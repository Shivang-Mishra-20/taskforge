#!/usr/bin/env python3
"""
Task Forge AI — FastAPI HTTP Server
Runs on port 7860 for Hugging Face Spaces.

Endpoints:
  GET  /              → info page
  GET  /health        → {"status":"ok"} — HF Spaces ping
  POST /reset         → {"scenario":"easy|medium|hard","seed":42}
  POST /step          → {"action_type":"...","task_id":"...","new_priority":null}
  GET  /state         → current environment state
  GET  /grade         → grader score for current episode
  POST /run           → run full inference episode, returns [START]/[STEP]/[END] log
"""

import os
import sys
import json
import subprocess
from typing import Optional, Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel
import uvicorn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from environment.env import TaskForgeEnv
from environment.models import Action, ActionType


# ─── Score safety ─────────────────────────────────────────────────────────────

def _safe_score(score: float) -> float:
    """Clamp score to strictly open interval (0, 1) — never 0.0 or 1.0.
    Applies epsilon guard BEFORE rounding to prevent e.g. round(0.99995, 4) → 1.0."""
    epsilon = 1e-6
    score = max(epsilon, min(1 - epsilon, float(score)))
    score = min(score, 0.9999)  # pre-rounding guard
    return round(score, 4)


# ─── App ─────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Task Forge AI",
    description="Real-Time SaaS Operations RL Environment — OpenEnv Compliant",
    version="1.0.0",
)

# Single global env instance (stateful per-session)
_env: Optional[TaskForgeEnv] = None


def get_env() -> TaskForgeEnv:
    global _env
    if _env is None:
        _env = TaskForgeEnv(scenario="easy", seed=42)
    return _env


# ─── Request models ───────────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    scenario: str = "easy"
    seed: int = 42


class StepRequest(BaseModel):
    action_type: str = "start_task"
    task_id: str = "T001"
    new_priority: Optional[int] = None


class RunRequest(BaseModel):
    scenario: str = "easy"
    seed: int = 42
    max_steps: int = 200


# ─── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "name":        "Task Forge AI",
        "version":     "1.0.0",
        "description": "SaaS Operations RL Environment — OpenEnv Compliant",
        "scenarios":   ["easy", "medium", "hard"],
        "endpoints": {
            "health":  "GET  /health",
            "reset":   "POST /reset  {scenario, seed}",
            "step":    "POST /step   {action_type, task_id, new_priority}",
            "state":   "GET  /state",
            "grade":   "GET  /grade",
            "run":     "POST /run    {scenario, seed, max_steps}",
        },
    }


@app.get("/health")
def health():
    """HF Spaces ping endpoint — must return 200."""
    return {"status": "ok", "env": "task-forge-ai"}


@app.post("/reset")
def reset(req: Optional[ResetRequest] = None):
    """Reset environment to initial state. Returns first observation."""
    global _env
    scenario = req.scenario if req else "easy"
    seed = req.seed if req else 42
    if scenario not in ("easy", "medium", "hard"):
        scenario = "easy"
    _env = TaskForgeEnv(scenario=scenario, seed=seed)
    obs = _env.reset()
    return {
        "observation": obs.model_dump(),
        "scenario": scenario,
        "seed": seed,
    }


@app.post("/step")
def step(req: StepRequest):
    """Execute one action. Returns observation, reward, done, info."""
    env = get_env()
    try:
        action = Action(
            action_type=ActionType(req.action_type),
            task_id=req.task_id,
            new_priority=req.new_priority,
        )
    except ValueError as e:
        raise HTTPException(400, f"Invalid action: {e}")

    try:
        obs, reward, done, info = env.step(action)
    except RuntimeError as e:
        raise HTTPException(409, str(e))

    return {
        "observation": obs.model_dump(),
        "reward":      reward.model_dump(),
        "done":        done,
        "info":        info,
    }


@app.get("/state")
def state():
    """Return full current environment state."""
    env = get_env()
    return env.state()


@app.get("/grade")
def grade():
    """Run deterministic grader on current episode."""
    env = get_env()
    result = env.grade()
    data = result.model_dump()
    # Clamp score to strictly open (0, 1) — Phase 2 fail-fast requirement
    if "score" in data:
        data["score"] = _safe_score(data["score"])
    return data


@app.post("/run")
def run_inference(req: RunRequest):
    """
    Run a complete inference episode using the built-in rule-based agent.
    Returns structured log lines in [START]/[STEP]/[END] format.
    """
    if req.scenario not in ("easy", "medium", "hard", "all"):
        raise HTTPException(400, "scenario must be easy/medium/hard/all")

    args = [
        sys.executable, "inference.py",
        "--scenario", req.scenario,
        "--seed",     str(req.seed),
        "--max-steps", str(req.max_steps),
    ]
    result = subprocess.run(
        args,
        capture_output=True,
        text=True,
        cwd=os.path.dirname(os.path.abspath(__file__)),
        timeout=1200,  # 20 min max
    )

    lines      = result.stdout.strip().split("\n")
    start_logs = [l for l in lines if l.startswith("[START]")]
    step_logs  = [l for l in lines if l.startswith("[STEP]")]
    end_logs   = [l for l in lines if l.startswith("[END]")]

    # Parse END for score summary — clamp scores to strictly open (0, 1)
    scores = []
    for el in end_logs:
        try:
            entry = json.loads(el[6:])
            # Clamp top-level score field and per-reward values
            if "score" in entry:
                entry["score"] = _safe_score(entry["score"])
            if "rewards" in entry and isinstance(entry["rewards"], list):
                entry["rewards"] = [
                    round(max(1e-6, min(1 - 1e-6, float(r))), 4)
                    if 0.0 <= float(r) <= 1.0 else r
                    for r in entry["rewards"]
                ]
            scores.append(entry)
        except Exception:
            pass

    return {
        "stdout":      result.stdout,
        "returncode":  result.returncode,
        "start_count": len(start_logs),
        "step_count":  len(step_logs),
        "end_count":   len(end_logs),
        "scores":      scores,
    }


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
    )
