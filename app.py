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

import os, sys, json, subprocess
from typing import Optional, Any, Dict
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from environment.env import TaskForgeEnv
from environment.models import Action, ActionType

app = FastAPI(title="Task Forge AI", description="Real-Time SaaS Operations RL Environment", version="1.0.0")

_env: Optional[TaskForgeEnv] = None

def get_env() -> TaskForgeEnv:
    global _env
    if _env is None:
        _env = TaskForgeEnv(scenario="easy", seed=42)
    return _env

class ResetRequest(BaseModel):
    scenario: str = "easy"
    seed: int = 42

class StepRequest(BaseModel):
    action: Dict[str, Any]

class RunRequest(BaseModel):
    scenario: str = "easy"
    seed: int = 42
    max_steps: int = 200

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.get("/metadata")
def metadata():
    return {
        "name": "Task Forge AI",
        "description": "Real-Time SaaS Operations RL Environment where an agent manages bugs, escalations, deadlines and customer issues across an 8-hour workday.",
        "version": "1.0.0",
        "author": "CaptainMishra",
    }

@app.get("/schema")
def schema():
    return {
        "action": {
            "type": "object",
            "properties": {
                "action_type": {"type": "string", "enum": ["start_task","delegate_task","delay_task","drop_task","reprioritize"]},
                "task_id": {"type": "string"},
                "new_priority": {"type": "integer", "minimum": 1, "maximum": 5},
            },
            "required": ["action_type", "task_id"],
        },
        "observation": {
            "type": "object",
            "properties": {
                "current_time": {"type": "number"},
                "time_remaining": {"type": "number"},
                "task_list": {"type": "array"},
                "customer_satisfaction": {"type": "number"},
                "team_capacity": {"type": "number"},
                "active_escalations": {"type": "integer"},
                "completed_tasks": {"type": "integer"},
                "missed_deadlines": {"type": "integer"},
                "cumulative_reward": {"type": "number"},
                "episode_step": {"type": "integer"},
            },
        },
        "state": {
            "type": "object",
            "properties": {
                "scenario": {"type": "string"},
                "current_time": {"type": "number"},
                "done": {"type": "boolean"},
            },
        },
    }

@app.post("/mcp")
def mcp(request: Dict[str, Any] = {}):
    return {"jsonrpc": "2.0", "id": request.get("id", 1), "result": {"tools": []}}

@app.get("/")
def root():
    return {"name": "Task Forge AI", "version": "1.0.0", "description": "SaaS Operations RL Environment"}

@app.post("/reset")
def reset(req: Optional[ResetRequest] = None):
    global _env
    scenario = req.scenario if req else "easy"
    seed = req.seed if req else 42
    if scenario not in ("easy", "medium", "hard"):
        scenario = "easy"
    _env = TaskForgeEnv(scenario=scenario, seed=seed)
    obs = _env.reset()
    return {"observation": obs.model_dump(), "reward": None, "done": False}

@app.post("/step")
def step(req: StepRequest):
    env = get_env()
    action_data = req.action
    try:
        action = Action(
            action_type=ActionType(action_data.get("action_type", "start_task")),
            task_id=str(action_data.get("task_id", "")),
            new_priority=action_data.get("new_priority"),
        )
    except ValueError as e:
        raise HTTPException(400, f"Invalid action: {e}")
    try:
        obs, reward, done, info = env.step(action)
    except RuntimeError as e:
        raise HTTPException(409, str(e))
    return {"observation": obs.model_dump(), "reward": reward.value, "done": done}

@app.get("/state")
def state():
    return get_env().state()

@app.get("/grade")
def grade():
    return get_env().grade().model_dump()

@app.post("/run")
def run_inference(req: RunRequest):
    args = [sys.executable, "inference.py", "--scenario", req.scenario, "--seed", str(req.seed), "--max-steps", str(req.max_steps)]
    result = subprocess.run(args, capture_output=True, text=True, cwd=os.path.dirname(os.path.abspath(__file__)), timeout=1200)
    lines = result.stdout.strip().split("\n")
    scores = []
    for l in lines:
        if l.startswith("[END]"):
            try: scores.append(json.loads(l[6:]))
            except: pass
    return {"stdout": result.stdout, "returncode": result.returncode, "scores": scores}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("app:app", host="0.0.0.0", port=port, log_level="info")