#!/usr/bin/env python3
"""
Task Forge AI — Inference Script
OpenEnv-compliant. Strict [START]/[STEP]/[END] log format matching sample script.

Free Mode (no API key):
    python inference.py --scenario easy

API Mode:
    API_BASE_URL=https://api.openai.com/v1 MODEL_NAME=gpt-4o-mini \
    OPENAI_API_KEY=sk-... python inference.py --scenario hard

All Scenarios:
    python inference.py --scenario all
"""

import os
import sys
import json
import time
import argparse
import logging
from typing import Optional, Dict, Any, List

# ─── Logging setup ───────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("taskforge")

# ─── Env imports ─────────────────────────────────────────────────────────────

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from environment.env import TaskForgeEnv
from environment.models import Action, ActionType, Observation, Task, TaskStatus

# ─── Config (from env vars, exactly as required) ─────────────────────────────

API_BASE_URL   = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME     = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN       = os.environ.get("HF_TOKEN")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

USE_API = bool(OPENAI_API_KEY or HF_TOKEN) and bool(API_BASE_URL)

# ─── Structured log helpers (matching sample script exactly) ─────────────────

def log_start(task: str, env: str, model: str) -> None:
    payload = {
        "task":      task,
        "env":       env,
        "model":     model,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    print(f"[START] {json.dumps(payload)}", flush=True)


def log_step(step: int, action: Any, reward: float,
             done: bool, error: Optional[str]) -> None:
    
    epsilon = 1e-6
    safe_reward = max(-(1 - epsilon), min(1 - epsilon, reward))
    safe_reward = min(safe_reward, 0.9999)  # guard before rounding

    payload = {
        "step":   step,
        "action": action,
        "reward": round(safe_reward, 4),
        "done":   done,
        "error":  error,
    }

    print(f"[STEP] {json.dumps(payload)}", flush=True)


def log_end(success: bool, steps: int, score: float,
            rewards: List[float]) -> None:
    
    epsilon = 1e-6
    safe_score = max(epsilon, min(1 - epsilon, score))
    safe_score = min(safe_score, 0.9999)  # guard before rounding

    payload = {
        "success": success,
        "steps":   steps,
        "score":   round(safe_score, 4),
        "rewards": [round(r, 4) for r in rewards],
    }

    print(f"[END] {json.dumps(payload)}", flush=True)


# ─── Rule-Based Agent ────────────────────────────────────────────────────────

class RuleBasedAgent:
    """Deterministic priority-heuristic agent — zero API cost."""

    def decide(self, obs: Observation) -> Action:
        tasks = obs.task_list

        # 1. Level-3 escalations first
        crit = self._filter(tasks, min_esc=3,
                            statuses=[TaskStatus.PENDING, TaskStatus.DELAYED])
        if crit:
            return Action(action_type=ActionType.START_TASK, task_id=crit[0].id)

        # 2. Overdue low-impact tasks → drop to clear queue
        overdue = [
            t for t in tasks
            if t.status in (TaskStatus.PENDING, TaskStatus.DELAYED)
            and t.deadline < obs.current_time
        ]
        for t in sorted(overdue, key=lambda x: x.customer_impact):
            if t.customer_impact < 0.3:
                return Action(action_type=ActionType.DROP_TASK, task_id=t.id)

        # 3. Ready tasks (deps satisfied)
        ready = self._ready_tasks(tasks, obs)
        if ready:
            best = sorted(ready, key=lambda t: self._score(t, obs), reverse=True)[0]
            if (obs.team_capacity > 0.5
                    and best.priority <= 3
                    and best.customer_impact < 0.6
                    and best.delegation_attempts < 2):
                return Action(action_type=ActionType.DELEGATE_TASK, task_id=best.id)
            return Action(action_type=ActionType.START_TASK, task_id=best.id)

        # 4. Reprioritize escalated pending tasks
        for t in tasks:
            if (t.status == TaskStatus.PENDING
                    and t.escalation_level >= 2
                    and t.priority < 4):
                return Action(action_type=ActionType.REPRIORITIZE,
                              task_id=t.id, new_priority=5)

        # 5. Delay low-priority tasks nearing deadline
        delayable = [
            t for t in tasks
            if t.status == TaskStatus.PENDING
            and t.escalation_level == 0
            and t.priority <= 2
            and t.deadline - obs.current_time < 60
        ]
        if delayable:
            return Action(action_type=ActionType.DELAY_TASK, task_id=delayable[0].id)

        # 6. Start any remaining pending task
        fallback = [t for t in tasks if t.status == TaskStatus.PENDING]
        if fallback:
            best = sorted(fallback, key=lambda t: self._score(t, obs), reverse=True)[0]
            return Action(action_type=ActionType.START_TASK, task_id=best.id)

        # 7. Drop very low value delayed tasks
        droppable = [t for t in tasks
                     if t.status in (TaskStatus.PENDING, TaskStatus.DELAYED)
                     and t.customer_impact < 0.2]
        if droppable:
            return Action(action_type=ActionType.DROP_TASK, task_id=droppable[0].id)

        # 8. Wait: harmless reprioritize on any actionable task
        waitable = [t for t in tasks
                    if t.status in (TaskStatus.PENDING, TaskStatus.DELAYED)]
        if waitable:
            worst = min(waitable, key=lambda t: t.priority)
            return Action(action_type=ActionType.REPRIORITIZE,
                          task_id=worst.id, new_priority=worst.priority)

        in_prog = [t for t in tasks if t.status == TaskStatus.IN_PROGRESS]
        if in_prog:
            return Action(action_type=ActionType.REPRIORITIZE,
                          task_id=in_prog[0].id, new_priority=in_prog[0].priority)

        delegated = [t for t in tasks if t.status == TaskStatus.DELEGATED]
        if delegated:
            return Action(action_type=ActionType.REPRIORITIZE,
                          task_id=delegated[0].id, new_priority=delegated[0].priority)

        return Action(action_type=ActionType.REPRIORITIZE,
                      task_id=tasks[0].id, new_priority=tasks[0].priority)

    def _filter(self, tasks, min_esc=0, statuses=None):
        if statuses is None:
            statuses = [TaskStatus.PENDING]
        return [t for t in tasks
                if t.escalation_level >= min_esc and t.status in statuses]

    def _ready_tasks(self, tasks: List[Task], obs: Observation) -> List[Task]:
        completed_ids = {t.id for t in tasks if t.status == TaskStatus.COMPLETED}
        return [
            t for t in tasks
            if t.status in (TaskStatus.PENDING, TaskStatus.DELAYED)
            and all(dep in completed_ids for dep in t.dependencies)
        ]

    def _score(self, task: Task, obs: Observation) -> float:
        time_left = max(1.0, task.deadline - obs.current_time)
        urgency    = 1.0 / time_left * 100
        importance = (task.priority / 5.0 * 0.4
                      + task.customer_impact * 0.35
                      + task.revenue_impact * 0.25)
        esc_bonus  = task.escalation_level * 0.15
        return urgency * 0.4 + importance * 0.45 + esc_bonus * 0.15


# ─── LLM Agent (OpenAI-compatible) ───────────────────────────────────────────

class LLMAgent:
    """Uses OpenAI client with API_BASE_URL / MODEL_NAME / HF_TOKEN vars.
    Falls back to RuleBasedAgent on any error or missing key."""

    def __init__(self):
        self.fallback = RuleBasedAgent()
        self.client = None
        try:
            from openai import OpenAI
            kwargs: Dict[str, Any] = {}
            if API_BASE_URL:
                kwargs["base_url"] = API_BASE_URL
            api_key = OPENAI_API_KEY or HF_TOKEN or "dummy-key"
            self.client = OpenAI(api_key=api_key, **kwargs)
        except ImportError:
            logger.warning("openai not installed — using rule-based fallback")

    def decide(self, obs: Observation) -> Action:
        if self.client is None:
            return self.fallback.decide(obs)
        system = (
            "You are a SaaS ops manager. Given tasks, pick ONE action as JSON:\n"
            '{"action_type":"start_task|delegate_task|delay_task|drop_task|reprioritize",'
            '"task_id":"<id>","new_priority":null}\n'
            "Rules: handle escalation>=3 first, respect deps, maximise CSAT & revenue."
        )
        user = (
            f"Time: {obs.current_time:.0f}m / {obs.time_remaining:.0f}m left | "
            f"CSAT={obs.customer_satisfaction:.2f} capacity={obs.team_capacity:.2f}\n"
            + "\n".join(
                f"[{t.id}] {t.type} pri={t.priority} dl={t.deadline:.0f}m "
                f"cust={t.customer_impact:.1f} esc={t.escalation_level} "
                f"status={t.status} deps={t.dependencies} | {t.description[:40]}"
                for t in obs.task_list
            )
            + "\nJSON only:"
        )
        try:
            resp = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "system", "content": system},
                           {"role": "user",   "content": user}],
                temperature=0.0,
                max_tokens=80,
            )
            raw = resp.choices[0].message.content.strip()
            raw = raw.replace("```json", "").replace("```", "").strip()
            d = json.loads(raw)
            return Action(
                action_type=ActionType(d["action_type"]),
                task_id=str(d["task_id"]),
                new_priority=d.get("new_priority"),
            )
        except Exception as exc:
            logger.warning(f"LLM error ({exc}), fallback to rule-based")
            return self.fallback.decide(obs)


# ─── Episode runner ───────────────────────────────────────────────────────────

TASK_NAMES = {
    "easy":   "Morning Triage",
    "medium": "Payment Crisis",
    "hard":   "Total Outage",
}
BENCHMARK = "task-forge-ai"
SUCCESS_SCORE_THRESHOLD = 0.5
MAX_TOTAL_REWARD = 10.0   # normalisation denominator


def run_episode(scenario: str, seed: int = 42,
                max_steps: int = 200) -> Dict[str, Any]:
    env        = TaskForgeEnv(scenario=scenario, seed=seed)
    agent      = LLMAgent() if USE_API else RuleBasedAgent()
    model_name = MODEL_NAME if USE_API else "RuleBasedAgent"
    task_name  = TASK_NAMES.get(scenario, scenario)

    rewards:      List[float] = []
    steps_taken:  int         = 0
    score:        float       = 0.0
    success:      bool        = False

    log_start(task=task_name, env=BENCHMARK, model=model_name)

    try:
        obs  = env.reset()
        done = False

        for step in range(1, max_steps + 1):
            if done:
                break

            action = agent.decide(obs)
            obs, reward, done, info = env.step(action)

            reward_val = reward.value
            error_msg  = info.get("error") if not info.get("action_valid", True) else None

            rewards.append(reward_val)
            steps_taken = step

            # Serialize action for logging
            action_log = {
                "type":         action.action_type,
                "task_id":      action.task_id,
                "new_priority": action.new_priority,
            }
            log_step(step=step, action=action_log,
                     reward=reward_val, done=done, error=error_msg)

            if done:
                break

        # Normalise score to [0,1]
        total_reward = sum(rewards)
        epsilon = 1e-6
        score = total_reward / MAX_TOTAL_REWARD
        score = min(score, 0.9999)  # guard before clamp
        score = max(epsilon, min(1 - epsilon, score))
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        log_end(success=success, steps=steps_taken,
                score=score, rewards=rewards)

    # Also compute grader score for summary display
    grade_result  = env.grade()
    final_state   = env.state()

    return {
        "scenario":      scenario,
        "grade_score":   grade_result.score,
        "grade_passed":  grade_result.passed,
        "score":         score,
        "success":       success,
        "total_steps":   steps_taken,
        "total_reward":  round(total_reward, 4),
    }


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Task Forge AI — OpenEnv Inference")
    parser.add_argument("--scenario",
                        choices=["easy", "medium", "hard", "all"],
                        default="all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=200)
    args = parser.parse_args()

    scenarios = (["easy", "medium", "hard"]
                 if args.scenario == "all" else [args.scenario])

    results = {}
    for sc in scenarios:
        print(f"\n{'='*60}", flush=True)
        print(f"Running scenario: {sc.upper()}", flush=True)
        print(f"{'='*60}", flush=True)
        results[sc] = run_episode(sc, seed=args.seed, max_steps=args.max_steps)

    print(f"\n{'='*60}", flush=True)
    print("SUMMARY", flush=True)
    print(f"{'='*60}", flush=True)
    for sc, r in results.items():
        status = "✅ PASS" if r["grade_passed"] else "❌ FAIL"
        print(
            f"{sc.upper():8s} | grade={r['grade_score']:.4f} | "
            f"score={r['score']:.4f} | steps={r['total_steps']:3d} | "
            f"reward={r['total_reward']:+.3f} | {status}",
            flush=True,
        )


if __name__ == "__main__":
    main()
