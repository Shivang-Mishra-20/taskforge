"""
Task Forge AI — Core RL Environment
Implements OpenEnv spec: step(), reset(), state()
"""

from __future__ import annotations

import copy
import random
from typing import Any, Dict, List, Optional, Tuple

from environment.models import (
    Action,
    ActionType,
    Observation,
    Reward,
    RewardBreakdown,
    Task,
    TaskStatus,
    TaskType,
)
from environment.tasks import get_dynamic_events, get_task_list
from environment.graders import grade_episode, EpisodeResult


# ─── Constants ───────────────────────────────────────────────────────────────

WORKDAY_MINUTES = 480        # 8-hour workday
MAX_MISSED_DEADLINES = 5     # Early termination threshold
MIN_CSAT = 0.2               # Early termination threshold
TIME_PER_STEP = 5.0          # Minutes that pass per environment step
MAX_STEPS = 200              # Safety cap

DELEGATION_SUCCESS_RATES: Dict[str, float] = {
    "easy":   0.85,
    "medium": 0.70,
    "hard":   0.50,
}

# How much time progresses when delegating (agent hands off)
DELEGATION_TIME_COST = 10.0  # 10 min to hand off

# Dynamic event injection windows (minutes into episode)
DYNAMIC_EVENT_WINDOWS = {
    "hard": [(30, 50), (90, 120)],
}


class TaskForgeEnv:
    """
    Task Forge AI — SaaS Operations RL Environment.

    Implements the OpenEnv interface:
        reset()  → Observation
        step(action) → (Observation, Reward, done, info)
        state()  → dict
    """

    def __init__(self, scenario: str = "easy", seed: int = 42):
        assert scenario in ("easy", "medium", "hard"), \
            f"scenario must be easy/medium/hard, got {scenario!r}"
        self.scenario = scenario
        self.seed = seed
        self._rng = random.Random(seed)

        # Will be set on reset()
        self._tasks: List[Task] = []
        self._time: float = 0.0
        self._step_count: int = 0
        self._customer_satisfaction: float = 1.0
        self._team_capacity: float = 1.0
        self._cumulative_reward: float = 0.0
        self._done: bool = False
        self._early_terminated: bool = False
        self._dynamic_events_injected: List[str] = []
        self._missed_deadlines: int = 0

        # Track ongoing in-progress tasks
        self._in_progress: Dict[str, float] = {}  # task_id → completion_time

        self.reset()

    # ─── Public API ──────────────────────────────────────────────────────────

    def reset(self) -> Observation:
        """Reset environment to initial state. Returns first observation."""
        self._rng = random.Random(self.seed)
        self._tasks = get_task_list(self.scenario)
        self._time = 0.0
        self._step_count = 0
        self._customer_satisfaction = 1.0
        self._team_capacity = 1.0
        self._cumulative_reward = 0.0
        self._done = False
        self._early_terminated = False
        self._dynamic_events_injected = []
        self._missed_deadlines = 0
        self._in_progress = {}

        # Add estimated_time noise (±15%)
        for task in self._tasks:
            noise = self._rng.uniform(-0.15, 0.15)
            task.estimated_time = max(5.0, task.estimated_time * (1 + noise))

        return self._build_observation(dynamic_event=None)

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        """
        Execute one action. Returns (observation, reward, done, info).
        """
        if self._done:
            raise RuntimeError("Episode is done. Call reset() first.")

        # Validate action
        obs = self._build_observation()
        valid, reason = action.validate_action(obs)
        info: Dict[str, Any] = {"action_valid": valid, "reason": reason}

        dynamic_event_desc: Optional[str] = None

        if not valid:
            # Invalid action → small time penalty, no task progress
            reward_val, breakdown = self._compute_reward(
                action_result="invalid",
                task=None,
                time_spent=TIME_PER_STEP,
            )
            self._advance_time(TIME_PER_STEP)
            info["error"] = reason
        else:
            task = next(t for t in self._tasks if t.id == action.task_id)

            # ── Dispatch action ──
            if action.action_type == ActionType.START_TASK:
                time_spent, action_result = self._do_start_task(task)
            elif action.action_type == ActionType.DELEGATE_TASK:
                time_spent, action_result = self._do_delegate_task(task)
            elif action.action_type == ActionType.DELAY_TASK:
                time_spent, action_result = self._do_delay_task(task)
            elif action.action_type == ActionType.DROP_TASK:
                time_spent, action_result = self._do_drop_task(task)
            elif action.action_type == ActionType.REPRIORITIZE:
                time_spent, action_result = self._do_reprioritize(task, action.new_priority)
            else:
                time_spent, action_result = TIME_PER_STEP, "unknown_action"

            reward_val, breakdown = self._compute_reward(
                action_result=action_result,
                task=task,
                time_spent=time_spent,
            )
            self._advance_time(time_spent)
            info["action_result"] = action_result
            info["time_spent"] = time_spent

        # Process in-progress tasks that may have completed
        self._process_in_progress()

        # Check for missed deadlines
        self._check_missed_deadlines()

        # Inject dynamic events (hard scenario only)
        dynamic_event_desc = self._maybe_inject_dynamic_event()

        # Update CSAT
        self._update_customer_satisfaction()

        # Check termination conditions
        done = self._check_termination()
        self._done = done

        self._cumulative_reward += reward_val
        self._step_count += 1

        reward = Reward(
            value=max(-1.0, min(1.0, reward_val)),
            breakdown=breakdown,
            normalized=max(0.0, min(1.0, (reward_val + 1.0) / 2.0)),
        )

        new_obs = self._build_observation(dynamic_event=dynamic_event_desc)
        info["cumulative_reward"] = self._cumulative_reward
        info["step"] = self._step_count

        return new_obs, reward, done, info

    def state(self) -> Dict[str, Any]:
        """Return full serialisable environment state."""
        obs = self._build_observation()
        return {
            "scenario": self.scenario,
            "current_time": self._time,
            "time_remaining": max(0.0, WORKDAY_MINUTES - self._time),
            "step_count": self._step_count,
            "customer_satisfaction": self._customer_satisfaction,
            "team_capacity": self._team_capacity,
            "cumulative_reward": self._cumulative_reward,
            "done": self._done,
            "early_terminated": self._early_terminated,
            "missed_deadlines": self._missed_deadlines,
            "tasks": [t.model_dump() for t in self._tasks],
        }

    def grade(self) -> EpisodeResult:
        """Run deterministic grader and return score."""
        return grade_episode(
            scenario=self.scenario,
            tasks=self._tasks,
            customer_satisfaction=self._customer_satisfaction,
            missed_deadlines=self._missed_deadlines,
            episode_steps=self._step_count,
            early_terminated=self._early_terminated,
        )

    # ─── Action Handlers ─────────────────────────────────────────────────────

    def _do_start_task(self, task: Task) -> Tuple[float, str]:
        """Agent starts working on task themselves."""
        task.status = TaskStatus.IN_PROGRESS
        task.started_at = self._time
        # Actual time = estimated ± noise
        noise = self._rng.uniform(-0.10, 0.20)
        actual = max(5.0, task.estimated_time * (1 + noise))
        task.actual_time = actual
        # Schedule completion
        completion_time = self._time + actual
        self._in_progress[task.id] = completion_time
        return TIME_PER_STEP, "started"

    def _do_delegate_task(self, task: Task) -> Tuple[float, str]:
        """Agent delegates to team member."""
        task.delegation_attempts += 1
        success_rate = DELEGATION_SUCCESS_RATES[self.scenario]
        # Degrade with each attempt
        effective_rate = success_rate * (0.8 ** (task.delegation_attempts - 1))
        success = self._rng.random() < effective_rate

        if success:
            task.status = TaskStatus.DELEGATED
            task.assigned_to = f"delegate_{task.id}"
            # Delegation completes task in 1.3x estimated time (less efficient)
            actual = task.estimated_time * self._rng.uniform(1.1, 1.5)
            task.actual_time = actual
            completion_time = self._time + DELEGATION_TIME_COST + actual
            self._in_progress[task.id] = completion_time
            # Team capacity temporarily reduced
            self._team_capacity = max(0.0, self._team_capacity - 0.2)
            return DELEGATION_TIME_COST, "delegated_success"
        else:
            # Delegation failed → task returns to pending
            task.status = TaskStatus.PENDING
            task.assigned_to = None
            self._customer_satisfaction = max(0.0, self._customer_satisfaction - 0.02)
            return DELEGATION_TIME_COST, "delegated_failure"

    def _do_delay_task(self, task: Task) -> Tuple[float, str]:
        """Delay the task — push deadline by 30 minutes, small penalty."""
        if task.escalation_level >= 2:
            # Can't delay escalated tasks
            return TIME_PER_STEP, "delay_rejected_escalated"
        task.deadline += 30.0
        task.status = TaskStatus.DELAYED
        return TIME_PER_STEP, "delayed"

    def _do_drop_task(self, task: Task) -> Tuple[float, str]:
        """Drop the task entirely."""
        task.status = TaskStatus.DROPPED
        task.completed_at = self._time
        # CSAT hit proportional to customer impact
        self._customer_satisfaction = max(
            0.0,
            self._customer_satisfaction - task.customer_impact * 0.15
        )
        return TIME_PER_STEP, "dropped"

    def _do_reprioritize(self, task: Task, new_priority: Optional[int]) -> Tuple[float, str]:
        """Change task priority."""
        if new_priority is not None:
            task.priority = new_priority
        return TIME_PER_STEP, "reprioritized"

    # ─── Internals ───────────────────────────────────────────────────────────

    def _advance_time(self, minutes: float):
        self._time = min(self._time + minutes, WORKDAY_MINUTES + 1)

    def _process_in_progress(self):
        """Check which in-progress tasks have completed."""
        to_remove = []
        for task_id, completion_time in self._in_progress.items():
            if self._time >= completion_time:
                task = next((t for t in self._tasks if t.id == task_id), None)
                if task:
                    task.status = TaskStatus.COMPLETED
                    task.completed_at = completion_time
                    # Restore some team capacity if delegated
                    if task.assigned_to:
                        self._team_capacity = min(1.0, self._team_capacity + 0.2)
                    # CSAT improvement for completed tasks
                    self._customer_satisfaction = min(
                        1.0,
                        self._customer_satisfaction + task.customer_impact * 0.05
                    )
                to_remove.append(task_id)
        for tid in to_remove:
            del self._in_progress[tid]

    def _check_missed_deadlines(self):
        """Mark tasks that missed their deadline."""
        for task in self._tasks:
            if (
                task.status in (TaskStatus.PENDING, TaskStatus.DELAYED, TaskStatus.IN_PROGRESS)
                and self._time > task.deadline
                and task.id not in self._in_progress
            ):
                # Only count first time
                if task.status != TaskStatus.FAILED:
                    task.status = TaskStatus.FAILED
                    self._missed_deadlines += 1
                    # CSAT hit
                    self._customer_satisfaction = max(
                        0.0,
                        self._customer_satisfaction - task.customer_impact * 0.1
                    )
                    # Escalation
                    if task.escalation_level < 3:
                        task.escalation_level += 1

    def _maybe_inject_dynamic_event(self) -> Optional[str]:
        """Inject dynamic events for the hard scenario."""
        if self.scenario != "hard":
            return None

        windows = DYNAMIC_EVENT_WINDOWS.get("hard", [])
        for start, end in windows:
            if start <= self._time <= end:
                # Get uinjected events
                candidate_events = get_dynamic_events(self._time)
                for event in candidate_events:
                    if event.id not in self._dynamic_events_injected:
                        # 30% chance per step within window
                        if self._rng.random() < 0.30:
                            self._tasks.append(event)
                            self._dynamic_events_injected.append(event.id)
                            return f"⚡ Dynamic Event: {event.description}"
        return None

    def _update_customer_satisfaction(self):
        """Natural CSAT decay from unresolved escalations."""
        active_escalations = sum(
            1 for t in self._tasks
            if t.escalation_level >= 2
            and t.status not in (TaskStatus.COMPLETED, TaskStatus.DROPPED)
        )
        if active_escalations > 0:
            decay = 0.005 * active_escalations
            self._customer_satisfaction = max(0.0, self._customer_satisfaction - decay)

        # Team capacity slowly recovers
        self._team_capacity = min(1.0, self._team_capacity + 0.02)

    def _check_termination(self) -> bool:
        """Check episode termination conditions."""
        if self._time >= WORKDAY_MINUTES:
            return True
        if self._step_count >= MAX_STEPS:
            return True
        if self._missed_deadlines >= MAX_MISSED_DEADLINES:
            self._early_terminated = True
            return True
        if self._customer_satisfaction <= MIN_CSAT:
            self._early_terminated = True
            return True
        # All tasks resolved
        unresolved = [
            t for t in self._tasks
            if t.status in (TaskStatus.PENDING, TaskStatus.IN_PROGRESS, TaskStatus.DELAYED)
            and t.id not in self._in_progress
        ]
        if not unresolved and not self._in_progress:
            return True
        return False

    # ─── Reward ──────────────────────────────────────────────────────────────

    def _compute_reward(
        self,
        action_result: str,
        task: Optional[Task],
        time_spent: float,
    ) -> Tuple[float, RewardBreakdown]:
        """Dense reward function returning value ∈ [-1, 1]."""
        b = RewardBreakdown()

        if action_result == "invalid":
            b.inefficiency_penalty = -0.05
            b.total = b.inefficiency_penalty
            return b.total, b

        if task is None:
            b.total = 0.0
            return 0.0, b

        # ── Positive rewards ──

        if action_result in ("started", "delegated_success"):
            # Progress reward proportional to task importance
            importance = (task.priority / 5.0) * 0.5 + task.customer_impact * 0.3 + task.revenue_impact * 0.2
            b.task_completion = importance * 0.1  # Partial reward for starting

            # Bonus for tackling escalated tasks quickly
            if task.escalation_level >= 2:
                b.escalation_penalty = 0.05  # Actually a bonus here (positive)
                b.task_completion += task.escalation_level * 0.02

        if action_result == "reprioritized":
            b.task_completion = 0.01  # Tiny credit for active management

        # ── Negative rewards ──

        if action_result == "delegated_failure":
            b.delegation_penalty = -0.08
            # Extra penalty for failing on important tasks
            b.delegation_penalty -= task.customer_impact * 0.05

        if action_result == "dropped":
            b.missed_deadline_penalty = -(task.customer_impact * 0.15 + task.revenue_impact * 0.10)
            if task.escalation_level >= 2:
                b.escalation_penalty = -0.15

        if action_result == "delay_rejected_escalated":
            b.escalation_penalty = -0.10

        if action_result == "delayed":
            # Small penalty for delaying — encourages proactive handling
            b.inefficiency_penalty = -0.02

        # ── Time efficiency ──
        # Penalise wasting time on low-priority tasks when escalations exist
        active_escalations = sum(
            1 for t in self._tasks
            if t.escalation_level >= 2
            and t.status not in (TaskStatus.COMPLETED, TaskStatus.DROPPED, TaskStatus.FAILED)
        )
        if active_escalations > 0 and task.priority <= 2:
            b.inefficiency_penalty -= 0.03  # Wasting time

        # ── Missed deadlines penalty (step-level) ──
        newly_missed = sum(
            1 for t in self._tasks
            if t.status == TaskStatus.FAILED
            and t.completed_at is None
            and self._time > t.deadline
        )
        b.missed_deadline_penalty += newly_missed * -0.05

        b.total = (
            b.task_completion
            + b.deadline_bonus
            + b.customer_impact_bonus
            + b.escalation_penalty
            + b.missed_deadline_penalty
            + b.delegation_penalty
            + b.inefficiency_penalty
            + b.early_termination_penalty
        )

        return max(-1.0, min(1.0, b.total)), b

    # ─── Observation Builder ─────────────────────────────────────────────────

    def _build_observation(self, dynamic_event: Optional[str] = None) -> Observation:
        active_escalations = sum(
            1 for t in self._tasks
            if t.escalation_level >= 1
            and t.status not in (TaskStatus.COMPLETED, TaskStatus.DROPPED, TaskStatus.FAILED)
        )
        completed = sum(1 for t in self._tasks if t.status == TaskStatus.COMPLETED)
        return Observation(
            current_time=round(self._time, 2),
            time_remaining=round(max(0.0, WORKDAY_MINUTES - self._time), 2),
            task_list=copy.deepcopy(self._tasks),
            customer_satisfaction=round(self._customer_satisfaction, 4),
            team_capacity=round(self._team_capacity, 4),
            active_escalations=active_escalations,
            completed_tasks=completed,
            missed_deadlines=self._missed_deadlines,
            cumulative_reward=round(self._cumulative_reward, 4),
            episode_step=self._step_count,
            dynamic_event=dynamic_event,
        )
