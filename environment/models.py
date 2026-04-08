"""
Pydantic models for Task Forge AI OpenEnv environment.
Typed definitions for Observation, Action, Reward.
"""

from __future__ import annotations
from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


# ─── Enums ──────────────────────────────────────────────────────────────────

class TaskType(str, Enum):
    BUG = "bug"
    SUPPORT = "support"
    SALES = "sales"
    INTERNAL = "internal"
    FEATURE = "feature"


class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    DELEGATED = "delegated"
    COMPLETED = "completed"
    FAILED = "failed"
    DELAYED = "delayed"
    DROPPED = "dropped"


class ActionType(str, Enum):
    START_TASK = "start_task"
    DELEGATE_TASK = "delegate_task"
    DELAY_TASK = "delay_task"
    DROP_TASK = "drop_task"
    REPRIORITIZE = "reprioritize"


# ─── Task Model ─────────────────────────────────────────────────────────────

class Task(BaseModel):
    id: str
    type: TaskType
    priority: int = Field(ge=1, le=5, description="1=lowest, 5=highest")
    deadline: float = Field(description="Minutes from simulation start")
    estimated_time: float = Field(description="Estimated minutes to complete")
    customer_impact: float = Field(ge=0.0, le=1.0, description="0=none, 1=critical")
    revenue_impact: float = Field(ge=0.0, le=1.0, description="0=none, 1=very high")
    dependencies: List[str] = Field(default_factory=list, description="Task IDs that must complete first")
    status: TaskStatus = TaskStatus.PENDING
    escalation_level: int = Field(default=0, ge=0, le=3, description="0=none, 3=CEO involved")
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    assigned_to: Optional[str] = None  # None = self, else delegate name
    delegation_attempts: int = 0
    actual_time: Optional[float] = None  # filled on completion
    description: str = ""


# ─── Observation / State ─────────────────────────────────────────────────────

class Observation(BaseModel):
    current_time: float = Field(description="Current simulation time in minutes")
    time_remaining: float = Field(description="Minutes left in workday (480 min)")
    task_list: List[Task]
    customer_satisfaction: float = Field(ge=0.0, le=1.0)
    team_capacity: float = Field(ge=0.0, le=1.0, description="Available delegation capacity")
    active_escalations: int = Field(ge=0, description="Number of active escalations")
    completed_tasks: int
    missed_deadlines: int
    cumulative_reward: float
    episode_step: int
    dynamic_event: Optional[str] = None  # Description of any dynamic event this step


# ─── Action ─────────────────────────────────────────────────────────────────

class Action(BaseModel):
    action_type: ActionType
    task_id: str
    new_priority: Optional[int] = Field(default=None, ge=1, le=5)

    model_config = {"use_enum_values": True}

    def validate_action(self, obs: Observation) -> tuple[bool, str]:
        """Returns (is_valid, reason)."""
        task_ids = {t.id for t in obs.task_list}
        if self.task_id not in task_ids:
            return False, f"Task {self.task_id} not found"

        task = next(t for t in obs.task_list if t.id == self.task_id)

        if self.action_type == ActionType.REPRIORITIZE and self.new_priority is None:
            return False, "new_priority required for reprioritize action"

        if task.status in (TaskStatus.COMPLETED, TaskStatus.DROPPED, TaskStatus.FAILED):
            return False, f"Task {self.task_id} is already {task.status}"

        if self.action_type == ActionType.START_TASK:
            # Check dependencies
            for dep_id in task.dependencies:
                dep_tasks = [t for t in obs.task_list if t.id == dep_id]
                if dep_tasks and dep_tasks[0].status != TaskStatus.COMPLETED:
                    return False, f"Dependency {dep_id} not yet completed"

        return True, "ok"


# ─── Reward ─────────────────────────────────────────────────────────────────

class RewardBreakdown(BaseModel):
    task_completion: float = 0.0
    deadline_bonus: float = 0.0
    customer_impact_bonus: float = 0.0
    escalation_penalty: float = 0.0
    missed_deadline_penalty: float = 0.0
    delegation_penalty: float = 0.0
    inefficiency_penalty: float = 0.0
    early_termination_penalty: float = 0.0
    total: float = 0.0


class Reward(BaseModel):
    value: float = Field(ge=-1.0, le=1.0)
    breakdown: RewardBreakdown
    normalized: float = Field(gt=0.0, lt=1.0, description="Normalized for graders (0,1)")


# ─── Episode Result ──────────────────────────────────────────────────────────

class EpisodeResult(BaseModel):
    task_id: str
    score: float = Field(gt=0.0, lt=1.0)
    details: Dict[str, Any]
    passed: bool