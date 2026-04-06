"""
Task definitions for Task Forge AI.
Defines initial states for easy, medium, and hard tasks.
"""

from __future__ import annotations
import copy
from typing import List
from environment.models import Task, TaskType, TaskStatus


# ─── TASK 1: Easy ────────────────────────────────────────────────────────────
# Low complexity, minimal conflicts, clear priorities

EASY_TASKS: List[dict] = [
    {
        "id": "T001",
        "type": TaskType.BUG,
        "priority": 5,
        "deadline": 120.0,  # 2 hours
        "estimated_time": 30.0,
        "customer_impact": 0.8,
        "revenue_impact": 0.3,
        "dependencies": [],
        "status": TaskStatus.PENDING,
        "escalation_level": 1,
        "description": "Critical login bug affecting enterprise customers",
    },
    {
        "id": "T002",
        "type": TaskType.SUPPORT,
        "priority": 3,
        "deadline": 240.0,
        "estimated_time": 20.0,
        "customer_impact": 0.4,
        "revenue_impact": 0.1,
        "dependencies": [],
        "status": TaskStatus.PENDING,
        "escalation_level": 0,
        "description": "Customer inquiry about billing invoice",
    },
    {
        "id": "T003",
        "type": TaskType.INTERNAL,
        "priority": 1,
        "deadline": 480.0,
        "estimated_time": 45.0,
        "customer_impact": 0.0,
        "revenue_impact": 0.0,
        "dependencies": [],
        "status": TaskStatus.PENDING,
        "escalation_level": 0,
        "description": "Update internal documentation wiki",
    },
    {
        "id": "T004",
        "type": TaskType.SALES,
        "priority": 4,
        "deadline": 180.0,
        "estimated_time": 25.0,
        "customer_impact": 0.5,
        "revenue_impact": 0.7,
        "dependencies": [],
        "status": TaskStatus.PENDING,
        "escalation_level": 0,
        "description": "Prepare enterprise demo for prospect call at 3pm",
    },
    {
        "id": "T005",
        "type": TaskType.FEATURE,
        "priority": 2,
        "deadline": 420.0,
        "estimated_time": 60.0,
        "customer_impact": 0.2,
        "revenue_impact": 0.2,
        "dependencies": [],
        "status": TaskStatus.PENDING,
        "escalation_level": 0,
        "description": "Implement dark mode toggle (low priority backlog)",
    },
]

# ─── TASK 2: Medium ──────────────────────────────────────────────────────────
# Deadline conflicts, dependency chains, resource pressure

MEDIUM_TASKS: List[dict] = [
    {
        "id": "M001",
        "type": TaskType.BUG,
        "priority": 5,
        "deadline": 60.0,  # Very tight
        "estimated_time": 45.0,
        "customer_impact": 0.9,
        "revenue_impact": 0.8,
        "dependencies": [],
        "status": TaskStatus.PENDING,
        "escalation_level": 2,
        "description": "Payment processing failure – customers can't checkout",
    },
    {
        "id": "M002",
        "type": TaskType.BUG,
        "priority": 4,
        "deadline": 90.0,
        "estimated_time": 30.0,
        "customer_impact": 0.6,
        "revenue_impact": 0.5,
        "dependencies": ["M001"],  # Depends on M001 fix
        "status": TaskStatus.PENDING,
        "escalation_level": 1,
        "description": "Post-payment email confirmations failing (depends on M001)",
    },
    {
        "id": "M003",
        "type": TaskType.SUPPORT,
        "priority": 4,
        "deadline": 120.0,
        "estimated_time": 20.0,
        "customer_impact": 0.7,
        "revenue_impact": 0.4,
        "dependencies": [],
        "status": TaskStatus.PENDING,
        "escalation_level": 2,
        "description": "VIP customer threatening churn – needs immediate call",
    },
    {
        "id": "M004",
        "type": TaskType.SALES,
        "priority": 3,
        "deadline": 120.0,
        "estimated_time": 35.0,
        "customer_impact": 0.3,
        "revenue_impact": 0.9,
        "dependencies": [],
        "status": TaskStatus.PENDING,
        "escalation_level": 0,
        "description": "Finalize $500K contract – legal review needed",
    },
    {
        "id": "M005",
        "type": TaskType.FEATURE,
        "priority": 2,
        "deadline": 300.0,
        "estimated_time": 90.0,
        "customer_impact": 0.4,
        "revenue_impact": 0.3,
        "dependencies": ["M001", "M002"],  # Can't start until bugs fixed
        "status": TaskStatus.PENDING,
        "escalation_level": 0,
        "description": "Enhanced checkout UX (blocked by payment bug fixes)",
    },
    {
        "id": "M006",
        "type": TaskType.INTERNAL,
        "priority": 1,
        "deadline": 480.0,
        "estimated_time": 30.0,
        "customer_impact": 0.0,
        "revenue_impact": 0.0,
        "dependencies": [],
        "status": TaskStatus.PENDING,
        "escalation_level": 0,
        "description": "Monthly server cost review",
    },
    {
        "id": "M007",
        "type": TaskType.SUPPORT,
        "priority": 3,
        "deadline": 180.0,
        "estimated_time": 15.0,
        "customer_impact": 0.5,
        "revenue_impact": 0.2,
        "dependencies": [],
        "status": TaskStatus.PENDING,
        "escalation_level": 1,
        "description": "API rate limiting complaints from 5 customers",
    },
]

# ─── TASK 3: Hard ────────────────────────────────────────────────────────────
# Dynamic chaos, stochastic delegation, cascading failures

HARD_TASKS: List[dict] = [
    {
        "id": "H001",
        "type": TaskType.BUG,
        "priority": 5,
        "deadline": 45.0,  # Extremely tight
        "estimated_time": 60.0,  # Impossible alone
        "customer_impact": 1.0,
        "revenue_impact": 1.0,
        "dependencies": [],
        "status": TaskStatus.PENDING,
        "escalation_level": 3,  # CEO involved
        "description": "CRITICAL: Complete data outage – ALL customers affected",
    },
    {
        "id": "H002",
        "type": TaskType.BUG,
        "priority": 5,
        "deadline": 90.0,
        "estimated_time": 40.0,
        "customer_impact": 0.9,
        "revenue_impact": 0.8,
        "dependencies": ["H001"],
        "status": TaskStatus.PENDING,
        "escalation_level": 2,
        "description": "Database replication lag – data inconsistency post-outage",
    },
    {
        "id": "H003",
        "type": TaskType.SUPPORT,
        "priority": 5,
        "deadline": 60.0,
        "estimated_time": 20.0,
        "customer_impact": 0.9,
        "revenue_impact": 0.7,
        "dependencies": [],
        "status": TaskStatus.PENDING,
        "escalation_level": 3,
        "description": "150+ support tickets flooding in – triage & communicate",
    },
    {
        "id": "H004",
        "type": TaskType.SALES,
        "priority": 4,
        "deadline": 120.0,
        "estimated_time": 30.0,
        "customer_impact": 0.5,
        "revenue_impact": 1.0,
        "dependencies": [],
        "status": TaskStatus.PENDING,
        "escalation_level": 1,
        "description": "Board demo in 2hrs – product must be working",
    },
    {
        "id": "H005",
        "type": TaskType.FEATURE,
        "priority": 2,
        "deadline": 360.0,
        "estimated_time": 120.0,
        "customer_impact": 0.4,
        "revenue_impact": 0.5,
        "dependencies": ["H001", "H002"],
        "status": TaskStatus.PENDING,
        "escalation_level": 0,
        "description": "New data export feature – promised to enterprise client",
    },
    {
        "id": "H006",
        "type": TaskType.INTERNAL,
        "priority": 3,
        "deadline": 240.0,
        "estimated_time": 45.0,
        "customer_impact": 0.1,
        "revenue_impact": 0.2,
        "dependencies": ["H001"],
        "status": TaskStatus.PENDING,
        "escalation_level": 0,
        "description": "Post-mortem documentation (required for insurance claim)",
    },
    {
        "id": "H007",
        "type": TaskType.BUG,
        "priority": 4,
        "deadline": 150.0,
        "estimated_time": 35.0,
        "customer_impact": 0.6,
        "revenue_impact": 0.4,
        "dependencies": ["H001", "H002"],
        "status": TaskStatus.PENDING,
        "escalation_level": 1,
        "description": "Mobile app crashing on login after outage",
    },
    {
        "id": "H008",
        "type": TaskType.SUPPORT,
        "priority": 3,
        "deadline": 200.0,
        "estimated_time": 25.0,
        "customer_impact": 0.7,
        "revenue_impact": 0.3,
        "dependencies": [],
        "status": TaskStatus.PENDING,
        "escalation_level": 2,
        "description": "Enterprise SLA breach notification – respond or pay penalty",
    },
]

# ─── Dynamic Events (injected randomly during hard scenario) ─────────────────

DYNAMIC_EVENTS = [
    {
        "id": "DE001",
        "type": TaskType.BUG,
        "priority": 5,
        "deadline_offset": 30.0,  # 30 min from now
        "estimated_time": 20.0,
        "customer_impact": 0.8,
        "revenue_impact": 0.6,
        "dependencies": [],
        "escalation_level": 2,
        "description": "New: CDN cache poisoning discovered – inject mitigation",
    },
    {
        "id": "DE002",
        "type": TaskType.SUPPORT,
        "priority": 4,
        "deadline_offset": 45.0,
        "estimated_time": 15.0,
        "customer_impact": 0.9,
        "revenue_impact": 0.8,
        "dependencies": [],
        "escalation_level": 3,
        "description": "New: Forbes reporter asking for comment on outage",
    },
    {
        "id": "DE003",
        "type": TaskType.INTERNAL,
        "priority": 3,
        "deadline_offset": 60.0,
        "estimated_time": 20.0,
        "customer_impact": 0.0,
        "revenue_impact": 0.0,
        "dependencies": [],
        "escalation_level": 0,
        "description": "New: Key engineer just called in sick",
    },
]


def get_task_list(scenario: str) -> List[Task]:
    """Return fresh Task objects for the given scenario."""
    mapping = {
        "easy": EASY_TASKS,
        "medium": MEDIUM_TASKS,
        "hard": HARD_TASKS,
    }
    raw = mapping.get(scenario, EASY_TASKS)
    return [Task(**copy.deepcopy(d)) for d in raw]


def get_dynamic_events(current_time: float) -> List[Task]:
    """Return dynamic event tasks with deadlines relative to current_time."""
    events = []
    for d in DYNAMIC_EVENTS:
        data = copy.deepcopy(d)
        data["deadline"] = current_time + data.pop("deadline_offset")
        data["status"] = TaskStatus.PENDING
        data["started_at"] = None
        data["completed_at"] = None
        data["assigned_to"] = None
        data["delegation_attempts"] = 0
        data["actual_time"] = None
        events.append(Task(**data))
    return events
