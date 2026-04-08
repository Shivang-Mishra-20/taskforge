"""
Deterministic graders for Task Forge AI.
Each grader returns a score strictly in (0.001, 0.999) as required by Phase 2.
"""

from __future__ import annotations
from typing import Dict, Any, List
from environment.models import Task, TaskStatus, EpisodeResult


def _clamp(score: float) -> float:
    epsilon = 1e-4

    score = max(epsilon, min(1 - epsilon, score))

    if score <= 0.0:
        score = epsilon
    elif score >= 1.0:
        score = 1 - epsilon

    return score


def _compute_deadline_rate(tasks: List[Task]) -> float:
    completable = [t for t in tasks if t.status in (
        TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.DROPPED, TaskStatus.IN_PROGRESS
    )]
    if not completable:
        return 0.5
    on_time = sum(
        1 for t in tasks
        if t.status == TaskStatus.COMPLETED
        and t.completed_at is not None
        and t.completed_at <= t.deadline
    )
    return on_time / max(len(completable), 1)


def _weighted_completion_score(tasks: List[Task]) -> float:
    total_weight = 0.0
    earned_weight = 0.0
    for t in tasks:
        weight = (t.priority / 5.0) * 0.4 + t.customer_impact * 0.4 + t.revenue_impact * 0.2
        total_weight += weight
        if t.status == TaskStatus.COMPLETED:
            on_time_bonus = 1.0
            if t.completed_at is not None and t.completed_at <= t.deadline:
                on_time_bonus = 1.1
            elif t.completed_at is not None and t.completed_at > t.deadline:
                late_ratio = (t.completed_at - t.deadline) / max(t.deadline, 1.0)
                on_time_bonus = max(0.3, 1.0 - late_ratio)
            earned_weight += weight * min(on_time_bonus, 1.0)
    if total_weight == 0:
        return 0.5
    return earned_weight / total_weight


def _escalation_response_score(tasks: List[Task], missed_deadlines: int) -> float:
    escalated = [t for t in tasks if t.escalation_level >= 2]
    if not escalated:
        return 0.9
    resolved = sum(1 for t in escalated if t.status == TaskStatus.COMPLETED)
    base = resolved / len(escalated)
    penalty = min(0.5, missed_deadlines * 0.1)
    return max(0.001, base - penalty)


# ─── GRADER 1: Easy ──────────────────────────────────────────────────────────

def grade_easy(
    tasks: List[Task],
    customer_satisfaction: float,
    missed_deadlines: int,
    episode_steps: int,
    early_terminated: bool,
) -> EpisodeResult:
    details: Dict[str, Any] = {}

    t001 = next((t for t in tasks if t.id == "T001"), None)
    t004 = next((t for t in tasks if t.id == "T004"), None)

    t001_score = 0.1
    if t001:
        if t001.status == TaskStatus.COMPLETED:
            if t001.completed_at is not None and t001.completed_at <= t001.deadline:
                t001_score = 0.95
            else:
                t001_score = 0.5
        details["T001_status"] = t001.status
        details["T001_on_time"] = (
            t001.completed_at is not None and t001.completed_at <= t001.deadline
        )

    t004_score = 0.1
    if t004:
        if t004.status == TaskStatus.COMPLETED:
            if t004.completed_at is not None and t004.completed_at <= t004.deadline:
                t004_score = 0.95
            else:
                t004_score = 0.4
        details["T004_status"] = t004.status

    deadline_rate = _compute_deadline_rate(tasks)
    weighted_score = _weighted_completion_score(tasks)
    csat_score = min(0.95, customer_satisfaction)
    early_penalty = 0.4 if early_terminated else 0.0

    score = (
        t001_score * 0.35
        + t004_score * 0.20
        + deadline_rate * 0.15
        + weighted_score * 0.15
        + csat_score * 0.15
    ) - early_penalty

    details.update({
        "deadline_rate": deadline_rate,
        "weighted_completion": weighted_score,
        "customer_satisfaction": customer_satisfaction,
        "missed_deadlines": missed_deadlines,
        "early_terminated": early_terminated,
        "episode_steps": episode_steps,
    })

    final = _clamp(score)
    return EpisodeResult(
        task_id="easy",
        score=final,
        details=details,
        passed=final >= 0.6,
    )


# ─── GRADER 2: Medium ────────────────────────────────────────────────────────

def grade_medium(
    tasks: List[Task],
    customer_satisfaction: float,
    missed_deadlines: int,
    episode_steps: int,
    early_terminated: bool,
) -> EpisodeResult:
    details: Dict[str, Any] = {}

    m001 = next((t for t in tasks if t.id == "M001"), None)
    m002 = next((t for t in tasks if t.id == "M002"), None)
    m003 = next((t for t in tasks if t.id == "M003"), None)
    m004 = next((t for t in tasks if t.id == "M004"), None)

    m001_score = 0.05
    if m001 and m001.status == TaskStatus.COMPLETED:
        ratio = (m001.deadline - (m001.completed_at or m001.deadline)) / max(m001.deadline, 1)
        m001_score = min(0.95, 0.7 + ratio * 0.25)
    details["M001_completed"] = m001.status == TaskStatus.COMPLETED if m001 else False

    dep_bonus = 0.0
    if m002 and m001:
        if m002.status == TaskStatus.COMPLETED and m001.status == TaskStatus.COMPLETED:
            if (m001.completed_at or 0) <= (m002.started_at or float("inf")):
                dep_bonus = 0.14
    details["dependency_chain_respected"] = dep_bonus > 0

    m003_score = 0.05
    if m003 and m003.status == TaskStatus.COMPLETED:
        m003_score = min(0.95, customer_satisfaction)
    details["M003_completed"] = m003.status == TaskStatus.COMPLETED if m003 else False

    m004_score = 0.05
    if m004 and m004.status == TaskStatus.COMPLETED:
        if m004.completed_at is not None and m004.completed_at <= m004.deadline:
            m004_score = 0.95
        else:
            m004_score = 0.3
    details["M004_completed"] = m004.status == TaskStatus.COMPLETED if m004 else False

    escalation_score = _escalation_response_score(tasks, missed_deadlines)
    weighted = _weighted_completion_score(tasks)
    early_penalty = 0.4 if early_terminated else 0.0

    score = (
        m001_score * 0.25
        + dep_bonus
        + m003_score * 0.20
        + m004_score * 0.15
        + escalation_score * 0.15
        + weighted * 0.10
    ) - early_penalty

    details.update({
        "escalation_score": escalation_score,
        "weighted_completion": weighted,
        "customer_satisfaction": customer_satisfaction,
        "missed_deadlines": missed_deadlines,
        "early_terminated": early_terminated,
    })

    final = _clamp(score)
    return EpisodeResult(
        task_id="medium",
        score=final,
        details=details,
        passed=final >= 0.55,
    )


# ─── GRADER 3: Hard ──────────────────────────────────────────────────────────

def grade_hard(
    tasks: List[Task],
    customer_satisfaction: float,
    missed_deadlines: int,
    episode_steps: int,
    early_terminated: bool,
) -> EpisodeResult:
    details: Dict[str, Any] = {}

    h001 = next((t for t in tasks if t.id == "H001"), None)
    h003 = next((t for t in tasks if t.id == "H003"), None)
    h004 = next((t for t in tasks if t.id == "H004"), None)
    h008 = next((t for t in tasks if t.id == "H008"), None)

    h001_score = 0.05
    if h001:
        if h001.status == TaskStatus.COMPLETED:
            h001_score = 0.78
            if h001.completed_at is not None and h001.completed_at <= h001.deadline:
                h001_score = 0.95
        elif h001.status == TaskStatus.DELEGATED:
            h001_score = 0.5
    details["H001_score"] = h001_score

    h003_score = 0.05
    if h003 and h003.status == TaskStatus.COMPLETED:
        if h003.completed_at is not None and h003.completed_at <= h003.deadline:
            h003_score = 0.95
        else:
            h003_score = 0.5
    details["H003_completed"] = h003.status == TaskStatus.COMPLETED if h003 else False

    h004_score = 0.05
    if h004 and h004.status == TaskStatus.COMPLETED:
        if h004.completed_at is not None and h004.completed_at <= h004.deadline:
            h004_score = 0.95
        else:
            h004_score = 0.2
    details["H004_score"] = h004_score

    h008_score = 0.05
    if h008 and h008.status == TaskStatus.COMPLETED:
        h008_score = 0.78
    details["H008_completed"] = h008.status == TaskStatus.COMPLETED if h008 else False

    dynamic_tasks = [t for t in tasks if t.id.startswith("DE")]
    dynamic_missed = sum(1 for t in dynamic_tasks if t.status not in (
        TaskStatus.COMPLETED, TaskStatus.DELEGATED
    ))
    cascade_penalty = min(0.24, dynamic_missed * 0.1)
    details["cascade_penalty"] = cascade_penalty

    deadline_penalty = min(0.29, missed_deadlines * 0.05)
    escalation_score = _escalation_response_score(tasks, missed_deadlines)
    csat_score = min(0.95, customer_satisfaction)
    early_penalty = 0.49 if early_terminated else 0.0

    score = (
        h001_score * 0.30
        + h003_score * 0.15
        + h004_score * 0.15
        + h008_score * 0.10
        + escalation_score * 0.10
        + csat_score * 0.10
        - cascade_penalty
        - deadline_penalty
        - early_penalty
    )

    details.update({
        "escalation_score": escalation_score,
        "customer_satisfaction": customer_satisfaction,
        "missed_deadlines": missed_deadlines,
        "early_terminated": early_terminated,
    })

    final = _clamp(score)
    return EpisodeResult(
        task_id="hard",
        score=final,
        details=details,
        passed=final >= 0.45,
    )


# ─── Dispatcher ──────────────────────────────────────────────────────────────

GRADERS = {
    "easy":   grade_easy,
    "medium": grade_medium,
    "hard":   grade_hard,
}


def grade_episode(
    scenario: str,
    tasks: List[Task],
    customer_satisfaction: float,
    missed_deadlines: int,
    episode_steps: int,
    early_terminated: bool,
) -> EpisodeResult:
    grader = GRADERS.get(scenario, grade_easy)
    result = grader(
        tasks=tasks,
        customer_satisfaction=customer_satisfaction,
        missed_deadlines=missed_deadlines,
        episode_steps=episode_steps,
        early_terminated=early_terminated,
    )
    # Final safety clamp — score is ALWAYS strictly between 0 and 1
    result.score = _clamp(result.score)
    return result
