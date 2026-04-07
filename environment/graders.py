"""
Deterministic graders for Task Forge AI.
Each grader returns a score ∈ [0.0, 1.0] based on episode outcomes.
"""

from __future__ import annotations
from typing import Dict, Any, List
from environment.models import Task, TaskStatus, EpisodeResult


def _compute_deadline_rate(tasks: List[Task]) -> float:
    """Fraction of completable tasks finished before deadline."""
    completable = [t for t in tasks if t.status in (
        TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.DROPPED, TaskStatus.IN_PROGRESS
    )]
    if not completable:
        return 0.0
    on_time = sum(
        1 for t in tasks
        if t.status == TaskStatus.COMPLETED
        and t.completed_at is not None
        and t.completed_at <= t.deadline
    )
    return on_time / max(len(completable), 1)


def _weighted_completion_score(tasks: List[Task]) -> float:
    """Weighted score: high-priority + high-impact tasks count more."""
    total_weight = 0.0
    earned_weight = 0.0
    for t in tasks:
        weight = (t.priority / 5.0) * 0.4 + t.customer_impact * 0.4 + t.revenue_impact * 0.2
        total_weight += weight
        if t.status == TaskStatus.COMPLETED:
            on_time_bonus = 1.0
            if t.completed_at is not None and t.completed_at <= t.deadline:
                on_time_bonus = 1.2  # 20% bonus for on-time
            elif t.completed_at is not None and t.completed_at > t.deadline:
                # Partial credit, decaying with lateness
                late_ratio = (t.completed_at - t.deadline) / max(t.deadline, 1.0)
                on_time_bonus = max(0.3, 1.0 - late_ratio)
            earned_weight += weight * min(on_time_bonus, 1.0)
    if total_weight == 0:
        return 0.0
    return min(earned_weight / total_weight, 1.0)


def _escalation_response_score(tasks: List[Task], missed_deadlines: int) -> float:
    """Score based on how well escalations were handled."""
    escalated = [t for t in tasks if t.escalation_level >= 2]
    if not escalated:
        return 1.0
    resolved = sum(1 for t in escalated if t.status == TaskStatus.COMPLETED)
    base = resolved / len(escalated)
    # Penalise for each missed deadline
    penalty = min(0.5, missed_deadlines * 0.1)
    return max(0.0, base - penalty)


# ─── GRADER 1: Easy ──────────────────────────────────────────────────────────

def grade_easy(
    tasks: List[Task],
    customer_satisfaction: float,
    missed_deadlines: int,
    episode_steps: int,
    early_terminated: bool,
) -> EpisodeResult:
    """
    Easy grader — rewards completing the critical bug (T001) and
    sales task (T004) before deadline. Minimal conflict resolution needed.
    """
    details: Dict[str, Any] = {}

    # Critical task completion (T001 — bug fix)
    t001 = next((t for t in tasks if t.id == "T001"), None)
    t004 = next((t for t in tasks if t.id == "T004"), None)

    t001_score = 0.0
    if t001:
        if t001.status == TaskStatus.COMPLETED:
            if t001.completed_at is not None and t001.completed_at <= t001.deadline:
                t001_score = 1.0
            else:
                t001_score = 0.5  # Completed but late
        details["T001_status"] = t001.status
        details["T001_on_time"] = (
            t001.completed_at is not None and t001.completed_at <= t001.deadline
        )

    t004_score = 0.0
    if t004:
        if t004.status == TaskStatus.COMPLETED:
            if t004.completed_at is not None and t004.completed_at <= t004.deadline:
                t004_score = 1.0
            else:
                t004_score = 0.4
        details["T004_status"] = t004.status

    # Overall efficiency
    deadline_rate = _compute_deadline_rate(tasks)
    weighted_score = _weighted_completion_score(tasks)
    csat_score = customer_satisfaction

    # Early termination hard penalty
    early_penalty = 0.4 if early_terminated else 0.0

    score = (
        t001_score * 0.35
        + t004_score * 0.20
        + deadline_rate * 0.15
        + weighted_score * 0.15
        + csat_score * 0.15
    ) - early_penalty

    score = max(0.001, min(0.999, score))
    details.update({
        "deadline_rate": deadline_rate,
        "weighted_completion": weighted_score,
        "customer_satisfaction": customer_satisfaction,
        "missed_deadlines": missed_deadlines,
        "early_terminated": early_terminated,
        "episode_steps": episode_steps,
    })

    return EpisodeResult(
        task_id="easy",
        score=round(score, 4),
        details=details,
        passed=score >= 0.6,
    )


# ─── GRADER 2: Medium ────────────────────────────────────────────────────────

def grade_medium(
    tasks: List[Task],
    customer_satisfaction: float,
    missed_deadlines: int,
    episode_steps: int,
    early_terminated: bool,
) -> EpisodeResult:
    """
    Medium grader — rewards handling payment bug (M001) first (dependency chain),
    retaining VIP customer (M003), and closing the contract (M004).
    """
    details: Dict[str, Any] = {}

    m001 = next((t for t in tasks if t.id == "M001"), None)
    m002 = next((t for t in tasks if t.id == "M002"), None)
    m003 = next((t for t in tasks if t.id == "M003"), None)
    m004 = next((t for t in tasks if t.id == "M004"), None)

    # M001 — payment bug (critical path)
    m001_score = 0.0
    if m001 and m001.status == TaskStatus.COMPLETED:
        ratio = (m001.deadline - (m001.completed_at or m001.deadline)) / max(m001.deadline, 1)
        m001_score = min(1.0, 0.7 + ratio * 0.3)
    details["M001_completed"] = m001.status == TaskStatus.COMPLETED if m001 else False

    # Dependency chain adherence: M002 only starts after M001
    dep_bonus = 0.0
    if m002 and m001:
        if m002.status == TaskStatus.COMPLETED and m001.status == TaskStatus.COMPLETED:
            # Reward correct sequencing
            if (m001.completed_at or 0) <= (m002.started_at or float("inf")):
                dep_bonus = 0.15
    details["dependency_chain_respected"] = dep_bonus > 0

    # M003 — VIP customer retention
    m003_score = 0.0
    if m003 and m003.status == TaskStatus.COMPLETED:
        m003_score = customer_satisfaction  # CSAT is the measure
    details["M003_completed"] = m003.status == TaskStatus.COMPLETED if m003 else False

    # M004 — contract closure
    m004_score = 0.0
    if m004 and m004.status == TaskStatus.COMPLETED:
        if m004.completed_at is not None and m004.completed_at <= m004.deadline:
            m004_score = 1.0
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

    score = max(0.001, min(0.999, score))
    details.update({
        "escalation_score": escalation_score,
        "weighted_completion": weighted,
        "customer_satisfaction": customer_satisfaction,
        "missed_deadlines": missed_deadlines,
        "early_terminated": early_terminated,
    })

    return EpisodeResult(
        task_id="medium",
        score=round(score, 4),
        details=details,
        passed=score >= 0.55,
    )


# ─── GRADER 3: Hard ──────────────────────────────────────────────────────────

def grade_hard(
    tasks: List[Task],
    customer_satisfaction: float,
    missed_deadlines: int,
    episode_steps: int,
    early_terminated: bool,
) -> EpisodeResult:
    """
    Hard grader — chaotic environment. Rewards decisive triage of the outage
    (H001), stakeholder communication (H003), and protecting the board demo (H004).
    Penalties are severe for cascade failures.
    """
    details: Dict[str, Any] = {}

    h001 = next((t for t in tasks if t.id == "H001"), None)
    h003 = next((t for t in tasks if t.id == "H003"), None)
    h004 = next((t for t in tasks if t.id == "H004"), None)
    h008 = next((t for t in tasks if t.id == "H008"), None)

    # H001 — outage resolution (highest weight)
    h001_score = 0.0
    if h001:
        if h001.status == TaskStatus.COMPLETED:
            # Even late completion is valuable here
            h001_score = 0.8
            if h001.completed_at is not None and h001.completed_at <= h001.deadline:
                h001_score = 1.0
        elif h001.status == TaskStatus.DELEGATED:
            h001_score = 0.5  # Delegation is acceptable for outage
    details["H001_score"] = h001_score

    # H003 — communication under pressure
    h003_score = 0.0
    if h003 and h003.status == TaskStatus.COMPLETED:
        if h003.completed_at is not None and h003.completed_at <= h003.deadline:
            h003_score = 1.0
        else:
            h003_score = 0.5
    details["H003_completed"] = h003.status == TaskStatus.COMPLETED if h003 else False

    # H004 — board demo protection
    h004_score = 0.0
    if h004 and h004.status == TaskStatus.COMPLETED:
        if h004.completed_at is not None and h004.completed_at <= h004.deadline:
            h004_score = 1.0
        else:
            h004_score = 0.2
    details["H004_score"] = h004_score

    # H008 — SLA breach response
    h008_score = 0.0
    if h008 and h008.status == TaskStatus.COMPLETED:
        h008_score = 0.8
    details["H008_completed"] = h008.status == TaskStatus.COMPLETED if h008 else False

    # Cascade penalty: dynamic events missed
    dynamic_tasks = [t for t in tasks if t.id.startswith("DE")]
    dynamic_missed = sum(1 for t in dynamic_tasks if t.status not in (
        TaskStatus.COMPLETED, TaskStatus.DELEGATED
    ))
    cascade_penalty = min(0.25, dynamic_missed * 0.1)
    details["cascade_penalty"] = cascade_penalty

    # Missed deadline penalty for hard scenario
    deadline_penalty = min(0.30, missed_deadlines * 0.05)

    escalation_score = _escalation_response_score(tasks, missed_deadlines)
    csat_score = customer_satisfaction

    early_penalty = 0.5 if early_terminated else 0.0

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

    score = max(0.001, min(0.999, score))
    details.update({
        "escalation_score": escalation_score,
        "customer_satisfaction": customer_satisfaction,
        "missed_deadlines": missed_deadlines,
        "early_terminated": early_terminated,
    })

    return EpisodeResult(
        task_id="hard",
        score=round(score, 4),
        details=details,
        passed=score >= 0.45,
    )


# ─── Dispatcher ──────────────────────────────────────────────────────────────

GRADERS = {
    "easy": grade_easy,
    "medium": grade_medium,
    "hard": grade_hard,
}


def _clamp(score: float) -> float:
    """Clamp score to strictly open interval (0, 1) as required by Phase 2."""
    return round(max(0.001, min(0.999, score)), 4)


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
    result.score = _clamp(result.score)
    return result