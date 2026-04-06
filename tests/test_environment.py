"""
Tests for Task Forge AI environment.
Validates OpenEnv compliance and grader correctness.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from environment.env import TaskForgeEnv
from environment.models import Action, ActionType, Observation, Reward, TaskStatus
from environment.graders import grade_easy, grade_medium, grade_hard
from environment.tasks import get_task_list, get_dynamic_events


# ─── Environment Tests ───────────────────────────────────────────────────────

class TestEnvironmentSpec:
    """Validates OpenEnv spec compliance."""

    def test_reset_returns_observation(self):
        env = TaskForgeEnv("easy", seed=42)
        obs = env.reset()
        assert isinstance(obs, Observation)
        assert obs.current_time == 0.0
        assert obs.time_remaining == 480.0
        assert len(obs.task_list) > 0
        assert 0.0 <= obs.customer_satisfaction <= 1.0

    def test_step_returns_correct_types(self):
        env = TaskForgeEnv("easy", seed=42)
        env.reset()
        action = Action(action_type=ActionType.START_TASK, task_id="T001")
        obs, reward, done, info = env.step(action)
        assert isinstance(obs, Observation)
        assert isinstance(reward, Reward)
        assert isinstance(done, bool)
        assert isinstance(info, dict)

    def test_state_returns_dict(self):
        env = TaskForgeEnv("easy", seed=42)
        env.reset()
        state = env.state()
        assert isinstance(state, dict)
        assert "current_time" in state
        assert "tasks" in state
        assert "done" in state

    def test_reward_in_valid_range(self):
        env = TaskForgeEnv("easy", seed=42)
        env.reset()
        for task_id in ["T001", "T002", "T003"]:
            action = Action(action_type=ActionType.START_TASK, task_id=task_id)
            _, reward, done, _ = env.step(action)
            assert -1.0 <= reward.value <= 1.0
            assert 0.0 <= reward.normalized <= 1.0
            if done:
                break

    def test_done_after_reset_is_false(self):
        env = TaskForgeEnv("easy", seed=42)
        env.reset()
        assert env._done is False

    def test_error_on_step_when_done(self):
        env = TaskForgeEnv("easy", seed=42)
        env.reset()
        env._done = True
        with pytest.raises(RuntimeError):
            env.step(Action(action_type=ActionType.DROP_TASK, task_id="T001"))

    def test_invalid_action_handled_gracefully(self):
        env = TaskForgeEnv("easy", seed=42)
        env.reset()
        # Non-existent task
        action = Action(action_type=ActionType.START_TASK, task_id="INVALID_999")
        obs, reward, done, info = env.step(action)
        assert not info["action_valid"]
        assert "INVALID_999" in info.get("error", "")

    def test_dependency_validation(self):
        env = TaskForgeEnv("medium", seed=42)
        obs = env.reset()
        # M002 depends on M001 — should be invalid to start before M001
        action = Action(action_type=ActionType.START_TASK, task_id="M002")
        valid, reason = action.validate_action(obs)
        assert not valid
        assert "M001" in reason

    def test_deterministic_with_same_seed(self):
        def run_ep(seed):
            env = TaskForgeEnv("hard", seed=seed)
            obs = env.reset()
            rewards = []
            for task_id in ["H001", "H003", "H004"]:
                a = Action(action_type=ActionType.START_TASK, task_id=task_id)
                _, r, done, _ = env.step(a)
                rewards.append(round(r.value, 6))
                if done:
                    break
            return rewards

        assert run_ep(42) == run_ep(42)

    def test_different_seeds_differ(self):
        def first_est_time(seed):
            env = TaskForgeEnv("easy", seed=seed)
            env.reset()
            return env._tasks[0].estimated_time

        assert first_est_time(42) != first_est_time(99)


# ─── Scenario Tests ───────────────────────────────────────────────────────────

class TestScenarios:
    @pytest.mark.parametrize("scenario", ["easy", "medium", "hard"])
    def test_scenario_loads(self, scenario):
        env = TaskForgeEnv(scenario, seed=42)
        obs = env.reset()
        assert len(obs.task_list) > 0

    def test_easy_has_5_tasks(self):
        env = TaskForgeEnv("easy", seed=42)
        obs = env.reset()
        assert len(obs.task_list) == 5

    def test_medium_has_7_tasks(self):
        env = TaskForgeEnv("medium", seed=42)
        obs = env.reset()
        assert len(obs.task_list) == 7

    def test_hard_has_8_tasks(self):
        env = TaskForgeEnv("hard", seed=42)
        obs = env.reset()
        assert len(obs.task_list) == 8

    def test_hard_highest_escalation(self):
        env = TaskForgeEnv("hard", seed=42)
        obs = env.reset()
        max_esc = max(t.escalation_level for t in obs.task_list)
        assert max_esc == 3


# ─── Action Tests ─────────────────────────────────────────────────────────────

class TestActions:
    def test_start_task_changes_status(self):
        env = TaskForgeEnv("easy", seed=42)
        env.reset()
        action = Action(action_type=ActionType.START_TASK, task_id="T001")
        obs, _, _, info = env.step(action)
        # Task should be in_progress or completed (if very short)
        t001 = next(t for t in obs.task_list if t.id == "T001")
        assert t001.status in (TaskStatus.IN_PROGRESS, TaskStatus.COMPLETED)

    def test_drop_task_marks_dropped(self):
        env = TaskForgeEnv("easy", seed=42)
        env.reset()
        action = Action(action_type=ActionType.DROP_TASK, task_id="T003")
        obs, reward, _, info = env.step(action)
        t003 = next(t for t in obs.task_list if t.id == "T003")
        assert t003.status == TaskStatus.DROPPED
        # Should have negative reward for dropping
        assert reward.value <= 0.0

    def test_reprioritize_changes_priority(self):
        env = TaskForgeEnv("easy", seed=42)
        env.reset()
        action = Action(action_type=ActionType.REPRIORITIZE, task_id="T005", new_priority=5)
        obs, _, _, _ = env.step(action)
        t005 = next(t for t in obs.task_list if t.id == "T005")
        assert t005.priority == 5

    def test_delay_extends_deadline(self):
        env = TaskForgeEnv("easy", seed=42)
        obs = env.reset()
        original_deadline = next(t for t in obs.task_list if t.id == "T005").deadline
        action = Action(action_type=ActionType.DELAY_TASK, task_id="T005")
        obs, _, _, _ = env.step(action)
        new_deadline = next(t for t in obs.task_list if t.id == "T005").deadline
        assert new_deadline == original_deadline + 30.0

    def test_delay_rejected_for_escalated(self):
        env = TaskForgeEnv("hard", seed=42)
        obs = env.reset()
        # H001 has escalation_level=3 — cannot be delayed
        action = Action(action_type=ActionType.DELAY_TASK, task_id="H001")
        _, reward, _, info = env.step(action)
        assert info["action_result"] == "delay_rejected_escalated"


# ─── Grader Tests ─────────────────────────────────────────────────────────────

class TestGraders:
    def _make_tasks(self, scenario):
        return get_task_list(scenario)

    def test_easy_grader_score_in_range(self):
        tasks = self._make_tasks("easy")
        result = grade_easy(tasks, 0.8, 0, 10, False)
        assert 0.0 <= result.score <= 1.0

    def test_medium_grader_score_in_range(self):
        tasks = self._make_tasks("medium")
        result = grade_medium(tasks, 0.7, 1, 20, False)
        assert 0.0 <= result.score <= 1.0

    def test_hard_grader_score_in_range(self):
        tasks = self._make_tasks("hard")
        result = grade_hard(tasks, 0.5, 3, 30, False)
        assert 0.0 <= result.score <= 1.0

    def test_early_termination_penalised(self):
        tasks = self._make_tasks("easy")
        result_normal = grade_easy(tasks, 0.8, 0, 10, False)
        result_early = grade_easy(tasks, 0.8, 0, 10, True)
        assert result_early.score < result_normal.score

    def test_completing_critical_task_improves_score(self):
        tasks_base = self._make_tasks("easy")
        tasks_done = self._make_tasks("easy")

        # Complete T001 on time
        t001 = next(t for t in tasks_done if t.id == "T001")
        t001.status = TaskStatus.COMPLETED
        t001.completed_at = 60.0  # Before deadline of 120

        score_base = grade_easy(tasks_base, 0.8, 0, 10, False).score
        score_done = grade_easy(tasks_done, 0.8, 0, 10, False).score
        assert score_done > score_base

    def test_episode_result_has_required_fields(self):
        tasks = self._make_tasks("easy")
        result = grade_easy(tasks, 0.8, 0, 10, False)
        assert hasattr(result, "task_id")
        assert hasattr(result, "score")
        assert hasattr(result, "details")
        assert hasattr(result, "passed")


# ─── Full Episode Test ────────────────────────────────────────────────────────

class TestFullEpisode:
    """Run a complete episode with the rule-based agent logic."""

    def _greedy_action(self, obs):
        """Simple greedy: always start highest-priority pending task."""
        from environment.models import TaskStatus
        ready = [
            t for t in obs.task_list
            if t.status == TaskStatus.PENDING
        ]
        if not ready:
            # Drop the least important
            pending = [t for t in obs.task_list if t.status in (
                TaskStatus.PENDING, TaskStatus.DELAYED
            )]
            if pending:
                worst = min(pending, key=lambda t: t.priority)
                return Action(action_type=ActionType.DROP_TASK, task_id=worst.id)
            return Action(action_type=ActionType.START_TASK, task_id=obs.task_list[0].id)
        best = max(ready, key=lambda t: t.priority)
        return Action(action_type=ActionType.START_TASK, task_id=best.id)

    @pytest.mark.parametrize("scenario", ["easy", "medium", "hard"])
    def test_episode_completes(self, scenario):
        env = TaskForgeEnv(scenario, seed=42)
        obs = env.reset()
        done = False
        steps = 0
        while not done and steps < 100:
            action = self._greedy_action(obs)
            obs, reward, done, info = env.step(action)
            assert -1.0 <= reward.value <= 1.0
            steps += 1
        assert done or steps == 100

    @pytest.mark.parametrize("scenario", ["easy", "medium", "hard"])
    def test_grade_returns_valid_score(self, scenario):
        env = TaskForgeEnv(scenario, seed=42)
        env.reset()
        # Do a few steps
        for task_id in [env._tasks[0].id, env._tasks[-1].id]:
            action = Action(action_type=ActionType.DROP_TASK, task_id=task_id)
            _, _, done, _ = env.step(action)
            if done:
                break
        result = env.grade()
        assert 0.0 <= result.score <= 1.0
