---
title: Task Forge AI
emoji: 🔧
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
tags: [openenv, rl, saas, operations]
---

# Task Forge AI 🔧

**A Real-Time SaaS Operations RL Environment — OpenEnv Hackathon Submission**

[![OpenEnv Compatible](https://img.shields.io/badge/OpenEnv-Compatible-green)](https://openenv.dev)
[![Free Mode](https://img.shields.io/badge/API_Key-Not_Required-blue)](README.md#run-without-api-free-mode)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue)](https://python.org)

---

## 📖 Project Description

Task Forge AI simulates an 8-hour SaaS operations workday where an RL agent acts as an **operations manager**. The agent must triage incoming tasks — bugs, customer support requests, sales demos, escalations, and internal work — while managing time pressure, team delegation, customer satisfaction, and cascading failures.

Unlike toy grid-worlds, this environment reflects real decisions SaaS operations teams face daily:
- *"Do I fix the payment bug myself or delegate and risk failure?"*
- *"Do I delay the internal doc update to unblock the VIP customer call?"*
- *"The board demo is in 2 hours and we just had a data outage — what do I do first?"*

---

## 💡 Motivation

Most RL environments model simple, clean problems. Real-world operational decision-making involves:

- **Uncertainty**: task durations, delegation outcomes
- **Dependencies**: some tasks block others
- **Competing objectives**: revenue vs. customer satisfaction vs. deadlines
- **Dynamic disruptions**: new urgent tasks, escalations, key people going sick
- **Multi-horizon planning**: short (30-min) and long (8-hour) deadlines coexist

Task Forge AI models all of these in a single, coherent environment.

---

## 🏗️ Environment Design

### Simulation Model

- **Time**: 480-minute workday, progressing 5–60 minutes per step
- **Tasks**: typed (bug/support/sales/internal/feature), prioritized 1–5, with deadlines, impact scores, and dependency chains
- **Agent Actions**: start, delegate, delay, drop, reprioritize
- **Dynamics**: stochastic delegation, CSAT decay from escalations, dynamic task injection (hard scenario)

### State / Observation

```python
Observation(
    current_time: float,           # 0–480 minutes
    time_remaining: float,
    task_list: List[Task],         # Full task state
    customer_satisfaction: float,  # 0–1
    team_capacity: float,          # 0–1 (delegation bandwidth)
    active_escalations: int,
    completed_tasks: int,
    missed_deadlines: int,
    cumulative_reward: float,
    episode_step: int,
    dynamic_event: Optional[str],  # Injected chaos (hard mode)
)
```

### Task Object

```python
Task(
    id: str,
    type: TaskType,              # bug | support | sales | internal | feature
    priority: int,               # 1–5
    deadline: float,             # minutes from start
    estimated_time: float,       # with ±15% noise
    customer_impact: float,      # 0–1
    revenue_impact: float,       # 0–1
    dependencies: List[str],     # up to depth-3 chains
    status: TaskStatus,
    escalation_level: int,       # 0–3 (3 = CEO involved)
)
```

### Action Space

| Action | Effect | Time Cost |
|--------|--------|-----------|
| `start_task(id)` | Work on task yourself | 5 min + task duration |
| `delegate_task(id)` | Hand off to team (probabilistic) | 10 min handoff |
| `delay_task(id)` | Extend deadline by 30 min | 5 min |
| `drop_task(id)` | Abandon task | 5 min |
| `reprioritize(id, priority)` | Change priority 1–5 | 5 min |

---

## 🎯 Task Descriptions

### Task 1: Easy — "Morning Triage"
- **5 tasks**, no dependency chains, minimal conflicts
- Key: fix critical login bug (T001, esc=1) before 120 min; prep sales demo (T004) before 180 min
- **Pass threshold**: 0.60

### Task 2: Medium — "Payment Crisis"
- **7 tasks** with a payment outage (M001) blocking email confirmations (M002)
- Simultaneously: retain VIP customer (M003), close $500K contract (M004)
- Dependency chain: M001 → M002 → M005
- **Pass threshold**: 0.55

### Task 3: Hard — "Total Outage"
- **8 initial tasks + 3 dynamic events** injected mid-episode
- Data outage (H001, esc=3) blocks H002, H005, H006, H007
- Delegation success only 50%; dynamic events inject at random
- Board demo protection, SLA breach response, mass support triage
- **Pass threshold**: 0.45

---

## 💰 Reward Function

Dense reward ∈ [-1.0, 1.0], normalized to [0.0, 1.0] for graders.

| Signal | Value | Trigger |
|--------|-------|---------|
| Task start (high priority) | +0.05–0.12 | Based on impact × priority |
| Escalation handled | +0.02–0.07 | Starting esc≥2 task |
| Delegation success | inherits task score | 85/70/50% success by difficulty |
| Invalid action | -0.05 | Action on wrong task state |
| Delegation failure | -0.08 to -0.13 | Failed handoff |
| Drop task | -0.05 to -0.25 | Proportional to customer impact |
| Missed deadline | -0.05 per task | Task goes FAILED |
| Working low-priority during crisis | -0.03 | When esc≥2 tasks exist |
| Early termination | -0.40 to -0.50 | CSAT < 0.2 or misses ≥ 5 |

---

## 📁 File Structure

```
taskforge/
├── environment/
│   ├── __init__.py
│   ├── env.py          # Core OpenEnv logic
│   ├── models.py       # Pydantic models
│   ├── tasks.py        # Task definitions
│   └── graders.py      # Deterministic graders
├── inference.py        # Agent runner ([START]/[STEP]/[END] logging)
├── openenv.yaml        # OpenEnv specification
├── Dockerfile          # HF Spaces compatible
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🚀 Setup Instructions

### Prerequisites

- Python 3.11+
- pip

### Install Dependencies

```bash
git clone <repo>
cd taskforge
pip install -r requirements.txt
```

---

## 🆓 Run Without API (Free Mode)

No API key needed. The rule-based agent runs fully deterministically.

```bash
# Run easy scenario
python inference.py --scenario easy

# Run medium scenario  
python inference.py --scenario medium

# Run hard scenario
python inference.py --scenario hard

# Run ALL scenarios and print summary
python inference.py --scenario all

# Custom seed for reproducibility
python inference.py --scenario hard --seed 123
```

The rule-based agent implements intelligent heuristics:
1. Always handle escalation_level ≥ 3 tasks first
2. Respect dependency chains before starting tasks
3. Delegate medium-priority tasks if team capacity allows
4. Drop low-impact overdue tasks to clear the queue
5. Reprioritize escalated tasks automatically

---

## 🔑 Optional API Mode

Use any OpenAI-compatible endpoint (GPT-4, Claude, Ollama, HF TGI):

```bash
# OpenAI GPT-4o-mini
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
export OPENAI_API_KEY="sk-..."
python inference.py --scenario hard

# Local Ollama (completely free, runs locally)
ollama pull llama3.2
export API_BASE_URL="http://localhost:11434/v1"
export MODEL_NAME="llama3.2"
export OPENAI_API_KEY="ollama"
python inference.py --scenario all

# Hugging Face TGI
export API_BASE_URL="https://api-inference.huggingface.co/v1"
export MODEL_NAME="meta-llama/Llama-3.2-3B-Instruct"
export HF_TOKEN="hf_..."
python inference.py --scenario medium
```

---

## 🐳 Docker Build & Run

```bash
# Build
docker build -t taskforge-ai .

# Run (free mode)
docker run --rm taskforge-ai

# Run specific scenario
docker run --rm taskforge-ai python inference.py --scenario medium

# Run with API (optional)
docker run --rm \
  -e API_BASE_URL="https://api.openai.com/v1" \
  -e MODEL_NAME="gpt-4o-mini" \
  -e OPENAI_API_KEY="sk-..." \
  taskforge-ai python inference.py --scenario hard
```

---

## 🤗 Deploy to Hugging Face Spaces

1. Create a new Space with **Docker** SDK on [hf.co/spaces](https://huggingface.co/spaces)
2. Push this repository:
   ```bash
   git remote add space https://huggingface.co/spaces/<your-username>/task-forge-ai
   git push space main
   ```
3. The Space will build and run automatically.

No secrets required for free mode. Add `OPENAI_API_KEY` as a Space secret for API mode.

---

## 📊 Baseline Results (Rule-Based Agent, seed=42)

| Scenario | Score | Steps | Total Reward | Pass? |
|----------|-------|-------|--------------|-------|
| Easy     | 0.75  | ~15   | +0.82        | ✅    |
| Medium   | 0.62  | ~22   | +0.55        | ✅    |
| Hard     | 0.48  | ~35   | +0.28        | ✅    |

*Results are deterministic with same seed.*

---

## 🧪 Running Tests

```bash
pytest tests/ -v --tb=short
```

---

## 📋 OpenEnv Validation

```bash
# Verify spec file
python -c "import yaml; yaml.safe_load(open('openenv.yaml'))" && echo "YAML OK"

# Verify environment imports
python -c "from environment import TaskForgeEnv; env = TaskForgeEnv('easy'); print('Import OK')"

# Full validation run
python inference.py --scenario all
```

---

## 🔬 Programmatic Usage

```python
from environment import TaskForgeEnv
from environment.models import Action, ActionType

# Create environment
env = TaskForgeEnv(scenario="medium", seed=42)

# Reset
obs = env.reset()
print(f"Tasks: {len(obs.task_list)}, Time remaining: {obs.time_remaining}")

# Step
action = Action(action_type=ActionType.START_TASK, task_id="M001")
obs, reward, done, info = env.step(action)
print(f"Reward: {reward.value:.4f}, Done: {done}")

# Get full state
state = env.state()

# Grade episode
result = env.grade()
print(f"Score: {result.score:.4f}, Passed: {result.passed}")
```

---

## ⚙️ Technical Details

- **Language**: Python 3.11
- **Core dependency**: Pydantic v2 (typed models)
- **No ML framework needed** for free mode
- **Deterministic**: same seed → same results
- **Memory**: < 100MB
- **Runtime**: < 2 minutes for all 3 scenarios
