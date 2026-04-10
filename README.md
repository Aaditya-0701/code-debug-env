---
title: CodeDebugEnv
emoji: 🐛
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# CodeDebugEnv

An OpenEnv environment where agents are given buggy Python code and an objective. They must return fixed code. The fixed code is tested automatically against unit tests.

This environment is built with the [OpenEnv framework](https://github.com/meta-pytorch/OpenEnv) to demonstrate isolated, programmatic evaluation of LLMs solving coding tasks.

## Why this Environment?
This is designed for LLM fine-tuning using reinforcement learning (like GRPO). 
1. **Clear Task**: The LLM gets a block of Python code intentionally seeded with a bug.
2. **Deterministic Grader**: The LLM's returned string is pulled out and executed against hidden asserts.
3. **Reward Signal**: If all asserts pass, reward is `1.0`. Any exception or failure results in `0.0`.

## Installation

```bash
pip install openenv-core
git clone https://huggingface.co/spaces/<your-username>/code-debug-env
cd code-debug-env
uv sync
```

## Running the Server Locally
```bash
uv run server
# or using Docker:
docker build -t code-debug-env -f server/Dockerfile .
docker run -p 8000:8000 code-debug-env
```

## Using the Client

```python
from code_debug_env import CodeDebugEnv, CodeDebugAction

with CodeDebugEnv(base_url="http://localhost:8000").sync() as env:
    obs = env.reset()
    
    print(f"Buggy Code:\\n{obs.observation.buggy_code}")
    print(f"Hint: {obs.observation.hint}")
    
    # Try an action
    fixed_code = '...' # Your fixed python code
    result = env.step(CodeDebugAction(fixed_code=fixed_code))
    
    print(f"Passed? {result.done and result.reward == 1.0}")
    print(f"Test Result: {result.observation.test_results}")
```

## Types and Models

Everything is statically typed with Pydantic:

**Action:**
- `fixed_code` (str): the replacement Python code snippet

**Observation:**
- `buggy_code` (str)
- `test_results` (str)
- `hint` (str)
- `attempts_remaining` (int)

**State:**
- `correct_solution` (str)
- `test_cases` (str)
- `max_attempts` (int)
