"""
inference.py — CodeDebugEnv AI Agent
"""

from __future__ import annotations

import json
import os
import sys

from openai import OpenAI

# Add project root to path (needed when running from a different directory)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import CodeDebugAction
from server.code_debug_env_environment import CodeDebugEnvironment

# ── Environment variable setup ────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1/")
MODEL_NAME   = os.getenv("MODEL_NAME",   "meta-llama/Meta-Llama-3-8B-Instruct")
# You must export HF_TOKEN in your terminal before running this script
HF_TOKEN     = os.getenv("HF_TOKEN")

SYSTEM_PROMPT = """You are an expert Python programming assistant.
Your job is to fix the buggy code presented to you.

AVAILABLE ACTION (respond with exactly one JSON object per turn):

To submit your fixed code:
{"fixed_code": "def my_func():\\n    # fixed code goes here"}

1. RESPOND: Respond with ONLY a valid JSON object matching the schema.
2. NO MARKDOWN: Do not wrap the JSON output in markdown ```json ... ``` blocks.
3. FIX: Analyze the bug logically before providing the fix in your response. 
Ensure proper indentation in your fixed_code string."""

def call_llm(client: OpenAI, messages: list[dict]) -> dict:
    """Call the LLM and parse its JSON response into a dict."""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=250,
            temperature=0.1,
            timeout=30.0,
        )
        content = response.choices[0].message.content.strip()

        # Extract JSON even if the model wraps it in markdown code blocks
        if "```" in content:
            start = content.find("{")
            end   = content.rfind("}") + 1
            content = content[start:end]

        return json.loads(content)

    except Exception as e:
        print(f"  [WARN] LLM parse error: {e}", file=sys.stderr, flush=True)
        return {"fixed_code": "def failed_fallback(): pass"}

def run_task(env: CodeDebugEnvironment, client: OpenAI, task_level: int) -> float:
    """Run one task level and return the final reward score."""
    
    # ── MANDATORY LOG: START ──────────────────────────────────────────────
    print(f"[START] task=task_{task_level} env=code-debug-env model={MODEL_NAME}", flush=True)

    obs = env.reset(task_level=task_level)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": (
            "TASK: Fix the following python block:\n\n"
            f"```python\n{obs.buggy_code}\n```\n\n"
            f"HINT: {obs.hint}\n\n"
            "Respond with a JSON object containing the 'fixed_code' key."
        )},
    ]

    final_score = 0.0
    max_steps = 5
    rewards = []

    for step_num in range(1, max_steps + 1):
        action_dict = call_llm(client, messages)

        # Build a valid CodeDebugAction
        try:
            action = CodeDebugAction(**action_dict)
        except Exception:
            action = CodeDebugAction(fixed_code="def failed(): pass")

        # Execute action in environment
        obs = env.step(action)

        reward = obs.reward if obs.reward is not None else 0.0
        rewards.append(reward)
        done = obs.done

        # ── MANDATORY LOG: STEP ───────────────────────────────────────────
        print(f"[STEP] step={step_num} action=submit_code reward={reward:.2f} done={str(done).lower()} error=null", flush=True)

        # Update conversation with environment feedback
        messages.append({"role": "assistant", "content": json.dumps(action_dict)})
        messages.append({
            "role": "user",
            "content": (
                f"Result: {obs.test_results}\n"
                f"Reward: {obs.reward}\n"
                f"Done: {obs.done}\n\n"
                + (
                    "Task complete. You passed!" if obs.done
                    else "The test failed. Analyze the feedback and try again. Respond with ONLY the new JSON object."
                )
            ),
        })

        if obs.done:
            final_score = reward
            break

    # ── MANDATORY LOG: END ────────────────────────────────────────────────
    success = final_score >= 1.0
    clamped_score = max(0.01, min(0.99, final_score))
    steps = len(rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={clamped_score:.2f} rewards={','.join(f'{r:.2f}' for r in rewards)}", flush=True)
    return final_score


def main():
    """Run all task levels and report scores."""
    client = OpenAI(
        api_key=HF_TOKEN,
        base_url=API_BASE_URL,
    )

    # Create the environment locally
    env = CodeDebugEnvironment()

    total_score = 0.0
    # We have 2 levels in our environment
    for level in [1, 2]:
        score = run_task(env, client, level)
        total_score += score

    print(f"\nTotal Score: {total_score:.1f} / 2.0", flush=True)


if __name__ == "__main__":
    if not HF_TOKEN:
        print("Error: Please set HF_TOKEN environment variable.")
        sys.exit(1)
    main()
