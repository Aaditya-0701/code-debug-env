# Copyright (c) 2026
# Licensed under the BSD-style license.

"""
Code Debug Env Environment Implementation.

An environment where agents are given buggy Python code and must return working code.
Correctness is checked programmatically.
"""

import sys
import os
import io
import traceback
import textwrap
import random
from uuid import uuid4

# Ensure project root is on sys.path so `from models import ...` works everywhere
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openenv.core.env_server import Environment
from models import CodeDebugAction, CodeDebugObservation, CodeDebugState

PROBLEMS = [
    {
        "buggy_code": textwrap.dedent('''\
            def find_max(numbers):
                max_val = 0
                for num in numbers:
                    if num > max_val:
                        max_val = num
                return max_val
        '''),
        "asserts": textwrap.dedent('''\
            assert find_max([1, 2, 3]) == 3
            assert find_max([-1, -5, -3]) == -1
            assert find_max([0]) == 0
        '''),
        "hint": "What happens if all numbers in the list are negative?",
        "correct_code": textwrap.dedent('''\
            def find_max(numbers):
                max_val = float('-inf')
                for num in numbers:
                    if num > max_val:
                        max_val = num
                return max_val
        ''')
    },
    {
        "buggy_code": textwrap.dedent('''\
            def is_palindrome(word):
                return word == word[::-1]
        '''),
        "asserts": textwrap.dedent('''\
            assert is_palindrome("racecar") == True
            assert is_palindrome("Racecar") == True
            assert is_palindrome("hello") == False
        '''),
        "hint": "Consider case sensitivity. Is 'Racecar' a palindrome?",
        "correct_code": textwrap.dedent('''\
            def is_palindrome(word):
                word = word.lower()
                return word == word[::-1]
        ''')
    }
]

class CodeDebugEnvironment(Environment):
    """
    Code Debug Environment.
    Agents receive buggy Python code and an objective.
    They must return fixed code. The fixed code is tested automatically.
    """
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        super().__init__()
        self._state = CodeDebugState(episode_id=str(uuid4()), step_count=0)
        self._current_problem = None
        self._attempts = 5

    def reset(self, seed=None, episode_id=None, **kwargs) -> CodeDebugObservation:
        task_level = kwargs.get('task_level', 1)
        # Use task_level to select the problem uniquely (1-based index)
        # If task_level > number of problems, just bound it
        idx = max(0, min(len(PROBLEMS) - 1, task_level - 1))
        self._current_problem = PROBLEMS[idx]
        
        self._attempts = 5
        self._state = CodeDebugState(
            episode_id=episode_id or str(uuid4()), 
            step_count=0,
            correct_solution=self._current_problem["correct_code"],
            test_cases=self._current_problem["asserts"],
            max_attempts=self._attempts
        )

        return CodeDebugObservation(
            buggy_code=self._current_problem["buggy_code"],
            test_results="Run code to test.",
            hint="You have 5 attempts. Fix the bug.",
            attempts_remaining=self._attempts,
            done=False,
            reward=0.0
        )

    def step(self, action: CodeDebugAction, timeout_s=None, **kwargs) -> CodeDebugObservation:
        # Guard: auto-reset if reset() was never called (happens on stateless HTTP /step)
        if self._current_problem is None:
            self.reset()
        
        self._state.step_count += 1
        self._attempts -= 1
        
        fixed_code = action.fixed_code
        test_results = ""
        passed = False
        
        # Combine fixed code with assertions — use a real newline
        code_to_run = fixed_code + "\n" + self._current_problem["asserts"]
        
        # Capture stdout
        old_stdout = sys.stdout
        redirected_output = io.StringIO()
        sys.stdout = redirected_output
        
        local_scope = {}
        try:
            exec(code_to_run, {}, local_scope)
            passed = True
            test_results = "All tests passed! " + redirected_output.getvalue()
        except AssertionError as e:
            test_results = f"Test failed (AssertionError).\nOutput: {redirected_output.getvalue()}"
        except Exception as e:
            test_results = f"Error during execution: {traceback.format_exc()}"
        finally:
            sys.stdout = old_stdout

        done = passed or (self._attempts <= 0)
        reward = 1.0 if passed else 0.0
        
        if passed:
            hint = "Excellent! You fixed the code."
        elif self._attempts <= 0:
            hint = f"Out of attempts. Correct solution:\n{self._current_problem['correct_code']}"
        else:
            hint = self._current_problem['hint']

        return CodeDebugObservation(
            buggy_code=self._current_problem["buggy_code"],
            test_results=test_results,
            hint=hint,
            attempts_remaining=self._attempts,
            done=done,
            reward=reward
        )

    @property
    def state(self) -> CodeDebugState:
        return self._state
