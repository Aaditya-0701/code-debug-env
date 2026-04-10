# Copyright (c) 2026
# Licensed under the BSD-style license.

"""
Data models for the Code Debug Env Environment.
"""

from typing import List, Optional
from pydantic import Field
from openenv.core.env_server.types import Action, Observation, State

class CodeDebugAction(Action):
    """Action for the Code Debug Env - submitting fixed Python code."""
    thought: Optional[str] = Field(default=None, description="Reasoning about the bug and how to fix it.")
    fixed_code: str = Field(..., description="The fully corrected Python code.")

class CodeDebugObservation(Observation):
    """Observation returned after taking an action or resetting."""
    # done and reward are inherited
    buggy_code: str = Field(default="", description="The buggy Python code to be fixed.")
    test_results: str = Field(default="", description="Output from running the code against test cases.")
    hint: str = Field(default="", description="Optional hint about the bug.")
    attempts_remaining: int = Field(default=0, description="Number of attempts left before the episode ends.")

class CodeDebugState(State):
    """Full internal state of an episode."""
    # episode_id and step_count are inherited
    correct_solution: str = Field(default="")
    test_cases: str = Field(default="")
    max_attempts: int = Field(default=5)
