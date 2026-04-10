# Copyright (c) 2026
# Licensed under the BSD-style license.

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from .models import CodeDebugAction, CodeDebugObservation, CodeDebugState

class CodeDebugEnv(
    EnvClient[CodeDebugAction, CodeDebugObservation, CodeDebugState]
):
    """Client for the Code Debug Env."""

    def _step_payload(self, action: CodeDebugAction) -> Dict:
        return {
            "fixed_code": action.fixed_code,
        }

    def _parse_result(self, payload: Dict) -> StepResult[CodeDebugObservation]:
        obs_data = payload.get("observation", {})
        observation = CodeDebugObservation(
            buggy_code=obs_data.get("buggy_code", ""),
            test_results=obs_data.get("test_results", ""),
            hint=obs_data.get("hint", ""),
            attempts_remaining=obs_data.get("attempts_remaining", 0),
            done=payload.get("done", False),
            reward=payload.get("reward"),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> CodeDebugState:
        return CodeDebugState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            correct_solution=payload.get("correct_solution", ""),
            test_cases=payload.get("test_cases", ""),
            max_attempts=payload.get("max_attempts", 5),
        )
