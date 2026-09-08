"""Offline contract tests for the bounded hybrid invoke handler.

No Bedrock or other AWS endpoint is contacted: every model call is replaced by
an in-process stub before the generated handler is exercised.
"""

from __future__ import annotations

import asyncio
import copy
import json

import agent_base_hybrid as hybrid
from fallback import FWD1_CONFIG
from test_helpers import GAME_STATE


class _Logger:
    def __init__(self):
        self.infos = []
        self.warnings = []

    def info(self, message):
        self.infos.append(message)

    def warning(self, message):
        self.warnings.append(message)


class _App:
    def __init__(self):
        self.logger = _Logger()
        self.handler = None

    def entrypoint(self, function):
        self.handler = function
        return function


def _payload(team="home", player="agentId_3"):
    return {
        "prompt": json.dumps(
            {"gameState": copy.deepcopy(GAME_STATE), "teamId": team, "myPlayers": [player]}
        )
    }


def _command(command_type="MOVE_TO", player=99, team=99):
    return {
        "commandType": command_type,
        "playerId": player,
        "teamId": team,
        "parameters": {"target_x": 12, "target_y": -4, "sprint": False},
        "duration": 0,
    }


def _run(fast_path, fallback, model, payload=None):
    app = _App()
    original = hybrid._ask_sonnet
    hybrid._ask_sonnet = model
    try:
        handler = hybrid.create_hybrid_invoke_handler(
            app=app,
            system_prompt="Return one command as JSON.",
            my_player_id=3,
            position_label="FWD1",
            fallback_fn=fallback,
            fallback_cfg=FWD1_CONFIG,
            fast_path_fn=fast_path,
        )

        async def collect():
            chunks = []
            async for chunk in handler(payload if payload is not None else _payload(), None):
                chunks.append(chunk)
            return json.loads("".join(chunks))

        return asyncio.run(collect()), app.logger
    finally:
        hybrid._ask_sonnet = original


def _model_must_not_run(*_args):
    raise AssertionError("model branch unexpectedly executed")


def _fallback_must_not_run(*_args):
    raise AssertionError("fallback branch unexpectedly executed")


def test_fast_path_skips_model_and_forces_ownership():
    commands, logger = _run(
        lambda *_args: [_command("MOVE_TO")],
        _fallback_must_not_run,
        _model_must_not_run,
    )
    assert commands[0]["playerId"] == 3
    assert commands[0]["teamId"] == 0
    assert any("source=v4" in line for line in logger.infos)


def test_empty_fast_path_uses_model_and_forces_away_ownership():
    raw = json.dumps([_command("MOVE_TO")])
    commands, logger = _run(
        lambda *_args: None,
        _fallback_must_not_run,
        lambda *_args: raw,
        _payload(team="away", player="agentId_4"),
    )
    assert commands[0]["playerId"] == 4
    assert commands[0]["teamId"] == 1
    assert any("source=sonnet-4.6" in line for line in logger.infos)


def test_malformed_model_output_uses_fallback():
    commands, logger = _run(
        lambda *_args: [],
        lambda *_args: [_command("PRESS_BALL")],
        lambda *_args: "this is not JSON",
    )
    assert commands[0]["commandType"] == "PRESS_BALL"
    assert commands[0]["playerId"] == 3 and commands[0]["teamId"] == 0
    assert any("source=fallback" in line for line in logger.infos)


def test_model_exception_uses_fallback_without_aws():
    def fail_model(*_args):
        raise TimeoutError("bounded model timeout")

    commands, logger = _run(
        lambda *_args: None,
        lambda *_args: [_command("SET_STANCE")],
        fail_model,
    )
    assert commands[0]["commandType"] == "SET_STANCE"
    assert logger.warnings and "bounded model timeout" in logger.warnings[0]
    assert any("source=fallback" in line for line in logger.infos)


def test_fast_path_exception_goes_directly_to_fallback():
    def fail_fast(*_args):
        raise RuntimeError("bad local state")

    commands, logger = _run(
        fail_fast,
        lambda *_args: [_command("PRESS_BALL")],
        _model_must_not_run,
    )
    assert commands[0]["commandType"] == "PRESS_BALL"
    assert logger.warnings and "bad local state" in logger.warnings[0]


def test_bad_payload_and_bad_fallback_use_last_resort():
    def fail_fallback(*_args):
        raise RuntimeError("fallback unavailable")

    commands, logger = _run(
        lambda *_args: None,
        fail_fallback,
        _model_must_not_run,
        {"prompt": "not-json"},
    )
    assert commands == [
        {
            "commandType": "PRESS_BALL",
            "playerId": 3,
            "teamId": 0,
            "parameters": {"intensity": 0.6},
            "duration": 3,
        }
    ]
    assert any("source=last-resort" in line for line in logger.infos)


def test_empty_fallback_is_promoted_to_last_resort():
    commands, logger = _run(
        lambda *_args: None,
        lambda *_args: None,
        lambda *_args: "not-json",
    )
    assert commands[0]["commandType"] == "PRESS_BALL"
    assert commands[0]["playerId"] == 3 and commands[0]["teamId"] == 0
    assert any("source=last-resort" in line for line in logger.infos)


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
    print(f"{len(tests)} hybrid handler tests passed")
