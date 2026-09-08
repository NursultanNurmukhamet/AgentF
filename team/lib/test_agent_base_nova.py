"""Offline contract tests for the bounded Nova invoke handler.

No Bedrock or other AWS endpoint is contacted: every model call is replaced by
an in-process stub before the generated handler is exercised.
"""

from __future__ import annotations

import asyncio
import copy
import json

import agent_base_nova as nova
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
    original = nova._ask_nova
    nova._ask_nova = model
    try:
        handler = nova.create_hybrid_invoke_handler(
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
        nova._ask_nova = original


def _model_must_not_run(*_args):
    raise AssertionError("model branch unexpectedly executed")


def _fallback_must_not_run(*_args):
    raise AssertionError("fallback branch unexpectedly executed")


def test_valid_model_always_runs_and_beats_available_fast_path():
    calls = []

    def model(*_args):
        calls.append("model")
        return json.dumps([_command("MOVE_TO")])

    def fast(*_args):
        calls.append("fast")
        return [_command("PRESS_BALL")]

    commands, logger = _run(
        fast,
        _fallback_must_not_run,
        model,
    )
    assert calls == ["model"]
    assert commands[0]["commandType"] == "MOVE_TO"
    assert commands[0]["playerId"] == 3
    assert commands[0]["teamId"] == 0
    assert any("source=nova-micro" in line for line in logger.infos)


def test_model_forces_away_ownership_without_using_fast_path():
    raw = json.dumps([_command("MOVE_TO")])
    commands, logger = _run(
        lambda *_args: None,
        _fallback_must_not_run,
        lambda *_args: raw,
        _payload(team="away", player="agentId_4"),
    )
    assert commands[0]["playerId"] == 4
    assert commands[0]["teamId"] == 1
    assert any("source=nova-micro" in line for line in logger.infos)


def test_no_possession_constraint_precedes_state_passed_to_model():
    seen_states = []

    def model(_system_prompt, constrained_state):
        seen_states.append(constrained_state)
        return json.dumps([_command("MOVE_TO")])

    commands, logger = _run(
        _model_must_not_run,
        _fallback_must_not_run,
        model,
        _payload(player="agentId_4"),
    )
    assert commands[0]["commandType"] == "MOVE_TO"
    assert len(seen_states) == 1
    state = seen_states[0]
    assert state.startswith("TICK OWNERSHIP RULE: YOU DO NOT OWN THE BALL.")
    assert "PASS, SHOOT, and GK_DISTRIBUTE are STRICTLY FORBIDDEN" in state
    assert "MOVE_TO, PRESS_BALL, INTERCEPT, MARK, FOLLOW_PLAYER, SET_STANCE" in state
    assert "exactly ONE" in state
    assert state.index("TICK OWNERSHIP RULE") < state.index("Time:")
    assert any("source=nova-micro" in line for line in logger.infos)


def test_malformed_model_output_uses_fallback():
    commands, logger = _run(
        lambda *_args: [],
        lambda *_args: [_command("PRESS_BALL")],
        lambda *_args: "this is not JSON",
    )
    assert commands[0]["commandType"] == "PRESS_BALL"
    assert commands[0]["playerId"] == 3 and commands[0]["teamId"] == 0
    assert any("source=fallback" in line for line in logger.infos)


def test_model_exception_uses_fast_safety_before_fallback():
    def fail_model(*_args):
        raise TimeoutError("bounded model timeout")

    commands, logger = _run(
        lambda *_args: [_command("INTERCEPT")],
        _fallback_must_not_run,
        fail_model,
    )
    assert commands[0]["commandType"] == "INTERCEPT"
    assert logger.warnings and "bounded model timeout" in logger.warnings[0]
    assert any("source=v6-safety" in line for line in logger.infos)


def test_model_error_then_fast_error_uses_fallback():
    def fail_model(*_args):
        raise TimeoutError("bounded model timeout")

    def fail_fast(*_args):
        raise RuntimeError("bad local state")

    commands, logger = _run(
        fail_fast,
        lambda *_args: [_command("PRESS_BALL")],
        fail_model,
    )
    assert commands[0]["commandType"] == "PRESS_BALL"
    assert any("bounded model timeout" in line for line in logger.warnings)
    assert any("bad local state" in line for line in logger.warnings)
    assert any("source=fallback" in line for line in logger.infos)


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


def test_model_pass_without_possession_uses_fallback():
    payload = _payload()
    decoded = json.loads(payload["prompt"])
    decoded["gameState"]["ball"]["isFree"] = True
    decoded["gameState"]["ball"]["possessionAgentId"] = None
    payload["prompt"] = json.dumps(decoded)
    raw = json.dumps([_command("PASS")])
    commands, logger = _run(
        lambda *_args: None,
        lambda *_args: [_command("PRESS_BALL")],
        lambda *_args: raw,
        payload,
    )
    assert commands[0]["commandType"] == "PRESS_BALL"
    assert any("source=fallback" in line for line in logger.infos)


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
    print(f"{len(tests)} hybrid handler tests passed")
