"""V6 Direct Attack tactical regressions; all tests are offline."""

from __future__ import annotations

import copy
import json

import agent_base_nova as nova
from fast_path import build_fast_path
from test_agent_base_nova import _command, _run
from test_helpers import GAME_STATE


def _player(state: dict, team: str, player_id: int) -> dict:
    return next(
        player for player in state["players"]
        if player["teamCode"] == team
        and player["agentId"] == f"agentId_{player_id}"
    )


def _give_ball(
    state: dict, team: str, player_id: int, x: float, y: float = 0,
) -> dict:
    holder = _player(state, team, player_id)
    holder["position"] = {"x": x, "y": y}
    state["ball"].update({
        "position": {"x": x, "y": y},
        "velocity": {"x": 0, "y": 0, "z": 0},
        "isFree": False,
        "possessionAgentId": f"agentId_{player_id}",
    })
    return state


def _move_opponents_far(state: dict, team: str) -> None:
    far_x = 48 if team == "home" else -48
    for index, opponent in enumerate(
        player for player in state["players"] if player["teamCode"] != team
    ):
        opponent["position"] = {"x": far_x, "y": -30 + index * 12}


def _crowded_home_forward_state() -> dict:
    state = _give_ball(copy.deepcopy(GAME_STATE), "home", 3, 30, 0)
    _player(state, "home", 4)["position"] = {"x": 42, "y": 12}
    _player(state, "away", 1)["position"] = {"x": 31, "y": 1}
    _player(state, "away", 2)["position"] = {"x": 29, "y": -2}
    return state


def test_home_defender_clears_own_half_to_advanced_forward():
    state = _give_ball(copy.deepcopy(GAME_STATE), "home", 1, -30, 0)
    _player(state, "home", 3)["position"] = {"x": 22, "y": -7}
    _player(state, "home", 4)["position"] = {"x": 12, "y": 8}

    command = build_fast_path("DEF")(state, 0, 1)[0]

    assert command["commandType"] == "PASS"
    assert command["parameters"] == {"target_player_id": 3, "type": "AERIAL"}
    assert command["duration"] == 0


def test_away_defender_clears_mirrored_own_half_to_advanced_forward():
    state = _give_ball(copy.deepcopy(GAME_STATE), "away", 1, 30, 0)
    _player(state, "away", 3)["position"] = {"x": -22, "y": -7}
    _player(state, "away", 4)["position"] = {"x": -12, "y": 8}

    command = build_fast_path("DEF")(state, 1, 1)[0]

    assert command["commandType"] == "PASS"
    assert command["parameters"] == {"target_player_id": 3, "type": "AERIAL"}
    assert command["duration"] == 0


def test_own_half_clearance_shoots_full_power_when_nobody_crossed_midfield():
    for team, team_id, ball_x, direction in (
        ("home", 0, -30, 1),
        ("away", 1, 30, -1),
    ):
        state = _give_ball(copy.deepcopy(GAME_STATE), team, 1, ball_x, 0)
        for player_id, attack_x in ((2, -22), (3, -12), (4, -4)):
            _player(state, team, player_id)["position"] = {
                "x": direction * attack_x,
                "y": player_id * 3,
            }

        command = build_fast_path("DEF")(state, team_id, 1)[0]

        assert command["commandType"] == "SHOOT"
        assert command["parameters"] == {
            "aim_location": "CENTER",
            "power": 1.0,
        }
        directive = nova._tactical_directive(state, team_id, 1, "DEF")
        assert directive["kind"] == "OWN_HALF_CLEAR"
        assert directive["target_id"] is None


def test_goalkeeper_attempts_full_power_direct_goal_for_both_sides():
    for team, team_id, x in (("home", 0, -50), ("away", 1, 50)):
        state = _give_ball(copy.deepcopy(GAME_STATE), team, 0, x, 0)
        command = build_fast_path("GK")(state, team_id, 0)[0]
        assert command["commandType"] == "SHOOT"
        assert command["parameters"] == {
            "aim_location": "CENTER",
            "power": 1.0,
        }
        assert command["duration"] == 0


def test_crowded_forward_passes_before_considering_shot():
    state = _crowded_home_forward_state()

    command = build_fast_path("FWD1")(state, 0, 3)[0]

    assert command["commandType"] == "PASS"
    assert command["parameters"]["target_player_id"] == 4
    assert command["parameters"]["type"] in {"GROUND", "THROUGH", "AERIAL"}


def test_same_forward_shoots_when_pressure_is_removed():
    state = _crowded_home_forward_state()
    _move_opponents_far(state, "home")

    command = build_fast_path("FWD1")(state, 0, 3)[0]

    assert command["commandType"] == "SHOOT"
    assert command["parameters"]["power"] == 1.0


def test_pass_then_receiver_shoots_on_next_snapshot_without_memory():
    first = _give_ball(copy.deepcopy(GAME_STATE), "home", 2, 12, 0)
    _move_opponents_far(first, "home")
    _player(first, "home", 3)["position"] = {"x": 30, "y": 0}
    _player(first, "home", 4)["position"] = {"x": 10, "y": 18}

    pass_command = build_fast_path("MID")(first, 0, 2)[0]
    assert pass_command["commandType"] == "PASS"
    assert pass_command["parameters"]["target_player_id"] == 3

    second = _give_ball(copy.deepcopy(first), "home", 3, 30, 0)
    shot_command = build_fast_path("FWD1")(second, 0, 3)[0]
    assert shot_command["commandType"] == "SHOOT"
    assert shot_command["parameters"]["power"] == 1.0


def test_shot_radii_are_larger_but_still_bounded_and_mirrored():
    home_forward = _give_ball(copy.deepcopy(GAME_STATE), "home", 3, 23, 0)
    _move_opponents_far(home_forward, "home")
    assert build_fast_path("FWD1")(home_forward, 0, 3)[0]["commandType"] == "SHOOT"

    outside = _give_ball(copy.deepcopy(GAME_STATE), "home", 3, 22.5, 0)
    _move_opponents_far(outside, "home")
    assert build_fast_path("FWD1")(outside, 0, 3)[0]["commandType"] != "SHOOT"

    home_mid = _give_ball(copy.deepcopy(GAME_STATE), "home", 2, 27, 0)
    _move_opponents_far(home_mid, "home")
    assert build_fast_path("MID")(home_mid, 0, 2)[0]["commandType"] == "SHOOT"

    away_forward = _give_ball(copy.deepcopy(GAME_STATE), "away", 4, -23, 0)
    _move_opponents_far(away_forward, "away")
    assert build_fast_path("FWD2")(away_forward, 1, 4)[0]["commandType"] == "SHOOT"


def test_attacking_corner_still_recycles_instead_of_spamming_shots():
    state = _give_ball(copy.deepcopy(GAME_STATE), "home", 4, 45, 25)

    command = build_fast_path("FWD2")(state, 0, 4)[0]

    assert command["commandType"] != "SHOOT"
    assert command["commandType"] in {"PASS", "MOVE_TO"}


def test_directives_reject_noncompliant_model_commands():
    own_half = _give_ball(copy.deepcopy(GAME_STATE), "home", 1, -25, 0)
    own_half_directive = nova._tactical_directive(own_half, 0, 1, "DEF")
    assert own_half_directive["kind"] == "OWN_HALF_CLEAR"
    assert not nova._directive_command_is_legal(
        [_command("MOVE_TO")], own_half_directive,
    )

    crowded = _crowded_home_forward_state()
    crowded_directive = nova._tactical_directive(crowded, 0, 3, "FWD1")
    assert crowded_directive["kind"] == "CROWDED_PASS"
    assert not nova._directive_command_is_legal(
        [{
            "commandType": "SHOOT",
            "parameters": {"aim_location": "TR", "power": 1.0},
        }],
        crowded_directive,
    )

    goalkeeper = _give_ball(copy.deepcopy(GAME_STATE), "home", 0, -50, 0)
    goalkeeper_directive = nova._tactical_directive(goalkeeper, 0, 0, "GK")
    assert goalkeeper_directive["kind"] == "GK_DIRECT_SHOT"
    assert nova._directive_command_is_legal(
        [{
            "commandType": "SHOOT",
            "parameters": {"aim_location": "CENTER", "power": 1.0},
        }],
        goalkeeper_directive,
    )


def test_model_ignoring_crowded_pass_falls_back_to_v6_safety():
    state = _crowded_home_forward_state()
    payload = {
        "prompt": json.dumps({
            "gameState": state,
            "teamId": "home",
            "myPlayers": ["agentId_3"],
        })
    }
    model_shot = json.dumps([{
        "commandType": "SHOOT",
        "playerId": 3,
        "parameters": {"aim_location": "TR", "power": 1.0},
        "duration": 0,
    }])

    commands, logger = _run(
        build_fast_path("FWD1", full=False),
        lambda *_args: [_command("PRESS_BALL")],
        lambda *_args: model_shot,
        payload,
    )

    assert commands[0]["commandType"] == "PASS"
    assert commands[0]["parameters"]["target_player_id"] == 4
    assert any("source=v6-safety" in line for line in logger.infos)
    assert any("directive=CROWDED_PASS" in line for line in logger.warnings)


def test_stale_possession_far_from_ball_cannot_trigger_tactics():
    state = _give_ball(copy.deepcopy(GAME_STATE), "home", 3, 30, 0)
    state["ball"]["position"] = {"x": 35, "y": 0}

    assert nova._tactical_directive(state, 0, 3, "FWD1") is None


def test_illegal_possession_command_from_generic_fallback_is_rejected():
    payload = {
        "prompt": json.dumps({
            "gameState": copy.deepcopy(GAME_STATE),
            "teamId": "home",
            "myPlayers": ["agentId_4"],
        })
    }
    fallback_pass = [{
        "commandType": "PASS",
        "playerId": 4,
        "parameters": {"target_player_id": 3, "type": "GROUND"},
        "duration": 0,
    }]

    commands, logger = _run(
        lambda *_args: None,
        lambda *_args: fallback_pass,
        lambda *_args: "not-json",
        payload,
    )

    assert commands[0]["commandType"] == "PRESS_BALL"
    assert any("source=last-resort" in line for line in logger.infos)


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
    print(f"{len(tests)} V6 direct-attack tests passed")
