"""Offline regression tests for possession identity and the fast policy."""

import copy

from fast_path import build_fast_path
from fallback import FWD1_CONFIG, build_fallback
from state import get_possession_info, stamina_percent
from test_helpers import GAME_STATE


def _state_with_away_possession(player_id: int = 3):
    state = copy.deepcopy(GAME_STATE)
    away = next(p for p in state["players"] if p["teamCode"] == "away" and p["agentId"] == f"agentId_{player_id}")
    state["ball"]["possessionAgentId"] = f"agentId_{player_id}"
    state["ball"]["position"] = dict(away["position"])
    state["ball"]["isFree"] = False
    return state


def _one_touch_state(carrier_stamina: float = 80):
    state = copy.deepcopy(GAME_STATE)
    carrier = next(
        p for p in state["players"]
        if p["teamCode"] == "home" and p["agentId"] == "agentId_3"
    )
    static_outlet = next(
        p for p in state["players"]
        if p["teamCode"] == "home" and p["agentId"] == "agentId_2"
    )
    moving_outlet = next(
        p for p in state["players"]
        if p["teamCode"] == "home" and p["agentId"] == "agentId_4"
    )
    carrier.update({
        "position": {"x": 0, "y": 0},
        "velocity": {"x": 0, "y": 0},
        "stamina": carrier_stamina,
    })
    static_outlet.update({
        "position": {"x": 12, "y": -8},
        "velocity": {"x": 0, "y": 0},
        "stamina": 80,
    })
    moving_outlet.update({
        "position": {"x": 12, "y": 8},
        "velocity": {"x": 2, "y": 0},
        "stamina": 80,
    })
    state["players"] = [carrier, static_outlet, moving_outlet]
    state["ball"].update({
        "position": dict(carrier["position"]),
        "isFree": False,
        "possessionAgentId": "agentId_3",
    })
    return state


def test_possession_identity_uses_team_and_distance():
    state = _state_with_away_possession(3)
    pid, _, home_has_ball = get_possession_info(state["ball"], state["players"], 0)
    assert pid == 3
    assert home_has_ball is False
    _, _, away_has_ball = get_possession_info(state["ball"], state["players"], 1)
    assert away_has_ball is True


def test_opponent_with_same_player_id_is_not_our_possession():
    state = _state_with_away_possession(3)
    cmd = build_fast_path("FWD1")(state, 0, 3)[0]
    assert cmd["commandType"] != "SHOOT"


def test_fallback_does_not_claim_opponents_possession():
    state = _state_with_away_possession(3)
    cmd = build_fallback(FWD1_CONFIG)(state, 0, 3)[0]
    assert cmd["commandType"] != "SHOOT"


def test_only_nearest_fresh_outfielder_presses():
    state = _state_with_away_possession(3)
    away_holder = next(
        p for p in state["players"]
        if p["teamCode"] == "away" and p["agentId"] == "agentId_3"
    )
    home_forward = next(
        p for p in state["players"]
        if p["teamCode"] == "home" and p["agentId"] == "agentId_3"
    )
    midfielder = next(
        p for p in state["players"]
        if p["teamCode"] == "home" and p["agentId"] == "agentId_2"
    )
    away_holder["position"] = {"x": 10, "y": -12}
    state["ball"]["position"] = dict(away_holder["position"])
    home_forward["position"] = {"x": 9, "y": -12}
    midfielder["position"] = {"x": 6, "y": -12}
    commands = {
        pid: build_fast_path(label)(state, 0, pid)[0]
        for pid, label in [(1, "DEF"), (2, "MID"), (3, "FWD1"), (4, "FWD2")]
    }
    pressers = [pid for pid, cmd in commands.items() if cmd["commandType"] == "PRESS_BALL"]
    assert pressers == [3]


def test_fresh_presser_is_preferred_over_closer_low_stamina_player():
    state = _state_with_away_possession(3)
    holder = next(
        p for p in state["players"]
        if p["teamCode"] == "away" and p["agentId"] == "agentId_3"
    )
    midfielder = next(
        p for p in state["players"]
        if p["teamCode"] == "home" and p["agentId"] == "agentId_2"
    )
    forward = next(
        p for p in state["players"]
        if p["teamCode"] == "home" and p["agentId"] == "agentId_3"
    )
    holder["position"] = {"x": 0, "y": 0}
    state["ball"]["position"] = dict(holder["position"])
    for player in state["players"]:
        if player["teamCode"] == "home":
            player["position"] = {"x": -40, "y": 30}
    midfielder.update({"position": {"x": -8, "y": 0}, "stamina": 80})
    forward.update({"position": {"x": -1, "y": 0}, "stamina": 20})

    commands = {
        pid: build_fast_path(label)(state, 0, pid)[0]
        for pid, label in [(1, "DEF"), (2, "MID"), (3, "FWD1"), (4, "FWD2")]
    }
    pressers = [
        pid for pid, command in commands.items()
        if command["commandType"] == "PRESS_BALL"
    ]

    assert pressers == [2]


def test_free_ball_overrides_stale_possession_id():
    state = copy.deepcopy(GAME_STATE)
    forward = next(
        p for p in state["players"]
        if p["teamCode"] == "home" and p["agentId"] == "agentId_3"
    )
    forward["position"] = {"x": 40, "y": 0}
    state["ball"].update({
        "position": dict(forward["position"]),
        "isFree": True,
        "possessionAgentId": "agentId_3",
    })

    cmd = build_fast_path("FWD1")(state, 0, 3)[0]

    assert cmd["commandType"] == "INTERCEPT"


def test_closest_outfielder_intercepts_free_ball_instead_of_pressing():
    state = copy.deepcopy(GAME_STATE)
    midfielder = next(
        p for p in state["players"]
        if p["teamCode"] == "home" and p["agentId"] == "agentId_2"
    )
    state["ball"].update({
        "position": dict(midfielder["position"]),
        "isFree": True,
        "possessionAgentId": None,
    })

    cmd = build_fast_path("MID")(state, 0, 2)[0]

    assert cmd["commandType"] == "INTERCEPT"
    assert cmd["parameters"] == {"aggressive": False}


def test_shooting_range_uses_two_dimensional_distance():
    state = copy.deepcopy(GAME_STATE)
    forward = next(
        p for p in state["players"]
        if p["teamCode"] == "home" and p["agentId"] == "agentId_3"
    )
    forward["position"] = {"x": 40, "y": 35}
    state["ball"].update({
        "position": dict(forward["position"]),
        "isFree": False,
        "possessionAgentId": "agentId_3",
    })

    cmd = build_fast_path("FWD1")(state, 0, 3)[0]

    assert cmd["commandType"] != "SHOOT"


def test_forward_recycles_ball_out_of_attacking_corner():
    state = copy.deepcopy(GAME_STATE)
    forward = next(
        p for p in state["players"]
        if p["teamCode"] == "home" and p["agentId"] == "agentId_4"
    )
    forward["position"] = {"x": 50, "y": 34}
    state["ball"].update({
        "position": dict(forward["position"]),
        "isFree": False,
        "possessionAgentId": "agentId_4",
    })

    cmd = build_fast_path("FWD2")(state, 0, 4)[0]

    assert cmd["commandType"] == "PASS"
    assert cmd["parameters"]["target_player_id"] in {1, 2, 3}
    assert cmd["parameters"]["type"] == "GROUND"


def test_corner_escape_moves_infield_when_no_outlet_exists():
    state = copy.deepcopy(GAME_STATE)
    forward = next(
        p for p in state["players"]
        if p["teamCode"] == "home" and p["agentId"] == "agentId_4"
    )
    forward["position"] = {"x": 50, "y": 34}
    state["players"] = [
        p for p in state["players"]
        if p["teamCode"] != "home" or p["agentId"] == "agentId_4"
    ]
    state["ball"].update({
        "position": dict(forward["position"]),
        "isFree": False,
        "possessionAgentId": "agentId_4",
    })

    cmd = build_fast_path("FWD2")(state, 0, 4)[0]

    assert cmd["commandType"] == "MOVE_TO"
    assert cmd["parameters"]["target_x"] == 38.0
    assert cmd["parameters"]["target_y"] == 12.0


def test_away_corner_escape_is_mirrored():
    state = copy.deepcopy(GAME_STATE)
    forward = next(
        p for p in state["players"]
        if p["teamCode"] == "away" and p["agentId"] == "agentId_4"
    )
    forward["position"] = {"x": -50, "y": -34}
    state["players"] = [
        p for p in state["players"]
        if p["teamCode"] != "away" or p["agentId"] == "agentId_4"
    ]
    state["ball"].update({
        "position": dict(forward["position"]),
        "isFree": False,
        "possessionAgentId": "agentId_4",
    })

    cmd = build_fast_path("FWD2")(state, 1, 4)[0]

    assert cmd["commandType"] == "MOVE_TO"
    assert cmd["parameters"]["target_x"] == -38.0
    assert cmd["parameters"]["target_y"] == -12.0


def test_forward_passes_to_clear_upfield_outlet():
    state = copy.deepcopy(GAME_STATE)
    carrier = next(
        p for p in state["players"]
        if p["teamCode"] == "home" and p["agentId"] == "agentId_3"
    )
    outlet = next(
        p for p in state["players"]
        if p["teamCode"] == "home" and p["agentId"] == "agentId_4"
    )
    carrier["position"] = {"x": 0, "y": -8}
    outlet["position"] = {"x": 12, "y": 8}
    state["ball"].update({
        "position": dict(carrier["position"]),
        "isFree": False,
        "possessionAgentId": "agentId_3",
    })

    cmd = build_fast_path("FWD1")(state, 0, 3)[0]

    assert cmd["commandType"] == "PASS"
    assert cmd["parameters"] == {"target_player_id": 4, "type": "THROUGH"}


def test_build_up_does_not_pass_to_teammate_stuck_in_corner():
    state = copy.deepcopy(GAME_STATE)
    carrier = next(
        p for p in state["players"]
        if p["teamCode"] == "home" and p["agentId"] == "agentId_3"
    )
    corner_runner = next(
        p for p in state["players"]
        if p["teamCode"] == "home" and p["agentId"] == "agentId_4"
    )
    carrier["position"] = {"x": 20, "y": -8}
    corner_runner["position"] = {"x": 45, "y": 30}
    state["players"] = [
        p for p in state["players"]
        if p["teamCode"] != "home" or p is carrier or p is corner_runner
    ]
    state["ball"].update({
        "position": dict(carrier["position"]),
        "isFree": False,
        "possessionAgentId": "agentId_3",
    })

    cmd = build_fast_path("FWD1")(state, 0, 3)[0]

    assert cmd["commandType"] == "MOVE_TO"


def test_one_touch_prefers_forward_moving_outlet():
    state = _one_touch_state(80)

    cmd = build_fast_path("FWD1")(state, 0, 3)[0]

    assert cmd["commandType"] == "PASS"
    assert cmd["parameters"] == {"target_player_id": 4, "type": "THROUGH"}


def test_vertical_outlet_beats_short_lateral_option():
    state = copy.deepcopy(GAME_STATE)
    carrier = next(
        p for p in state["players"]
        if p["teamCode"] == "home" and p["agentId"] == "agentId_2"
    )
    lateral = next(
        p for p in state["players"]
        if p["teamCode"] == "home" and p["agentId"] == "agentId_1"
    )
    vertical = next(
        p for p in state["players"]
        if p["teamCode"] == "home" and p["agentId"] == "agentId_3"
    )
    carrier["position"] = {"x": 0, "y": 0}
    lateral.update({"position": {"x": 2, "y": 10}, "velocity": {"x": 0, "y": 0}})
    vertical.update({"position": {"x": 16, "y": 0}, "velocity": {"x": 2, "y": 0}})
    state["players"] = [carrier, lateral, vertical]
    state["ball"].update({
        "position": dict(carrier["position"]),
        "isFree": False,
        "possessionAgentId": "agentId_2",
    })

    cmd = build_fast_path("MID")(state, 0, 2)[0]

    assert cmd["commandType"] == "PASS"
    assert cmd["parameters"] == {"target_player_id": 3, "type": "THROUGH"}


def test_full_attack_shoots_from_twenty_four_units_centrally():
    state = copy.deepcopy(GAME_STATE)
    forward = next(
        p for p in state["players"]
        if p["teamCode"] == "home" and p["agentId"] == "agentId_3"
    )
    forward["position"] = {"x": 31, "y": 0}
    state["ball"].update({
        "position": dict(forward["position"]),
        "isFree": False,
        "possessionAgentId": "agentId_3",
    })

    cmd = build_fast_path("FWD1")(state, 0, 3)[0]

    assert cmd["commandType"] == "SHOOT"


def test_low_stamina_one_touch_uses_safe_ground_pass():
    state = _one_touch_state(20)

    cmd = build_fast_path("FWD1")(state, 0, 3)[0]

    assert cmd["commandType"] == "PASS"
    assert cmd["parameters"] == {"target_player_id": 4, "type": "GROUND"}


def test_three_stamina_tiers_control_attacking_sprint():
    state = copy.deepcopy(GAME_STATE)
    carrier = next(
        p for p in state["players"]
        if p["teamCode"] == "home" and p["agentId"] == "agentId_3"
    )
    carrier["position"] = {"x": 0, "y": -8}
    state["players"] = [carrier]
    state["ball"].update({
        "position": dict(carrier["position"]),
        "isFree": False,
        "possessionAgentId": "agentId_3",
    })

    sprint_by_stamina = {}
    for stamina in (80, 50, 20):
        carrier["stamina"] = stamina
        cmd = build_fast_path("FWD1")(state, 0, 3)[0]
        assert cmd["commandType"] == "MOVE_TO"
        sprint_by_stamina[stamina] = cmd["parameters"]["sprint"]

    assert sprint_by_stamina == {80: True, 50: False, 20: False}


def test_three_stamina_tiers_shrink_press_range_and_keep_emergency_press():
    state = _state_with_away_possession(3)
    holder = next(
        p for p in state["players"]
        if p["teamCode"] == "away" and p["agentId"] == "agentId_3"
    )
    midfielder = next(
        p for p in state["players"]
        if p["teamCode"] == "home" and p["agentId"] == "agentId_2"
    )
    for player in state["players"]:
        if player["teamCode"] == "home" and player is not midfielder:
            player["position"] = {"x": -45, "y": 30}

    holder["position"] = {"x": 0, "y": 0}
    state["ball"]["position"] = dict(holder["position"])
    midfielder.update({"position": {"x": -7, "y": 0}, "stamina": 80})
    fresh = build_fast_path("MID")(state, 0, 2)[0]

    midfielder["stamina"] = 50
    managed_far = build_fast_path("MID")(state, 0, 2)[0]
    midfielder["position"] = {"x": -5, "y": 0}
    managed_close = build_fast_path("MID")(state, 0, 2)[0]

    midfielder.update({"position": {"x": -3, "y": 0}, "stamina": 20})
    low_normal = build_fast_path("MID")(state, 0, 2)[0]

    holder["position"] = {"x": -30, "y": 0}
    state["ball"]["position"] = dict(holder["position"])
    midfielder["position"] = {"x": -27, "y": 0}
    low_emergency = build_fast_path("MID")(state, 0, 2)[0]

    assert fresh["commandType"] == "PRESS_BALL"
    assert managed_far["commandType"] != "PRESS_BALL"
    assert managed_close["commandType"] == "PRESS_BALL"
    assert low_normal["commandType"] != "PRESS_BALL"
    assert low_emergency["commandType"] == "PRESS_BALL"
    assert low_emergency["parameters"]["intensity"] == 0.35


def test_low_stamina_only_intercepts_loose_ball_in_immediate_danger():
    state = copy.deepcopy(GAME_STATE)
    midfielder = next(
        p for p in state["players"]
        if p["teamCode"] == "home" and p["agentId"] == "agentId_2"
    )
    for player in state["players"]:
        if player["teamCode"] == "home" and player is not midfielder:
            player["position"] = {"x": -45, "y": 30}
    midfielder.update({"position": {"x": 0, "y": 0}, "stamina": 20})
    state["ball"].update({
        "position": {"x": 0, "y": 0},
        "isFree": True,
        "possessionAgentId": None,
    })
    normal = build_fast_path("MID")(state, 0, 2)[0]

    midfielder["position"] = {"x": -27, "y": 0}
    state["ball"]["position"] = {"x": -30, "y": 0}
    emergency = build_fast_path("MID")(state, 0, 2)[0]

    assert normal["commandType"] != "INTERCEPT"
    assert emergency["commandType"] == "INTERCEPT"


def test_free_ball_prefers_fresh_collector_over_closer_managed_player():
    state = copy.deepcopy(GAME_STATE)
    midfielder = next(
        p for p in state["players"]
        if p["teamCode"] == "home" and p["agentId"] == "agentId_2"
    )
    forward = next(
        p for p in state["players"]
        if p["teamCode"] == "home" and p["agentId"] == "agentId_3"
    )
    for player in state["players"]:
        if player["teamCode"] == "home":
            player["position"] = {"x": -40, "y": 30}
    midfielder.update({"position": {"x": -10, "y": 0}, "stamina": 80})
    forward.update({"position": {"x": -2, "y": 0}, "stamina": 50})
    state["ball"].update({
        "position": {"x": 0, "y": 0},
        "isFree": True,
        "possessionAgentId": None,
    })

    mid = build_fast_path("MID")(state, 0, 2)[0]
    fwd = build_fast_path("FWD1")(state, 0, 3)[0]

    assert mid["commandType"] == "INTERCEPT"
    assert fwd["commandType"] != "INTERCEPT"


def test_low_stamina_sprints_only_for_emergency_recovery():
    state = _state_with_away_possession(3)
    holder = next(
        p for p in state["players"]
        if p["teamCode"] == "away" and p["agentId"] == "agentId_3"
    )
    midfielder = next(
        p for p in state["players"]
        if p["teamCode"] == "home" and p["agentId"] == "agentId_2"
    )
    midfielder.update({"position": {"x": -10, "y": 0}, "stamina": 20})
    holder["position"] = {"x": 0, "y": 0}
    state["ball"]["position"] = dict(holder["position"])
    normal = build_fast_path("MID")(state, 0, 2)[0]

    holder["position"] = {"x": -30, "y": 0}
    state["ball"]["position"] = dict(holder["position"])
    midfielder["position"] = {"x": -20, "y": 0}
    emergency = build_fast_path("MID")(state, 0, 2)[0]

    assert normal["commandType"] == emergency["commandType"] == "MOVE_TO"
    assert normal["parameters"]["sprint"] is False
    assert emergency["parameters"]["sprint"] is True


def test_full_attack_press_radius_stops_at_twelve():
    state = _state_with_away_possession(1)
    midfielder = next(
        p for p in state["players"]
        if p["teamCode"] == "home" and p["agentId"] == "agentId_2"
    )
    for player in state["players"]:
        if player["teamCode"] == "home" and player is not midfielder:
            player["position"] = {"x": -45, "y": 30}

    ball_pos = state["ball"]["position"]
    midfielder["position"] = {"x": ball_pos["x"] - 13, "y": ball_pos["y"]}
    outside = build_fast_path("MID")(state, 0, 2)[0]

    midfielder["position"] = {"x": ball_pos["x"] - 11, "y": ball_pos["y"]}
    inside = build_fast_path("MID")(state, 0, 2)[0]

    assert outside["commandType"] != "PRESS_BALL"
    assert inside["commandType"] == "PRESS_BALL"


def test_direct_attack_gets_mark_press_and_two_counter_outlets():
    state = _state_with_away_possession(3)
    holder = next(
        p for p in state["players"]
        if p["teamCode"] == "away" and p["agentId"] == "agentId_3"
    )
    holder["position"] = {"x": -20, "y": 0}
    state["ball"]["position"] = dict(holder["position"])
    midfielder = next(
        p for p in state["players"]
        if p["teamCode"] == "home" and p["agentId"] == "agentId_2"
    )
    midfielder["position"] = {"x": -15, "y": 0}

    commands = {
        pid: build_fast_path(label)(state, 0, pid)[0]
        for pid, label in [(1, "DEF"), (2, "MID"), (3, "FWD1"), (4, "FWD2")]
    }

    assert commands[1]["commandType"] == "MARK"
    assert commands[1]["parameters"]["target_player_id"] == 3
    assert commands[2]["commandType"] == "PRESS_BALL"
    assert commands[3]["commandType"] == "MOVE_TO"
    assert commands[4]["commandType"] == "MOVE_TO"
    assert commands[3]["parameters"]["target_y"] < 0
    assert commands[4]["parameters"]["target_y"] > 0


def test_distant_free_ball_keeps_shape_instead_of_intercepting():
    state = copy.deepcopy(GAME_STATE)
    state["ball"].update({
        "position": {"x": 0, "y": 30},
        "isFree": True,
        "possessionAgentId": None,
    })

    commands = [
        build_fast_path(label)(state, 0, pid)[0]
        for pid, label in [(1, "DEF"), (2, "MID"), (3, "FWD1"), (4, "FWD2")]
    ]

    assert all(command["commandType"] != "INTERCEPT" for command in commands)


def test_defender_clears_own_half_with_aerial_pass():
    state = copy.deepcopy(GAME_STATE)
    defender = next(
        p for p in state["players"]
        if p["teamCode"] == "home" and p["agentId"] == "agentId_1"
    )
    defender["position"] = {"x": -25, "y": 0}
    state["ball"].update({
        "position": dict(defender["position"]),
        "isFree": False,
        "possessionAgentId": "agentId_1",
    })

    cmd = build_fast_path("DEF")(state, 0, 1)[0]

    assert cmd["commandType"] == "PASS"
    assert cmd["parameters"]["type"] == "AERIAL"
    assert cmd["parameters"]["target_player_id"] in {2, 3, 4}


def test_control_action_requires_ball_close_to_reported_holder():
    state = copy.deepcopy(GAME_STATE)
    forward = next(
        p for p in state["players"]
        if p["teamCode"] == "home" and p["agentId"] == "agentId_3"
    )
    forward["position"] = {"x": 40, "y": 0}
    state["ball"].update({
        "position": {"x": 35, "y": 0},
        "isFree": False,
        "possessionAgentId": "agentId_3",
    })

    cmd = build_fast_path("FWD1")(state, 0, 3)[0]

    assert cmd["commandType"] == "MOVE_TO"


def test_attacking_forwards_keep_different_depth_and_width():
    state = copy.deepcopy(GAME_STATE)
    midfielder = next(
        p for p in state["players"]
        if p["teamCode"] == "home" and p["agentId"] == "agentId_2"
    )
    midfielder["position"] = {"x": 0, "y": 0}
    state["ball"].update({
        "position": dict(midfielder["position"]),
        "isFree": False,
        "possessionAgentId": "agentId_2",
    })

    first = build_fast_path("FWD1")(state, 0, 3)[0]
    second = build_fast_path("FWD2")(state, 0, 4)[0]

    assert first["commandType"] == second["commandType"] == "MOVE_TO"
    assert first["parameters"]["target_x"] > second["parameters"]["target_x"]
    assert first["parameters"]["target_y"] < 0 < second["parameters"]["target_y"]


def test_normalises_both_stamina_scales():
    assert stamina_percent(0.65) == 65.0
    assert stamina_percent(65) == 65.0


def test_home_and_away_attack_in_opposite_directions():
    home = build_fast_path("FWD1")(GAME_STATE, 0, 3)[0]
    away_state = _state_with_away_possession(3)
    away = build_fast_path("FWD1")(away_state, 1, 3)[0]
    assert home["commandType"] in {"MOVE_TO", "PASS", "SHOOT"}
    assert away["commandType"] in {"MOVE_TO", "PASS", "SHOOT"}
    if home["commandType"] == away["commandType"] == "MOVE_TO":
        assert home["parameters"]["target_x"] > 0
        assert away["parameters"]["target_x"] < 0


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
    print(f"{len(tests)} fast-path tests passed")
