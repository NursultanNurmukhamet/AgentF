"""Offline scenario regressions for the deterministic fast-path policy."""

from benchmark_fast_path import build_scenarios, run_scenario, validate_command


SCENARIOS = {scenario.name: scenario for scenario in build_scenarios()}


def test_scenario_set_covers_both_sides_and_phases():
    assert {scenario.team_id for scenario in SCENARIOS.values()} == {0, 1}
    categories = {scenario.category for scenario in SCENARIOS.values()}
    assert {
        "attack", "build_up", "goalkeeper", "high_press",
        "full_attack", "free_ball", "defence",
    } <= categories


def test_every_scenario_returns_one_valid_command_per_player():
    for scenario in SCENARIOS.values():
        commands = run_scenario(scenario, validate=False)
        assert set(commands) == {0, 1, 2, 3, 4}
        for player_id, command in commands.items():
            validate_command(command, scenario.team_id, player_id)


def test_home_and_away_forwards_shoot_near_goal():
    home = run_scenario(SCENARIOS["home_forward_shot"])
    away = run_scenario(SCENARIOS["away_forward_shot"])
    assert home[3]["commandType"] == "SHOOT"
    assert away[4]["commandType"] == "SHOOT"


def test_home_and_away_midfielders_pass_during_build_up():
    home = run_scenario(SCENARIOS["home_midfield_build"])
    away = run_scenario(SCENARIOS["away_midfield_build"])
    assert home[2]["commandType"] == "PASS"
    assert away[2]["commandType"] == "PASS"
    assert home[2]["parameters"]["type"] == "THROUGH"
    assert away[2]["parameters"]["type"] == "THROUGH"


def test_goalkeepers_attempt_full_power_direct_shots_for_both_sides():
    home = run_scenario(SCENARIOS["home_goalkeeper_possession"])
    away = run_scenario(SCENARIOS["away_goalkeeper_possession"])
    assert home[0]["commandType"] == "SHOOT"
    assert away[0]["commandType"] == "SHOOT"
    assert home[0]["parameters"]["power"] == 1.0
    assert away[0]["parameters"]["power"] == 1.0


def test_high_press_escape_is_long_and_forward_for_both_sides():
    home = run_scenario(SCENARIOS["home_high_press_escape"])
    away = run_scenario(SCENARIOS["away_high_press_escape"])
    for commands in (home, away):
        assert commands[2]["commandType"] == "PASS"
        assert commands[2]["parameters"]["target_player_id"] in {3, 4}
        assert commands[2]["parameters"]["type"] == "AERIAL"


def test_total_attack_has_one_presser_and_one_separate_lane_interceptor():
    for name in ("home_total_attack_cover", "away_total_attack_cover"):
        commands = run_scenario(SCENARIOS[name])
        pressers = [
            pid for pid in range(1, 5)
            if commands[pid]["commandType"] == "PRESS_BALL"
        ]
        interceptors = [
            pid for pid in range(1, 5)
            if commands[pid]["commandType"] == "INTERCEPT"
        ]
        assert len(pressers) == 1
        assert len(interceptors) == 1
        assert pressers[0] != interceptors[0]
        for player_id in (3, 4):
            command = commands[player_id]
            if command["commandType"] == "MOVE_TO":
                assert abs(command["parameters"]["target_y"]) <= 12


def test_free_ball_has_exactly_one_outfield_interceptor_per_team():
    for name in ("home_free_ball", "away_free_ball"):
        commands = run_scenario(SCENARIOS[name])
        interceptors = [pid for pid in range(1, 5) if commands[pid]["commandType"] == "INTERCEPT"]
        assert len(interceptors) == 1


def test_defensive_pressure_has_at_most_one_outfield_presser():
    for name in ("home_defends_danger", "away_defends_danger"):
        commands = run_scenario(SCENARIOS[name])
        pressers = [pid for pid in range(1, 5) if commands[pid]["commandType"] == "PRESS_BALL"]
        assert len(pressers) <= 1
        assert any(
            commands[pid]["commandType"] in {"PRESS_BALL", "MARK"}
            for pid in range(1, 5)
        )


def test_off_ball_forward_shape_is_mirrored_by_side():
    home = run_scenario(SCENARIOS["home_midfield_build"])
    away = run_scenario(SCENARIOS["away_midfield_build"])
    for player_id in (3, 4):
        assert home[player_id]["commandType"] == "MOVE_TO"
        assert away[player_id]["commandType"] == "MOVE_TO"
        assert home[player_id]["parameters"]["target_x"] > 0
        assert away[player_id]["parameters"]["target_x"] < 0


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
    print(f"{len(tests)} scenario tests passed")
