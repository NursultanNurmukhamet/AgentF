"""Standalone offline scenario runner and microbenchmark for ``fast_path``.

No AWS SDK, model, network, or AgentCore runtime is used.  The benchmark
measures the local Python decision function only, one player invocation at a
time, after a warm-up pass.
"""

from __future__ import annotations

import argparse
import copy
import time
from dataclasses import dataclass
from typing import Iterable

from fast_path import build_fast_path


ROLES = ((0, "GK"), (1, "DEF"), (2, "MID"), (3, "FWD1"), (4, "FWD2"))
ALLOWED_COMMANDS = {
    "MOVE_TO",
    "FOLLOW_PLAYER",
    "SHOOT",
    "PASS",
    "GK_DISTRIBUTE",
    "SLIDE_TACKLE",
    "PRESS_BALL",
    "MARK",
    "INTERCEPT",
    "SET_STANCE",
    "CLEAR_OVERRIDE",
    "RESET",
}


@dataclass(frozen=True)
class Scenario:
    """One complete game state viewed by all five agents of one team."""

    name: str
    category: str
    team_id: int
    state: dict


def _player(player_id: int, team_code: str, x: float, y: float) -> dict:
    return {
        "agentId": f"agentId_{player_id}",
        "teamCode": team_code,
        "position": {"x": x, "y": y},
        "velocity": {"x": 0.0, "y": 0.0},
        "orientation": 0.0,
        "stamina": 0.8,
        "currentAction": 0,
        "speed": 0.0,
        "isSprinting": False,
    }


def _base_state() -> dict:
    home = [
        _player(0, "home", -50, 0),
        _player(1, "home", -24, 0),
        _player(2, "home", -5, 0),
        _player(3, "home", 18, -9),
        _player(4, "home", 18, 9),
    ]
    away = [
        _player(0, "away", 50, 0),
        _player(1, "away", 24, 0),
        _player(2, "away", 5, 0),
        _player(3, "away", -18, 9),
        _player(4, "away", -18, -9),
    ]
    return {
        "tick": 1,
        "gameTime": 1.0,
        "playMode": "OPEN_PLAY",
        "score": {"home": 0, "away": 0},
        "ball": {
            "position": {"x": 0.0, "y": 0.0, "z": 0.0},
            "velocity": {"x": 0.0, "y": 0.0, "z": 0.0},
            "isFree": True,
            "possessionAgentId": None,
        },
        "players": home + away,
        "teamChat": [],
    }


def _find_player(state: dict, team_code: str, player_id: int) -> dict:
    return next(
        player
        for player in state["players"]
        if player["teamCode"] == team_code
        and player["agentId"] == f"agentId_{player_id}"
    )


def _possession_state(
    team_code: str,
    player_id: int,
    position: tuple[float, float] | None = None,
) -> dict:
    state = _base_state()
    holder = _find_player(state, team_code, player_id)
    if position is not None:
        holder["position"] = {"x": position[0], "y": position[1]}
    state["ball"]["position"] = {
        "x": holder["position"]["x"],
        "y": holder["position"]["y"],
        "z": 0.0,
    }
    state["ball"]["isFree"] = False
    state["ball"]["possessionAgentId"] = holder["agentId"]
    return state


def build_scenarios() -> tuple[Scenario, ...]:
    """Return representative HOME/AWAY attack, build-up and defence states."""
    free_home = _base_state()
    free_home["ball"]["position"] = {"x": 2.0, "y": 1.0, "z": 0.0}

    free_away = copy.deepcopy(free_home)

    home_high_press = _possession_state("home", 2, (-10.0, 0.0))
    _find_player(home_high_press, "home", 3)["velocity"] = {"x": 2.0, "y": 0.0}
    _find_player(home_high_press, "home", 4)["velocity"] = {"x": 1.5, "y": 0.0}
    _find_player(home_high_press, "away", 2)["position"] = {"x": -7.0, "y": -3.0}
    _find_player(home_high_press, "away", 3)["position"] = {"x": -5.0, "y": 3.0}

    away_high_press = _possession_state("away", 2, (10.0, 0.0))
    _find_player(away_high_press, "away", 3)["velocity"] = {"x": -2.0, "y": 0.0}
    _find_player(away_high_press, "away", 4)["velocity"] = {"x": -1.5, "y": 0.0}
    _find_player(away_high_press, "home", 2)["position"] = {"x": 7.0, "y": 3.0}
    _find_player(away_high_press, "home", 3)["position"] = {"x": 5.0, "y": -3.0}

    home_total_attack = _possession_state("away", 3, (10.0, 0.0))
    home_total_attack["ball"]["velocity"] = {"x": -3.0, "y": 0.0, "z": 0.0}
    _find_player(home_total_attack, "home", 1)["position"] = {"x": -30.0, "y": 0.0}
    _find_player(home_total_attack, "home", 2)["position"] = {"x": 0.0, "y": 0.0}
    _find_player(home_total_attack, "home", 3)["position"] = {"x": 8.0, "y": -6.0}
    _find_player(home_total_attack, "home", 4)["position"] = {"x": 4.0, "y": 6.0}

    away_total_attack = _possession_state("home", 3, (-10.0, 0.0))
    away_total_attack["ball"]["velocity"] = {"x": 3.0, "y": 0.0, "z": 0.0}
    _find_player(away_total_attack, "away", 1)["position"] = {"x": 30.0, "y": 0.0}
    _find_player(away_total_attack, "away", 2)["position"] = {"x": 0.0, "y": 0.0}
    _find_player(away_total_attack, "away", 3)["position"] = {"x": -8.0, "y": 6.0}
    _find_player(away_total_attack, "away", 4)["position"] = {"x": -4.0, "y": -6.0}

    return (
        Scenario(
            "home_forward_shot", "attack", 0,
            _possession_state("home", 3, (35.0, -6.0)),
        ),
        Scenario(
            "away_forward_shot", "attack", 1,
            _possession_state("away", 4, (-35.0, 6.0)),
        ),
        Scenario(
            "home_midfield_build", "build_up", 0,
            _possession_state("home", 2, (0.0, -2.0)),
        ),
        Scenario(
            "away_midfield_build", "build_up", 1,
            _possession_state("away", 2, (0.0, 2.0)),
        ),
        Scenario(
            "home_goalkeeper_possession", "goalkeeper", 0,
            _possession_state("home", 0, (-50.0, 0.0)),
        ),
        Scenario(
            "away_goalkeeper_possession", "goalkeeper", 1,
            _possession_state("away", 0, (50.0, 0.0)),
        ),
        Scenario("home_high_press_escape", "high_press", 0, home_high_press),
        Scenario("away_high_press_escape", "high_press", 1, away_high_press),
        Scenario("home_total_attack_cover", "full_attack", 0, home_total_attack),
        Scenario("away_total_attack_cover", "full_attack", 1, away_total_attack),
        Scenario("home_free_ball", "free_ball", 0, free_home),
        Scenario("away_free_ball", "free_ball", 1, free_away),
        Scenario(
            "home_defends_danger", "defence", 0,
            _possession_state("away", 3, (-20.0, 0.0)),
        ),
        Scenario(
            "away_defends_danger", "defence", 1,
            _possession_state("home", 3, (20.0, 0.0)),
        ),
    )


def validate_command(command: dict, expected_team: int, expected_player: int) -> None:
    """Raise ``AssertionError`` if a command violates the documented contract."""
    assert isinstance(command, dict), "command must be an object"
    required = {"commandType", "playerId", "teamId", "parameters", "duration"}
    assert required <= command.keys(), f"missing command fields: {required - command.keys()}"
    kind = command["commandType"]
    assert kind in ALLOWED_COMMANDS, f"unknown command type: {kind}"
    assert command["teamId"] == expected_team
    assert command["playerId"] == expected_player
    assert isinstance(command["parameters"], dict)
    assert isinstance(command["duration"], int) and command["duration"] >= 0

    params = command["parameters"]
    if kind == "MOVE_TO":
        assert -55 <= params["target_x"] <= 55
        assert -35 <= params["target_y"] <= 35
        assert isinstance(params["sprint"], bool)
    elif kind == "FOLLOW_PLAYER":
        _validate_player_id(params["target_player_id"])
        assert params["target_team"] in {"HOME", "AWAY"}
        assert isinstance(params["distance"], (int, float)) and params["distance"] >= 0
    elif kind == "SHOOT":
        assert params["aim_location"] in {"TL", "TR", "BL", "BR", "CENTER"}
        assert 0 <= params["power"] <= 1
    elif kind == "PASS":
        _validate_player_id(params["target_player_id"])
        assert params["type"] in {"GROUND", "AERIAL", "THROUGH"}
    elif kind == "GK_DISTRIBUTE":
        _validate_player_id(params["target_player_id"])
        assert params["method"] in {"THROW", "KICK"}
    elif kind == "SLIDE_TACKLE":
        _validate_player_id(params["target_player_id"])
        assert isinstance(params["sprint"], bool)
        assert isinstance(params["distance"], (int, float)) and params["distance"] >= 0
    elif kind == "PRESS_BALL":
        assert 0 <= params["intensity"] <= 1
    elif kind == "MARK":
        _validate_player_id(params["target_player_id"])
        assert params["tightness"] in {"LOOSE", "TIGHT"}
    elif kind == "INTERCEPT":
        assert isinstance(params["aggressive"], bool)
    elif kind == "SET_STANCE":
        assert params["stance"] in {0, 1, 2}


def _validate_player_id(value: object) -> None:
    assert isinstance(value, int) and 0 <= value <= 4


def run_scenario(scenario: Scenario, validate: bool = True) -> dict[int, dict]:
    """Run all five policies for a scenario and return one command per player."""
    commands: dict[int, dict] = {}
    for player_id, role in ROLES:
        result = build_fast_path(role)(scenario.state, scenario.team_id, player_id)
        assert isinstance(result, list) and len(result) == 1
        command = result[0]
        if validate:
            validate_command(command, scenario.team_id, player_id)
        commands[player_id] = command
    return commands


def percentile(sorted_values: list[int], percent: float) -> int:
    """Nearest-rank percentile for a non-empty sorted sample."""
    assert sorted_values
    index = max(0, min(len(sorted_values) - 1, int(len(sorted_values) * percent + 0.999999) - 1))
    return sorted_values[index]


def benchmark(
    scenarios: Iterable[Scenario],
    iterations: int = 2_000,
    warmup: int = 100,
) -> dict[str, float | int]:
    """Measure individual local decisions and return latency percentiles."""
    scenario_list = tuple(scenarios)
    policies = {player_id: build_fast_path(role) for player_id, role in ROLES}

    for _ in range(warmup):
        for scenario in scenario_list:
            for player_id, _ in ROLES:
                policies[player_id](scenario.state, scenario.team_id, player_id)

    samples_ns: list[int] = []
    wall_start = time.perf_counter_ns()
    for _ in range(iterations):
        for scenario in scenario_list:
            for player_id, _ in ROLES:
                started = time.perf_counter_ns()
                policies[player_id](scenario.state, scenario.team_id, player_id)
                samples_ns.append(time.perf_counter_ns() - started)
    wall_ns = time.perf_counter_ns() - wall_start
    samples_ns.sort()
    decisions = len(samples_ns)
    return {
        "scenarios": len(scenario_list),
        "decisions": decisions,
        "p50_ms": percentile(samples_ns, 0.50) / 1_000_000,
        "p95_ms": percentile(samples_ns, 0.95) / 1_000_000,
        "p99_ms": percentile(samples_ns, 0.99) / 1_000_000,
        "wall_ms": wall_ns / 1_000_000,
        "decisions_per_second": decisions / (wall_ns / 1_000_000_000),
    }


def _print_scenarios(scenarios: Iterable[Scenario]) -> None:
    print("Scenario command matrix")
    print("scenario                         side  commands P0..P4")
    for scenario in scenarios:
        commands = run_scenario(scenario)
        kinds = ", ".join(commands[player_id]["commandType"] for player_id, _ in ROLES)
        side = "HOME" if scenario.team_id == 0 else "AWAY"
        print(f"{scenario.name:32} {side:5} {kinds}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=2_000)
    parser.add_argument("--warmup", type=int, default=100)
    args = parser.parse_args()
    if args.iterations < 1 or args.warmup < 0:
        parser.error("--iterations must be positive and --warmup must be non-negative")

    scenarios = build_scenarios()
    _print_scenarios(scenarios)
    stats = benchmark(scenarios, iterations=args.iterations, warmup=args.warmup)
    print("\nLocal decision latency (Python only, excludes network/AgentCore/model)")
    print(
        f"{stats['decisions']} decisions across {stats['scenarios']} scenarios: "
        f"p50={stats['p50_ms']:.4f} ms, p95={stats['p95_ms']:.4f} ms, "
        f"p99={stats['p99_ms']:.4f} ms"
    )
    print(
        f"wall={stats['wall_ms']:.2f} ms, "
        f"throughput={stats['decisions_per_second']:.0f} decisions/s"
    )


if __name__ == "__main__":
    main()
