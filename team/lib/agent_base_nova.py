"""Latency-bounded, model-first Nova Micro policy with deterministic safety."""

from __future__ import annotations

import json
import os
import re
from typing import Callable

import boto3
from botocore.config import Config

from fallback import FallbackConfig, build_last_resort
from parsing import parse_commands
from state import (
    _is_my_team,
    _player_idx,
    dist,
    get_goal_positions,
    get_possession_player,
    summarize_state,
)


MODEL_ID = os.environ.get("AGENTF_MODEL_ID", "us.amazon.nova-micro-v1:0")
_POSSESSION_COMMANDS = {"PASS", "SHOOT", "GK_DISTRIBUTE"}
_ATTACKING_ROLES = {"MID", "FWD1", "FWD2"}
_SHOT_DISTANCE = {"DEF": 20.0, "MID": 28.0, "FWD1": 32.0, "FWD2": 32.0}
_CONTROL_DISTANCE = 4.0
_CROWD_RADIUS = 6.0
_TIGHT_PRESS_RADIUS = 3.0
_TRAILING_INT = re.compile(r"(-?\d+)$")
_CLIENT = None


def _normalise_id(value, default: int) -> int:
    if isinstance(value, bool) or value is None:
        return default
    if isinstance(value, (int, float)):
        return int(value)
    match = _TRAILING_INT.search(str(value).strip())
    return int(match.group(1)) if match else default


def _normalise_team(value) -> int:
    text = str(value).strip().lower()
    if text == "home":
        return 0
    if text == "away":
        return 1
    return 1 if _normalise_id(value, 0) == 1 else 0


def _client():
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = boto3.client(
            "bedrock-runtime",
            region_name=os.environ.get("AWS_REGION", "us-east-1"),
            config=Config(
                connect_timeout=0.25,
                read_timeout=1.35,
                retries={"max_attempts": 0},
            ),
        )
    return _CLIENT


def _ask_nova(system_prompt: str, state_summary: str) -> str:
    response = _client().converse(
        modelId=MODEL_ID,
        system=[{"text": system_prompt}],
        messages=[{"role": "user", "content": [{"text": state_summary}]}],
        inferenceConfig={"maxTokens": 64, "temperature": 0.0},
    )
    return "".join(
        part.get("text", "")
        for part in response.get("output", {}).get("message", {}).get("content", [])
    )


def _owns_ball(game_state: dict, team_id: int, player_id: int) -> bool:
    ball = game_state.get("ball", {})
    if ball.get("isFree"):
        return False
    holder = get_possession_player(ball, game_state.get("players", []))
    return bool(
        holder
        and _is_my_team(holder, team_id)
        and _player_idx(holder) == player_id
        and dist(holder.get("position", {}), ball.get("position", {}))
        <= _CONTROL_DISTANCE
    )


def _outlet_clearance(player: dict, opponents: list[dict]) -> float:
    position = player.get("position", {})
    return min(
        (dist(position, opponent.get("position", {})) for opponent in opponents),
        default=30.0,
    )


def _most_advanced_outlet(
    mine: list[dict], opponents: list[dict], player_id: int, attack_sign: int,
) -> dict | None:
    candidates = [
        player for player in mine
        if _player_idx(player) not in {0, player_id}
    ]
    if not candidates:
        return None
    far_half = [
        player for player in candidates
        if attack_sign * player.get("position", {}).get("x", 0) > 0
    ]
    if not far_half:
        return None
    return max(
        far_half,
        key=lambda player: (
            attack_sign * player.get("position", {}).get("x", 0),
            _outlet_clearance(player, opponents),
            -_player_idx(player),
        ),
    )


def _open_pressure_outlet(
    mine: list[dict], opponents: list[dict], player_id: int,
    carrier_position: dict, attack_sign: int,
) -> dict | None:
    options: list[tuple[float, int, dict]] = []
    for player in mine:
        target_id = _player_idx(player)
        if target_id not in {2, 3, 4} or target_id == player_id:
            continue
        target_position = player.get("position", {})
        target_attack_x = attack_sign * target_position.get("x", 0)
        in_corner = target_attack_x >= 42 and abs(target_position.get("y", 0)) >= 24
        if in_corner:
            continue
        clearance = _outlet_clearance(player, opponents)
        pass_distance = dist(carrier_position, target_position)
        score = clearance * 1.7 + target_attack_x * 0.8 - pass_distance * 0.15
        options.append((score, -target_id, player))
    return max(options, default=(0.0, 0, None))[2]


def _tactical_directive(
    game_state: dict, team_id: int, player_id: int, position_label: str,
) -> dict | None:
    """Return one state-specific tactical constraint for the model output."""
    if not _owns_ball(game_state, team_id, player_id):
        return None

    players = game_state.get("players", [])
    mine = [player for player in players if _is_my_team(player, team_id)]
    opponents = [player for player in players if not _is_my_team(player, team_id)]
    me = next((player for player in mine if _player_idx(player) == player_id), None)
    if me is None:
        return None

    position = me.get("position", {})
    attack_sign = 1 if team_id == 0 else -1
    ball_position = game_state.get("ball", {}).get("position", position)

    if position_label == "GK":
        return {
            "kind": "GK_DIRECT_SHOT",
            "instruction": (
                "TACTICAL PRIORITY: You own the ball as goalkeeper. Attempt the "
                "longest direct goal kick now: return SHOOT toward the opponent goal "
                "with power 1.0 and duration 0. Do not THROW, dribble, or hold it."
            ),
        }

    ball_attack_x = attack_sign * ball_position.get("x", 0)
    if ball_attack_x < 0:
        outlet = _most_advanced_outlet(mine, opponents, player_id, attack_sign)
        target_id = _player_idx(outlet) if outlet is not None else None
        if target_id is not None:
            instruction = (
                "TACTICAL PRIORITY: The ball is on our half. Clear it immediately "
                f"to the other half with PASS to teammate {target_id}, type AERIAL, "
                "duration 0. Do not dribble or make a short/backward pass."
            )
        else:
            instruction = (
                "TACTICAL PRIORITY: The ball is on our half and no outlet exists. "
                "Clear toward the opponent goal with SHOOT power 1.0, duration 0."
            )
        return {
            "kind": "OWN_HALF_CLEAR",
            "target_id": target_id,
            "instruction": instruction,
        }

    if position_label in _ATTACKING_ROLES:
        in_attacking_corner = (
            ball_attack_x >= 42 and abs(position.get("y", 0)) >= 24
        )
        if in_attacking_corner:
            outlet = _open_pressure_outlet(
                mine, opponents, player_id, position, attack_sign,
            )
            if outlet is not None:
                target_id = _player_idx(outlet)
                return {
                    "kind": "CORNER_RECYCLE",
                    "target_id": target_id,
                    "instruction": (
                        f"TACTICAL PRIORITY: Recycle out of the attacking corner. "
                        f"PASS now to open teammate {target_id}, duration 0. Do not "
                        "dribble deeper into the corner."
                    ),
                }

        nearby = [
            opponent for opponent in opponents
            if dist(position, opponent.get("position", {})) <= _CROWD_RADIUS
        ]
        tightly_pressed = any(
            dist(position, opponent.get("position", {})) <= _TIGHT_PRESS_RADIUS
            for opponent in opponents
        )
        if len(nearby) >= 2 or tightly_pressed:
            outlet = _open_pressure_outlet(
                mine, opponents, player_id, position, attack_sign,
            )
            if outlet is not None:
                target_id = _player_idx(outlet)
                return {
                    "kind": "CROWDED_PASS",
                    "target_id": target_id,
                    "instruction": (
                        f"TACTICAL PRIORITY: You are under heavy pressure. PASS now "
                        f"to open teammate {target_id} with GROUND, THROUGH, or AERIAL "
                        "and duration 0. Do not shoot or dribble this tick."
                    ),
                }

    max_shot_distance = _SHOT_DISTANCE.get(position_label)
    if max_shot_distance is not None:
        _, opponent_goal_x = get_goal_positions(team_id)
        goal_distance = dist(position, {"x": opponent_goal_x, "y": 0})
        if goal_distance <= max_shot_distance:
            return {
                "kind": "ATTACKER_SHOT",
                "instruction": (
                    f"TACTICAL PRIORITY: You are {goal_distance:.1f} units from goal. "
                    "Shoot on this first possession tick: return SHOOT toward a goal "
                    "corner with power at least 0.95 and duration 0. Do not pass."
                ),
            }

    return None


def _number(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _directive_command_is_legal(commands: list[dict], directive: dict | None) -> bool:
    if directive is None:
        return True
    if len(commands) != 1:
        return False
    command = commands[0]
    command_type = command.get("commandType")
    parameters = command.get("parameters", {})
    if not isinstance(parameters, dict):
        return False
    kind = directive.get("kind")

    if kind in {"GK_DIRECT_SHOT", "ATTACKER_SHOT"}:
        return command_type == "SHOOT" and _number(parameters.get("power")) >= 0.95
    if kind in {"CROWDED_PASS", "CORNER_RECYCLE"}:
        return (
            command_type == "PASS"
            and _normalise_id(parameters.get("target_player_id"), -1)
            == directive.get("target_id")
            and parameters.get("type") in {"GROUND", "THROUGH", "AERIAL"}
        )
    if kind == "OWN_HALF_CLEAR":
        target_id = directive.get("target_id")
        if target_id is None:
            return command_type == "SHOOT" and _number(parameters.get("power")) >= 0.95
        return (
            command_type == "PASS"
            and _normalise_id(parameters.get("target_player_id"), -1) == target_id
            and parameters.get("type") in {"AERIAL", "THROUGH"}
        )
    return True


def _model_commands_are_legal(
    commands: list[dict], game_state: dict, team_id: int, player_id: int,
    directive: dict | None = None,
) -> bool:
    needs_possession = any(
        command.get("commandType") in _POSSESSION_COMMANDS for command in commands
    )
    ownership_is_legal = not needs_possession or _owns_ball(
        game_state, team_id, player_id,
    )
    return ownership_is_legal and _directive_command_is_legal(commands, directive)


def _commands_are_valid(commands) -> bool:
    return bool(
        isinstance(commands, list)
        and len(commands) == 1
        and all(
            isinstance(command, dict) and bool(command.get("commandType"))
            for command in commands
        )
    )


def _ownership_instruction(game_state: dict, team_id: int, player_id: int) -> str:
    if _owns_ball(game_state, team_id, player_id):
        return (
            "TICK OWNERSHIP RULE: YOU OWN THE BALL. Return exactly ONE JSON "
            "command in the required one-element JSON array, with no prose."
        )
    return (
        "TICK OWNERSHIP RULE: YOU DO NOT OWN THE BALL. PASS, SHOOT, and "
        "GK_DISTRIBUTE are STRICTLY FORBIDDEN this tick. Choose exactly ONE "
        "off-ball command only from MOVE_TO, PRESS_BALL, INTERCEPT, MARK, "
        "FOLLOW_PLAYER, SET_STANCE. Return it in the required one-element JSON "
        "array, with no prose."
    )


def create_hybrid_invoke_handler(
    app,
    system_prompt: str,
    my_player_id: int,
    position_label: str,
    fallback_fn: Callable[[dict, int, int], list[dict]],
    fallback_cfg: FallbackConfig,
    fast_path_fn: Callable[[dict, int, int], list[dict] | None],
):
    """Ask Nova on every valid tick; enforce V6 tactics with bounded safety."""
    last_resort = build_last_resort(fallback_cfg, my_player_id)

    @app.entrypoint
    async def invoke(payload, context):
        prompt = payload.get("prompt", "{}") if isinstance(payload, dict) else "{}"
        game_state: dict = {}
        team_id = 0
        effective_pid = my_player_id
        commands: list[dict] = []
        source = "last-resort"
        directive: dict | None = None

        try:
            data = json.loads(prompt) if isinstance(prompt, str) else prompt
            if not isinstance(data, dict):
                raise ValueError("payload is not an object")
            game_state = data.get("gameState", {})
            if not isinstance(game_state, dict):
                raise ValueError("gameState is not an object")
            team_id = _normalise_team(data.get("teamId", 0))
            my_players = data.get("myPlayers", [my_player_id])
            raw_pid = my_players[0] if isinstance(my_players, list) and my_players else my_player_id
            effective_pid = _normalise_id(raw_pid, my_player_id)

            summary = summarize_state(game_state, team_id, effective_pid, position_label)
            directive = _tactical_directive(
                game_state, team_id, effective_pid, position_label,
            )
            tactical_instruction = (
                f"\n{directive['instruction']}" if directive is not None else ""
            )
            constrained_state = (
                f"{_ownership_instruction(game_state, team_id, effective_pid)}"
                f"{tactical_instruction}\n\n"
                f"{summary}"
            )
            raw = _ask_nova(system_prompt, constrained_state)
            candidate = parse_commands(raw, team_id, effective_pid)
            if _commands_are_valid(candidate) and _model_commands_are_legal(
                candidate, game_state, team_id, effective_pid, directive,
            ):
                commands = candidate
                source = "nova-micro"
            else:
                app.logger.warning(
                    f"{position_label} model returned invalid or illegal commands "
                    f"for directive={directive.get('kind') if directive else 'NORMAL'}"
                )
        except Exception as exc:
            app.logger.warning(f"{position_label} bounded model-first path failed: {exc}")

        # A deterministic decision is safety equipment, never the primary policy.
        if not _commands_are_valid(commands):
            try:
                candidate = fast_path_fn(game_state, team_id, effective_pid)
                if _commands_are_valid(candidate) and _model_commands_are_legal(
                    candidate, game_state, team_id, effective_pid, directive,
                ):
                    commands = candidate
                    source = "v6-safety"
            except Exception as exc:
                app.logger.warning(f"{position_label} fast safety failed: {exc}")

        if not _commands_are_valid(commands):
            try:
                candidate = fallback_fn(game_state, team_id, effective_pid)
                if _commands_are_valid(candidate) and _model_commands_are_legal(
                    candidate, game_state, team_id, effective_pid, None,
                ):
                    commands = candidate
                    source = "fallback"
            except Exception as exc:
                app.logger.warning(f"{position_label} fallback failed: {exc}")

        if not _commands_are_valid(commands):
            commands = [dict(last_resort)]
            source = "last-resort"
        else:
            commands = [dict(command) for command in commands]

        for command in commands:
            command["playerId"] = effective_pid
            command["teamId"] = team_id
        app.logger.info(f"{position_label} decision source={source} model={MODEL_ID}")
        yield json.dumps(commands, separators=(",", ":"))

    return invoke
