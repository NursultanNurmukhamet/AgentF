"""Latency-bounded Sonnet advisor on top of the deterministic v4 policy."""

from __future__ import annotations

import json
import re
from typing import Callable

import boto3
from botocore.config import Config

from fallback import FallbackConfig, build_last_resort
from parsing import parse_commands
from state import summarize_state


MODEL_ID = "us.anthropic.claude-sonnet-4-6"
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
            region_name="us-east-1",
            config=Config(
                connect_timeout=0.25,
                read_timeout=1.15,
                retries={"max_attempts": 0},
            ),
        )
    return _CLIENT


def _ask_sonnet(system_prompt: str, state_summary: str) -> str:
    response = _client().converse(
        modelId=MODEL_ID,
        system=[{"text": system_prompt}],
        messages=[{"role": "user", "content": [{"text": state_summary}]}],
        inferenceConfig={"maxTokens": 96, "temperature": 0.0},
    )
    return "".join(
        part.get("text", "")
        for part in response.get("output", {}).get("message", {}).get("content", [])
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
    """Use v4 for urgent actions and Sonnet 4.6 for non-urgent tactics."""
    last_resort = build_last_resort(fallback_cfg, my_player_id)

    @app.entrypoint
    async def invoke(payload, context):
        prompt = payload.get("prompt", "{}") if isinstance(payload, dict) else "{}"
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

            commands = fast_path_fn(game_state, team_id, effective_pid)
            source = "v4"
            if not commands:
                summary = summarize_state(game_state, team_id, effective_pid, position_label)
                raw = _ask_sonnet(system_prompt, summary)
                commands = parse_commands(raw, team_id, effective_pid)
                source = "sonnet-4.6"
            if not commands:
                commands = fallback_fn(game_state, team_id, effective_pid)
                source = "fallback"
        except Exception as exc:
            app.logger.warning(f"{position_label} bounded model path failed: {exc}")
            try:
                data = json.loads(prompt) if isinstance(prompt, str) else (prompt or {})
                game_state = data.get("gameState", {})
                team_id = _normalise_team(data.get("teamId", 0))
                my_players = data.get("myPlayers", [my_player_id])
                effective_pid = _normalise_id(my_players[0] if my_players else my_player_id, my_player_id)
                commands = fallback_fn(game_state, team_id, effective_pid)
                source = "fallback"
            except Exception:
                team_id, effective_pid = 0, my_player_id
                command = dict(last_resort)
                command["playerId"] = effective_pid
                commands = [command]
                source = "last-resort"

        if (
            not isinstance(commands, list)
            or not commands
            or not all(isinstance(command, dict) for command in commands)
        ):
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
