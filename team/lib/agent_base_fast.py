"""Deterministic AgentCore handler for latency-critical football agents.

This module intentionally has no Strands or Bedrock model imports.  A valid
game tick is handled by the fast policy; malformed or unfamiliar payloads go
straight to the rule fallback instead of blocking on an LLM call.
"""

from __future__ import annotations

import json
import re
from typing import Callable

from fallback import FallbackConfig, build_last_resort


_TRAILING_INT = re.compile(r"(-?\d+)$")


def _normalise_id(value, default: int) -> int:
    """Accept numeric IDs and current ``agentId_N`` string identifiers."""
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


def decide_fast(
    payload: dict,
    my_player_id: int,
    fast_path_fn: Callable[[dict, int, int], list[dict] | None],
    fallback_fn: Callable[[dict, int, int], list[dict]],
    last_resort: dict,
) -> tuple[list[dict], bool]:
    """Return commands and whether the rule fallback was required."""
    prompt = payload.get("prompt", "{}") if isinstance(payload, dict) else "{}"
    try:
        prompt_data = json.loads(prompt) if isinstance(prompt, str) else prompt
        if not isinstance(prompt_data, dict):
            prompt_data = {}
    except (TypeError, ValueError, json.JSONDecodeError):
        prompt_data = {}

    game_state = prompt_data.get("gameState", {})
    if not isinstance(game_state, dict):
        game_state = {}
    team_id = _normalise_team(prompt_data.get("teamId", 0))
    my_players = prompt_data.get("myPlayers", [my_player_id])
    raw_pid = my_players[0] if isinstance(my_players, list) and my_players else my_player_id
    effective_pid = _normalise_id(raw_pid, my_player_id)

    used_fallback = False
    try:
        commands = fast_path_fn(game_state, team_id, effective_pid)
    except Exception:
        commands = None

    if not commands:
        used_fallback = True
        try:
            commands = fallback_fn(game_state, team_id, effective_pid)
        except Exception:
            commands = None

    if not isinstance(commands, list) or not commands:
        command = dict(last_resort)
        command["playerId"] = effective_pid
        commands = [command]
        used_fallback = True

    # Keep ownership authoritative even when a fallback template was used.
    for command in commands:
        command["playerId"] = effective_pid
        command["teamId"] = team_id
    return commands, used_fallback


def create_fast_invoke_handler(
    app,
    my_player_id: int,
    position_label: str,
    fallback_fn: Callable[[dict, int, int], list[dict]],
    fallback_cfg: FallbackConfig,
    fast_path_fn: Callable[[dict, int, int], list[dict] | None],
):
    """Register an AgentCore entrypoint that can never call an LLM."""
    last_resort = build_last_resort(fallback_cfg, my_player_id)

    @app.entrypoint
    async def invoke(payload, context):
        commands, used_fallback = decide_fast(
            payload, my_player_id, fast_path_fn, fallback_fn, last_resort
        )
        if used_fallback:
            app.logger.warning(f"{position_label} used deterministic rule fallback")
        # The tournament caller currently consumes the AgentCore event stream.
        # Keep this as a single compact frame; a plain application/json return
        # is valid AgentCore HTTP but is rejected by the Player Portal fitness
        # parser (verified 2026-08-29).
        yield json.dumps(commands, separators=(",", ":"))

    return invoke
