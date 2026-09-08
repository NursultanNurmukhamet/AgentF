import asyncio
import inspect
import json

from agent_base_fast import create_fast_invoke_handler, decide_fast
from fallback import FallbackConfig


def _payload(my_player="agentId_3", team="home"):
    return {
        "prompt": json.dumps({
            "teamId": team,
            "myPlayers": [my_player],
            "gameState": {"players": [], "ball": {}},
        })
    }


def test_normalises_agent_id_and_team_without_model_path():
    seen = {}

    def fast(state, team_id, player_id):
        seen.update(team=team_id, player=player_id)
        return [{"commandType": "MOVE_TO", "parameters": {}}]

    commands, used_fallback = decide_fast(
        _payload(), 3, fast, lambda *_: [],
        {"commandType": "MOVE_TO", "playerId": 3, "teamId": 0, "parameters": {}},
    )

    assert seen == {"team": 0, "player": 3}
    assert commands[0]["playerId"] == 3
    assert commands[0]["teamId"] == 0
    assert not used_fallback


def test_fast_path_none_uses_rules_not_llm():
    fallback_calls = []

    def fallback(state, team_id, player_id):
        fallback_calls.append((team_id, player_id))
        return [{"commandType": "INTERCEPT", "parameters": {"aggressive": False}}]

    commands, used_fallback = decide_fast(
        _payload("4", "away"), 4, lambda *_: None, fallback,
        {"commandType": "MOVE_TO", "playerId": 4, "teamId": 0, "parameters": {}},
    )

    assert fallback_calls == [(1, 4)]
    assert commands[0]["commandType"] == "INTERCEPT"
    assert commands[0]["teamId"] == 1
    assert used_fallback


def test_malformed_payload_returns_last_resort():
    commands, used_fallback = decide_fast(
        {"prompt": "not-json"}, 2, lambda *_: None, lambda *_: [],
        {"commandType": "MOVE_TO", "playerId": 2, "teamId": 0,
         "parameters": {"target_x": 0, "target_y": 0, "sprint": False}},
    )

    assert commands[0]["commandType"] == "MOVE_TO"
    assert commands[0]["playerId"] == 2
    assert used_fallback


def test_handler_returns_single_compact_event_stream_frame():
    class App:
        class logger:
            @staticmethod
            def warning(_message):
                pass

        @staticmethod
        def entrypoint(fn):
            return fn

    handler = create_fast_invoke_handler(
        App(), 3, "MID", lambda *_: [], FallbackConfig(),
        lambda *_: [{"commandType": "MOVE_TO", "parameters": {}}],
    )
    assert inspect.isasyncgen(handler(_payload(), None))

    async def collect():
        return [item async for item in handler(_payload(), None)]

    frames = asyncio.run(collect())
    assert len(frames) == 1
    result = json.loads(frames[0])
    assert result[0]["playerId"] == 3
