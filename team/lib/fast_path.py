"""Low-latency counter-attacking policy for practice-match experiments.

The policy deliberately stays within the documented command and coordinate
ranges.  It can run without a model call, making decision latency predictable.
"""

from __future__ import annotations

from typing import Callable

from state import (
    _is_my_team,
    _player_idx,
    dist,
    get_goal_positions,
    get_possession_info,
    get_possession_player,
    stamina_percent,
)


CONTROL_DISTANCE = 4.0
INTERCEPT_DISTANCE = 12.0
PRESS_DISTANCE = 12.0
SHOT_DISTANCE_BY_ROLE = {
    "DEF": 20.0,
    "MID": 28.0,
    "FWD1": 32.0,
    "FWD2": 32.0,
}
_CROWD_RADIUS = 6.0
_TIGHT_PRESS_RADIUS = 3.0
FRESH_STAMINA = 65.0
MANAGED_STAMINA = 35.0


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _stamina_tier(stamina: float) -> str:
    if stamina >= FRESH_STAMINA:
        return "FRESH"
    if stamina >= MANAGED_STAMINA:
        return "MANAGED"
    return "LOW"


def _player_stamina_tier(player: dict) -> str:
    return _stamina_tier(stamina_percent(player.get("stamina", 100)))


def _cmd(kind: str, pid: int, tid: int, parameters: dict, duration: int = 0) -> list[dict]:
    return [{
        "commandType": kind,
        "playerId": pid,
        "teamId": tid,
        "parameters": parameters,
        "duration": duration,
    }]


def _most_advanced(players: list[dict], attack_sign: int, candidates: set[int]) -> dict | None:
    eligible = [p for p in players if _player_idx(p) in candidates]
    return max(eligible, key=lambda p: attack_sign * p.get("position", {}).get("x", 0), default=None)


def _outlet_clearance(player: dict, opponents: list[dict]) -> float:
    position = player.get("position", {})
    return min(
        (dist(position, opponent.get("position", {})) for opponent in opponents),
        default=30.0,
    )


def _clearance_outlet(
    players: list[dict], opponents: list[dict], player_id: int, attack_sign: int,
) -> dict | None:
    candidates = [
        player for player in players
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
    players: list[dict], opponents: list[dict], player_id: int,
    carrier_position: dict, attack_sign: int,
) -> dict | None:
    options: list[tuple[float, int, dict]] = []
    for player in players:
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


def _central_release(players: list[dict], player_id: int, attack_sign: int) -> dict | None:
    """Pick an outfield outlet that is central enough to escape a corner."""
    eligible = [
        p for p in players
        if _player_idx(p) not in {0, player_id}
        and abs(p.get("position", {}).get("y", 0)) <= 20
    ]
    return min(
        eligible,
        key=lambda p: (
            abs(p.get("position", {}).get("y", 0)),
            abs(attack_sign * p.get("position", {}).get("x", 0) - 30),
        ),
        default=None,
    )


def _safe_pass_target(
    players: list[dict], opponents: list[dict], player_id: int,
    pos: dict, attack_sign: int, stamina_tier: str,
) -> dict | None:
    """Pick an open moving outlet without forcing a ball into a corner."""
    options: list[tuple[float, dict]] = []
    max_distance = {"FRESH": 34.0, "MANAGED": 32.0, "LOW": 28.0}[stamina_tier]
    min_progress = -18.0 if stamina_tier == "LOW" else -12.0
    for player in players:
        if _player_idx(player) in {0, player_id}:
            continue
        target_pos = player.get("position", {})
        pass_distance = dist(pos, target_pos)
        progress = attack_sign * (target_pos.get("x", 0) - pos.get("x", 0))
        in_corner = (
            attack_sign * target_pos.get("x", 0) >= 42
            and abs(target_pos.get("y", 0)) >= 24
        )
        if pass_distance > max_distance or progress < min_progress or in_corner:
            continue
        velocity = player.get("velocity", {})
        velocity_x = velocity.get("x", 0)
        velocity_y = velocity.get("y", 0)
        outlet_motion = min(3.0, abs(velocity_x) + abs(velocity_y))
        forward_motion = _clamp(attack_sign * velocity_x, -2.0, 2.0)
        outlet_energy = stamina_percent(player.get("stamina", 100))
        clearance = min(
            (dist(target_pos, opponent.get("position", {})) for opponent in opponents),
            default=30.0,
        )
        score = (
            clearance * 1.2
            + progress * 1.5
            - pass_distance * 0.2
            - abs(target_pos.get("y", 0)) * 0.1
            + outlet_motion * 2.5
            + forward_motion * 1.5
            + (outlet_energy - 50.0) * 0.05
        )
        options.append((score, player))
    return max(options, key=lambda option: option[0], default=(0.0, None))[1]


def build_fast_path(position_label: str, full: bool = True) -> Callable:
    """Build a deterministic policy for one positional agent.

    With ``full=True`` every valid state is handled without an LLM call.  With
    ``full=False`` only possession, goalkeeper and urgent defensive decisions
    use the fast path; other states fall through to the model.
    """

    def decide(game_state: dict, team_id: int, player_id: int) -> list[dict] | None:
        players = game_state.get("players", [])
        ball = game_state.get("ball", {})
        ball_pos = ball.get("position", {})
        mine = [p for p in players if _is_my_team(p, team_id)]
        opponents = [p for p in players if not _is_my_team(p, team_id)]
        me = next((p for p in mine if _player_idx(p) == player_id), None)
        if me is None:
            return None

        pos = me.get("position", {})
        attack_sign = 1 if team_id == 0 else -1
        my_goal_x, opp_goal_x = get_goal_positions(team_id)
        ball_attack_x = attack_sign * ball_pos.get("x", 0)
        possession_id, _, we_have_ball = get_possession_info(ball, players, team_id)
        holder = get_possession_player(ball, players)
        ball_is_free = bool(ball.get("isFree"))
        if ball_is_free:
            # Some payloads can retain the last possessionAgentId for a loose
            # ball.  isFree is authoritative: nobody can act as the carrier.
            we_have_ball = False
            holder = None
        i_have_ball = (
            we_have_ball
            and possession_id == player_id
            and holder is not None
            and _is_my_team(holder, team_id)
            and dist(pos, ball_pos) <= CONTROL_DISTANCE
        )
        stamina = stamina_percent(me.get("stamina", 100))
        stamina_tier = _stamina_tier(stamina)

        # Possession decisions are bounded and cheap enough to make exactly.
        if i_have_ball:
            distance_to_goal = dist(pos, {"x": opp_goal_x, "y": 0})
            if position_label == "GK":
                # The protocol has no power field for GK_DISTRIBUTE.  A literal
                # direct goal attempt must therefore use SHOOT; the generic
                # fallback still retains GK_DISTRIBUTE if this branch ever fails.
                return _cmd("SHOOT", player_id, team_id, {
                    "aim_location": "CENTER",
                    "power": 1.0,
                })

            # Never dribble or play short on our own half.  Find the most
            # advanced outfield outlet already beyond midfield and send an aerial
            # clearance.  With no far-half outlet, kick directly toward goal.
            if ball_attack_x < 0:
                target = _clearance_outlet(
                    mine, opponents, player_id, attack_sign,
                )
                if target is not None:
                    return _cmd("PASS", player_id, team_id, {
                        "target_player_id": _player_idx(target),
                        "type": "AERIAL",
                    })
                return _cmd("SHOOT", player_id, team_id, {
                    "aim_location": "CENTER",
                    "power": 1.0,
                })

            # Do not let an attacker repeatedly drive or shoot into the corner.
            # Recycle centrally when possible; otherwise carry diagonally out.
            in_attacking_corner = ball_attack_x >= 42 and abs(pos.get("y", 0)) >= 24
            if position_label in ("MID", "FWD1", "FWD2") and in_attacking_corner:
                target = _open_pressure_outlet(
                    mine, opponents, player_id, pos, attack_sign,
                )
                if target is not None:
                    return _cmd("PASS", player_id, team_id, {
                        "target_player_id": _player_idx(target),
                        "type": "GROUND",
                    })
                return _cmd("MOVE_TO", player_id, team_id, {
                    "target_x": attack_sign * 38.0,
                    "target_y": 12.0 if pos.get("y", 0) > 0 else -12.0,
                    "sprint": stamina_tier == "FRESH",
                })

            # Under heavy pressure, release the ball before considering a shot.
            # One marker at tackling distance or two within six units counts as
            # crowded.  The receiver will use the shot rule on its next tick.
            if position_label in ("MID", "FWD1", "FWD2"):
                nearby = [
                    opponent for opponent in opponents
                    if dist(pos, opponent.get("position", {})) <= _CROWD_RADIUS
                ]
                tightly_pressed = any(
                    dist(pos, opponent.get("position", {})) <= _TIGHT_PRESS_RADIUS
                    for opponent in opponents
                )
                if len(nearby) >= 2 or tightly_pressed:
                    target = _open_pressure_outlet(
                        mine, opponents, player_id, pos, attack_sign,
                    )
                    if target is not None:
                        target_pos = target.get("position", {})
                        pass_distance = dist(pos, target_pos)
                        progress = attack_sign * (
                            target_pos.get("x", 0) - pos.get("x", 0)
                        )
                        pass_type = (
                            "AERIAL" if pass_distance > 18
                            else "THROUGH" if progress >= 6
                            else "GROUND"
                        )
                        return _cmd("PASS", player_id, team_id, {
                            "target_player_id": _player_idx(target),
                            "type": pass_type,
                        })

            shot_distance = SHOT_DISTANCE_BY_ROLE.get(position_label)
            if shot_distance is not None and distance_to_goal <= shot_distance:
                aim = "TR" if player_id in (2, 3) else "BL"
                return _cmd("SHOOT", player_id, team_id,
                            {"aim_location": aim, "power": 1.0})

            # Outside a clear shooting position, keep possession moving.  All
            # outfield roles use the same safety score, so a forward can recycle
            # instead of dribbling alone and a defender avoids hopeful long balls.
            target = _safe_pass_target(
                mine, opponents, player_id, pos, attack_sign, stamina_tier,
            )
            if target is not None:
                target_pos = target.get("position", {})
                target_progress = attack_sign * (
                    target_pos.get("x", 0) - pos.get("x", 0)
                )
                pass_type = (
                    "THROUGH"
                    if stamina_tier != "LOW"
                    and position_label in ("MID", "FWD1", "FWD2")
                    and target_progress >= 6
                    else "GROUND"
                )
                return _cmd("PASS", player_id, team_id,
                            {"target_player_id": _player_idx(target), "type": pass_type})

            # Without a safe outlet, carry in the role's lane.
            lane_y = -9.0 if position_label == "FWD1" else 9.0
            target_attack_x = _clamp(ball_attack_x + 12.0, 20.0, 45.0)
            return _cmd("MOVE_TO", player_id, team_id, {
                "target_x": attack_sign * target_attack_x,
                "target_y": lane_y,
                "sprint": stamina_tier == "FRESH",
            })

        # Goalkeeper positioning is deterministic and never crosses the box.
        if position_label == "GK":
            if ball_is_free and dist(pos, ball_pos) <= 5:
                return _cmd("INTERCEPT", player_id, team_id,
                            {"aggressive": False}, duration=1)
            return _cmd("MOVE_TO", player_id, team_id, {
                "target_x": my_goal_x + attack_sign * 4.0,
                "target_y": _clamp(ball_pos.get("y", 0) * 0.45, -12.0, 12.0),
                "sprint": False,
            })

        opponent_has_ball = holder is not None and not _is_my_team(holder, team_id)

        immediate_danger = ball_attack_x <= -25

        # Tier-aware loose-ball assignment.  Prefer one FRESH reactor, then one
        # MANAGED player; LOW joins only for immediate danger near our goal.
        collector = None
        for tier, limit in (
            ("FRESH", INTERCEPT_DISTANCE),
            ("MANAGED", 8.0),
            ("LOW", 4.0 if immediate_danger else 0.0),
        ):
            if limit <= 0:
                continue
            candidates = [
                p for p in mine
                if _player_idx(p) != 0
                and _player_stamina_tier(p) == tier
                and dist(p.get("position", {}), ball_pos) <= limit
            ]
            if candidates:
                collector = min(
                    candidates,
                    key=lambda p: dist(p.get("position", {}), ball_pos),
                )
                break
        if (
            ball_is_free
            and collector is not None
            and _player_idx(collector) == player_id
        ):
            return _cmd("INTERCEPT", player_id, team_id,
                        {"aggressive": False}, duration=1)

        # Full-Attack press: exactly one nearest eligible outfielder.  Prefer a
        # FRESH player within 12; fall back to MANAGED within 6, or LOW within 4
        # only in immediate danger.  This preserves the stamina tiers while
        # allowing DEF/FWD to press when they are the right player.
        presser = None
        if opponent_has_ball:
            for tier, limit in (
                ("FRESH", PRESS_DISTANCE),
                ("MANAGED", 6.0),
                ("LOW", 4.0 if immediate_danger else 0.0),
            ):
                if limit <= 0:
                    continue
                candidates = [
                    p for p in mine
                    if _player_idx(p) != 0
                    and _player_stamina_tier(p) == tier
                    and dist(p.get("position", {}), ball_pos) <= limit
                ]
                if candidates:
                    presser = min(
                        candidates,
                        key=lambda p: dist(p.get("position", {}), ball_pos),
                    )
                    break
        if presser is not None and _player_idx(presser) == player_id:
            return _cmd("PRESS_BALL", player_id, team_id, {
                "intensity": {
                    "FRESH": 0.5,
                    "MANAGED": 0.4,
                    "LOW": 0.35,
                }[stamina_tier],
            }, duration=1)

        # The defender protects the goal with a supported maintained MARK rather
        # than joining the chase.  Prefer the carrier in our half, otherwise the
        # opponent closest to goal.
        if position_label == "DEF" and opponent_has_ball and opponents:
            dangerous = holder if ball_attack_x <= 15 else min(
                opponents,
                key=lambda p: abs(p.get("position", {}).get("x", 0) - my_goal_x),
            )
            if abs(dangerous.get("position", {}).get("x", 0) - my_goal_x) <= 45:
                return _cmd("MARK", player_id, team_id, {
                    "target_player_id": _player_idx(dangerous),
                    "tightness": "TIGHT",
                }, duration=2)

        # A second player may cover a moving pass lane, but never joins the
        # presser at the current ball position.  DEF remains on MARK duty.
        velocity = ball.get("velocity", {})
        velocity_x = velocity.get("x", 0)
        velocity_y = velocity.get("y", 0)
        ball_motion = abs(velocity_x) + abs(velocity_y)
        if opponent_has_ball and ball_motion >= 0.75:
            projected = {
                "x": _clamp(ball_pos.get("x", 0) + velocity_x * 2.0, -55, 55),
                "y": _clamp(ball_pos.get("y", 0) + velocity_y * 2.0, -35, 35),
            }
            cover_candidates = [
                p for p in mine
                if _player_idx(p) in {2, 3, 4}
                and (presser is None or _player_idx(p) != _player_idx(presser))
                and (_player_stamina_tier(p) != "LOW" or immediate_danger)
            ]
            cover = min(
                cover_candidates,
                key=lambda p: dist(p.get("position", {}), projected),
                default=None,
            )
            if (
                cover is not None
                and _player_idx(cover) == player_id
                and dist(pos, projected) <= 10
            ):
                return _cmd("INTERCEPT", player_id, team_id,
                            {"aggressive": False}, duration=1)

        if not full:
            return None

        # Coordinated shape in attack-relative coordinates.  Without possession
        # the team becomes 1-1-2 with two separated counter outlets.  In
        # possession FWD1 runs beyond while FWD2 comes shorter as a connector.
        if we_have_ball:
            if position_label == "DEF":
                target_attack_x, target_y = _clamp(ball_attack_x - 18, -38, -14), 0.0
            elif position_label == "MID":
                target_attack_x = _clamp(ball_attack_x - 4, -8, 24)
                target_y = _clamp(ball_pos.get("y", 0) * 0.3, -10, 10)
            elif position_label == "FWD1":
                target_attack_x, target_y = _clamp(ball_attack_x + 14, 26, 43), -10.0
            else:
                target_attack_x, target_y = _clamp(ball_attack_x + 8, 20, 40), 10.0
        else:
            if position_label == "DEF":
                target_attack_x = _clamp(ball_attack_x - 10, -40, -24)
                target_y = _clamp(ball_pos.get("y", 0) * 0.3, -14, 14)
            elif position_label == "MID":
                target_attack_x = _clamp(ball_attack_x - 12, -22, 2)
                target_y = _clamp(ball_pos.get("y", 0) * 0.4, -12, 12)
            elif position_label == "FWD1":
                target_attack_x, target_y = _clamp(ball_attack_x + 8, 4, 18), -12.0
            else:
                target_attack_x, target_y = _clamp(ball_attack_x + 4, 0, 14), 12.0

        emergency_recovery = (
            immediate_danger
            and opponent_has_ball
            and position_label in ("DEF", "MID")
        )
        return _cmd("MOVE_TO", player_id, team_id, {
            "target_x": attack_sign * target_attack_x,
            "target_y": target_y,
            "sprint": bool(
                (we_have_ball and stamina_tier == "FRESH")
                or emergency_recovery
            ),
        })

    return decide
