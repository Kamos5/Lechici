"""Unit progression (XP + levels).

This module is intentionally small and self-contained so you can fine-tune
XP thresholds and per-level stat scaling later without touching combat code.

UPDATED: Custom XP thresholds:
- regular at 20 XP
- veteran at +30 XP (50 total)
- hero at +50 XP (100 total)
"""

from __future__ import annotations

from dataclasses import dataclass


LEVELS: tuple[str, ...] = ("novice", "regular", "veteran", "hero")

# XP thresholds (cumulative totals)
XP_THRESHOLDS: tuple[int, ...] = (0, 20, 50, 100)


def level_index_from_xp(xp: int) -> int:
    """Return level index based on cumulative thresholds."""
    if xp < XP_THRESHOLDS[1]:
        return 0  # novice
    if xp < XP_THRESHOLDS[2]:
        return 1  # regular
    if xp < XP_THRESHOLDS[3]:
        return 2  # veteran
    return 3  # hero


def level_name_from_index(idx: int) -> str:
    idx = max(0, min(len(LEVELS) - 1, int(idx)))
    return LEVELS[idx]


def _round_int(x: float) -> int:
    return int(x + 0.5)


@dataclass
class ProgressionTuning:
    """Central place to tweak scaling later."""

    hp_per_level_pct: float = 0.10
    atk_per_level_pct: float = 0.10


TUNING = ProgressionTuning()


def ensure_base_stats(unit) -> None:
    if getattr(unit, "_progression_base_initialized", False):
        return
    unit.base_max_hp = int(getattr(unit, "max_hp", 0) or 0)
    unit.base_attack_damage = int(getattr(unit, "attack_damage", 0) or 0)
    unit._progression_base_initialized = True


def apply_level_stats(unit) -> None:
    ensure_base_stats(unit)
    lvl = int(getattr(unit, "level", 0) or 0)

    old_max = int(getattr(unit, "max_hp", 0) or 0)
    base_hp = int(getattr(unit, "base_max_hp", old_max) or old_max)
    base_atk = int(getattr(unit, "base_attack_damage", int(getattr(unit, "attack_damage", 0) or 0)) or 0)

    hp_mult = 1.0 + (TUNING.hp_per_level_pct * lvl)
    atk_mult = 1.0 + (TUNING.atk_per_level_pct * lvl)

    new_max = _round_int(base_hp * hp_mult)
    new_atk = _round_int(base_atk * atk_mult)

    unit.max_hp = max(1, int(new_max))
    unit.attack_damage = max(0, int(new_atk))

    if hasattr(unit, "hp"):
        delta = unit.max_hp - old_max
        unit.hp = min(unit.max_hp, float(getattr(unit, "hp", unit.max_hp)) + float(delta))


def add_xp(unit, amount: int) -> bool:
    if amount <= 0:
        return False

    if not hasattr(unit, "xp"):
        unit.xp = 0
    if not hasattr(unit, "level"):
        unit.level = 0

    ensure_base_stats(unit)

    old_level = int(getattr(unit, "level", 0) or 0)
    unit.xp = int(getattr(unit, "xp", 0) or 0) + int(amount)

    new_level = level_index_from_xp(unit.xp)
    if new_level != old_level:
        unit.level = new_level
        apply_level_stats(unit)
        return True
    return False


def award_combat_xp(attacker, target, *, damage: float, target_hp_before: float) -> None:
    try:
        from units import Tree, Building
    except Exception:
        Tree = ()
        Building = ()

    if attacker is None or target is None:
        return
    if damage <= 0:
        return
    if isinstance(target, Tree):
        return

    if getattr(attacker, "player_id", None) == getattr(target, "player_id", None):
        return
    if getattr(target, "player_id", 0) == 0:
        return

    killed = (target_hp_before > 0) and (target_hp_before - damage <= 0)

    if isinstance(target, Building):
        add_xp(attacker, 40 if killed else 1)
    else:
        add_xp(attacker, 10 if killed else 1)
