
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import pygame

# This module is intentionally self-contained and "dummy-friendly":
# - It only needs pygame and basic game objects (players / units).
# - It stores snapshots so it remains valid even after units are removed.


@dataclass
class UnitSnapshot:
    unit_class: str
    name: str
    level: int
    kills: int = 0
    # icon surface is optional; UI will generate/fallback if missing
    icon: Optional[pygame.Surface] = None


@dataclass
class TimelineSeries:
    t: List[float] = field(default_factory=list)
    milk_collected: List[float] = field(default_factory=list)
    wood_collected: List[float] = field(default_factory=list)
    units_total: List[int] = field(default_factory=list)
    buildings_total: List[int] = field(default_factory=list)
    units_trained: List[int] = field(default_factory=list)
    units_lost: List[int] = field(default_factory=list)
    units_killed: List[int] = field(default_factory=list)


class StatsTracker:
    """
    Tracks mission-end stats + time series for charts.

    Design goals:
    - minimal coupling to game loop
    - event hooks for 'unit trained', 'building completed', 'kill', 'loss'
    - resource collection tracked via positive deltas of player resources
    - stores unit snapshots so MVP survives unit removal
    """
    def __init__(self, player_id: int = 1, sample_every_s: float = 1.0):
        self.player_id = player_id
        self.sample_every_s = max(0.25, float(sample_every_s))
        self._next_sample_t = 0.0

        # totals
        self.produced_units = 0
        self.produced_buildings = 0
        self.collected_milk = 0.0
        self.collected_wood = 0.0
        self.units_trained = 0
        self.units_lost = 0
        self.units_killed = 0

        # per-unit kill tracking
        self._unit_kills: Dict[int, int] = {}
        self._unit_snap: Dict[int, UnitSnapshot] = {}

        # resource delta tracking
        self._last_milk: Optional[float] = None
        self._last_wood: Optional[float] = None

        # time series
        self.series = TimelineSeries()

        # baseline counts (to avoid counting starting army as "produced")
        self._baseline_units: Optional[int] = None
        self._baseline_buildings: Optional[int] = None

    def set_baseline_from_players(self, players: List[Any]) -> None:
        p = next((pp for pp in players if getattr(pp, 'player_id', None) == self.player_id), None)
        if not p:
            self._baseline_units = 0
            self._baseline_buildings = 0
            self._last_milk = 0.0
            self._last_wood = 0.0
            return

        units = getattr(p, 'units', []) or []
        # buildings in this project are also Units with alpha == 255
        buildings = [u for u in units if getattr(u, '__class__', None) and u.__class__.__name__ in (
            'Barn','TownCenter','Barracks','ShamansHut','WarriorsLodge','KnightsEstate','Wall','Bridge','Road'
        ) or getattr(u, 'is_building', False)]
        # Better heuristic: treat as building if has 'production_time' and not 'speed'
        buildings2 = [u for u in units if hasattr(u, 'production_time') and not hasattr(u, 'base_speed')]
        buildings = buildings2 if buildings2 else buildings

        self._baseline_units = len([u for u in units if not _is_building(u) and getattr(u, 'player_id', None) == self.player_id])
        self._baseline_buildings = len([u for u in units if _is_building(u) and getattr(u, 'player_id', None) == self.player_id and getattr(u, 'alpha', 255) == 255])

        self._last_milk = float(getattr(p, 'milk', 0.0) or 0.0)
        self._last_wood = float(getattr(p, 'wood', 0.0) or 0.0)

    def on_unit_trained(self, unit: Any, player_id: int) -> None:
        if player_id != self.player_id:
            return
        if _is_building(unit):
            return
        self.produced_units += 1
        self.units_trained += 1

        uid = id(unit)
        self._ensure_unit_snapshot(unit)
        # keep series monotonic
        self._unit_kills.setdefault(uid, 0)

    def on_building_completed(self, building: Any, player_id: int) -> None:
        if player_id != self.player_id:
            return
        self.produced_buildings += 1

    def on_unit_lost(self, unit: Any) -> None:
        if getattr(unit, 'player_id', None) == self.player_id and not _is_neutral(unit):
            if not _is_building(unit):
                self.units_lost += 1
            else:
                # buildings lost are not explicitly requested, but could be added later
                pass

    def on_kill(self, attacker: Any, victim: Any) -> None:
        if getattr(attacker, 'player_id', None) != self.player_id:
            return
        if _is_neutral(victim):
            return
        self.units_killed += 1

        uid = id(attacker)
        self._ensure_unit_snapshot(attacker)
        self._unit_kills[uid] = self._unit_kills.get(uid, 0) + 1
        # mirror into snapshot for rendering
        self._unit_snap[uid].kills = self._unit_kills[uid]

    def update_resources(self, players: List[Any]) -> None:
        p = next((pp for pp in players if getattr(pp, 'player_id', None) == self.player_id), None)
        if not p:
            return
        milk = float(getattr(p, 'milk', 0.0) or 0.0)
        wood = float(getattr(p, 'wood', 0.0) or 0.0)
        if self._last_milk is None:
            self._last_milk = milk
        if self._last_wood is None:
            self._last_wood = wood

        dm = milk - self._last_milk
        dw = wood - self._last_wood
        # Only count positive deltas as "collected"
        if dm > 0:
            self.collected_milk += dm
        if dw > 0:
            self.collected_wood += dw

        self._last_milk = milk
        self._last_wood = wood

    def maybe_sample(self, t: float, players: List[Any]) -> None:
        if t < self._next_sample_t:
            return
        # sample next
        self._next_sample_t = t + self.sample_every_s

        self.update_resources(players)

        p = next((pp for pp in players if getattr(pp, 'player_id', None) == self.player_id), None)
        units_total = 0
        buildings_total = 0
        if p:
            units = getattr(p, 'units', []) or []
            units_total = len([u for u in units if not _is_building(u) and getattr(u, 'player_id', None) == self.player_id])
            buildings_total = len([u for u in units if _is_building(u) and getattr(u, 'player_id', None) == self.player_id and getattr(u, 'alpha', 255) == 255])

        # normalize totals by baseline, so charts start at 0 for produced totals
        if self._baseline_units is None:
            self._baseline_units = units_total
        if self._baseline_buildings is None:
            self._baseline_buildings = buildings_total

        self.series.t.append(float(t))
        self.series.milk_collected.append(float(self.collected_milk))
        self.series.wood_collected.append(float(self.collected_wood))
        self.series.units_total.append(int(max(0, units_total - (self._baseline_units or 0))))
        self.series.buildings_total.append(int(max(0, buildings_total - (self._baseline_buildings or 0))))
        self.series.units_trained.append(int(self.units_trained))
        self.series.units_lost.append(int(self.units_lost))
        self.series.units_killed.append(int(self.units_killed))

    def get_mvp(self) -> Optional[UnitSnapshot]:
        if not self._unit_snap:
            return None
        # by kills desc, then level desc
        best = None
        for snap in self._unit_snap.values():
            if best is None:
                best = snap
                continue
            if snap.kills > best.kills:
                best = snap
            elif snap.kills == best.kills and snap.level > best.level:
                best = snap
        return best

    def _ensure_unit_snapshot(self, unit: Any) -> None:
        uid = id(unit)
        if uid in self._unit_snap:
            # refresh level/name if changed
            self._unit_snap[uid].level = int(getattr(unit, 'level', self._unit_snap[uid].level) or 0)
            nm = getattr(unit, 'name', None)
            if nm:
                self._unit_snap[uid].name = str(nm)
            return

        snap = UnitSnapshot(
            unit_class=getattr(unit, '__class__', type(unit)).__name__,
            name=str(getattr(unit, 'name', None) or getattr(unit, '__class__', type(unit)).__name__),
            level=int(getattr(unit, 'level', 0) or 0),
            kills=0,
            icon=None,
        )
        # Try to resolve a miniature icon from Unit._unit_icons if present.
        try:
            from units import Unit  # local project import
            icon = Unit._unit_icons.get(snap.unit_class)
            if icon is not None:
                snap.icon = icon
        except Exception:
            pass
        self._unit_snap[uid] = snap


def _is_building(u: Any) -> bool:
    # In this codebase, buildings are Unit subclasses that generally have 'production_time'
    # and alpha construction, and don't have base_speed.
    if u is None:
        return False
    name = getattr(u, '__class__', type(u)).__name__
    if name in {'Building','Barn','TownCenter','Barracks','ShamansHut','WarriorsLodge','KnightsEstate','Wall','Bridge','Road'}:
        return True
    if hasattr(u, 'production_time') and not hasattr(u, 'base_speed'):
        return True
    return False


def _is_neutral(u: Any) -> bool:
    # Player 0 is neutral in this project (trees, map objects).
    return getattr(u, 'player_id', None) == 0


# ---------------- Multi-player wrapper ----------------

class MultiStats:
    """Track endgame stats for multiple non-Gaia players.

    The game previously used a single StatsTracker (player 1 only). This wrapper keeps
    the same call sites (stats.on_kill, stats.on_unit_lost, stats.maybe_sample, etc.)
    but routes each event to the correct per-player tracker.
    """

    def __init__(self, player_ids: List[int], sample_every_s: float = 1.0):
        self.trackers: Dict[int, StatsTracker] = {
            int(pid): StatsTracker(player_id=int(pid), sample_every_s=sample_every_s)
            for pid in (player_ids or [])
            if int(pid) != 0
        }

    def set_baseline_from_players(self, players: List[Any]) -> None:
        for tr in self.trackers.values():
            tr.set_baseline_from_players(players)

    def maybe_sample(self, t: float, players: List[Any]) -> None:
        for tr in self.trackers.values():
            tr.maybe_sample(t, players)

    def on_unit_trained(self, unit: Any, player_id: int) -> None:
        tr = self.trackers.get(int(player_id))
        if tr:
            tr.on_unit_trained(unit, player_id)

    def on_building_completed(self, building: Any, player_id: int) -> None:
        tr = self.trackers.get(int(player_id))
        if tr:
            tr.on_building_completed(building, player_id)

    def on_unit_lost(self, unit: Any) -> None:
        pid = int(getattr(unit, 'player_id', 0) or 0)
        tr = self.trackers.get(pid)
        if tr:
            tr.on_unit_lost(unit)

    def on_kill(self, attacker: Any, victim: Any) -> None:
        pid = int(getattr(attacker, 'player_id', 0) or 0)
        tr = self.trackers.get(pid)
        if tr:
            tr.on_kill(attacker, victim)

    # UI convenience helpers
    def get_mvp(self) -> Optional[UnitSnapshot]:
        """MVP for player 1 (human) by convention."""
        tr = self.trackers.get(1)
        return tr.get_mvp() if tr else None

    def get_tracker(self, player_id: int) -> Optional['StatsTracker']:
        return self.trackers.get(int(player_id))

    def iter_player_ids(self) -> List[int]:
        return sorted(self.trackers.keys())
