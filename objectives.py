# objectives.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

# ---------------------------------------------------------------------------
# Mission objectives (extensible)
# ---------------------------------------------------------------------------

DEFAULT_MISSION_DICT: Dict[str, Any] = {"type": "kill_all_enemies"}

def _clamp_int(v: Any, default: int, lo: int, hi: int) -> int:
    try:
        n = int(v)
    except Exception:
        return default
    return max(lo, min(hi, n))

class MissionObjective:
    """Base class for mission objectives.

    Contract:
      - `is_completed(...)` returns True when the player has achieved the objective.
      - `status_text(...)` returns short lines suitable for UI display.
      - `to_dict()` serializes to map JSON.

    This module is intentionally standalone so adding new objectives later
    doesn't require changing unrelated gameplay code.
    """

    type_id: str = "base"

    def to_dict(self) -> Dict[str, Any]:
        return {"type": self.type_id}

    def is_completed(self, *, current_time_s: float, players: Sequence[Any], player_id: int) -> bool:
        return False

    def status_text(self, *, current_time_s: float, players: Sequence[Any], player_id: int) -> list[str]:
        return ["Objective: (unknown)"]

    @staticmethod
    def from_dict(data: Optional[Dict[str, Any]]) -> "MissionObjective":
        if not isinstance(data, dict):
            data = DEFAULT_MISSION_DICT

        t = str(data.get("type", "kill_all_enemies")).lower()

        if t in ("kill_all", "kill_all_enemies", "default"):
            # Optional: allow specifying enemy ids; keep backwards compatible
            enemy_ids = data.get("enemy_ids")
            if isinstance(enemy_ids, list) and enemy_ids:
                try:
                    enemy_ids_int = [int(x) for x in enemy_ids]
                except Exception:
                    enemy_ids_int = [2]
            else:
                enemy_ids_int = [2]
            return KillAllEnemiesObjective(enemy_player_ids=enemy_ids_int)

        if t in ("survive", "survive_time", "survive_x_time"):
            seconds = _clamp_int(data.get("seconds", data.get("time", 300)), default=300, lo=5, hi=60 * 60 * 3)
            return SurviveTimeObjective(seconds=seconds)

        # Unknown -> default
        return KillAllEnemiesObjective(enemy_player_ids=[2])

@dataclass
class KillAllEnemiesObjective(MissionObjective):
    type_id: str = "kill_all_enemies"
    enemy_player_ids: list[int] = None

    def __post_init__(self) -> None:
        if not self.enemy_player_ids:
            self.enemy_player_ids = [2]

    def to_dict(self) -> Dict[str, Any]:
        return {"type": self.type_id, "enemy_ids": list(self.enemy_player_ids)}

    def is_completed(self, *, current_time_s: float, players: Sequence[Any], player_id: int) -> bool:
        # Completed when all enemy players have no units left (units include buildings).
        for p in players:
            pid = getattr(p, "player_id", None)
            if pid in self.enemy_player_ids:
                if getattr(p, "units", None):
                    if len(p.units) > 0:
                        return False
        return True

    def status_text(self, *, current_time_s: float, players: Sequence[Any], player_id: int) -> list[str]:
        return ["Objective:", "Kill all enemies"]

@dataclass
class SurviveTimeObjective(MissionObjective):
    type_id: str = "survive_time"
    seconds: int = 300  # duration

    def to_dict(self) -> Dict[str, Any]:
        return {"type": self.type_id, "seconds": int(self.seconds)}

    def is_completed(self, *, current_time_s: float, players: Sequence[Any], player_id: int) -> bool:
        return float(current_time_s) >= float(self.seconds)

    def status_text(self, *, current_time_s: float, players: Sequence[Any], player_id: int) -> list[str]:
        remaining = max(0, int(round(self.seconds - float(current_time_s))))
        mm = remaining // 60
        ss = remaining % 60
        return ["Objective:", f"Survive {self.seconds//60}m {self.seconds%60:02d}s", f"Remaining: {mm}:{ss:02d}"]
