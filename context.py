"""
context.py
Mutable shared game state (kept here to avoid circular imports after refactor).
Main sets these fields after world creation.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Set

# Set by main / worldgen
players: List[Any] = []
all_units: Set[Any] = set()
grass_tiles: List[Any] = []
river_tiles: List[Any] = []
spatial_grid: Any = None
waypoint_graph: Any = None

# Time (ms) updated each frame by main
current_time: int = 0

# Effects / animations
highlight_times: Dict[int, int] = {}
attack_animations: Dict[int, Dict[str, Any]] = {}

# Production / building
production_queues: Dict[int, Any] = {}
building_animations: Dict[Any, Dict[str, Any]] = {}

# Orders / misc
move_order_times: Dict[int, int] = {}
