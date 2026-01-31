# worldgen.py
from __future__ import annotations

import json
import math
import random
from typing import Any, List, Set, Tuple

import pygame
from perlin_noise import PerlinNoise
from pygame import Vector2

from constants import *
import context
import units as units_mod
from tiles import GrassTile, Dirt, River, Foundation, Mountain
from world_objects import Bridge, Road, Wall, MiscPassable
from units import Tree, Building
from player import Player, PlayerAI
from pathfinding import SpatialGrid, WaypointGraph


DEFAULT_TREE_VARIANT = "tree6"
# ---------------------------
# Editor map loader
# ---------------------------
def load_editor_map(path: str) -> Tuple[
    List[List[Any]],
    Set[Any],
    Set[Tuple[int, int]],
    List[Any],
]:
    """
    Load map created by map_editor.py (tiles + units).
    Returns:
        grass_tiles (2D list of tile objects),
        all_units (set of unit objects),
        river_tiles (set of (row,col) where River tiles exist)
    Save format expected:
      {
        "rows": ...,
        "cols": ...,
        "tile_size": ...,
        "tiles": [[ "GrassTile" | "Dirt" | "River" | "Bridge", ...], ...],
        "units": { "r,c": {"type": "Axeman", "player": 1}, ... }
      }
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    tiles_data = data.get("tiles")
    if not tiles_data:
        raise ValueError("Editor map missing 'tiles'")

    # Tiles
    grass_tiles: List[List[Any]] = []
    river_tiles: Set[Tuple[int, int]] = set()

    for r in range(GRASS_ROWS):
        row_tiles = []
        for c in range(GRASS_COLS):
            cell = tiles_data[r][c]

            # Backward compatible (string) + new dict format
            if isinstance(cell, dict):
                tname = cell.get("type", "GrassTile")
                variant = cell.get("variant")
            else:
                tname = cell
                variant = None

            x = c * TILE_SIZE
            y = r * TILE_SIZE

            if tname == "GrassTile":
                tile = GrassTile(x, y)
            elif tname == "Dirt":
                tile = Dirt(x, y)
            elif tname == "River":
                tile = River(x, y, variant=variant)
                river_tiles.add((r, c))
            elif tname == "Foundation":
                tile = Foundation(x, y)
            elif tname == "Mountain":
                tile = Mountain(x, y)
            else:
                tile = GrassTile(x, y)
            row_tiles.append(tile)
        grass_tiles.append(row_tiles)

    # Units
    units_raw = data.get("units", {})
    all_units: Set[Any] = set()

    if isinstance(units_raw, dict):
        for key, info in units_raw.items():
            if not isinstance(key, str) or "," not in key or not isinstance(info, dict):
                continue
            try:
                rs, cs = key.split(",")
                r = int(rs)
                c = int(cs)
            except Exception:
                continue
            if not (0 <= r < GRASS_ROWS and 0 <= c < GRASS_COLS):
                continue

            unit_type = str(info.get("type", ""))
            try:
                player_id = int(info.get("player", 0))
            except Exception:
                player_id = 0

            cls = getattr(units_mod, unit_type, None)
            if cls is None:
                continue

            # Place at tile center
            x = c * TILE_SIZE + TILE_HALF
            y = r * TILE_SIZE + TILE_HALF

            # Choose a player color consistent with init_game_world() players list
            if player_id == 0:
                pcol = WHITE_GRAY
            elif player_id == 1:
                pcol = BLUE
            else:
                pcol = PURPLE

            # Instantiate with best-effort signature matching your units.py conventions
            try:
                if issubclass(cls, Building):
                    unit = cls(x, y, player_id, pcol)
                elif cls is Tree:
                    # Editor may store "variant": "tree0"..."tree6"
                    variant = info.get("variant", DEFAULT_TREE_VARIANT)
                    if not isinstance(variant, str) or not variant.startswith("tree"):
                        variant = DEFAULT_TREE_VARIANT

                    # Tree(x, y, size, color, player_id, player_color, variant=...)
                    unit = Tree(x, y, TILE_SIZE, WHITE_GRAY, player_id, pcol, variant=variant)
                else:
                    # Units: (x, y, player_id, player_color) in your codebase
                    unit = cls(x, y, player_id, pcol)
            except Exception:
                # If some unit has a different signature, skip it rather than crashing load
                continue

            all_units.add(unit)

    # World objects (overlay layer: drawn above tiles, below units)
    world_objects: List[Any] = []
    objects_raw = data.get("objects", {})

    if isinstance(objects_raw, dict):
        for key, info in objects_raw.items():
            if not isinstance(key, str) or "," not in key or not isinstance(info, dict):
                continue
            try:
                rs, cs = key.split(",")
                r = int(rs)
                c = int(cs)
            except Exception:
                continue
            if not (0 <= r < GRASS_ROWS and 0 <= c < GRASS_COLS):
                continue

            obj_type = str(info.get("type", ""))
            x = c * TILE_SIZE
            y = r * TILE_SIZE

            if obj_type == "Bridge":
                variant = info.get("variant")
                passable = bool(info.get("passable", True))
                try:
                    world_objects.append(Bridge(x, y, variant=variant, passable=passable))
                except TypeError:
                    # if your Bridge ctor is still Bridge(x,y) only, fall back safely
                    world_objects.append(Bridge(x, y))
            elif obj_type == "Road":
                variant = info.get("variant")
                passable = bool(info.get("passable", True))
                try:
                    world_objects.append(Road(x, y, variant=variant, passable=passable))
                except TypeError:
                    world_objects.append(Road(x, y))

            elif obj_type == "Wall":
                variant = info.get("variant")
                # ignore passable from save: walls are always blocked
                try:
                    world_objects.append(Wall(x, y, variant=variant))
                except TypeError:
                    world_objects.append(Wall(x, y))

            elif obj_type == "MiscPassable":
                variant = info.get("variant")
                passable = bool(info.get("passable", True))
                try:
                    world_objects.append(MiscPassable(x, y, variant=variant))
                except TypeError:
                    world_objects.append(MiscPassable(x, y))

    return grass_tiles, all_units, river_tiles, world_objects


# ---------------------------
# River generation (unchanged)
# ---------------------------
def generate_river(grass_tiles, tile_size, rows, cols, num_bridges=2):
    """
    Generate a 3-tile-wide continuous river from bottom-left to top-right, staying within 20 tiles
    of the straight line, ensuring it reaches (cols-1, 0), with 3-tile-wide bridges that extend at
    least one tile over grass or dirt on both ends, sharing x or y values. Place up to num_bridges,
    ensuring at least one by increasing bridge length, preferring shorter bridges.
    Args:
        grass_tiles: 2D list of tile objects
        tile_size: Size of each tile (20 pixels)
        rows: Number of rows (60)
        cols: Number of columns (60)
        num_bridges: Desired number of bridges to place (default 2)
    Returns:
        Set of (row, col) tuples representing river tiles for obstacle tracking
    """
    print(f"Starting river generation with {num_bridges} bridges requested...")
    river_tiles = set()
    bridge_locations = []  # Store bridge locations

    # Define exact start and end points
    start_x = 0
    start_y = rows - 1  # Bottom-left (col=0, row=59)
    end_x = cols - 1
    end_y = 0  # Top-right (col=59, row=0)

    # Explicitly set start and end tiles as River
    grass_tiles[rows - 1][0] = River(0, (rows - 1) * tile_size)
    river_tiles.add((rows - 1, 0))
    grass_tiles[0][cols - 1] = River((cols - 1) * tile_size, 0)
    river_tiles.add((0, cols - 1))

    # Generate river using Perlin noise
    noise = PerlinNoise(octaves=1, seed=random.randint(0, 1000000))  # Smoother path
    river_width = 3  # River width in tiles
    river_path = [(int(start_x), int(start_y))]  # Start point

    # Calculate steps based on diagonal distance
    diagonal_distance = math.sqrt((end_x - start_x) ** 2 + (end_y - start_y) ** 2)
    steps = int(diagonal_distance * 3)  # Sufficient sampling
    x_step = (end_x - start_x) / steps
    y_step = (end_y - start_y) / steps
    current_x = start_x
    current_y = start_y

    # Generate initial path
    for step in range(steps):
        # Reduce noise influence near the end to ensure reaching (cols-1, 0)
        noise_scale = min(cols, rows) / 75 * (1 - step / steps)  # Reduced noise
        noise_value = noise([current_x / cols, current_y / rows])
        current_x += x_step + noise_value * noise_scale
        current_y += y_step + noise_value * noise_scale
        current_x = max(0, min(cols - 1, current_x))
        current_y = max(0, min(rows - 1, current_y))
        tile_x = int(current_x)
        tile_y = int(current_y)
        if 0 <= tile_y < rows and 0 <= tile_x < cols:
            if (tile_x, tile_y) not in river_path:
                river_path.append((tile_x, tile_y))

    # Ensure end point is included
    if (int(end_x), int(end_y)) not in river_path:
        river_path.append((int(end_x), int(end_y)))

    # Constrain river path to within 20 tiles of the straight line
    max_deviation = 20
    constrained_path = [river_path[0]]  # Keep start point
    for x, y in river_path[1:-1]:  # Skip start and end
        # Parametric line: x = t * (cols-1), y = (1-t) * (rows-1), t in [0,1]
        t = ((x - start_x) * (end_x - start_x) + (y - start_y) * (end_y - start_y)) / (diagonal_distance ** 2)
        t = max(0, min(1, t))  # Clamp t to [0,1]
        closest_x = start_x + t * (end_x - start_x)
        closest_y = start_y + t * (end_y - start_y)
        # Compute perpendicular distance
        dist = math.sqrt((x - closest_x) ** 2 + (y - closest_y) ** 2)
        if dist > max_deviation:
            # Project point to within 20 tiles
            factor = max_deviation / dist
            new_x = closest_x + factor * (x - closest_x)
            new_y = closest_y + factor * (y - closest_y)
            tile_x = int(max(0, min(cols - 1, new_x)))
            tile_y = int(max(0, min(rows - 1, new_y)))
            print(f"Adjusted point ({x}, {y}) to ({tile_x}, {tile_y}), dist was {dist:.2f}")
        else:
            tile_x = x
            tile_y = y
        if (tile_x, tile_y) not in constrained_path:
            constrained_path.append((tile_x, tile_y))
    constrained_path.append((int(end_x), int(end_y)))  # Ensure end point

    # Interpolate to ensure continuous river path
    interpolated_path = []
    for i in range(len(constrained_path) - 1):
        x1, y1 = constrained_path[i]
        x2, y2 = constrained_path[i + 1]
        interpolated_path.append((x1, y1))
        # Use Bresenham's line algorithm for integer grid points
        dx = x2 - x1
        dy = y2 - y1
        steps = max(abs(dx), abs(dy))
        if steps > 1:
            x_step = dx / steps
            y_step = dy / steps
            for t in range(1, int(steps)):
                tile_x = int(x1 + t * x_step)
                tile_y = int(y1 + t * y_step)
                if 0 <= tile_x < cols and 0 <= tile_y < rows and (tile_x, tile_y) not in interpolated_path:
                    interpolated_path.append((tile_x, tile_y))
                    print(f"Interpolated point ({tile_x}, {tile_y}) between ({x1}, {y1}) and ({x2}, {y2})")
    interpolated_path.append(constrained_path[-1])  # Ensure end point

    # Place River tiles along the interpolated path
    for x, y in interpolated_path:
        for dy in range(-river_width // 2, river_width // 2 + 1):
            for dx in range(-river_width // 2, river_width // 2 + 1):
                tile_x = x + dx
                tile_y = y + dy
                if 0 <= tile_x < cols and 0 <= tile_y < rows:  # Strict bounds check
                    river_tiles.add((tile_y, tile_x))
                else:
                    print(f"Skipped out-of-bounds River tile at ({tile_x}, {tile_y})")

    # Ensure 3-tile-wide region around end point
    for dy in range(-river_width // 2, river_width // 2 + 1):
        for dx in range(-river_width // 2, river_width // 2 + 1):
            tile_x = end_x + dx
            tile_y = end_y + dy
            if 0 <= tile_x < cols and 0 <= tile_y < rows:
                river_tiles.add((tile_y, tile_x))

    # Update grass_tiles with River tiles
    for row, col in river_tiles:
        grass_tiles[row][col] = River(col * tile_size, row * tile_size, variant="river5")

    # Find valid bridge spans (3 tiles wide, at least 3 River tiles, land on both ends)
    valid_spans = []
    bridge_min_length = 5  # Start with minimum length
    bridge_max_length = 10  # Maximum length to try
    min_bridge_distance = 5  # Minimum distance between bridge centers

    # Try increasing bridge lengths until at least one span is selected
    selected_spans = []
    for length in range(bridge_min_length, bridge_max_length + 1):
        print(f"Trying bridge length {length}...")
        valid_spans = []  # Reset spans for each length
        for row in range(1, rows - 1):  # Avoid edges for 3-row width
            for col in range(cols):
                if isinstance(grass_tiles[row][col], River) or isinstance(grass_tiles[row][col], Mountain):
                    # Check horizontal spans (3 rows wide)
                    if col + length - 1 < cols:  # Ensure within bounds
                        valid = True
                        for dr in [-1, 0, 1]:
                            if not (0 <= row + dr < rows):
                                valid = False
                                break
                            start_tile = grass_tiles[row + dr][col - 1] if col > 0 else None
                            end_tile = grass_tiles[row + dr][col + length - 1]
                            if not (start_tile and isinstance(start_tile, (GrassTile, Dirt)) and
                                    isinstance(end_tile, (GrassTile, Dirt))):
                                valid = False
                                break
                        if valid:
                            river_count = sum(1 for i in range(col, col + length - 1)
                                              if isinstance(grass_tiles[row][i], River))
                            if river_count >= 3:
                                span_tiles = [(col + i, row + dr) for dr in [-1, 0, 1] for i in range(-1, length)
                                             if 0 <= row + dr < rows and 0 <= col + i < cols]
                                span_center = (col + (length - 1) // 2, row)
                                valid_spans.append({
                                    "center": span_center,
                                    "tiles": span_tiles,
                                    "direction": "horizontal",
                                    "score": abs(col + (length - 1) // 2 - cols / 2) + abs(row - rows / 2),
                                    "length": length
                                })
        for col in range(1, cols - 1):  # Avoid edges for 3-column width
            for row in range(rows):
                if isinstance(grass_tiles[row][col], River):
                    # Check vertical spans (3 columns wide)
                    if row + length - 1 < rows:
                        valid = True
                        for dc in [-1, 0, 1]:
                            if not (0 <= col + dc < cols):
                                valid = False
                                break
                            start_tile = grass_tiles[row - 1][col + dc] if row > 0 else None
                            end_tile = grass_tiles[row + length - 1][col + dc]
                            if not (start_tile and isinstance(start_tile, (GrassTile, Dirt)) and
                                    isinstance(end_tile, (GrassTile, Dirt))):
                                valid = False
                                break
                        if valid:
                            river_count = sum(1 for i in range(row, row + length - 1)
                                              if isinstance(grass_tiles[i][col], River))
                            if river_count >= 3:
                                span_tiles = [(col + dc, row + i) for dc in [-1, 0, 1] for i in range(-1, length)
                                             if 0 <= col + dc < cols and 0 <= row + i < rows]
                                span_center = (col, row + (length - 1) // 2)
                                valid_spans.append({
                                    "center": span_center,
                                    "tiles": span_tiles,
                                    "direction": "vertical",
                                    "score": abs(col - cols / 2) + abs(row + (length - 1) // 2 - rows / 2),
                                    "length": length
                                })

        selected_spans = []
        if valid_spans:
            valid_spans.sort(key=lambda x: (x["length"], x["score"]))
            selected_spans.append(valid_spans[0])
            for span in valid_spans[1:]:
                if len(selected_spans) >= num_bridges:
                    break
                valid = True
                for chosen in selected_spans:
                    dist = min(abs(span["center"][0] - chosen["center"][0]),
                               abs(span["center"][1] - chosen["center"][1]))
                    if dist < min_bridge_distance:
                        valid = False
                        break
                if valid:
                    selected_spans.append(span)

        print(f"For length {length}: Found {len(valid_spans)} valid spans, selected {len(selected_spans)}")
        if selected_spans:
            break

    # Place Bridge tiles
    for span in selected_spans:
        span_tiles = span["tiles"]
        bridge_locations.append(span["center"])
        for tile_x, tile_y in span_tiles:
            if 0 <= tile_y < rows and 0 <= tile_x < cols:
                grass_tiles[tile_y][tile_x] = Bridge(tile_x * tile_size, tile_y * tile_size)

    # Re-apply River tiles to avoid overwrites
    for row, col in river_tiles:
        if not isinstance(grass_tiles[row][col], Bridge):
            grass_tiles[row][col] = River(col * tile_size, row * tile_size, variant="river5")

    print(f"Selected bridge locations (centers): {bridge_locations}")
    return river_tiles


# ---------------------------
# Main world initializer
# ---------------------------
def init_game_world(random_world: bool = False):
    """
    Initializes the game world.
    If random=False, loads maps/editor_map.json created by map_editor.py.
    If random=True, uses existing procedural generation (unchanged).
    Returns:
        grass_tiles, needs_regrowth, river_tiles, players, all_units, spatial_grid, waypoint_graph, player2_ai
    """
    needs_regrowth: Set[Tuple[int, int]] = set()

    # Create players (same as before)
    players = [
        Player(0, WHITE_GRAY, 0, 0),
        Player(1, BLUE, 0, 0),
        Player(2, PURPLE, 800, 900),
    ]

    # Reset production queues for these players
    context.production_queues.clear()
    for p in players:
        context.production_queues[p.player_id] = []

    if not random_world:
        # ---- Load editor map ----
        grass_tiles, loaded_units, river_tiles, world_objects  = load_editor_map("maps/test_map.json")
        context.world_objects = world_objects  # easiest way to make it visible to main.py

        # Assign units to players (and rebuild all_units from player lists)
        for u in loaded_units:
            pid = getattr(u, "player_id", 0)
            if 0 <= pid < len(players):
                players[pid].units.append(u)

        all_units: Set[Any] = set()
        for p in players:
            all_units.update(p.units)

        # --- rebuild player state after loading ---
        for p in players:
            # recompute barns list (only completed barns if you use alpha==255 in-game)
            p.barns = [u for u in p.units if u.__class__.__name__ == "Barn" and getattr(u, "alpha", 255) == 255]

            # reset cow-in-barn tracking (or ensure keys exist)
            if not hasattr(p, "cow_in_barn") or p.cow_in_barn is None:
                p.cow_in_barn = {}
            for b in p.barns:
                p.cow_in_barn.setdefault(b, None)

            # recompute counts/limits bookkeeping
            p.unit_count = sum(1 for u in p.units if (u.__class__.__name__ != "Tree" and not isinstance(u, Building)) ) # or whatever your rule is
            p.building_count = sum(1 for u in p.units if isinstance(u, Building) and getattr(u, "alpha", 255) == 255)

    else:
        # ---- Existing random generation (as before) ----
        grass_tiles = [[GrassTile(col * TILE_SIZE, row * TILE_SIZE)
                        for col in range(GRASS_COLS)] for row in range(GRASS_ROWS)]

        river_tiles = generate_river(grass_tiles, TILE_SIZE, GRASS_ROWS, GRASS_COLS)

        # Combine all units created by players
        all_units: Set[Any] = set()
        for player in players:
            all_units.update(player.units)

    # Place Dirt tiles under buildings (works for both random + loaded)
    for unit in all_units:
        if isinstance(unit, Building):
            buildings_cols = range(
                int((unit.pos.x - unit.size / 2) // TILE_SIZE),
                int(math.ceil((unit.pos.x + unit.size / 2) / TILE_SIZE))
            )
            buildings_rows = range(
                int((unit.pos.y - unit.size / 2) // TILE_SIZE),
                int(math.ceil((unit.pos.y + unit.size / 2) / TILE_SIZE))
            )
            for row in buildings_rows:
                for col in buildings_cols:
                    if 0 <= row < GRASS_ROWS and 0 <= col < GRASS_COLS:
                        grass_tiles[row][col] = Dirt(col * TILE_SIZE, row * TILE_SIZE)

    # Random-only tree spawning (keep original behavior)
    if random_world:
        def is_tile_occupied(row, col, all_units_local):
            if not (0 <= row < GRASS_ROWS and 0 <= col < GRASS_COLS):
                return True
            if isinstance(grass_tiles[row][col], River) or isinstance(grass_tiles[row][col], Mountain):
                return True
            for unit in all_units_local:
                if isinstance(unit, (Building, Tree)):
                    unit_row = int(unit.pos.y // TILE_SIZE)
                    unit_col = int(unit.pos.x // TILE_SIZE)
                    size = unit.size // TILE_SIZE
                    half_size = size // 2
                    for r in range(unit_row - half_size, unit_row + half_size + 1):
                        for c in range(unit_col - half_size, unit_col + half_size + 1):
                            if r == row and c == col:
                                return True
            return False

        # Create tree objects
        tree_objects = []
        total_tiles = GRASS_ROWS * GRASS_COLS
        target_tree_count = int(total_tiles * 0.3)
        eligible_tiles = []
        for row in range(GRASS_ROWS):
            for col in range(GRASS_COLS):
                tile = grass_tiles[row][col]
                if isinstance(tile, GrassTile) and not isinstance(tile, Dirt) and not is_tile_occupied(row, col, all_units):
                    eligible_tiles.append((row, col))

        while len(tree_objects) < target_tree_count and eligible_tiles:
            patch_size = random.randint(0, 3)
            center_row, center_col = random.choice(eligible_tiles)
            patch = []
            for dr in range(-(patch_size - (patch_size // 2)), patch_size):
                for dc in range(-(patch_size - (patch_size // 2)), patch_size):
                    r, c = center_row + dr, center_col + dc
                    if (0 <= r < GRASS_ROWS and 0 <= c < GRASS_COLS and
                        isinstance(grass_tiles[r][c], GrassTile) and
                        not isinstance(grass_tiles[r][c], Dirt) and
                        not is_tile_occupied(r, c, all_units) and
                        (r, c) in eligible_tiles):
                        patch.append((r, c))
            if patch:
                for r, c in patch:
                    tree = Tree(c * TILE_SIZE + TILE_HALF, r * TILE_SIZE + TILE_HALF,
                                TILE_SIZE, WHITE_GRAY, 0, WHITE_GRAY)
                    tree_objects.append(tree)
                    eligible_tiles.remove((r, c))
            if (center_row, center_col) in eligible_tiles:
                eligible_tiles.remove((center_row, center_col))

        players[0].units.extend(tree_objects)
        all_units.update(tree_objects)

    # Initialize spatial grid
    spatial_grid = SpatialGrid(GRID_CELL_SIZE, MAP_WIDTH, MAP_HEIGHT)
    for unit in all_units:
        spatial_grid.add_unit(unit)

    # Initialize WaypointGraph
    waypoint_graph = WaypointGraph(grass_tiles, all_units, TILE_SIZE, MAP_WIDTH, MAP_HEIGHT, spatial_grid)

    # Initialize AI for Player 2
    player2_ai = PlayerAI(players[2], grass_tiles, spatial_grid, all_units, context.production_queues)

    # Export into shared context for modules that rely on it
    context.players = players
    context.all_units = all_units
    context.grass_tiles = grass_tiles
    context.river_tiles = river_tiles
    context.spatial_grid = spatial_grid
    context.waypoint_graph = waypoint_graph

    return grass_tiles, needs_regrowth, river_tiles, players, all_units, spatial_grid, waypoint_graph, player2_ai
