import pygame
import sys
import random
import math
from pygame.math import Vector2
from heapq import heappush, heappop
import time
# Add game state enum for managing screens
from enum import Enum
from perlin_noise import PerlinNoise

import grid_actions
# 👇 ADD THIS
from constants import *
import context
from effects import update_effects, draw_effects
from utils import is_tile_occupied, find_valid_spawn_tiles
from worldgen import init_game_world
from world_objects import Bridge, Road
from tiles import GrassTile, Dirt, River
from units import Unit, Tree, Building, Barn, TownCenter, Barracks, KnightsEstate, WarriorsLodge, Axeman, Knight, Bear, Archer, Cow, ShamansHut, Wall
from player import Player, PlayerAI
from pathfinding import SpatialGrid, WaypointGraph
import ui
from camera import Camera


def run_game() -> int:
    """
    Runs the RTS game.
    Returns:
        1 if victory
        0 if defeat
    """
    # Initialize Pygame
    pygame.init()

    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Lechites")
    clock = pygame.time.Clock()

    ui_layout = ui.build_ui_layout()

    grid_buttons = ui_layout["grid_buttons"]
    quit_button = ui_layout["quit_button"]

    # Universal grid hotkeys (4x3): q w e r / a s d f / z x c v
    GRID_HOTKEYS = {
        pygame.K_q: (0, 0), pygame.K_w: (0, 1), pygame.K_e: (0, 2), pygame.K_r: (0, 3),
        pygame.K_a: (1, 0), pygame.K_s: (1, 1), pygame.K_d: (1, 2), pygame.K_f: (1, 3),
        pygame.K_z: (2, 0), pygame.K_x: (2, 1), pygame.K_c: (2, 2), pygame.K_v: (2, 3),
    }

    button_player0 = ui_layout["button_player0"]
    button_player1 = ui_layout["button_player1"]
    button_player2 = ui_layout["button_player2"]

    # Camera
    camera = Camera(view_width=VIEW_WIDTH, view_height=VIEW_HEIGHT, map_width=MAP_WIDTH, map_height=MAP_HEIGHT,
                    screen_width=SCREEN_WIDTH, screen_height=SCREEN_HEIGHT,
                    scroll_margin=SCROLL_MARGIN, scroll_speed=SCROLL_SPEED)

    # UI icons + sizes
    ICON_SIZE = ui_layout["icon_size"]
    ICON_MARGIN = ui_layout["icon_margin"]
    icons = ui.load_ui_icons()
    wood_icon = icons["wood"]
    milk_icon = icons["milk"]
    unit_icon = icons["unit"]
    building_icon = icons["building"]

    # Move order and highlight tracking
    move_order_times = {}
    highlight_times = {}
    attack_animations = []  # List to store attack animation data (start_pos, end_pos, start_time)

    # Production queue and animation tracking
    production_queues = {}  # Maps building to {'unit_type': Class, 'start_time': float, 'player_id': int}
    building_animations = {}  # Maps building to {'start_time': float, 'alpha': float} for transparency animation

    class GameState(Enum):
        RUNNING = 1
        DEFEAT = 2
        VICTORY = 3

    # Initialize game state
    game_state = GameState.RUNNING

    # Pause state (Pause/Break toggles)
    paused = False
    paused_by_player_id = None
    resume_button_rect = None

    # Simulation time that does NOT advance while paused
    game_time = 0.0


    # Fonts
    fonts = ui.create_fonts()
    font = fonts["font"]
    small_font = fonts["small_font"]
    button_font = fonts["button_font"]
    end_button_font = fonts["end_button_font"]

    # --- World init (refactored) --- (refactored) ---
    grass_tiles, needs_regrowth, river_tiles, players, all_units, spatial_grid, waypoint_graph, player2_ai = init_game_world()

    # Wire up shared mutable state for refactored modules
    context.players = players
    context.all_units = all_units
    context.grass_tiles = grass_tiles
    context.river_tiles = river_tiles
    context.spatial_grid = spatial_grid
    context.waypoint_graph = waypoint_graph

    # Share these animation/queue dicts defined above with modules
    context.highlight_times = highlight_times
    context.attack_animations = attack_animations
    context.production_queues = production_queues
    context.building_animations = building_animations
    context.move_order_times = move_order_times

    context.effects = []

    selection_end = None
    selecting = False
    current_player = None

    # --- Mouse selection state (click vs box-select) ---
    pending_click = False
    pending_click_unit = None
    mouse_down_screen = None
    mouse_down_world = None
    drag_threshold_px = 6  # pixels

    # --- NEW: minimap drag state ---
    minimap_dragging = False

    # --- NEW: right-mouse camera drag (pan) state ---
    right_mouse_down = False
    right_mouse_dragging = False
    right_mouse_last = None  # Vector2 (screen coords)
    right_drag_threshold_px = 4  # pixels

    # UI rectangles/fonts already created via ui.build_ui_layout() and ui.create_fonts()

    # --- SPLIT POINT (End of Part 1) --- (End of Part 1) ---
    # The game loop and remaining code will continue in Part 2.

    # --- Part 2: Continuation from Part 1 ---

    # Game loop
    running = True
    tile_surface = pygame.Surface((VIEW_WIDTH, VIEW_HEIGHT), pygame.SRCALPHA)
    placing_building = False
    building_to_place = None
    building_pos = None

    # Building footprint is class-defined (Building.SIZE_TILES == 3 by default).
    # Walls override it to 1.
    def _placement_size_for(cls) -> int:
        # Roads are tile-sized world objects
        if cls is Road:
            return TILE_SIZE

        size_tiles = int(getattr(cls, "SIZE_TILES", max(1, int(BUILDING_SIZE // TILE_SIZE))))
        return size_tiles * TILE_SIZE

    # Map (U, D, L, R) -> variant
    _WALL_VARIANT = {
        (0, 0, 0, 0): "wall6",
        (1, 0, 0, 0): "wall2",  # up
        (0, 1, 0, 0): "wall2",  # down
        (1, 1, 0, 0): "wall2",  # up+down

        (0, 0, 1, 0): "wall6",  # left
        (0, 0, 0, 1): "wall6",  # right
        (0, 0, 1, 1): "wall6",  # left+right

        (1, 0, 0, 1): "wall1",  # up+right
        (0, 1, 0, 1): "wall3",  # down+right
        (1, 1, 0, 1): "wall4",  # up+down+right

        (1, 0, 1, 0): "wall5",  # up+left
        (1, 0, 1, 1): "wall7",  # left+right+up

        (0, 1, 1, 0): "wall8",  # left+down
        (1, 1, 1, 0): "wall9",  # up+down+left
        (0, 1, 1, 1): "wall10",  # left+right+down

        (1, 1, 1, 1): "wall11",  # up+down+left+right
    }

    def _tile_of(u) -> tuple[int, int]:
        return (int(u.pos.x // TILE_SIZE), int(u.pos.y // TILE_SIZE))

    def _find_wall_at(all_units, tx: int, ty: int, player_id: int):
        # placement-time scan (simple + reliable)
        cx = tx * TILE_SIZE + TILE_HALF
        cy = ty * TILE_SIZE + TILE_HALF
        for u in all_units:
            if isinstance(u, Wall) and u.player_id == player_id:
                if int(u.pos.x) == int(cx) and int(u.pos.y) == int(cy):
                    return u
        return None

    def _compute_wall_variant(all_units, wall: Wall) -> str:
        tx, ty = _tile_of(wall)
        pid = wall.player_id

        up = 1 if _find_wall_at(all_units, tx, ty - 1, pid) else 0
        down = 1 if _find_wall_at(all_units, tx, ty + 1, pid) else 0
        left = 1 if _find_wall_at(all_units, tx - 1, ty, pid) else 0
        right = 1 if _find_wall_at(all_units, tx + 1, ty, pid) else 0

        return _WALL_VARIANT.get((up, down, left, right), "wall6")

    def _refresh_wall_and_neighbors(all_units, wall: Wall) -> None:
        """Recompute variant for wall and its 4-neighbors (same player)."""
        tx, ty = _tile_of(wall)
        pid = wall.player_id

        candidates = [wall]
        for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            w2 = _find_wall_at(all_units, tx + dx, ty + dy, pid)
            if w2:
                candidates.append(w2)

        for w in candidates:
            v = _compute_wall_variant(all_units, w)
            w.set_variant(v)

    # --- Road auto-connect (like Wall) ---
    # Map (U, D, L, R) -> road variant (road1..road15)
    _ROAD_VARIANT = {
        (0, 0, 0, 0): "road10",  # no adjacent
        (0, 0, 0, 1): "road1",  # right
        (0, 0, 1, 0): "road2",  # left
        (0, 1, 0, 0): "road3",  # down
        (1, 0, 0, 0): "road4",  # up

        (1, 0, 0, 1): "road5",  # up + left
        (1, 1, 0, 0): "road6",  # up + down
        (0, 1, 0, 1): "road7",  # right + down
        (1, 1, 0, 1): "road8",  # up + right + down
        (1, 0, 1, 0): "road9",  # up + right

        (0, 0, 1, 1): "road10",  # left + right
        (1, 0, 1, 1): "road11",  # left + up + right
        (0, 1, 1, 0): "road12",  # left + down
        (1, 1, 1, 0): "road13",  # up + down + left
        (0, 1, 1, 1): "road14",  # left + right + down
        (1, 1, 1, 1): "road15",  # all 4
    }

    def _tile_of_world_object(obj) -> tuple[int, int]:
        return (int(obj.pos.x // TILE_SIZE), int(obj.pos.y // TILE_SIZE))

    def _find_road_at(world_objects, tx: int, ty: int):
        # roads are stored in context.world_objects (ownership ignored)
        for o in world_objects:
            if isinstance(o, Road):
                ox, oy = _tile_of_world_object(o)
                if ox == tx and oy == ty:
                    return o
        return None


    def _compute_road_variant(world_objects, road: Road) -> str:
        tx, ty = _tile_of_world_object(road)

        up = 1 if _find_road_at(world_objects, tx, ty - 1) else 0
        down = 1 if _find_road_at(world_objects, tx, ty + 1) else 0
        left = 1 if _find_road_at(world_objects, tx - 1, ty) else 0
        right = 1 if _find_road_at(world_objects, tx + 1, ty) else 0

        return _ROAD_VARIANT.get((up, down, left, right), "road10")

    def _refresh_road_and_neighbors(world_objects, road: Road) -> None:
        """Recompute variant for road and its 4-neighbors (ownership ignored)."""
        tx, ty = _tile_of_world_object(road)

        candidates = [road]
        for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            r2 = _find_road_at(world_objects, tx + dx, ty + dy)
            if r2:
                candidates.append(r2)

        for r in candidates:
            v = _compute_road_variant(world_objects, r)
            r.set_variant(v)

    def _building_footprint_tiles(b) -> set[tuple[int, int]]:
        size_tiles = int(getattr(b, 'SIZE_TILES', max(1, int(b.size // TILE_SIZE))))
        cx = int(b.pos.x // TILE_SIZE)
        cy = int(b.pos.y // TILE_SIZE)
        half = size_tiles // 2
        out = set()
        for dy in range(-half, half + 1):
            for dx in range(-half, half + 1):
                out.add((cx + dx, cy + dy))
        return out

    def _road_can_place(player, world_objects, all_units, tx: int, ty: int) -> bool:
        """Road must connect to this player's TownCenter (directly adjacent) or via an existing connected road chain."""
        if not player:
            return False

        pid = player.player_id
        town_centers = [u for u in all_units if isinstance(u, TownCenter) and u.player_id == pid and getattr(u, 'alpha', 255) == 255]
        if not town_centers:
            return False

        # Adjacent to any town center footprint => ok
        for tc in town_centers:
            fp = _building_footprint_tiles(tc)
            for fx, fy in fp:
                if abs(fx - tx) + abs(fy - ty) == 1:
                    return True

        # Build a dict of all roads by tile (ownership ignored)
        roads = {}
        for o in world_objects:
            if isinstance(o, Road):
                rx, ry = _tile_of_world_object(o)
                roads[(rx, ry)] = o

        if not roads:
            return False

        # Seed BFS with all roads adjacent to TC footprints
        seeds = []
        for tc in town_centers:
            fp = _building_footprint_tiles(tc)
            for fx, fy in fp:
                for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                    k = (fx + dx, fy + dy)
                    if k in roads:
                        seeds.append(k)

        if not seeds:
            return False

        visited = set(seeds)
        stack = list(seeds)
        while stack:
            x, y = stack.pop()
            for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                nk = (x + dx, y + dy)
                if nk in roads and nk not in visited:
                    visited.add(nk)
                    stack.append(nk)

        # New road must be adjacent to some connected road
        for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            if (tx + dx, ty + dy) in visited:
                return True

        return False


    def _connected_road_tiles(player, world_objects, all_units) -> set[tuple[int, int]]:
        """Return set of road tiles connected (via road graph) to this player's TownCenter footprint.

        Ownership of roads is ignored: any road may be part of the connected graph.
        """
        if not player:
            return set()

        pid = player.player_id
        town_centers = [u for u in all_units if isinstance(u, TownCenter) and u.player_id == pid and getattr(u, 'alpha', 255) == 255]
        if not town_centers:
            return set()

        # all roads by tile (ignore player_id)
        roads = set()
        for o in world_objects:
            if isinstance(o, Road):
                rx, ry = _tile_of_world_object(o)
                roads.add((rx, ry))

        if not roads:
            return set()

        # Seed BFS with any road adjacent to TC footprint
        seeds = []
        for tc in town_centers:
            fp = _building_footprint_tiles(tc)
            for fx, fy in fp:
                for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                    k = (fx + dx, fy + dy)
                    if k in roads:
                        seeds.append(k)

        if not seeds:
            return set()

        visited = set(seeds)
        stack = list(seeds)
        while stack:
            x, y = stack.pop()
            for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                nk = (x + dx, y + dy)
                if nk in roads and nk not in visited:
                    visited.add(nk)
                    stack.append(nk)

        return visited


    def _building_can_place_near_connected_road(player, world_objects, all_units, center_tx: int, center_ty: int, size_tiles: int) -> bool:
        """Buildings must touch (Manhattan-adjacent) a connected road tile."""
        connected = _connected_road_tiles(player, world_objects, all_units)
        if not connected:
            return False

        half = size_tiles // 2
        for dy in range(-half, half + 1):
            for dx in range(-half, half + 1):
                fx, fy = (center_tx + dx, center_ty + dy)
                for ox, oy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                    if (fx + ox, fy + oy) in connected:
                        return True
        return False

    current_player = players[1]

    # --- Multi-selection grouping helpers (UI/stat-screen style) ---
    def _normalize_selection_for_player(p):
        """If units+buildings are selected, keep only units selected (buildings get deselected)."""
        if not p:
            return
        selected_units = [u for u in p.units if getattr(u, "selected", False) and not isinstance(u, Building)]
        if selected_units:
            for u in p.units:
                if getattr(u, "selected", False) and isinstance(u, Building):
                    u.selected = False

    def _compute_selection_groups(p):
        """Return list of (type_name, [entities]) for current player's selection, filtered (units over buildings)."""
        if not p:
            return []
        selected = [u for u in p.units if getattr(u, "selected", False)]
        if not selected:
            return []
        # If any non-building selected -> ignore buildings entirely
        if any(not isinstance(u, Building) for u in selected):
            selected = [u for u in selected if not isinstance(u, Building)]
        else:
            selected = [u for u in selected if isinstance(u, Building)]
        groups = {}
        for u in selected:
            k = u.__class__.__name__
            groups.setdefault(k, []).append(u)
        # stable deterministic ordering by type name
        return [(k, groups[k]) for k in sorted(groups.keys())]

    def _update_player_selection_groups(p):
        """Store groups + keep active index valid."""
        if not p:
            return
        groups = _compute_selection_groups(p)
        setattr(p, "selection_groups", groups)
        idx = int(getattr(p, "active_selection_group_index", 0) or 0)
        if not groups:
            idx = 0
        else:
            idx = max(0, min(idx, len(groups) - 1))
        setattr(p, "active_selection_group_index", idx)

    def _cycle_selection_group(p):
        if not p:
            return
        groups = _compute_selection_groups(p)
        if len(groups) <= 1:
            setattr(p, "selection_groups", groups)
            setattr(p, "active_selection_group_index", 0)
            return
        idx = int(getattr(p, "active_selection_group_index", 0) or 0)
        idx = (idx + 1) % len(groups)
        setattr(p, "selection_groups", groups)
        setattr(p, "active_selection_group_index", idx)


    # --- Control groups (Ctrl+1..0 to save, 1..0 to recall) ---
    def _ensure_control_groups(p):
        if not p:
            return
        if not isinstance(getattr(p, "control_groups", None), dict):
            setattr(p, "control_groups", {})

    def _prune_control_groups(p):
        """Remove dead/invalid units from saved groups (and delete empty groups)."""
        if not p:
            return
        _ensure_control_groups(p)
        cg = getattr(p, "control_groups")
        alive = set(all_units)
        to_del = []
        for idx, ents in list(cg.items()):
            kept = [u for u in (ents or []) if u in alive and getattr(u, "player_id", None) == p.player_id]
            if kept:
                cg[idx] = kept
            else:
                to_del.append(idx)
        for idx in to_del:
            cg.pop(idx, None)

    def _save_control_group(p, idx: int):
        """Save current selection into group idx (1..9,0). Units can be in only one group."""
        if not p:
            return
        _ensure_control_groups(p)
        cg = getattr(p, "control_groups")

        selected = [u for u in p.units if getattr(u, "selected", False)]
        # Empty selection clears the group
        if not selected:
            cg.pop(idx, None)
            return

        # Remove these units from all other groups (units can be only in one group)
        for k, ents in list(cg.items()):
            cg[k] = [u for u in (ents or []) if u not in selected]
            if not cg[k]:
                cg.pop(k, None)

        # Save (preserve order, unique)
        seen = set()
        saved = []
        for u in selected:
            if u not in seen:
                seen.add(u)
                saved.append(u)
        cg[idx] = saved

    def _recall_control_group(p, idx: int, *, center: bool = False):
        if not p:
            return
        _ensure_control_groups(p)
        _prune_control_groups(p)
        cg = getattr(p, "control_groups")
        ents = cg.get(idx)
        if not ents:
            # Clear active marker if group doesn't exist
            setattr(p, "active_control_group_idx", None)
            return

        # Deselect everything first
        for u in all_units:
            u.selected = False

        for u in ents:
            u.selected = True

        _normalize_selection_for_player(p)
        _update_player_selection_groups(p)

        # Track which control group is currently active (used by UI)
        setattr(p, "active_control_group_idx", idx)

        if center and ents:
            first = ents[0]
            # Center camera on the first unit/building in the group,
            # clamped to world bounds (so borders block as close as possible).
            target_x = float(first.pos.x) - (VIEW_WIDTH / 2)
            target_y = float(first.pos.y) - (VIEW_HEIGHT / 2)

            camera.x = max(0, min(target_x, MAP_WIDTH - VIEW_WIDTH))
            camera.y = max(0, min(target_y, MAP_HEIGHT - VIEW_HEIGHT))


    def _handle_right_click_command(screen_pos: tuple[int, int], current_time: float) -> None:
        """Existing right-click behavior (move/attack/rally/cancel placement)."""
        nonlocal placing_building, building_to_place, building_pos

        mouse_pos = Vector2(screen_pos)

        if placing_building:
            print(f"Building placement canceled for {building_to_place.__name__}")
            placing_building = False
            building_to_place = None
            building_pos = None
        elif VIEW_MARGIN_LEFT <= mouse_pos.x <= VIEW_BOUNDS_X and VIEW_MARGIN_TOP <= mouse_pos.y <= VIEW_BOUNDS_Y:
            click_pos = camera.screen_to_world(mouse_pos, view_margin_left=VIEW_MARGIN_LEFT, view_margin_top=VIEW_MARGIN_TOP)
            tile_x = int(click_pos.x // TILE_SIZE)
            tile_y = int(click_pos.y // TILE_SIZE)
            snapped_pos = (tile_x * TILE_SIZE + TILE_HALF, tile_y * TILE_SIZE + TILE_HALF)
            clicked_unit = None
            for unit in all_units:
                if unit.is_clicked(Vector2(mouse_pos.x - VIEW_MARGIN_LEFT, mouse_pos.y - VIEW_MARGIN_TOP), camera.x, camera.y):
                    clicked_unit = unit
                    break
            selected_buildings = []
            selected_non_buildings = 0
            if current_player:
                for unit in current_player.units:
                    if unit.selected:
                        if isinstance(unit, Building):
                            selected_buildings.append(unit)
                        else:
                            selected_non_buildings += 1

            # If ONLY buildings are selected and we right-click empty ground -> set rally point for ALL selected buildings.
            if selected_buildings and selected_non_buildings == 0 and not clicked_unit:
                for b in selected_buildings:
                    b.rally_point = Vector2(snapped_pos)
                print(f"Set rally point for {len(selected_buildings)} building(s) to {snapped_pos}")
                move_order_times[snapped_pos] = current_time
            else:
                for unit in all_units:
                    if unit.selected and not isinstance(unit, (Building, Tree)):
                        clicked_tree = clicked_unit if isinstance(clicked_unit, Tree) and clicked_unit.player_id == 0 else None
                        if clicked_tree and isinstance(unit, Axeman) and unit.special == 0 and not unit.depositing:
                            unit.target = clicked_tree.pos
                            unit.autonomous_target = False

                        elif clicked_unit and not isinstance(clicked_unit, Tree) and (clicked_unit.player_id != unit.player_id or clicked_unit.player_id == 0):
                            unit.target = clicked_unit
                            unit.autonomous_target = False

                        elif (clicked_unit
                              and isinstance(clicked_unit, TownCenter)
                              and clicked_unit.player_id == unit.player_id
                              and isinstance(unit, Axeman)
                              and unit.special > 0):
                            # Manual deposit order: behave exactly like the automatic deposit
                            unit.target = Vector2(clicked_unit.pos)  # IMPORTANT: use TC center, not raw click_pos
                            unit.depositing = True
                            unit.return_pos = None  # optional; prevents returning-to-old-tree logic
                            unit.autonomous_target = False
                            unit.path = waypoint_graph.get_path(unit.pos, unit.target, unit) if waypoint_graph else [unit.pos, unit.target]
                            unit.path_index = 0

                        else:
                            # Normal move order should cancel a deposit run
                            if isinstance(unit, Axeman):
                                unit.depositing = False

                            unit.target = Vector2(click_pos.x, click_pos.y)
                            unit.autonomous_target = False

    while running:

        dt = clock.get_time() / 1000  # Delta time for frame-rate independent updates
        if (not paused) and game_state == GameState.RUNNING:
            game_time += dt
        current_time = game_time
        context.current_time = current_time
        if (not paused) and game_state == GameState.RUNNING:
            update_effects(getattr(context, "effects", []), current_time, dt)

        # Check for Defeat or Victory conditions
        if (not paused) and game_state == GameState.RUNNING:
            player1 = next((p for p in players if p.player_id == 1), None)
            player2 = next((p for p in players if p.player_id == 2), None)

            # Defeat: Player 1 has no units or buildings
            if player1 and not player1.units:
                game_state = GameState.DEFEAT
                print("Player 1 has no units left. Showing Defeat screen.")

            # Victory: Player 2 has no units or buildings
            elif player2 and not player2.units:
                game_state = GameState.VICTORY
                print("Player 2 has no units left. Showing Victory screen.")

        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
                sys.exit()

            # Pause/Break toggles pause even while paused (only during active gameplay)
            if event.type == pygame.KEYDOWN:
                pause_keys = {pygame.K_PAUSE}
                if hasattr(pygame, "K_BREAK"):
                    pause_keys.add(pygame.K_BREAK)
                if event.key in pause_keys and game_state == GameState.RUNNING:
                    paused = not paused
                    if paused:
                        paused_by_player_id = (getattr(current_player, "player_id", None) or 1)
                    else:
                        paused_by_player_id = None
                        resume_button_rect = None
                    continue

                # While paused: ignore all other key input
                if paused:
                    continue

            # While paused: only allow clicking Resume
            if paused and event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if resume_button_rect and resume_button_rect.collidepoint(event.pos):
                    paused = False
                    paused_by_player_id = None
                    resume_button_rect = None
                continue
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if game_state in (GameState.DEFEAT, GameState.VICTORY):
                    # Check if Quit button is clicked
                    if quit_button.collidepoint(event.pos):
                        print("Quit button clicked. Exiting game.")
                        # pygame.quit()
                        # sys.exit()
                elif game_state == GameState.RUNNING:
                    if event.button == 1:  # Left click
                        mouse_pos = Vector2(event.pos)
                        clicked_something = False

                        # --- Control group bookmarks (above bottom stat panel) ---
                        if current_player:
                            visible_bookmarks = ui.get_visible_control_group_bookmarks(current_player, all_units)
                            clicked_idx = None
                            for idx, rect in visible_bookmarks.items():
                                if rect.collidepoint(event.pos):
                                    clicked_idx = idx
                                    break
                            if clicked_idx is not None:
                                # Double click centers camera on the first unit in the group
                                last = getattr(current_player, "_control_group_last_click", {})
                                last_t = float(last.get(clicked_idx, -9999.0))
                                is_double = (current_time - last_t) <= 0.35
                                _recall_control_group(current_player, clicked_idx, center=is_double)
                                last[clicked_idx] = float(current_time)
                                setattr(current_player, "_control_group_last_click", last)

                                # Consume click: don't start box-select / map selection
                                pending_click = False
                                pending_click_unit = None
                                mouse_down_screen = None
                                mouse_down_world = None
                                selecting = False
                                selection_start = None
                                selection_end = None
                                continue

                        # --- NEW: minimap click/drag begins here ---
                        mm_world = ui.minimap_screen_to_world(mouse_pos)
                        if mm_world is not None:
                            minimap_dragging = True
                            camera.center_on(mm_world)

                            # Cancel selection/drag-box state so minimap drag doesn't turn into box-select
                            pending_click = False
                            pending_click_unit = None
                            mouse_down_screen = None
                            mouse_down_world = None
                            selecting = False
                            selection_start = None
                            selection_end = None

                            continue  # consume this click entirely

                        click_pos = camera.screen_to_world(
                            mouse_pos,
                            view_margin_left=VIEW_MARGIN_LEFT,
                            view_margin_top=VIEW_MARGIN_TOP,
                        )
                        # Check LEFT grid for either unit commands (priority) or production
                        # Check LEFT grid for either unit commands (priority) or production
                        grid_button_clicked = False
                        cell = grid_actions.cell_from_mouse(grid_buttons, event.pos)
                        if cell and current_player:
                            r, c = cell
                            handled, placing_building, building_to_place = grid_actions.execute_grid_cell(
                                r, c,
                                current_player=current_player,
                                grid_buttons=grid_buttons,
                                production_queues=production_queues,
                                current_time=current_time,
                                placing_building=placing_building,
                                building_to_place=building_to_place,
                            )
                            grid_button_clicked = handled

                        # If the click was on the left grid, DO NOT treat it as a map click
                        if grid_button_clicked:
                            clicked_something = True

                        # --- Confirm building placement on LEFT CLICK in the world ---
                        if placing_building and (not grid_button_clicked) and VIEW_MARGIN_LEFT <= mouse_pos.x <= VIEW_BOUNDS_X and VIEW_MARGIN_TOP <= mouse_pos.y <= VIEW_BOUNDS_Y:
                            tile_x = int(click_pos.x // TILE_SIZE)
                            tile_y = int(click_pos.y // TILE_SIZE)
                            building_pos = Vector2(tile_x * TILE_SIZE + TILE_HALF, tile_y * TILE_SIZE + TILE_HALF)

                            building_size = _placement_size_for(building_to_place)
                            building_size_tiles = int(building_size // TILE_SIZE)
                            valid_placement = True
                            for row in range(tile_y - building_size_tiles // 2, tile_y + building_size_tiles // 2 + 1):
                                for col in range(tile_x - building_size_tiles // 2, tile_x + building_size_tiles // 2 + 1):
                                    if not (0 <= row < GRASS_ROWS and 0 <= col < GRASS_COLS) or is_tile_occupied(row, col, all_units, context.grass_tiles):
                                        valid_placement = False
                                        break
                                if not valid_placement:
                                    break

                            if valid_placement and building_to_place:
                                # NEW RULE: buildings (except Wall) must be placed next to a road connected to your TownCenter
                                if building_to_place is not Road and getattr(building_to_place, '__name__', '') != 'Wall':
                                    world_objects = getattr(context, 'world_objects', [])
                                    if not _building_can_place_near_connected_road(current_player, world_objects, all_units, tile_x, tile_y, building_size_tiles):
                                        print('Building must be placed next to a connected road.')
                                        valid_placement = False

                                # Special-case: Road is a tile-sized world object
                                if building_to_place is Road:
                                    world_objects = getattr(context, "world_objects", None)
                                    if world_objects is None:
                                        context.world_objects = []
                                        world_objects = context.world_objects

                                    # road placement uses TOP-LEFT coords
                                    # extra rule: must connect to player's TownCenter via a road chain

                                    if isinstance(context.grass_tiles[tile_y][tile_x], River):
                                        print("Cannot place Road on River.")
                                        valid_placement = False

                                    if not valid_placement:
                                        print("Invalid placement position: occupied or out of bounds")
                                    elif _find_road_at(world_objects, tile_x, tile_y):
                                        print("There is already a road here.")
                                    elif not _road_can_place(current_player, world_objects, all_units, tile_x, tile_y):
                                        print("Road must connect (directly or indirectly) to your TownCenter.")
                                    elif current_player.milk < Road.milk_cost or current_player.wood < Road.wood_cost:
                                        print("Not enough resources to place Road.")
                                    else:
                                        current_player.milk -= Road.milk_cost
                                        current_player.wood -= Road.wood_cost

                                        new_road = Road(tile_x * TILE_SIZE, tile_y * TILE_SIZE, player_id=current_player.player_id)
                                        world_objects.append(new_road)

                                        # auto-connect: update this road + adjacent roads (same player)
                                        _refresh_road_and_neighbors(world_objects, new_road)

                                        print(f"Placed Road at tile ({tile_x},{tile_y}) for Player {current_player.player_id}")

                                        # Reset placement state
                                        placing_building = False
                                        building_to_place = None
                                        building_pos = None
                                else:
                                    if not valid_placement:
                                        print('Invalid placement: must be next to a connected road (or clear space).')
                                    elif current_player.milk < building_to_place.milk_cost or current_player.wood < building_to_place.wood_cost:
                                        print('Not enough resources to place building.')
                                    else:

                                        current_player.milk -= building_to_place.milk_cost
                                        current_player.wood -= building_to_place.wood_cost

                                        # They fade in via alpha and also "build up" HP from 1 -> max_hp over production_time.
                                        new_building = building_to_place(building_pos.x, building_pos.y, current_player.player_id, current_player.color)
                                        # Start buildings under construction.
                                        # They fade in via alpha and also "build up" HP from 1 -> max_hp over production_time.
                                        new_building.alpha = 0
                                        new_building.hp = 1
                                        current_player.add_unit(new_building)
                                        all_units.add(new_building)
                                        spatial_grid.add_unit(new_building)

                                        # --- Wall auto-connect: update this wall + adjacent walls (same player) ---
                                        if isinstance(new_building, Wall):
                                            _refresh_wall_and_neighbors(all_units, new_building)

                                        # Paint area to dirt (same as your old placement code)
                                        # (Skip for tile-sized walls so they don't leave a dirt "footprint".)
                                        if getattr(building_to_place, "__name__", "") != "Wall":
                                            for row in range(tile_y - building_size_tiles // 2, tile_y + building_size_tiles // 2 + 1):
                                                for col in range(tile_x - building_size_tiles // 2, tile_x + building_size_tiles // 2 + 1):
                                                    if 0 <= row < GRASS_ROWS and 0 <= col < GRASS_COLS:
                                                        grass_tiles[row][col] = Dirt(col * TILE_SIZE, row * TILE_SIZE)

                                        highlight_times[new_building] = current_time
                                        building_animations[new_building] = {
                                            'start_time': current_time,
                                            'alpha': 0,
                                            'town_center': None,
                                            # used to make construction HP progression frame-rate independent
                                            'last_time': current_time,
                                        }

                                        print(f"Placed {building_to_place.__name__} at {building_pos} for Player {current_player.player_id}")

                                        # Reset placement state
                                        placing_building = False
                                        building_to_place = None
                                        building_pos = None
                            else:
                                print("Invalid placement position: occupied or out of bounds")

                            clicked_something = True  # consume click; don't select units

                        else:
                            # Otherwise continue with normal map selection / pending click
                            if not grid_button_clicked and VIEW_MARGIN_LEFT <= mouse_pos.x <= VIEW_BOUNDS_X and VIEW_MARGIN_TOP <= mouse_pos.y <= VIEW_BOUNDS_Y:
                                unit_clicked = None
                                for unit in all_units:
                                    if unit.is_clicked(Vector2(mouse_pos.x - VIEW_MARGIN_LEFT, mouse_pos.y - VIEW_MARGIN_TOP), camera.x, camera.y):
                                        unit_clicked = unit
                                        clicked_something = True
                                        break

                                # SHIFT+click keeps toggle behavior
                                if unit_clicked and (pygame.key.get_mods() & pygame.KMOD_SHIFT):
                                    unit_clicked.selected = not unit_clicked.selected
                                    pending_click = False
                                    pending_click_unit = None
                                    mouse_down_screen = None
                                    mouse_down_world = None
                                    selecting = False
                                    selection_start = None
                                    selection_end = None
                                else:
                                    # Defer single-click selection until mouse-up (keeps your drag-box behavior)
                                    if not placing_building:
                                        pending_click = True
                                        pending_click_unit = unit_clicked
                                        mouse_down_screen = Vector2(event.pos)
                                        mouse_down_world = click_pos
                                        selection_start = click_pos
                                        selection_end = click_pos
                                        selecting = False

                    elif event.button == 3:  # Right click (hold + drag pans camera)
                        right_mouse_down = True
                        right_mouse_dragging = False
                        right_mouse_last = Vector2(event.pos)

            elif event.type == pygame.KEYDOWN and game_state == GameState.RUNNING:
                # TAB cycles the active selection group (works for units and buildings)
                if event.key == pygame.K_TAB and current_player:
                    _cycle_selection_group(current_player)
                    _normalize_selection_for_player(current_player)
                    continue

                # Control groups: Ctrl+1..0 saves current selection; 1..0 recalls (double-tap centers camera)
                if current_player and event.key in (
                    pygame.K_0, pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4,
                    pygame.K_5, pygame.K_6, pygame.K_7, pygame.K_8, pygame.K_9,
                ):
                    key_to_idx = {
                        pygame.K_1: 1, pygame.K_2: 2, pygame.K_3: 3, pygame.K_4: 4,
                        pygame.K_5: 5, pygame.K_6: 6, pygame.K_7: 7, pygame.K_8: 8,
                        pygame.K_9: 9, pygame.K_0: 0,
                    }
                    idx = key_to_idx.get(event.key)
                    if idx is not None:
                        if pygame.key.get_mods() & pygame.KMOD_CTRL:
                            _save_control_group(current_player, idx)
                            # Saving a group also marks it as the "active" one for bookmark highlighting.
                            setattr(current_player, "active_control_group_idx", idx)
                        else:
                            # Double-tap digit -> center camera on first unit in that group.
                            last = getattr(current_player, "_control_group_last_keypress", {})
                            last_t = float(last.get(idx, -9999.0))
                            is_double = (current_time - last_t) <= 0.35
                            _recall_control_group(current_player, idx, center=is_double)
                            last[idx] = float(current_time)
                            setattr(current_player, "_control_group_last_keypress", last)
                        continue

                cell = grid_actions.cell_from_key(event.key)
                if cell and current_player:
                    r, c = cell
                    handled, placing_building, building_to_place = grid_actions.execute_grid_cell(
                        r, c,
                        current_player=current_player,
                        grid_buttons=grid_buttons,
                        production_queues=production_queues,
                        current_time=current_time,
                        placing_building=placing_building,
                        building_to_place=building_to_place,
                    )

            elif event.type == pygame.MOUSEBUTTONUP:
                if game_state == GameState.RUNNING and event.button == 3:
                    # End right-mouse drag-pan, or execute a normal right-click command if it was just a click.
                    if right_mouse_down:
                        if right_mouse_dragging:
                            right_mouse_down = False
                            right_mouse_dragging = False
                            right_mouse_last = None
                            continue
                        else:
                            right_mouse_down = False
                            right_mouse_dragging = False
                            right_mouse_last = None
                            _handle_right_click_command(event.pos, current_time)
                            continue

                elif game_state == GameState.RUNNING and event.button == 1:
                    # Finish box-select
                    # --- NEW: end minimap drag ---
                    if minimap_dragging:
                        minimap_dragging = False
                        continue

                    if selecting and current_player:
                        mouse_pos = Vector2(event.pos)
                        if VIEW_MARGIN_LEFT <= mouse_pos.x <= VIEW_BOUNDS_X and VIEW_MARGIN_TOP <= mouse_pos.y <= VIEW_BOUNDS_Y:
                            selection_end = camera.screen_to_world(mouse_pos, view_margin_left=VIEW_MARGIN_LEFT, view_margin_top=VIEW_MARGIN_TOP)
                            selecting = False
                            # Shift+box adds; otherwise replace selection
                            if not (pygame.key.get_mods() & pygame.KMOD_SHIFT):
                                for player in players:
                                    player.deselect_all_units()
                                _update_player_selection_groups(current_player)

                            x1, x2 = min(selection_start.x, selection_end.x), max(selection_start.x, selection_end.x)
                            y1, y2 = min(selection_start.y, selection_end.y), max(selection_start.y, selection_end.y)
                            for unit in current_player.units:
                                if x1 <= unit.pos.x <= x2 and y1 <= unit.pos.y <= y2:
                                    unit.selected = True

                            _normalize_selection_for_player(current_player)
                            _update_player_selection_groups(current_player)
                            selection_start = None
                            selection_end = None

                        pending_click = False
                        pending_click_unit = None
                        mouse_down_screen = None
                        mouse_down_world = None

                    # Resolve pending single-click (only if we never turned it into a drag)
                    elif pending_click and current_player:
                        unit_clicked = pending_click_unit
                        pending_click = False
                        pending_click_unit = None
                        mouse_down_screen = None
                        mouse_down_world = None

                        if unit_clicked:
                            for player in players:
                                player.deselect_all_units()
                            unit_clicked.selected = True
                            _normalize_selection_for_player(current_player)
                            _update_player_selection_groups(current_player)
                            if unit_clicked.player_id == current_player.player_id:
                                selected_barn = unit_clicked if isinstance(unit_clicked, Barn) and unit_clicked.alpha == 255 else None
                                selected_barracks = unit_clicked if isinstance(unit_clicked, Barracks) and unit_clicked.alpha == 255 else None
                                selected_town_center = unit_clicked if isinstance(unit_clicked, TownCenter) and unit_clicked.alpha == 255 else None
                                selected_shamans_hut = unit_clicked if isinstance(unit_clicked, ShamansHut) and unit_clicked.alpha == 255 else None
                            else:
                                selected_barn = None
                                selected_barracks = None
                                selected_town_center = None
                                selected_shamans_hut = None
                            placing_building = False
                            building_to_place = None
                            building_pos = None
                            print(f"Selected unit {unit_clicked.__class__.__name__} at {unit_clicked.pos} (Player {unit_clicked.player_id}) via map click")
                            selected_units = [u for u in current_player.units if u.selected]
                            for selected_unit in selected_units:
                                if isinstance(selected_unit, (Axeman, Archer, Knight, Bear)) and selected_unit != unit_clicked:
                                    selected_unit.target = unit_clicked
                                    selected_unit.path = waypoint_graph.get_path(selected_unit.pos, unit_clicked.pos, selected_unit)
                                    selected_unit.path_index = 0
                                    selected_unit.autonomous_target = False
                                    highlight_times[unit_clicked] = current_time
                                    print(f"Set target to {unit_clicked.__class__.__name__} at {unit_clicked.pos} for {selected_unit.__class__.__name__} at {selected_unit.pos}")
                        else:
                            # Clicked empty ground: just clear selection
                            if not placing_building:
                                for player in players:
                                    player.deselect_all_units()
                                _update_player_selection_groups(current_player)
            elif event.type == pygame.MOUSEMOTION:
                if game_state == GameState.RUNNING:
                    # --- NEW: minimap dragging ---
                    if minimap_dragging:
                        mouse_pos = Vector2(event.pos)
                        mm_world = ui.minimap_screen_to_world(mouse_pos)
                        if mm_world is not None:
                            camera.center_on(mm_world)
                        continue  # don't let minimap drag turn into box select / placement updates

                    # --- NEW: right-mouse drag pans camera ---
                    if right_mouse_down and right_mouse_last is not None:
                        now = Vector2(event.pos)
                        delta = now - right_mouse_last

                        # Decide if this is a drag (vs a click) once we move a little.
                        if (not right_mouse_dragging) and delta.length_squared() >= (right_drag_threshold_px * right_drag_threshold_px):
                            right_mouse_dragging = True

                        if right_mouse_dragging:
                            # Camera moves with mouse movement (drag right -> camera moves right).
                            camera.pan(-delta.x, -delta.y)
                            right_mouse_last = now
                            continue  # don't let camera drag interfere with selection/placement updates
                        else:
                            # Not yet a drag: keep updating last position so delta is measured from latest.
                            right_mouse_last = now

                    # If we have a pending click, turn it into a box-select once we move far enough.
                    if pending_click and not selecting and mouse_down_screen is not None:
                        delta = Vector2(event.pos) - mouse_down_screen
                        if delta.length_squared() >= (drag_threshold_px * drag_threshold_px):
                            # Start box selecting (even if the drag began over a unit/building)
                            selecting = True
                            pending_click = False
                            pending_click_unit = None
                            # Shift+drag-box adds to selection; normal drag-box replaces selection
                            if not (pygame.key.get_mods() & pygame.KMOD_SHIFT):
                                for player in players:
                                    player.deselect_all_units()
                                _update_player_selection_groups(current_player)
                            selection_start = mouse_down_world
                            mouse_pos = Vector2(event.pos)
                            if VIEW_MARGIN_LEFT <= mouse_pos.x <= VIEW_BOUNDS_X and VIEW_MARGIN_TOP <= mouse_pos.y <= VIEW_BOUNDS_Y:
                                selection_end = camera.screen_to_world(mouse_pos, view_margin_left=VIEW_MARGIN_LEFT, view_margin_top=VIEW_MARGIN_TOP)

                    if selecting:
                        mouse_pos = Vector2(event.pos)
                        if VIEW_MARGIN_LEFT <= mouse_pos.x <= VIEW_BOUNDS_X and VIEW_MARGIN_TOP <= mouse_pos.y <= VIEW_BOUNDS_Y:
                            selection_end = camera.screen_to_world(mouse_pos, view_margin_left=VIEW_MARGIN_LEFT, view_margin_top=VIEW_MARGIN_TOP)
                    if placing_building:
                        mouse_pos = Vector2(event.pos)
                        if VIEW_MARGIN_LEFT <= mouse_pos.x <= VIEW_BOUNDS_X and VIEW_MARGIN_TOP <= mouse_pos.y <= VIEW_BOUNDS_Y:
                            click_pos = camera.screen_to_world(mouse_pos, view_margin_left=VIEW_MARGIN_LEFT, view_margin_top=VIEW_MARGIN_TOP)
                            tile_x = int(click_pos.x // TILE_SIZE)
                            tile_y = int(click_pos.y // TILE_SIZE)
                            building_pos = Vector2(tile_x * TILE_SIZE + TILE_HALF, tile_y * TILE_SIZE + TILE_HALF)

        if game_state == GameState.RUNNING and (not paused):
            _normalize_selection_for_player(current_player)
            _update_player_selection_groups(current_player)
            # Update camera
            mouse_pos_screen = pygame.mouse.get_pos()
            if not minimap_dragging and not right_mouse_dragging:
                camera.update(mouse_pos_screen)

            # Check for building selection
            barn_selected = any(isinstance(unit, Barn) and unit.selected and unit.alpha == 255 for unit in (current_player.units if current_player else []))
            barracks_selected = any(isinstance(unit, Barracks) and unit.selected and unit.alpha == 255 for unit in (current_player.units if current_player else []))
            town_center_selected = any(isinstance(unit, TownCenter) and unit.selected and unit.alpha == 255 for unit in (current_player.units if current_player else []))
            town_shamans_hut = any(isinstance(unit, ShamansHut) and unit.selected and unit.alpha == 255 for unit in (current_player.units if current_player else []))

            # --- NEW: arrow-key camera scroll ---
            keys = pygame.key.get_pressed()

            dx = (keys[pygame.K_RIGHT] - keys[pygame.K_LEFT]) * CAMERA_KEY_SPEED
            dy = (keys[pygame.K_DOWN] - keys[pygame.K_UP]) * CAMERA_KEY_SPEED

            if (dx or dy) and not right_mouse_dragging:
                camera.x = max(0, min(camera.x + dx, MAP_WIDTH - VIEW_WIDTH))
                camera.y = max(0, min(camera.y + dy, MAP_HEIGHT - VIEW_HEIGHT))

            # Update production queues
            units_to_spawn = []
            for building, queue in list(production_queues.items()):
                if building not in all_units:
                    del production_queues[building]
                    continue
                production_time = queue['unit_type'].production_time
                if current_time - queue['start_time'] >= production_time:
                    player = next(p for p in players if p.player_id == queue['player_id'])
                    if not issubclass(queue['unit_type'], Building):
                        valid_spawn_tiles = find_valid_spawn_tiles(building, all_units, grass_tiles, TILE_SIZE, radius=2)
                        if valid_spawn_tiles:
                            spawn_pos = random.choice(valid_spawn_tiles)
                            new_unit = queue['unit_type'](spawn_pos.x, spawn_pos.y, player.player_id, player.color)
                            if building.rally_point:
                                new_unit.target = Vector2(building.rally_point)
                                print(f"Set rally point {new_unit.target} for {new_unit.__class__.__name__} spawned at {new_unit.pos}")
                            units_to_spawn.append((new_unit, player))
                            print(f"Queued {new_unit.__class__.__name__} to spawn at random tile {spawn_pos} for Player {player.player_id}")
                        else:
                            print(f"No valid spawn tiles found around {building.__class__.__name__} at {building.pos}, delaying production")
                            continue
                    del production_queues[building]
                    print(f"Production complete for {queue['unit_type'].__name__} at {building.pos} for Player {player.player_id}")

            # Spawn completed units
            for unit, player in units_to_spawn:
                player.add_unit(unit)
                all_units.add(unit)
                spatial_grid.add_unit(unit)
                player.barns = [u for u in player.units if isinstance(u, Barn) and u.alpha == 255]
                highlight_times[unit] = current_time
                print(f"Spawned {unit.__class__.__name__} at {unit.pos} for Player {player.player_id}")

            # Update building animations (construction)
            for building, anim in list(building_animations.items()):
                if building not in all_units:
                    del building_animations[building]
                    continue

                # HP should be visible and grow at a constant rate during construction.
                # If damaged during construction, damage reduces current hp and growth continues.
                if building.alpha < 255:
                    last_time = anim.get('last_time', anim.get('start_time', current_time))
                    dtt = max(0.0, current_time - last_time)
                    anim['last_time'] = current_time
                    prod_time = float(getattr(building, 'production_time', 1.0) or 1.0)
                    rate = (max(1.0, float(getattr(building, 'max_hp', 1))) - 1.0) / max(0.001, prod_time)
                    building.hp = min(float(getattr(building, 'max_hp', building.hp)), float(building.hp) + rate * dtt)

                elapsed = current_time - anim['start_time']
                if elapsed >= building.production_time:
                    building.alpha = 255
                    player = next(p for p in players if p.player_id == building.player_id)
                    if building not in player.units or building.alpha < 255:
                        player.building_count += 1
                    if isinstance(building, Barn):
                        player.barns = [u for u in player.units if isinstance(u, Barn) and u.alpha == 255]
                    del building_animations[building]
                else:
                    building.alpha = int(255 * (elapsed / building.production_time))

            # Update AI for Player 2
            player2_ai.update(current_time, waypoint_graph, spatial_grid, all_units, grass_tiles, production_queues)
            player2_ai.update_attack_units(current_time, waypoint_graph, spatial_grid, all_units, grass_tiles, production_queues)

            # Update grass regrowth
            for row, col in list(needs_regrowth):
                tile = grass_tiles[row][col]
                tile.regrow(REGROWTH_RATE)
                if tile.grass_level >= 1.0:
                    needs_regrowth.remove((row, col))

            # Update units
            axemen = [unit for unit in all_units if isinstance(unit, Axeman)]
            units_to_remove = set()
            for player in players:
                for unit in player.units:
                    unit._corrections = []
                    if isinstance(unit, Axeman):
                        unit.move(all_units, spatial_grid, waypoint_graph)
                    elif isinstance(unit, Cow):
                        unit.move(all_units, spatial_grid, waypoint_graph)
                    else:
                        unit.move(all_units, spatial_grid, waypoint_graph)
                    if unit.target and isinstance(unit.target, Unit) and not isinstance(unit.target, Tree) and unit.target in all_units:
                        unit.attack(unit.target, current_time)
                        if unit.target.hp <= 0:
                            units_to_remove.add(unit.target)
                    unit.resolve_collisions(all_units, spatial_grid)
                    if hasattr(unit, '_corrections') and unit._corrections:
                        unit.pos += sum(unit._corrections, Vector2(0, 0))
                    unit.keep_in_bounds()
                    spatial_grid.add_unit(unit)
                    if isinstance(unit, Cow):
                        unit.harvest_grass(grass_tiles, player.barns, player.cow_in_barn, player, spatial_grid)
                        if isinstance(grass_tiles[int(unit.pos.y // TILE_SIZE)][int(unit.pos.x // TILE_SIZE)], GrassTile):
                            needs_regrowth.add((int(unit.pos.y // TILE_SIZE), int(unit.pos.x // TILE_SIZE)))
                    elif isinstance(unit, Axeman):
                        unit.chop_tree([u for u in all_units if isinstance(u, Tree) and u.player_id == 0])
                for barn in player.barns:
                    if barn in player.cow_in_barn and player.cow_in_barn[barn]:
                        cow = player.cow_in_barn[barn]
                        if cow.is_in_barn(barn) and cow.special > 0:
                            over_limit = max(0, player.unit_count - player.unit_limit) if player.unit_limit is not None else 0
                            multiplier = max(0.0, 1.0 - (0.1 * over_limit))
                            milk_harvested = barn.harvest_rate / 60 * multiplier
                            cow.special = max(0, cow.special - barn.harvest_rate / 60)
                            player.milk = min(player.milk + milk_harvested, player.max_milk)
                            if cow.special <= 0:
                                player.cow_in_barn[barn] = None
                                cow.target = cow.return_pos
                                cow.returning = True
                                cow.return_pos = None
                                print(f"Cow at {cow.pos} fully drained in barn, targeting return_pos {cow.target}")
                        else:
                            player.cow_in_barn[barn] = None
                    else:
                        for unit in player.units:
                            if isinstance(unit, Cow) and unit.is_in_barn(barn):
                                player.cow_in_barn[barn] = unit
                                break

            # Projectile damage happens asynchronously (on-hit), so collect dead units globally too
            for u in list(all_units):
                if isinstance(u, Unit) and not isinstance(u, Tree) and u.hp <= 0:
                    units_to_remove.add(u)

            # Remove dead units
            for unit in units_to_remove:
                if unit in building_animations:
                    anim = building_animations[unit]
                    town_center = anim.get('town_center')
                    if town_center and town_center in production_queues and production_queues[town_center]['unit_type'] == unit.__class__:
                        del production_queues[town_center]
                        print(f"Canceled TownCenter production queue for {unit.__class__.__name__} due to destruction")
                    del building_animations[unit]
                for player in players:
                    if unit in player.units:
                        print(f"Removing unit {unit.__class__.__name__} at {unit.pos} for Player {unit.player_id}, building_count before: {player.building_count}")
                        player.remove_unit(unit)
                        print(f"After removing {unit.__class__.__name__}, building_count now: {player.building_count}")
                all_units.discard(unit)
                spatial_grid.remove_unit(unit)
                print(f"Unit {unit.__class__.__name__} at {unit.pos} destroyed")

            # Keep control groups clean (hide bookmarks when groups are empty)
            if current_player:
                _prune_control_groups(current_player)

        # Rendering
        screen.fill((0, 0, 0))  # Clear screen

        if game_state == GameState.RUNNING:
            # Draw tiles
            tile_surface.fill((0, 0, 0, 0))
            start_col = max(0, int(camera.x // TILE_SIZE))
            end_col = min(GRASS_COLS, int((camera.x + VIEW_WIDTH + TILE_SIZE) // TILE_SIZE))
            start_row = max(0, int(camera.y // TILE_SIZE))
            end_row = min(GRASS_ROWS, int((camera.y + VIEW_HEIGHT + TILE_SIZE) // TILE_SIZE))
            for row in range(start_row, end_row):
                for col in range(start_col, end_col):
                    grass_tiles[row][col].draw(tile_surface, camera.x, camera.y)
                    tile_pos = (col * TILE_SIZE + TILE_HALF, row * TILE_SIZE + TILE_HALF)
                    if tile_pos in move_order_times and current_time - move_order_times[tile_pos] <= 0.5:
                        x1 = tile_pos[0] - TILE_QUARTER // 2 - camera.x
                        y1 = tile_pos[1] - TILE_QUARTER // 2 - camera.y
                        x2 = tile_pos[0] + TILE_QUARTER // 2 - camera.x
                        y2 = tile_pos[1] + TILE_QUARTER // 2 - camera.y
                        pygame.draw.line(tile_surface, WHITE, (x1, y1), (x2, y2), 2)
                        pygame.draw.line(tile_surface, WHITE, (x2, y1), (x1, y2), 2)
            if current_player:
                selected_building = next((unit for unit in current_player.units if isinstance(unit, Building) and unit.selected), None)
                if selected_building and selected_building.rally_point:
                    rally_pos = (int(selected_building.rally_point.x), int(selected_building.rally_point.y))
                    x1 = rally_pos[0] - TILE_QUARTER // 2 - camera.x
                    y1 = rally_pos[1] - TILE_QUARTER // 2 - camera.y
                    x2 = rally_pos[0] + TILE_QUARTER // 2 - camera.x
                    y2 = rally_pos[1] + TILE_QUARTER // 2 - camera.y
                    pygame.draw.line(tile_surface, ORANGE, (x1, y1), (x2, y2), 2)
                    pygame.draw.line(tile_surface, ORANGE, (x2, y1), (x1, y2), 2)
            screen.blit(tile_surface, (VIEW_MARGIN_LEFT, VIEW_MARGIN_TOP))

            # Clean up move orders and highlights
            move_order_times = {k: v for k, v in move_order_times.items() if current_time - v <= 1}
            highlight_times = {k: v for k, v in highlight_times.items() if current_time - v <= 0.4}
            attack_animations[:] = [anim for anim in attack_animations if current_time - anim['start_time'] < 0.2]

            for obj in getattr(context, "world_objects", []):
                obj.draw(tile_surface, camera.x, camera.y)

            screen.blit(tile_surface, (VIEW_MARGIN_LEFT, VIEW_MARGIN_TOP))

            # Draw units (existing)
            for unit in all_units:
                if isinstance(unit, Building):
                    unit.draw(screen, camera.x, camera.y)
            for unit in all_units:
                if not isinstance(unit, Building):
                    if isinstance(unit, Tree):
                        unit.draw(screen, camera.x, camera.y, axemen)
                    else:
                        unit.draw(screen, camera.x, camera.y)

            # 👇 ADD THIS: draw arrows / effects
            draw_effects(
                getattr(context, "effects", []),
                screen,
                camera.x,
                camera.y,
                current_time,
            )

            # Draw attack animations
            for anim in attack_animations:
                start_x = anim['start_pos'].x - camera.x + VIEW_MARGIN_LEFT
                start_y = anim['start_pos'].y - camera.y + VIEW_MARGIN_TOP
                end_x = anim['end_pos'].x - camera.x + VIEW_MARGIN_LEFT
                end_y = anim['end_pos'].y - camera.y + VIEW_MARGIN_TOP
                pygame.draw.line(screen, anim['color'], (start_x, start_y), (end_x, end_y), 2)

            # Draw building/road preview during placement
            if placing_building and building_pos:
                building_size = _placement_size_for(building_to_place)
                tile_x = int(building_pos.x // TILE_SIZE)
                tile_y = int(building_pos.y // TILE_SIZE)

                # compute occupancy validity (buildings use footprint; road is 1 tile)
                valid_placement = True
                building_size_tiles = int(building_size // TILE_SIZE)
                for row in range(tile_y - building_size_tiles // 2, tile_y + building_size_tiles // 2 + 1):
                    for col in range(tile_x - building_size_tiles // 2, tile_x + building_size_tiles // 2 + 1):
                        if not (0 <= row < GRASS_ROWS and 0 <= col < GRASS_COLS) or is_tile_occupied(row, col, all_units, context.grass_tiles):
                            valid_placement = False
                            break
                    if not valid_placement:
                        break
                # NEW RULE: buildings (except Wall) must be placed next to a connected road
                if valid_placement and building_to_place is not Road and getattr(building_to_place, '__name__', '') != 'Wall':
                    world_objects = getattr(context, 'world_objects', [])
                    if not _building_can_place_near_connected_road(current_player, world_objects, all_units, tile_x, tile_y, building_size_tiles):
                        valid_placement = False


                if building_to_place is Road:
                    world_objects = getattr(context, "world_objects", [])
                    # additional rule: road must connect to TownCenter chain
                    if valid_placement and isinstance(context.grass_tiles[tile_y][tile_x], River):
                        valid_placement = False
                    if valid_placement and (_find_road_at(world_objects, tile_x, tile_y) is not None):
                        valid_placement = False
                    if valid_placement and (not _road_can_place(current_player, world_objects, all_units, tile_x, tile_y)):
                        valid_placement = False

                    # preview uses a road variant that matches nearby placed roads
                    tmp = Road(tile_x * TILE_SIZE, tile_y * TILE_SIZE, player_id=current_player.player_id)
                    v = _compute_road_variant(world_objects + [tmp], tmp)
                    tmp.set_variant(v)
                    img = Road._variant_images.get(tmp.variant)
                    if img is None:
                        Road._variant_images[tmp.variant] = Road._load_variant_image(tmp.variant)
                        img = Road._variant_images.get(tmp.variant)

                    px = tmp.pos.x - camera.x + VIEW_MARGIN_LEFT
                    py = tmp.pos.y - camera.y + VIEW_MARGIN_TOP

                    if img is not None:
                        preview_image = img.copy()
                        preview_image.set_alpha(128)
                        screen.blit(preview_image, (px, py))
                    else:
                        surf = pygame.Surface((TILE_SIZE, TILE_SIZE), pygame.SRCALPHA)
                        surf.fill((90, 90, 90, 128))
                        screen.blit(surf, (px, py))

                    border_color = GREEN if valid_placement else RED
                    pygame.draw.rect(screen, border_color, (px, py, TILE_SIZE, TILE_SIZE), 2)

                else:
                    # building preview (existing behavior)
                    temp_building = building_to_place(building_pos.x, building_pos.y, current_player.player_id, current_player.color)
                    cls_name = building_to_place.__name__
                    image = Unit._images.get(cls_name)
                    snapped_building_pos = Vector2(tile_x * TILE_SIZE + TILE_HALF, tile_y * TILE_SIZE + TILE_HALF)
                    x = snapped_building_pos.x - camera.x + VIEW_MARGIN_LEFT
                    y = snapped_building_pos.y - camera.y + VIEW_MARGIN_TOP

                    if image:
                        preview_image = image.copy()
                        preview_image.set_alpha(128)
                        image_rect = preview_image.get_rect(center=(int(x), int(y)))
                        screen.blit(preview_image, image_rect)
                    else:
                        preview_surface = pygame.Surface((building_size, building_size), pygame.SRCALPHA)
                        preview_surface.fill(temp_building.color[:3] + (128,))
                        screen.blit(preview_surface, (x - building_size / 2, y - building_size / 2))

                    border_color = GREEN if valid_placement else RED
                    snapped_x = snapped_building_pos.x - camera.x + VIEW_MARGIN_LEFT
                    snapped_y = snapped_building_pos.y - camera.y + VIEW_MARGIN_TOP
                    pygame.draw.rect(screen, border_color, (snapped_x - building_size / 2, snapped_y - building_size / 2, building_size, building_size), 2)

            # Draw selection rectangle
            if selecting and selection_start and selection_end and current_player:
                rect = pygame.Rect(
                    min(selection_start.x - camera.x + VIEW_MARGIN_LEFT, selection_end.x - camera.x + VIEW_MARGIN_LEFT),
                    min(selection_start.y - camera.y + VIEW_MARGIN_TOP, selection_end.y - camera.y + VIEW_MARGIN_TOP),
                    abs(selection_end.x - selection_start.x),
                    abs(selection_end.y - selection_start.y)
                )
                pygame.draw.rect(screen, current_player.color, rect, 3)

            # Draw UI (refactored)
            fps = clock.get_fps()
            ui.draw_game_ui(
                screen=screen,
                grid_buttons=grid_buttons,
                current_player=current_player,
                production_queues=production_queues,
                building_animations=building_animations,
                current_time=current_time,
                all_units=all_units,
                icons=icons,
                fonts=fonts,
                fps=fps,
                grass_tiles=grass_tiles,
                camera=camera,
            )

        elif game_state == GameState.DEFEAT:
            # Draw Defeat screen
            ui.draw_end_screen(
                screen,
                mode_text=("Defeat! Player 1 has lost all units.", (100, 0, 0)),
                quit_button=quit_button,
                fonts=fonts,
            )
            running = False

        elif game_state == GameState.VICTORY:
            # Draw Victory screen
            ui.draw_end_screen(
                screen,
                mode_text=("Victory! Player 2 has been defeated!", (0, 100, 0)),
                quit_button=quit_button,
                fonts=fonts,
            )
            running = False
        # Pause overlay (drawn on top of everything)
        if paused and game_state == GameState.RUNNING:
            resume_button_rect = ui.draw_pause_overlay(
                screen,
                player_id=(paused_by_player_id or (getattr(current_player, 'player_id', None) or 1)),
                fonts=fonts,
            )

        pygame.display.flip()
        clock.tick(60)

    # pygame.quit()
    return 1 if game_state == GameState.VICTORY else 0
    # sys.exit()


if __name__ == "__main__":
    result = run_game()
    print(f"Game finished with result: {result}")