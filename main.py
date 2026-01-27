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
from utils import is_tile_occupied, find_valid_spawn_tiles
from worldgen import init_game_world
from tiles import GrassTile, Dirt, River, Bridge
from units import Unit, Tree, Building, Barn, TownCenter, Barracks, Axeman, Knight, Archer, Cow, ShamansHut
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


    selection_end = None
    selecting = False
    current_player = None

    # --- Mouse selection state (click vs box-select) ---
    pending_click = False
    pending_click_unit = None
    mouse_down_screen = None
    mouse_down_world = None
    drag_threshold_px = 6  # pixels

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
    building_size = BUILDING_SIZE

    current_player = players[1]



    while running:

        current_time = pygame.time.get_ticks() / 1000
        context.current_time = current_time
        dt = clock.get_time() / 1000  # Delta time for frame-rate independent updates

        # Check for Defeat or Victory conditions
        if game_state == GameState.RUNNING:
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
                        click_pos = camera.screen_to_world(mouse_pos, view_margin_left=VIEW_MARGIN_LEFT, view_margin_top=VIEW_MARGIN_TOP)
                        clicked_something = False

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
                                # Deduct NOW (only on successful placement)
                                if current_player.milk < building_to_place.milk_cost or current_player.wood < building_to_place.wood_cost:
                                    print("Not enough resources to place building.")
                                else:
                                    current_player.milk -= building_to_place.milk_cost
                                    current_player.wood -= building_to_place.wood_cost

                                    new_building = building_to_place(building_pos.x, building_pos.y, current_player.player_id, current_player.color)
                                    new_building.alpha = 0
                                    current_player.add_unit(new_building)
                                    all_units.add(new_building)
                                    spatial_grid.add_unit(new_building)

                                    # Paint area to dirt (same as your old placement code)
                                    for row in range(tile_y - building_size_tiles // 2, tile_y + building_size_tiles // 2 + 1):
                                        for col in range(tile_x - building_size_tiles // 2, tile_x + building_size_tiles // 2 + 1):
                                            if 0 <= row < GRASS_ROWS and 0 <= col < GRASS_COLS:
                                                grass_tiles[row][col] = Dirt(col * TILE_SIZE, row * TILE_SIZE)

                                    highlight_times[new_building] = current_time
                                    building_animations[new_building] = {
                                        'start_time': current_time,
                                        'alpha': 0,
                                        'town_center': None
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
                    elif event.button == 3:  # Right click
                        mouse_pos = Vector2(event.pos)
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
                            selected_building = None
                            selected_non_buildings = 0
                            if current_player:
                                for unit in current_player.units:
                                    if unit.selected:
                                        if isinstance(unit, Building):
                                            if selected_building is None:
                                                selected_building = unit
                                            else:
                                                selected_building = None
                                        else:
                                            selected_non_buildings += 1
                            if selected_building and selected_non_buildings == 0 and not clicked_unit:
                                selected_building.rally_point = Vector2(snapped_pos)
                                print(f"Set rally point for {selected_building.__class__.__name__} at {selected_building.pos} to {selected_building.rally_point}")
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
            elif event.type == pygame.KEYDOWN and game_state == GameState.RUNNING:
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
                if game_state == GameState.RUNNING and event.button == 1:
                    # Finish box-select
                    if selecting and current_player:
                        mouse_pos = Vector2(event.pos)
                        if VIEW_MARGIN_LEFT <= mouse_pos.x <= VIEW_BOUNDS_X and VIEW_MARGIN_TOP <= mouse_pos.y <= VIEW_BOUNDS_Y:
                            selection_end = camera.screen_to_world(mouse_pos, view_margin_left=VIEW_MARGIN_LEFT, view_margin_top=VIEW_MARGIN_TOP)
                            selecting = False
                            for player in players:
                                player.deselect_all_units()
                            for unit in current_player.units:
                                unit.selected = (min(selection_start.x, selection_end.x) <= unit.pos.x <= max(selection_start.x, selection_end.x) and
                                                 min(selection_start.y, selection_end.y) <= unit.pos.y <= max(selection_start.y, selection_end.y))
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
                                if isinstance(selected_unit, (Axeman, Archer, Knight)) and selected_unit != unit_clicked:
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
            elif event.type == pygame.MOUSEMOTION:
                if game_state == GameState.RUNNING:
                    # If we have a pending click, turn it into a box-select once we move far enough.
                    if pending_click and not selecting and mouse_down_screen is not None:
                        delta = Vector2(event.pos) - mouse_down_screen
                        if delta.length_squared() >= (drag_threshold_px * drag_threshold_px):
                            # Start box selecting (even if the drag began over a unit/building)
                            selecting = True
                            pending_click = False
                            pending_click_unit = None
                            for player in players:
                                player.deselect_all_units()
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

        if game_state == GameState.RUNNING:
            # Update camera
            mouse_pos_screen = pygame.mouse.get_pos()
            camera.update(mouse_pos_screen)

            # Check for building selection
            barn_selected = any(isinstance(unit, Barn) and unit.selected and unit.alpha == 255 for unit in (current_player.units if current_player else []))
            barracks_selected = any(isinstance(unit, Barracks) and unit.selected and unit.alpha == 255 for unit in (current_player.units if current_player else []))
            town_center_selected = any(isinstance(unit, TownCenter) and unit.selected and unit.alpha == 255 for unit in (current_player.units if current_player else []))
            town_shamans_hut = any(isinstance(unit, ShamansHut) and unit.selected and unit.alpha == 255 for unit in (current_player.units if current_player else []))

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

            # Update building animations
            for building, anim in list(building_animations.items()):
                if building not in all_units:
                    del building_animations[building]
                    continue
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

            # Draw units
            for unit in all_units:
                if isinstance(unit, Building):
                    unit.draw(screen, camera.x, camera.y)
            for unit in all_units:
                if not isinstance(unit, Building):
                    if isinstance(unit, Tree):
                        unit.draw(screen, camera.x, camera.y, axemen)
                    else:
                        unit.draw(screen, camera.x, camera.y)

            # Draw attack animations
            for anim in attack_animations:
                start_x = anim['start_pos'].x - camera.x + VIEW_MARGIN_LEFT
                start_y = anim['start_pos'].y - camera.y + VIEW_MARGIN_TOP
                end_x = anim['end_pos'].x - camera.x + VIEW_MARGIN_LEFT
                end_y = anim['end_pos'].y - camera.y + VIEW_MARGIN_TOP
                pygame.draw.line(screen, anim['color'], (start_x, start_y), (end_x, end_y), 2)

            # Draw building preview during placement
            if placing_building and building_pos:
                temp_building = building_to_place(building_pos.x, building_pos.y, current_player.player_id, current_player.color)
                cls_name = building_to_place.__name__
                image = Unit._images.get(cls_name)
                tile_x = int(building_pos.x // TILE_SIZE)
                tile_y = int(building_pos.y // TILE_SIZE)
                snapped_building_pos = Vector2(tile_x * TILE_SIZE + TILE_HALF, tile_y * TILE_SIZE + TILE_HALF)
                x = snapped_building_pos.x - camera.x + VIEW_MARGIN_LEFT
                y = snapped_building_pos.y - camera.y + VIEW_MARGIN_TOP
                valid_placement = True
                building_size_tiles = int(building_size // TILE_SIZE)
                for row in range(tile_y - building_size_tiles // 2, tile_y + building_size_tiles // 2 + 1):
                    for col in range(tile_x - building_size_tiles // 2, tile_x + building_size_tiles // 2 + 1):
                        if not (0 <= row < GRASS_ROWS and 0 <= col < GRASS_COLS) or is_tile_occupied(row, col, all_units, context.grass_tiles):
                            valid_placement = False
                            break
                    if not valid_placement:
                        break
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
                current_time=current_time,
                all_units=all_units,
                icons=icons,
                fonts=fonts,
                fps=fps,
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

        pygame.display.flip()
        clock.tick(60)

    # pygame.quit()
    return 1 if game_state == GameState.VICTORY else 0
    # sys.exit()

if __name__ == "__main__":
    result = run_game()
    print(f"Game finished with result: {result}")