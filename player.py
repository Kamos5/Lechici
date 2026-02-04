from __future__ import annotations

import random
from typing import List, Optional, Tuple

import pygame
from pygame.math import Vector2

from constants import *
from tiles import GrassTile, Dirt, River
from units import Archer, Axeman, Barn, Barracks, Building, Cow, Knight, Bear, Strzyga, Priestess, Shaman, TownCenter, Tree, Unit, ShamansHut, WarriorsLodge, KnightsEstate, Wall
import context
from utils import is_tile_occupied
from world_objects import Road


class Player:
    def __init__(self, player_id, color, start_x, start_y):
        self.player_id = player_id
        self.color = color
        self.milk = 400.0
        self.max_milk = 1000
        self.wood = 500.0
        self.max_wood = 1000
        self.units = []
        self.cow_in_barn = {}
        self.barns = []
        self.unit_limit = None if player_id == 0 else 10
        self.unit_count = 0
        self.building_limit = None if player_id == 0 else 5
        self.building_count = 0
        self.used_male_names = set()  # Track used male names for Axeman, Archer, Knight, Bear, Strzyga, Priestess, Shaman
        self.used_female_names = set()  # Track used female names for Cow
        offset_x = start_x * SCALE
        offset_y = start_y * SCALE
        # if self.player_id > 0:
        #     initial_units = [
        #         Axeman(offset_x + 100 * SCALE, offset_y + 100 * SCALE, player_id, self.color),
        #         Knight(offset_x + 150 * SCALE, offset_y + 100 * SCALE, player_id, self.color),
        #         Archer(offset_x + 200 * SCALE, offset_y + 100 * SCALE, player_id, self.color),
        #         Cow(offset_x + 260 * SCALE, offset_y + 100 * SCALE, player_id, self.color),
        #         Cow(offset_x + 280 * SCALE, offset_y + 100 * SCALE, player_id, self.color),
        #         Barn(offset_x + 230 * SCALE, offset_y + 150 * SCALE, player_id, self.color),
        #         TownCenter(offset_x + 150 * SCALE, offset_y + 150 * SCALE, player_id, self.color),
        #         Barracks(offset_x + 70 * SCALE, offset_y + 150 * SCALE, player_id, self.color),
        #     ]
        #     for unit in initial_units:
        #         self.add_unit(unit)
        #     self.unit_count = len([unit for unit in self.units if not isinstance(unit, Building)])
        #     self.building_count = len([unit for unit in self.units if isinstance(unit, Building)])
        #     self.barns = [unit for unit in self.units if isinstance(unit, Barn)]
        #     print(f"Player {self.player_id} initialized with {self.unit_count}/{self.unit_limit} units and {self.building_count}/{self.building_limit} buildings")

    def add_unit(self, unit):
        self.units.append(unit)
        # Assign a unique name to Axeman, Archer, Knight, Bear, Strzyga, Priestess, Shaman, or Cow
        if isinstance(unit, (Axeman, Archer, Knight, Bear, Strzyga, Priestess, Shaman, Cow)) and unit.name is None:
            if isinstance(unit, Cow) or isinstance(unit, Priestess) or isinstance(unit, Strzyga):
                # Assign female name to Cow
                available_names = [name for name in FEMALE_NAMES if name not in self.used_female_names]
                if available_names:
                    unit.name = random.choice(available_names)
                    self.used_female_names.add(unit.name)
                    print(f"Assigned female name {unit.name} to Cow or Priestess or Strzyga for Player {self.player_id}")
                else:
                    unit.name = f"Cow_{len(self.used_female_names) + 1}"
                    self.used_female_names.add(unit.name)
                    print(f"No female names available, assigned {unit.name} to Cow or Priestess or Strzyga for Player {self.player_id}")
            else:
                # Assign male name to Axeman, Archer, Knight, Bear, Strzyga, Priestess, Shaman
                available_names = [name for name in UNIQUE_MALE_NAMES if name not in self.used_male_names]
                if available_names:
                    unit.name = random.choice(available_names)
                    self.used_male_names.add(unit.name)
                    print(f"Assigned male name {unit.name} to {unit.__class__.__name__} for Player {self.player_id}")
                else:
                    unit.name = f"{unit.__class__.__name__}_{len(self.used_male_names) + 1}"
                    self.used_male_names.add(unit.name)
                    print(f"No male names available, assigned {unit.name} to {unit.__class__.__name__} for Player {self.player_id}")
        if isinstance(unit, Wall):
            return

        if isinstance(unit, Building):
            if self.building_limit is not None:
                self.building_count += 1
                print(f"Player {self.player_id} building count increased to {self.building_count}/{self.building_limit}")
        else:
            if self.unit_limit is not None:
                self.unit_count += 1
                print(f"Player {self.player_id} unit count increased to {self.unit_count}/{self.unit_limit}")

    def remove_unit(self, unit):
        if unit in self.units:
            self.units.remove(unit)
            # Remove the unit's name from the appropriate name set
            if unit.name:
                if isinstance(unit, Cow) and unit.name in self.used_female_names:
                    self.used_female_names.remove(unit.name)
                    print(f"Removed female name {unit.name} from Player {self.player_id}")
                elif isinstance(unit, (Axeman, Archer, Knight, Bear, Strzyga, Priestess, Shaman)) and unit.name in self.used_male_names:
                    self.used_male_names.remove(unit.name)
                    print(f"Removed male name {unit.name} from Player {self.player_id}")
            if isinstance(unit, Wall):
                return
            if isinstance(unit, Building):
                if self.building_limit is not None:
                    self.building_count -= 1
                    print(f"Player {self.player_id} building count decreased to {self.building_count}/{self.building_limit}")
                if isinstance(unit, Barn):
                    self.barns = [u for u in self.units if isinstance(u, Barn)]
            else:
                if self.unit_limit is not None:
                    self.unit_count -= 1
                    print(f"Player {self.player_id} unit count decreased to {self.unit_count}/{self.unit_limit}")

    def select_all_units(self):
        for unit in self.units:
            unit.selected = True

    def deselect_all_units(self):
        for unit in self.units:
            unit.selected = False

class PlayerAI:
    def __init__(self, player, grass_tiles, spatial_grid, all_units, production_queues):
        self.player = player
        self.context = context
        self.context.grass_tiles = context.grass_tiles
        self.context.spatial_grid = context.spatial_grid
        self.context.all_units = context.all_units
        self.context.production_queues = context.production_queues
        self.last_decision_time = 0
        self.decision_interval = 5.0  # Make decisions every 5 seconds
        self.building_size = BUILDING_SIZE
        self.building_size_tiles = int(self.building_size // TILE_SIZE)
        self.max_barns = 2
        self.max_units = 15
        # Attack wave variables
        self.wave_start_time = 0  # Initialize to 0; set in update
        self.wave_number = 0  # Track current wave
        self.attack_units = []  # Units currently assigned to attack
        self.target_player_id = 1  # Target Player 1
        self.target_camp_pos = None  # Center of Player 1's camp
        self.detection_range = 4 * TILE_SIZE  # 6 tiles for enemy detection
        self.detection_interval = 1.0  # Check every 1 second for performance
        self.last_detection_time = 0


    # --- Roads & connectivity (AI) ---
    _ROAD_VARIANT = {
        (0, 0, 0, 0): "road10",
        (0, 0, 0, 1): "road1",
        (0, 0, 1, 0): "road2",
        (0, 1, 0, 0): "road3",
        (1, 0, 0, 0): "road4",
        (1, 0, 0, 1): "road5",
        (1, 1, 0, 0): "road6",
        (0, 1, 0, 1): "road7",
        (1, 1, 0, 1): "road8",
        (1, 0, 1, 0): "road9",
        (0, 0, 1, 1): "road10",
        (1, 0, 1, 1): "road11",
        (0, 1, 1, 0): "road12",
        (1, 1, 1, 0): "road13",
        (0, 1, 1, 1): "road14",
        (1, 1, 1, 1): "road15",
    }

    def _tile_of_world_object(self, obj) -> tuple[int, int]:
        return (int(obj.pos.x // TILE_SIZE), int(obj.pos.y // TILE_SIZE))

    def _find_road_at(self, tx: int, ty: int) -> Road | None:
        for o in getattr(self.context, 'world_objects', []):
            if isinstance(o, Road):
                ox, oy = self._tile_of_world_object(o)
                if ox == tx and oy == ty:
                    return o
        return None

    def _building_footprint_tiles(self, center_tx: int, center_ty: int, size_tiles: int) -> list[tuple[int, int]]:
        half = size_tiles // 2
        out = []
        for dy in range(-half, half + 1):
            for dx in range(-half, half + 1):
                out.append((center_tx + dx, center_ty + dy))
        return out

    def _connected_road_tiles(self) -> set[tuple[int, int]]:
        """Road tiles reachable from this player's TownCenter via 4-neighbor road chain.

        Roads are ownership-less: any Road may be used.
        """
        world_objects = getattr(self.context, 'world_objects', [])
        all_roads = set()
        for o in world_objects:
            if isinstance(o, Road):
                all_roads.add(self._tile_of_world_object(o))
        if not all_roads:
            return set()

        town_centers = [u for u in self.context.all_units if isinstance(u, TownCenter) and u.player_id == self.player.player_id and getattr(u, 'alpha', 255) == 255]
        if not town_centers:
            return set()

        # seed from any road adjacent to TC footprint
        seeds = []
        for tc in town_centers:
            tc_tx, tc_ty = int(tc.pos.x // TILE_SIZE), int(tc.pos.y // TILE_SIZE)
            fp = self._building_footprint_tiles(tc_tx, tc_ty, self.building_size_tiles)
            for fx, fy in fp:
                for dx, dy in ((0,-1),(0,1),(-1,0),(1,0)):
                    k = (fx+dx, fy+dy)
                    if k in all_roads:
                        seeds.append(k)
        if not seeds:
            return set()

        visited = set(seeds)
        stack = list(seeds)
        while stack:
            x, y = stack.pop()
            for dx, dy in ((0,-1),(0,1),(-1,0),(1,0)):
                nk = (x+dx, y+dy)
                if nk in all_roads and nk not in visited:
                    visited.add(nk)
                    stack.append(nk)
        return visited

    def _is_near_connected_road(self, center_tx: int, center_ty: int, size_tiles: int) -> bool:
        connected = self._connected_road_tiles()
        if not connected:
            return False
        for fx, fy in self._building_footprint_tiles(center_tx, center_ty, size_tiles):
            for dx, dy in ((0,-1),(0,1),(-1,0),(1,0)):
                if (fx+dx, fy+dy) in connected:
                    return True
        return False

    def _compute_road_variant(self, road: Road) -> str:
        tx, ty = self._tile_of_world_object(road)
        up = 1 if self._find_road_at(tx, ty-1) else 0
        down = 1 if self._find_road_at(tx, ty+1) else 0
        left = 1 if self._find_road_at(tx-1, ty) else 0
        right = 1 if self._find_road_at(tx+1, ty) else 0
        return self._ROAD_VARIANT.get((up, down, left, right), 'road10')

    def _refresh_road_and_neighbors(self, road: Road) -> None:
        tx, ty = self._tile_of_world_object(road)
        candidates = [road]
        for dx, dy in ((0,-1),(0,1),(-1,0),(1,0)):
            r2 = self._find_road_at(tx+dx, ty+dy)
            if r2:
                candidates.append(r2)
        for r in candidates:
            r.set_variant(self._compute_road_variant(r))

    def _road_tile_placeable(self, tx: int, ty: int) -> bool:
        if not (0 <= tx < GRASS_COLS and 0 <= ty < GRASS_ROWS):
            return False
        if self._find_road_at(tx, ty):
            return False
        # don't allow roads on rivers
        if isinstance(self.context.grass_tiles[ty][tx], River):
            return False
        if is_tile_occupied(ty, tx, self.context.all_units, self.context.grass_tiles):
            return False
        return True

    def _place_road_tile(self, tx: int, ty: int) -> bool:
        if not self._road_tile_placeable(tx, ty):
            return False
        if self.player.milk < Road.milk_cost or self.player.wood < Road.wood_cost:
            return False

        self.player.milk -= Road.milk_cost
        self.player.wood -= Road.wood_cost

        if not hasattr(self.context, 'world_objects') or self.context.world_objects is None:
            self.context.world_objects = []

        r = Road(tx * TILE_SIZE, ty * TILE_SIZE, player_id=self.player.player_id)
        self.context.world_objects.append(r)
        self._refresh_road_and_neighbors(r)
        return True

    def _ensure_initial_road_from_tc(self) -> None:
        """If we have a TC but zero connected roads, try to place 1 road adjacent to it."""
        if self._connected_road_tiles():
            return
        town_centers = [u for u in self.context.all_units if isinstance(u, TownCenter) and u.player_id == self.player.player_id and getattr(u, 'alpha', 255) == 255]
        if not town_centers:
            return
        tc = town_centers[0]
        tc_tx, tc_ty = int(tc.pos.x // TILE_SIZE), int(tc.pos.y // TILE_SIZE)
        fp = self._building_footprint_tiles(tc_tx, tc_ty, self.building_size_tiles)
        # try any adjacent tile
        for fx, fy in fp:
            for dx, dy in ((0,-1),(0,1),(-1,0),(1,0)):
                if self._place_road_tile(fx+dx, fy+dy):
                    return

    def _build_road_to_build_site(self, center_tx: int, center_ty: int, size_tiles: int) -> None:
        """Build a Manhattan-shortest road from the existing connected network (or TC) to a tile adjacent to the build footprint."""
        self._ensure_initial_road_from_tc()
        connected = self._connected_road_tiles()

        # pick target adjacent tile around building footprint that is placeable
        targets = []
        for fx, fy in self._building_footprint_tiles(center_tx, center_ty, size_tiles):
            for dx, dy in ((0,-1),(0,1),(-1,0),(1,0)):
                tx, ty = fx+dx, fy+dy
                if self._road_tile_placeable(tx, ty):
                    targets.append((tx, ty))
        if not targets:
            return

        # choose nearest start tile from connected (or a TC-adjacent road we just placed)
        if not connected:
            return

        def manh(a,b):
            return abs(a[0]-b[0]) + abs(a[1]-b[1])

        best = None
        for t in targets:
            for s in connected:
                d = manh(s, t)
                if best is None or d < best[0]:
                    best = (d, s, t)
        if not best:
            return
        _, s, t = best

        # lay an L-shaped path (x then y), falling back to y then x if blocked
        def lay_path(path):
            for px, py in path:
                # skip tiles already road
                if self._find_road_at(px, py):
                    continue
                if not self._place_road_tile(px, py):
                    return False
            return True

        sx, sy = s
        tx, ty = t
        path1 = []
        # step x
        step = 1 if tx >= sx else -1
        for x in range(sx, tx + step, step):
            path1.append((x, sy))
        # step y
        step = 1 if ty >= sy else -1
        for y in range(sy, ty + step, step):
            path1.append((tx, y))

        path2 = []
        step = 1 if ty >= sy else -1
        for y in range(sy, ty + step, step):
            path2.append((sx, y))
        step = 1 if tx >= sx else -1
        for x in range(sx, tx + step, step):
            path2.append((x, ty))

        if not lay_path(path1):
            lay_path(path2)

    def _ensure_roads_around_building(self, building) -> None:
        center_tx, center_ty = int(building.pos.x // TILE_SIZE), int(building.pos.y // TILE_SIZE)
        fp = self._building_footprint_tiles(center_tx, center_ty, self.building_size_tiles)
        # place roads on the ring around the footprint (4-neighbor)
        for fx, fy in fp:
            for dx, dy in ((0,-1),(0,1),(-1,0),(1,0)):
                tx, ty = fx+dx, fy+dy
                # only build if it would be connected (we just built to the site) and placeable
                if self._road_tile_placeable(tx, ty):
                    self._place_road_tile(tx, ty)



    def find_valid_building_position(self, center_pos, require_road: bool = True):
        """Find a clear building center position near center_pos.

        If require_road=True, the chosen position must touch a road tile that is connected to this AI player's TownCenter.
        """
        center_tile_x = int(center_pos.x // TILE_SIZE)
        center_tile_y = int(center_pos.y // TILE_SIZE)
        search_radius = 8
        for dr in range(-search_radius, search_radius + 1):
            for dc in range(-search_radius, search_radius + 1):
                tile_x = center_tile_x + dc
                tile_y = center_tile_y + dr

                # optional road constraint
                if require_road and not self._is_near_connected_road(tile_x, tile_y, self.building_size_tiles):
                    continue

                valid = True
                for row in range(tile_y - self.building_size_tiles // 2, tile_y + self.building_size_tiles // 2 + 1):
                    for col in range(tile_x - self.building_size_tiles // 2, tile_x + self.building_size_tiles // 2 + 1):
                        if not (0 <= row < GRASS_ROWS and 0 <= col < GRASS_COLS) or is_tile_occupied(row, col, self.context.all_units, self.context.grass_tiles):
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    return Vector2(tile_x * TILE_SIZE + TILE_HALF, tile_y * TILE_SIZE + TILE_HALF)
        return None

    def find_closest_enemy(self, unit, enemy_units):
        """Find the closest enemy unit or building to attack."""
        if not enemy_units:
            return None
        return min(enemy_units, key=lambda e: unit.pos.distance_to(e.pos))

    def find_closest_enemy_building(self, unit, enemy_buildings):
        """Find the closest enemy building to attack."""
        if not enemy_buildings:
            return None
        return min(enemy_buildings, key=lambda e: unit.pos.distance_to(e.pos))

    def initiate_wave(self, current_time, waypoint_graph):
        """Initiate attack waves based on time elapsed."""
        if self.wave_start_time == 0:
            self.wave_start_time = context.current_time
        elapsed_time = context.current_time - self.wave_start_time
        target_player = next((p for p in context.players if p.player_id == self.target_player_id), None)
        if not target_player:
            return

        # Set fallback camp position (used in update_attack_units if no enemies)
        if not self.target_camp_pos:
            town_centers = [u for u in target_player.units if isinstance(u, TownCenter)]
            if town_centers:
                self.target_camp_pos = town_centers[0].pos
            else:
                self.target_camp_pos = Vector2(150, 150)

        # Get all Player 1 buildings
        enemy_buildings = [u for u in target_player.units if isinstance(u, (TownCenter, Barn, Barracks, ShamansHut, KnightsEstate, WarriorsLodge)) and u.hp > 0 and u.alpha == 255]

        # Wave 1: After 30 seconds, 1 Archer
        if self.wave_number == 0 and elapsed_time >= 150:
            archers = [u for u in self.player.units if isinstance(u, Archer) and u not in self.attack_units]
            if archers:
                archer = random.choice(archers)
                self.attack_units.append(archer)
                if enemy_buildings:
                    target_building = self.find_closest_enemy_building(archer, enemy_buildings)
                    archer.target = target_building
                    archer.path = context.waypoint_graph.get_path(archer.pos, target_building.pos, archer)
                    print(f"AI Wave 1: Archer at {archer.pos} sent to Player 1 {target_building.__class__.__name__} at {target_building.pos}")
                else:
                    archer.target = self.target_camp_pos
                    archer.path = context.waypoint_graph.get_path(archer.pos, self.target_camp_pos, archer)
                    print(f"AI Wave 1: Archer at {archer.pos} sent to Player 1 camp at {self.target_camp_pos} (no buildings found)")
                archer.autonomous_target = False
                archer.path_index = 0
                self.wave_number = 1
                self.last_decision_time = context.current_time

        # Wave 2: After 90 seconds, 2 random military units
        elif self.wave_number == 1 and elapsed_time >= 400:
            military_units = [u for u in self.player.units if isinstance(u, (Axeman, Archer, Knight, Bear, Strzyga, Priestess, Shaman)) and u not in self.attack_units]
            if len(military_units) >= 2:
                selected_units = random.sample(military_units, 2)
                for unit in selected_units:
                    self.attack_units.append(unit)
                    if isinstance(unit, Axeman):
                        unit.depositing = False
                        unit.special = 0
                        unit.return_pos = None
                    if enemy_buildings:
                        target_building = self.find_closest_enemy_building(unit, enemy_buildings)
                        unit.target = target_building
                        unit.path = context.waypoint_graph.get_path(unit.pos, target_building.pos, unit)
                        print(f"AI Wave 2: {unit.__class__.__name__} at {unit.pos} sent to Player 1 {target_building.__class__.__name__} at {target_building.pos}")
                    else:
                        unit.target = self.target_camp_pos
                        unit.path = context.waypoint_graph.get_path(unit.pos, self.target_camp_pos, unit)
                        print(f"AI Wave 2: {unit.__class__.__name__} at {unit.pos} sent to Player 1 camp at {self.target_camp_pos} (no buildings found)")
                    unit.autonomous_target = False
                    unit.path_index = 0
                self.wave_number = 2
                self.last_decision_time = context.current_time

        elif self.wave_number >= 2 and context.current_time >= self.last_decision_time + 150:
            military_units = [u for u in self.player.units if isinstance(u, (Axeman, Archer, Knight, Bear, Strzyga, Priestess, Shaman)) and u not in self.attack_units]
            num_units = min(random.randint(5, 10), len(military_units))
            if num_units > 0:
                selected_units = random.sample(military_units, num_units)
                for unit in selected_units:
                    self.attack_units.append(unit)
                    if isinstance(unit, Axeman):
                        unit.depositing = False
                        unit.special = 0
                        unit.return_pos = None
                    if enemy_buildings:
                        target_building = self.find_closest_enemy_building(unit, enemy_buildings)
                        unit.target = target_building
                        unit.path = context.waypoint_graph.get_path(unit.pos, target_building.pos, unit)
                        print(f"AI Wave {self.wave_number + 1}: {unit.__class__.__name__} at {unit.pos} sent to Player 1 {target_building.__class__.__name__} at {target_building.pos}")
                    else:
                        unit.target = self.target_camp_pos
                        unit.path = context.waypoint_graph.get_path(unit.pos, self.target_camp_pos, unit)
                        print(f"AI Wave {self.wave_number + 1}: {unit.__class__.__name__} at {unit.pos} sent to Player 1 camp at {self.target_camp_pos} (no buildings found)")
                    unit.autonomous_target = False
                    unit.path_index = 0
                self.wave_number += 1
                self.last_decision_time = context.current_time

    def update_attack_units(self, current_time, waypoint_graph, spatial_grid, all_units, grass_tiles, production_queues):
        """Update attack units and detect nearby enemies within 6 tiles."""
        if context.current_time - self.last_detection_time < self.detection_interval:
            return
        self.last_detection_time = context.current_time

        target_player = next((p for p in context.players if p.player_id == self.target_player_id), None)
        if not target_player:
            return

        # Update existing attack units
        enemy_units = [u for u in context.all_units if u.player_id == self.target_player_id and not isinstance(u, Tree) and u.hp > 0 and u.alpha == 255]
        for unit in self.attack_units[:]:
            if unit not in self.player.units or unit.hp <= 0:
                self.attack_units.remove(unit)
                continue
            # If unit has reached the camp or has no target, find the closest enemy
            if not unit.target or (isinstance(unit.target, Vector2) and unit.pos.distance_to(unit.target) <= unit.size):
                closest_enemy = self.find_closest_enemy(unit, enemy_units)
                if closest_enemy:
                    unit.target = closest_enemy
                    unit.autonomous_target = True
                    unit.path = context.waypoint_graph.get_path(unit.pos, closest_enemy.pos, unit)
                    unit.path_index = 0
                    print(f"AI: {unit.__class__.__name__} at {unit.pos} targeting enemy {closest_enemy.__class__.__name__} at {closest_enemy.pos}")
                else:
                    # Return to camp if no enemies
                    unit.target = self.target_camp_pos
                    unit.autonomous_target = True
                    unit.path = context.waypoint_graph.get_path(unit.pos, self.target_camp_pos, unit)
                    unit.path_index = 0
                    print(f"AI: {unit.__class__.__name__} at {unit.pos} returning to Player 1 camp at {self.target_camp_pos}")

        # Detect nearby enemies for idle military units
        military_units = [u for u in self.player.units if isinstance(u, (Axeman, Archer, Knight, Bear, Strzyga, Priestess, Shaman)) and not u.target and u not in self.attack_units]
        for unit in military_units:
            nearby_enemies = context.spatial_grid.get_nearby_units(unit, radius=self.detection_range)
            nearby_enemies = [e for e in nearby_enemies if e in enemy_units and e.hp > 0]
            if nearby_enemies:
                closest_enemy = self.find_closest_enemy(unit, nearby_enemies)
                if closest_enemy:
                    unit.target = closest_enemy
                    unit.autonomous_target = True
                    unit.path = context.waypoint_graph.get_path(unit.pos, closest_enemy.pos, unit)
                    unit.path_index = 0
                    print(f"AI: {unit.__class__.__name__} at {unit.pos} detected and targeting Player 1 {closest_enemy.__class__.__name__} at {closest_enemy.pos} (distance: {unit.pos.distance_to(closest_enemy.pos):.1f})")

    def update(self, current_time, waypoint_graph, spatial_grid, all_units, grass_tiles, production_queues):
        """Update AI decisions including attack waves and detection."""
        if context.current_time - self.last_decision_time < self.decision_interval:
            return
        self.last_decision_time = context.current_time

        # Initiate attack waves
        self.initiate_wave(context.current_time, context.waypoint_graph)

        # Manage resources
        self.manage_cows()
        self.manage_axemen()

        # Build structures and units
        self.manage_buildings(context.current_time)
        self.manage_unit_production(context.current_time)

    def manage_cows(self):
        # Existing method unchanged
        for cow in [u for u in self.player.units if isinstance(u, Cow) and u not in self.attack_units]:
            if not cow.target and cow.special < 100:
                tile_x = int(cow.pos.x // TILE_SIZE)
                tile_y = int(cow.pos.y // TILE_SIZE)
                adjacent_tiles = [
                    (tile_x, tile_y - 1), (tile_x, tile_y + 1), (tile_x - 1, tile_y), (tile_x + 1, tile_y),
                    (tile_x - 1, tile_y - 1), (tile_x + 1, tile_y - 1), (tile_x - 1, tile_y + 1), (tile_x + 1, tile_y + 1)
                ]
                random.shuffle(adjacent_tiles)
                for adj_x, adj_y in adjacent_tiles:
                    if (0 <= adj_x < GRASS_COLS and 0 <= adj_y < GRASS_ROWS and
                        isinstance(self.context.grass_tiles[adj_y][adj_x], GrassTile) and
                        not isinstance(self.context.grass_tiles[adj_y][adj_x], Dirt) and
                        self.context.grass_tiles[adj_y][adj_x].grass_level > 0.5 and
                        cow.is_tile_walkable(adj_x, adj_y, self.context.spatial_grid)):
                        cow.target = self.context.grass_tiles[adj_y][adj_x].center
                        cow.autonomous_target = True
                        # print(f"AI: Cow at {cow.pos} assigned to grass tile at {cow.target}")
                        break
            elif cow.special >= 100 and not cow.target:
                barns = [u for u in self.player.units if isinstance(u, Barn) and u.alpha == 255 and u not in self.player.cow_in_barn]
                if barns:
                    barn = min(barns, key=lambda b: cow.pos.distance_to(b.pos))
                    target_x = barn.pos.x - barn.size / 2 + TILE_HALF
                    target_y = barn.pos.y + barn.size / 2 - TILE_HALF
                    cow.target = Vector2(target_x, target_y)
                    cow.return_pos = Vector2(cow.pos)
                    cow.assigned_corner = cow.target
                    cow.autonomous_target = True
                    # print(f"AI: Cow at {cow.pos} assigned to barn at {cow.target} for milk deposit")

    def manage_axemen(self):
        # Existing method unchanged
        trees = [u for u in self.context.all_units if isinstance(u, Tree) and u.player_id == 0]
        axemen = [u for u in self.player.units if isinstance(u, Axeman) and u not in self.attack_units]
        chopping_axemen = [a for a in axemen if a.target and isinstance(a.target, Vector2)]
        depositing_axemen = [a for a in axemen if a.depositing]
        num_chopping = len(chopping_axemen)

        if self.player.wood >= self.player.max_wood:
            for axeman in chopping_axemen:
                axeman.target = None
                axeman.autonomous_target = False
                print(f"AI: Axeman at {axeman.pos} stopped harvesting tree due to wood limit reached ({self.player.wood}/{self.player.max_wood})")
            num_chopping = 0

        min_chopping = 2 if self.player.wood < self.player.max_wood else 0
        if num_chopping < min_chopping and trees and self.player.wood < self.player.max_wood:
            idle_axemen = [a for a in axemen if not a.target and a.special == 0 and not a.depositing]
            for axeman in idle_axemen[:min_chopping - num_chopping]:
                closest_tree = min(trees, key=lambda t: axeman.pos.distance_to(t.pos))
                axeman.target = closest_tree.pos
                axeman.autonomous_target = True
                print(f"AI: Axeman at {axeman.pos} assigned to tree at {axeman.target} to meet chopping quota")
                num_chopping += 1

        if self.player.wood < self.player.max_wood:
            for axeman in axemen:
                if not axeman.target and axeman.special == 0 and not axeman.depositing and trees:
                    closest_tree = min(trees, key=lambda t: axeman.pos.distance_to(t.pos))
                    axeman.target = closest_tree.pos
                    axeman.autonomous_target = True
                    print(f"AI: Axeman at {axeman.pos} assigned to tree at {axeman.target}")

        for axeman in axemen:
            if axeman.special >= 100 and not axeman.target:
                town_centers = [u for u in self.player.units if isinstance(u, TownCenter) and u.alpha == 255]
                if town_centers:
                    closest_town = min(town_centers, key=lambda t: axeman.pos.distance_to(t.pos))
                    axeman.target = closest_town.pos
                    axeman.depositing = True
                    print(f"AI: Axeman at {axeman.pos} assigned to TownCenter at {axeman.target} for wood deposit")

    def manage_buildings(self, current_time):
        """Manage building construction with context.current_time passed."""
        town_centers = [u for u in self.player.units if isinstance(u, TownCenter) and u.alpha == 255 and u not in self.context.production_queues]
        if not town_centers:
            return
        town_center = random.choice(town_centers)
        barn_count = len([u for u in self.player.units if isinstance(u, Barn)])
        if (barn_count < self.max_barns and
                self.player.milk >= Barn.milk_cost and self.player.wood >= Barn.wood_cost and
                (self.player.building_limit is None or self.player.building_count < self.player.building_limit)):
            pos = self.find_valid_building_position(town_center.pos, require_road=True)
            if not pos:
                # try again without road constraint, then build road path to it
                pos = self.find_valid_building_position(town_center.pos, require_road=False)
                if pos:
                    self._build_road_to_build_site(int(pos.x // TILE_SIZE), int(pos.y // TILE_SIZE), self.building_size_tiles)
            if pos and not self._is_near_connected_road(int(pos.x // TILE_SIZE), int(pos.y // TILE_SIZE), self.building_size_tiles):
                pos = None
            if pos:
                new_building = Barn(pos.x, pos.y, self.player.player_id, self.player.color)
                new_building.alpha = 0
                self.player.add_unit(new_building)
                self.context.all_units.add(new_building)
                self.context.spatial_grid.add_unit(new_building)
                for row in range(int(pos.y // TILE_SIZE) - self.building_size_tiles // 2, int(pos.y // TILE_SIZE) + self.building_size_tiles // 2 + 1):
                    for col in range(int(pos.x // TILE_SIZE) - self.building_size_tiles // 2, int(pos.x // TILE_SIZE) + self.building_size_tiles // 2 + 1):
                        if 0 <= row < GRASS_ROWS and 0 <= col < GRASS_COLS:
                            self.context.grass_tiles[row][col] = Dirt(col * TILE_SIZE, row * TILE_SIZE)
                self.context.production_queues[town_center] = {
                    'unit_type': Barn,
                    'start_time': context.current_time,
                    'player_id': self.player.player_id
                }
                context.building_animations[new_building] = {
                    'start_time': context.current_time,
                    'alpha': 0,
                    'town_center': town_center
                }
                self.player.milk -= Barn.milk_cost
                self.player.wood -= Barn.wood_cost
                print(f"AI: Placed Barn at {pos} for Player {self.player.player_id} (Barn count: {barn_count + 1}/{self.max_barns})")
                self._ensure_roads_around_building(new_building)
        elif (self.player.milk >= Barracks.milk_cost and self.player.wood >= Barracks.wood_cost and
              (self.player.building_limit is None or self.player.building_count < self.player.building_limit)):
            pos = self.find_valid_building_position(town_center.pos, require_road=True)
            if not pos:
                # try again without road constraint, then build road path to it
                pos = self.find_valid_building_position(town_center.pos, require_road=False)
                if pos:
                    self._build_road_to_build_site(int(pos.x // TILE_SIZE), int(pos.y // TILE_SIZE), self.building_size_tiles)
            if pos and not self._is_near_connected_road(int(pos.x // TILE_SIZE), int(pos.y // TILE_SIZE), self.building_size_tiles):
                pos = None
            if pos:
                new_building = Barracks(pos.x, pos.y, self.player.player_id, self.player.color)
                new_building.alpha = 0
                self.player.add_unit(new_building)
                self.context.all_units.add(new_building)
                self.context.spatial_grid.add_unit(new_building)
                for row in range(int(pos.y // TILE_SIZE) - self.building_size_tiles // 2, int(pos.y // TILE_SIZE) + self.building_size_tiles // 2 + 1):
                    for col in range(int(pos.x // TILE_SIZE) - self.building_size_tiles // 2, int(pos.x // TILE_SIZE) + self.building_size_tiles // 2 + 1):
                        if 0 <= row < GRASS_ROWS and 0 <= col < GRASS_COLS:
                            self.context.grass_tiles[row][col] = Dirt(col * TILE_SIZE, row * TILE_SIZE)
                self.context.production_queues[town_center] = {
                    'unit_type': Barracks,
                    'start_time': context.current_time,
                    'player_id': self.player.player_id
                }
                context.building_animations[new_building] = {
                    'start_time': context.current_time,
                    'alpha': 0,
                    'town_center': town_center
                }
                self.player.milk -= Barracks.milk_cost
                self.player.wood -= Barracks.wood_cost
                print(f"AI: Placed Barracks at {pos} for Player {self.player.player_id}")
                self._ensure_roads_around_building(new_building)

    def manage_unit_production(self, current_time):
        """Manage unit production with context.current_time passed."""
        barns = [u for u in self.player.units if isinstance(u, Barn) and u.alpha == 255 and u not in self.context.production_queues]
        barracks = [u for u in self.player.units if isinstance(u, Barracks) and u.alpha == 255 and u not in self.context.production_queues]
        barn_count = len([u for u in self.player.units if isinstance(u, Barn)])
        cow_count = len([u for u in self.player.units if isinstance(u, Cow)])
        axeman_count = len([u for u in self.player.units if isinstance(u, Axeman)])
        archer_count = len([u for u in self.player.units if isinstance(u, Archer)])
        knight_count = len([u for u in self.player.units if isinstance(u, Knight)])
        total_unit_count = self.player.unit_count
        chopping_axemen = len([a for a in self.player.units if isinstance(a, Axeman) and a.target and isinstance(a.target, Vector2)])

        max_cows = barn_count * 2
        if (barns and cow_count < max_cows and total_unit_count < self.max_units and
                self.player.milk >= Cow.milk_cost and self.player.wood >= Cow.wood_cost):
            barn = random.choice(barns)
            self.context.production_queues[barn] = {
                'unit_type': Cow,
                'start_time': context.current_time,
                'player_id': self.player.player_id
            }
            self.player.milk -= Cow.milk_cost
            self.player.wood -= Cow.wood_cost
            print(f"AI: Queued Cow production in Barn at {barn.pos} for Player {self.player.player_id} (Cow count: {cow_count + 1}/{max_cows}, Barn count: {barn_count})")

        if barracks and total_unit_count < self.max_units:
            barracks = random.choice(barracks)
            total_combat_units = axeman_count + archer_count + knight_count
            if total_combat_units == 0:
                desired_axemen = desired_archers = desired_knights = 0
            else:
                desired_axemen = int(total_combat_units * 0.4)
                desired_archers = int(total_combat_units * 0.4)
                desired_knights = int(total_combat_units * 0.2)

            if chopping_axemen < 2 and self.player.wood < self.player.max_wood:
                if self.player.milk >= Axeman.milk_cost and self.player.wood >= Axeman.wood_cost:
                    self.context.production_queues[barracks] = {
                        'unit_type': Axeman,
                        'start_time': context.current_time,
                        'player_id': self.player.player_id
                    }
                    self.player.milk -= Axeman.milk_cost
                    self.player.wood -= Axeman.wood_cost
                    print(f"AI: Queued Axeman production in Barracks at {barracks.pos} to meet chopping quota (Chopping Axemen: {chopping_axemen}/2)")
                    return

            axeman_deficit = desired_axemen - axeman_count if total_combat_units > 0 else 1
            archer_deficit = desired_archers - archer_count if total_combat_units > 0 else 1
            knight_deficit = desired_knights - knight_count if total_combat_units > 0 else 0

            if axeman_deficit >= archer_deficit and axeman_deficit >= knight_deficit:
                if self.player.milk >= Axeman.milk_cost and self.player.wood >= Axeman.wood_cost:
                    self.context.production_queues[barracks] = {
                        'unit_type': Axeman,
                        'start_time': context.current_time,
                        'player_id': self.player.player_id
                    }
                    self.player.milk -= Axeman.milk_cost
                    self.player.wood -= Axeman.wood_cost
                    print(f"AI: Queued Axeman production in Barracks at {barracks.pos} (Deficit: Axeman={axeman_deficit}, Archer={archer_deficit}, Knight={knight_deficit})")
            elif archer_deficit >= knight_deficit:
                if self.player.milk >= Archer.milk_cost and self.player.wood >= Archer.wood_cost:
                    self.context.production_queues[barracks] = {
                        'unit_type': Archer,
                        'start_time': context.current_time,
                        'player_id': self.player.player_id
                    }
                    self.player.milk -= Archer.milk_cost
                    self.player.wood -= Archer.wood_cost
                    print(f"AI: Queued Archer production in Barracks at {barracks.pos} (Deficit: Axeman={axeman_deficit}, Archer={archer_deficit}, Knight={knight_deficit})")
            else:
                if self.player.milk >= Knight.milk_cost and self.player.wood >= Knight.wood_cost:
                    self.context.production_queues[barracks] = {
                        'unit_type': Knight,
                        'start_time': context.current_time,
                        'player_id': self.player.player_id
                    }
                    self.player.milk -= Knight.milk_cost
                    self.player.wood -= Knight.wood_cost
                    print(f"AI: Queued Knight production in Barracks at {barracks.pos} (Deficit: Axeman={axeman_deficit}, Archer={archer_deficit}, Knight={knight_deficit})")


