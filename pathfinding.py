from __future__ import annotations

import math
import time
from heapq import heappop, heappush
from typing import Dict, List, Optional, Set, Tuple

import pygame
from pygame.math import Vector2

from constants import *
from world_objects import Bridge, Road, MiscPassable
from tiles import Dirt, GrassTile, Foundation
from units import Axeman, Barn, Building, Cow, TownCenter, Tree, Unit
from world_objects import Road


class WaypointGraph:
    def __init__(self, grass_tiles, all_units, tile_size, map_width, map_height, spatial_grid):
        self.grass_tiles = grass_tiles
        self.all_units = all_units
        self.tile_size = tile_size
        self.map_width = map_width
        self.map_height = map_height
        self.spatial_grid = spatial_grid
        self.cols = len(grass_tiles[0])
        self.rows = len(grass_tiles)
        self.path_cache = {}  # Cache: (start_tile, target_tile, unit_type) -> path
        self.cache_timeout = 0.5  # Increased to reduce recalculations
        self.last_cache_clean = time.time()
        self.max_distance = 50  # Reduced from 30 to limit search space
        self.walkable_cache = {}  # Cache walkable tiles per frame
        self.frame_walkable_cache = {}  # Cache walkable tiles for current frame
        self.frame_count = 0  # Track frame for cache invalidation

    def is_walkable(self, tile_x, tile_y, unit):
        # Check map bounds
        if not (0 <= tile_x < self.cols and 0 <= tile_y < self.rows):
            return False

        # Cache key for performance
        cache_key = (tile_x, tile_y, unit.__class__.__name__, self.frame_count)
        if cache_key in self.frame_walkable_cache:
            return self.frame_walkable_cache[cache_key]

        # Check tile type
        if not isinstance(self.grass_tiles[tile_y][tile_x], (GrassTile, Dirt, Bridge, Road, Foundation, MiscPassable)):
            self.frame_walkable_cache[cache_key] = False
            return False

        # Calculate the position where the unit's center would be if it moves to this tile
        target_pos = Vector2(tile_x * self.tile_size + self.tile_size / 2, tile_y * self.tile_size + self.tile_size / 2)
        unit_rect = pygame.Rect(target_pos.x - unit.size / 2, target_pos.y - unit.size / 2, unit.size, unit.size)

        # Get nearby units for collision checks
        nearby_units = self.spatial_grid.get_nearby_units(unit, radius=self.tile_size * 1.0)

        for other in nearby_units:
            # Skip the target unit/building if it's the unit's intended target
            if unit.target == other or (isinstance(unit, Axeman) and unit.target == other.pos and (isinstance(other, Tree) or isinstance(other, TownCenter))):
                continue
            # Allow Cows to enter Barns
            if isinstance(unit, Cow) and isinstance(other, Barn) and other.player_id == unit.player_id:
                continue
            # Check for collision with Trees or Buildings
            if isinstance(other, (Tree, Building)):
                other_rect = pygame.Rect(other.pos.x - other.size / 2, other.pos.y - other.size / 2, other.size, other.size)
                if unit_rect.colliderect(other_rect):
                    self.frame_walkable_cache[cache_key] = False
                    return False

        self.frame_walkable_cache[cache_key] = True
        return True

    def get_neighbors(self, tile_x, tile_y, unit):
        neighbors = []
        directions = [
            (0, -1, 1.0),  # Up
            (0, 1, 1.0),  # Down
            (-1, 0, 1.0),  # Left
            (1, 0, 1.0),  # Right
            (-1, -1, 1.414),  # Up-Left
            (1, -1, 1.414),  # Up-Right
            (-1, 1, 1.414),  # Down-Left
            (1, 1, 1.414),  # Down-Right
        ]

        for dx, dy, cost in directions:
            nx, ny = tile_x + dx, tile_y + dy
            if not self.is_walkable(nx, ny, unit):
                continue

            # Prevent diagonal corner cutting:
            if dx != 0 and dy != 0:
                if not (self.is_walkable(tile_x + dx, tile_y, unit) and
                        self.is_walkable(tile_x, tile_y + dy, unit)):
                    continue

            neighbors.append((nx, ny, cost))

        return neighbors

    def manhattan_distance(self, tile1, tile2):
        """Calculate Manhattan distance (heuristic for A*)."""
        return abs(tile1[0] - tile2[0]) + abs(tile1[1] - tile2[1])

    def euclidean_distance(self, tile1, tile2):
        """Calculate Euclidean distance for path cost and validation."""
        return math.sqrt((tile2[0] - tile1[0]) ** 2 + (tile2[1] - tile1[1]) ** 2)

    def get_path(self, start_pos, target_pos, unit):
        self.frame_count += 1
        if self.frame_count % 60 == 0:
            self.frame_walkable_cache = {}

        current_time = time.time()
        if current_time - self.last_cache_clean > self.cache_timeout:
            self.path_cache = {k: v for k, v in self.path_cache.items() if current_time - v['time'] < self.cache_timeout}
            self.last_cache_clean = current_time

        start_tile = (int(start_pos.x // self.tile_size), int(start_pos.y // self.tile_size))
        target_tile = (int(target_pos.x // self.tile_size), int(target_pos.y // self.tile_size))
        cache_key = (start_tile, target_tile, unit.__class__.__name__)

        if cache_key in self.path_cache and current_time - self.path_cache[cache_key]['time'] < self.cache_timeout:
            path = self.path_cache[cache_key]['path']
            # If targeting a Unit, replace the final point with the target's exact position
            if isinstance(unit.target, Unit) and not isinstance(unit.target, Tree) and path:
                path[-1] = Vector2(target_pos)
            return path

        if not self.is_walkable(start_tile[0], start_tile[1], unit):
            return []
        if not (0 <= target_tile[0] < self.cols and 0 <= target_tile[1] < self.rows):
            return []

        open_set = [(0, start_tile, 0)]  # (f_score, tile, g_score)
        came_from = {}
        g_scores = {start_tile: 0}
        f_scores = {start_tile: self.manhattan_distance(start_tile, target_tile)}
        closest_tile = start_tile
        closest_distance = self.euclidean_distance(start_tile, target_tile)

        while open_set:
            current_f, current, current_g = heappop(open_set)
            if current == target_tile:
                closest_tile = current
                break
            if current_g > self.max_distance:
                break
            for neighbor_x, neighbor_y, move_cost in self.get_neighbors(current[0], current[1], unit):
                neighbor = (neighbor_x, neighbor_y)
                tentative_g = g_scores[current] + move_cost
                if tentative_g < g_scores.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_scores[neighbor] = tentative_g
                    f_score = tentative_g + self.manhattan_distance(neighbor, target_tile) * 1.01
                    f_scores[neighbor] = f_score
                    heappush(open_set, (f_score, neighbor, tentative_g))
                    dist = self.euclidean_distance(neighbor, target_tile)
                    if dist < closest_distance:
                        closest_distance = dist
                        if dist < self.tile_size / self.tile_size:
                            closest_tile = neighbor
                            break

        path = []
        current = closest_tile
        while current in came_from:
            path.append(Vector2(current[0] * self.tile_size + self.tile_size / 2,
                                current[1] * self.tile_size + self.tile_size / 2))
            current = came_from[current]
        if path or closest_tile == start_tile:
            path.append(Vector2(start_tile[0] * self.tile_size + self.tile_size / 2,
                                start_tile[1] * self.tile_size + self.tile_size / 2))
        path.reverse()

        # If targeting a Unit, append the target's exact position as the final point
        if isinstance(unit.target, Unit) and not isinstance(unit.target, Tree):
            if path:
                path[-1] = Vector2(target_pos)
            else:
                path = [Vector2(start_pos), Vector2(target_pos)]
        # Smooth path while preserving the final point
        if path:
            final_point = path[-1]
            path = self.smooth_path(path[:-1], unit)
            path.append(final_point)

        self.path_cache[cache_key] = {'path': path, 'time': current_time}
        return path

    def smooth_path(self, path, unit):
        """Simplified path smoothing to reduce CPU usage."""
        if len(path) < 2:
            return path
        smoothed_path = [path[0]]
        i = 0
        while i < len(path) - 1:
            current = smoothed_path[-1]
            next_idx = min(i + 2, len(path) - 1)  # Look ahead one or two points
            next_point = path[next_idx]
            if self.is_line_of_sight_clear(current, next_point, unit):
                smoothed_path.append(next_point)
                i = next_idx
            else:
                smoothed_path.append(path[i + 1])
                i += 1
            if len(smoothed_path) > 10:  # Limit smoothing iterations
                smoothed_path.extend(path[i + 1:])
                break
        return smoothed_path

    def is_line_of_sight_clear(self, start, end, unit):
        dist = start.distance_to(end)
        if dist <= 0.001:
            return True

        # --- 1) Tile sampling along the segment (prevents "river/wall" corner cutting) ---
        step = max(8.0, self.tile_size * 0.25)  # sample at least 4 points per tile
        steps = int(dist / step) + 1

        for i in range(steps + 1):
            t = i / steps
            p = start.lerp(end, t)
            tx = int(p.x // self.tile_size)
            ty = int(p.y // self.tile_size)
            if not self.is_walkable(tx, ty, unit):
                return False

        # --- 2) Keep your rectangle intersection for big obstacles (extra safety) ---
        nearby_units = self.spatial_grid.get_nearby_units(unit, radius=dist + self.tile_size)
        for other in nearby_units:
            if isinstance(other, Tree) or (
                    isinstance(other, Building) and not (isinstance(unit, Cow) and isinstance(other, Barn))
            ):
                other_rect = pygame.Rect(other.pos.x - other.size / 2, other.pos.y - other.size / 2, other.size, other.size)
                if self.line_intersects_rect(start, end, other_rect):
                    return False

        return True

    def line_intersects_rect(self, start, end, rect):
        """Check if a line intersects a rectangle (unchanged for performance)."""
        def line_intersects_segment(p1, p2, s1, s2):
            def ccw(A, B, C):
                return (C.y - A.y) * (B.x - A.x) > (B.y - A.y) * (C.x - A.x)
            return ccw(p1, s1, s2) != ccw(p2, s1, s2) and ccw(p1, p2, s1) != ccw(p1, p2, s2)

        corners = [
            Vector2(rect.left, rect.top),
            Vector2(rect.right, rect.top),
            Vector2(rect.right, rect.bottom),
            Vector2(rect.left, rect.bottom)
        ]
        for i in range(4):
            if line_intersects_segment(start, end, corners[i], corners[(i + 1) % 4]):
                return True
        return False

# SpatialGrid class
class SpatialGrid:
    def __init__(self, cell_size, map_width, map_height):
        self.cell_size = cell_size
        self.cols = int(math.ceil(map_width / cell_size))
        self.rows = int(math.ceil(map_height / cell_size))
        self.grid = [[[] for _ in range(self.cols)] for _ in range(self.rows)]
        self.unit_cells = {}

    def add_unit(self, unit):
        if unit in self.unit_cells:
            for row, col in self.unit_cells[unit]:
                if unit in self.grid[row][col]:
                    self.grid[row][col].remove(unit)
        half_size = unit.size / 2
        min_col = max(0, int((unit.pos.x - half_size) / self.cell_size))
        max_col = min(self.cols - 1, int((unit.pos.x + half_size) / self.cell_size))
        min_row = max(0, int((unit.pos.y - half_size) / self.cell_size))
        max_row = min(self.rows - 1, int((unit.pos.y + half_size) / self.cell_size))
        cells = []
        for row in range(min_row, max_row + 1):
            for col in range(min_col, max_col + 1):
                self.grid[row][col].append(unit)
                cells.append((row, col))
        self.unit_cells[unit] = cells

    def get_nearby_units(self, unit, radius=None):
        if radius is None:
            radius = self.cell_size + unit.size
        half_size = unit.size / 2
        min_col = max(0, int((unit.pos.x - half_size - radius) / self.cell_size))
        max_col = min(self.cols - 1, int((unit.pos.x + half_size + radius) / self.cell_size))
        min_row = max(0, int((unit.pos.y - half_size - radius) / self.cell_size))
        max_row = min(self.rows - 1, int((unit.pos.y + half_size + radius) / self.cell_size))
        nearby = set()
        for row in range(min_row, max_row + 1):
            for col in range(min_col, max_col + 1):
                nearby.update(self.grid[row][col])
        if unit in nearby:
            nearby.remove(unit)
        return nearby

    def remove_unit(self, unit):
        if unit in self.unit_cells:
            for row, col in self.unit_cells[unit]:
                if unit in self.grid[row][col]:
                    self.grid[row][col].remove(unit)
            del self.unit_cells[unit]

# SimpleTile base class
