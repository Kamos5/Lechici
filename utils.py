import pygame
from pygame import Vector2

from constants import TILE_SIZE, GRASS_ROWS, GRASS_COLS
from context import grass_tiles
from tiles import GrassTile, River, Dirt
from units import Building, Tree


def is_tile_occupied(row, col, all_units,grass_tiles):
    if not (0 <= row < GRASS_ROWS and 0 <= col < GRASS_COLS):
        return True
    if isinstance(grass_tiles[row][col], River):
        return True
    for unit in all_units:
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


def find_valid_spawn_tiles(building, units, grass_tiles, tile_size, radius=2):
    """Find valid, unoccupied tiles around a building for spawning units."""
    valid_tiles = []
    center_tile_x = int(building.pos.x // tile_size)
    center_tile_y = int(building.pos.y // tile_size)
    building_size_tiles = int(building.size // tile_size)

    # Define search area around the building (slightly outside its footprint)
    for dr in range(-radius - building_size_tiles // 2, radius + building_size_tiles // 2 + 1):
        for dc in range(-radius - building_size_tiles // 2, radius + building_size_tiles // 2 + 1):
            tile_x = center_tile_x + dc
            tile_y = center_tile_y + dr
            # Check if tile is within map bounds
            if not (0 <= tile_x < GRASS_COLS and 0 <= tile_y < GRASS_ROWS):
                continue
            # Check if tile is walkable (GrassTile, not Dirt, and not occupied)
            tile_rect = pygame.Rect(tile_x * tile_size, tile_y * tile_size, tile_size, tile_size)
            if isinstance(grass_tiles[tile_y][tile_x], GrassTile) and not isinstance(grass_tiles[tile_y][tile_x], Dirt):
                occupied = False
                for unit in units:
                    unit_rect = pygame.Rect(unit.pos.x - unit.size / 2, unit.pos.y - unit.size / 2, unit.size, unit.size)
                    if tile_rect.colliderect(unit_rect):
                        occupied = True
                        break
                if not occupied:
                    valid_tiles.append(Vector2(tile_x * tile_size + tile_size / 2, tile_y * tile_size + tile_size / 2))

    return valid_tiles