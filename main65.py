import pygame
import sys
import random
import math
from pygame.math import Vector2
from heapq import heappush, heappop
import time

# Initialize Pygame
pygame.init()

# Screen settings
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 1000
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Lechites")
clock = pygame.time.Clock()

# Colors
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
PURPLE = (128, 0, 128)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
BROWN = (75, 25, 0)
GREEN = (0, 255, 0)
GRASS_GREEN = (0, 100, 0)
GRASS_BROWN = (139, 69, 19)
TREE_GREEN = (0, 50, 0)
DARK_GRAY = (50, 50, 50, 128)
GRAY = (128, 128, 128, 128)
TOWN_CENTER_GRAY = (100, 100, 100, 128)
BLACK = (0, 0, 0)
LIGHT_GRAY = (150, 150, 150)
HIGHLIGHT_GRAY = (200, 200, 200)
BORDER_OUTER = (50, 50, 50)
BORDER_INNER = (200, 200, 200)
PANEL_COLOR = (100, 100, 100)
ORANGE = (255, 165, 0)

# Tile settings
TILE_SIZE = 20
TILE_HALF = TILE_SIZE // 2
TILE_QUARTER = TILE_SIZE // 4
GRASS_ROWS = 60
GRASS_COLS = 60
MAP_WIDTH = GRASS_COLS * TILE_SIZE
MAP_HEIGHT = GRASS_ROWS * TILE_SIZE

# UI settings
VIEW_MARGIN_LEFT = 10
VIEW_MARGIN_RIGHT = 10
VIEW_MARGIN_TOP = 40
VIEW_MARGIN_BOTTOM = 0
PANEL_HEIGHT = 150
VIEW_WIDTH = SCREEN_WIDTH - (VIEW_MARGIN_LEFT + VIEW_MARGIN_RIGHT)
VIEW_HEIGHT = SCREEN_HEIGHT - (VIEW_MARGIN_TOP + VIEW_MARGIN_BOTTOM + PANEL_HEIGHT)
PANEL_Y = SCREEN_HEIGHT - PANEL_HEIGHT - VIEW_MARGIN_BOTTOM
VIEW_BOUNDS_X = VIEW_MARGIN_LEFT + VIEW_WIDTH
VIEW_BOUNDS_Y = VIEW_MARGIN_TOP + VIEW_HEIGHT

# Button settings
BUTTON_WIDTH = 100
BUTTON_HEIGHT = 30
BUTTON_MARGIN = 5
BUTTON_PLAYER0_POS = (VIEW_MARGIN_LEFT + VIEW_WIDTH - BUTTON_WIDTH - 10, VIEW_MARGIN_TOP + 40)
BUTTON_PLAYER1_POS = (VIEW_MARGIN_LEFT + VIEW_WIDTH - BUTTON_WIDTH - 10, VIEW_MARGIN_TOP + 40 + (BUTTON_HEIGHT + BUTTON_MARGIN))
BUTTON_PLAYER2_POS = (VIEW_MARGIN_LEFT + VIEW_WIDTH - BUTTON_WIDTH - 10, VIEW_MARGIN_TOP + 40 + 2 * (BUTTON_HEIGHT + BUTTON_MARGIN))

# List of unique names for units
UNIQUE_MALE_NAMES = [
    "Bolesław", "Mieszko", "Władysław", "Kazimierz", "Dobrosław", "Mirosław", "Sławomir", "Wojciech",
    "Stanisław", "Zbigniew", "Bogumił", "Jaromir", "Witosław", "Czesław", "Leszek", "Radosław",
    "Witold", "Ziemowit", "Przemysław", "Bohusław", "Lubomir", "Wacław", "Zdzisław", "Mieczysław",
    "Radomił", "Świętosław", "Bronisław", "Gniewomir", "Siemowit", "Bohdan", "Jarosław", "Krzysztof",
    "Władimir", "Domarad", "Sulimir", "Bezprym", "Sambor", "Rostysław", "Wyszesław", "Bogusław"
]

FEMALE_NAMES = [
    "Milena", "Zofia", "Wanda", "Dobrawa", "Ludmiła", "Bronisława", "Jadwiga", "Elżbieta",
    "Radosława", "Bogumiła", "Stanisława", "Wiesława", "Mirosława", "Sławomira", "Zdzisława",
    "Czesława", "Jaromira", "Witosława", "Przemysława", "Bohusława", "Lubomira", "Wacława",
    "Radomiła", "Świętosława", "Dobrosława", "Gosława", "Mieczysława", "Władysława", "Kazimiera",
    "Ziemowita", "Bogna", "Danuta", "Halina", "Irena", "Krystyna", "Aldona", "Jolanta",
    "Beata", "Agnieszka", "Ewa"
]

# 3x1 Grid Button settings
GRID_BUTTON_ROWS = 3
GRID_BUTTON_COLS = 1
GRID_BUTTON_WIDTH = 110
GRID_BUTTON_HEIGHT = 40
GRID_BUTTON_MARGIN = 5
GRID_BUTTON_START_X = VIEW_BOUNDS_X - GRID_BUTTON_WIDTH - BUTTON_MARGIN
GRID_BUTTON_START_Y = PANEL_Y + (PANEL_HEIGHT - (GRID_BUTTON_ROWS * GRID_BUTTON_HEIGHT + (GRID_BUTTON_ROWS - 1) * GRID_BUTTON_MARGIN)) // 2

# Initialize 3x1 grid of button rectangles
grid_buttons = []
for row in range(GRID_BUTTON_ROWS):
    row_buttons = []
    for col in range(GRID_BUTTON_COLS):
        x = GRID_BUTTON_START_X + col * (GRID_BUTTON_WIDTH + GRID_BUTTON_MARGIN)
        y = GRID_BUTTON_START_Y + row * (GRID_BUTTON_HEIGHT + GRID_BUTTON_MARGIN)
        row_buttons.append(pygame.Rect(x, y, GRID_BUTTON_WIDTH, GRID_BUTTON_HEIGHT))
    grid_buttons.append(row_buttons)

# Camera settings
camera_x = 0
camera_y = 0
SCROLL_SPEED = 10
SCROLL_MARGIN = 20

# Icon settings
ICON_SIZE = 32
ICON_MARGIN = 5

# Load icons with error handling
try:
    wood_icon = pygame.image.load("wood_icon.png").convert_alpha()
    milk_icon = pygame.image.load("milk_icon.png").convert_alpha()
    unit_icon = pygame.image.load("unit_icon.png").convert_alpha() if pygame.image.get_extended() else pygame.Surface((20, 20))
    building_icon = pygame.image.load("building_icon.png").convert_alpha() if pygame.image.get_extended() else pygame.Surface((20, 20))
    wood_icon = pygame.transform.scale(wood_icon, (20, 20))
    milk_icon = pygame.transform.scale(milk_icon, (20, 20))
    unit_icon = pygame.transform.scale(unit_icon, (20, 20))
    building_icon = pygame.transform.scale(building_icon, (20, 20))
except (pygame.error, FileNotFoundError) as e:
    print(f"Failed to load icons: {e}")
    wood_icon = pygame.Surface((20, 20))
    milk_icon = pygame.Surface((20, 20))
    unit_icon = pygame.Surface((20, 20))
    building_icon = pygame.Surface((20, 20))
    wood_icon.fill(BROWN)
    milk_icon.fill(WHITE)
    unit_icon.fill(LIGHT_GRAY)
    building_icon.fill(LIGHT_GRAY)

# Spatial grid settings
GRID_CELL_SIZE = 60

# Move order and highlight tracking
move_order_times = {}
highlight_times = {}
attack_animations = []  # List to store attack animation data (start_pos, end_pos, start_time)

# Production queue and animation tracking
production_queues = {}  # Maps building to {'unit_type': Class, 'start_time': float, 'player_id': int}
building_animations = {}  # Maps building to {'start_time': float, 'alpha': float} for transparency animation

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
        if not isinstance(self.grass_tiles[tile_y][tile_x], (GrassTile, Dirt)):
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
        """Get walkable neighbors (cardinal only for performance, diagonals optional)."""
        neighbors = []
        directions = [
            (0, -1, 1.0),  # Up
            (0, 1, 1.0),  # Down
            (-1, 0, 1.0),  # Left
            (1, 0, 1.0),  # Right
            (-1, -1, 1.414),  # Up-Left
            (1, -1, 1.414),  # Up-Right
            (-1, 1, 1.414),  # Down-Left
            (1, 1, 1.414)  # Down-Right
            # Diagonals commented out to reduce branching; enable if smoother paths needed
            # (-1, -1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (1, 1, 1.414)
        ]
        for dx, dy, cost in directions:
            new_x, new_y = tile_x + dx, tile_y + dy
            if self.is_walkable(new_x, new_y, unit):
                # Simplified: obstacle penalty removed to reduce computation
                neighbors.append((new_x, new_y, cost))
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
        """Check line of sight with reduced radius for performance."""
        nearby_units = self.spatial_grid.get_nearby_units(unit, radius=start.distance_to(end) * 0.75)  # Reduced radius
        for other in nearby_units:
            if isinstance(other, Tree) or (isinstance(other, Building) and not (isinstance(unit, Cow) and isinstance(other, Barn))):
                unit_rect = pygame.Rect(other.pos.x - other.size / 2, other.pos.y - other.size / 2, other.size, other.size)
                if self.line_intersects_rect(start, end, unit_rect):
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
class SimpleTile:
    _image = None

    def __init__(self, x, y):
        self.pos = Vector2(x, y)
        self.center = Vector2(x + TILE_HALF, y + TILE_HALF)

    def draw(self, surface, camera_x, camera_y):
        if (self.pos.x < camera_x - TILE_SIZE or self.pos.x > camera_x + VIEW_WIDTH or
            self.pos.y < camera_y - TILE_SIZE or self.pos.y > camera_y + VIEW_HEIGHT):
            return
        if self._image:
            surface.blit(self._image, (self.pos.x - camera_x, self.pos.y - camera_y))
        else:
            pygame.draw.rect(surface, GRASS_BROWN, (self.pos.x - camera_x, self.pos.y - camera_y, TILE_SIZE, TILE_SIZE))

    def harvest(self, amount):
        return 0.0

    def regrow(self, amount):
        pass

# GrassTile class
class GrassTile(SimpleTile):
    _grass_image = None
    _dirt_image = None
    _surface_cache = {}

    def __init__(self, x, y):
        super().__init__(x, y)
        self.grass_level = 1.0
        if GrassTile._grass_image is None or GrassTile._dirt_image is None:
            GrassTile.load_images()

    @classmethod
    def load_images(cls):
        try:
            cls._grass_image = pygame.image.load("grass.png").convert_alpha()
            cls._grass_image = pygame.transform.scale(cls._grass_image, (TILE_SIZE, TILE_SIZE))
            cls._dirt_image = pygame.image.load("dirt.png").convert_alpha()
            cls._dirt_image = pygame.transform.scale(cls._dirt_image, (TILE_SIZE, TILE_SIZE))
        except (pygame.error, FileNotFoundError) as e:
            print(f"Failed to load grass.png or dirt.png: {e}")
            cls._grass_image = None
            cls._dirt_image = None

    def draw(self, surface, camera_x, camera_y):
        if (self.pos.x < camera_x - TILE_SIZE or self.pos.x > camera_x + VIEW_WIDTH or
            self.pos.y < camera_y - TILE_SIZE or self.pos.y > camera_y + VIEW_HEIGHT):
            return
        if self._grass_image and self._dirt_image:
            level_key = round(self.grass_level * 100)
            if level_key not in self._surface_cache:
                blended_surface = pygame.Surface((TILE_SIZE, TILE_SIZE), pygame.SRCALPHA)
                blended_surface.blit(self._grass_image, (0, 0))
                dirt_alpha = int((1.0 - self.grass_level) * 255)
                dirt_overlay = self._dirt_image.copy()
                dirt_overlay.set_alpha(dirt_alpha)
                blended_surface.blit(dirt_overlay, (0, 0))
                self._surface_cache[level_key] = blended_surface
            surface.blit(self._surface_cache[level_key], (self.pos.x - camera_x, self.pos.y - camera_y))
        else:
            color = (
                int(GRASS_GREEN[0] * self.grass_level + GRASS_BROWN[0] * (1 - self.grass_level)),
                int(GRASS_GREEN[1] * self.grass_level + GRASS_BROWN[1] * (1 - self.grass_level)),
                int(GRASS_GREEN[2] * self.grass_level + GRASS_BROWN[2] * (1 - self.grass_level))
            )
            pygame.draw.rect(surface, color, (self.pos.x - camera_x, self.pos.y - camera_y, TILE_SIZE, TILE_SIZE))

    def harvest(self, amount):
        old_level = self.grass_level
        self.grass_level = max(0.0, self.grass_level - amount)
        return old_level - self.grass_level

    def regrow(self, amount):
        if self.grass_level < 1.0:
            self.grass_level = min(1.0, self.grass_level + amount)

# Dirt class
class Dirt(SimpleTile):
    _image = None

    def __init__(self, x, y):
        super().__init__(x, y)
        if Dirt._image is None:
            Dirt.load_image()

    @classmethod
    def load_image(cls):
        try:
            cls._image = pygame.image.load(f"{cls.__name__.lower()}.png").convert_alpha()
            cls._image = pygame.transform.scale(cls._image, (TILE_SIZE, TILE_SIZE))
        except (pygame.error, FileNotFoundError) as e:
            print(f"Failed to load {cls.__name__.lower()}.png: {e}")
            cls._image = None

# Unit class
class Unit:
    _images = {}
    _unit_icons = {}
    milk_cost = 0
    wood_cost = 0
    production_time = 15.0

    def __init__(self, x, y, size, speed, color, player_id, player_color):
        self.pos = Vector2(x, y)
        self.target = None
        self.speed = speed
        self.selected = False
        self.size = size
        self.min_distance = self.size
        self.color = color
        self.player_id = player_id
        self.player_color = player_color
        self.velocity = Vector2(0, 0)
        self.damping = 0.95
        self.hp = 50
        self.max_hp = 50
        self.mana = 0
        self.special = 0
        self.attack_damage = 0
        self.attack_range = 0
        self.attack_cooldown = 1.0
        self.last_attack_time = 0
        self.armor = 0
        self.name = None
        cls_name = self.__class__.__name__
        if cls_name not in Unit._images:
            self.load_images(cls_name, size)
        self.alpha = 255
        self.path = []  # Store pathfinding waypoints
        self.path_index = 0
        self.last_target = None
        self.attackers = []  # List of (attacker, timestamp) tuples
        self.attacker_timeout = 5.0  # Remove attackers after 5 seconds

    @classmethod
    def load_images(cls, cls_name, size):
        try:
            cls._images[cls_name] = pygame.image.load(f"{cls_name.lower()}.png").convert_alpha()
            cls._images[cls_name] = pygame.transform.scale(cls._images[cls_name], (int(size), int(size)))
        except (pygame.error, FileNotFoundError) as e:
            print(f"Failed to load {cls_name.lower()}.png: {e}")
            cls._images[cls_name] = None
        try:
            cls._unit_icons[cls_name] = pygame.image.load(f"{cls_name.lower()}_icon.png").convert_alpha()
            cls._unit_icons[cls_name] = pygame.transform.scale(cls._unit_icons[cls_name], (ICON_SIZE, ICON_SIZE))
        except (pygame.error, FileNotFoundError) as e:
            print(f"Failed to load {cls_name.lower()}_icon.png: {e}")
            cls._unit_icons[cls_name] = None

    def should_highlight(self, current_time):
        return self in highlight_times and current_time - highlight_times[self] <= 0.4

    def draw(self, screen, camera_x, camera_y):
        if (self.pos.x < camera_x - self.size / 2 or self.pos.x > camera_x + VIEW_WIDTH + self.size / 2 or
            self.pos.y < camera_y - self.size / 2 or self.pos.y > camera_y + VIEW_HEIGHT + self.size / 2):
            return
        cls_name = self.__class__.__name__
        image = self._images.get(cls_name)
        x = self.pos.x - camera_x + VIEW_MARGIN_LEFT
        y = self.pos.y - camera_y + VIEW_MARGIN_TOP
        if not image:
            color = GREEN if self.selected else self.color
            surface = pygame.Surface((self.size, self.size), pygame.SRCALPHA)
            surface.fill(color[:3] + (self.alpha,))
            screen.blit(surface, (x - self.size / 2, y - self.size / 2))
        else:
            image_surface = image.copy()
            image_surface.set_alpha(self.alpha)
            image_rect = image_surface.get_rect(center=(int(x), int(y)))
            screen.blit(image_surface, image_rect)
        if self.selected:
            pygame.draw.rect(screen, self.player_color, (x - self.size / 2, y - self.size / 2, self.size, self.size), 1)
        if self.should_highlight(current_time):
            pygame.draw.rect(screen, WHITE, (x - self.size / 2, y - self.size / 2, self.size, self.size), 1)
        if not isinstance(self, Tree):
            bar_width = 16
            bar_height = 4
            bar_offset = 2
            bar_x = x - bar_width / 2
            bar_y = y - self.size / 2 - bar_height - bar_offset
            pygame.draw.rect(screen, BLACK, (bar_x, bar_y, bar_width, bar_height))
            fill_width = (self.hp / self.max_hp) * bar_width
            pygame.draw.rect(screen, self.player_color, (bar_x, bar_y, fill_width, bar_height))

        # Draw pathfinding path
        if self.path and len(self.path[self.path_index:]) >= 2:  # Ensure at least 2 points
            points = [(p.x - camera_x + VIEW_MARGIN_LEFT, p.y - camera_y + VIEW_MARGIN_TOP) for p in self.path[self.path_index:]]
            pygame.draw.lines(screen, WHITE, False, points, 1)

    def move(self, units, spatial_grid=None, waypoint_graph=None):
        if not self.target:
            self.path = []
            self.path_index = 0
            self.velocity = Vector2(0, 0)
            return

        # Determine target position and stop distance
        if isinstance(self.target, Unit) and not isinstance(self.target, Tree):
            if self.target.hp <= 0 or self.target not in units:  # Check if target is dead or removed
                self.target = None
                self.path = []
                self.path_index = 0
                self.last_target = None
                self.velocity = Vector2(0, 0)
                print(f"Target lost for {self.__class__.__name__} at {self.pos}, clearing target")
                return
            target_pos = self.target.pos
            stop_distance = self.attack_range
            # Check if target moved significantly since last path calculation
            if self.path and self.path_index < len(self.path):
                last_waypoint = self.path[-1]
                if (target_pos - last_waypoint).length() > self.size:  # Target moved more than unit size
                    self.path = []  # Force path recalculation
                    self.path_index = 0
        else:
            target_pos = self.target
            stop_distance = self.size / 4 if isinstance(self, Axeman) else self.size / 2

        # Recalculate path if needed
        if (self.target != self.last_target or not self.path or self.path_index >= len(self.path) or
                self.is_path_blocked(units, spatial_grid, waypoint_graph)):
            self.path = waypoint_graph.get_path(self.pos, target_pos, self) if waypoint_graph else []
            self.path_index = 0
            self.last_target = self.target
            if not self.path:
                if self.pos.distance_to(target_pos) < self.size * 2 and self.is_line_of_sight_clear(target_pos, units, spatial_grid):
                    self.path = [self.pos, target_pos]
                else:
                    print(f"No path found for {self.__class__.__name__} from {self.pos} to {target_pos}")
                    self.target = None
                    self.path = []
                    self.path_index = 0
                    self.last_target = None
                    self.velocity = Vector2(0, 0)
                    return

        # Follow the path
        if self.path_index < len(self.path):
            next_point = self.path[self.path_index]
            direction = next_point - self.pos
            distance = direction.length()
            if distance > stop_distance:
                try:
                    self.velocity = direction.normalize() * self.speed
                except ValueError:
                    self.path_index += 1
                    if self.path_index >= len(self.path):
                        if isinstance(self, Axeman) and isinstance(self.target, Vector2):
                            self.velocity = Vector2(0, 0)
                            print(f"Axeman at {self.pos} stopped at path end, distance to {self.target}: {self.pos.distance_to(self.target):.1f}")
                        else:
                            self.target = None
                            self.path = []
                            self.path_index = 0
                            self.last_target = None
                    return
            else:
                self.path_index += 1
                if self.path_index >= len(self.path):
                    if isinstance(self, Axeman) and isinstance(self.target, Vector2):
                        self.velocity = Vector2(0, 0)
                        print(f"Axeman at {self.pos} reached path end, distance to {self.target}: {self.pos.distance_to(self.target):.1f}")
                    elif not isinstance(self.target, Unit):  # Only clear non-Unit targets
                        self.target = None
                        self.path = []
                        self.path_index = 0
                        self.last_target = None
        else:
            self.velocity = Vector2(0, 0)

        # Apply velocity and damping
        self.velocity *= self.damping
        self.pos += self.velocity

    def is_path_blocked(self, units, spatial_grid, waypoint_graph):
        """Check if the current path is blocked, optimized to reduce checks."""
        if not self.path or self.path_index >= len(self.path):
            return False
        # Only check the next few waypoints to reduce overhead
        check_limit = min(self.path_index + 5, len(self.path))
        for point in self.path[self.path_index:check_limit]:
            tile_x = int(point.x // waypoint_graph.tile_size)
            tile_y = int(point.y // waypoint_graph.tile_size)
            if not waypoint_graph.is_walkable(tile_x, tile_y, self):
                return True
        return False

    def is_line_of_sight_clear(self, target_pos, units, spatial_grid):
        """Check if there's a clear line of sight to the target."""
        if not spatial_grid:
            return True
        nearby_units = spatial_grid.get_nearby_units(self, radius=self.pos.distance_to(target_pos))
        for unit in nearby_units:
            if isinstance(unit, Tree) or (isinstance(unit, Building) and not (isinstance(self, Cow) and isinstance(unit, Barn))):
                unit_rect = pygame.Rect(unit.pos.x - unit.size / 2, unit.pos.y - unit.size / 2, unit.size, unit.size)
                if self.line_intersects_rect(self.pos, target_pos, unit_rect):
                    return False
        return True

    def line_intersects_rect(self, start, end, rect):
        """Check if a line from start to end intersects a rectangle."""
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

    def update_attackers(self, attacker, current_time):
        """Add or update an attacker with a timestamp."""
        # Remove expired attackers
        self.attackers = [(a, t) for a, t in self.attackers if current_time - t < self.attacker_timeout]
        # Update or add attacker
        for i, (existing_attacker, _) in enumerate(self.attackers):
            if existing_attacker == attacker:
                self.attackers[i] = (attacker, current_time)
                return
        self.attackers.append((attacker, current_time))

    def get_closest_attacker(self):
        """Return the closest living attacker."""
        if not self.attackers:
            return None
        valid_attackers = [(a, t) for a, t in self.attackers if a.hp > 0 and a in all_units]
        if not valid_attackers:
            return None
        return min(valid_attackers, key=lambda x: self.pos.distance_to(x[0].pos))[0]

    def attack(self, target, current_time):
        if not isinstance(target, Unit) or isinstance(target, Tree) or target.hp <= 0 or target not in all_units:
            return
        distance = (self.pos - target.pos).length()
        max_range = self.attack_range + self.size / 2 + target.size / 2
        if distance <= max_range:
            if current_time - self.last_attack_time >= self.attack_cooldown:
                attack_animations.append({
                    'start_pos': self.pos,
                    'end_pos': target.pos,
                    'color': self.color,
                    'start_time': current_time
                })
                damage = max(0, self.attack_damage - target.armor)
                target.hp -= damage
                self.last_attack_time = current_time
                print(f"{self.__class__.__name__} at {self.pos} attacked {target.__class__.__name__} at {target.pos}, dealing {damage} damage")
                # Notify target of attack for defensive behavior
                if isinstance(target, (Axeman, Archer, Knight)) and target.hp > 0:
                    target.update_attackers(self, current_time)
                    # Trigger counter-attack if no target or autonomous target
                    if not target.target or target.autonomous_target:
                        closest_attacker = target.get_closest_attacker()
                        if closest_attacker:
                            target.target = closest_attacker
                            target.autonomous_target = True
                            target.path = []  # Clear path to recalculate
                            target.path_index = 0
                            print(f"{target.__class__.__name__} at {target.pos} counter-attacking closest attacker {closest_attacker.__class__.__name__} at {closest_attacker.pos}")
        elif isinstance(self, Archer):
            # Ensure Archers recalculate path if out of range
            self.path = []
            self.path_index = 0

    def resolve_collisions(self, units, spatial_grid):
        if isinstance(self, (Building, Tree)):
            return
        corrections = []
        nearby_units = spatial_grid.get_nearby_units(self)
        epsilon = 0.001
        for other in nearby_units:
            if other is not self:
                distance = self.pos.distance_to(other.pos)
                if isinstance(self, Cow) and isinstance(other, Barn):
                    barn_corners = [
                        Vector2(other.pos.x - other.size / 2, other.pos.y - other.size / 2),
                        Vector2(other.pos.x + other.size / 2, other.pos.y - other.size / 2),
                        Vector2(other.pos.x - other.size / 2, other.pos.y + other.size / 2),
                        Vector2(other.pos.x + other.size / 2, other.pos.y + other.size / 2)
                    ]
                    nearest_corner = barn_corners[2]
                    distance = self.pos.distance_to(nearest_corner)
                    corner_radius = 10
                    if distance < corner_radius and distance > epsilon and (self.target is None or self.target != nearest_corner):
                        overlap = corner_radius - distance
                        direction = (self.pos - nearest_corner).normalize()
                        corrections.append(direction * overlap)
                elif isinstance(other, Tree) and other.player_id == 0:
                    combined_min_distance = (self.size + TILE_SIZE) / 2
                    if distance < combined_min_distance * 2 and distance > epsilon:
                        if distance < combined_min_distance:
                            overlap = combined_min_distance - distance
                            direction = (self.pos - other.pos).normalize()
                            corrections.append(direction * overlap)
                elif isinstance(self, Cow) and isinstance(other, Building) and not isinstance(other, Barn):
                    combined_min_distance = (self.size + other.size) / 2
                    if distance < combined_min_distance * 2 and distance > epsilon:
                        if distance < combined_min_distance:
                            overlap = combined_min_distance - distance
                            direction = (self.pos - other.pos).normalize()
                            corrections.append(direction * overlap)
                elif not isinstance(self, Cow) and isinstance(other, Building):
                    combined_min_distance = (self.size + other.size) / 2
                    if distance < combined_min_distance * 2 and distance > epsilon:
                        if distance < combined_min_distance:
                            overlap = combined_min_distance - distance
                            direction = (self.pos - other.pos).normalize()
                            corrections.append(direction * overlap)
                elif not isinstance(self, (Building, Tree)) and not isinstance(other, (Building, Tree)):
                    combined_min_distance = (self.size + other.size) / 2
                    if distance < combined_min_distance * 2 and distance > epsilon:
                        if distance < combined_min_distance:
                            overlap = combined_min_distance - distance
                            direction = (self.pos - other.pos).normalize()
                            correction = direction * overlap * 0.5
                            corrections.append(correction)
                            other_corrections = getattr(other, '_corrections', [])
                            other_corrections.append(-correction)
                            other._corrections = other_corrections
        if corrections:
            self.pos += sum(corrections, Vector2(0, 0))

    def keep_in_bounds(self):
        self.pos.x = max(self.size / 2, min(MAP_WIDTH - self.size / 2, self.pos.x))
        self.pos.y = max(self.size / 2, min(MAP_HEIGHT - self.size / 2, self.pos.y))

    def harvest_grass(self, grass_tiles):
        pass

    def is_clicked(self, click_pos, camera_x, camera_y):
        adjusted_pos = Vector2(click_pos.x + camera_x, click_pos.y + camera_y)
        return (abs(adjusted_pos.x - self.pos.x) <= self.size / 2 and
                abs(adjusted_pos.y - self.pos.y) <= self.size / 2)

# Tree class
class Tree(Unit):
    _image = None
    _tinted_images = {}
    _selected = False
    _last_color_change_time = 0
    _color_index = 0
    _colors = [RED, GREEN, BLUE, YELLOW]

    def __init__(self, x, y, size, color, player_id, player_color):
        super().__init__(x, y, size=TILE_SIZE, speed=0, color=color, player_id=player_id, player_color=player_color)
        self.hp = 180
        self.max_hp = 180
        self.attack_damage = 0  # Trees cannot attack
        self.attack_range = 0
        self.armor = 0  # Trees not affected by combat damage
        if Tree._image is None:
            Tree.load_image()
        self.pos.x = x
        self.pos.y = y

    @classmethod
    def load_image(cls):
        try:
            cls._image = pygame.image.load("tree.png").convert_alpha()
            scale_factor = min(TILE_SIZE / cls._image.get_width(), TILE_SIZE / cls._image.get_height())
            new_size = (int(cls._image.get_width() * scale_factor), int(cls._image.get_height() * scale_factor))
            cls._image = pygame.transform.scale(cls._image, new_size)
            for i, color in enumerate(cls._colors):
                tinted_image = cls._image.copy()
                mask_surface = pygame.Surface(tinted_image.get_size(), pygame.SRCALPHA)
                mask_surface.fill(color + (128,))
                tinted_image.blit(mask_surface, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
                cls._tinted_images[i] = tinted_image
        except (pygame.error, FileNotFoundError) as e:
            print(f"Failed to load tree.png: {e}")
            cls._image = None

    def draw(self, screen, camera_x, camera_y, axemen_targets):
        if (self.pos.x < camera_x - TILE_SIZE or self.pos.x > camera_x + VIEW_WIDTH + TILE_SIZE or
            self.pos.y < camera_y - TILE_SIZE or self.pos.y > camera_y + VIEW_HEIGHT + TILE_SIZE):
            return
        x = self.pos.x - camera_x + VIEW_MARGIN_LEFT
        y = self.pos.y - camera_y + VIEW_MARGIN_TOP
        if self._image:
            is_targeted = any(self.pos.distance_to(axeman.pos) <= TILE_SIZE and axeman.target == self.pos for axeman in axemen_targets)
            if is_targeted:
                if current_time - self._last_color_change_time >= 1:
                    self._color_index = (self._color_index + 1) % len(self._colors)
                    self._last_color_change_time = current_time
                image = self._tinted_images[self._color_index]
            else:
                image = self._image
            image_rect = image.get_rect(center=(int(x), int(y)))
            screen.blit(image, image_rect)
            if self.should_highlight(current_time):
                pygame.draw.rect(screen, WHITE, (x - TILE_HALF, y - TILE_HALF, TILE_SIZE, TILE_SIZE), 1)
            if self._selected:
                pygame.draw.rect(screen, YELLOW, (x - TILE_HALF, y - TILE_HALF, TILE_SIZE, TILE_SIZE), 1)
        else:
            pygame.draw.rect(screen, TREE_GREEN, (x - TILE_HALF, y - TILE_HALF, TILE_SIZE, TILE_SIZE))
            if self.should_highlight(current_time):
                pygame.draw.rect(screen, WHITE, (x - TILE_HALF, y - TILE_HALF, TILE_SIZE, TILE_SIZE), 1)
            if self._selected:
                pygame.draw.rect(screen, YELLOW, (x - TILE_HALF, y - TILE_HALF, TILE_SIZE, TILE_SIZE), 1)

    def is_clicked(self, click_pos, camera_x, camera_y):
        adjusted_pos = Vector2(click_pos.x + camera_x, click_pos.y + camera_y)
        tile_rect = pygame.Rect(self.pos.x - TILE_HALF, self.pos.y - TILE_HALF, TILE_SIZE, TILE_SIZE)
        return tile_rect.collidepoint(adjusted_pos)

    def set_selected(self, selected):
        self._selected = selected

    def move(self, units, spatial_grid=None, waypoint_graph=None):
        pass

    def attack(self, target, current_time):
        return False  # Trees cannot attack

# Building class
class Building(Unit):

    production_time = 30.0  # Production time for buildings (seconds)
    def __init__(self, x, y, size, color, player_id, player_color):
        super().__init__(x, y, size, speed=0, color=color, player_id=player_id, player_color=player_color)
        self.hp = 150  # High HP for buildings
        self.max_hp = 150
        self.armor = 5  # Buildings have armor
        self.attack_damage = 0  # Buildings cannot attack
        self.attack_range = 0
        self.rally_point = None  # Initialize rally point as None

    def move(self, units, spatial_grid=None, waypoint_graph=None):
        pass

# Barn class
class Barn(Building):
    milk_cost = 0
    wood_cost = 300
    def __init__(self, x, y, player_id, player_color):
        super().__init__(x, y, size=60, color=DARK_GRAY, player_id=player_id, player_color=player_color)
        self.harvest_rate = 60.0

# TownCenter class
class TownCenter(Building):
    milk_cost = 0
    wood_cost = 800
    def __init__(self, x, y, player_id, player_color):
        super().__init__(x, y, size=60, color=TOWN_CENTER_GRAY, player_id=player_id, player_color=player_color)
        self.hp = 200  # High HP for buildings
        self.max_hp = 200
        self.armor = 5

# Barracks class
class Barracks(Building):
    milk_cost = 0
    wood_cost = 500
    def __init__(self, x, y, player_id, player_color):
        super().__init__(x, y, size=60, color=TOWN_CENTER_GRAY, player_id=player_id, player_color=player_color)

# Axeman class
class Axeman(Unit):
    milk_cost = 300
    wood_cost = 0

    def __init__(self, x, y, player_id, player_color):
        super().__init__(x, y, size=16, speed=2, color=RED, player_id=player_id, player_color=player_color)
        self.chop_damage = 1
        self.special = 0
        self.return_pos = None
        self.depositing = False
        self.attack_damage = 10
        self.attack_range = 20
        self.attack_cooldown = 1.0
        self.armor = 2

    def move(self, units, spatial_grid=None, waypoint_graph=None):
        self.velocity = Vector2(0, 0)

        # Idle state: No target, remain idle unless commanded
        if not self.target and not self.depositing:
            return  # Do not automatically target trees unless commanded

        # Depositing state: Move to TownCenter to deposit wood
        if self.depositing and self.target:
            target_unit = next((unit for unit in units if isinstance(unit, TownCenter) and unit.pos == self.target and unit.player_id == self.player_id), None)
            if not target_unit:
                # print(f"Axeman at {self.pos} failed to find TownCenter at {self.target}")
                self.target = None
                self.depositing = False
                self.special = 0
                return
            if self.pos.distance_to(self.target) <= self.size + target_unit.size / 2:
                for player in players:
                    if player.player_id == self.player_id:
                        over_limit = max(0, player.unit_count - player.unit_limit) if player.unit_limit is not None else 0
                        multiplier = max(0.0, 1.0 - (0.1 * over_limit))
                        wood_deposited = 100 * multiplier
                        player.wood = min(player.max_wood, player.wood + wood_deposited)
                        # print(f"Axeman at {self.pos} deposited {wood_deposited:.1f} wood for Player {self.player_id}, player wood now {player.wood}")
                        break
                self.special = 0
                self.depositing = False
                self.target = self.return_pos
                # self.return_pos = None  # Clear return_pos to prepare for next cycle
                # print(f"Axeman at {self.pos} returning to {self.target}")
            else:
                # print(f"Axeman at {self.pos} moving to TownCenter at {self.target}, distance: {self.pos.distance_to(self.target):.1f}")
                super().move(units, spatial_grid, waypoint_graph)
            return

        # Returning state: Move to return_pos and target a new tree when close
        if self.target and self.target == self.return_pos:
            if self.pos.distance_to(self.target) <= TILE_SIZE * 1.5:  # Allow larger distance (30 pixels)
                self.target = None
                # Look for the closest tree and pathfind to it
                nearby_units = spatial_grid.get_nearby_units(self, radius=1000)
                trees = [unit for unit in nearby_units if isinstance(unit, Tree) and unit.player_id == 0]
                if trees:
                    closest_tree = min(trees, key=lambda tree: self.pos.distance_to(tree.pos))
                    self.target = closest_tree.pos
                    self.return_pos = None
                    # Force path recalculation to ensure proper pathfinding
                    self.path = waypoint_graph.get_path(self.pos, self.target, self) if waypoint_graph else [self.pos, self.target]
                    self.path_index = 0
                    # print(f"Axeman at {self.pos} near return position, targeting new tree at {self.target}, path: {self.path}")
                else:
                    # print(f"Axeman at {self.pos} near return position, no trees found nearby")
                    self.return_pos = None
            else:
                # print(f"Axeman at {self.pos} moving to return position at {self.target}, distance: {self.pos.distance_to(self.target):.1f}")
                super().move(units, spatial_grid, waypoint_graph)
            return

        # Chopping state: Move to tree or stop to chop
        target_tree = next((unit for unit in units if isinstance(unit, Tree) and unit.player_id == 0 and unit.pos == self.target), None)
        if target_tree and self.pos.distance_to(self.target) <= TILE_SIZE:
            self.velocity = Vector2(0, 0)  # Stop to chop
            # print(f"Axeman at {self.pos} stopped to chop tree at {self.target}")
        else:
            super().move(units, spatial_grid, waypoint_graph)

    def chop_tree(self, trees):
        if self.special > 0 or self.depositing or self.target == self.return_pos:
            return
        target_tree = next((tree for tree in trees if isinstance(tree, Tree) and tree.player_id == 0 and self.target == tree.pos and self.pos.distance_to(tree.pos) <= TILE_SIZE), None)
        if target_tree:
            target_tree.hp -= self.chop_damage
            # print(f"Axeman at {self.pos} chopping tree at {target_tree.pos}, tree HP: {target_tree.hp}")
            if target_tree.hp <= 0:
                self.return_pos = Vector2(self.pos)
                players[0].remove_unit(target_tree)
                all_units.remove(target_tree)
                spatial_grid.remove_unit(target_tree)
                self.special += 100
                print(f"Tree at {target_tree.pos} chopped down by Axeman at {self.pos}, special = {self.special}")
                # Set TownCenter as target for depositing
                town_centers = [unit for unit in all_units if isinstance(unit, TownCenter) and unit.player_id == self.player_id and unit.alpha == 255]
                if town_centers:
                    closest_town = min(town_centers, key=lambda town: self.pos.distance_to(town.pos))
                    self.target = closest_town.pos
                    self.depositing = True
                    # print(f"Axeman at {self.pos} targeting TownCenter at {self.target} for wood deposit")
                else:
                    # print(f"Axeman at {self.pos} found no TownCenter, clearing state")
                    self.special = 0
                    self.depositing = False
                    self.target = None
                    self.return_pos = None

# Knight class
class Knight(Unit):
    milk_cost = 500
    wood_cost = 400

    def __init__(self, x, y, player_id, player_color):
        super().__init__(x, y, size=16, speed=4, color=BLUE, player_id=player_id, player_color=player_color)
        self.attack_damage = 17
        self.attack_range = 20
        self.attack_cooldown = 1.0
        self.armor = 5

# Archer class
class Archer(Unit):
    milk_cost = 400
    wood_cost = 200

    def __init__(self, x, y, player_id, player_color):
        super().__init__(x, y, size=16, speed=3, color=YELLOW, player_id=player_id, player_color=player_color)
        self.attack_damage = 15
        self.attack_range = 100
        self.attack_cooldown = 1.5
        self.armor = 0

# Cow class
class Cow(Unit):
    milk_cost = 400
    wood_cost = 0
    returning = False
    return_pos = None

    def __init__(self, x, y, player_id, player_color):
        super().__init__(x, y, size=16, speed=4, color=BROWN, player_id=player_id, player_color=player_color)
        self.harvest_rate = 0.01
        self.assigned_corner = None
        self.autonomous_target = False
        self.attack_damage = 5
        self.attack_range = 20
        self.attack_cooldown = 2.0
        self.armor = 1

    def draw(self, screen, camera_x, camera_y):
        if (self.pos.x < camera_x - self.size / 2 or self.pos.x > camera_x + VIEW_WIDTH + self.size / 2 or
            self.pos.y < camera_y - self.size / 2 or self.pos.y > camera_y + VIEW_HEIGHT + self.size / 2):
            return
        cls_name = self.__class__.__name__
        image = self._images.get(cls_name)
        x = self.pos.x - camera_x + VIEW_MARGIN_LEFT
        y = self.pos.y - camera_y + VIEW_MARGIN_TOP
        if not image:
            color = GREEN if self.selected else self.color
            pygame.draw.rect(screen, color, (x - self.size / 2, y - self.size / 2, self.size, self.size))
        else:
            image_rect = image.get_rect(center=(int(x), int(y)))
            screen.blit(image, image_rect)
        if self.selected:
            pygame.draw.rect(screen, self.player_color, (x - self.size / 2, y - self.size / 2, self.size, self.size), 1)
        if self.should_highlight(current_time):
            pygame.draw.rect(screen, WHITE, (x - self.size / 2, y - self.size / 2, self.size, self.size), 1)
        bar_width = 16
        bar_height = 4
        bar_offset = 2
        bar_x = x - bar_width / 2
        bar_y = y - self.size / 2 - bar_height - bar_offset
        pygame.draw.rect(screen, BLACK, (bar_x, bar_y, bar_width, bar_height))
        fill_width = (self.hp / self.max_hp) * bar_width
        pygame.draw.rect(screen, self.player_color, (bar_x, bar_y, fill_width, bar_height))
        fill_width = (self.special / 100) * bar_width
        pygame.draw.rect(screen, WHITE, (bar_x, bar_y + bar_height + 1, fill_width, bar_height))

    def move(self, units, spatial_grid=None, waypoint_graph=None):
        self.velocity = Vector2(0, 0)
        if not self.target:
            return
        target_pos = self.target
        stop_distance = self.size / 2  # Use size/2 for grass tiles
        # Snap target to tile center for grass tiles
        if isinstance(self.target, Vector2):
            target_tile_x = int(self.target.x // TILE_SIZE)
            target_tile_y = int(self.target.y // TILE_SIZE)
            target_pos = Vector2(target_tile_x * TILE_SIZE + TILE_HALF, target_tile_y * TILE_SIZE + TILE_HALF)
            self.target = target_pos
        # Recalculate path if target changed or path is blocked
        if (self.target != self.last_target or not self.path or self.path_index >= len(self.path) or
                self.is_path_blocked(units, spatial_grid, waypoint_graph)):
            self.path = waypoint_graph.get_path(self.pos, target_pos, self) if waypoint_graph else []
            self.path_index = 0
            self.last_target = self.target
            if not self.path:
                if self.pos.distance_to(target_pos) < self.size * 1.5 and self.is_line_of_sight_clear(target_pos, units, spatial_grid):
                    self.path = [self.pos, target_pos]
                else:
                    print(f"No path found for Cow from {self.pos} to {target_pos}")
                    tile_x = int(target_pos.x // TILE_SIZE)
                    tile_y = int(target_pos.y // TILE_SIZE)
                    adjacent_tiles = [
                        (tile_x, tile_y - 1), (tile_x, tile_y + 1), (tile_x - 1, tile_y), (tile_x + 1, tile_y)
                    ]
                    for adj_x, adj_y in adjacent_tiles:
                        if (0 <= adj_x < GRASS_COLS and 0 <= adj_y < GRASS_ROWS and
                                waypoint_graph.is_walkable(adj_x, adj_y, self)):
                            adj_pos = Vector2(adj_x * TILE_SIZE + TILE_HALF, adj_y * TILE_SIZE + TILE_HALF)
                            self.path = [self.pos, adj_pos]
                            print(f"Retrying path to adjacent tile {adj_pos} for Cow")
                            break
                    else:
                        self.target = None
                        return
        if self.path_index < len(self.path):
            next_point = self.path[self.path_index]
            direction = next_point - self.pos
            distance = direction.length()
            if distance > stop_distance:
                try:
                    self.velocity = direction.normalize() * self.speed
                except ValueError:
                    self.path_index += 1
                    if self.path_index >= len(self.path):
                        self.target = None
                        self.path = []
                        self.path_index = 0
                        self.last_target = None
                    return
            else:
                self.path_index += 1
                if self.path_index >= len(self.path):
                    # Snap to tile center when reaching the target
                    self.pos = Vector2(int(self.pos.x // TILE_SIZE) * TILE_SIZE + TILE_HALF,
                                       int(self.pos.y // TILE_SIZE) * TILE_SIZE + TILE_HALF)
                    self.target = None
                    self.path = []
                    self.path_index = 0
                    self.last_target = None
        self.velocity *= self.damping
        self.pos += self.velocity

    def is_in_barn(self, barn):
        return (isinstance(barn, Barn) and
                barn.pos.x - barn.size / 2 <= self.pos.x <= barn.pos.x + barn.size / 2 and
                barn.pos.y - barn.size / 2 <= self.pos.y <= barn.pos.y + barn.size / 2)

    def is_in_barn_any(self, barns):
        return any(self.is_in_barn(barn) for barn in barns)

    def is_tile_walkable(self, tile_x, tile_y, spatial_grid):
        tile_rect = pygame.Rect(tile_x * TILE_SIZE, tile_y * TILE_SIZE, TILE_SIZE, TILE_SIZE)
        nearby_units = spatial_grid.get_nearby_units(self, radius=TILE_SIZE)
        for unit in nearby_units:
            if isinstance(unit, Tree) or (isinstance(unit, Building) and not isinstance(unit, Barn)):
                unit_rect = pygame.Rect(unit.pos.x - unit.size / 2, unit.pos.y - unit.size / 2, unit.size, unit.size)
                if tile_rect.colliderect(unit_rect):
                    return False
        return True

    def harvest_grass(self, grass_tiles, barns, cow_in_barn, player, spatial_grid):
        if self.is_in_barn_any(barns) and self.special > 0:
            return
        if self.special >= 100 and not self.target:
            available_barns = [barn for barn in barns if barn.player_id == self.player_id and (barn not in cow_in_barn or cow_in_barn[barn] is None)]
            if available_barns:
                closest_barn = min(available_barns, key=lambda barn: self.pos.distance_to(barn.pos))
                target_x = closest_barn.pos.x - closest_barn.size / 2 + TILE_HALF
                target_y = closest_barn.pos.y + closest_barn.size / 2 - TILE_HALF
                self.target = Vector2(target_x, target_y)
                self.return_pos = Vector2(self.pos)
                self.assigned_corner = self.target
                self.autonomous_target = True
                print(f"Cow at {self.pos} targeting barn at {self.target}")
            return
        if self.special < 100 and not self.target and not self.velocity.length() > 0.5:
            tile_x = int(self.pos.x // TILE_SIZE)
            tile_y = int(self.pos.y // TILE_SIZE)
            if 0 <= tile_x < GRASS_COLS and 0 <= tile_y < GRASS_ROWS:
                tile = grass_tiles[tile_y][tile_x]
                if isinstance(tile, GrassTile) and not isinstance(tile, Dirt):
                    nearby_units = spatial_grid.get_nearby_units(self, radius=TILE_SIZE)
                    has_tree = any(isinstance(unit, Tree) and unit.player_id == 0 and
                                   unit.pos.x - TILE_HALF <= self.pos.x <= unit.pos.x + TILE_HALF and
                                   unit.pos.y - TILE_HALF <= self.pos.y <= unit.pos.y + TILE_HALF
                                   for unit in nearby_units)
                    if not has_tree:
                        harvested = tile.harvest(self.harvest_rate)
                        self.special = min(100, self.special + harvested * 50)
                        if tile.grass_level == 0:
                            adjacent_tiles = [
                                (tile_x, tile_y - 1), (tile_x, tile_y + 1), (tile_x - 1, tile_y), (tile_x + 1, tile_y),
                                (tile_x - 1, tile_y - 1), (tile_x + 1, tile_y - 1), (tile_x - 1, tile_y + 1), (tile_x + 1, tile_y + 1)
                            ]
                            random.shuffle(adjacent_tiles)
                            for adj_x, adj_y in adjacent_tiles:
                                if (0 <= adj_x < GRASS_COLS and 0 <= adj_y < GRASS_ROWS and
                                    isinstance(grass_tiles[adj_y][adj_x], GrassTile) and
                                    not isinstance(grass_tiles[adj_y][adj_x], Dirt) and
                                    grass_tiles[adj_y][adj_x].grass_level > 0.5 and
                                    self.is_tile_walkable(adj_x, adj_y, spatial_grid)):
                                    self.target = grass_tiles[adj_y][adj_x].center
                                    self.return_pos = Vector2(self.pos)
                                    self.autonomous_target = True
                                    print(f"Cow at {self.pos} targeting adjacent grass tile at {self.target}")
                                    break

# Player class
class Player:
    def __init__(self, player_id, color, start_x, start_y):
        self.player_id = player_id
        self.color = color
        self.milk = 500.0
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
        self.used_male_names = set()  # Track used male names for Axeman, Archer, Knight
        self.used_female_names = set()  # Track used female names for Cow
        offset_x = start_x
        offset_y = start_y
        if self.player_id > 0:
            initial_units = [
                Axeman(offset_x + 100, offset_y + 100, player_id, self.color),
                Knight(offset_x + 150, offset_y + 100, player_id, self.color),
                Archer(offset_x + 200, offset_y + 100, player_id, self.color),
                Cow(offset_x + 260, offset_y + 100, player_id, self.color),
                Barn(offset_x + 230, offset_y + 150, player_id, self.color),
                TownCenter(offset_x + 150, offset_y + 150, player_id, self.color),
                Barracks(offset_x + 70, offset_y + 150, player_id, self.color),
            ]
            for unit in initial_units:
                self.add_unit(unit)
            self.unit_count = len([unit for unit in self.units if not isinstance(unit, Building)])
            self.building_count = len([unit for unit in self.units if isinstance(unit, Building)])
            self.barns = [unit for unit in self.units if isinstance(unit, Barn)]
            print(f"Player {self.player_id} initialized with {self.unit_count}/{self.unit_limit} units and {self.building_count}/{self.building_limit} buildings")

    def add_unit(self, unit):
        self.units.append(unit)
        # Assign a unique name to Axeman, Archer, Knight, or Cow
        if isinstance(unit, (Axeman, Archer, Knight, Cow)) and unit.name is None:
            if isinstance(unit, Cow):
                # Assign female name to Cow
                available_names = [name for name in FEMALE_NAMES if name not in self.used_female_names]
                if available_names:
                    unit.name = random.choice(available_names)
                    self.used_female_names.add(unit.name)
                    print(f"Assigned female name {unit.name} to Cow for Player {self.player_id}")
                else:
                    unit.name = f"Cow_{len(self.used_female_names) + 1}"
                    self.used_female_names.add(unit.name)
                    print(f"No female names available, assigned {unit.name} to Cow for Player {self.player_id}")
            else:
                # Assign male name to Axeman, Archer, Knight
                available_names = [name for name in UNIQUE_MALE_NAMES if name not in self.used_male_names]
                if available_names:
                    unit.name = random.choice(available_names)
                    self.used_male_names.add(unit.name)
                    print(f"Assigned male name {unit.name} to {unit.__class__.__name__} for Player {self.player_id}")
                else:
                    unit.name = f"{unit.__class__.__name__}_{len(self.used_male_names) + 1}"
                    self.used_male_names.add(unit.name)
                    print(f"No male names available, assigned {unit.name} to {unit.__class__.__name__} for Player {self.player_id}")
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
                elif isinstance(unit, (Axeman, Archer, Knight)) and unit.name in self.used_male_names:
                    self.used_male_names.remove(unit.name)
                    print(f"Removed male name {unit.name} from Player {self.player_id}")
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
        self.grass_tiles = grass_tiles
        self.spatial_grid = spatial_grid
        self.all_units = all_units
        self.production_queues = production_queues
        self.last_decision_time = 0
        self.decision_interval = 5.0  # Make decisions every 5 seconds
        self.building_size = 60
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

    def find_valid_building_position(self, center_pos):
        # Unchanged
        center_tile_x = int(center_pos.x // TILE_SIZE)
        center_tile_y = int(center_pos.y // TILE_SIZE)
        search_radius = 5
        for dr in range(-search_radius, search_radius + 1):
            for dc in range(-search_radius, search_radius + 1):
                tile_x = center_tile_x + dc
                tile_y = center_tile_y + dr
                valid = True
                for row in range(tile_y - self.building_size_tiles // 2, tile_y + self.building_size_tiles // 2 + 1):
                    for col in range(tile_x - self.building_size_tiles // 2, tile_x + self.building_size_tiles // 2 + 1):
                        if not (0 <= row < GRASS_ROWS and 0 <= col < GRASS_COLS) or is_tile_occupied(row, col, self.all_units):
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

    def initiate_wave(self, current_time, waypoint_graph):
        """Initiate attack waves based on time elapsed."""
        if self.wave_start_time == 0:
            self.wave_start_time = current_time
        elapsed_time = current_time - self.wave_start_time
        target_player = next((p for p in players if p.player_id == self.target_player_id), None)
        if not target_player:
            return

        # Find Player 1's TownCenter as the camp center
        if not self.target_camp_pos:
            town_centers = [u for u in target_player.units if isinstance(u, TownCenter)]
            if town_centers:
                self.target_camp_pos = town_centers[0].pos
            else:
                self.target_camp_pos = Vector2(150, 150)

        # Wave 1: After 30 seconds, 1 Archer
        if self.wave_number == 0 and elapsed_time >= 30:
            archers = [u for u in self.player.units if isinstance(u, Archer) and u not in self.attack_units]
            if archers:
                archer = random.choice(archers)
                self.attack_units.append(archer)
                archer.target = self.target_camp_pos
                archer.autonomous_target = False
                archer.path = waypoint_graph.get_path(archer.pos, self.target_camp_pos, archer)
                archer.path_index = 0
                print(f"AI Wave 1: Archer at {archer.pos} sent to Player 1 camp at {self.target_camp_pos}")
                self.wave_number = 1
                self.last_decision_time = current_time

        # Wave 2: After 90 seconds, 2 random military units
        elif self.wave_number == 1 and elapsed_time >= 90:
            military_units = [u for u in self.player.units if isinstance(u, (Axeman, Archer, Knight)) and u not in self.attack_units]
            if len(military_units) >= 2:
                selected_units = random.sample(military_units, 2)
                for unit in selected_units:
                    self.attack_units.append(unit)
                    if isinstance(unit, Axeman):
                        unit.target = self.target_camp_pos
                        unit.depositing = False
                        unit.special = 0
                        unit.return_pos = None
                        print(f"AI Wave 2: Axeman at {unit.pos} stopped chopping, targeting Player 1 camp at {self.target_camp_pos}")
                    else:
                        unit.target = self.target_camp_pos
                    unit.autonomous_target = False
                    unit.path = waypoint_graph.get_path(unit.pos, self.target_camp_pos, unit)
                    unit.path_index = 0
                    print(f"AI Wave 2: {unit.__class__.__name__} at {unit.pos} sent to Player 1 camp at {self.target_camp_pos}")
                self.wave_number = 2
                self.last_decision_time = current_time

        # Wave 3: After 150 seconds, 5–10 random military units
        elif self.wave_number == 2 and elapsed_time >= 150:
            military_units = [u for u in self.player.units if isinstance(u, (Axeman, Archer, Knight)) and u not in self.attack_units]
            num_units = min(random.randint(5, 10), len(military_units))
            if num_units > 0:
                selected_units = random.sample(military_units, num_units)
                for unit in selected_units:
                    self.attack_units.append(unit)
                    if isinstance(unit, Axeman):
                        unit.target = self.target_camp_pos
                        unit.depositing = False
                        unit.special = 0
                        unit.return_pos = None
                        print(f"AI Wave 3: Axeman at {unit.pos} stopped chopping, targeting Player 1 camp at {self.target_camp_pos}")
                    else:
                        unit.target = self.target_camp_pos
                    unit.autonomous_target = False
                    unit.path = waypoint_graph.get_path(unit.pos, self.target_camp_pos, unit)
                    unit.path_index = 0
                    print(f"AI Wave 3: {unit.__class__.__name__} at {unit.pos} sent to Player 1 camp at {self.target_camp_pos}")
                self.wave_number = 3
                self.last_decision_time = current_time

    def update_attack_units(self, current_time, waypoint_graph, spatial_grid, all_units, grass_tiles, production_queues):
        """Update attack units and detect nearby enemies within 6 tiles."""
        if current_time - self.last_detection_time < self.detection_interval:
            return
        self.last_detection_time = current_time

        target_player = next((p for p in players if p.player_id == self.target_player_id), None)
        if not target_player:
            return

        # Update existing attack units
        enemy_units = [u for u in all_units if u.player_id == self.target_player_id and not isinstance(u, Tree) and u.hp > 0 and u.alpha == 255]
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
                    unit.path = waypoint_graph.get_path(unit.pos, closest_enemy.pos, unit)
                    unit.path_index = 0
                    print(f"AI: {unit.__class__.__name__} at {unit.pos} targeting enemy {closest_enemy.__class__.__name__} at {closest_enemy.pos}")
                else:
                    # Return to camp if no enemies
                    unit.target = self.target_camp_pos
                    unit.autonomous_target = True
                    unit.path = waypoint_graph.get_path(unit.pos, self.target_camp_pos, unit)
                    unit.path_index = 0
                    print(f"AI: {unit.__class__.__name__} at {unit.pos} returning to Player 1 camp at {self.target_camp_pos}")

        # Detect nearby enemies for idle military units
        military_units = [u for u in self.player.units if isinstance(u, (Axeman, Archer, Knight)) and not u.target and u not in self.attack_units]
        for unit in military_units:
            nearby_enemies = spatial_grid.get_nearby_units(unit, radius=self.detection_range)
            nearby_enemies = [e for e in nearby_enemies if e in enemy_units and e.hp > 0]
            if nearby_enemies:
                closest_enemy = self.find_closest_enemy(unit, nearby_enemies)
                if closest_enemy:
                    unit.target = closest_enemy
                    unit.autonomous_target = True
                    unit.path = waypoint_graph.get_path(unit.pos, closest_enemy.pos, unit)
                    unit.path_index = 0
                    print(f"AI: {unit.__class__.__name__} at {unit.pos} detected and targeting Player 1 {closest_enemy.__class__.__name__} at {closest_enemy.pos} (distance: {unit.pos.distance_to(closest_enemy.pos):.1f})")

    def update(self, current_time, waypoint_graph, spatial_grid, all_units, grass_tiles, production_queues):
        """Update AI decisions including attack waves and detection."""
        if current_time - self.last_decision_time < self.decision_interval:
            return
        self.last_decision_time = current_time

        # Initiate attack waves
        self.initiate_wave(current_time, waypoint_graph)

        # Manage resources
        self.manage_cows()
        self.manage_axemen()

        # Build structures and units
        self.manage_buildings(current_time)
        self.manage_unit_production(current_time)

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
                        isinstance(self.grass_tiles[adj_y][adj_x], GrassTile) and
                        not isinstance(self.grass_tiles[adj_y][adj_x], Dirt) and
                        self.grass_tiles[adj_y][adj_x].grass_level > 0.5 and
                        cow.is_tile_walkable(adj_x, adj_y, self.spatial_grid)):
                        cow.target = self.grass_tiles[adj_y][adj_x].center
                        cow.autonomous_target = True
                        print(f"AI: Cow at {cow.pos} assigned to grass tile at {cow.target}")
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
                    print(f"AI: Cow at {cow.pos} assigned to barn at {cow.target} for milk deposit")

    def manage_axemen(self):
        # Existing method unchanged
        trees = [u for u in self.all_units if isinstance(u, Tree) and u.player_id == 0]
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
        """Manage building construction with current_time passed."""
        town_centers = [u for u in self.player.units if isinstance(u, TownCenter) and u.alpha == 255 and u not in self.production_queues]
        if not town_centers:
            return
        town_center = random.choice(town_centers)
        barn_count = len([u for u in self.player.units if isinstance(u, Barn)])
        if (barn_count < self.max_barns and
                self.player.milk >= Barn.milk_cost and self.player.wood >= Barn.wood_cost and
                (self.player.building_limit is None or self.player.building_count < self.player.building_limit)):
            pos = self.find_valid_building_position(town_center.pos)
            if pos:
                new_building = Barn(pos.x, pos.y, self.player.player_id, self.player.color)
                new_building.alpha = 0
                self.player.add_unit(new_building)
                self.all_units.add(new_building)
                self.spatial_grid.add_unit(new_building)
                self.player.milk -= Barn.milk_cost
                self.player.wood -= Barn.wood_cost
                for row in range(int(pos.y // TILE_SIZE) - self.building_size_tiles // 2, int(pos.y // TILE_SIZE) + self.building_size_tiles // 2 + 1):
                    for col in range(int(pos.x // TILE_SIZE) - self.building_size_tiles // 2, int(pos.x // TILE_SIZE) + self.building_size_tiles // 2 + 1):
                        if 0 <= row < GRASS_ROWS and 0 <= col < GRASS_COLS:
                            self.grass_tiles[row][col] = Dirt(col * TILE_SIZE, row * TILE_SIZE)
                self.production_queues[town_center] = {
                    'unit_type': Barn,
                    'start_time': current_time,
                    'player_id': self.player.player_id
                }
                building_animations[new_building] = {
                    'start_time': current_time,
                    'alpha': 0,
                    'town_center': town_center
                }
                print(f"AI: Placed Barn at {pos} for Player {self.player.player_id} (Barn count: {barn_count + 1}/{self.max_barns})")
        elif (self.player.milk >= Barracks.milk_cost and self.player.wood >= Barracks.wood_cost and
              (self.player.building_limit is None or self.player.building_count < self.player.building_limit)):
            pos = self.find_valid_building_position(town_center.pos)
            if pos:
                new_building = Barracks(pos.x, pos.y, self.player.player_id, self.player.color)
                new_building.alpha = 0
                self.player.add_unit(new_building)
                self.all_units.add(new_building)
                self.spatial_grid.add_unit(new_building)
                self.player.milk -= Barracks.milk_cost
                self.player.wood -= Barracks.wood_cost
                for row in range(int(pos.y // TILE_SIZE) - self.building_size_tiles // 2, int(pos.y // TILE_SIZE) + self.building_size_tiles // 2 + 1):
                    for col in range(int(pos.x // TILE_SIZE) - self.building_size_tiles // 2, int(pos.x // TILE_SIZE) + self.building_size_tiles // 2 + 1):
                        if 0 <= row < GRASS_ROWS and 0 <= col < GRASS_COLS:
                            self.grass_tiles[row][col] = Dirt(col * TILE_SIZE, row * TILE_SIZE)
                self.production_queues[town_center] = {
                    'unit_type': Barracks,
                    'start_time': current_time,
                    'player_id': self.player.player_id
                }
                building_animations[new_building] = {
                    'start_time': current_time,
                    'alpha': 0,
                    'town_center': town_center
                }
                print(f"AI: Placed Barracks at {pos} for Player {self.player.player_id}")

    def manage_unit_production(self, current_time):
        """Manage unit production with current_time passed."""
        barns = [u for u in self.player.units if isinstance(u, Barn) and u.alpha == 255 and u not in self.production_queues]
        barracks = [u for u in self.player.units if isinstance(u, Barracks) and u.alpha == 255 and u not in self.production_queues]
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
            self.production_queues[barn] = {
                'unit_type': Cow,
                'start_time': current_time,
                'player_id': self.player.player_id
            }
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
                    self.production_queues[barracks] = {
                        'unit_type': Axeman,
                        'start_time': current_time,
                        'player_id': self.player.player_id
                    }
                    print(f"AI: Queued Axeman production in Barracks at {barracks.pos} to meet chopping quota (Chopping Axemen: {chopping_axemen}/2)")
                    return

            axeman_deficit = desired_axemen - axeman_count if total_combat_units > 0 else 1
            archer_deficit = desired_archers - archer_count if total_combat_units > 0 else 1
            knight_deficit = desired_knights - knight_count if total_combat_units > 0 else 0

            if axeman_deficit >= archer_deficit and axeman_deficit >= knight_deficit:
                if self.player.milk >= Axeman.milk_cost and self.player.wood >= Axeman.wood_cost:
                    self.production_queues[barracks] = {
                        'unit_type': Axeman,
                        'start_time': current_time,
                        'player_id': self.player.player_id
                    }
                    print(f"AI: Queued Axeman production in Barracks at {barracks.pos} (Deficit: Axeman={axeman_deficit}, Archer={archer_deficit}, Knight={knight_deficit})")
            elif archer_deficit >= knight_deficit:
                if self.player.milk >= Archer.milk_cost and self.player.wood >= Archer.wood_cost:
                    self.production_queues[barracks] = {
                        'unit_type': Archer,
                        'start_time': current_time,
                        'player_id': self.player.player_id
                    }
                    print(f"AI: Queued Archer production in Barracks at {barracks.pos} (Deficit: Axeman={axeman_deficit}, Archer={archer_deficit}, Knight={knight_deficit})")
            else:
                if self.player.milk >= Knight.milk_cost and self.player.wood >= Knight.wood_cost:
                    self.production_queues[barracks] = {
                        'unit_type': Knight,
                        'start_time': current_time,
                        'player_id': self.player.player_id
                    }
                    print(f"AI: Queued Knight production in Barracks at {barracks.pos} (Deficit: Axeman={axeman_deficit}, Archer={archer_deficit}, Knight={knight_deficit})")

# Create grass field
grass_tiles = [[GrassTile(col * TILE_SIZE, row * TILE_SIZE) for col in range(GRASS_COLS)] for row in range(GRASS_ROWS)]
needs_regrowth = set()

# Create players
players = [
    Player(0, GRAY, 0, 0),
    Player(1, BLUE, 0, 0),
    Player(2, PURPLE, 0, 600)
]

# Combine all units
all_units = set()
for player in players:
    all_units.update(player.units)

# Place Dirt tiles under buildings
for unit in all_units:
    if isinstance(unit, Building):
        buildings_cols = range(int((unit.pos.x - unit.size / 2) // TILE_SIZE), int(math.ceil((unit.pos.x + unit.size / 2) / TILE_SIZE)))
        buildings_rows = range(int((unit.pos.y - unit.size / 2) // TILE_SIZE), int(math.ceil((unit.pos.y + unit.size / 2) / TILE_SIZE)))
        for row in buildings_rows:
            for col in buildings_cols:
                if 0 <= row < GRASS_ROWS and 0 <= col < GRASS_COLS:
                    grass_tiles[row][col] = Dirt(col * TILE_SIZE, row * TILE_SIZE)

def is_tile_occupied(row, col, units):
    tile_rect = pygame.Rect(col * TILE_SIZE, row * TILE_SIZE, TILE_SIZE, TILE_SIZE)
    for unit in units:
        unit_rect = pygame.Rect(unit.pos.x - unit.size / 2, unit.pos.y - unit.size / 2, unit.size, unit.size)
        if tile_rect.colliderect(unit_rect):
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

# Create tree objects
tree_objects = []
total_tiles = GRASS_ROWS * GRASS_COLS
target_tree_count = int(total_tiles * 0.25)
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
            tree = Tree(c * TILE_SIZE + TILE_HALF, r * TILE_SIZE + TILE_HALF, TILE_SIZE, GRAY, 0, GRAY)
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
player2_ai = PlayerAI(players[2], grass_tiles, spatial_grid, all_units, production_queues)

# Selection rectangle and player selection mode
selection_start = None
selection_end = None
selecting = False
current_player = None

# Button rectangles
button_player0 = pygame.Rect(BUTTON_PLAYER0_POS[0], BUTTON_PLAYER0_POS[1], BUTTON_WIDTH, BUTTON_HEIGHT)
button_player1 = pygame.Rect(BUTTON_PLAYER1_POS[0], BUTTON_PLAYER1_POS[1], BUTTON_WIDTH, BUTTON_HEIGHT)
button_player2 = pygame.Rect(BUTTON_PLAYER2_POS[0], BUTTON_PLAYER2_POS[1], BUTTON_WIDTH, BUTTON_HEIGHT)

# Precompute constants
regrowth_rate = 0.001

# Fonts
font = pygame.font.SysFont(None, 24)
small_font = pygame.font.SysFont(None, 16)
button_font = pygame.font.SysFont(None, 20)

# --- SPLIT POINT (End of Part 1) ---
# The game loop and remaining code will continue in Part 2.

# --- Part 2: Continuation from Part 1 ---

# Game loop
running = True
tile_surface = pygame.Surface((VIEW_WIDTH, VIEW_HEIGHT), pygame.SRCALPHA)
placing_building = False
building_to_place = None
building_pos = None
building_size = 60

while running:
    current_time = pygame.time.get_ticks() / 1000
    dt = clock.get_time() / 1000  # Delta time for frame-rate independent updates

    # Update camera
    mouse_pos_screen = pygame.mouse.get_pos()
    if mouse_pos_screen[0] < SCROLL_MARGIN:
        camera_x = max(0, camera_x - SCROLL_SPEED)
    if mouse_pos_screen[0] > SCREEN_WIDTH - SCROLL_MARGIN:
        camera_x = min(MAP_WIDTH - VIEW_WIDTH, camera_x + SCROLL_SPEED)
    if mouse_pos_screen[1] < SCROLL_MARGIN:
        camera_y = max(0, camera_y - SCROLL_SPEED)
    if mouse_pos_screen[1] > SCREEN_HEIGHT - SCROLL_MARGIN:
        camera_y = min(MAP_HEIGHT - VIEW_HEIGHT, camera_y + SCROLL_SPEED)

    # Check for building selection (only fully constructed buildings for production)
    barn_selected = any(isinstance(unit, Barn) and unit.selected and unit.alpha == 255 for unit in (current_player.units if current_player else []))
    barracks_selected = any(isinstance(unit, Barracks) and unit.selected and unit.alpha == 255 for unit in (current_player.units if current_player else []))
    town_center_selected = any(isinstance(unit, TownCenter) and unit.selected and unit.alpha == 255 for unit in (current_player.units if current_player else []))

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
                # Find valid spawn tiles around the building
                valid_spawn_tiles = find_valid_spawn_tiles(building, all_units, grass_tiles, TILE_SIZE, radius=2)
                if valid_spawn_tiles:
                    # Pick a random valid tile
                    spawn_pos = random.choice(valid_spawn_tiles)
                    new_unit = queue['unit_type'](spawn_pos.x, spawn_pos.y, player.player_id, player.color)
                    if building.rally_point:  # Set the unit's target to the rally point
                        new_unit.target = Vector2(building.rally_point)
                        print(f"Set rally point {new_unit.target} for {new_unit.__class__.__name__} spawned at {new_unit.pos}")
                    units_to_spawn.append((new_unit, player))
                    print(f"Queued {new_unit.__class__.__name__} to spawn at random tile {spawn_pos} for Player {player.player_id}")
                else:
                    print(f"No valid spawn tiles found around {building.__class__.__name__} at {building.pos}, delaying production")
                    continue  # Skip spawning if no valid tiles are found
            del production_queues[building]
            print(f"Production complete for {queue['unit_type'].__name__} at {building.pos} for Player {player.player_id}")

    # Spawn completed units
    for unit, player in units_to_spawn:
        player.add_unit(unit)
        all_units.add(unit)
        spatial_grid.add_unit(unit)
        player.milk -= unit.milk_cost
        player.wood -= unit.wood_cost
        player.barns = [u for u in player.units if isinstance(u, Barn) and u.alpha == 255]
        highlight_times[unit] = current_time
        print(f"Spawned {unit.__class__.__name__} at {unit.pos} for Player {player.player_id}")

    # Update building animations (construction fade)
    for building, anim in list(building_animations.items()):
        if building not in all_units:
            del building_animations[building]
            continue
        elapsed = current_time - anim['start_time']
        if elapsed >= building.production_time:
            building.alpha = 255  # Fully opaque
            player = next(p for p in players if p.player_id == building.player_id)
            # Only increment if not previously counted
            if building not in player.units or building.alpha < 255:
                player.building_count += 1
            if isinstance(building, Barn):
                player.barns = [u for u in player.units if isinstance(u, Barn) and u.alpha == 255]
            del building_animations[building]
        else:
            building.alpha = int(255 * (elapsed / building.production_time))  # Fade from 0 to 255

    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                mouse_pos = Vector2(event.pos)
                # Check player selection buttons
                if button_player0.collidepoint(event.pos):
                    print("Player 0 button clicked")
                    current_player = players[0]
                    for player in players:
                        player.deselect_all_units()
                    placing_building = False
                    building_to_place = None
                    building_pos = None
                elif button_player1.collidepoint(event.pos):
                    print("Player 1 button clicked")
                    current_player = players[1]
                    for player in players:
                        player.deselect_all_units()
                    placing_building = False
                    building_to_place = None
                    building_pos = None
                elif button_player2.collidepoint(event.pos):
                    print("Player 2 button clicked")
                    current_player = players[2]
                    for player in players:
                        player.deselect_all_units()
                    placing_building = False
                    building_to_place = None
                    building_pos = None
                # Check grid buttons for spawning units or initiating building placement
                elif current_player:
                    grid_button_clicked = False
                    selected_barn = next((unit for unit in current_player.units if isinstance(unit, Barn) and unit.selected and unit.alpha == 255), None)
                    selected_barracks = next((unit for unit in current_player.units if isinstance(unit, Barracks) and unit.selected and unit.alpha == 255), None)
                    selected_town_center = next((unit for unit in current_player.units if isinstance(unit, TownCenter) and unit.selected and unit.alpha == 255), None)
                    if barn_selected and grid_buttons[0][0].collidepoint(event.pos) and selected_barn:
                        if selected_barn in production_queues:
                            print(f"Barn at {selected_barn.pos} is already producing")
                        elif (current_player.milk >= Cow.milk_cost and current_player.wood >= Cow.wood_cost and
                              (current_player.unit_limit is None or current_player.unit_count < current_player.unit_limit)):
                            production_queues[selected_barn] = {
                                'unit_type': Cow,
                                'start_time': current_time,
                                'player_id': current_player.player_id
                            }
                            grid_button_clicked = True
                            print(f"Started production of Cow at Barn {selected_barn.pos} for Player {current_player.player_id}")
                        else:
                            print(f"Cannot queue Cow: milk={current_player.milk}/{Cow.milk_cost}, wood={current_player.wood}/{Cow.wood_cost}, units={current_player.unit_count}/{current_player.unit_limit}")
                    elif barracks_selected and selected_barracks:
                        if selected_barracks in production_queues:
                            print(f"Barracks at {selected_barracks.pos} is already producing")
                        elif grid_buttons[0][0].collidepoint(event.pos):
                            if (current_player.milk >= Axeman.milk_cost and current_player.wood >= Axeman.wood_cost and
                                (current_player.unit_limit is None or current_player.unit_count < current_player.unit_limit)):
                                production_queues[selected_barracks] = {
                                    'unit_type': Axeman,
                                    'start_time': current_time,
                                    'player_id': current_player.player_id
                                }
                                grid_button_clicked = True
                                print(f"Started production of Axeman at Barracks {selected_barracks.pos} for Player {current_player.player_id}")
                            else:
                                print(f"Cannot queue Axeman: milk={current_player.milk}/{Axeman.milk_cost}, wood={current_player.wood}/{Axeman.wood_cost}, units={current_player.unit_count}/{current_player.unit_limit}")
                        elif grid_buttons[1][0].collidepoint(event.pos):
                            if (current_player.milk >= Archer.milk_cost and current_player.wood >= Archer.wood_cost and
                                (current_player.unit_limit is None or current_player.unit_count < current_player.unit_limit)):
                                production_queues[selected_barracks] = {
                                    'unit_type': Archer,
                                    'start_time': current_time,
                                    'player_id': current_player.player_id
                                }
                                grid_button_clicked = True
                                print(f"Started production of Archer at Barracks {selected_barracks.pos} for Player {current_player.player_id}")
                            else:
                                print(f"Cannot queue Archer: milk={current_player.milk}/{Archer.milk_cost}, wood={current_player.wood}/{Archer.wood_cost}, units={current_player.unit_count}/{current_player.unit_limit}")
                        elif grid_buttons[2][0].collidepoint(event.pos):
                            if (current_player.milk >= Knight.milk_cost and current_player.wood >= Knight.wood_cost and
                                (current_player.unit_limit is None or current_player.unit_count < current_player.unit_limit)):
                                production_queues[selected_barracks] = {
                                    'unit_type': Knight,
                                    'start_time': current_time,
                                    'player_id': current_player.player_id
                                }
                                grid_button_clicked = True
                                print(f"Started production of Knight at Barracks {selected_barracks.pos} for Player {current_player.player_id}")
                            else:
                                print(f"Cannot queue Knight: milk={current_player.milk}/{Knight.milk_cost}, wood={current_player.wood}/{Knight.wood_cost}, units={current_player.unit_count}/{current_player.unit_limit}")
                    elif town_center_selected and selected_town_center:
                        if grid_buttons[0][0].collidepoint(event.pos):
                            if (current_player.milk >= Barn.milk_cost and current_player.wood >= Barn.wood_cost and
                                (current_player.building_limit is None or current_player.building_count < current_player.building_limit) and
                                selected_town_center not in production_queues):
                                placing_building = True
                                building_to_place = Barn
                                grid_button_clicked = True
                                print(f"Initiated placement of Barn for Player {current_player.player_id}")
                            else:
                                print(f"Cannot place Barn: milk={current_player.milk}/{Barn.milk_cost}, wood={current_player.wood}/{Barn.wood_cost}, buildings={current_player.building_count}/{current_player.building_limit}, producing={selected_town_center in production_queues}")
                        elif grid_buttons[1][0].collidepoint(event.pos):
                            if (current_player.milk >= Barracks.milk_cost and current_player.wood >= Barracks.wood_cost and
                                (current_player.building_limit is None or current_player.building_count < current_player.building_limit) and
                                selected_town_center not in production_queues):
                                placing_building = True
                                building_to_place = Barracks
                                grid_button_clicked = True
                                print(f"Initiated placement of Barracks for Player {current_player.player_id}")
                            else:
                                print(f"Cannot place Barracks: milk={current_player.milk}/{Barracks.milk_cost}, wood={current_player.wood}/{Barracks.wood_cost}, buildings={current_player.building_count}/{current_player.building_limit}, producing={selected_town_center in production_queues}")
                        elif grid_buttons[2][0].collidepoint(event.pos):
                            if (current_player.milk >= TownCenter.milk_cost and current_player.wood >= TownCenter.wood_cost and
                                (current_player.building_limit is None or current_player.building_count < current_player.building_limit) and
                                selected_town_center not in production_queues):
                                placing_building = True
                                building_to_place = TownCenter
                                grid_button_clicked = True
                                print(f"Initiated placement of TownCenter for Player {current_player.player_id}")
                            else:
                                print(f"Cannot place TownCenter: milk={current_player.milk}/{TownCenter.milk_cost}, wood={current_player.wood}/{TownCenter.wood_cost}, buildings={current_player.building_count}/{current_player.building_limit}, producing={selected_town_center in production_queues}")
                    # Check unit icon clicks
                    if not grid_button_clicked and PANEL_Y <= mouse_pos.y <= PANEL_Y + ICON_SIZE + 10:
                        selected_units = [unit for unit in current_player.units if unit.selected]
                        icon_x = VIEW_MARGIN_LEFT + 10
                        icon_y = PANEL_Y + 10
                        for i, unit in enumerate(selected_units):
                            icon_rect = pygame.Rect(icon_x + i * (ICON_SIZE + ICON_MARGIN), icon_y, ICON_SIZE, ICON_SIZE)
                            if icon_rect.collidepoint(mouse_pos):
                                current_player.deselect_all_units()
                                unit.selected = True
                                selecting = False
                                selection_start = None
                                selection_end = None
                                print(f"Selected unit {unit.__class__.__name__} at {unit.pos} via icon click")
                                break
                    # Handle game view clicks for unit selection or building placement
                    if not grid_button_clicked and VIEW_MARGIN_LEFT <= mouse_pos.x <= VIEW_BOUNDS_X and VIEW_MARGIN_TOP <= mouse_pos.y <= VIEW_BOUNDS_Y:
                        click_pos = Vector2(mouse_pos.x - VIEW_MARGIN_LEFT + camera_x, mouse_pos.y - VIEW_MARGIN_TOP + camera_y)
                        if placing_building:
                            # Snap to the center of the tile under the mouse
                            tile_x = int(click_pos.x // TILE_SIZE)
                            tile_y = int(click_pos.y // TILE_SIZE)
                            building_pos = Vector2(tile_x * TILE_SIZE + TILE_HALF, tile_y * TILE_SIZE + TILE_HALF)
                            building_size_tiles = int(building_size // TILE_SIZE)
                            valid_placement = True
                            for row in range(tile_y - building_size_tiles // 2, tile_y + building_size_tiles // 2 + 1):
                                for col in range(tile_x - building_size_tiles // 2, tile_x + building_size_tiles // 2 + 1):
                                    if not (0 <= row < GRASS_ROWS and 0 <= col < GRASS_COLS) or is_tile_occupied(row, col, all_units):
                                        valid_placement = False
                                        break
                                if not valid_placement:
                                    break
                            if valid_placement:
                                new_building = building_to_place(building_pos.x, building_pos.y, current_player.player_id, current_player.color)
                                new_building.alpha = 0  # Start fully transparent
                                current_player.add_unit(new_building)
                                all_units.add(new_building)
                                spatial_grid.add_unit(new_building)
                                current_player.milk -= building_to_place.milk_cost
                                current_player.wood -= building_to_place.wood_cost
                                for row in range(tile_y - building_size_tiles // 2, tile_y + building_size_tiles // 2 + 1):
                                    for col in range(tile_x - building_size_tiles // 2, tile_x + building_size_tiles // 2 + 1):
                                        if 0 <= row < GRASS_ROWS and 0 <= col < GRASS_COLS:
                                            grass_tiles[row][col] = Dirt(col * TILE_SIZE, row * TILE_SIZE)
                                highlight_times[new_building] = current_time
                                building_animations[new_building] = {
                                    'start_time': current_time,
                                    'alpha': 0,
                                    'town_center': selected_town_center
                                }
                                production_queues[selected_town_center] = {
                                    'unit_type': building_to_place,
                                    'start_time': current_time,
                                    'player_id': current_player.player_id
                                }
                                print(f"Placed {building_to_place.__name__} at {building_pos} for Player {current_player.player_id} with construction fade")
                                placing_building = False
                                building_to_place = None
                                building_pos = None
                            else:
                                print(f"Invalid placement position at {building_pos}: occupied or out of bounds")
                        else:
                            unit_clicked = None
                            if current_player:
                                for unit in current_player.units:
                                    if unit.is_clicked(Vector2(mouse_pos.x - VIEW_MARGIN_LEFT, mouse_pos.y - VIEW_MARGIN_TOP), camera_x, camera_y):
                                        unit_clicked = unit
                                        break
                            if unit_clicked:
                                for player in players:
                                    player.deselect_all_units()
                                unit_clicked.selected = True
                                selecting = False
                                selection_start = None
                                selection_end = None
                                print(f"Selected unit {unit_clicked.__class__.__name__} at {unit_clicked.pos} via map click")
                            else:
                                for player in players:
                                    player.deselect_all_units()
                                selection_start = click_pos
                                selecting = True
            elif event.button == 3:
                mouse_pos = Vector2(event.pos)
                if placing_building:
                    print(f"Building placement canceled for {building_to_place.__name__}")
                    placing_building = False
                    building_to_place = None
                    building_pos = None
                elif VIEW_MARGIN_LEFT <= mouse_pos.x <= VIEW_BOUNDS_X and VIEW_MARGIN_TOP <= mouse_pos.y <= VIEW_BOUNDS_Y:
                    click_pos = Vector2(mouse_pos.x - VIEW_MARGIN_LEFT + camera_x, mouse_pos.y - VIEW_MARGIN_TOP + camera_y)
                    tile_x = int(click_pos.x // TILE_SIZE)
                    tile_y = int(click_pos.y // TILE_SIZE)
                    snapped_pos = (tile_x * TILE_SIZE + TILE_HALF, tile_y * TILE_SIZE + TILE_HALF)
                    clicked_unit = None
                    for unit in all_units:
                        if unit.is_clicked(Vector2(mouse_pos.x - VIEW_MARGIN_LEFT, mouse_pos.y - VIEW_MARGIN_TOP), camera_x, camera_y):
                            clicked_unit = unit
                            break
                    # Check if only a single building is selected and no non-building units
                    selected_building = None
                    selected_non_buildings = 0
                    if current_player:
                        for unit in current_player.units:
                            if unit.selected:
                                if isinstance(unit, Building):
                                    if selected_building is None:
                                        selected_building = unit
                                    else:
                                        selected_building = None  # Multiple buildings selected, disable rally point
                                else:
                                    selected_non_buildings += 1
                    if selected_building and selected_non_buildings == 0 and not clicked_unit:
                        # Set rally point for the selected building
                        selected_building.rally_point = Vector2(snapped_pos)
                        print(f"Set rally point for {selected_building.__class__.__name__} at {selected_building.pos} to {selected_building.rally_point}")
                        move_order_times[snapped_pos] = current_time  # Use move_order_times for visual feedback
                    else:
                        # Existing movement and attack logic
                        for unit in all_units:
                            if unit.selected and not isinstance(unit, (Building, Tree)):
                                clicked_tree = clicked_unit if isinstance(clicked_unit, Tree) and clicked_unit.player_id == 0 else None
                                if clicked_tree and isinstance(unit, Axeman) and unit.special == 0 and not unit.depositing:
                                    unit.target = clicked_tree.pos
                                    unit.autonomous_target = False
                                    print(f"Axeman at {unit.pos} assigned to tree at {unit.target}")
                                    highlight_times[clicked_tree] = current_time
                                elif clicked_unit and not isinstance(clicked_unit, Tree) and (clicked_unit.player_id != unit.player_id or clicked_unit.player_id == 0):
                                    unit.target = clicked_unit
                                    unit.autonomous_target = False
                                    if isinstance(unit, Cow):
                                        unit.returning = False
                                    highlight_times[clicked_unit] = current_time
                                    print(f"Unit {unit.__class__.__name__} at {unit.pos} targeting enemy {clicked_unit.__class__.__name__} at {clicked_unit.pos}")
                                else:
                                    unit.target = Vector2(click_pos.x, click_pos.y)
                                    unit.autonomous_target = False
                                    if isinstance(unit, Cow):
                                        unit.returning = False
                                    move_order_times[snapped_pos] = current_time
                                    print(f"Move order recorded at snapped pos {snapped_pos}")
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1 and selecting and current_player:
                mouse_pos = Vector2(event.pos)
                if VIEW_MARGIN_LEFT <= mouse_pos.x <= VIEW_BOUNDS_X and VIEW_MARGIN_TOP <= mouse_pos.y <= VIEW_BOUNDS_Y:
                    selection_end = Vector2(mouse_pos.x - VIEW_MARGIN_LEFT + camera_x, mouse_pos.y - VIEW_MARGIN_TOP + camera_y)
                    selecting = False
                    for player in players:
                        player.deselect_all_units()
                    for unit in current_player.units:
                        unit.selected = (min(selection_start.x, selection_end.x) <= unit.pos.x <= max(selection_start.x, selection_end.x) and
                                        min(selection_start.y, selection_end.y) <= unit.pos.y <= max(selection_start.y, selection_end.y))
                    selection_start = None
                    selection_end = None
        elif event.type == pygame.MOUSEMOTION:
            if selecting:
                mouse_pos = Vector2(event.pos)
                if VIEW_MARGIN_LEFT <= mouse_pos.x <= VIEW_BOUNDS_X and VIEW_MARGIN_TOP <= mouse_pos.y <= VIEW_BOUNDS_Y:
                    selection_end = Vector2(mouse_pos.x - VIEW_MARGIN_LEFT + camera_x, mouse_pos.y - VIEW_MARGIN_TOP + camera_y)
            if placing_building:
                mouse_pos = Vector2(event.pos)
                if VIEW_MARGIN_LEFT <= mouse_pos.x <= VIEW_BOUNDS_X and VIEW_MARGIN_TOP <= mouse_pos.y <= VIEW_BOUNDS_Y:
                    click_pos = Vector2(mouse_pos.x - VIEW_MARGIN_LEFT + camera_x, mouse_pos.y - VIEW_MARGIN_TOP + camera_y)
                    tile_x = int(click_pos.x // TILE_SIZE)
                    tile_y = int(click_pos.y // TILE_SIZE)
                    building_pos = Vector2(tile_x * TILE_SIZE + TILE_HALF, tile_y * TILE_SIZE + TILE_HALF)

    # Update AI for Player 2
    player2_ai.update(current_time, waypoint_graph, spatial_grid, all_units, grass_tiles, production_queues)
    player2_ai.update_attack_units(current_time, waypoint_graph, spatial_grid, all_units, grass_tiles, production_queues)

    # Update grass regrowth
    for row, col in list(needs_regrowth):
        tile = grass_tiles[row][col]
        tile.regrow(regrowth_rate)
        if tile.grass_level >= 1.0:
            needs_regrowth.remove((row, col))

    # Update grass regrowth
    for row, col in list(needs_regrowth):
        tile = grass_tiles[row][col]
        tile.regrow(regrowth_rate)
        if tile.grass_level >= 1.0:
            needs_regrowth.remove((row, col))

    # Update units
    axemen = [unit for unit in all_units if isinstance(unit, Axeman)]
    units_to_remove = []
    for player in players:
        for unit in player.units:
            unit._corrections = []
            if isinstance(unit, Axeman):
                unit.move(all_units, spatial_grid, waypoint_graph)
            elif isinstance(unit, Cow):
                unit.move(all_units, spatial_grid, waypoint_graph)
            else:
                unit.move(all_units, spatial_grid, waypoint_graph)
            # Handle attacks
            if unit.target and isinstance(unit.target, Unit) and not isinstance(unit.target, Tree) and unit.target in all_units:
                unit.attack(unit.target, current_time)
                if unit.target.hp <= 0:
                    units_to_remove.append(unit.target)
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
            # Cancel TownCenter production queue if building is destroyed during construction
            anim = building_animations[unit]
            town_center = anim.get('town_center')
            if town_center and town_center in production_queues and production_queues[town_center]['unit_type'] == unit.__class__:
                del production_queues[town_center]
                print(f"Canceled TownCenter production queue for {unit.__class__.__name__} due to destruction")
            del building_animations[unit]
        for player in players:
            if unit in player.units:
                player.remove_unit(unit)
                if isinstance(unit, Building) and unit.alpha == 255:
                    player.building_count -= 1
                    if isinstance(unit, Barn):
                        player.barns = [u for u in player.units if isinstance(u, Barn) and u.alpha == 255]
        all_units.discard(unit)
        spatial_grid.remove_unit(unit)
        print(f"Unit {unit.__class__.__name__} at {unit.pos} destroyed")

    # Draw tiles
    tile_surface.fill((0, 0, 0, 0))
    start_col = max(0, int(camera_x // TILE_SIZE))
    end_col = min(GRASS_COLS, int((camera_x + VIEW_WIDTH + TILE_SIZE) // TILE_SIZE))
    start_row = max(0, int(camera_y // TILE_SIZE))
    end_row = min(GRASS_ROWS, int((camera_y + VIEW_HEIGHT + TILE_SIZE) // TILE_SIZE))
    for row in range(start_row, end_row):
        for col in range(start_col, end_col):
            grass_tiles[row][col].draw(tile_surface, camera_x, camera_y)
            tile_pos = (col * TILE_SIZE + TILE_HALF, row * TILE_SIZE + TILE_HALF)
            if tile_pos in move_order_times and current_time - move_order_times[tile_pos] <= 0.5:
                x1 = tile_pos[0] - TILE_QUARTER // 2 - camera_x
                y1 = tile_pos[1] - TILE_QUARTER // 2 - camera_y
                x2 = tile_pos[0] + TILE_QUARTER // 2 - camera_x
                y2 = tile_pos[1] + TILE_QUARTER // 2 - camera_y
                pygame.draw.line(tile_surface, WHITE, (x1, y1), (x2, y2), 2)
                pygame.draw.line(tile_surface, WHITE, (x2, y1), (x1, y2), 2)
    # Draw rally point for the selected building
    if current_player:
        selected_building = next((unit for unit in current_player.units if isinstance(unit, Building) and unit.selected), None)
        if selected_building and selected_building.rally_point:
            rally_pos = (int(selected_building.rally_point.x), int(selected_building.rally_point.y))
            x1 = rally_pos[0] - TILE_QUARTER // 2 - camera_x
            y1 = rally_pos[1] - TILE_QUARTER // 2 - camera_y
            x2 = rally_pos[0] + TILE_QUARTER // 2 - camera_x
            y2 = rally_pos[1] + TILE_QUARTER // 2 - camera_y
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
            unit.draw(screen, camera_x, camera_y)
    for unit in all_units:
        if not isinstance(unit, Building):
            if isinstance(unit, Tree):
                unit.draw(screen, camera_x, camera_y, axemen)
            else:
                unit.draw(screen, camera_x, camera_y)

    # Draw attack animations
    for anim in attack_animations:
        start_x = anim['start_pos'].x - camera_x + VIEW_MARGIN_LEFT
        start_y = anim['start_pos'].y - camera_y + VIEW_MARGIN_TOP
        end_x = anim['end_pos'].x - camera_x + VIEW_MARGIN_LEFT
        end_y = anim['end_pos'].y - camera_y + VIEW_MARGIN_TOP
        pygame.draw.line(screen, anim['color'], (start_x, start_y), (end_x, end_y), 2)

    # Draw building preview during placement
    if placing_building and building_pos:
        temp_building = building_to_place(building_pos.x, building_pos.y, current_player.player_id, current_player.color)
        cls_name = building_to_place.__name__
        image = Unit._images.get(cls_name)
        tile_x = int(building_pos.x // TILE_SIZE)
        tile_y = int(building_pos.y // TILE_SIZE)
        snapped_building_pos = Vector2(tile_x * TILE_SIZE + TILE_HALF, tile_y * TILE_SIZE + TILE_HALF)
        x = snapped_building_pos.x - camera_x + VIEW_MARGIN_LEFT
        y = snapped_building_pos.y - camera_y + VIEW_MARGIN_TOP
        valid_placement = True
        building_size_tiles = int(building_size // TILE_SIZE)
        for row in range(tile_y - building_size_tiles // 2, tile_y + building_size_tiles // 2 + 1):
            for col in range(tile_x - building_size_tiles // 2, tile_x + building_size_tiles // 2 + 1):
                if not (0 <= row < GRASS_ROWS and 0 <= col < GRASS_COLS) or is_tile_occupied(row, col, all_units):
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
        snapped_x = snapped_building_pos.x - camera_x + VIEW_MARGIN_LEFT
        snapped_y = snapped_building_pos.y - camera_y + VIEW_MARGIN_TOP
        pygame.draw.rect(screen, border_color, (snapped_x - building_size / 2, snapped_y - building_size / 2, building_size, building_size), 2)

    # Draw selection rectangle
    if selecting and selection_start and selection_end and current_player:
        rect = pygame.Rect(
            min(selection_start.x - camera_x + VIEW_MARGIN_LEFT, selection_end.x - camera_x + VIEW_MARGIN_LEFT),
            min(selection_start.y - camera_y + VIEW_MARGIN_TOP, selection_end.y - camera_y + VIEW_MARGIN_TOP),
            abs(selection_end.x - selection_start.x),
            abs(selection_end.y - selection_start.y)
        )
        pygame.draw.rect(screen, current_player.color, rect, 3)

    # Draw UI
    pygame.draw.rect(screen, PANEL_COLOR, (0, 0, VIEW_MARGIN_LEFT, SCREEN_HEIGHT))
    pygame.draw.rect(screen, PANEL_COLOR, (SCREEN_WIDTH - VIEW_MARGIN_RIGHT, 0, VIEW_MARGIN_RIGHT, SCREEN_HEIGHT))
    pygame.draw.rect(screen, PANEL_COLOR, (0, 0, SCREEN_WIDTH, VIEW_MARGIN_TOP))
    pygame.draw.rect(screen, PANEL_COLOR, (VIEW_MARGIN_LEFT, PANEL_Y, VIEW_WIDTH, PANEL_HEIGHT))

    player_button_color_0 = GRAY if current_player and current_player.player_id == 0 else LIGHT_GRAY
    player_button_color_1 = BLUE if current_player and current_player.player_id == 1 else LIGHT_GRAY
    player_button_color_2 = PURPLE if current_player and current_player.player_id == 2 else LIGHT_GRAY
    pygame.draw.rect(screen, player_button_color_0, button_player0)
    pygame.draw.rect(screen, player_button_color_1, button_player1)
    pygame.draw.rect(screen, player_button_color_2, button_player2)
    player0_text = button_font.render("Player 0", True, BLACK)
    player1_text = button_font.render("Player 1", True, BLACK)
    player2_text = button_font.render("Player 2", True, BLACK)
    screen.blit(player0_text, (BUTTON_PLAYER0_POS[0] + 10, BUTTON_PLAYER0_POS[1] + 5))
    screen.blit(player1_text, (BUTTON_PLAYER1_POS[0] + 10, BUTTON_PLAYER1_POS[1] + 5))
    screen.blit(player2_text, (BUTTON_PLAYER2_POS[0] + 10, BUTTON_PLAYER2_POS[1] + 5))

    # Draw 3x1 grid of buttons with moving line animation
    for row in range(GRID_BUTTON_ROWS):
        for col in range(GRID_BUTTON_COLS):
            button_rect = grid_buttons[row][col]
            selected_barn = next((unit for unit in current_player.units if isinstance(unit, Barn) and unit.selected and unit.alpha == 255), None) if current_player else None
            selected_barracks = next((unit for unit in current_player.units if isinstance(unit, Barracks) and unit.selected and unit.alpha == 255), None) if current_player else None
            selected_town_center = next((unit for unit in current_player.units if isinstance(unit, TownCenter) and unit.selected and unit.alpha == 255), None) if current_player else None

            if current_player and selected_barn and row == 0 and col == 0:
                # Barn button for Cow
                spawn_button_color = HIGHLIGHT_GRAY if (current_player.milk >= Cow.milk_cost and current_player.wood >= Cow.wood_cost and
                                                        (current_player.unit_limit is None or current_player.unit_count < current_player.unit_limit) and
                                                        selected_barn not in production_queues) else GRAY
                pygame.draw.rect(screen, spawn_button_color, button_rect)
                screen.blit(Unit._unit_icons.get('Cow'), (button_rect.x + 8, button_rect.y + 4))
                screen.blit(milk_icon, (button_rect.x + 44, button_rect.y + 2))
                spawn_food_text = small_font.render(f"{Cow.milk_cost}", True, BLACK)
                screen.blit(spawn_food_text, (button_rect.x + 70, button_rect.y + 6))
                screen.blit(wood_icon, (button_rect.x + 44, button_rect.y + 22))
                spawn_wood_text = small_font.render(f"{Cow.wood_cost}", True, BLACK)
                screen.blit(spawn_wood_text, (button_rect.x + 70, button_rect.y + 26))
                if selected_barn in production_queues and production_queues[selected_barn]['unit_type'] == Cow:
                    progress = (current_time - production_queues[selected_barn]['start_time']) / Cow.production_time
                    progress_width = int(progress * (GRID_BUTTON_WIDTH - 4))
                    pygame.draw.rect(screen, GREEN, (button_rect.x + 2, button_rect.y + 2, progress_width, 4))
            elif current_player and selected_barracks:
                if row == 0 and col == 0:
                    # Barracks button for Axeman
                    spawn_button_color = HIGHLIGHT_GRAY if (current_player.milk >= Axeman.milk_cost and current_player.wood >= Axeman.wood_cost and
                                                            (current_player.unit_limit is None or current_player.unit_count < current_player.unit_limit) and
                                                            selected_barracks not in production_queues) else GRAY
                    pygame.draw.rect(screen, spawn_button_color, button_rect)
                    screen.blit(Unit._unit_icons.get('Axeman'), (button_rect.x + 8, button_rect.y + 4))
                    screen.blit(milk_icon, (button_rect.x + 44, button_rect.y + 2))
                    spawn_food_text = small_font.render(f"{Axeman.milk_cost}", True, BLACK)
                    screen.blit(spawn_food_text, (button_rect.x + 70, button_rect.y + 6))
                    screen.blit(wood_icon, (button_rect.x + 44, button_rect.y + 22))
                    spawn_wood_text = small_font.render(f"{Axeman.wood_cost}", True, BLACK)
                    screen.blit(spawn_wood_text, (button_rect.x + 70, button_rect.y + 26))
                    if selected_barracks in production_queues and production_queues[selected_barracks]['unit_type'] == Axeman:
                        progress = (current_time - production_queues[selected_barracks]['start_time']) / Axeman.production_time
                        progress_width = int(progress * (GRID_BUTTON_WIDTH - 4))
                        pygame.draw.rect(screen, GREEN, (button_rect.x + 2, button_rect.y + 2, progress_width, 4))
                elif row == 1 and col == 0:
                    # Barracks button for Archer
                    spawn_button_color = HIGHLIGHT_GRAY if (current_player.milk >= Archer.milk_cost and current_player.wood >= Archer.wood_cost and
                                                            (current_player.unit_limit is None or current_player.unit_count < current_player.unit_limit) and
                                                            selected_barracks not in production_queues) else GRAY
                    pygame.draw.rect(screen, spawn_button_color, button_rect)
                    screen.blit(Unit._unit_icons.get('Archer'), (button_rect.x + 8, button_rect.y + 4))
                    screen.blit(milk_icon, (button_rect.x + 44, button_rect.y + 2))
                    spawn_food_text = small_font.render(f"{Archer.milk_cost}", True, BLACK)
                    screen.blit(spawn_food_text, (button_rect.x + 70, button_rect.y + 6))
                    screen.blit(wood_icon, (button_rect.x + 44, button_rect.y + 22))
                    spawn_wood_text = small_font.render(f"{Archer.wood_cost}", True, BLACK)
                    screen.blit(spawn_wood_text, (button_rect.x + 70, button_rect.y + 26))
                    if selected_barracks in production_queues and production_queues[selected_barracks]['unit_type'] == Archer:
                        progress = (current_time - production_queues[selected_barracks]['start_time']) / Archer.production_time
                        progress_width = int(progress * (GRID_BUTTON_WIDTH - 4))
                        pygame.draw.rect(screen, GREEN, (button_rect.x + 2, button_rect.y + 2, progress_width, 4))
                elif row == 2 and col == 0:
                    # Barracks button for Knight
                    spawn_button_color = HIGHLIGHT_GRAY if (current_player.milk >= Knight.milk_cost and current_player.wood >= Knight.wood_cost and
                                                            (current_player.unit_limit is None or current_player.unit_count < current_player.unit_limit) and
                                                            selected_barracks not in production_queues) else GRAY
                    pygame.draw.rect(screen, spawn_button_color, button_rect)
                    screen.blit(Unit._unit_icons.get('Knight'), (button_rect.x + 8, button_rect.y + 4))
                    screen.blit(milk_icon, (button_rect.x + 44, button_rect.y + 2))
                    spawn_food_text = small_font.render(f"{Knight.milk_cost}", True, BLACK)
                    screen.blit(spawn_food_text, (button_rect.x + 70, button_rect.y + 6))
                    screen.blit(wood_icon, (button_rect.x + 44, button_rect.y + 22))
                    spawn_wood_text = small_font.render(f"{Knight.wood_cost}", True, BLACK)
                    screen.blit(spawn_wood_text, (button_rect.x + 70, button_rect.y + 26))
                    if selected_barracks in production_queues and production_queues[selected_barracks]['unit_type'] == Knight:
                        progress = (current_time - production_queues[selected_barracks]['start_time']) / Knight.production_time
                        progress_width = int(progress * (GRID_BUTTON_WIDTH - 4))
                        pygame.draw.rect(screen, GREEN, (button_rect.x + 2, button_rect.y + 2, progress_width, 4))
            elif current_player and selected_town_center:
                if row == 0 and col == 0:
                    # TownCenter button for Barn
                    spawn_button_color = HIGHLIGHT_GRAY if (current_player.milk >= Barn.milk_cost and current_player.wood >= Barn.wood_cost and
                                                            (current_player.building_limit is None or current_player.building_count < current_player.building_limit) and
                                                            selected_town_center not in production_queues) else GRAY
                    pygame.draw.rect(screen, spawn_button_color, button_rect)
                    screen.blit(Unit._unit_icons.get('Barn'), (button_rect.x + 8, button_rect.y + 4))
                    screen.blit(milk_icon, (button_rect.x + 44, button_rect.y + 2))
                    spawn_food_text = small_font.render(f"{Barn.milk_cost}", True, BLACK)
                    screen.blit(spawn_food_text, (button_rect.x + 70, button_rect.y + 6))
                    screen.blit(wood_icon, (button_rect.x + 44, button_rect.y + 22))
                    spawn_wood_text = small_font.render(f"{Barn.wood_cost}", True, BLACK)
                    screen.blit(spawn_wood_text, (button_rect.x + 70, button_rect.y + 26))
                    if selected_town_center in production_queues and production_queues[selected_town_center]['unit_type'] == Barn:
                        progress = (current_time - production_queues[selected_town_center]['start_time']) / Barn.production_time
                        progress_width = int(progress * (GRID_BUTTON_WIDTH - 4))
                        pygame.draw.rect(screen, GREEN, (button_rect.x + 2, button_rect.y + 2, progress_width, 4))
                elif row == 1 and col == 0:
                    # TownCenter button for Barracks
                    spawn_button_color = HIGHLIGHT_GRAY if (current_player.milk >= Barracks.milk_cost and current_player.wood >= Barracks.wood_cost and
                                                            (current_player.building_limit is None or current_player.building_count < current_player.building_limit) and
                                                            selected_town_center not in production_queues) else GRAY
                    pygame.draw.rect(screen, spawn_button_color, button_rect)
                    screen.blit(Unit._unit_icons.get('Barracks'), (button_rect.x + 8, button_rect.y + 4))
                    screen.blit(milk_icon, (button_rect.x + 44, button_rect.y + 2))
                    spawn_food_text = small_font.render(f"{Barracks.milk_cost}", True, BLACK)
                    screen.blit(spawn_food_text, (button_rect.x + 70, button_rect.y + 6))
                    screen.blit(wood_icon, (button_rect.x + 44, button_rect.y + 22))
                    spawn_wood_text = small_font.render(f"{Barracks.wood_cost}", True, BLACK)
                    screen.blit(spawn_wood_text, (button_rect.x + 70, button_rect.y + 26))
                    if selected_town_center in production_queues and production_queues[selected_town_center]['unit_type'] == Barracks:
                        progress = (current_time - production_queues[selected_town_center]['start_time']) / Barracks.production_time
                        progress_width = int(progress * (GRID_BUTTON_WIDTH - 4))
                        pygame.draw.rect(screen, GREEN, (button_rect.x + 2, button_rect.y + 2, progress_width, 4))
                elif row == 2 and col == 0:
                    # TownCenter button for TownCenter
                    spawn_button_color = HIGHLIGHT_GRAY if (current_player.milk >= TownCenter.milk_cost and current_player.wood >= TownCenter.wood_cost and
                                                            (current_player.building_limit is None or current_player.building_count < current_player.building_limit) and
                                                            selected_town_center not in production_queues) else GRAY
                    pygame.draw.rect(screen, spawn_button_color, button_rect)
                    screen.blit(Unit._unit_icons.get('TownCenter'), (button_rect.x + 8, button_rect.y + 4))
                    screen.blit(milk_icon, (button_rect.x + 44, button_rect.y + 2))
                    spawn_food_text = small_font.render(f"{TownCenter.milk_cost}", True, BLACK)
                    screen.blit(spawn_food_text, (button_rect.x + 70, button_rect.y + 6))
                    screen.blit(wood_icon, (button_rect.x + 44, button_rect.y + 22))
                    spawn_wood_text = small_font.render(f"{TownCenter.wood_cost}", True, BLACK)
                    screen.blit(spawn_wood_text, (button_rect.x + 70, button_rect.y + 26))
                    if selected_town_center in production_queues and production_queues[selected_town_center]['unit_type'] == TownCenter:
                        progress = (current_time - production_queues[selected_town_center]['start_time']) / TownCenter.production_time
                        progress_width = int(progress * (GRID_BUTTON_WIDTH - 4))
                        pygame.draw.rect(screen, GREEN, (button_rect.x + 2, button_rect.y + 2, progress_width, 4))
            else:
                pygame.draw.rect(screen, LIGHT_GRAY, button_rect)

    # Draw unit icons for selected units
    icon_x = VIEW_MARGIN_LEFT + 10
    icon_y = PANEL_Y + 10
    if current_player:
        selected_units = [unit for unit in current_player.units if unit.selected and unit.player_id == current_player.player_id]
        for unit in selected_units:
            # Only show grid buttons for fully constructed buildings
            if isinstance(unit, Building) and unit.alpha < 255:
                continue
            cls_name = unit.__class__.__name__
            unit_icon_img = Unit._unit_icons.get(cls_name)
            if unit_icon_img:
                screen.blit(unit_icon_img, (icon_x, icon_y))
            else:
                pygame.draw.rect(screen, WHITE, (icon_x, icon_y, ICON_SIZE, ICON_SIZE))
            # If exactly one unit is selected, display name and stats
            if len(selected_units) == 1:
                # Display "Name - Class"
                display_text = f"{unit.name or cls_name} - {cls_name}"
                text_surface = small_font.render(display_text, True, current_player.color)
                screen.blit(text_surface, (icon_x, icon_y + ICON_SIZE + 5))
                # Display HP
                hp_text = f"HP: {int(unit.hp)}/{int(unit.max_hp)}"
                hp_surface = small_font.render(hp_text, True, current_player.color)
                screen.blit(hp_surface, (icon_x, icon_y + ICON_SIZE + 20))
                # Display Attack
                attack_text = f"Attack: {unit.attack_damage}"
                attack_surface = small_font.render(attack_text, True, current_player.color)
                screen.blit(attack_surface, (icon_x, icon_y + ICON_SIZE + 35))
                # Display Armor
                armor_text = f"Armor: {unit.armor}"
                armor_surface = small_font.render(armor_text, True, current_player.color)
                screen.blit(armor_surface, (icon_x, icon_y + ICON_SIZE + 50))
                # Display Speed
                speed_text = f"Speed: {unit.speed}"
                speed_surface = small_font.render(speed_text, True, current_player.color)
                screen.blit(speed_surface, (icon_x, icon_y + ICON_SIZE + 65))
            icon_x += ICON_SIZE + ICON_MARGIN

    # Draw additional info
    if current_player:
        screen.blit(milk_icon, (VIEW_MARGIN_LEFT + 10, 10))
        resources_text = font.render(
            f": {current_player.milk:.0f}/{current_player.max_milk}",
            True, current_player.color
        )
        screen.blit(resources_text, (VIEW_MARGIN_LEFT + 30, 15))
        screen.blit(wood_icon, (VIEW_MARGIN_LEFT + 120, 10))
        wood_text = f": {current_player.wood:.0f}/{current_player.max_wood}"
        resources_text = font.render(wood_text, True, current_player.color)
        screen.blit(resources_text, (VIEW_MARGIN_LEFT + 140, 15))
        if current_player.unit_limit is not None:
            screen.blit(unit_icon, (VIEW_MARGIN_LEFT + 230, 10))
            unit_text = f": {current_player.unit_count}/{current_player.unit_limit}"
            text_color = ORANGE if current_player.unit_count > current_player.unit_limit else current_player.color
            unit_text_surface = font.render(unit_text, True, text_color)
            screen.blit(unit_text_surface, (VIEW_MARGIN_LEFT + 250, 15))
        if current_player.building_limit is not None:
            screen.blit(building_icon, (VIEW_MARGIN_LEFT + 320, 10))
            building_text = f": {current_player.building_count}/{current_player.building_limit}"
            text_color = ORANGE if current_player.building_count > current_player.building_limit else current_player.color
            building_text_surface = font.render(building_text, True, text_color)
            screen.blit(building_text_surface, (VIEW_MARGIN_LEFT + 340, 15))

    fps = clock.get_fps()
    fps_text = font.render(f"FPS: {int(fps)}", True, WHITE)
    screen.blit(fps_text, (VIEW_MARGIN_LEFT + VIEW_WIDTH - 80, VIEW_MARGIN_TOP + 10))

    selected_count = sum(1 for unit in all_units if unit.selected and unit.player_id == (current_player.player_id if current_player else -1))
    selected_text = font.render(f"Selected Units: {selected_count}", True, BLACK)
    screen.blit(selected_text, (VIEW_MARGIN_LEFT + 10, PANEL_Y + 120))

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
sys.exit()