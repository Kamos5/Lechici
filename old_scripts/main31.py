import pygame
import sys
import random
import math
from pygame.math import Vector2

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

wood_icon = pygame.image.load("../assets/wood_icon.png").convert_alpha()
milk_icon = pygame.image.load("../assets/milk_icon.png").convert_alpha()

# Scale icons
wood_icon = pygame.transform.scale(wood_icon, (20, 20))
milk_icon = pygame.transform.scale(milk_icon, (20, 20))

# Spatial grid settings
GRID_CELL_SIZE = 60

# Move order tracking
move_order_times = {}

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
            cls._grass_image = pygame.image.load("../assets/grass.png").convert_alpha()
            cls._grass_image = pygame.transform.scale(cls._grass_image, (TILE_SIZE, TILE_SIZE))
            cls._dirt_image = pygame.image.load("../assets/dirt.png").convert_alpha()
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
        self.hp = 100
        self.mana = 0
        self.special = 0
        cls_name = self.__class__.__name__
        if cls_name not in Unit._images:
            self.load_images(cls_name, size)

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

    def move(self, units):
        self.velocity = Vector2(0, 0)
        if self.target:
            direction = self.target - self.pos
            distance_to_target = direction.length()
            if distance_to_target > self.size:  # Stop within unit's size without snapping
                self.velocity = direction.normalize() * self.speed
            else:
                self.target = None  # Stop moving without snapping position
        self.velocity *= self.damping
        self.pos += self.velocity

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
                    nearest_corner = min(barn_corners, key=lambda corner: self.pos.distance_to(corner))
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
        if Tree._image is None:
            Tree.load_image()
        self.pos.x = x
        self.pos.y = y

    @classmethod
    def load_image(cls):
        try:
            cls._image = pygame.image.load("../assets/tree.png").convert_alpha()
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
                current_time = pygame.time.get_ticks() / 1000
                if current_time - self._last_color_change_time >= 1:
                    self._color_index = (self._color_index + 1) % len(self._colors)
                    self._last_color_change_time = current_time
                image = self._tinted_images[self._color_index]
            else:
                image = self._image
            image_rect = image.get_rect(center=(int(x), int(y)))
            screen.blit(image, image_rect)
            if is_targeted:
                pygame.draw.rect(screen, WHITE, (x - TILE_HALF, y - TILE_HALF, TILE_SIZE, TILE_SIZE), 1)
            if self._selected:
                pygame.draw.rect(screen, YELLOW, (x - TILE_HALF, y - TILE_HALF, TILE_SIZE, TILE_SIZE), 1)
        else:
            pygame.draw.rect(screen, TREE_GREEN, (x - TILE_HALF, y - TILE_HALF, TILE_SIZE, TILE_SIZE))

    def is_clicked(self, click_pos, camera_x, camera_y):
        adjusted_pos = Vector2(click_pos.x + camera_x, click_pos.y + camera_y)
        tile_rect = pygame.Rect(self.pos.x - TILE_HALF, self.pos.y - TILE_HALF, TILE_SIZE, TILE_SIZE)
        return tile_rect.collidepoint(adjusted_pos)

    def set_selected(self, selected):
        self._selected = selected

    def move(self, units):
        pass

# Building class
class Building(Unit):
    def __init__(self, x, y, size, color, player_id, player_color):
        super().__init__(x, y, size, speed=0, color=color, player_id=player_id, player_color=player_color)

    def move(self, units):
        pass

# Barn class
class Barn(Building):
    def __init__(self, x, y, player_id, player_color):
        super().__init__(x, y, size=60, color=DARK_GRAY, player_id=player_id, player_color=player_color)
        self.harvest_rate = 60.0

# TownCenter class
class TownCenter(Building):
    def __init__(self, x, y, player_id, player_color):
        super().__init__(x, y, size=60, color=TOWN_CENTER_GRAY, player_id=player_id, player_color=player_color)

# Barracks class
class Barracks(Building):
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

    def move(self, units, spatial_grid):
        self.velocity = Vector2(0, 0)
        if self.special >= 100 and not self.depositing:
            town_centers = [unit for unit in all_units if isinstance(unit, TownCenter) and unit.player_id == self.player_id]
            if town_centers:
                closest_town = min(town_centers, key=lambda town: self.pos.distance_to(town.pos))
                self.target = closest_town.pos
                self.depositing = True
                print(f"Axeman at {self.pos} targeting TownCenter at {self.target} for wood deposit")
            else:
                print(f"Axeman at {self.pos} found no TownCenter, clearing state")
                self.special = 0
                self.depositing = False
                self.target = None
                self.return_pos = None
        elif self.depositing and self.target:
            direction = self.target - self.pos
            distance_to_target = direction.length()
            target_unit = next((unit for unit in all_units if isinstance(unit, TownCenter) and unit.pos == self.target), None)
            tolerance = self.size + target_unit.size / 2 if target_unit else self.size
            if distance_to_target <= tolerance:
                for player in players:
                    if player.player_id == self.player_id:
                        player.wood = min(player.max_wood, player.wood + 100)
                        print(f"Axeman at {self.pos} deposited 100 wood for Player {self.player_id}, player wood now {player.wood}")
                        break
                self.special = 0
                self.depositing = False
                self.target = self.return_pos
                print(f"Axeman at {self.pos} returning to {self.return_pos}")
            else:
                self.velocity = direction.normalize() * self.speed
        elif self.return_pos and self.target == self.return_pos:
            direction = self.target - self.pos
            distance_to_target = direction.length()
            if distance_to_target <= self.size:
                self.target = None
                self.return_pos = None
                nearby_units = spatial_grid.get_nearby_units(self, radius=1000)
                trees = [unit for unit in nearby_units if isinstance(unit, Tree) and unit.player_id == 0]
                if trees:
                    closest_tree = min(trees, key=lambda tree: self.pos.distance_to(tree.pos))
                    self.target = closest_tree.pos
                    print(f"Axeman at {self.pos} targeting new tree at {self.target}")
                else:
                    print(f"Axeman at {self.pos} found no trees to harvest")
            else:
                self.velocity = direction.normalize() * self.speed
        elif self.target:
            direction = self.target - self.pos
            distance_to_target = direction.length()
            if distance_to_target > self.size:
                self.velocity = direction.normalize() * self.speed
            else:
                self.target = None
        self.velocity *= self.damping
        self.pos += self.velocity

    def chop_tree(self, trees):
        if self.special > 0:
            return
        for tree in trees:
            if (isinstance(tree, Tree) and tree.player_id == 0 and
                self.target == tree.pos and self.pos.distance_to(tree.pos) <= TILE_SIZE):
                tree.hp -= self.chop_damage
                if tree.hp <= 0:
                    self.return_pos = Vector2(self.pos)
                    players[0].units.remove(tree)
                    all_units.remove(tree)
                    spatial_grid.remove_unit(tree)
                    self.special += 100
                    print(f"Tree at {tree.pos} chopped down by Axeman at {self.pos}, special = {self.special}")
                    self.target = None
                break

# Knight class
class Knight(Unit):
    milk_cost = 500
    wood_cost = 400

    def __init__(self, x, y, player_id, player_color):
        super().__init__(x, y, size=16, speed=4, color=BLUE, player_id=player_id, player_color=player_color)

# Archer class
class Archer(Unit):
    milk_cost = 400
    wood_cost = 200

    def __init__(self, x, y, player_id, player_color):
        super().__init__(x, y, size=16, speed=3, color=YELLOW, player_id=player_id, player_color=player_color)

# Cow class
class Cow(Unit):
    milk_cost = 400
    wood_cost = 0

    def __init__(self, x, y, player_id, player_color):
        super().__init__(x, y, size=16, speed=4, color=BROWN, player_id=player_id, player_color=player_color)
        self.harvest_rate = 0.01
        self.assigned_corner = None

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
        bar_width = 16
        bar_height = 4
        bar_offset = 2
        bar_x = x - bar_width / 2
        bar_y = y - self.size / 2 - bar_height - bar_offset
        pygame.draw.rect(screen, BLACK, (bar_x, bar_y, bar_width, bar_height))
        fill_width = (self.special / 100) * bar_width
        pygame.draw.rect(screen, WHITE, (bar_x, bar_y, fill_width, bar_height))

    def move(self, units):
        self.velocity = Vector2(0, 0)
        if self.target:
            direction = self.target - self.pos
            distance_to_target = direction.length()
            if distance_to_target > self.size:
                self.velocity = direction.normalize() * self.speed
            else:
                self.target = None
                self.assigned_corner = None
        self.velocity *= self.damping
        self.pos += self.velocity

    def is_in_barn(self, barn):
        return (isinstance(barn, Barn) and
                barn.pos.x - barn.size / 2 <= self.pos.x <= barn.pos.x + barn.size / 2 and
                barn.pos.y - barn.size / 2 <= self.pos.y <= barn.pos.y + barn.size / 2)

    def is_in_barn_any(self, barns):
        return any(self.is_in_barn(barn) for barn in barns)

    def harvest_grass(self, grass_tiles, barns, cow_in_barn, player, spatial_grid):
        if self.is_in_barn_any(barns) and self.special > 0:
            return
        if self.special >= 100:
            if not self.target:
                available_barns = [barn for barn in barns if barn.player_id == self.player_id and (barn not in cow_in_barn or cow_in_barn[barn] is None)]
                if available_barns:
                    closest_barn = min(available_barns, key=lambda barn: self.pos.distance_to(barn.pos))
                    target_x = closest_barn.pos.x - closest_barn.size / 2 + TILE_HALF
                    target_y = closest_barn.pos.y + closest_barn.size / 2 - TILE_HALF
                    self.target = Vector2(target_x, target_y)
            return
        for barn in barns:
            if self.special == 0 and self.is_in_barn(barn):
                tile_x = int(self.pos.x // TILE_SIZE)
                tile_y = int(self.pos.y // TILE_SIZE)
                adjacent_tiles = [
                    (tile_x, tile_y - 1), (tile_x, tile_y + 1), (tile_x - 1, tile_y), (tile_x + 1, tile_y),
                    (tile_x - 1, tile_y - 1), (tile_x + 1, tile_y - 1), (tile_x - 1, tile_y + 1), (tile_x + 1, tile_y + 1)
                ]
                random.shuffle(adjacent_tiles)
                for adj_x, adj_y in adjacent_tiles:
                    if 0 <= adj_x < GRASS_COLS and 0 <= adj_y < GRASS_ROWS:
                        tile = grass_tiles[adj_y][adj_x]
                        if isinstance(tile, GrassTile) and not isinstance(tile, Dirt) and tile.grass_level > 0.5:
                            self.target = tile.center
                            return
                return
        if not self.target or self.velocity.length() < 0.5:
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
                                if 0 <= adj_x < GRASS_COLS and 0 <= adj_y < GRASS_ROWS:
                                    adj_tile = grass_tiles[adj_y][adj_x]
                                    if isinstance(adj_tile, GrassTile) and not isinstance(adj_tile, Dirt) and adj_tile.grass_level > 0.5:
                                        self.target = adj_tile.center
                                        break

# Player class
class Player:
    def __init__(self, player_id, color, start_x, start_y):
        self.player_id = player_id
        self.color = color
        self.milk = 0.0
        self.max_milk = 1000
        self.wood = 500.0  # Starting wood set to 500
        self.max_wood = 1000
        self.units = []
        self.cow_in_barn = {}
        self.barns = []
        offset_x = start_x
        offset_y = start_y
        if self.player_id > 0:
            self.units.extend([
                Axeman(offset_x + 100, offset_y + 100, player_id, self.color),
                Knight(offset_x + 150, offset_y + 100, player_id, self.color),
                Archer(offset_x + 200, offset_y + 100, player_id, self.color),
                Cow(offset_x + 260, offset_y + 100, player_id, self.color),
                Barn(offset_x + 230, offset_y + 150, player_id, self.color),
                TownCenter(offset_x + 150, offset_y + 150, player_id, self.color),
                Barracks(offset_x + 70, offset_y + 150, player_id, self.color)
            ])
            self.barns = [unit for unit in self.units if isinstance(unit, Barn)]

    def select_all_units(self):
        for unit in self.units:
            unit.selected = True

    def deselect_all_units(self):
        for unit in self.units:
            unit.selected = False

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
    patch_size = random.randint(0, 2)
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

# Game loop
running = True
tile_surface = pygame.Surface((VIEW_WIDTH, VIEW_HEIGHT), pygame.SRCALPHA)
while running:
    current_time = pygame.time.get_ticks() / 1000

    # Update camera
    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT]:
        camera_x = max(0, camera_x - SCROLL_SPEED)
    if keys[pygame.K_RIGHT]:
        camera_x = min(MAP_WIDTH - VIEW_WIDTH, camera_x + SCROLL_SPEED)
    if keys[pygame.K_UP]:
        camera_y = max(0, camera_y - SCROLL_SPEED)
    if keys[pygame.K_DOWN]:
        camera_y = min(MAP_HEIGHT - VIEW_HEIGHT, camera_y + SCROLL_SPEED)

    # Check for building selection
    barn_selected = any(isinstance(unit, Barn) and unit.selected for unit in (current_player.units if current_player else []))
    barracks_selected = any(isinstance(unit, Barracks) and unit.selected for unit in (current_player.units if current_player else []))

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
                elif button_player1.collidepoint(event.pos):
                    print("Player 1 button clicked")
                    current_player = players[1]
                    for player in players:
                        player.deselect_all_units()
                elif button_player2.collidepoint(event.pos):
                    print("Player 2 button clicked")
                    current_player = players[2]
                    for player in players:
                        player.deselect_all_units()
                else:
                    # Check if click is on grid buttons for spawning
                    grid_button_clicked = False
                    if barn_selected and current_player and row == 0 and col == 0:
                        if grid_buttons[0][0].collidepoint(event.pos) and current_player.milk >= Cow.milk_cost and current_player.wood >= Cow.wood_cost:
                            selected_barn = next((unit for unit in current_player.units if isinstance(unit, Barn) and unit.selected), None)
                            if selected_barn:
                                new_cow = Cow(selected_barn.pos.x + selected_barn.size / 2 + 20, selected_barn.pos.y, current_player.player_id, current_player.color)
                                current_player.units.append(new_cow)
                                all_units.add(new_cow)
                                spatial_grid.add_unit(new_cow)
                                current_player.milk -= Cow.milk_cost
                                current_player.wood -= Cow.wood_cost
                                current_player.barns = [unit for unit in current_player.units if isinstance(unit, Barn)]
                                grid_button_clicked = True
                    elif barracks_selected and current_player:
                        selected_barracks = next((unit for unit in current_player.units if isinstance(unit, Barracks) and unit.selected), None)
                        if selected_barracks:
                            if grid_buttons[0][0].collidepoint(event.pos) and current_player.milk >= Axeman.milk_cost and current_player.wood >= Axeman.wood_cost:
                                new_unit = Axeman(selected_barracks.pos.x + selected_barracks.size / 2 + 20, selected_barracks.pos.y, current_player.player_id, current_player.color)
                                current_player.units.append(new_unit)
                                all_units.add(new_unit)
                                spatial_grid.add_unit(new_unit)
                                current_player.milk -= Axeman.milk_cost
                                current_player.wood -= Axeman.wood_cost
                                grid_button_clicked = True
                            elif grid_buttons[1][0].collidepoint(event.pos) and current_player.milk >= Archer.milk_cost and current_player.wood >= Archer.wood_cost:
                                new_unit = Archer(selected_barracks.pos.x + selected_barracks.size / 2 + 20, selected_barracks.pos.y, current_player.player_id, current_player.color)
                                current_player.units.append(new_unit)
                                all_units.add(new_unit)
                                spatial_grid.add_unit(new_unit)
                                current_player.milk -= Archer.milk_cost
                                current_player.wood -= Archer.wood_cost
                                grid_button_clicked = True
                            elif grid_buttons[2][0].collidepoint(event.pos) and current_player.milk >= Knight.milk_cost and current_player.wood >= Knight.wood_cost:
                                new_unit = Knight(selected_barracks.pos.x + selected_barracks.size / 2 + 20, selected_barracks.pos.y, current_player.player_id, current_player.color)
                                current_player.units.append(new_unit)
                                all_units.add(new_unit)
                                spatial_grid.add_unit(new_unit)
                                current_player.milk -= Knight.milk_cost
                                current_player.wood -= Knight.wood_cost
                                grid_button_clicked = True

                    # Handle game view clicks for unit selection
                    if not grid_button_clicked and VIEW_MARGIN_LEFT <= mouse_pos.x <= VIEW_BOUNDS_X and VIEW_MARGIN_TOP <= mouse_pos.y <= VIEW_BOUNDS_Y:
                        click_pos = Vector2(mouse_pos.x - VIEW_MARGIN_LEFT + camera_x, mouse_pos.y - VIEW_MARGIN_TOP + camera_y)
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
                            selecting = False  # Reset selection state
                            selection_start = None
                            selection_end = None
                        else:
                            for player in players:
                                player.deselect_all_units()  # Deselect all units on empty click
                            selection_start = click_pos
                            selecting = True
            elif event.button == 3:
                mouse_pos = Vector2(event.pos)
                if VIEW_MARGIN_LEFT <= mouse_pos.x <= VIEW_BOUNDS_X and VIEW_MARGIN_TOP <= mouse_pos.y <= VIEW_BOUNDS_Y:
                    click_pos = Vector2(mouse_pos.x - VIEW_MARGIN_LEFT + camera_x, mouse_pos.y - VIEW_MARGIN_TOP + camera_y)
                    for unit in all_units:
                        if unit.selected and not isinstance(unit, (Building, Tree)):
                            clicked_tree = next((tree for tree in all_units if isinstance(tree, Tree) and tree.is_clicked(Vector2(mouse_pos.x - VIEW_MARGIN_LEFT, mouse_pos.y - VIEW_MARGIN_TOP), camera_x, camera_y)), None)
                            if clicked_tree and isinstance(unit, Axeman) and unit.special == 0 and not unit.depositing:
                                unit.target = clicked_tree.pos
                                print(f"Axeman at {unit.pos} assigned to tree at {unit.target}")
                            else:
                                unit.target = Vector2(click_pos.x, click_pos.y)
                                move_order_times[(click_pos.x, click_pos.y)] = current_time
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
                    # Reset selection positions to forget last mouse_pos
                    selection_start = None
                    selection_end = None
        elif event.type == pygame.MOUSEMOTION and selecting:
            mouse_pos = Vector2(event.pos)
            if VIEW_MARGIN_LEFT <= mouse_pos.x <= VIEW_BOUNDS_X and VIEW_MARGIN_TOP <= mouse_pos.y <= VIEW_BOUNDS_Y:
                selection_end = Vector2(mouse_pos.x - VIEW_MARGIN_LEFT + camera_x, mouse_pos.y - VIEW_MARGIN_TOP + camera_y)

    # Update grass regrowth
    for row, col in list(needs_regrowth):
        tile = grass_tiles[row][col]
        tile.regrow(regrowth_rate)
        if tile.grass_level >= 1.0:
            needs_regrowth.remove((row, col))

    # Update units
    axemen = [unit for unit in all_units if isinstance(unit, Axeman)]
    for player in players:
        for unit in player.units:
            unit._corrections = []
            if isinstance(unit, Axeman):
                unit.move(all_units, spatial_grid)
            else:
                unit.move(all_units)
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
                if cow.is_in_barn(barn) and player.milk < player.max_milk and cow.special > 0:
                    cow.special = max(0, cow.special - barn.harvest_rate / 60)
                    player.milk = min(player.milk + barn.harvest_rate / 60, player.max_milk)
                    if cow.special <= 0:
                        player.cow_in_barn[barn] = None
                else:
                    player.cow_in_barn[barn] = None
            else:
                for unit in player.units:
                    if isinstance(unit, Cow) and unit.is_in_barn(barn):
                        player.cow_in_barn[barn] = unit
                        break

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
            if tile_pos in move_order_times and current_time - move_order_times[tile_pos] <= 0.4:
                x1 = tile_pos[0] - TILE_QUARTER // 2 - camera_x
                y1 = tile_pos[1] - TILE_QUARTER // 2 - camera_y
                x2 = tile_pos[0] + TILE_QUARTER // 2 - camera_x
                y2 = tile_pos[1] + TILE_QUARTER // 2 - camera_y
                pygame.draw.line(tile_surface, WHITE, (x1, y1), (x2, y2), 1)
                pygame.draw.line(tile_surface, WHITE, (x2, y1), (x1, y2), 1)
    screen.blit(tile_surface, (VIEW_MARGIN_LEFT, VIEW_MARGIN_TOP))

    # Clean up move orders
    move_order_times = {k: v for k, v in move_order_times.items() if current_time - v <= 1}

    # Draw units
    for unit in all_units:
        if isinstance(unit, Tree):
            unit.draw(screen, camera_x, camera_y, axemen)
        else:
            unit.draw(screen, camera_x, camera_y)

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

    # Draw 3x1 grid of buttons
    for row in range(GRID_BUTTON_ROWS):
        for col in range(GRID_BUTTON_COLS):
            button_rect = grid_buttons[row][col]
            if barn_selected and current_player and row == 0 and col == 0:
                spawn_button_color = HIGHLIGHT_GRAY if current_player.milk >= Cow.milk_cost and current_player.wood >= Cow.wood_cost else GRAY
                pygame.draw.rect(screen, spawn_button_color, button_rect)
                screen.blit(Unit._unit_icons.get('Cow'), (button_rect.x + 8, button_rect.y + 4))
                screen.blit(milk_icon, (button_rect.x + 44, button_rect.y + 2))
                spawn_food_text = small_font.render(f"{Cow.milk_cost}", True, BLACK)
                screen.blit(spawn_food_text, (button_rect.x + 70, button_rect.y + 6))
                screen.blit(wood_icon, (button_rect.x + 44, button_rect.y + 22))
                spawn_wood_text = small_font.render(f"{Cow.wood_cost}", True, BLACK)
                screen.blit(spawn_wood_text, (button_rect.x + 70, button_rect.y + 26))
            elif barracks_selected and current_player:
                if row == 0 and col == 0:
                    spawn_button_color = HIGHLIGHT_GRAY if current_player.milk >= Axeman.milk_cost and current_player.wood >= Axeman.wood_cost else GRAY
                    pygame.draw.rect(screen, spawn_button_color, button_rect)
                    screen.blit(Unit._unit_icons.get('Axeman'), (button_rect.x + 8, button_rect.y + 4))
                    screen.blit(milk_icon, (button_rect.x + 44, button_rect.y + 2))
                    spawn_food_text = small_font.render(f"{Axeman.milk_cost}", True, BLACK)
                    screen.blit(spawn_food_text, (button_rect.x + 70, button_rect.y + 6))
                    screen.blit(wood_icon, (button_rect.x + 44, button_rect.y + 22))
                    spawn_wood_text = small_font.render(f"{Axeman.wood_cost}", True, BLACK)
                    screen.blit(spawn_wood_text, (button_rect.x + 70, button_rect.y + 26))
                elif row == 1 and col == 0:
                    spawn_button_color = HIGHLIGHT_GRAY if current_player.milk >= Archer.milk_cost and current_player.wood >= Archer.wood_cost else GRAY
                    pygame.draw.rect(screen, spawn_button_color, button_rect)
                    screen.blit(Unit._unit_icons.get('Archer'), (button_rect.x + 8, button_rect.y + 4))
                    screen.blit(milk_icon, (button_rect.x + 44, button_rect.y + 2))
                    spawn_food_text = small_font.render(f"{Archer.milk_cost}", True, BLACK)
                    screen.blit(spawn_food_text, (button_rect.x + 70, button_rect.y + 6))
                    screen.blit(wood_icon, (button_rect.x + 44, button_rect.y + 22))
                    spawn_wood_text = small_font.render(f"{Archer.wood_cost}", True, BLACK)
                    screen.blit(spawn_wood_text, (button_rect.x + 70, button_rect.y + 26))
                elif row == 2 and col == 0:
                    spawn_button_color = HIGHLIGHT_GRAY if current_player.milk >= Knight.milk_cost and current_player.wood >= Knight.wood_cost else GRAY
                    pygame.draw.rect(screen, spawn_button_color, button_rect)
                    screen.blit(Unit._unit_icons.get('Knight'), (button_rect.x + 8, button_rect.y + 4))
                    screen.blit(milk_icon, (button_rect.x + 44, button_rect.y + 2))
                    spawn_food_text = small_font.render(f"{Knight.milk_cost}", True, BLACK)
                    screen.blit(spawn_food_text, (button_rect.x + 70, button_rect.y + 6))
                    screen.blit(wood_icon, (button_rect.x + 44, button_rect.y + 22))
                    spawn_wood_text = small_font.render(f"{Knight.wood_cost}", True, BLACK)
                    screen.blit(spawn_wood_text, (button_rect.x + 70, button_rect.y + 26))
            else:
                pygame.draw.rect(screen, LIGHT_GRAY, button_rect)

    # Draw unit icons for selected units
    icon_x = VIEW_MARGIN_LEFT + 10
    icon_y = PANEL_Y + 10
    for unit in all_units:
        if unit.selected and current_player and unit.player_id == current_player.player_id:
            cls_name = unit.__class__.__name__
            unit_icon = Unit._unit_icons.get(cls_name)
            if unit_icon:
                screen.blit(unit_icon, (icon_x, icon_y))
            else:
                pygame.draw.rect(screen, WHITE, (icon_x, icon_y, ICON_SIZE, ICON_SIZE))
            icon_x += ICON_SIZE + ICON_MARGIN

    # Draw additional info
    if current_player:
        screen.blit(milk_icon, (VIEW_MARGIN_LEFT + 10, 10))
        resources_text = font.render(
            f": {current_player.milk:.0f}/{current_player.max_milk}",
            True, current_player.color
        )
        screen.blit(resources_text, (VIEW_MARGIN_LEFT + 40, 15))
        screen.blit(wood_icon, (VIEW_MARGIN_LEFT + 130, 10))
        resources_text = font.render(
            f": {current_player.wood:.0f}/{current_player.max_wood}",
            True, current_player.color
        )
        screen.blit(resources_text, (VIEW_MARGIN_LEFT + 160, 15))

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