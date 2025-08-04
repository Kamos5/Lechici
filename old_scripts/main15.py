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
pygame.display.set_caption("Simple RTS Game with Cow Image and Unit Icons")
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
GRASS_ROWS = 60
GRASS_COLS = 60
MAP_WIDTH = GRASS_COLS * TILE_SIZE  # 1200
MAP_HEIGHT = GRASS_ROWS * TILE_SIZE  # 1200

# UI settings
BORDER_WIDTH = 10
VIEW_X = 10  # 10
VIEW_Y = 40  # 10
PANEL_HEIGHT = 150
VIEW_WIDTH = SCREEN_WIDTH - 2 * VIEW_X  # 980
VIEW_HEIGHT = SCREEN_HEIGHT - 1 * VIEW_Y - 150  # 830
PANEL_Y = SCREEN_HEIGHT - PANEL_HEIGHT  # 850

# Button settings
BUTTON_WIDTH = 150
BUTTON_HEIGHT = 40
BUTTON_MARGIN = 10
BUTTON_PLAYER1_POS = (SCREEN_WIDTH // 2 - BUTTON_WIDTH - BUTTON_MARGIN - 50, PANEL_Y + 10)
BUTTON_PLAYER2_POS = (SCREEN_WIDTH // 2 - BUTTON_WIDTH - BUTTON_MARGIN - 50, PANEL_Y + 10 + BUTTON_HEIGHT + BUTTON_MARGIN)
BUTTON_SPAWN_COW_POS = (SCREEN_WIDTH // 2 + BUTTON_WIDTH + BUTTON_MARGIN - 50, PANEL_Y + 10)

# Camera settings
camera_x = 0
camera_y = 0
SCROLL_SPEED = 10
SCROLL_MARGIN = 20

# Icon settings
ICON_SIZE = 32
ICON_MARGIN = 5

# SimpleTile base class
class SimpleTile:
    _image = None

    def __init__(self, x, y):
        self.pos = Vector2(x, y)

    def draw(self, screen, camera_x, camera_y):
        pass

    def harvest(self, amount):
        return 0.0

    def regrow(self, amount):
        pass

# GrassTile class
class GrassTile(SimpleTile):
    _grass_image = None
    _dirt_image = None

    def __init__(self, x, y):
        super().__init__(x, y)
        self.grass_level = 1.0
        if GrassTile._grass_image is None or GrassTile._dirt_image is None:
            GrassTile.load_images()

    @classmethod
    def load_images(cls):
        try:
            cls._grass_image = pygame.image.load("../grass.png").convert_alpha()
            cls._grass_image = pygame.transform.scale(cls._grass_image, (TILE_SIZE, TILE_SIZE))
            cls._dirt_image = pygame.image.load("../dirt.png").convert_alpha()
            cls._dirt_image = pygame.transform.scale(cls._dirt_image, (TILE_SIZE, TILE_SIZE))
        except (pygame.error, FileNotFoundError) as e:
            print(f"Failed to load grass.png or dirt.png: {e}")
            cls._grass_image = None
            cls._dirt_image = None

    def draw(self, screen, camera_x, camera_y):
        if self._grass_image and self._dirt_image:
            blended_surface = pygame.Surface((TILE_SIZE, TILE_SIZE), pygame.SRCALPHA)
            blended_surface.blit(self._grass_image, (0, 0))
            dirt_alpha = int((1.0 - self.grass_level) * 255)
            dirt_overlay = self._dirt_image.copy()
            dirt_overlay.set_alpha(dirt_alpha)
            blended_surface.blit(dirt_overlay, (0, 0))
            screen.blit(blended_surface, (self.pos.x - camera_x, self.pos.y - camera_y))
        else:
            color = (
                int(GRASS_GREEN[0] * self.grass_level + GRASS_BROWN[0] * (1 - self.grass_level)),
                int(GRASS_GREEN[1] * self.grass_level + GRASS_BROWN[1] * (1 - self.grass_level)),
                int(GRASS_GREEN[2] * self.grass_level + GRASS_BROWN[2] * (1 - self.grass_level))
            )
            pygame.draw.rect(screen, color, (self.pos.x - camera_x, self.pos.y - camera_y, TILE_SIZE, TILE_SIZE))

    def harvest(self, amount):
        old_level = self.grass_level
        self.grass_level = max(0.0, self.grass_level - amount)
        harvested = old_level - self.grass_level
        return harvested

    def regrow(self, amount):
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

    def draw(self, screen, camera_x, camera_y):
        if self._image:
            screen.blit(self._image, (self.pos.x - camera_x, self.pos.y - camera_y))
        else:
            pygame.draw.rect(screen, GRASS_BROWN, (self.pos.x - camera_x, self.pos.y - camera_y, TILE_SIZE, TILE_SIZE))

# Tree class
class Tree(SimpleTile):
    _image = None
    _selected = False

    def __init__(self, x, y):
        super().__init__(x, y)
        if Tree._image is None:
            Tree.load_image()
        self.pos.x = x - TILE_SIZE / 2  # Adjust x to center of tile
        self.pos.y = y - TILE_SIZE / 2  # Adjust y to center of tile

    @classmethod
    def load_image(cls):
        try:
            cls._image = pygame.image.load("../tree.png").convert_alpha()
            original_width, original_height = cls._image.get_size()
            scale_factor = min(TILE_SIZE / original_width, TILE_SIZE / original_height)
            new_width = int(original_width * scale_factor)
            new_height = int(original_height * scale_factor)
            cls._image = pygame.transform.scale(cls._image, (new_width, new_height))
        except (pygame.error, FileNotFoundError) as e:
            print(f"Failed to load tree.png: {e}")
            cls._image = None

    def draw(self, screen, camera_x, camera_y):
        if self._image:
            image_rect = self._image.get_rect(center=(int(self.pos.x + TILE_SIZE / 2 - camera_x), int(self.pos.y + TILE_SIZE / 2 - camera_y)))
            screen.blit(self._image, image_rect)
            if self._selected:
                pygame.draw.rect(screen, YELLOW, (self.pos.x - camera_x, self.pos.y - camera_y, TILE_SIZE, TILE_SIZE), 2)  # Yellow border when selected
        else:
            pygame.draw.rect(screen, TREE_GREEN, (self.pos.x - camera_x, self.pos.y - camera_y, TILE_SIZE, TILE_SIZE))

    def is_clicked(self, click_pos, camera_x, camera_y):
        adjusted_pos = Vector2(click_pos.x + camera_x, click_pos.y + camera_y)
        tile_rect = pygame.Rect(self.pos.x, self.pos.y, TILE_SIZE, TILE_SIZE)
        return tile_rect.collidepoint(adjusted_pos)

    def set_selected(self, selected):
        self._selected = selected

# Unit class
class Unit:
    _images = {}
    _unit_icons = {}

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
            cls._images[cls_name] = pygame.image.load(f"{cls_name.lower()}.png")
            cls._images[cls_name] = pygame.transform.scale(cls._images[cls_name], (int(size), int(size)))
        except (pygame.error, FileNotFoundError) as e:
            print(f"Failed to load {cls_name.lower()}.png: {e}")
            cls._images[cls_name] = None
        try:
            cls._unit_icons[cls_name] = pygame.image.load(f"{cls_name.lower()}_icon.png")
            cls._unit_icons[cls_name] = pygame.transform.scale(cls._unit_icons[cls_name], (ICON_SIZE, ICON_SIZE))
        except (pygame.error, FileNotFoundError) as e:
            print(f"Failed to load {cls_name.lower()}_icon.png: {e}")
            cls._unit_icons[cls_name] = None

    def draw(self, screen, camera_x, camera_y):
        cls_name = self.__class__.__name__
        image = self._images.get(cls_name)
        if not image:
            color = GREEN if self.selected else self.color
            pygame.draw.rect(screen, color, (self.pos.x - self.size / 2 - camera_x, self.pos.y - self.size / 2 - camera_y, self.size, self.size))
        else:
            image_rect = image.get_rect(center=(int(self.pos.x - camera_x), int(self.pos.y - camera_y)))
            screen.blit(image, image_rect)
        if self.selected:
            pygame.draw.rect(screen, self.player_color, (self.pos.x - self.size / 2 - camera_x, self.pos.y - self.size / 2 - camera_y, self.size, self.size), 1)

    def move(self, units):
        self.velocity = Vector2(0, 0)
        if self.target:
            direction = self.target - self.pos
            distance_to_target = direction.length()
            if distance_to_target > 2:
                self.velocity = direction.normalize() * self.speed
            else:
                self.pos = Vector2(self.target)
                self.target = None
        self.velocity *= self.damping
        self.pos += self.velocity

    def resolve_collisions(self, units):
        for other in units:
            if other is not self:
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
                    if distance < corner_radius and distance > 0:
                        if self.target is not None and self.target != nearest_corner:
                            overlap = corner_radius - distance
                            direction = (self.pos - nearest_corner).normalize()
                            self.pos += direction * overlap
                elif isinstance(self, Cow) and isinstance(other, Building) and not isinstance(other, Barn):
                    distance = self.pos.distance_to(other.pos)
                    combined_min_distance = (self.size + other.size) / 2
                    if distance < combined_min_distance and distance > 0:
                        overlap = combined_min_distance - distance
                        direction = (self.pos - other.pos).normalize()
                        self.pos += direction * overlap
                elif not isinstance(self, Cow) and isinstance(other, Building):
                    distance = self.pos.distance_to(other.pos)
                    combined_min_distance = (self.size + other.size) / 2
                    if distance < combined_min_distance and distance > 0:
                        overlap = combined_min_distance - distance
                        direction = (self.pos - other.pos).normalize()
                        self.pos += direction * overlap
                elif not isinstance(self, Building) and not isinstance(other, Building):
                    distance = self.pos.distance_to(other.pos)
                    combined_min_distance = (self.size + other.size) / 2
                    if distance < combined_min_distance and distance > 0:
                        overlap = combined_min_distance - distance
                        direction = (self.pos - other.pos).normalize()
                        correction = direction * overlap * 0.5
                        self.pos += correction
                        other.pos -= correction
                elif isinstance(self, Building):
                    pass

    def keep_in_bounds(self):
        self.pos.x = max(self.size / 2, min(MAP_WIDTH - self.size / 2, self.pos.x))
        self.pos.y = max(self.size / 2, min(MAP_HEIGHT - self.size / 2, self.pos.y))

    def harvest_grass(self, grass_tiles):
        pass

    def is_clicked(self, click_pos, camera_x, camera_y):
        adjusted_pos = Vector2(click_pos.x + camera_x, click_pos.y + camera_y)
        return (abs(adjusted_pos.x - self.pos.x) <= self.size / 2 and
                abs(adjusted_pos.y - self.pos.y) <= self.size / 2)

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
        self.harvest_rate = 60 / 60  # Fixed to 1 per second at 60 FPS

# TownCenter class
class TownCenter(Building):
    def __init__(self, x, y, player_id, player_color):
        super().__init__(x, y, size=60, color=TOWN_CENTER_GRAY, player_id=player_id, player_color=player_color)

# Axeman class
class Axeman(Unit):
    def __init__(self, x, y, player_id, player_color):
        super().__init__(x, y, size=16, speed=2, color=RED, player_id=player_id, player_color=player_color)

# Knight class
class Knight(Unit):
    def __init__(self, x, y, player_id, player_color):
        super().__init__(x, y, size=16, speed=4, color=BLUE, player_id=player_id, player_color=player_color)

# Archer class
class Archer(Unit):
    def __init__(self, x, y, player_id, player_color):
        super().__init__(x, y, size=16, speed=3, color=YELLOW, player_id=player_id, player_color=player_color)

# Cow class
class Cow(Unit):
    def __init__(self, x, y, player_id, player_color):
        super().__init__(x, y, size=16, speed=4, color=BROWN, player_id=player_id, player_color=player_color)
        self.harvest_rate = 0.01
        self.assigned_corner = None

    def draw(self, screen, camera_x, camera_y):
        cls_name = self.__class__.__name__
        image = self._images.get(cls_name)
        if not image:
            color = GREEN if self.selected else self.color
            pygame.draw.rect(screen, color, (self.pos.x - self.size / 2 - camera_x, self.pos.y - self.size / 2 - camera_y, self.size, self.size))
        else:
            image_rect = image.get_rect(center=(int(self.pos.x - camera_x), int(self.pos.y - camera_y)))
            screen.blit(image, image_rect)
        if self.selected:
            pygame.draw.rect(screen, self.player_color, (self.pos.x - self.size / 2 - camera_x, self.pos.y - self.size / 2 - camera_y, self.size, self.size), 1)
        # Draw health bar
        bar_width = 16
        bar_height = 4
        bar_offset = 2
        bar_x = self.pos.x - bar_width / 2 - camera_x
        bar_y = self.pos.y - self.size / 2 - bar_height - bar_offset - camera_y
        pygame.draw.rect(screen, BLACK, (bar_x, bar_y, bar_width, bar_height))
        fill_width = (self.special / 100) * bar_width
        pygame.draw.rect(screen, WHITE, (bar_x, bar_y, fill_width, bar_height))

    def move(self, units):
        self.velocity = Vector2(0, 0)
        if self.target:
            direction = self.target - self.pos
            distance_to_target = direction.length()
            if distance_to_target > 2:
                self.velocity = direction.normalize() * self.speed
            else:
                self.pos = Vector2(self.target)
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

    def harvest_grass(self, grass_tiles, barns, cow_in_barn, player):
        if self.is_in_barn_any(barns) and self.special > 0:
            return
        if self.special >= 100:
            if not self.target:
                available_barns = [barn for barn in barns if barn.player_id == self.player_id and isinstance(barn, Barn) and (barn not in cow_in_barn or cow_in_barn[barn] is None)]
                if available_barns:
                    closest_barn = min(available_barns, key=lambda barn: self.pos.distance_to(barn.pos))
                    # Target the bottom-left tile center of the barn
                    target_x = closest_barn.pos.x - closest_barn.size / 2 + TILE_SIZE / 2
                    target_y = closest_barn.pos.y + closest_barn.size / 2 - TILE_SIZE / 2
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
                        if isinstance(grass_tiles[adj_y][adj_x], GrassTile) and not isinstance(grass_tiles[adj_y][adj_x], Dirt) and grass_tiles[adj_y][adj_x].grass_level > 0.5:
                            self.target = Vector2(adj_x * TILE_SIZE + TILE_SIZE / 2, adj_y * TILE_SIZE + TILE_SIZE / 2)
                            break
                return
        if not self.target or self.velocity.length() < 0.5:
            tile_x = int(self.pos.x // TILE_SIZE)
            tile_y = int(self.pos.y // TILE_SIZE)
            if 0 <= tile_x < GRASS_COLS and 0 <= tile_y < GRASS_ROWS:
                if isinstance(grass_tiles[tile_y][tile_x], GrassTile) and not isinstance(grass_tiles[tile_y][tile_x], Dirt):
                    # Check if the current position has a tree overlay
                    has_tree = any(tree.pos.x <= self.pos.x <= tree.pos.x + TILE_SIZE and
                                 tree.pos.y <= self.pos.y <= tree.pos.y + TILE_SIZE
                                 for tree in tree_objects)
                    if not has_tree:
                        harvested = grass_tiles[tile_y][tile_x].harvest(self.harvest_rate)
                        self.special = min(100, self.special + harvested * 50)
                        if grass_tiles[tile_y][tile_x].grass_level == 0:
                            adjacent_tiles = [
                                (tile_x, tile_y - 1), (tile_x, tile_y + 1), (tile_x - 1, tile_y), (tile_x + 1, tile_y),
                                (tile_x - 1, tile_y - 1), (tile_x + 1, tile_y - 1), (tile_x - 1, tile_y + 1), (tile_x + 1, tile_y + 1)
                            ]
                            random.shuffle(adjacent_tiles)
                            for adj_x, adj_y in adjacent_tiles:
                                if 0 <= adj_x < GRASS_COLS and 0 <= adj_y < GRASS_ROWS:
                                    if isinstance(grass_tiles[adj_y][adj_x], GrassTile) and not isinstance(grass_tiles[adj_y][adj_x], Dirt) and grass_tiles[adj_y][adj_x].grass_level > 0.5:
                                        self.target = Vector2(adj_x * TILE_SIZE + TILE_SIZE / 2, adj_y * TILE_SIZE + TILE_SIZE / 2)
                                        break

# Player class
class Player:
    def __init__(self, player_id, color, start_x, start_y):
        self.player_id = player_id
        self.color = color
        self.milk = 0.0
        self.units = []
        self.cow_in_barn = {}
        offset_x = start_x
        offset_y = start_y
        self.units.extend([
            Axeman(offset_x + 100, offset_y + 100, player_id, self.color),
            Knight(offset_x + 150, offset_y + 100, player_id, self.color),
            Archer(offset_x + 200, offset_y + 100, player_id, self.color),
            Cow(offset_x + 300, offset_y + 100, player_id, self.color),
            Barn(offset_x + 310, offset_y + 150, player_id, self.color),
            TownCenter(offset_x + 150, offset_y + 150, player_id, self.color)
        ])

    def select_all_units(self):
        for unit in self.units:
            unit.selected = True

    def deselect_all_units(self):
        for unit in self.units:
            unit.selected = False

# Create grass field
grass_tiles = [[GrassTile(col * TILE_SIZE, row * TILE_SIZE) for col in range(GRASS_COLS)] for row in range(GRASS_ROWS)]

# Create players
players = [
    Player(1, BLUE, 0, 0),
    Player(2, PURPLE, 0, 600)
]

# Combine all units
all_units = []
for player in players:
    all_units.extend(player.units)

# Place Dirt tiles under buildings
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

def is_tile_occupied(row, col, units):
    tile_rect = pygame.Rect(col * TILE_SIZE, row * TILE_SIZE, TILE_SIZE, TILE_SIZE)
    for unit in units:
        unit_rect = pygame.Rect(unit.pos.x - unit.size / 2, unit.pos.y - unit.size / 2, unit.size, unit.size)
        if tile_rect.colliderect(unit_rect):
            return True
    return False

# Create a separate list for tree objects to overlay on 30% of grass tiles in small patches
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
            tree_objects.append(Tree(c * TILE_SIZE + TILE_SIZE / 2, r * TILE_SIZE + TILE_SIZE / 2))
            eligible_tiles.remove((r, c))
    if (center_row, center_col) in eligible_tiles:
        eligible_tiles.remove((center_row, center_col))

# Selection rectangle and player selection mode
selection_start = None
selection_end = None
selecting = False
current_player = None

# Button rectangles
button_player1 = pygame.Rect(BUTTON_PLAYER1_POS[0], BUTTON_PLAYER1_POS[1], BUTTON_WIDTH, BUTTON_HEIGHT)
button_player2 = pygame.Rect(BUTTON_PLAYER2_POS[0], BUTTON_PLAYER2_POS[1], BUTTON_WIDTH, BUTTON_HEIGHT)
button_spawn_cow = pygame.Rect(BUTTON_SPAWN_COW_POS[0], BUTTON_SPAWN_COW_POS[1], BUTTON_WIDTH, BUTTON_HEIGHT)

# Game loop
running = True
font = pygame.font.SysFont(None, 24)
while running:
    # Update camera for arrow key scrolling
    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT]:
        camera_x = max(0, camera_x - SCROLL_SPEED)
    if keys[pygame.K_RIGHT]:
        camera_x = min(MAP_WIDTH - VIEW_WIDTH, camera_x + SCROLL_SPEED)
    if keys[pygame.K_UP]:
        camera_y = max(0, camera_y - SCROLL_SPEED)
    if keys[pygame.K_DOWN]:
        camera_y = min(MAP_HEIGHT - VIEW_HEIGHT, camera_y + SCROLL_SPEED)

    barn_selected = False
    if current_player is not None:
        for player in players:
            if player.player_id == current_player.player_id:
                for unit in player.units:
                    if isinstance(unit, Barn) and unit.selected:
                        barn_selected = True
                        break

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                mouse_pos = Vector2(event.pos)
                if VIEW_X <= mouse_pos.x <= VIEW_X + VIEW_WIDTH and VIEW_Y <= mouse_pos.y <= VIEW_Y + VIEW_HEIGHT:
                    click_pos = Vector2(mouse_pos.x - VIEW_X + camera_x, mouse_pos.y - VIEW_Y + camera_y)
                    unit_clicked = None
                    for unit in all_units:
                        if current_player and unit.player_id == current_player.player_id and unit.is_clicked(Vector2(mouse_pos.x - VIEW_X, mouse_pos.y - VIEW_Y), camera_x, camera_y):
                            unit_clicked = unit
                            break
                    if unit_clicked:
                        for player in players:
                            player.deselect_all_units()
                        unit_clicked.selected = True
                        print(f"Selected unit at {unit_clicked.pos} (Player {unit_clicked.player_id})")
                    else:
                        selection_start = click_pos
                        selecting = True
                        print(f"Selection started at: {selection_start}")
                elif button_player1.collidepoint(event.pos):
                    current_player = players[0]
                    for player in players:
                        player.deselect_all_units()
                    print("Selected Player 1; only Player 1 units can be selected")
                elif button_player2.collidepoint(event.pos):
                    current_player = players[1]
                    for player in players:
                        player.deselect_all_units()
                    print("Selected Player 2; only Player 2 units can be selected")
                elif button_spawn_cow.collidepoint(event.pos) and current_player is not None and barn_selected:
                    for player in players:
                        if player.player_id == current_player.player_id and player.milk >= 500:
                            selected_barn = None
                            for unit in player.units:
                                if isinstance(unit, Barn) and unit.selected:
                                    selected_barn = unit
                                    break
                            if selected_barn:
                                new_cow = Cow(
                                    selected_barn.pos.x + selected_barn.size / 2 + 20,
                                    selected_barn.pos.y,
                                    player.player_id,
                                    player.color
                                )
                                player.units.append(new_cow)
                                all_units.append(new_cow)
                                player.milk -= 500
                                print(f"Spawned new cow for Player {player.player_id} at {new_cow.pos}")
            elif event.button == 3:
                mouse_pos = Vector2(event.pos)
                if VIEW_X <= mouse_pos.x <= VIEW_X + VIEW_WIDTH and VIEW_Y <= mouse_pos.y <= VIEW_Y + VIEW_HEIGHT:
                    for unit in all_units:
                        if unit.selected and not isinstance(unit, Building):
                            unit.target = Vector2(mouse_pos.x - VIEW_X + camera_x, mouse_pos.y - VIEW_Y + camera_y)
                            print(f"Set target for unit at {unit.pos} to {unit.target}")
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1 and selecting and current_player is not None:
                mouse_pos = Vector2(event.pos)
                if VIEW_X <= mouse_pos.x <= VIEW_X + VIEW_WIDTH and VIEW_Y <= mouse_pos.y <= VIEW_Y + VIEW_HEIGHT:
                    selection_end = Vector2(mouse_pos.x - VIEW_X + camera_x, mouse_pos.y - VIEW_Y + camera_y)
                    selecting = False
                    for player in players:
                        if player.player_id == current_player.player_id:
                            for unit in player.units:
                                unit.selected = (min(selection_start.x, selection_end.x) <= unit.pos.x <= max(selection_start.x, selection_end.x) and
                                                min(selection_start.y, selection_end.y) <= unit.pos.y <= max(selection_start.y, selection_end.y))
                    print(f"Selection ended at: {selection_end}")
        elif event.type == pygame.MOUSEMOTION and selecting:
            mouse_pos = Vector2(event.pos)
            if VIEW_X <= mouse_pos.x <= VIEW_X + VIEW_WIDTH and VIEW_Y <= mouse_pos.y <= VIEW_Y + VIEW_HEIGHT:
                selection_end = Vector2(mouse_pos.x - VIEW_X + camera_x, mouse_pos.y - VIEW_Y + camera_y)

    # Update grass regrowth
    regrowth_rate = 0.001
    for row in grass_tiles:
        for tile in row:
            tile.regrow(regrowth_rate)

    # Update units and handle cow-barn interactions
    for player in players:
        barns = [unit for unit in player.units if isinstance(unit, Barn)]
        for unit in player.units:
            unit.move(all_units)
            unit.resolve_collisions(all_units)
            unit.keep_in_bounds()
            if isinstance(unit, Cow):
                unit.harvest_grass(grass_tiles, barns, player.cow_in_barn, player)
        for barn in barns:
            if barn in player.cow_in_barn and player.cow_in_barn[barn]:
                cow = player.cow_in_barn[barn]
                if cow.is_in_barn(barn):
                    if player.milk < 1000 and cow.special > 0:
                        cow.special = max(0, cow.special - barn.harvest_rate)
                        player.milk = min(1000, player.milk + barn.harvest_rate)
                    if cow.special <= 0:
                        player.cow_in_barn[barn] = None
                else:
                    player.cow_in_barn[barn] = None
            else:
                for unit in player.units:
                    if isinstance(unit, Cow) and unit.is_in_barn(barn):
                        player.cow_in_barn[barn] = unit
                        break

    # Draw only visible tiles with grass first, then trees as overlays
    start_col = int(camera_x // TILE_SIZE)
    end_col = min(GRASS_COLS, int((camera_x + VIEW_WIDTH) // TILE_SIZE) + 1)
    start_row = int(camera_y // TILE_SIZE)
    end_row = min(GRASS_ROWS, int((camera_y + VIEW_HEIGHT) // TILE_SIZE) + 1)
    # First pass: Draw dirt tiles (under buildings)
    for row in range(start_row, end_row):
        for col in range(start_col, end_col):
            tile = grass_tiles[row][col]
            if isinstance(tile, Dirt):
                tile.draw(screen, camera_x - VIEW_X, camera_y - VIEW_Y)
    # Second pass: Draw grass tiles
    for row in range(start_row, end_row):
        for col in range(start_col, end_col):
            tile = grass_tiles[row][col]
            if isinstance(tile, GrassTile):
                tile.draw(screen, camera_x - VIEW_X, camera_y - VIEW_Y)
    # Third pass: Draw tree objects on top of grass, excluding UI area
    for tree in tree_objects:
        # Convert tree position to screen coordinates
        screen_y = tree.pos.y + TILE_SIZE / 2 - camera_y  # Adjusted for centered position
        tree.draw(screen, camera_x - VIEW_X, camera_y - VIEW_Y)

    # Draw selection rectangle
    if selecting and selection_start and selection_end and current_player is not None:
        rect = pygame.Rect(
            min(selection_start.x - camera_x + VIEW_X, selection_end.x - camera_x + VIEW_X),
            min(selection_start.y - camera_y + VIEW_Y, selection_end.y - camera_y + VIEW_Y),
            abs(selection_end.x - selection_start.x),
            abs(selection_end.y - selection_start.y)
        )
        selection_color = next(player.color for player in players if player.player_id == current_player.player_id)
        pygame.draw.rect(screen, selection_color, rect, 3)

    # Draw buildings first, then units to ensure units are on top
    for unit in all_units:
        if isinstance(unit, Building):
            screen_y = unit.pos.y + TILE_SIZE / 2 - camera_y
            unit.draw(screen, camera_x - VIEW_X, camera_y - VIEW_Y)
    for unit in all_units:
        if not isinstance(unit, Building):
            screen_y = unit.pos.y + TILE_SIZE / 2 - camera_y
            unit.draw(screen, camera_x - VIEW_X, camera_y - VIEW_Y)

    # Draw

    # Left Border
    pygame.draw.rect(screen, PANEL_COLOR, (0, 0, VIEW_X, SCREEN_HEIGHT))
    # Right Border
    pygame.draw.rect(screen, PANEL_COLOR, (SCREEN_WIDTH - VIEW_X, 0, SCREEN_WIDTH, SCREEN_HEIGHT))
    # Top Border
    pygame.draw.rect(screen, PANEL_COLOR, (0, 0, SCREEN_WIDTH, VIEW_Y))
    # Bottom Border
    pygame.draw.rect(screen, PANEL_COLOR, (VIEW_X, PANEL_Y, VIEW_WIDTH, PANEL_HEIGHT))

    # Draw UI
    player_button_color_1 = BLUE if current_player and current_player.player_id == 1 else LIGHT_GRAY
    player_button_color_2 = PURPLE if current_player and current_player.player_id == 2 else LIGHT_GRAY
    spawn_button_color = HIGHLIGHT_GRAY if barn_selected and current_player is not None and current_player.milk >= 500 else LIGHT_GRAY
    pygame.draw.rect(screen, player_button_color_1, button_player1)
    pygame.draw.rect(screen, player_button_color_2, button_player2)
    pygame.draw.rect(screen, spawn_button_color, button_spawn_cow)
    player1_text = font.render("Select Player 1", True, BLACK)
    player2_text = font.render("Select Player 2", True, BLACK)
    spawn_cow_text = font.render("Spawn Cow", True, BLACK)
    screen.blit(player1_text, (BUTTON_PLAYER1_POS[0] + 10, BUTTON_PLAYER1_POS[1] + 10))
    screen.blit(player2_text, (BUTTON_PLAYER2_POS[0] + 10, BUTTON_PLAYER2_POS[1] + 10))
    screen.blit(spawn_cow_text, (BUTTON_SPAWN_COW_POS[0] + 10, BUTTON_SPAWN_COW_POS[1] + 10))

    # Draw unit icons for selected units
    icon_x = VIEW_X + 10
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
    fps = clock.get_fps()
    fps_text = font.render(f"FPS: {int(fps)}", True, WHITE)
    screen.blit(fps_text, (VIEW_X + 10, PANEL_Y + 60))
    for i, player in enumerate(players):
        milk_text = font.render(f"Player {player.player_id} Milk: {player.milk:.2f}", True, player.color)
        screen.blit(milk_text, (VIEW_X + 10, PANEL_Y + 90 + i * 30))
    selected_count = sum(1 for unit in all_units if unit.selected and unit.player_id == (current_player.player_id if current_player else 0))
    selected_text = font.render(f"Selected Units: {selected_count}", True, BLACK)
    screen.blit(selected_text, (VIEW_X + 10, PANEL_Y + 150))

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
sys.exit()

