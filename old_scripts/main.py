import pygame
import sys
import random
from pygame.math import Vector2

# Initialize Pygame
pygame.init()

# Screen settings
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Simple RTS Game with Cow Special Bar")
clock = pygame.time.Clock()

# Colors
WHITE = (255, 255, 255)  # Cow and selection rectangle
BLUE = (0, 0, 255)  # Soldier
RED = (255, 0, 0)  # Tank and special bar background
YELLOW = (255, 255, 0)  # Scout
GREEN = (0, 255, 0)  # Selected unit highlight and special bar fill
GRASS_GREEN = (0, 100, 0)  # Full grass
GRASS_BROWN = (139, 69, 19)  # Depleted grass

# Grass tile settings
TILE_SIZE = 20
GRASS_ROWS = SCREEN_HEIGHT // TILE_SIZE  # 30 rows
GRASS_COLS = SCREEN_WIDTH // TILE_SIZE  # 40 cols


# Grass field
class GrassTile:
    def __init__(self, x, y):
        self.pos = Vector2(x, y)
        self.grass_level = 1.0

    def draw(self, screen):
        color = (
            int(GRASS_GREEN[0] * self.grass_level + GRASS_BROWN[0] * (1 - self.grass_level)),
            int(GRASS_GREEN[1] * self.grass_level + GRASS_BROWN[1] * (1 - self.grass_level)),
            int(GRASS_GREEN[2] * self.grass_level + GRASS_BROWN[2] * (1 - self.grass_level))
        )
        pygame.draw.rect(screen, color, (self.pos.x, self.pos.y, TILE_SIZE, TILE_SIZE))

    def harvest(self, amount):
        old_level = self.grass_level
        self.grass_level = max(0.0, self.grass_level - amount)
        harvested = old_level - self.grass_level
        print(f"Harvesting tile at ({self.pos.x}, {self.pos.y}): grass_level = {self.grass_level}")
        return harvested

    def regrow(self, amount):
        self.grass_level = min(1.0, self.grass_level + amount)


# Create grass field
grass_tiles = [[GrassTile(col * TILE_SIZE, row * TILE_SIZE) for col in range(GRASS_COLS)] for row in range(GRASS_ROWS)]


# Base Unit class
class Unit:
    def __init__(self, x, y, size, speed, color):
        self.pos = Vector2(x, y)  # No snapping
        self.target = None
        self.speed = speed
        self.selected = False
        self.size = size
        self.min_distance = self.size
        self.color = color
        self.velocity = Vector2(0, 0)
        self.damping = 0.95
        self.hp = 100
        self.mana = 0
        self.special = 0

    def draw(self, screen):
        color = GREEN if self.selected else self.color
        pygame.draw.rect(screen, color, (self.pos.x - self.size / 2, self.pos.y - self.size / 2, self.size, self.size))

    def move(self, units):
        self.velocity = Vector2(0, 0)
        if self.target:
            direction = self.target - self.pos
            distance_to_target = direction.length()
            if distance_to_target > 5:
                self.velocity = direction.normalize() * self.speed
                print(f"Moving unit at {self.pos} toward {self.target}, velocity: {self.velocity}")
            else:
                self.target = None
        self.velocity *= self.damping
        self.pos += self.velocity

    def resolve_collisions(self, units):
        for other in units:
            if other is not self:
                distance = self.pos.distance_to(other.pos)
                combined_min_distance = (self.size + other.size) / 2
                if distance < combined_min_distance and distance > 0:
                    overlap = combined_min_distance - distance
                    direction = (self.pos - other.pos).normalize()
                    correction = direction * overlap * 0.5
                    self.pos += correction
                    other.pos -= correction

    def keep_in_bounds(self):
        self.pos.x = max(self.size / 2, min(SCREEN_WIDTH - self.size / 2, self.pos.x))
        self.pos.y = max(self.size / 2, min(SCREEN_HEIGHT - self.size / 2, self.pos.y))

    def harvest_grass(self, grass_tiles):
        pass

    def is_clicked(self, click_pos):
        return (abs(click_pos.x - self.pos.x) <= self.size / 2 and
                abs(click_pos.y - self.pos.y) <= self.size / 2)


class Soldier(Unit):
    def __init__(self, x, y):
        super().__init__(x, y, size=16, speed=5, color=BLUE)


class Tank(Unit):
    def __init__(self, x, y):
        super().__init__(x, y, size=16, speed=2, color=RED)


class Scout(Unit):
    def __init__(self, x, y):
        super().__init__(x, y, size=16, speed=8, color=YELLOW)


class Cow(Unit):
    def __init__(self, x, y):
        super().__init__(x, y, size=16, speed=4, color=WHITE)
        self.harvest_rate = 0.01

    def draw(self, screen):
        # Draw the Cow unit
        color = GREEN if self.selected else self.color
        pygame.draw.rect(screen, color, (self.pos.x - self.size / 2, self.pos.y - self.size / 2, self.size, self.size))
        # Draw special bar above the Cow
        bar_width = 16
        bar_height = 4
        bar_offset = 2  # Space between Cow and bar
        bar_x = self.pos.x - bar_width / 2
        bar_y = self.pos.y - self.size / 2 - bar_height - bar_offset
        # Background (empty bar)
        pygame.draw.rect(screen, RED, (bar_x, bar_y, bar_width, bar_height))
        # Fill (special level)
        fill_width = (self.special / 100) * bar_width
        pygame.draw.rect(screen, GREEN, (bar_x, bar_y, fill_width, bar_height))

    def move(self, units):
        self.velocity = Vector2(0, 0)
        if self.target and self.special < 100:  # Only move if not full
            direction = self.target - self.pos
            distance_to_target = direction.length()
            if distance_to_target > 5:
                self.velocity = direction.normalize() * self.speed
                print(f"Moving cow at {self.pos} toward {self.target}, velocity: {self.velocity}")
            else:
                self.pos = Vector2(self.target)
                self.target = None
                print(f"Cow centered at {self.pos}")
        self.velocity *= self.damping
        self.pos += self.velocity

    def harvest_grass(self, grass_tiles):
        if self.special >= 100:  # Stop harvesting if full
            print(f"Cow at {self.pos} is full (special = {self.special}), stopping harvest")
            return
        if not self.target or self.velocity.length() < 0.5:
            tile_x = int(self.pos.x // TILE_SIZE)
            tile_y = int(self.pos.y // TILE_SIZE)
            if 0 <= tile_x < GRASS_COLS and 0 <= tile_y < GRASS_ROWS:
                harvested = grass_tiles[tile_y][tile_x].harvest(self.harvest_rate)
                self.special = min(100, self.special + harvested * 5)  # 0.01 grass = 0.05 special
                print(f"Cow at {self.pos} special = {self.special}")
                if grass_tiles[tile_y][tile_x].grass_level == 0:
                    adjacent_tiles = [
                        (tile_x, tile_y - 1),  # Up
                        (tile_x, tile_y + 1),  # Down
                        (tile_x - 1, tile_y),  # Left
                        (tile_x + 1, tile_y),  # Right
                        (tile_x - 1, tile_y - 1),  # Up-Left
                        (tile_x + 1, tile_y - 1),  # Up-Right
                        (tile_x - 1, tile_y + 1),  # Down-Left
                        (tile_x + 1, tile_y + 1)  # Down-Right
                    ]
                    random.shuffle(adjacent_tiles)
                    for adj_x, adj_y in adjacent_tiles:
                        if 0 <= adj_x < GRASS_COLS and 0 <= adj_y < GRASS_ROWS:
                            if grass_tiles[adj_y][adj_x].grass_level > 0.5:
                                self.target = Vector2(adj_x * TILE_SIZE + TILE_SIZE / 2, adj_y * TILE_SIZE + TILE_SIZE / 2)
                                print(f"Cow at {self.pos} moving to adjacent tile ({adj_x}, {adj_y}) with grass_level {grass_tiles[adj_y][adj_x].grass_level}")
                                break


# Create units
units = [
    Soldier(100, 100),
    Soldier(150, 150),
    Tank(200, 100),
    Tank(250, 150),
    Scout(300, 100),
    Scout(350, 150),
    Cow(400, 100),
    Cow(450, 150)
]

# Selection rectangle
selection_start = None
selection_end = None
selecting = False

# Game loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                click_pos = Vector2(event.pos)
                unit_clicked = None
                for unit in units:
                    if unit.is_clicked(click_pos):
                        unit_clicked = unit
                        break
                if unit_clicked:
                    for unit in units:
                        unit.selected = (unit == unit_clicked)
                    print(f"Selected unit at {unit_clicked.pos}")
                else:
                    selection_start = click_pos
                    selecting = True
                    print(f"Selection started at: {selection_start}")
            elif event.button == 3:
                for unit in units:
                    if unit.selected:
                        unit.target = Vector2(event.pos)  # No snapping, allows manual movement
                        print(f"Set target for unit at {unit.pos} to {unit.target}")
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1 and selecting:
                selection_end = Vector2(event.pos)
                selecting = False
                print(f"Selection ended at: {selection_end}")
                for unit in units:
                    unit.selected = (min(selection_start.x, selection_end.x) <= unit.pos.x <= max(selection_start.x, selection_end.x) and
                                     min(selection_start.y, selection_end.y) <= unit.pos.y <= max(selection_start.y, selection_end.y))
        elif event.type == pygame.MOUSEMOTION and selecting:
            selection_end = Vector2(event.pos)
            print(f"Selection updating: {selection_start}, {selection_end}")

    # Update grass regrowth
    regrowth_rate = 0.001
    for row in grass_tiles:
        for tile in row:
            tile.regrow(regrowth_rate)

    # Update units
    for unit in units:
        unit.move(units)
        unit.resolve_collisions(units)
        unit.keep_in_bounds()
        unit.harvest_grass(grass_tiles)

    # Draw
    screen.fill(WHITE)

    # Draw grass tiles
    for row in grass_tiles:
        for tile in row:
            tile.draw(screen)

    # Draw selection rectangle
    if selecting and selection_start and selection_end:
        rect = pygame.Rect(
            min(selection_start.x, selection_end.x),
            min(selection_start.y, selection_end.y),
            abs(selection_end.x - selection_start.x),
            abs(selection_end.y - selection_start.y)
        )
        pygame.draw.rect(screen, WHITE, rect, 3)
        print(f"Drawing selection rect: {rect}")

    # Draw units
    for unit in units:
        unit.draw(screen)

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
sys.exit()