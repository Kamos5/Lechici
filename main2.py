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
pygame.display.set_caption("Simple RTS Game with Barn Corner Collisions")
clock = pygame.time.Clock()

# Colors
WHITE = (255, 255, 255)  # Selection rectangle
BLUE = (0, 0, 255)  # Soldier and milk text
RED = (255, 0, 0)  # Tank and Cow special bar background
YELLOW = (255, 255, 0)  # Scout
GREEN = (0, 255, 0)  # Selected unit highlight and Cow special bar fill
GRASS_GREEN = (0, 100, 0)  # Full grass
GRASS_BROWN = (139, 69, 19)  # Depleted grass
GRAY = (128, 128, 128)  # Barn

# Grass tile settings
TILE_SIZE = 20
GRASS_ROWS = SCREEN_HEIGHT // TILE_SIZE  # 30 rows
GRASS_COLS = SCREEN_WIDTH // TILE_SIZE  # 40 cols

# Global variables
milk = 0.0  # Tracks total milk collected, max 1000
cow_in_barn = None  # Tracks which Cow is in the Barn (None if empty)


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
        self.pos = Vector2(x, y)
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
                # Handle Cow-Barn collisions at corners
                if isinstance(self, Cow) and isinstance(other, Barn):
                    barn_corners = [
                        Vector2(other.pos.x - other.size / 2, other.pos.y - other.size / 2),  # Top-left
                        Vector2(other.pos.x + other.size / 2, other.pos.y - other.size / 2),  # Top-right
                        Vector2(other.pos.x - other.size / 2, other.pos.y + other.size / 2),  # Bottom-left
                        Vector2(other.pos.x + other.size / 2, other.pos.y + other.size / 2)  # Bottom-right
                    ]
                    # Find nearest corner
                    nearest_corner = min(barn_corners, key=lambda corner: self.pos.distance_to(corner))
                    distance = self.pos.distance_to(nearest_corner)
                    corner_radius = 8  # Collision radius around corners
                    combined_min_distance = (self.size + other.size) / 2
                    if distance < corner_radius:
                        overlap = corner_radius - distance
                        direction = (self.pos - nearest_corner).normalize()
                        self.pos += direction * overlap  # Move only the Cow
                        print(f"Cow at {self.pos} collided with Barn corner at {nearest_corner}")
                    continue
                # Handle Barn-other collisions (Barn immovable)
                elif isinstance(self, Barn) and not isinstance(other, Barn) and not isinstance(other, Cow):
                    distance = self.pos.distance_to(other.pos)
                    combined_min_distance = (self.size + other.size) / 2
                    if distance < combined_min_distance and distance > 0:
                        overlap = combined_min_distance - distance
                        direction = (other.pos - self.pos).normalize()
                        other.pos += direction * overlap  # Move only the other unit
                        print(f"Barn at {self.pos} pushed unit at {other.pos}")
                elif isinstance(other, Barn) and not isinstance(self, Barn):
                    distance = self.pos.distance_to(other.pos)
                    combined_min_distance = (self.size + other.size) / 2
                    if distance < combined_min_distance and distance > 0:
                        overlap = combined_min_distance - distance
                        direction = (self.pos - other.pos).normalize()
                        self.pos += direction * overlap  # Move only self (not Barn)
                        print(f"Unit at {self.pos} pushed by Barn at {other.pos}")
                # Standard collision for non-Barn pairs
                else:
                    distance = self.pos.distance_to(other.pos)
                    combined_min_distance = (self.size + other.size) / 2
                    if distance < combined_min_distance and distance > 0:
                        overlap = combined_min_distance - distance
                        direction = (self.pos - other.pos).normalize()
                        correction = direction * overlap * 0.5
                        self.pos += correction
                        other.pos -= correction
                        print(f"Collision between units at {self.pos} and {other.pos}")

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
        color = GREEN if self.selected else self.color
        pygame.draw.rect(screen, color, (self.pos.x - self.size / 2, self.pos.y - self.size / 2, self.size, self.size))
        bar_width = 16
        bar_height = 4
        bar_offset = 2
        bar_x = self.pos.x - bar_width / 2
        bar_y = self.pos.y - self.size / 2 - bar_height - bar_offset
        pygame.draw.rect(screen, RED, (bar_x, bar_y, bar_width, bar_height))
        fill_width = (self.special / 100) * bar_width
        pygame.draw.rect(screen, GREEN, (bar_x, bar_y, fill_width, bar_height))

    def move(self, units):
        self.velocity = Vector2(0, 0)
        if self.target:
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

    def is_in_barn(self, barn):
        return (barn.pos.x - barn.size / 2 <= self.pos.x <= barn.pos.x + barn.size / 2 and
                barn.pos.y - barn.size / 2 <= self.pos.y <= barn.pos.y + barn.size / 2)

    def harvest_grass(self, grass_tiles, barn, cow_in_barn):
        if self.special >= 100:
            if not self.target:  # Only set target if not already moving
                if cow_in_barn is None and not self.is_in_barn(barn):
                    self.target = Vector2(barn.pos)  # Move to Barn center
                    print(f"Cow at {self.pos} (special = {self.special}) moving to Barn at {barn.pos}")
                elif cow_in_barn is not self:
                    self.target = Vector2(barn.pos.x + barn.size / 2 + 6, barn.pos.y)  # Wait outside Barn
                    print(f"Cow at {self.pos} (special = {self.special}) moving to wait near Barn at {self.target}")
            return
        if not self.target or self.velocity.length() < 0.5:
            tile_x = int(self.pos.x // TILE_SIZE)
            tile_y = int(self.pos.y // TILE_SIZE)
            if 0 <= tile_x < GRASS_COLS and 0 <= tile_y < GRASS_ROWS:
                harvested = grass_tiles[tile_y][tile_x].harvest(self.harvest_rate)
                self.special = min(100, self.special + harvested * 20)  # 20 tiles = 100 special
                print(f"Cow at {self.pos} special = {self.special}")
                if grass_tiles[tile_y][tile_x].grass_level == 0:
                    adjacent_tiles = [
                        (tile_x, tile_y - 1),
                        (tile_x, tile_y + 1),
                        (tile_x - 1, tile_y),
                        (tile_x + 1, tile_y),
                        (tile_x - 1, tile_y - 1),
                        (tile_x + 1, tile_y - 1),
                        (tile_x - 1, tile_y + 1),
                        (tile_x + 1, tile_y + 1)
                    ]
                    random.shuffle(adjacent_tiles)
                    for adj_x, adj_y in adjacent_tiles:
                        if 0 <= adj_x < GRASS_COLS and 0 <= adj_y < GRASS_ROWS:
                            if grass_tiles[adj_y][adj_x].grass_level > 0.5:
                                self.target = Vector2(adj_x * TILE_SIZE + TILE_SIZE / 2, adj_y * TILE_SIZE + TILE_SIZE / 2)
                                print(f"Cow at {self.pos} moving to adjacent tile ({adj_x}, {adj_y}) with grass_level {grass_tiles[adj_y][adj_x].grass_level}")
                                break


class Barn(Unit):
    def __init__(self, x, y):
        super().__init__(x, y, size=48, speed=0, color=GRAY)

    def move(self, units):
        pass  # Barn is immovable


# Create units
units = [
    Soldier(100, 100),
    Soldier(150, 150),
    Tank(200, 100),
    Tank(250, 150),
    Scout(300, 100),
    Scout(350, 150),
    Cow(400, 100),
    Cow(450, 150),
    Barn(600, 400)
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
                    if unit.selected and not isinstance(unit, Barn):  # Prevent Barn from being targeted
                        unit.target = Vector2(event.pos)
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

    # Find the Barn
    barn = next(unit for unit in units if isinstance(unit, Barn))

    # Update units
    for unit in units:
        unit.move(units)
        unit.resolve_collisions(units)
        unit.keep_in_bounds()
        if isinstance(unit, Cow):
            unit.harvest_grass(grass_tiles, barn, cow_in_barn)

    # Handle Cow in Barn
    if cow_in_barn:
        if cow_in_barn.is_in_barn(barn):
            cow_in_barn.special = max(0, cow_in_barn.special - 0.1667)  # 10 per second at 60 FPS
            if milk < 1000:  # Cap milk at 1000
                milk = min(1000, milk + 0.1667)
            print(f"Cow at {cow_in_barn.pos} in Barn, special = {cow_in_barn.special}, milk = {milk}")
            if cow_in_barn.special <= 0:
                cow_in_barn = None  # Free the Barn
                print("Barn is now free")
        else:
            cow_in_barn = None  # Cow left the Barn
            print("Cow left Barn, Barn is now free")
    else:
        # Check for a Cow in the Barn
        for unit in units:
            if isinstance(unit, Cow) and unit.is_in_barn(barn):
                cow_in_barn = unit
                print(f"Cow at {unit.pos} entered Barn")
                break

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

    # Draw milk counter
    font = pygame.font.SysFont(None, 24)
    milk_text = font.render(f"Milk: {milk:.2f}", True, BLUE)
    screen.blit(milk_text, (10, 10))

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
sys.exit()