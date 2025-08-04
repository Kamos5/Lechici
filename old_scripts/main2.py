import pygame
import sys
import random
import math
from pygame.math import Vector2

# Initialize Pygame
pygame.init()

# Class definitions
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
        return harvested

    def regrow(self, amount):
        self.grass_level = min(1.0, self.grass_level + amount)

class Dirt(GrassTile):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.grass_level = 0.0

    def draw(self, screen):
        pygame.draw.rect(screen, GRASS_BROWN, (self.pos.x, self.pos.y, TILE_SIZE, TILE_SIZE))

    def regrow(self, amount):
        pass  # Dirt does not regrow

    def harvest(self, amount):
        return 0.0  # Dirt has no grass to harvest

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
                # Handle Cow-Barn corner collisions
                if isinstance(self, Cow) and isinstance(other, Barn):
                    barn_corners = [
                        Vector2(other.pos.x - other.size / 2, other.pos.y - other.size / 2),  # Top-left
                        Vector2(other.pos.x + other.size / 2, other.pos.y - other.size / 2),  # Top-right
                        Vector2(other.pos.x - other.size / 2, other.pos.y + other.size / 2),  # Bottom-left
                        Vector2(other.pos.x + other.size / 2, other.pos.y + other.size / 2)  # Bottom-right
                    ]
                    nearest_corner = min(barn_corners, key=lambda corner: self.pos.distance_to(corner))
                    distance = self.pos.distance_to(nearest_corner)
                    corner_radius = 10
                    if distance < corner_radius and distance > 0:
                        # Only push cow away if targeting a different corner
                        if self.target is not None and self.target != nearest_corner:
                            overlap = corner_radius - distance
                            direction = (self.pos - nearest_corner).normalize()
                            self.pos += direction * overlap
                            print(f"Cow at {self.pos} pushed away from barn corner at {nearest_corner}")
                # Handle Barn-other collisions (Barn immovable), but skip for Cows
                elif isinstance(self, Barn) and not isinstance(other, Barn) and not isinstance(other, Cow):
                    distance = self.pos.distance_to(other.pos)
                    combined_min_distance = (self.size + other.size) / 2
                    if distance < combined_min_distance and distance > 0:
                        overlap = combined_min_distance - distance
                        direction = (other.pos - self.pos).normalize()
                        other.pos += direction * overlap
                        print(f"Barn at {self.pos} pushed unit at {other.pos}")
                # Handle other-Barn collisions, but skip for Cows (Barn immovable)
                elif isinstance(other, Barn) and not isinstance(self, Barn) and not isinstance(self, Cow):
                    distance = self.pos.distance_to(other.pos)
                    combined_min_distance = (self.size + other.size) / 2
                    if distance < combined_min_distance and distance > 0:
                        overlap = combined_min_distance - distance
                        direction = (self.pos - other.pos).normalize()
                        self.pos += direction * overlap
                        print(f"Unit at {self.pos} pushed by barn at {other.pos}")
                # Standard collision for non-Barn pairs (including Cow-Cow, Cow-other, other-other)
                elif not isinstance(self, Barn) and not isinstance(other, Barn):
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
        self.assigned_corner = None  # Track assigned corner (top-left or top-right)

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
            if distance_to_target > 2:  # Reduced threshold for precise corner alignment
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

    def harvest_grass(self, grass_tiles, barns, cow_in_barn):
        global milk
        # Check if cow is in any barn
        for barn in barns:
            if self.is_in_barn(barn) and self.special > 0:
                # Cow is in a barn and still has special, do not change target
                return
        if self.special >= 100:
            if not self.target:  # Only set target if not already moving
                # Find the closest barn with no cow inside
                available_barns = [barn for barn in barns if barn not in cow_in_barn or cow_in_barn[barn] is None]
                if available_barns:
                    closest_barn = min(available_barns, key=lambda barn: self.pos.distance_to(barn.pos))
                    # Define barn corners
                    barn_corners = [
                        Vector2(closest_barn.pos.x - closest_barn.size / 2, closest_barn.pos.y - closest_barn.size / 2),  # Top-left
                        Vector2(closest_barn.pos.x + closest_barn.size / 2, closest_barn.pos.y - closest_barn.size / 2)   # Top-right
                    ]
                    # Choose the closest top corner that isn't occupied
                    for corner in sorted(barn_corners, key=lambda c: self.pos.distance_to(c)):
                        # Check if corner is free (no other cow is too close)
                        corner_free = True
                        for unit in units:
                            if isinstance(unit, Cow) and unit is not self:
                                if unit.pos.distance_to(corner) < 10:
                                    corner_free = False
                                    break
                        if corner_free:
                            self.target = corner
                            self.assigned_corner = corner
                            break
                    if self.target:
                        print(f"Cow at {self.pos} (special = {self.special}) moving to corner {self.target} of barn at {closest_barn.pos}")
                    else:
                        # If both corners are occupied, wait near the closest barn
                        self.target = Vector2(closest_barn.pos.x + closest_barn.size / 2 + 10, closest_barn.pos.y)
                        print(f"Cow at {self.pos} waiting near barn at {self.target}")
                else:
                    # No available barns, wait near the closest barn
                    closest_barn = min(barns, key=lambda barn: self.pos.distance_to(barn.pos))
                    self.target = Vector2(closest_barn.pos.x + closest_barn.size / 2 + 10, closest_barn.pos.y)
                    print(f"Cow at {self.pos} (special = {self.special}) waiting near barn at {self.target}")
            return
        # Check if cow is in any barn with special == 0
        for barn in barns:
            if self.special == 0 and self.is_in_barn(barn):
                # Cow in barn has depleted special, set target to a grass tile
                tile_x = int(self.pos.x // TILE_SIZE)
                tile_y = int(self.pos.y // TILE_SIZE)
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
                            print(f"Cow at {self.pos} (special = 0) leaving barn at {barn.pos} to tile ({adj_x}, {adj_y}) with grass_level {grass_tiles[adj_y][adj_x].grass_level}")
                            break
                return
        if not self.target or self.velocity.length() < 0.5:
            tile_x = int(self.pos.x // TILE_SIZE)
            tile_y = int(self.pos.y // TILE_SIZE)
            if 0 <= tile_x < GRASS_COLS and 0 <= tile_y < GRASS_ROWS:
                harvested = grass_tiles[tile_y][tile_x].harvest(self.harvest_rate)
                self.special = min(100, self.special + harvested * 50)
                print(f"Cow at {self.pos} harvested grass, special = {self.special}")
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
        super().__init__(x, y, size=48, speed=0, color=GRAY)  # Use GRAY with alpha
        self.harvest_rate = 1

    def draw(self, screen):
        # Create a surface with per-pixel alpha
        barn_surface = pygame.Surface((self.size, self.size), pygame.SRCALPHA)
        # Draw the barn rectangle on the surface with the GRAY color (includes alpha)
        pygame.draw.rect(barn_surface, self.color, (0, 0, self.size, self.size))
        # Blit the surface to the main screen at the barn's position
        screen.blit(barn_surface, (self.pos.x - self.size / 2, self.pos.y - self.size / 2))

    def move(self, units):
        pass  # Barn is immovable

# Screen settings
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Simple RTS Game with Multiple Barns")
clock = pygame.time.Clock()

# Colors
WHITE = (255, 255, 255)  # Selection rectangle
BLUE = (0, 0, 255)  # Soldier and milk text
RED = (255, 0, 0)  # Tank and Cow special bar background
YELLOW = (255, 255, 0)  # Scout
GREEN = (0, 255, 0)  # Selected unit highlight and Cow special bar fill
GRASS_GREEN = (0, 100, 0)  # Full grass
GRASS_BROWN = (139, 69, 19)  # Depleted grass and Dirt
GRAY = (128, 128, 128, 128)  # Barn, 50% transparent (alpha = 128)

# Grass tile settings
TILE_SIZE = 20
GRASS_ROWS = SCREEN_HEIGHT // TILE_SIZE  # 30 rows
GRASS_COLS = SCREEN_WIDTH // TILE_SIZE  # 40 cols

# Global variables
milk = 0.0  # Tracks total milk collected, max 1000
cow_in_barn = {}  # Tracks which Cow is in each Barn (Barn instance -> Cow or None)

# Create grass field
grass_tiles = [[GrassTile(col * TILE_SIZE, row * TILE_SIZE) for col in range(GRASS_COLS)] for row in range(GRASS_ROWS)]

# Create units
units = [
    Soldier(100, 100),
    Soldier(150, 150),
    Tank(200, 100),
    Tank(250, 150),
    Scout(300, 100),
    Scout(350, 150),
    Cow(450, 150),
    Barn(510, 90),  # Aligned to grid center at col 25, row 4
    Barn(590, 90),  # Aligned to grid center at col 29, row 4
    Barn(670, 390)   # Aligned to grid center at col 33, row 4
]

# Place Dirt tiles under all barns
for barn in [unit for unit in units if isinstance(unit, Barn)]:
    buildings_cols = range(
        int((barn.pos.x - barn.size / 2) // TILE_SIZE),
        int(math.ceil((barn.pos.x + barn.size / 2) / TILE_SIZE))
    )
    buildings_rows = range(
        int((barn.pos.y - barn.size / 2) // TILE_SIZE),
        int(math.ceil((barn.pos.y + barn.size / 2) / TILE_SIZE))
    )
    for row in buildings_rows:
        for col in buildings_cols:
            if 0 <= row < GRASS_ROWS and 0 <= col < GRASS_COLS:
                grass_tiles[row][col] = Dirt(col * TILE_SIZE, row * TILE_SIZE)

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
                    if unit.selected and not isinstance(unit, Barn):
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

    # Find all barns
    barns = [unit for unit in units if isinstance(unit, Barn)]

    # Update units
    for unit in units:
        unit.move(units)
        unit.resolve_collisions(units)
        unit.keep_in_bounds()
        if isinstance(unit, Cow):
            unit.harvest_grass(grass_tiles, barns, cow_in_barn)

    # Handle Cow in Barn
    for barn in barns:
        if barn in cow_in_barn and cow_in_barn[barn]:
            cow = cow_in_barn[barn]
            if cow.is_in_barn(barn):
                cow.special = max(0, cow.special - barn.harvest_rate)
                if milk < 1000 and cow.special > 0:
                    milk = min(1000, milk + barn.harvest_rate)
                print(f"Cow at {cow.pos} in Barn at {barn.pos}, special = {cow.special}, milk = {milk}")
                if cow.special <= 0:
                    cow_in_barn[barn] = None
                    print(f"Barn at {barn.pos} is now free")
            else:
                cow_in_barn[barn] = None
                print(f"Cow left Barn at {barn.pos}, Barn is now free")
        else:
            for unit in units:
                if isinstance(unit, Cow) and unit.is_in_barn(barn):
                    cow_in_barn[barn] = unit
                    print(f"Cow at {unit.pos} entered Barn at {barn.pos}")
                    break

    # Draw
    screen.fill(WHITE)
    for row in grass_tiles:
        for tile in row:
            tile.draw(screen)
    if selecting and selection_start and selection_end:
        rect = pygame.Rect(
            min(selection_start.x, selection_end.x),
            min(selection_start.y, selection_end.y),
            abs(selection_end.x - selection_start.x),
            abs(selection_end.y - selection_start.y)
        )
        pygame.draw.rect(screen, WHITE, rect, 3)
        print(f"Drawing selection rect: {rect}")
    for unit in units:
        unit.draw(screen)
    font = pygame.font.SysFont(None, 24)
    milk_text = font.render(f"Milk: {milk:.2f}", True, BLUE)
    screen.blit(milk_text, (10, 10))
    pygame.display.flip()
    clock.tick(60)

pygame.quit()
sys.exit()