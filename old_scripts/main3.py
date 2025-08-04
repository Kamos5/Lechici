import pygame
import sys
import random
import math
from pygame.math import Vector2

# Initialize Pygame
pygame.init()

# Screen settings
SCREEN_WIDTH = 1600
SCREEN_HEIGHT = 1200
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Simple RTS Game with Player Selection")
clock = pygame.time.Clock()

# Colors
WHITE = (255, 255, 255)  # Background
BLUE = (0, 0, 255)  # Player 1 units and selection rectangle
PURPLE = (128, 0, 128)  # Player 2 units and selection rectangle
RED = (255, 0, 0)  # Cow special bar background
GREEN = (0, 255, 0)  # Selected unit highlight and Cow special bar fill
GRASS_GREEN = (0, 100, 0)  # Full grass
GRASS_BROWN = (139, 69, 19)  # Depleted grass and Dirt
GRAY = (128, 128, 128, 128)  # Barn, 50% transparent
BLACK = (0, 0, 0)  # Button text
LIGHT_GRAY = (200, 200, 200)  # Button background

# Grass tile settings
TILE_SIZE = 20
GRASS_ROWS = SCREEN_HEIGHT // TILE_SIZE  # 60 rows
GRASS_COLS = SCREEN_WIDTH // TILE_SIZE  # 80 cols

# Button settings
BUTTON_WIDTH = 150
BUTTON_HEIGHT = 40
BUTTON_MARGIN = 10
BUTTON_PLAYER1_POS = (SCREEN_WIDTH - BUTTON_WIDTH - BUTTON_MARGIN, BUTTON_MARGIN)
BUTTON_PLAYER2_POS = (SCREEN_WIDTH - BUTTON_WIDTH - BUTTON_MARGIN, BUTTON_MARGIN + BUTTON_HEIGHT + 10)

# GrassTile class
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

# Dirt class
class Dirt(GrassTile):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.grass_level = 0.0

    def draw(self, screen):
        pygame.draw.rect(screen, GRASS_BROWN, (self.pos.x, self.pos.y, TILE_SIZE, TILE_SIZE))

    def regrow(self, amount):
        pass

    def harvest(self, amount):
        return 0.0

# Unit class
class Unit:
    def __init__(self, x, y, size, speed, color, player_id):
        self.pos = Vector2(x, y)
        self.target = None
        self.speed = speed
        self.selected = False
        self.size = size
        self.min_distance = self.size
        self.color = color
        self.player_id = player_id
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
                elif isinstance(self, Barn) and not isinstance(other, Barn) and not isinstance(other, Cow):
                    distance = self.pos.distance_to(other.pos)
                    combined_min_distance = (self.size + other.size) / 2
                    if distance < combined_min_distance and distance > 0:
                        overlap = combined_min_distance - distance
                        direction = (other.pos - self.pos).normalize()
                        other.pos += direction * overlap
                elif isinstance(other, Barn) and not isinstance(self, Barn) and not isinstance(self, Cow):
                    distance = self.pos.distance_to(other.pos)
                    combined_min_distance = (self.size + other.size) / 2
                    if distance < combined_min_distance and distance > 0:
                        overlap = combined_min_distance - distance
                        direction = (self.pos - other.pos).normalize()
                        self.pos += direction * overlap
                elif not isinstance(self, Barn) and not isinstance(other, Barn):
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

# Soldier, Tank, Scout, Cow, Barn classes
class Soldier(Unit):
    def __init__(self, x, y, player_id, player_color):
        super().__init__(x, y, size=16, speed=5, color=player_color, player_id=player_id)

class Tank(Unit):
    def __init__(self, x, y, player_id, player_color):
        super().__init__(x, y, size=16, speed=2, color=player_color, player_id=player_id)

class Scout(Unit):
    def __init__(self, x, y, player_id, player_color):
        super().__init__(x, y, size=16, speed=8, color=player_color, player_id=player_id)

class Cow(Unit):
    def __init__(self, x, y, player_id, player_color):
        super().__init__(x, y, size=16, speed=4, color=player_color, player_id=player_id)
        self.harvest_rate = 0.01
        self.assigned_corner = None

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
            if distance_to_target > 2:
                self.velocity = direction.normalize() * self.speed
            else:
                self.pos = Vector2(self.target)
                self.target = None
        self.velocity *= self.damping
        self.pos += self.velocity

    def is_in_barn(self, barn):
        return (barn.pos.x - barn.size / 2 <= self.pos.x <= barn.pos.x + barn.size / 2 and
                barn.pos.y - barn.size / 2 <= self.pos.y <= barn.pos.y + barn.size / 2)

    def harvest_grass(self, grass_tiles, barns, cow_in_barn, player):
        if self.is_in_barn_any(barns) and self.special > 0:
            return
        if self.special >= 100:
            if not self.target:
                available_barns = [barn for barn in barns if barn.player_id == self.player_id and (barn not in cow_in_barn or cow_in_barn[barn] is None)]
                if available_barns:
                    closest_barn = min(available_barns, key=lambda barn: self.pos.distance_to(barn.pos))
                    barn_corners = [
                        Vector2(closest_barn.pos.x - closest_barn.size / 2, closest_barn.pos.y - closest_barn.size / 2),
                        Vector2(closest_barn.pos.x + closest_barn.size / 2, closest_barn.pos.y - closest_barn.size / 2)
                    ]
                    for corner in sorted(barn_corners, key=lambda c: self.pos.distance_to(c)):
                        corner_free = True
                        for unit in player.units:
                            if isinstance(unit, Cow) and unit is not self:
                                if unit.pos.distance_to(corner) < 10:
                                    corner_free = False
                                    break
                        if corner_free:
                            self.target = corner
                            self.assigned_corner = corner
                            break
                    if not self.target:
                        closest_barn = min(available_barns, key=lambda barn: self.pos.distance_to(barn.pos))
                        self.target = Vector2(closest_barn.pos.x + closest_barn.size / 2 + 10, closest_barn.pos.y)
                else:
                    closest_barn = min(barns, key=lambda barn: self.pos.distance_to(barn.pos))
                    self.target = Vector2(closest_barn.pos.x + closest_barn.size / 2 + 10, closest_barn.pos.y)
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
                        if grass_tiles[adj_y][adj_x].grass_level > 0.5:
                            self.target = Vector2(adj_x * TILE_SIZE + TILE_SIZE / 2, adj_y * TILE_SIZE + TILE_SIZE / 2)
                            break
                return
        if not self.target or self.velocity.length() < 0.5:
            tile_x = int(self.pos.x // TILE_SIZE)
            tile_y = int(self.pos.y // TILE_SIZE)
            if 0 <= tile_x < GRASS_COLS and 0 <= tile_y < GRASS_ROWS:
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
                            if grass_tiles[adj_y][adj_x].grass_level > 0.5:
                                self.target = Vector2(adj_x * TILE_SIZE + TILE_SIZE / 2, adj_y * TILE_SIZE + TILE_SIZE / 2)
                                break

    def is_in_barn_any(self, barns):
        return any(self.is_in_barn(barn) for barn in barns)

class Barn(Unit):
    def __init__(self, x, y, player_id):
        super().__init__(x, y, size=48, speed=0, color=GRAY, player_id=player_id)
        self.harvest_rate = 1

    def draw(self, screen):
        barn_surface = pygame.Surface((self.size, self.size), pygame.SRCALPHA)
        pygame.draw.rect(barn_surface, self.color, (0, 0, self.size, self.size))
        screen.blit(barn_surface, (self.pos.x - self.size / 2, self.pos.y - self.size / 2))

    def move(self, units):
        pass

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
            Soldier(offset_x + 100, offset_y + 100, player_id, self.color),
            Soldier(offset_x + 150, offset_y + 150, player_id, self.color),
            Tank(offset_x + 200, offset_y + 100, player_id, self.color),
            Tank(offset_x + 250, offset_y + 150, player_id, self.color),
            Scout(offset_x + 300, offset_y + 100, player_id, self.color),
            Scout(offset_x + 350, offset_y + 150, player_id, self.color),
            Cow(offset_x + 450, offset_y + 150, player_id, self.color),
            Barn(offset_x + 510, offset_y + 90, player_id),
            Barn(offset_x + 590, offset_y + 90, player_id),
            Barn(offset_x + 670, offset_y + 390, player_id)
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
    Player(1, BLUE, 0, 0),  # Player 1 (BLUE) starts top-left
    Player(2, PURPLE, 800, 600)  # Player 2 (PURPLE) starts bottom-right
]

# Combine all units
all_units = []
for player in players:
    all_units.extend(player.units)

# Place Dirt tiles under barns
for unit in all_units:
    if isinstance(unit, Barn):
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

# Selection rectangle and player selection mode
selection_start = None
selection_end = None
selecting = False
current_player = None  # None means no player selected; must click a player button first

# Button rectangles
button_player1 = pygame.Rect(BUTTON_PLAYER1_POS[0], BUTTON_PLAYER1_POS[1], BUTTON_WIDTH, BUTTON_HEIGHT)
button_player2 = pygame.Rect(BUTTON_PLAYER2_POS[0], BUTTON_PLAYER2_POS[1], BUTTON_WIDTH, BUTTON_HEIGHT)

# Game loop
running = True
font = pygame.font.SysFont(None, 24)
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                click_pos = Vector2(event.pos)
                # Check for button clicks
                if button_player1.collidepoint(event.pos):
                    current_player = 1
                    for player in players:
                        player.deselect_all_units()
                    #players[0].select_all_units()
                    print("Selected Player 1; only Player 1 units can be selected")
                elif button_player2.collidepoint(event.pos):
                    current_player = 2
                    for player in players:
                        player.deselect_all_units()
                    #players[1].select_all_units()
                    print("Selected Player 2; only Player 2 units can be selected")
                elif current_player is not None:
                    # Only allow unit selection if a player is selected
                    unit_clicked = None
                    for unit in all_units:
                        if unit.player_id == current_player and unit.is_clicked(click_pos):
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
            elif event.button == 3:
                for unit in all_units:
                    if unit.selected and not isinstance(unit, Barn):
                        unit.target = Vector2(event.pos)
                        print(f"Set target for unit at {unit.pos} to {unit.target}")
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1 and selecting and current_player is not None:
                selection_end = Vector2(event.pos)
                selecting = False
                for player in players:
                    if player.player_id == current_player:
                        for unit in player.units:
                            unit.selected = (min(selection_start.x, selection_end.x) <= unit.pos.x <= max(selection_start.x, selection_end.x) and
                                             min(selection_start.y, selection_end.y) <= unit.pos.y <= max(selection_start.y, selection_end.y))
                print(f"Selection ended at: {selection_end}")
        elif event.type == pygame.MOUSEMOTION and selecting:
            selection_end = Vector2(event.pos)

    # Update grass regrowth
    regrowth_rate = 0.001
    for row in grass_tiles:
        for tile in row:
            tile.regrow(regrowth_rate)

    # Update units and handle cow-barn interactions per player
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
                    cow.special = max(0, cow.special - barn.harvest_rate)
                    if player.milk < 1000 and cow.special > 0:
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

    # Draw
    screen.fill(WHITE)
    for row in grass_tiles:
        for tile in row:
            tile.draw(screen)
    if selecting and selection_start and selection_end and current_player is not None:
        rect = pygame.Rect(
            min(selection_start.x, selection_end.x),
            min(selection_start.y, selection_end.y),
            abs(selection_end.x - selection_start.x),
            abs(selection_end.y - selection_start.y)
        )
        # Use the current player's color for the selection rectangle
        selection_color = next(player.color for player in players if player.player_id == current_player)
        pygame.draw.rect(screen, selection_color, rect, 3)
    for unit in all_units:
        unit.draw(screen)
    # Draw buttons
    pygame.draw.rect(screen, LIGHT_GRAY, button_player1)
    pygame.draw.rect(screen, LIGHT_GRAY, button_player2)
    player1_text = font.render("Select Player 1", True, BLACK)
    player2_text = font.render("Select Player 2", True, BLACK)
    screen.blit(player1_text, (BUTTON_PLAYER1_POS[0] + 10, BUTTON_PLAYER1_POS[1] + 10))
    screen.blit(player2_text, (BUTTON_PLAYER2_POS[0] + 10, BUTTON_PLAYER2_POS[1] + 10))
    # Draw milk for each player
    for i, player in enumerate(players):
        milk_text = font.render(f"Player {player.player_id} Milk: {player.milk:.2f}", True, player.color)
        screen.blit(milk_text, (10, 10 + i * 30))
    pygame.display.flip()
    clock.tick(60)

pygame.quit()
sys.exit()