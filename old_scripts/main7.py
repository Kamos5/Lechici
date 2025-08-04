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
pygame.display.set_caption("Simple RTS Game with Larger Map and Scrolling")
clock = pygame.time.Clock()

# Colors
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
PURPLE = (128, 0, 128)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
GRASS_GREEN = (0, 100, 0)
GRASS_BROWN = (139, 69, 19)
TREE_GREEN = (0, 50, 0)
GRAY = (128, 128, 128, 128)
TOWN_CENTER_GRAY = (100, 100, 100, 128)
BLACK = (0, 0, 0)
LIGHT_GRAY = (150, 150, 150)
HIGHLIGHT_GRAY = (200, 200, 200)

# Tile settings
TILE_SIZE = 20
GRASS_ROWS = 100  # 2x larger (2000 pixels)
GRASS_COLS = 100  # 2x larger (2000 pixels)
MAP_WIDTH = GRASS_COLS * TILE_SIZE  # 2000
MAP_HEIGHT = GRASS_ROWS * TILE_SIZE  # 2000

# Button settings
BUTTON_WIDTH = 150
BUTTON_HEIGHT = 40
BUTTON_MARGIN = 10
BUTTON_PLAYER1_POS = (SCREEN_WIDTH - BUTTON_WIDTH - BUTTON_MARGIN, BUTTON_MARGIN)
BUTTON_PLAYER2_POS = (SCREEN_WIDTH - BUTTON_WIDTH - BUTTON_MARGIN, BUTTON_MARGIN + BUTTON_HEIGHT + 10)
BUTTON_SPAWN_COW_POS = (SCREEN_WIDTH - BUTTON_WIDTH - BUTTON_MARGIN, BUTTON_MARGIN + 2 * (BUTTON_HEIGHT + 10))

# Camera settings
camera_x = 0
camera_y = 0
SCROLL_SPEED = 10  # Pixels per frame
SCROLL_MARGIN = 50  # Pixels from edge to trigger scrolling

# SimpleTile base class
class SimpleTile:
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
    def __init__(self, x, y):
        super().__init__(x, y)
        self.grass_level = 1.0

    def draw(self, screen, camera_x, camera_y):
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
    def __init__(self, x, y):
        super().__init__(x, y)

    def draw(self, screen, camera_x, camera_y):
        pygame.draw.rect(screen, GRASS_BROWN, (self.pos.x - camera_x, self.pos.y - camera_y, TILE_SIZE, TILE_SIZE))

# Tree class
class Tree(SimpleTile):
    def __init__(self, x, y):
        super().__init__(x, y)

    def draw(self, screen, camera_x, camera_y):
        pygame.draw.rect(screen, TREE_GREEN, (self.pos.x - camera_x, self.pos.y - camera_y, TILE_SIZE, TILE_SIZE))

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

    def draw(self, screen, camera_x, camera_y):
        color = GREEN if self.selected else self.color
        pygame.draw.rect(screen, color, (self.pos.x - self.size / 2 - camera_x, self.pos.y - self.size / 2 - camera_y, self.size, self.size))

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
    def __init__(self, x, y, size, color, player_id):
        super().__init__(x, y, size, speed=0, color=color, player_id=player_id)

    def move(self, units):
        pass

# Barn class
class Barn(Building):
    def __init__(self, x, y, player_id):
        super().__init__(x, y, size=48, color=GRAY, player_id=player_id)
        self.harvest_rate = 60 / 60  # 1 per second at 60 FPS

    def draw(self, screen, camera_x, camera_y):
        color = GREEN if self.selected else self.color
        barn_surface = pygame.Surface((self.size, self.size), pygame.SRCALPHA)
        pygame.draw.rect(barn_surface, color, (0, 0, self.size, self.size))
        screen.blit(barn_surface, (self.pos.x - self.size / 2 - camera_x, self.pos.y - self.size / 2 - camera_y))

# TownCenter class
class TownCenter(Building):
    def __init__(self, x, y, player_id):
        super().__init__(x, y, size=64, color=TOWN_CENTER_GRAY, player_id=player_id)

    def draw(self, screen, camera_x, camera_y):
        color = GREEN if self.selected else self.color
        town_center_surface = pygame.Surface((self.size, self.size), pygame.SRCALPHA)
        pygame.draw.rect(town_center_surface, color, (0, 0, self.size, self.size))
        screen.blit(town_center_surface, (self.pos.x - self.size / 2 - camera_x, self.pos.y - self.size / 2 - camera_y))

# Axeman class
class Axeman(Unit):
    def __init__(self, x, y, player_id, player_color):
        super().__init__(x, y, size=16, speed=5, color=player_color, player_id=player_id)

# Knight class
class Knight(Unit):
    def __init__(self, x, y, player_id, player_color):
        super().__init__(x, y, size=16, speed=2, color=player_color, player_id=player_id)

# Archer class
class Archer(Unit):
    def __init__(self, x, y, player_id, player_color):
        super().__init__(x, y, size=16, speed=8, color=player_color, player_id=player_id)

# Cow class
class Cow(Unit):
    def __init__(self, x, y, player_id, player_color):
        super().__init__(x, y, size=16, speed=4, color=player_color, player_id=player_id)
        self.harvest_rate = 0.01
        self.assigned_corner = None

    def draw(self, screen, camera_x, camera_y):
        color = GREEN if self.selected else self.color
        pygame.draw.rect(screen, color, (self.pos.x - self.size / 2 - camera_x, self.pos.y - self.size / 2 - camera_y, self.size, self.size))
        bar_width = 16
        bar_height = 4
        bar_offset = 2
        bar_x = self.pos.x - bar_width / 2 - camera_x
        bar_y = self.pos.y - self.size / 2 - bar_height - bar_offset - camera_y
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
                        if isinstance(grass_tiles[adj_y][adj_x], GrassTile) and not isinstance(grass_tiles[adj_y][adj_x], (Dirt, Tree)) and grass_tiles[adj_y][adj_x].grass_level > 0.5:
                            self.target = Vector2(adj_x * TILE_SIZE + TILE_SIZE / 2, adj_y * TILE_SIZE + TILE_SIZE / 2)
                            break
                return
        if not self.target or self.velocity.length() < 0.5:
            tile_x = int(self.pos.x // TILE_SIZE)
            tile_y = int(self.pos.y // TILE_SIZE)
            if 0 <= tile_x < GRASS_COLS and 0 <= tile_y < GRASS_ROWS:
                if isinstance(grass_tiles[tile_y][tile_x], GrassTile) and not isinstance(grass_tiles[tile_y][tile_x], (Dirt, Tree)):
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
                                if isinstance(grass_tiles[adj_y][adj_x], GrassTile) and not isinstance(grass_tiles[adj_y][adj_x], (Dirt, Tree)) and grass_tiles[adj_y][adj_x].grass_level > 0.5:
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
            Barn(offset_x + 310, offset_y + 150, player_id),
            TownCenter(offset_x + 150, offset_y + 150, player_id)
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

# Convert 30% of grass tiles to tree tiles in patches
def is_tile_occupied(row, col, units):
    tile_rect = pygame.Rect(col * TILE_SIZE, row * TILE_SIZE, TILE_SIZE, TILE_SIZE)
    for unit in units:
        unit_rect = pygame.Rect(unit.pos.x - unit.size / 2, unit.pos.y - unit.size / 2, unit.size, unit.size)
        if tile_rect.colliderect(unit_rect):
            return True
    return False

total_tiles = GRASS_ROWS * GRASS_COLS
target_tree_tiles = int(total_tiles * 0.3)
eligible_tiles = []
for row in range(GRASS_ROWS):
    for col in range(GRASS_COLS):
        if isinstance(grass_tiles[row][col], GrassTile) and not isinstance(grass_tiles[row][col], Dirt) and not is_tile_occupied(row, col, all_units):
            eligible_tiles.append((row, col))

tree_tiles = []
while len(tree_tiles) < target_tree_tiles and eligible_tiles:
    patch_size = random.randint(0, 2)
    center_row, center_col = random.choice(eligible_tiles)
    patch = []
    for dr in range(-(patch_size - (patch_size // 2)), patch_size):
        for dc in range(-(patch_size - (patch_size // 2)), patch_size):
            r, c = center_row + dr, center_col + dc
            if (0 <= r < GRASS_ROWS and 0 <= c < GRASS_COLS and
                isinstance(grass_tiles[r][c], GrassTile) and
                not isinstance(grass_tiles[r][c], (Dirt, Tree)) and
                not is_tile_occupied(r, c, all_units) and
                (r, c) in eligible_tiles):
                patch.append((r, c))
    if patch:
        for r, c in patch:
            grass_tiles[r][c] = Tree(c * TILE_SIZE, r * TILE_SIZE)
            tree_tiles.append((r, c))
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
    # Update camera for edge scrolling
    mouse_pos = pygame.mouse.get_pos()
    if mouse_pos[0] < SCROLL_MARGIN:
        camera_x = max(0, camera_x - SCROLL_SPEED)
    elif mouse_pos[0] > SCREEN_WIDTH - SCROLL_MARGIN:
        camera_x = min(MAP_WIDTH - SCREEN_WIDTH, camera_x + SCROLL_SPEED)
    if mouse_pos[1] < SCROLL_MARGIN:
        camera_y = max(0, camera_y - SCROLL_SPEED)
    elif mouse_pos[1] > SCREEN_HEIGHT - SCROLL_MARGIN:
        camera_y = min(MAP_HEIGHT - SCREEN_HEIGHT, camera_y + SCROLL_SPEED)

    # Update camera for arrow key scrolling
    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT]:
        camera_x = max(0, camera_x - SCROLL_SPEED)
    if keys[pygame.K_RIGHT]:
        camera_x = min(MAP_WIDTH - SCREEN_WIDTH, camera_x + SCROLL_SPEED)
    if keys[pygame.K_UP]:
        camera_y = max(0, camera_y - SCROLL_SPEED)
    if keys[pygame.K_DOWN]:
        camera_y = min(MAP_HEIGHT - SCREEN_HEIGHT, camera_y + SCROLL_SPEED)

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
                click_pos = Vector2(event.pos[0] + camera_x, event.pos[1] + camera_y)
                if button_player1.collidepoint(event.pos):
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
                elif current_player is not None:
                    unit_clicked = None
                    for unit in all_units:
                        if unit.player_id == current_player.player_id and unit.is_clicked(Vector2(event.pos), camera_x, camera_y):
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
                    if unit.selected and not isinstance(unit, Building):
                        unit.target = Vector2(event.pos[0] + camera_x, event.pos[1] + camera_y)
                        print(f"Set target for unit at {unit.pos} to {unit.target}")
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1 and selecting and current_player is not None:
                selection_end = Vector2(event.pos[0] + camera_x, event.pos[1] + camera_y)
                selecting = False
                for player in players:
                    if player.player_id == current_player.player_id:
                        for unit in player.units:
                            unit.selected = (min(selection_start.x, selection_end.x) <= unit.pos.x <= max(selection_start.x, selection_end.x) and
                                             min(selection_start.y, selection_end.y) <= unit.pos.y <= max(selection_start.y, selection_end.y))
                print(f"Selection ended at: {selection_end}")
        elif event.type == pygame.MOUSEMOTION and selecting:
            selection_end = Vector2(event.pos[0] + camera_x, event.pos[1] + camera_y)

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
                    if player.milk < 1000 and cow.special >= 0:
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
    # Draw only visible tiles
    start_col = int(camera_x // TILE_SIZE)
    end_col = min(GRASS_COLS, int((camera_x + SCREEN_WIDTH) // TILE_SIZE) + 1)
    start_row = int(camera_y // TILE_SIZE)
    end_row = min(GRASS_ROWS, int((camera_y + SCREEN_HEIGHT) // TILE_SIZE) + 1)
    for row in range(start_row, end_row):
        for col in range(start_col, end_col):
            grass_tiles[row][col].draw(screen, camera_x, camera_y)
    if selecting and selection_start and selection_end and current_player is not None:
        rect = pygame.Rect(
            min(selection_start.x - camera_x, selection_end.x - camera_x),
            min(selection_start.y - camera_y, selection_end.y - camera_y),
            abs(selection_end.x - selection_start.x),
            abs(selection_end.y - selection_start.y)
        )
        selection_color = next(player.color for player in players if player.player_id == current_player.player_id)
        pygame.draw.rect(screen, selection_color, rect, 3)
    for unit in all_units:
        unit.draw(screen, camera_x, camera_y)
    # Draw UI (in screen space)
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
    fps = clock.get_fps()
    fps_text = font.render(f"FPS: {int(fps)}", True, BLACK)
    screen.blit(fps_text, (10, 10))
    for i, player in enumerate(players):
        milk_text = font.render(f"Player {player.player_id} Milk: {player.milk:.2f}", True, player.color)
        screen.blit(milk_text, (10, 40 + i * 30))
    pygame.display.flip()
    clock.tick(60)

pygame.quit()
sys.exit()