from __future__ import annotations

import math
import os
import random
from typing import Dict, List, Optional, Tuple

import pygame
from PIL import Image
from pygame.math import Vector2

from constants import *
import context
from tiles import GrassTile, Dirt

# --- Team-color mask swap (shared by game + editor) ---

# Which color(s) in each sprite should be replaced with player_color.
# Put the "team mask" color(s) used in your PNGs here.
TEAM_MASKS: Dict[str, List[str]] = {
    # Buildings
    "Barracks": ["#6f0000"],
    "TownCenter": ["#6f0000"],
    "Barn": ["#6f0000"],
    # Units (if your unit sprites have a team mask too)
    "Axeman": ["#700000"],
    "Knight": ["#6b0000"],
    # "Archer": ["#6f0000"],
    # "Cow": ["#6f0000"],
    "ShamansHut": ["#6f0000"],
}

# Cache tinted result per (class_name, desired_px, player_color_rgb)
_TEAM_SPRITE_CACHE: Dict[Tuple[str, int, Tuple[int, int, int]], pygame.Surface] = {}


def remap_red_gradient(
    src: pygame.Surface,
    red_min: int,
    red_max: int,
    team_color: Tuple[int, int, int],
) -> pygame.Surface:
    """
    Remap pixels whose RED channel is in [red_min, red_max]
    into a shaded version of team_color.
    Preserves alpha and shading.
    """
    surf = src.copy().convert_alpha()
    px = pygame.PixelArray(surf)

    tr, tg, tb = team_color
    span = max(1, red_max - red_min)

    for x in range(surf.get_width()):
        for y in range(surf.get_height()):
            c = surf.unmap_rgb(px[x, y])
            if red_min <= c.r <= red_max and c.g == 0 and c.b == 0:
                t = (c.r - red_min) / span  # 0..1
                nr = int(tr * t)
                ng = int(tg * t)
                nb = int(tb * t)
                px[x, y] = (nr, ng, nb, c.a)

    del px
    return surf

def get_team_sprite(cls_name: str, desired_px: int, player_color: Tuple[int, int, int]) -> Optional[pygame.Surface]:
    """
    Return sprite for cls_name scaled to desired_px.
    If cls_name has TEAM_MASKS, replace those colors with player_color.
    Works in-game (Unit.draw) and in map_editor (preview sprites).
    """
    # Ensure base sprite is loaded at the requested size
    if cls_name not in Unit._images or Unit._images.get(cls_name) is None:
        Unit.load_images(cls_name, desired_px)

    base = Unit._images.get(cls_name)
    if base is None:
        return None

    if cls_name not in TEAM_MASKS:
        return base

    key = (cls_name, int(desired_px), tuple(player_color[:3]))
    if key in _TEAM_SPRITE_CACHE:
        return _TEAM_SPRITE_CACHE[key]

    tinted = remap_red_gradient(
        base,
        red_min=0x20,
        red_max=0xFF,
        team_color=key[2],
    )

    _TEAM_SPRITE_CACHE[key] = tinted
    return tinted


# ------------------ Animated sprites (GIF) support ------------------

# Cache per (folder_key, size_px, player_color_rgb, flip_x) -> list[Surface]
_ANIM_CACHE: Dict[Tuple[str, int, Tuple[int, int, int], bool], List[pygame.Surface]] = {}


def _load_gif_frames(path: str) -> List[pygame.Surface]:
    """
    Load all frames from an animated GIF using Pillow.
    Returns list of pygame.Surface in original pixel size.
    """
    img = Image.open(path)
    frames: List[pygame.Surface] = []

    try:
        while True:
            frame = img.convert("RGBA")
            w, h = frame.size
            surf = pygame.image.fromstring(frame.tobytes(), (w, h), "RGBA").convert_alpha()
            frames.append(surf)
            img.seek(img.tell() + 1)
    except EOFError:
        pass

    return frames


def _get_team_anim_frames(
    folder_key: str,
    gif_path: str,
    desired_px: int,
    cls_name_for_tint: str,
    player_color: Tuple[int, int, int],
    flip_x: bool = False,
) -> List[pygame.Surface]:
    """
    Universal: load & cache gif frames, scale to desired_px,
    optionally apply team tint like in get_team_sprite, optionally flip horizontally.
    """
    key = (folder_key, int(desired_px), tuple(player_color[:3]), bool(flip_x))
    if key in _ANIM_CACHE:
        return _ANIM_CACHE[key]

    if not os.path.exists(gif_path):
        _ANIM_CACHE[key] = []
        return []

    frames = _load_gif_frames(gif_path)
    out: List[pygame.Surface] = []

    for fr in frames:
        fr2 = pygame.transform.scale(fr, (int(desired_px), int(desired_px)))

        if flip_x:
            fr2 = pygame.transform.flip(fr2, True, False)

        # team tint if configured for this class
        if cls_name_for_tint in TEAM_MASKS:
            fr2 = remap_red_gradient(
                fr2,
                red_min=0x20,
                red_max=0xFF,
                team_color=tuple(player_color[:3]),
            )

        out.append(fr2)

    _ANIM_CACHE[key] = out
    return out

class Unit:
    _images = {}
    _unit_icons = {}
    milk_cost = 0
    wood_cost = 0
    production_time = 15.0

    def __init__(self, x, y, size, speed, color, player_id, player_color):
        self.pos = Vector2(x, y)
        self.target = None
        self.base_speed = speed
        self.speed = speed * SCALE
        self.selected = False
        self.size = size
        self.min_distance = self.size
        self.color = color
        self.player_id = player_id
        self.player_color = player_color
        self.velocity = Vector2(0, 0)
        self.damping = 0.95
        self.hp = 50
        self.view_distance = 5
        self.max_hp = 50
        self.mana = 0
        self.special = 0
        self.attack_damage = 0
        self.attack_range = 0
        self.attack_cooldown = 1.0
        self.last_attack_time = 0
        self.armor = 0
        self.name = None
        self.worldObject = False
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
            cls._images[cls_name] = pygame.image.load(f"assets/{cls_name.lower()}.png").convert_alpha()
            cls._images[cls_name] = pygame.transform.scale(cls._images[cls_name], (int(size), int(size)))
        except (pygame.error, FileNotFoundError) as e:
            print(f"Failed to load assets/{cls_name.lower()}.png: {e}")
            cls._images[cls_name] = None
        try:
            cls._unit_icons[cls_name] = pygame.image.load(f"assets/{cls_name.lower()}_icon.png").convert_alpha()
            cls._unit_icons[cls_name] = pygame.transform.scale(cls._unit_icons[cls_name], (ICON_SIZE, ICON_SIZE))
        except (pygame.error, FileNotFoundError) as e:
            print(f"Failed to load assets/{cls_name.lower()}_icon.png: {e}")
            cls._unit_icons[cls_name] = None

    def should_highlight(self, current_time):
        return self in context.highlight_times and context.current_time - context.highlight_times[self] <= 0.4

    def draw(self, screen, camera_x, camera_y):
        if (self.pos.x < camera_x - self.size / 2 or self.pos.x > camera_x + VIEW_WIDTH + self.size / 2 or
            self.pos.y < camera_y - self.size / 2 or self.pos.y > camera_y + VIEW_HEIGHT + self.size / 2):
            return
        cls_name = self.__class__.__name__
        image = get_team_sprite(cls_name, int(self.size), tuple(self.player_color[:3]))
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
        if self.should_highlight(context.current_time):
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
            # pygame.draw.lines(screen, WHITE, False, points, 1)

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
                self.is_path_blocked(units, context.spatial_grid, context.waypoint_graph)):
            self.path = context.waypoint_graph.get_path(self.pos, target_pos, self) if context.waypoint_graph else []
            self.path_index = 0
            self.last_target = self.target
            if not self.path:
                if self.pos.distance_to(target_pos) < self.size * 2 and self.is_line_of_sight_clear(target_pos, units, context.spatial_grid):
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
                            # print(f"Axeman at {self.pos} stopped at path end, distance to {self.target}: {self.pos.distance_to(self.target):.1f}")
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
                        # print(f"Axeman at {self.pos} reached path end, distance to {self.target}: {self.pos.distance_to(self.target):.1f}")
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
            tile_x = int(point.x // context.waypoint_graph.tile_size)
            tile_y = int(point.y // context.waypoint_graph.tile_size)
            if not context.waypoint_graph.is_walkable(tile_x, tile_y, self):
                return True
        return False

    def is_line_of_sight_clear(self, target_pos, units, spatial_grid):
        """Check if there's a clear line of sight to the target."""
        if not context.spatial_grid:
            return True
        nearby_units = context.spatial_grid.get_nearby_units(self, radius=self.pos.distance_to(target_pos))
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
        self.attackers = [(a, t) for a, t in self.attackers if context.current_time - t < self.attacker_timeout]
        # Update or add attacker
        for i, (existing_attacker, _) in enumerate(self.attackers):
            if existing_attacker == attacker:
                self.attackers[i] = (attacker, context.current_time)
                return
        self.attackers.append((attacker, context.current_time))

    def get_closest_attacker(self):
        """Return the closest living attacker."""
        if not self.attackers:
            return None
        valid_attackers = [(a, t) for a, t in self.attackers if a.hp > 0 and a in context.all_units]
        if not valid_attackers:
            return None
        return min(valid_attackers, key=lambda x: self.pos.distance_to(x[0].pos))[0]

    def attack(self, target, current_time):
        if not isinstance(target, Unit) or isinstance(target, Tree) or target.hp <= 0 or target not in context.all_units:
            return
        distance = (self.pos - target.pos).length()
        max_range = self.attack_range + self.size / 2 + target.size / 2
        if distance <= max_range:
            if context.current_time - self.last_attack_time >= self.attack_cooldown:
                context.attack_animations.append({
                    'start_pos': self.pos,
                    'end_pos': target.pos,
                    'color': self.color,
                    'start_time': context.current_time
                })
                damage = max(0, self.attack_damage - target.armor)
                target.hp -= damage
                self.last_attack_time = context.current_time
                print(f"{self.__class__.__name__} at {self.pos} attacked {target.__class__.__name__} at {target.pos}, dealing {damage} damage")
                # Notify target of attack for defensive behavior
                if isinstance(target, (Axeman, Archer, Knight)) and target.hp > 0:
                    target.update_attackers(self, context.current_time)
                    # Trigger counter-attack if no target or autonomous target
                    if not target.target or target.autonomous_target:
                        closest_attacker = target.get_closest_attacker()
                        if closest_attacker:
                            target.target = closest_attacker
                            target.autonomous_target = True
                            target.path = []  # Clear path to recalculate
                            target.path_index = 0
                            print(f"{target.__class__.__name__} at {target.pos} counter-attacking closest attacker {closest_attacker.__class__.__name__} at {closest_attacker.pos}")
        # elif isinstance(self, Archer):
            # Ensure Archers recalculate path if out of range
            # self.path = []
            # self.path_index = 0

    def resolve_collisions(self, units, spatial_grid):
        if isinstance(self, (Building, Tree)):
            return
        corrections = []
        nearby_units = context.spatial_grid.get_nearby_units(self)
        epsilon = 0.01
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
    # per-variant caches
    _images = {}          # variant -> base image surface
    _tinted_images = {}   # (variant, color_index) -> tinted image

    _selected = False
    _last_color_change_time = 0
    _color_index = 0
    _colors = [RED, GREEN, BLUE, YELLOW]

    # define your variants in one place (sprites + stats)
    VARIANTS = {
        "tree0": {
            "sprite": "assets/tree0.png",
            "hp": 900,
            "max_hp": 900,
        },
        "tree1": {
            "sprite": "assets/tree1.png",
            "hp": 900,
            "max_hp": 900,
        },
        "tree2": {
            "sprite": "assets/tree2.png",
            "hp": 900,
            "max_hp": 900,
        },
        "tree3": {
            "sprite": "assets/tree3.png",
            "hp": 900,
            "max_hp": 900,
        },
        "tree4": {
            "sprite": "assets/tree4.png",
            "hp": 900,
            "max_hp": 900,
        },
        "tree5": {
            "sprite": "assets/tree5.png",
            "hp": 900,
            "max_hp": 900,
        },
        "tree6": {
            "sprite": "assets/tree6.png",
            "hp": 900,
            "max_hp": 900,
        }
    }

    def __init__(self, x, y, size, color, player_id, player_color, variant: str = "tree6"):
        super().__init__(x, y, size=TILE_SIZE, speed=0, color=color, player_id=player_id, player_color=player_color)

        # pick variant (fallback to oak if unknown)
        self.variant = variant if variant in self.VARIANTS else "tree6"
        cfg = self.VARIANTS[self.variant]

        self.hp = cfg["hp"]
        self.max_hp = cfg["max_hp"]
        self.attack_damage = 0
        self.attack_range = 0
        self.armor = 0

        self.worldObject = True
        self.minimapColor = "#023020"

        # lazy-load sprite for this variant (and tint variants)
        if self.variant not in Tree._images:
            Tree.load_image(self.variant)

        self.pos.x = x
        self.pos.y = y

    @classmethod
    def load_image(cls, variant: str):
        cfg = cls.VARIANTS.get(variant)
        if not cfg:
            return

        path = cfg["sprite"]
        try:
            img = pygame.image.load(path).convert_alpha()
            scale_factor = min(TILE_SIZE / img.get_width(), TILE_SIZE / img.get_height())
            new_size = (int(img.get_width() * scale_factor), int(img.get_height() * scale_factor))
            img = pygame.transform.scale(img, new_size)

            cls._images[variant] = img

            # build tinted versions for targeting animation
            for i, color in enumerate(cls._colors):
                tinted = img.copy()
                mask_surface = pygame.Surface(tinted.get_size(), pygame.SRCALPHA)
                mask_surface.fill(color + (128,))
                tinted.blit(mask_surface, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
                cls._tinted_images[(variant, i)] = tinted

        except (pygame.error, FileNotFoundError) as e:
            print(f"Failed to load {path}: {e}")
            cls._images[variant] = None

    def draw(self, screen, camera_x, camera_y, axemen_targets):
        if (self.pos.x < camera_x - TILE_SIZE or self.pos.x > camera_x + VIEW_WIDTH + TILE_SIZE or
            self.pos.y < camera_y - TILE_SIZE or self.pos.y > camera_y + VIEW_HEIGHT + TILE_SIZE):
            return

        x = self.pos.x - camera_x + VIEW_MARGIN_LEFT
        y = self.pos.y - camera_y + VIEW_MARGIN_TOP

        img = self._images.get(self.variant)

        if img:
            is_targeted = any(
                self.pos.distance_to(axeman.pos) <= TILE_SIZE and axeman.target == self.pos
                for axeman in axemen_targets
            )

            if is_targeted:
                if context.current_time - self._last_color_change_time >= 1:
                    self._color_index = (self._color_index + 1) % len(self._colors)
                    self._last_color_change_time = context.current_time
                img_to_draw = self._tinted_images[(self.variant, self._color_index)]
            else:
                img_to_draw = img

            rect = img_to_draw.get_rect(center=(int(x), int(y)))
            screen.blit(img_to_draw, rect)

            if self.should_highlight(context.current_time):
                pygame.draw.rect(screen, WHITE, (x - TILE_HALF, y - TILE_HALF, TILE_SIZE, TILE_SIZE), 1)
            if self._selected:
                pygame.draw.rect(screen, YELLOW, (x - TILE_HALF, y - TILE_HALF, TILE_SIZE, TILE_SIZE), 1)
        else:
            pygame.draw.rect(screen, TREE_GREEN, (x - TILE_HALF, y - TILE_HALF, TILE_SIZE, TILE_SIZE))
            if self.should_highlight(context.current_time):
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
        return False

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
        super().__init__(x, y, size=BUILDING_SIZE, color=DARK_GRAY, player_id=player_id, player_color=player_color)
        self.harvest_rate = 60.0

# Barn class
class ShamansHut(Building):
    milk_cost = 200
    wood_cost = 300
    def __init__(self, x, y, player_id, player_color):
        super().__init__(x, y, size=BUILDING_SIZE, color=DARK_GRAY, player_id=player_id, player_color=player_color)


# TownCenter class
class TownCenter(Building):
    milk_cost = 0
    wood_cost = 800
    def __init__(self, x, y, player_id, player_color):
        super().__init__(x, y, size=BUILDING_SIZE, color=TOWN_CENTER_GRAY, player_id=player_id, player_color=player_color)
        self.hp = 200  # High HP for buildings
        self.max_hp = 200
        self.armor = 5

# Barracks class
class Barracks(Building):
    milk_cost = 0
    wood_cost = 500
    def __init__(self, x, y, player_id, player_color):
        super().__init__(x, y, size=BUILDING_SIZE, color=TOWN_CENTER_GRAY, player_id=player_id, player_color=player_color)

# Axeman class
class Axeman(Unit):
    milk_cost = 300
    wood_cost = 0

    # folder z animacjami:
    # assets/units/axeman/walk_D.gif, walk_L.gif, walk_LD.gif, walk_LU.gif, walk_M.gif, walk_U.gif
    _ANIM_DIR = "assets/units/axeman"

    # klatki na sekundę (czas na klatkę)
    _FRAME_TIME = 0.30  # 0.10s = 10 FPS; dostosuj jak chcesz
    _IDLE_SPEED_EPS2 = 0.05  # threshold dla uznania "stoi"

    def __init__(self, x, y, player_id, player_color):
        super().__init__(x, y, size=UNIT_SIZE, speed=2, color=RED, player_id=player_id, player_color=player_color)
        self.chop_damage = 1
        self.special = 0
        self.return_pos = None
        self.depositing = False
        self.attack_damage = 10
        self.attack_range = 20
        self.attack_cooldown = 1.0
        self.armor = 2

        # kierunek "ostatni" żeby idle trzymał sensowny zwrot (opcjonalne)
        self._last_facing = "D"

    def _facing_from_velocity(self) -> str:
        v = self.velocity
        if v.length_squared() < self._IDLE_SPEED_EPS2:
            return "M"  # idle

        # pygame: y rośnie w dół
        x, y = v.x, v.y

        # preferuj 8-kierunków w prosty sposób
        if abs(x) < 0.35 and y < 0:
            return "U"
        if abs(x) < 0.35 and y > 0:
            return "D"
        if abs(y) < 0.35 and x < 0:
            return "L"
        if abs(y) < 0.35 and x > 0:
            return "R"

        if x < 0 and y < 0:
            return "LU"
        if x < 0 and y > 0:
            return "LD"
        if x > 0 and y < 0:
            return "RU"
        if x > 0 and y > 0:
            return "RD"

        return "D"

    def _get_anim_frames_for_facing(self, facing: str) -> List[pygame.Surface]:
        """
        Zwraca listę klatek dla:
        D, L, LD, LU, M, U
        a R/RD/RU robi jako mirror z L/LD/LU.
        """
        cls_name = self.__class__.__name__
        size_px = int(self.size)
        pc = tuple(self.player_color[:3])

        # idle
        if facing == "M":
            path = os.path.join(self._ANIM_DIR, "walk_M.gif")
            return _get_team_anim_frames(f"{cls_name}/walk_M", path, size_px, cls_name, pc, flip_x=False)

        # proste (bez odbicia)
        if facing in ("D", "L", "LD", "LU", "U"):
            path = os.path.join(self._ANIM_DIR, f"walk_{facing}.gif")
            return _get_team_anim_frames(f"{cls_name}/walk_{facing}", path, size_px, cls_name, pc, flip_x=False)

        # odbicia lustrzane
        if facing == "R":
            path = os.path.join(self._ANIM_DIR, "walk_L.gif")
            return _get_team_anim_frames(f"{cls_name}/walk_R_from_L", path, size_px, cls_name, pc, flip_x=True)

        if facing == "RD":
            path = os.path.join(self._ANIM_DIR, "walk_LD.gif")
            return _get_team_anim_frames(f"{cls_name}/walk_RD_from_LD", path, size_px, cls_name, pc, flip_x=True)

        if facing == "RU":
            path = os.path.join(self._ANIM_DIR, "walk_LU.gif")
            return _get_team_anim_frames(f"{cls_name}/walk_RU_from_LU", path, size_px, cls_name, pc, flip_x=True)

        # fallback
        path = os.path.join(self._ANIM_DIR, "walk_M.gif")
        return _get_team_anim_frames(f"{cls_name}/walk_M", path, size_px, cls_name, pc, flip_x=False)

    def draw(self, screen, camera_x, camera_y):
        # culling jak w Unit.draw()
        if (self.pos.x < camera_x - self.size / 2 or self.pos.x > camera_x + VIEW_WIDTH + self.size / 2 or
            self.pos.y < camera_y - self.size / 2 or self.pos.y > camera_y + VIEW_HEIGHT + self.size / 2):
            return

        facing = self._facing_from_velocity()
        if facing != "M":
            self._last_facing = facing

        frames = self._get_anim_frames_for_facing(facing)
        if not frames:
            # fallback do starego rysowania sprite
            return super().draw(screen, camera_x, camera_y)

        # wybór klatki: gdy idle (M) też animuje jeśli GIF ma kilka klatek
        t = context.current_time
        idx = int(t / self._FRAME_TIME) % len(frames)
        image = frames[idx]

        x = self.pos.x - camera_x + VIEW_MARGIN_LEFT
        y = self.pos.y - camera_y + VIEW_MARGIN_TOP

        image_surface = image.copy()
        image_surface.set_alpha(self.alpha)
        image_rect = image_surface.get_rect(center=(int(x), int(y)))
        screen.blit(image_surface, image_rect)

        if self.selected:
            pygame.draw.rect(screen, self.player_color,
                             (x - self.size / 2, y - self.size / 2, self.size, self.size), 1)

        if self.should_highlight(context.current_time):
            pygame.draw.rect(screen, WHITE,
                             (x - self.size / 2, y - self.size / 2, self.size, self.size), 1)

        # HP bar jak w Unit.draw()
        bar_width = 16
        bar_height = 4
        bar_offset = 2
        bar_x = x - bar_width / 2
        bar_y = y - self.size / 2 - bar_height - bar_offset
        pygame.draw.rect(screen, BLACK, (bar_x, bar_y, bar_width, bar_height))
        fill_width = (self.hp / self.max_hp) * bar_width
        pygame.draw.rect(screen, self.player_color, (bar_x, bar_y, fill_width, bar_height))

        # ścieżka jak w Unit.draw (zostawiam wyłączone)
        if self.path and len(self.path[self.path_index:]) >= 2:
            points = [(p.x - camera_x + VIEW_MARGIN_LEFT, p.y - camera_y + VIEW_MARGIN_TOP)
                      for p in self.path[self.path_index:]]
            # pygame.draw.lines(screen, WHITE, False, points, 1)

    def move(self, units, spatial_grid=None, waypoint_graph=None):
        self.velocity = Vector2(0, 0)

        # Idle state: No target, remain idle unless commanded
        if not self.target and not self.depositing:
            return  # Do not automatically target trees unless commanded

        # Depositing state: Move to TownCenter to deposit wood
        if self.depositing and self.target:
            target_unit = next(
                (unit for unit in units
                 if isinstance(unit, TownCenter)
                 and unit.player_id == self.player_id
                 and isinstance(self.target, Vector2)
                 and unit.pos.distance_to(self.target) < 1.0),  # tolerance avoids float equality issues
                None
            )

            if not target_unit:
                self.depositing = False
            else:
                if self.pos.distance_to(self.target) <= self.size + target_unit.size / 2:
                    for player in context.players:
                        if player.player_id == self.player_id:
                            over_limit = max(0, player.unit_count - player.unit_limit) if player.unit_limit is not None else 0
                            multiplier = max(0.0, 1.0 - (0.1 * over_limit))
                            wood_deposited = self.special * multiplier
                            player.wood = min(player.max_wood, player.wood + wood_deposited)
                            break
                    self.special = 0
                    self.depositing = False
                    self.target = self.return_pos
                else:
                    super().move(units, context.spatial_grid, context.waypoint_graph)
                return

        # Returning state: Move to return_pos and target a new tree when close
        if self.target and self.target == self.return_pos:
            if self.pos.distance_to(self.target) <= TILE_SIZE * 1.5:
                self.target = None
                nearby_units = context.spatial_grid.get_nearby_units(self, radius=1000)
                trees = [unit for unit in nearby_units if isinstance(unit, Tree) and unit.player_id == 0]
                if trees:
                    closest_tree = min(trees, key=lambda tree: self.pos.distance_to(tree.pos))
                    self.target = closest_tree.pos
                    self.return_pos = None
                    self.path = context.waypoint_graph.get_path(self.pos, self.target, self) if context.waypoint_graph else [self.pos, self.target]
                    self.path_index = 0
                else:
                    self.return_pos = None
            else:
                super().move(units, context.spatial_grid, context.waypoint_graph)
            return

        # Chopping state
        target_tree = next((unit for unit in units if isinstance(unit, Tree) and unit.player_id == 0 and unit.pos == self.target), None)
        if target_tree and self.pos.distance_to(self.target) <= TILE_SIZE:
            self.velocity = Vector2(0, 0)
        else:
            super().move(units, context.spatial_grid, context.waypoint_graph)

    def chop_tree(self, trees):
        if self.special > 0 or self.depositing or self.target == self.return_pos:
            return
        target_tree = next((tree for tree in trees if isinstance(tree, Tree) and tree.player_id == 0 and self.target == tree.pos and self.pos.distance_to(tree.pos) <= TILE_SIZE), None)
        if target_tree:
            target_tree.hp -= self.chop_damage
            if target_tree.hp <= 0:
                self.return_pos = Vector2(self.pos)
                context.players[0].remove_unit(target_tree)
                context.all_units.remove(target_tree)
                context.spatial_grid.remove_unit(target_tree)
                self.special = 25
                print(f"Tree at {target_tree.pos} chopped down by Axeman at {self.pos}, special = {self.special}")
                town_centers = [unit for unit in context.all_units if isinstance(unit, TownCenter) and unit.player_id == self.player_id and unit.alpha == 255]
                if town_centers:
                    closest_town = min(town_centers, key=lambda town: self.pos.distance_to(town.pos))
                    self.target = closest_town.pos
                    self.path = context.waypoint_graph.get_path(self.pos, closest_town.pos, self)
                    self.path_index = 0
                    self.depositing = True
                else:
                    self.special = 0
                    self.depositing = False
                    self.target = None
                    self.return_pos = None

# Knight class
class Knight(Unit):
    milk_cost = 500
    wood_cost = 400

    def __init__(self, x, y, player_id, player_color):
        super().__init__(x, y, size=UNIT_SIZE, speed=2.5, color=BLUE, player_id=player_id, player_color=player_color)
        self.attack_damage = 17
        self.attack_range = 20
        self.attack_cooldown = 1.0
        self.armor = 5

# Archer class
class Archer(Unit):
    milk_cost = 400
    wood_cost = 200

    def __init__(self, x, y, player_id, player_color):
        super().__init__(x, y, size=UNIT_SIZE, speed=2.3, color=YELLOW, player_id=player_id, player_color=player_color)
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
        super().__init__(x, y, size=UNIT_SIZE, speed=2, color=BROWN, player_id=player_id, player_color=player_color)
        self.harvest_rate = 0.005
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
        image = get_team_sprite(cls_name, int(self.size), tuple(self.player_color[:3]))
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
        if self.should_highlight(context.current_time):
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
                self.is_path_blocked(units, context.spatial_grid, context.waypoint_graph)):
            self.path = context.waypoint_graph.get_path(self.pos, target_pos, self) if context.waypoint_graph else []
            self.path_index = 0
            self.last_target = self.target
            if not self.path:
                if self.pos.distance_to(target_pos) < self.size * 1.5 and self.is_line_of_sight_clear(target_pos, units, context.spatial_grid):
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
                                context.waypoint_graph.is_walkable(adj_x, adj_y, self)):
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
        nearby_units = context.spatial_grid.get_nearby_units(self, radius=TILE_SIZE)
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
                tile = context.grass_tiles[tile_y][tile_x]
                if isinstance(tile, GrassTile) and not isinstance(tile, Dirt):
                    nearby_units = context.spatial_grid.get_nearby_units(self, radius=TILE_SIZE)
                    has_tree = any(isinstance(unit, Tree) and unit.player_id == 0 and
                                   unit.pos.x - TILE_HALF <= self.pos.x <= unit.pos.x + TILE_HALF and
                                   unit.pos.y - TILE_HALF <= self.pos.y <= unit.pos.y + TILE_HALF
                                   for unit in nearby_units)
                    if not has_tree:
                        harvested = tile.harvest(self.harvest_rate)
                        self.special = min(100, self.special + harvested * 10)
                        if tile.grass_level == 0:
                            adjacent_tiles = [
                                (tile_x, tile_y - 1), (tile_x, tile_y + 1), (tile_x - 1, tile_y), (tile_x + 1, tile_y),
                                (tile_x - 1, tile_y - 1), (tile_x + 1, tile_y - 1), (tile_x - 1, tile_y + 1), (tile_x + 1, tile_y + 1)
                            ]
                            random.shuffle(adjacent_tiles)
                            for adj_x, adj_y in adjacent_tiles:
                                if (0 <= adj_x < GRASS_COLS and 0 <= adj_y < GRASS_ROWS and
                                    isinstance(context.grass_tiles[adj_y][adj_x], GrassTile) and
                                    not isinstance(context.grass_tiles[adj_y][adj_x], Dirt) and
                                    context.grass_tiles[adj_y][adj_x].grass_level > 0.5 and
                                    self.is_tile_walkable(adj_x, adj_y, context.spatial_grid)):
                                    self.target = context.grass_tiles[adj_y][adj_x].center
                                    self.return_pos = Vector2(self.pos)
                                    self.autonomous_target = True
                                    print(f"Cow at {self.pos} targeting adjacent grass tile at {self.target}")
                                    break

# Player class
