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

# XP/level progression (kept separate for easier tuning)
import progression

# --- Team-color mask swap (shared by game + editor) ---

# Which color(s) in each sprite should be replaced with player_color.
# Put the "team mask" color(s) used in your PNGs here.
TEAM_MASKS: Dict[str, List[str]] = {
    # Buildings
    "Barracks": ["#700000"],
    "TownCenter": ["#700000"],
    "Barn": ["#700000"],
    # Units (if your unit sprites have a team mask too)
    "Axeman": ["#700000"],
    "Knight": ["#700000"],
    "Bear": ["#700000"],
    "Strzyga": ["#700000"],
    "Priestess": ["#700000"],
    "Shaman": ["#700000"],
    "Archer": ["#700000"],
    "Swordsman": ["#700000"],
    "Spearman": ["#700000"],
    # "Cow": ["#6f0000"],
    "ShamansHut": ["#700000"],
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

# ------------------ Building damage fire (GIF) support ------------------

# Cache per (level, size_px) -> list[Surface]
_FIRE_ANIM_CACHE: Dict[Tuple[int, int], List[pygame.Surface]] = {}

# Remember missing flame assets so we don't spam the console.
_MISSING_FIRE_WARNED: set[tuple[int, int]] = set()

def _fire_gif_path(level: int) -> Optional[str]:
    """Resolve flame gif path.

    We try multiple common layouts because asset folders vary between projects.
    """
    level = int(level)

    candidates = [
        os.path.join("assets", "effects", "flame", f"flame{level}.gif"),
        os.path.join("assets", "effects", f"flame{level}.gif"),
        os.path.join("assets", "effects", "flames", f"flame{level}.gif"),
        os.path.join("assets", "effects", "fire", f"flame{level}.gif"),
        os.path.join("assets", "effects", "fire", "flame", f"flame{level}.gif"),
        os.path.join("assets", "effects", "flame", f"flame_{level}.gif"),
        os.path.join("assets", "effects", f"flame_{level}.gif"),
    ]

    for c in candidates:
        if os.path.exists(c):
            return c
        d = os.path.dirname(c)
        base = os.path.basename(c)
        if os.path.isdir(d):
            try:
                for fn in os.listdir(d):
                    if fn.lower() == base.lower():
                        return os.path.join(d, fn)
            except Exception:
                pass
    return None

def get_fire_frames(level: int, size_px: int) -> List[pygame.Surface]:
    """Load & cache flame GIF frames scaled to (size_px, size_px)."""
    key = (int(level), int(size_px))
    if key in _FIRE_ANIM_CACHE:
        return _FIRE_ANIM_CACHE[key]

    path = _fire_gif_path(level)
    if not path or not os.path.exists(path):
        _FIRE_ANIM_CACHE[key] = []
        if key not in _MISSING_FIRE_WARNED:
            _MISSING_FIRE_WARNED.add(key)
            print(f"[fire] Missing flame asset for level={level}: put flame{level}.gif under assets/effects/flame/ (or similar).")
        return []

    frames = _load_gif_frames(path)
    out: List[pygame.Surface] = []
    for fr in frames:
        out.append(pygame.transform.scale(fr, (int(size_px), int(size_px))))
    _FIRE_ANIM_CACHE[key] = out
    return out


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
    *,
    fit_to_tile: bool = True,
    scale_factor: float = 1.0,
) -> List[pygame.Surface]:
    """
    Universal: load & cache gif frames.

    Modes:
    - fit_to_tile=True  -> scale each frame to (desired_px, desired_px)   (old behavior)
    - fit_to_tile=False -> keep original frame size, but multiply by scale_factor
                           (preserves aspect ratio, does NOT "fit into tile")
    """
    # cache key must include mode + factor, otherwise you'll reuse wrong-sized frames
    key = (
        folder_key,
        int(desired_px),
        tuple(player_color[:3]),
        bool(flip_x),
        bool(fit_to_tile),
        int(round(scale_factor * 1000)),
    )
    if key in _ANIM_CACHE:
        return _ANIM_CACHE[key]

    if not os.path.exists(gif_path):
        _ANIM_CACHE[key] = []
        return []

    frames = _load_gif_frames(gif_path)
    out: List[pygame.Surface] = []

    for fr in frames:
        if fit_to_tile:
            fr2 = pygame.transform.scale(fr, (int(desired_px), int(desired_px)))
        else:
            # preserve original dims, just apply global factor (e.g. SCALE)
            w, h = fr.get_width(), fr.get_height()
            nw = max(1, int(round(w * scale_factor)))
            nh = max(1, int(round(h * scale_factor)))
            fr2 = pygame.transform.scale(fr, (nw, nh))

        if flip_x:
            fr2 = pygame.transform.flip(fr2, True, False)

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

# ------------------ Standard unit GIF paths (walk_M.gif) ------------------

GIF_UNITS = {"Axeman", "Archer", "Knight", "Cow", "Bear", "Strzyga", "Priestess", "Shaman", "Swordsman", "Spearman"}

ATTACK_D_USES_LD = {"Axeman", "Knight", "Bear", "Strzyga", "Priestess", "Shaman", "Swordsman", "Spearman"}

ATTACK_NO_SCALE = {"Axeman", "Knight", "Bear", "Strzyga", "Priestess", "Shaman", "Swordsman", "Spearman"}

# ------------------ Sprite debug border + per-unit attack offsets ------------------

# Draw a 1px border around the *actual sprite surface rect* (not the tile).
# Useful when aligning oversized GIF frames.
SPRITE_BORDER_ENABLED: bool = False
SPRITE_BORDER_WIDTH: int = 1
SPRITE_BORDER_COLOR: Tuple[int, int, int] = (255, 0, 255)  # magenta for visibility

# Per-unit, per-facing attack animation offsets in "base pixels" (SCALE=1).
# Positive x -> right, positive y -> down.
# These offsets are automatically multiplied by SCALE at draw time.
#
# You can tweak these by hand for any unit if needed:
#   ATTACK_ANIM_OFFSETS["Axeman"]["D"] = (3, -2)
ATTACK_ANIM_OFFSETS: Dict[str, Dict[str, Tuple[float, float]]] = {
    # Defaults (all zero). Add/override per unit & facing as you tune.
    "Axeman": {"M": (0, 0), "D": (-2, 2), "U": (0, -3), "L": (6, 6), "R": (-6, 6),
               "LD": (6, -6), "RD": (-6, -6), "LU": (6, 6), "RU": (-6, 6)},
    "Knight": {"M": (0, 0), "D": (-2, 4), "U": (3, -3), "L": (-2, -1), "R": (2, -1),
               "LD": (0, 0), "RD": (0, 0), "LU": (0, 0), "RU": (0, 0)},
    "Bear": {"M": (0, 0), "D": (-2, 4), "U": (3, -3), "L": (-2, -1), "R": (2, -1),
             "LD": (0, 0), "RD": (0, 0), "LU": (0, 0), "RU": (0, 0)},
    "Strzyga": {"M": (0, 0), "D": (-2, 4), "U": (3, -3), "L": (-2, -1), "R": (2, -1),
                "LD": (0, 0), "RD": (0, 0), "LU": (0, 0), "RU": (0, 0)},
    "Archer": {"M": (0, 0), "D": (0, 0), "U": (0, 0), "L": (0, 0), "R": (0, 0),
               "LD": (0, 0), "RD": (0, 0), "LU": (0, 0), "RU": (0, 0)},
    "Swordsman": {"M": (0, 0), "D": (2, 0), "U": (3, -3), "L": (-2, -1), "R": (2, -1),
               "LD": (0, 0), "RD": (0, 0), "LU": (0, 0), "RU": (0, 0)},
}

def _scaled_attack_offset(cls_name: str, facing: str) -> Tuple[int, int]:
    """Return (dx, dy) for attack animation in screen pixels, auto-scaled by SCALE."""
    facing = facing or "D"
    per_unit = ATTACK_ANIM_OFFSETS.get(cls_name, {})
    ox, oy = per_unit.get(facing, per_unit.get("D", (0, 0)))
    return (int(round(float(ox) * SCALE)), int(round(float(oy) * SCALE)))

def _blit_sprite_with_border(screen: pygame.Surface, surf: pygame.Surface, rect: pygame.Rect) -> None:
    """Blit surf and optionally draw a 1px border around its rect."""
    screen.blit(surf, rect)
    if SPRITE_BORDER_ENABLED and SPRITE_BORDER_WIDTH > 0:
        pygame.draw.rect(screen, SPRITE_BORDER_COLOR, rect, SPRITE_BORDER_WIDTH)

def unit_walk_gif_path(cls_name: str, facing: str) -> str:
    # assets/units/{unit}/walk_<facing>.gif
    return os.path.join("assets", "units", cls_name.lower(), f"walk_{facing}.gif")

def unit_attack_gif_path(cls_name: str, facing: str) -> str:
    # assets/units/{unit}/attack_<facing>.gif
    return os.path.join("assets", "units", cls_name.lower(), f"attack_{facing}.gif")


def get_unit_attack_frames(
    cls_name: str,
    desired_px: int,
    player_color: Tuple[int, int, int],
    facing: str,
) -> List[pygame.Surface]:
    """
    Attack frames for any facing.
    Mirrors right facings from left assets.

    Scaling:
    - default: fit_to_tile=True  -> scale to (desired_px, desired_px)
    - for cls_name in ATTACK_NO_SCALE: fit_to_tile=False, but still multiply by SCALE
      (keeps original aspect ratio and doesn't "fit into tile").
    """
    pc = tuple(player_color[:3])
    facing = facing or "M"

    # scaling mode (exception uses SCALE but doesn't tile-fit)
    fit_to_tile = (cls_name not in ATTACK_NO_SCALE)
    scale_factor = SCALE if not fit_to_tile else 1.0

    def load(fkey: str, path: str, flip: bool) -> List[pygame.Surface]:
        return _get_team_anim_frames(
            fkey, path, desired_px, cls_name, pc, flip_x=flip,
            fit_to_tile=fit_to_tile,
            scale_factor=scale_factor,
        )

    # ---- mirroring first (keeps it robust) ----
    if facing == "R":
        gif_path = unit_attack_gif_path(cls_name, "L")
        folder_key = f"{cls_name}/attack_L"
        return load(folder_key, gif_path, flip=True)

    if facing == "RD":
        gif_path = unit_attack_gif_path(cls_name, "LD")
        folder_key = f"{cls_name}/attack_LD"
        return load(folder_key, gif_path, flip=True)

    if facing == "RU":
        gif_path = unit_attack_gif_path(cls_name, "LU")
        folder_key = f"{cls_name}/attack_LU"
        return load(folder_key, gif_path, flip=True)

    # ---- special rule: treat D as LD for selected units ----
    if facing == "D" and cls_name in ATTACK_D_USES_LD:
        facing = "LD"

    # direct facings
    if facing in ("M", "D", "L", "LD", "LU", "U"):
        gif_path = unit_attack_gif_path(cls_name, facing)
        folder_key = f"{cls_name}/attack_{facing}"
        return load(folder_key, gif_path, flip=False)

    # fallback (also respects the D->LD rule if configured)
    fallback = "LD" if (cls_name in ATTACK_D_USES_LD) else "D"
    gif_path = unit_attack_gif_path(cls_name, fallback)
    folder_key = f"{cls_name}/attack_{fallback}"
    return load(folder_key, gif_path, flip=False)

def get_unit_walk_frames(
    cls_name: str,
    desired_px: int,
    player_color: Tuple[int, int, int],
    facing: str,
) -> List[pygame.Surface]:
    """
    Standard walk frames for any facing.
    Mirrors right facings from left assets (same convention as your Axeman/Cow).
    """
    pc = tuple(player_color[:3])
    facing = facing or "M"

    # Direct faces that should exist as files (if you have them)
    if facing in ("M", "D", "L", "LD", "LU", "U"):
        gif_path = unit_walk_gif_path(cls_name, facing)
        folder_key = f"{cls_name}/walk_{facing}"
        return _get_team_anim_frames(folder_key, gif_path, desired_px, cls_name, pc, flip_x=False)

    # Mirror right variants from left
    if facing == "R":
        gif_path = unit_walk_gif_path(cls_name, "L")
        folder_key = f"{cls_name}/walk_R_from_L"
        return _get_team_anim_frames(folder_key, gif_path, desired_px, cls_name, pc, flip_x=True)

    if facing == "RD":
        gif_path = unit_walk_gif_path(cls_name, "LD")
        folder_key = f"{cls_name}/walk_RD_from_LD"
        return _get_team_anim_frames(folder_key, gif_path, desired_px, cls_name, pc, flip_x=True)

    if facing == "RU":
        gif_path = unit_walk_gif_path(cls_name, "LU")
        folder_key = f"{cls_name}/walk_RU_from_LU"
        return _get_team_anim_frames(folder_key, gif_path, desired_px, cls_name, pc, flip_x=True)

    # Fallback
    gif_path = unit_walk_gif_path(cls_name, "M")
    folder_key = f"{cls_name}/walk_M"
    return _get_team_anim_frames(folder_key, gif_path, desired_px, cls_name, pc, flip_x=False)

def get_unit_walk_first_frame(
    cls_name: str,
    desired_px: int,
    player_color: Tuple[int, int, int],
    facing: str = "M",
) -> Optional[pygame.Surface]:
    frames = get_unit_walk_frames(cls_name, desired_px, player_color, facing=facing)
    return frames[0] if frames else None

class Unit:
    _images = {}
    _unit_icons = {}
    _underbuilding_images = {}  # building construction sprites (e.g. *_c.png)
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
        self.view_distance = 6
        self.aggro_distance = 5
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
        # Trees also need their base icon available in Unit._unit_icons (used by UI).
        if cls_name not in Unit._images:
            self.load_images(cls_name, size)
        self.alpha = 255
        self.path = []  # Store pathfinding waypoints
        self.path_index = 0
        self.last_target = None
        self.attackers = []  # List of (attacker, timestamp) tuples
        self.attacker_timeout = 5.0  # Remove attackers after 5 seconds

        # --- Basic autonomous combat behavior (guard stance) ---
        # If True, current self.target was acquired automatically (not a direct player order).
        self.autonomous_target: bool = False
        # Small throttle to avoid reacquiring targets every frame (prevents path/position jitter).
        self._last_auto_acquire_time: float = -1e9
        self._auto_acquire_cooldown: float = 0.30

        # --- Player-issued advanced orders (attack-move, patrol) ---
        # Attack-move: walk to a destination, but if an enemy enters view_distance on the way,
        # engage once and do NOT resume the attack-move afterwards.
        self.attack_move_active: bool = False
        self.attack_move_dest: Optional[Vector2] = None

        # Patrol: walk between an anchor (position at time of order) and a destination.
        # If an enemy enters view_distance, engage once and do NOT resume patrol afterwards.
        self.patrol_active: bool = False
        self.patrol_anchor: Optional[Vector2] = None
        self.patrol_dest: Optional[Vector2] = None
        self.patrol_leg: str = "to_dest"  # "to_dest" or "to_anchor"

        # --- Progression (XP + levels) ---
        # Starts at novice (level 0) with 0 XP.
        self.xp: int = 0
        self.level: int = 0
        # Base stats are snapshotted lazily on first XP gain (see progression.ensure_base_stats)
        self.base_max_hp: int | None = None
        self.base_attack_damage: int | None = None
        self._progression_base_initialized: bool = False

    @classmethod
    def load_images(cls, cls_name, size):
        # Buildings: assets/units/buildings/{buildingname}.png
        # (for now) building_icon == building
        is_building = cls_name in {
            "Barn", "TownCenter", "Barracks", "ShamansHut",
            "WarriorsLodge", "KnightsEstate", "Ruin", "Wall",
        }

        # Tree is a variant-based world object (similar to Wall), but we still
        # want a default sprite + icon in the shared Unit caches.
        is_tree = (cls_name == "Tree")

        if is_building or is_tree:
            base = f"assets/units/buildings/{cls_name.lower()}"
            if cls_name == "Wall":
                # Wall sprites are variant-based: wall1.png ... wall12.png
                base = "assets/units/buildings/wall6"
            elif cls_name == "Tree":
                # Tree sprites are variant-based and live in assets/units/tree/
                # Default to tree6 if nothing else is specified.
                base = "assets/units/tree/tree6"
            sprite_path = f"{base}.png"
            icon_path = sprite_path  # ..._icon kept intentionally (future-proof)
            underbuilding_path = f"{base}_c.png"
        else:
            # Units: standardized to assets/units/{unit}/walk_M.gif
            # Miniatures: first frame of that gif
            sprite_path = None
            icon_path = None
            underbuilding_path = None

            # --- main sprite ---
        try:
            if not is_building and cls_name in GIF_UNITS:
                # Use first frame of walk_M.gif as base sprite (static fallback)
                cls._images[cls_name] = get_unit_walk_first_frame(cls_name, int(size), (255, 255, 255), facing="M")
            else:
                cls._images[cls_name] = pygame.image.load(sprite_path).convert_alpha()
                cls._images[cls_name] = pygame.transform.scale(cls._images[cls_name], (int(size), int(size)))
        except (pygame.error, FileNotFoundError, TypeError) as e:
            print(f"Failed to load base sprite for {cls_name}: {e}")
            cls._images[cls_name] = None

            # --- icon sprite ---
        try:
            if not is_building and cls_name in GIF_UNITS:
                # icon = first frame of walk_M.gif
                cls._unit_icons[cls_name] = get_unit_walk_first_frame(cls_name, ICON_SIZE, (255, 255, 255), facing="M")
            else:
                cls._unit_icons[cls_name] = pygame.image.load(icon_path).convert_alpha()
                cls._unit_icons[cls_name] = pygame.transform.scale(cls._unit_icons[cls_name], (ICON_SIZE, ICON_SIZE))
        except (pygame.error, FileNotFoundError, TypeError) as e:
            print(f"Failed to load icon for {cls_name}: {e}")
            cls._unit_icons[cls_name] = None

        # --- underbuilding sprite (construction), ruin is the exception ---
        if is_building and cls_name != "Ruin" and underbuilding_path:
            try:
                cls._underbuilding_images[cls_name] = pygame.image.load(underbuilding_path).convert_alpha()
                cls._underbuilding_images[cls_name] = pygame.transform.scale(
                    cls._underbuilding_images[cls_name], (int(size), int(size))
                )
            except (pygame.error, FileNotFoundError) as e:
                print(f"Failed to load {underbuilding_path}: {e}")
                cls._underbuilding_images[cls_name] = cls._images.get(cls_name)

    def should_highlight(self, current_time):
        return self in context.highlight_times and context.current_time - context.highlight_times[self] <= 0.4

    @staticmethod
    def _health_color(pct: float) -> Tuple[int, int, int]:
        """Green (1.0) -> Red (0.0) color for HP fill."""
        pct = max(0.0, min(1.0, pct))
        r = int(round(255 * (1.0 - pct)))
        g = int(round(255 * pct))
        return (r, g, 0)

    def draw_health_bar(self, screen: pygame.Surface, x: float, y: float) -> None:
        """Draw HP bar above the unit only if not at full health.

        - Bar width ~ unit size (with small padding)
        - Filled part uses green->red gradient based on HP%
        - Unfilled part uses player color
        """
        if isinstance(self, Tree):
            return
        if self.max_hp <= 0 or self.hp >= self.max_hp:
            return

        pct = max(0.0, min(1.0, self.hp / self.max_hp))

        pad = 4
        bar_w = max(10, int(self.size) - pad * 2)
        bar_h = 5
        bar_offset = 3

        bar_x = x - bar_w / 2
        bar_y = y - self.size / 2 - bar_h - bar_offset

        # Unfilled part in player color
        pc = tuple(self.player_color[:3])
        pygame.draw.rect(screen, pc, (bar_x, bar_y, bar_w, bar_h))

        # Filled part in HP gradient
        fill_w = int(round(bar_w * pct))
        if fill_w > 0:
            pygame.draw.rect(screen, self._health_color(pct), (bar_x, bar_y, fill_w, bar_h))

        # Outline
        pygame.draw.rect(screen, BLACK, (bar_x, bar_y, bar_w, bar_h), 1)

    # ---------------- Guard stance / autonomous targeting ----------------

    def _can_guard_auto_acquire(self) -> bool:
        """Return True when this unit should auto-acquire enemy targets (guard stance)."""
        # World objects and non-combat units (e.g., buildings) should never auto-acquire.
        if isinstance(self, (Building, Tree)):
            return False
        if getattr(self, "attack_damage", 0) <= 0:
            return False

        cls_name = self.__class__.__name__
        # Cows are explicitly excluded.
        if cls_name == "Cow":
            return False

        # Axemen: excluded only while doing worker-ish tasks (wood harvesting / repairing).
        # - harvesting wood is identified by having a Vector2 target that matches a Tree pos
        # - depositing/returning also counts as busy
        # TODO: when repairing is implemented, add a flag/state check here to disable guard while repairing.
        if cls_name == "Axeman":
            if getattr(self, "depositing", False):
                return False
            if isinstance(getattr(self, "target", None), Vector2):
                tpos = getattr(self, "target", None)
                if tpos is not None:
                    # If currently chopping a tree, don't auto-acquire.
                    for u in getattr(context, "all_units", []) or []:
                        if isinstance(u, Tree) and u.player_id == 0 and u.pos == tpos:
                            return False

        # Normal move command: target is Vector2 and this wasn't an autonomous target.
        if isinstance(getattr(self, "target", None), Vector2) and not getattr(self, "autonomous_target", False):
            return False

        return True

    def _guard_auto_acquire(self, units) -> None:
        """If idle (no target), pick a nearby enemy within aggro_distance and start chasing/attacking."""
        if self.target is not None:
            return

        if not self._can_guard_auto_acquire():
            return

        now = float(getattr(context, "current_time", 0.0) or 0.0)
        if (now - getattr(self, "_last_auto_acquire_time", -1e9)) < getattr(self, "_auto_acquire_cooldown", 0.30):
            return
        self._last_auto_acquire_time = now

        # Only use spatial grid if available; otherwise fall back to scanning.
        radius_px = float(getattr(self, "aggro_distance", 0)) * TILE_SIZE
        if radius_px <= 0:
            return

        grid = getattr(context, "spatial_grid", None)
        candidates_units = []
        candidates_buildings = []
        if grid is not None:
            nearby = grid.get_nearby_units(self, radius=radius_px)
        else:
            nearby = units

        for u in nearby:
            if u is self:
                continue
            if not isinstance(u, Unit) or isinstance(u, Tree):
                continue
            if getattr(u, "hp", 0) <= 0 or u not in units:
                continue

            # Ignore Gaia (player0) unless the player explicitly orders an attack.
            if getattr(u, "player_id", None) == 0:
                continue

            if getattr(u, "player_id", None) == getattr(self, "player_id", None):
                continue

            d = self.pos.distance_to(u.pos)
            if d > radius_px:
                continue

            # Units take priority over buildings.
            if isinstance(u, Building):
                candidates_buildings.append((d, u))
            else:
                candidates_units.append((d, u))

        if candidates_units:
            candidates_units.sort(key=lambda t: t[0])
            target = candidates_units[0][1]
        elif candidates_buildings:
            candidates_buildings.sort(key=lambda t: t[0])
            target = candidates_buildings[0][1]
        else:
            return
        self.target = target
        self.autonomous_target = True
        self.path = []
        self.path_index = 0
        self.last_target = None

    # ---------------- Attack-move / Patrol helpers ----------------

    def _clear_advanced_orders(self) -> None:
        """Clear attack-move / patrol state. Call this whenever a new direct order is issued."""
        self.attack_move_active = False
        self.attack_move_dest = None
        self.patrol_active = False
        self.patrol_anchor = None
        self.patrol_dest = None
        self.patrol_leg = "to_dest"

        # Also cancel any repair order (Axeman repairs should stop on any new manual order).
        if hasattr(self, 'repair_target'):
            self.repair_target = None
        if hasattr(self, '_repair_last_time'):
            self._repair_last_time = None

    def _scan_for_enemy_in_view(self, units) -> Optional["Unit"]:
        """Return closest enemy (unit/building) within view_distance. Excludes Trees."""
        view_px = float(getattr(self, "view_distance", 0)) * TILE_SIZE
        if view_px <= 0:
            return None

        grid = getattr(context, "spatial_grid", None)
        nearby = grid.get_nearby_units(self, radius=view_px) if grid is not None else (units or [])

        best_unit = None
        best_unit_d = 1e18
        best_building = None
        best_building_d = 1e18

        for u in nearby:
            if u is self:
                continue
            if not isinstance(u, Unit) or isinstance(u, Tree):
                continue
            if getattr(u, "hp", 0) <= 0 or (units is not None and u not in units):
                continue

            # Ignore Gaia (player0) for auto-scans (attack-move / patrol). Only direct attack orders should target Gaia.
            if getattr(u, "player_id", None) == 0:
                continue

            if getattr(u, "player_id", None) == getattr(self, "player_id", None):
                continue

            d = self.pos.distance_to(u.pos)
            if d > view_px:
                continue

            # Units take priority over buildings.
            if isinstance(u, Building):
                if d < best_building_d:
                    best_building = u
                    best_building_d = d
            else:
                if d < best_unit_d:
                    best_unit = u
                    best_unit_d = d

        return best_unit if best_unit is not None else best_building

    def draw(self, screen, camera_x, camera_y):
        if (self.pos.x < camera_x - self.size / 2 or self.pos.x > camera_x + VIEW_WIDTH + self.size / 2 or
            self.pos.y < camera_y - self.size / 2 or self.pos.y > camera_y + VIEW_HEIGHT + self.size / 2):
            return
        cls_name = self.__class__.__name__

        # If a building is still being constructed (alpha < 255), use *_c.png where available.
        # Ruin is the exception: it only has the normal sprite.
        if isinstance(self, Building) and self.alpha < 255 and cls_name != "Ruin":
            image = Unit._underbuilding_images.get(cls_name) or get_team_sprite(
                cls_name, int(self.size), tuple(self.player_color[:3])
            )
        else:
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
            _blit_sprite_with_border(screen, image_surface, image_rect)
        # Fire overlay for damaged buildings (drawn on top of building sprite).
        if isinstance(self, Building) and hasattr(self, "_draw_damage_fire"):
            try:
                self._draw_damage_fire(screen, x, y)
            except Exception as e:
                _fire_warn_once(f"[fire] draw failed for {type(self).__name__}: {e}")
        if self.selected:
            pygame.draw.rect(screen, self.player_color, (x - self.size / 2, y - self.size / 2, self.size, self.size), 1)
        if self.should_highlight(context.current_time):
            pygame.draw.rect(screen, WHITE, (x - self.size / 2, y - self.size / 2, self.size, self.size), 1)
        # HP bar (only when not fully healed)
        self.draw_health_bar(screen, x, y)

        # Draw pathfinding path
        if self.path and len(self.path[self.path_index:]) >= 2:  # Ensure at least 2 points
            points = [(p.x - camera_x + VIEW_MARGIN_LEFT, p.y - camera_y + VIEW_MARGIN_TOP) for p in self.path[self.path_index:]]
            # pygame.draw.lines(screen, WHITE, False, points, 1)

    def move(self, units, spatial_grid=None, waypoint_graph=None):
        # --- Advanced player orders: attack-move / patrol ---
        if getattr(self, "attack_move_active", False) or getattr(self, "patrol_active", False):
            # If an enemy enters view_distance while travelling, engage once and do NOT resume.
            enemy = self._scan_for_enemy_in_view(units)
            if enemy is not None and not isinstance(getattr(self, "target", None), Unit):
                self.target = enemy
                self.autonomous_target = False
                self._clear_advanced_orders()
                self.path = []
                self.path_index = 0
                self.last_target = None

            # Patrol: keep bouncing between anchor and destination (only while not fighting).
            if getattr(self, "patrol_active", False) and enemy is None and not isinstance(getattr(self, "target", None), Unit):
                if self.patrol_anchor is None or self.patrol_dest is None:
                    self._clear_advanced_orders()
                else:
                    # Determine which point we are *currently* patrolling towards.
                    desired_goal = Vector2(self.patrol_dest) if self.patrol_leg == "to_dest" else Vector2(self.patrol_anchor)

                    goal = self.target if isinstance(getattr(self, "target", None), Vector2) else None
                    if goal is None:
                        # If our travel target got cleared (e.g. temporary path failure), re-issue the same leg
                        # instead of toggling back and forth every frame (this was breaking group patrol).
                        self.target = Vector2(desired_goal)
                        self.autonomous_target = False
                        self.path = []
                        self.path_index = 0
                        self.last_target = None
                    else:
                        # Consider "reached" when very close to the goal.
                        if self.pos.distance_to(goal) <= max(6.0, float(self.size) * 0.35):
                            # Flip to the other leg.
                            self.patrol_leg = "to_anchor" if self.patrol_leg == "to_dest" else "to_dest"
                            next_goal = Vector2(self.patrol_dest) if self.patrol_leg == "to_dest" else Vector2(self.patrol_anchor)

                            self.target = Vector2(next_goal)
                            self.autonomous_target = False
                            self.path = []
                            self.path_index = 0
                            self.last_target = None

        # Guard stance: if idle and allowed, auto-acquire a nearby enemy.
        if not self.target:
            self._guard_auto_acquire(units)

        # If still no target, remain idle.
        if not self.target:
            self.path = []
            self.path_index = 0
            self.velocity = Vector2(0, 0)
            return

        # If this was an autonomous chase and the target is far beyond view_distance,
        # stop following (prevents chasing across the whole map).
        if (
            getattr(self, "autonomous_target", False)
            and isinstance(self.target, Unit)
            and not isinstance(self.target, Tree)
        ):
            view_px = float(getattr(self, "view_distance", 0)) * TILE_SIZE
            if view_px > 0 and self.pos.distance_to(self.target.pos) > view_px:
                self.target = None
                self.autonomous_target = False
                self.path = []
                self.path_index = 0
                self.last_target = None
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
            # Stop distance for combat targets: include both radii so melee can get right up to the target.
            # Small epsilon keeps units from hovering with a visible gap.
            stop_distance = max(2.0, float(self.attack_range) + (float(self.size) + float(self.target.size)) / 2.0 - 2.0)
            # Check if target moved significantly since last path calculation
            if self.path and self.path_index < len(self.path):
                last_waypoint = self.path[-1]
                if (target_pos - last_waypoint).length() > self.size:  # Target moved more than unit size
                    self.path = []  # Force path recalculation
                    self.path_index = 0
        else:
            target_pos = self.target
            # Move-to-point stop distance: keep it small so units can approach closely without leaving gaps.
            stop_distance = (self.size / 4) if isinstance(self, Axeman) else max(2.0, float(self.size) * 0.20)

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
                        if isinstance(self, Axeman) and isinstance(self.target, Vector2) and not (getattr(self, 'patrol_active', False) or getattr(self, 'attack_move_active', False)):
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
                    if isinstance(self, Axeman) and isinstance(self.target, Vector2) and not (getattr(self, 'patrol_active', False) or getattr(self, 'attack_move_active', False)):
                        self.velocity = Vector2(0, 0)
                        # print(f"Axeman at {self.pos} reached path end, distance to {self.target}: {self.pos.distance_to(self.target):.1f}")
                    elif not isinstance(self.target, Unit):  # Only clear non-Unit targets
                        self.target = None
                        self.path = []
                        self.path_index = 0
                        self.last_target = None
        else:
            self.velocity = Vector2(0, 0)

        # Apply velocity using sub-steps to prevent tunneling through thin gaps
        delta = Vector2(self.velocity)

        max_step = TILE_SIZE * 0.25  # <= quarter-tile per micro-step
        dist = delta.length()

        if dist > 0:
            steps = max(1, int(math.ceil(dist / max_step)))
            step_vec = delta / steps

            for _ in range(steps):
                self.pos += step_vec
                # resolve collisions immediately so we can't jump through blockers
                self.resolve_collisions(units, context.spatial_grid)

        # apply damping after movement (friction-like)
        self.velocity *= self.damping

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
        # If we are in a long-running player order (patrol / attack-move), being attacked counts as engaging in combat.
        if hasattr(self, "_clear_advanced_orders"):
            self._clear_advanced_orders()
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
        valid_attackers = [(a, t) for a, t in self.attackers if a.hp > 0 and a in context.all_units and getattr(a, 'player_id', None) != 0]
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
                hp_before = float(getattr(target, "hp", 0))
                target.hp -= damage
                # Track last attacker for endgame stats attribution
                setattr(target, '_last_attacker', self)
                # Progression XP: 1 per damaging hit, bonus on kill/destroy.
                progression.award_combat_xp(self, target, damage=damage, target_hp_before=hp_before)
                self.last_attack_time = context.current_time
                print(f"{self.__class__.__name__} at {self.pos} attacked {target.__class__.__name__} at {target.pos}, dealing {damage} damage")
                # Notify target of attack for defensive behavior
                if isinstance(target, (Axeman, Archer, Knight, Bear, Strzyga, Priestess, Shaman, Swordsman, Spearman)) and target.hp > 0:
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

        def rect_for(u):
            return pygame.Rect(u.pos.x - u.size / 2, u.pos.y - u.size / 2, u.size, u.size)

        def aabb_push_out(moving_rect: pygame.Rect, static_rect: pygame.Rect):
            # Returns a Vector2 correction that minimally separates moving_rect from static_rect
            if not moving_rect.colliderect(static_rect):
                return Vector2(0, 0)

            dx1 = static_rect.right - moving_rect.left  # push moving right
            dx2 = moving_rect.right - static_rect.left  # push moving left
            dy1 = static_rect.bottom - moving_rect.top  # push moving down
            dy2 = moving_rect.bottom - static_rect.top  # push moving up

            # Choose smallest magnitude axis push
            push_x = dx1 if dx1 < dx2 else -dx2
            push_y = dy1 if dy1 < dy2 else -dy2

            if abs(push_x) < abs(push_y):
                return Vector2(push_x, 0)
            else:
                return Vector2(0, push_y)

        nearby_units = context.spatial_grid.get_nearby_units(self)
        self_rect = rect_for(self)

        total_correction = Vector2(0, 0)

        for other in nearby_units:
            if other is self:
                continue

            # --- STATIC obstacles: Buildings + Trees (AABB push-out prevents corner sticking) ---
            if isinstance(other, (Tree, Building)):
                # Keep your special case: Cow can enter Barn
                if isinstance(self, Cow) and isinstance(other, Barn) and other.player_id == self.player_id:
                    continue

                other_rect = rect_for(other)
                corr = aabb_push_out(self_rect, other_rect)
                if corr.length_squared() > 0:
                    total_correction += corr
                    # update rect so multiple collisions stack correctly
                    self_rect.move_ip(corr.x, corr.y)
                continue

            # --- UNIT vs UNIT: friendly-only nudging + avoid stacking ---
            # Treat units as soft circles for separation.
            delta = self.pos - other.pos
            dist2 = delta.length_squared()
            if dist2 < 1e-8:
                # prevent NaNs
                delta = Vector2(1, 0)
                dist2 = 1.0

            dist = math.sqrt(dist2)
            min_dist = (self.size + other.size) / 2.0
            if dist < min_dist:
                overlap = min_dist - dist
                n = delta / dist  # collision normal (from other -> self)

                same_player = (getattr(other, "player_id", None) == getattr(self, "player_id", None))

                # Never push enemies: if we overlap an enemy, only we get corrected.
                if not same_player:
                    corr_self = n * overlap
                    total_correction += corr_self
                    self_rect.move_ip(corr_self.x, corr_self.y)

                    # remove velocity component that drives us INTO the enemy
                    vn_self = self.velocity.dot(-n)
                    if vn_self > 0:
                        self.velocity += n * vn_self
                    continue

                # Friendly nudging: allow much stronger pushing of idle friendlies
                def _is_idle(u: Unit) -> bool:
                    if isinstance(u, (Building, Tree)):
                        return True
                    if getattr(u, "target", None) is not None:
                        return False
                    if getattr(u, "attack_move_active", False) or getattr(u, "patrol_active", False):
                        return False
                    if getattr(u, "autonomous_target", False):
                        return False
                    if getattr(u, "velocity", Vector2(0, 0)).length_squared() > 1.0:
                        return False
                    # worker-like busy flags
                    if getattr(u, "depositing", False):
                        return False
                    return True

                self_idle = _is_idle(self)
                other_idle = _is_idle(other)

                # weights decide how much each side moves (sum == 1.0)
                if (not self_idle) and other_idle:
                    w_self, w_other = 0.15, 0.85  # active unit plows through idle
                elif self_idle and (not other_idle):
                    w_self, w_other = 0.85, 0.15  # idle yields less if the other is active
                else:
                    w_self, w_other = 0.5, 0.5

                corr_self = n * (overlap * w_self)
                corr_other = -n * (overlap * w_other)

                total_correction += corr_self
                self_rect.move_ip(corr_self.x, corr_self.y)

                # Apply immediately to the other unit so the pusher can get through.
                other.pos += corr_other

                # remove velocity components that drive units INTO each other (reduces "conga lines")
                vn_self = self.velocity.dot(-n)
                if vn_self > 0:
                    self.velocity += n * vn_self

                vn_other = other.velocity.dot(n)
                if vn_other > 0:
                    other.velocity -= n * vn_other
        if total_correction.length_squared() > 0:
            self.pos += total_correction

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
    _variant_icons = {}   # variant -> icon surface (scaled from sprite)
    _c_images = {}  # variant -> construction sprite (variant_c) or base if missing

    _selected = False
    _last_color_change_time = 0
    _color_index = 0
    _colors = [RED, GREEN, BLUE, YELLOW]

    # Define variants in one place (sprites + stats).
    # Sprites are expected under assets/units/tree/ (e.g. assets/units/tree/tree0.png).
    VARIANTS = {
        "tree0": {
            "sprite": "assets/units/tree/tree0.png",
            "hp": 900,
            "max_hp": 900,
        },
        "tree1": {
            "sprite": "assets/units/tree/tree1.png",
            "hp": 900,
            "max_hp": 900,
        },
        "tree2": {
            "sprite": "assets/units/tree/tree2.png",
            "hp": 900,
            "max_hp": 900,
        },
        "tree3": {
            "sprite": "assets/units/tree/tree3.png",
            "hp": 900,
            "max_hp": 900,
        },
        "tree4": {
            "sprite": "assets/units/tree/tree4.png",
            "hp": 900,
            "max_hp": 900,
        },
        "tree5": {
            "sprite": "assets/units/tree/tree5.png",
            "hp": 900,
            "max_hp": 900,
        },
        "tree6": {
            "sprite": "assets/units/tree/tree6.png",
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

            # Optional construction sprite (variant_c). If missing, use the base sprite.
            c_path = path[:-4] + "_c.png"
            try:
                c_img = pygame.image.load(c_path).convert_alpha()
                scale_factor_c = min(TILE_SIZE / c_img.get_width(), TILE_SIZE / c_img.get_height())
                new_size_c = (int(c_img.get_width() * scale_factor_c), int(c_img.get_height() * scale_factor_c))
                c_img = pygame.transform.scale(c_img, new_size_c)
                cls._c_images[variant] = c_img
            except (pygame.error, FileNotFoundError):
                cls._c_images[variant] = img

            # Miniature/icon for UI: use the SAME sprite of this variant (no separate icon assets)
            try:
                cls._variant_icons[variant] = pygame.transform.smoothscale(img, (ICON_SIZE, ICON_SIZE))
            except Exception:
                cls._variant_icons[variant] = img

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
            cls._variant_icons[variant] = None

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
            _blit_sprite_with_border(screen, img_to_draw, rect)

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
        # When selecting a specific tree variant, keep the UI icon consistent
        # with the actual sprite used for this instance.
        if selected:
            icon = Tree._variant_icons.get(self.variant)
            if icon is not None:
                Unit._unit_icons["Tree"] = icon
        # Make sure the UI icon matches the selected tree variation.
        # (UI caches icons by class name only, so we update it on selection.)
        if selected:
            icon = Tree._variant_icons.get(self.variant)
            if icon is None and self.variant not in Tree._images:
                Tree.load_image(self.variant)
                icon = Tree._variant_icons.get(self.variant)
            if icon is not None:
                Unit._unit_icons["Tree"] = icon

    def move(self, units, spatial_grid=None, waypoint_graph=None):
        pass

    def attack(self, target, current_time):
        return False

# Building class
class Building(Unit):

    production_time = 30.0  # Production time for buildings (seconds)

    # Footprint size in tiles (N x N). Most buildings are 3x3. Override in subclasses.
    SIZE_TILES = 3

    @classmethod
    def size_px(cls) -> int:
        """Pixel size (square) derived from SIZE_TILES."""
        return int(cls.SIZE_TILES) * TILE_SIZE

    def __init__(self, x, y, *, color, player_id, player_color):
        super().__init__(x, y, self.size_px(), speed=0, color=color, player_id=player_id, player_color=player_color)
        self.hp = 150  # High HP for buildings
        self.max_hp = 150
        self.armor = 5  # Buildings have armor
        self.attack_damage = 0  # Buildings cannot attack
        self.attack_range = 0
        self.rally_point = None  # Initialize rally point as None

        # --- Damage fire overlay state (buildings only) ---
        self._fire_level: int = 0
        self._fire_slots: List[Tuple[int, int]] = []
        self._fire_jitter: List[Tuple[int, int]] = []

    
_FIRE_FRAME_TIME: float = 0.12

# One-time warnings so asset/path issues don't get swallowed silently.
_fire_warned: set = set()
def _fire_warn_once(msg: str) -> None:
    if msg in _fire_warned:
        return
    _fire_warned.add(msg)
    try:
        print(msg)
    except Exception:
        pass

def _compute_fire_state(self) -> tuple[int, int]:
    """Return (flame_level, flame_count) based on building HP percent.
    Levels map to flame1/2/3 assets. Count increases 1→2→3 as damage worsens,
    and decreases during repair without changing already-chosen positions."""
    mx = float(getattr(self, "max_hp", 0) or 0)
    if mx <= 0:
        return (0, 0)
    hp = float(getattr(self, "hp", 0) or 0)
    pct = max(0.0, min(1.0, hp / mx))
    if pct > 0.75:
        return (0, 0)
    if pct > 0.50:
        return (1, 1)
    if pct > 0.25:
        return (2, 2)
    return (3, 3)

def _ensure_fire_slots(self) -> None:
    """Pick 3 stable offsets (and jitters) once per building."""
    if not hasattr(self, "_fire_slots") or not isinstance(getattr(self, "_fire_slots"), list):
        self._fire_slots = []
    if not hasattr(self, "_fire_jitter") or not isinstance(getattr(self, "_fire_jitter"), list):
        self._fire_jitter = []
    if len(self._fire_slots) >= 3 and len(self._fire_jitter) >= 3:
        return

    candidates = [(ox, oy) for ox in (-1, 0, 1) for oy in (-1, 0, 1)]
    # Ensure we get 3 unique positions; if not enough candidates, fall back safely.
    slots = random.sample(candidates, 3) if len(candidates) >= 3 else candidates
    self._fire_slots = slots

    j = max(1, int(round(TILE_SIZE * 0.15)))
    self._fire_jitter = [(random.randint(-j, j), random.randint(-j, j)) for _ in range(len(self._fire_slots))]

def _draw_damage_fire(self, screen: pygame.Surface, x: float, y: float) -> None:
    # Only for 3x3 buildings and only when fully visible.
    if getattr(self, "SIZE_TILES", 1) < 3:
        return
    if getattr(self, "alpha", 255) < 255:
        return

    level, count = self._compute_fire_state()
    if count <= 0 or level <= 0:
        return

    self._ensure_fire_slots()
    frames = get_fire_frames(level, int(TILE_SIZE))
    if not frames:
        _fire_warn_once(f"[fire] missing frames for level={level} (check flame assets paths/names)")
        return

    now = float(getattr(context, "current_time", 0.0) or 0.0)
    idx = int(now / float(self._FIRE_FRAME_TIME)) % len(frames)
    fr = frames[idx]

    # Lift a bit so flames appear on the roof/top of the 3x3 sprite.
    lift = int(round(TILE_SIZE * 0.35))

    # Draw first N slots to keep earlier flames fixed as intensity grows/shrinks.
    n = min(int(count), len(self._fire_slots))
    for i in range(n):
        ox, oy = self._fire_slots[i]
        jx, jy = self._fire_jitter[i] if i < len(self._fire_jitter) else (0, 0)
        fx = float(x) + float(ox) * float(TILE_SIZE) + float(jx)
        fy = float(y) + float(oy) * float(TILE_SIZE) + float(jy) - float(lift)
        rect = fr.get_rect(center=(int(fx), int(fy)))
        screen.blit(fr, rect)

# Bind fire helpers as Building methods so they are accessible as self._...()
Building._FIRE_FRAME_TIME = _FIRE_FRAME_TIME
Building._compute_fire_state = _compute_fire_state
Building._ensure_fire_slots = _ensure_fire_slots
Building._draw_damage_fire = _draw_damage_fire

# Barn class
class Barn(Building):
    milk_cost = 0
    wood_cost = 300
    def __init__(self, x, y, player_id, player_color):
        super().__init__(x, y, color=DARK_GRAY, player_id=player_id, player_color=player_color)
        self.harvest_rate = 60.0

# Barn class
class ShamansHut(Building):
    milk_cost = 200
    wood_cost = 300
    def __init__(self, x, y, player_id, player_color):
        super().__init__(x, y, color=DARK_GRAY, player_id=player_id, player_color=player_color)

# New buildings (treat like ShamansHut for now)
class KnightsEstate(Building):
    milk_cost = 100
    wood_cost = 100
    def __init__(self, x, y, player_id, player_color):
        super().__init__(x, y, color=DARK_GRAY, player_id=player_id, player_color=player_color)

class WarriorsLodge(Building):
    milk_cost = 100
    wood_cost = 100
    def __init__(self, x, y, player_id, player_color):
        super().__init__(x, y, color=DARK_GRAY, player_id=player_id, player_color=player_color)

class Ruin(Building):
    milk_cost = 100
    wood_cost = 100
    def __init__(self, x, y, player_id, player_color):
        super().__init__(x, y, color=DARK_GRAY, player_id=player_id, player_color=player_color)


# Wall building (tile-sized). Behaves like other buildings (hp, player, blocking, selectable),
# but uses per-variant sprites from assets/units/buildings/wallN.png
class Wall(Building):
    milk_cost = 0
    wood_cost = 5

    _variant_images: Dict[str, Optional[pygame.Surface]] = {}
    DEFAULT_VARIANT = "wall6"
    VARIANT_MIN = 1
    VARIANT_MAX = 12

    production_time = 5.0

    # Walls are 1x1 tile
    SIZE_TILES = 1

    def __init__(self, x, y, player_id, player_color, variant: Optional[str] = None):
        super().__init__(x, y, color=DARK_GRAY, player_id=player_id, player_color=player_color)

        self.variant = self._sanitize_variant(variant) or self.DEFAULT_VARIANT

        # Walls: sturdy, but tweakable
        self.hp = 150
        self.max_hp = 150
        self.armor = 6
        self.attack_damage = 0
        self.attack_range = 0

        if self.variant not in Wall._variant_images:
            Wall._variant_images[self.variant] = Wall._load_variant_image(self.variant)

    @staticmethod
    def _sanitize_variant(v: Optional[str]) -> Optional[str]:
        if not isinstance(v, str) or not v.startswith("wall"):
            return None
        try:
            idx = int(v[len("wall"):])
        except Exception:
            return None
        return v if Wall.VARIANT_MIN <= idx <= Wall.VARIANT_MAX else None

    @classmethod
    def _load_variant_image(cls, variant: str) -> Optional[pygame.Surface]:
        # NOTE: user moved wall sprites under assets/units/buildings/
        path = os.path.join("assets", "units", "buildings", f"{variant}.png")
        try:
            img = pygame.image.load(path).convert_alpha()
            return pygame.transform.scale(img, (TILE_SIZE, TILE_SIZE))
        except (pygame.error, FileNotFoundError) as e:
            print(f"Failed to load {path}: {e}")
            return None

    def set_variant(self, variant: str) -> None:
        """Set variant and lazy-load its sprite if needed."""
        v = self._sanitize_variant(variant) or self.DEFAULT_VARIANT
        self.variant = v
        if v not in Wall._variant_images:
            Wall._variant_images[v] = Wall._load_variant_image(v)

    def draw(self, screen, camera_x, camera_y):
        # Same culling as Unit.draw()
        if (self.pos.x < camera_x - self.size / 2 or self.pos.x > camera_x + VIEW_WIDTH + self.size / 2 or
            self.pos.y < camera_y - self.size / 2 or self.pos.y > camera_y + VIEW_HEIGHT + self.size / 2):
            return

        x = self.pos.x - camera_x + VIEW_MARGIN_LEFT
        y = self.pos.y - camera_y + VIEW_MARGIN_TOP

        img = Wall._variant_images.get(self.variant)
        if img is not None:
            image_surface = img.copy()
            image_surface.set_alpha(self.alpha)
            rect = image_surface.get_rect(center=(int(x), int(y)))
            _blit_sprite_with_border(screen, image_surface, rect)
        else:
            surface = pygame.Surface((self.size, self.size), pygame.SRCALPHA)
            surface.fill((60, 60, 60, int(self.alpha)))
            screen.blit(surface, (x - self.size / 2, y - self.size / 2))

        if self.selected:
            pygame.draw.rect(screen, self.player_color, (x - self.size / 2, y - self.size / 2, self.size, self.size), 1)
        if self.should_highlight(context.current_time):
            pygame.draw.rect(screen, WHITE, (x - self.size / 2, y - self.size / 2, self.size, self.size), 1)

        self.draw_health_bar(screen, x, y)



# TownCenter class
class TownCenter(Building):
    milk_cost = 0
    wood_cost = 800
    def __init__(self, x, y, player_id, player_color):
        super().__init__(x, y, color=TOWN_CENTER_GRAY, player_id=player_id, player_color=player_color)
        self.hp = 100  # High HP for buildings
        self.max_hp = 200
        self.armor = 5

# Barracks class
class Barracks(Building):
    milk_cost = 0
    wood_cost = 500
    def __init__(self, x, y, player_id, player_color):
        super().__init__(x, y, color=TOWN_CENTER_GRAY, player_id=player_id, player_color=player_color)

# Axeman class
class Axeman(Unit):
    milk_cost = 300
    wood_cost = 0

    # folder z animacjami:
    # assets/units/axeman/walk_D.gif, walk_L.gif, walk_LD.gif, walk_LU.gif, walk_M.gif, walk_U.gif
    _ANIM_DIR = "assets/units/axeman"

    # klatki na sekundę (czas na klatkę)
    _FRAME_TIME = 0.5  # 0.10s = 10 FPS; dostosuj jak chcesz
    _IDLE_SPEED_EPS2 = 0.05  # threshold dla uznania "stoi"

    _ATTACK_FRAME_TIME = 0.40

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

        # ---- attack animation state (like Archer) ----
        self._attacking_until = 0.0
        self._attack_facing = "D"

        self.repair_target = None          # target building (or None)
        self._repair_last_time = None
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

        now = context.current_time

        # Decide which animation to show
        if now < self._attacking_until:
            facing = self._attack_facing or "D"
            frames = get_unit_attack_frames("Axeman", int(self.size), tuple(self.player_color[:3]), facing=facing)
            frame_time = self._ATTACK_FRAME_TIME
        else:
            facing = self._facing_from_velocity()
            if facing != "M":
                self._last_facing = facing
            frames = self._get_anim_frames_for_facing(facing)
            frame_time = self._FRAME_TIME

        if not frames:
            return super().draw(screen, camera_x, camera_y)

        idx = int(now / frame_time) % len(frames)
        image = frames[idx]

        x = self.pos.x - camera_x + VIEW_MARGIN_LEFT
        y = self.pos.y - camera_y + VIEW_MARGIN_TOP

        image_surface = image.copy()
        image_surface.set_alpha(self.alpha)

        # --- special alignment for oversized ATTACK frames ---
        if now < self._attacking_until:
            CORE_W, CORE_H = 16, 14  # "original" sprite box you want to align

            w, h = image_surface.get_width(), image_surface.get_height()

            # anchor tells where the core box sits inside the big image:
            # (ax, ay) in {"L","C","R"} x {"T","C","B"}
            # We'll compute dx/dy so that the core's center equals (x, y).
            if facing in ("L", "LU"):
                ax, ay = "R", "B"  # bottom-right
            elif facing == "LD":
                ax, ay = "R", "T"  # top-right
            elif facing in ("R", "RU"):
                ax, ay = "L", "B"  # bottom-left (mirrored from L/LU)
            elif facing == "RD":
                ax, ay = "L", "T"  # top-left (mirrored from LD)
            else:
                # D/U/M etc: default center the big image on unit
                ax, ay = "C", "C"

            # where is the core's top-left inside the big image?
            if ax == "L":
                core_x0 = 0
            elif ax == "C":
                core_x0 = (w - CORE_W) // 2
            else:  # "R"
                core_x0 = w - CORE_W

            if ay == "T":
                core_y0 = 0
            elif ay == "C":
                core_y0 = (h - CORE_H) // 2
            else:  # "B"
                core_y0 = h - CORE_H

            # core center in image-local coordinates:
            core_cx = core_x0 + CORE_W / 2
            core_cy = core_y0 + CORE_H / 2

            # shift whole image so core center lands on (x,y)
            dx = int(x - core_cx)
            dy = int(y - core_cy)
            image_rect = image_surface.get_rect(topleft=(dx, dy))
        else:
            image_rect = image_surface.get_rect(center=(int(x), int(y)))

        # Per-unit manual offset for ATTACK animations (auto-scaled by SCALE)
        if now < self._attacking_until:
            dx, dy = _scaled_attack_offset("Axeman", facing)
            image_rect.move_ip(dx, dy)

        _blit_sprite_with_border(screen, image_surface, image_rect)

        if self.selected:
            pygame.draw.rect(screen, self.player_color,
                             (x - self.size / 2, y - self.size / 2, self.size, self.size), 1)

        if self.should_highlight(context.current_time):
            pygame.draw.rect(screen, WHITE,
                             (x - self.size / 2, y - self.size / 2, self.size, self.size), 1)

        # HP bar (only when not fully healed)
        self.draw_health_bar(screen, x, y)

        # ścieżka jak w Unit.draw (zostawiam wyłączone)
        if self.path and len(self.path[self.path_index:]) >= 2:
            points = [(p.x - camera_x + VIEW_MARGIN_LEFT, p.y - camera_y + VIEW_MARGIN_TOP)
                      for p in self.path[self.path_index:]]
            # pygame.draw.lines(screen, WHITE, False, points, 1)

    def move(self, units, spatial_grid=None, waypoint_graph=None):
        self.velocity = Vector2(0, 0)

        # Repair order: move to target building first and ignore all auto behavior.
        if self.repair_target is not None:
            b = self.repair_target
            # Cancel if invalid / dead / already fully healed
            if getattr(b, 'hp', 0) <= 0 or getattr(b, 'player_id', None) != self.player_id:
                self.repair_target = None
            elif getattr(b, 'hp', 0) >= getattr(b, 'max_hp', b.hp):
                self.repair_target = None
            else:
                # Keep a point-target at the building center; Unit.move keeps Axeman point-targets.
                self.target = Vector2(b.pos)
                self.autonomous_target = False
                # Do not allow harvesting loops while repairing
                self.depositing = False
                # Move toward the building; stop near it (repair happens in repair_building).
                # Use normal pathing so formations avoid obstacles.
                super().move(units, context.spatial_grid, context.waypoint_graph)
                return

        # Idle state: No target, remain idle unless commanded
        if not self.target and not self.depositing:
            # Guard stance (Axeman only when not chopping wood / depositing).
            self._guard_auto_acquire(units)
            if not self.target:
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
        if self.special > 0 or self.depositing or self.target == self.return_pos or self.repair_target is not None:
            return

        target_tree = next(
            (
                tree for tree in trees
                if isinstance(tree, Tree)
                   and tree.player_id == 0
                   and self.target == tree.pos
                   and self.pos.distance_to(tree.pos) <= TILE_SIZE
            ),
            None,
        )
        if not target_tree:
            return

        # ------------------ trigger SAME attack animation while chopping ------------------
        now = context.current_time

        # face toward the tree (same directional rules as in attack())
        v = (target_tree.pos - self.pos)
        if v.length_squared() > 1e-6:
            x, y = v.x, v.y
            if abs(x) < 0.35 * abs(y) and y < 0:
                self._attack_facing = "U"
            elif abs(x) < 0.35 * abs(y) and y > 0:
                self._attack_facing = "D"
            elif abs(y) < 0.35 * abs(x) and x < 0:
                self._attack_facing = "L"
            elif abs(y) < 0.35 * abs(x) and x > 0:
                self._attack_facing = "R"
            elif x < 0 and y < 0:
                self._attack_facing = "LU"
            elif x < 0 and y > 0:
                self._attack_facing = "LD"
            elif x > 0 and y < 0:
                self._attack_facing = "RU"
            elif x > 0 and y > 0:
                self._attack_facing = "RD"
            else:
                self._attack_facing = "D"
        else:
            self._attack_facing = self._last_facing or "D"

        # set attacking window long enough to play the gif once
        attack_frames = get_unit_attack_frames(
            "Axeman",
            int(self.size),
            tuple(self.player_color[:3]),
            facing=self._attack_facing,
        )
        frame_time = self._ATTACK_FRAME_TIME
        anim_len = (len(attack_frames) * frame_time) if attack_frames else 0.35
        self._attacking_until = max(self._attacking_until, now + anim_len)
        # -------------------------------------------------------------------------------

        # actual chop damage
        target_tree.hp -= self.chop_damage

        if target_tree.hp <= 0:
            self.return_pos = Vector2(self.pos)
            context.players[0].remove_unit(target_tree)
            context.all_units.remove(target_tree)
            context.spatial_grid.remove_unit(target_tree)
            self.special = 25
            print(f"Tree at {target_tree.pos} chopped down by Axeman at {self.pos}, special = {self.special}")

            town_centers = [
                unit for unit in context.all_units
                if isinstance(unit, TownCenter)
                   and unit.player_id == self.player_id
                   and unit.alpha == 255
            ]
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

    def attack(self, target, current_time):
        # Use existing generic logic, but start attack anim window when a hit is made.
        if not isinstance(target, Unit) or isinstance(target, Tree) or target.hp <= 0 or target not in context.all_units:
            return

        distance = (self.pos - target.pos).length()
        max_range = self.attack_range + self.size / 2 + target.size / 2
        if distance > max_range:
            return

        if context.current_time - self.last_attack_time < self.attack_cooldown:
            return

        # lock facing for attack (use last movement if close to zero velocity)
        v = (target.pos - self.pos)
        if v.length_squared() > 1e-6:
            # reuse the same facing rules as movement
            self._attack_facing = self._facing_from_velocity() if self.velocity.length_squared() > 1e-6 else self._last_facing
            # better: face toward target
            x, y = v.x, v.y
            if abs(x) < 0.35 * abs(y) and y < 0:
                self._attack_facing = "U"
            elif abs(x) < 0.35 * abs(y) and y > 0:
                self._attack_facing = "D"
            elif abs(y) < 0.35 * abs(x) and x < 0:
                self._attack_facing = "L"
            elif abs(y) < 0.35 * abs(x) and x > 0:
                self._attack_facing = "R"
            elif x < 0 and y < 0:
                self._attack_facing = "LU"
            elif x < 0 and y > 0:
                self._attack_facing = "LD"
            elif x > 0 and y < 0:
                self._attack_facing = "RU"
            elif x > 0 and y > 0:
                self._attack_facing = "RD"
            else:
                self._attack_facing = "D"
        else:
            self._attack_facing = self._last_facing or "D"

        # compute anim length from frames
        attack_frames = get_unit_attack_frames("Axeman", int(self.size), tuple(self.player_color[:3]), facing=self._attack_facing)
        frame_time = self._ATTACK_FRAME_TIME
        anim_len = (len(attack_frames) * frame_time) if attack_frames else 0.35
        self._attacking_until = max(self._attacking_until, context.current_time + anim_len)

        # do normal damage + aggro logic
        damage = max(0, self.attack_damage - target.armor)
        hp_before = float(getattr(target, "hp", 0))
        target.hp -= damage
        # Progression XP: 1 per damaging hit, bonus on kill/destroy.
        progression.award_combat_xp(self, target, damage=damage, target_hp_before=hp_before)
        self.last_attack_time = context.current_time
        print(f"{self.__class__.__name__} at {self.pos} attacked {target.__class__.__name__} at {target.pos}, dealing {damage} damage")

        if isinstance(target, (Axeman, Archer, Knight, Bear, Strzyga, Priestess, Shaman, Swordsman, Spearman)) and target.hp > 0:
            target.update_attackers(self, context.current_time)
            if not target.target or getattr(target, "autonomous_target", False):
                closest_attacker = target.get_closest_attacker()
                if closest_attacker:
                    target.target = closest_attacker
                    target.autonomous_target = True
                    target.path = []
                    target.path_index = 0
# Knight class
    def repair_building(self):
        """If this axeman has a repair_target and is in range, repair it.

        Repair speed scales linearly with number of axemen because each axeman applies its own HP/s.
        Wood cost scales with HP repaired: full repair (0->max_hp) costs building.wood_cost wood.
        """
        if self.repair_target is None:
            return

        b = self.repair_target
        if b not in context.all_units or getattr(b, 'hp', 0) <= 0:
            self.repair_target = None
            return

        if getattr(b, 'player_id', None) != self.player_id:
            self.repair_target = None
            return

        max_hp = float(getattr(b, 'max_hp', getattr(b, 'hp', 0) or 0))
        if max_hp <= 0:
            self.repair_target = None
            return

        cur_hp = float(getattr(b, 'hp', 0))
        if cur_hp >= max_hp - 1e-6:
            # finished
            self.repair_target = None
            self.target = None
            return

        # Must be close enough to repair (near building edge).
        from constants import REPAIR_SPEED_MULT, REPAIR_REACH_PADDING
        reach = (float(getattr(b, 'size', 0)) / 2.0) + (float(getattr(self, 'size', 0)) / 2.0) + float(REPAIR_REACH_PADDING)
        if self.pos.distance_to(b.pos) > reach:
            return

        now = float(context.current_time)
        last = float(self._repair_last_time or now)
        dt = max(0.0, now - last)
        self._repair_last_time = now
        if dt <= 0.0:
            return

        # Construction HP/s for this building (same formula as main.py uses during construction).
        prod_time = float(getattr(b, 'production_time', 1.0) or 1.0)
        build_rate = (max(1.0, max_hp) - 1.0) / max(0.001, prod_time)
        repair_rate = build_rate * float(REPAIR_SPEED_MULT)
        desired_hp = repair_rate * dt
        if desired_hp <= 0.0:
            return

        # Wood cost per HP: full repair costs wood_cost.
        wood_cost = float(getattr(b, 'wood_cost', 0) or 0)
        cost_per_hp = (wood_cost / max_hp) if wood_cost > 0 else 0.0

        # Pay wood from the owning player, proportional to HP repaired.
        player = next((p for p in context.players if getattr(p, 'player_id', None) == self.player_id), None)
        if player is None:
            return

        if cost_per_hp > 0.0:
            affordable_hp = float(getattr(player, 'wood', 0)) / cost_per_hp if getattr(player, 'wood', 0) > 0 else 0.0
            hp_gain = min(desired_hp, affordable_hp, max_hp - cur_hp)
            if hp_gain <= 0.0:
                # Can't afford any repair right now.
                return
            player.wood = max(0.0, float(getattr(player, 'wood', 0)) - hp_gain * cost_per_hp)
        else:
            hp_gain = min(desired_hp, max_hp - cur_hp)

        # Apply repair
        b.hp = min(max_hp, float(getattr(b, 'hp', 0)) + hp_gain)

        # Trigger the same attack animation while repairing (swinging).
        v = (b.pos - self.pos)
        if v.length_squared() > 1e-6:
            x, y = v.x, v.y
            if abs(x) < 0.35 * abs(y) and y < 0:
                self._attack_facing = 'U'
            elif abs(x) < 0.35 * abs(y) and y > 0:
                self._attack_facing = 'D'
            elif abs(y) < 0.35 * abs(x) and x < 0:
                self._attack_facing = 'L'
            elif abs(y) < 0.35 * abs(x) and x > 0:
                self._attack_facing = 'R'
            elif x < 0 and y < 0:
                self._attack_facing = 'LU'
            elif x < 0 and y > 0:
                self._attack_facing = 'LD'
            elif x > 0 and y < 0:
                self._attack_facing = 'RU'
            elif x > 0 and y > 0:
                self._attack_facing = 'RD'
            else:
                self._attack_facing = 'D'
        else:
            self._attack_facing = self._last_facing or 'D'

        attack_frames = get_unit_attack_frames('Axeman', int(self.size), tuple(self.player_color[:3]), facing=self._attack_facing)
        frame_time = self._ATTACK_FRAME_TIME
        anim_len = (len(attack_frames) * frame_time) if attack_frames else 0.35
        self._attacking_until = max(self._attacking_until, now + anim_len)

        if float(getattr(b, 'hp', 0)) >= max_hp - 1e-6:
            # done
            self.repair_target = None
            self.target = None



class Knight(Unit):
    milk_cost = 500
    wood_cost = 400

    _WALK_FRAME_TIME = 0.5
    _ATTACK_FRAME_TIME = 0.4  # attack speed; change if you want faster/slower
    _IDLE_SPEED_EPS2 = 0.05

    def __init__(self, x, y, player_id, player_color):
        super().__init__(x, y, size=UNIT_SIZE, speed=2.5, color=BLUE, player_id=player_id, player_color=player_color)
        self.attack_damage = 17
        self.attack_range = 20
        self.attack_cooldown = 1.0
        self.armor = 5

        # attack animation state (same concept as Axeman/Archer)
        self._attacking_until = 0.0
        self._attack_facing = "D"
        self._last_facing = "D"

    def _facing_from_velocity(self) -> str:
        v = self.velocity
        if v.length_squared() < self._IDLE_SPEED_EPS2:
            return "M"

        x, y = v.x, v.y
        if abs(x) < 0.35 and y < 0: return "U"
        if abs(x) < 0.35 and y > 0: return "D"
        if abs(y) < 0.35 and x < 0: return "L"
        if abs(y) < 0.35 and x > 0: return "R"
        if x < 0 and y < 0: return "LU"
        if x < 0 and y > 0: return "LD"
        if x > 0 and y < 0: return "RU"
        if x > 0 and y > 0: return "RD"
        return "D"

    def attack(self, target, current_time):
        # same gating as Unit.attack, but with attack animation window like Axeman
        if not isinstance(target, Unit) or isinstance(target, Tree) or target.hp <= 0 or target not in context.all_units:
            return

        distance = (self.pos - target.pos).length()
        max_range = self.attack_range + self.size / 2 + target.size / 2
        if distance > max_range:
            return

        if context.current_time - self.last_attack_time < self.attack_cooldown:
            return

        # face toward target
        v = (target.pos - self.pos)
        if v.length_squared() > 1e-6:
            x, y = v.x, v.y
            if abs(x) < 0.35 * abs(y) and y < 0: self._attack_facing = "U"
            elif abs(x) < 0.35 * abs(y) and y > 0: self._attack_facing = "D"
            elif abs(y) < 0.35 * abs(x) and x < 0: self._attack_facing = "L"
            elif abs(y) < 0.35 * abs(x) and x > 0: self._attack_facing = "R"
            elif x < 0 and y < 0: self._attack_facing = "LU"
            elif x < 0 and y > 0: self._attack_facing = "LD"
            elif x > 0 and y < 0: self._attack_facing = "RU"
            elif x > 0 and y > 0: self._attack_facing = "RD"
            else: self._attack_facing = "D"
        else:
            self._attack_facing = self._last_facing or "D"

        # start attack animation window
        attack_frames = get_unit_attack_frames("Knight", int(self.size), tuple(self.player_color[:3]), facing=self._attack_facing)
        anim_len = (len(attack_frames) * self._ATTACK_FRAME_TIME) if attack_frames else 0.35
        self._attacking_until = max(self._attacking_until, context.current_time + anim_len)

        # deal damage (melee = immediate)
        damage = max(0, self.attack_damage - target.armor)
        hp_before = float(getattr(target, "hp", 0))
        target.hp -= damage
        # Progression XP: 1 per damaging hit, bonus on kill/destroy.
        progression.award_combat_xp(self, target, damage=damage, target_hp_before=hp_before)
        self.last_attack_time = context.current_time
        print(f"{self.__class__.__name__} at {self.pos} attacked {target.__class__.__name__} at {target.pos}, dealing {damage} damage")

        # Notify target of attack for defensive behavior (same as Unit.attack)
        if isinstance(target, (Axeman, Archer, Knight, Bear, Strzyga, Priestess, Shaman, Swordsman, Spearman)) and target.hp > 0:
            target.update_attackers(self, context.current_time)
            if not target.target or getattr(target, "autonomous_target", False):
                closest_attacker = target.get_closest_attacker()
                if closest_attacker:
                    target.target = closest_attacker
                    target.autonomous_target = True
                    target.path = []
                    target.path_index = 0

    def draw(self, screen, camera_x, camera_y):
        if (self.pos.x < camera_x - self.size / 2 or self.pos.x > camera_x + VIEW_WIDTH + self.size / 2 or
            self.pos.y < camera_y - self.size / 2 or self.pos.y > camera_y + VIEW_HEIGHT + self.size / 2):
            return

        now = context.current_time

        if now < self._attacking_until:
            facing = self._attack_facing or "D"
            frames = get_unit_attack_frames("Knight", int(self.size), tuple(self.player_color[:3]), facing=facing)
            frame_time = self._ATTACK_FRAME_TIME
        else:
            facing = self._facing_from_velocity()
            if facing != "M":
                self._last_facing = facing
            frames = get_unit_walk_frames("Knight", int(self.size), tuple(self.player_color[:3]), facing=facing)
            frame_time = self._WALK_FRAME_TIME

        if not frames:
            return super().draw(screen, camera_x, camera_y)

        idx = int(now / frame_time) % len(frames)
        image = frames[idx]

        x = self.pos.x - camera_x + VIEW_MARGIN_LEFT
        y = self.pos.y - camera_y + VIEW_MARGIN_TOP

        image_surface = image.copy()
        image_surface.set_alpha(self.alpha)
        rect = image_surface.get_rect(center=(int(x), int(y)))
        # Per-unit manual offset for ATTACK animations (auto-scaled by SCALE)
        if now < self._attacking_until:
            dx, dy = _scaled_attack_offset("Knight", facing)
            rect.move_ip(dx, dy)

        _blit_sprite_with_border(screen, image_surface, rect)

        if self.selected:
            pygame.draw.rect(screen, self.player_color, (x - self.size / 2, y - self.size / 2, self.size, self.size), 1)
        if self.should_highlight(context.current_time):
            pygame.draw.rect(screen, WHITE, (x - self.size / 2, y - self.size / 2, self.size, self.size), 1)

        # HP bar (only when not fully healed)
        self.draw_health_bar(screen, x, y)

class Swordsman(Unit):
    milk_cost = 500
    wood_cost = 400

    _WALK_FRAME_TIME = 0.5
    _ATTACK_FRAME_TIME = 0.4  # attack speed; change if you want faster/slower
    _IDLE_SPEED_EPS2 = 0.05

    def __init__(self, x, y, player_id, player_color):
        super().__init__(x, y, size=UNIT_SIZE, speed=2.5, color=BLUE, player_id=player_id, player_color=player_color)
        self.attack_damage = 17
        self.attack_range = 20
        self.attack_cooldown = 1.0
        self.armor = 5

        # attack animation state (same concept as Axeman/Archer)
        self._attacking_until = 0.0
        self._attack_facing = "D"
        self._last_facing = "D"

    def _facing_from_velocity(self) -> str:
        v = self.velocity
        if v.length_squared() < self._IDLE_SPEED_EPS2:
            return "M"

        x, y = v.x, v.y
        if abs(x) < 0.35 and y < 0: return "U"
        if abs(x) < 0.35 and y > 0: return "D"
        if abs(y) < 0.35 and x < 0: return "L"
        if abs(y) < 0.35 and x > 0: return "R"
        if x < 0 and y < 0: return "LU"
        if x < 0 and y > 0: return "LD"
        if x > 0 and y < 0: return "RU"
        if x > 0 and y > 0: return "RD"
        return "D"

    def attack(self, target, current_time):
        # same gating as Unit.attack, but with attack animation window like Axeman
        if not isinstance(target, Unit) or isinstance(target, Tree) or target.hp <= 0 or target not in context.all_units:
            return

        distance = (self.pos - target.pos).length()
        max_range = self.attack_range + self.size / 2 + target.size / 2
        if distance > max_range:
            return

        if context.current_time - self.last_attack_time < self.attack_cooldown:
            return

        # face toward target
        v = (target.pos - self.pos)
        if v.length_squared() > 1e-6:
            x, y = v.x, v.y
            if abs(x) < 0.35 * abs(y) and y < 0: self._attack_facing = "U"
            elif abs(x) < 0.35 * abs(y) and y > 0: self._attack_facing = "D"
            elif abs(y) < 0.35 * abs(x) and x < 0: self._attack_facing = "L"
            elif abs(y) < 0.35 * abs(x) and x > 0: self._attack_facing = "R"
            elif x < 0 and y < 0: self._attack_facing = "LU"
            elif x < 0 and y > 0: self._attack_facing = "LD"
            elif x > 0 and y < 0: self._attack_facing = "RU"
            elif x > 0 and y > 0: self._attack_facing = "RD"
            else: self._attack_facing = "D"
        else:
            self._attack_facing = self._last_facing or "D"

        # start attack animation window
        attack_frames = get_unit_attack_frames("Swordsman", int(self.size), tuple(self.player_color[:3]), facing=self._attack_facing)
        anim_len = (len(attack_frames) * self._ATTACK_FRAME_TIME) if attack_frames else 0.35
        self._attacking_until = max(self._attacking_until, context.current_time + anim_len)

        # deal damage (melee = immediate)
        damage = max(0, self.attack_damage - target.armor)
        hp_before = float(getattr(target, "hp", 0))
        target.hp -= damage
        # Progression XP: 1 per damaging hit, bonus on kill/destroy.
        progression.award_combat_xp(self, target, damage=damage, target_hp_before=hp_before)
        self.last_attack_time = context.current_time
        print(f"{self.__class__.__name__} at {self.pos} attacked {target.__class__.__name__} at {target.pos}, dealing {damage} damage")

        # Notify target of attack for defensive behavior (same as Unit.attack)
        if isinstance(target, (Axeman, Archer, Knight, Bear, Strzyga, Priestess, Shaman, Swordsman, Spearman)) and target.hp > 0:
            target.update_attackers(self, context.current_time)
            if not target.target or getattr(target, "autonomous_target", False):
                closest_attacker = target.get_closest_attacker()
                if closest_attacker:
                    target.target = closest_attacker
                    target.autonomous_target = True
                    target.path = []
                    target.path_index = 0

    def draw(self, screen, camera_x, camera_y):
        if (self.pos.x < camera_x - self.size / 2 or self.pos.x > camera_x + VIEW_WIDTH + self.size / 2 or
            self.pos.y < camera_y - self.size / 2 or self.pos.y > camera_y + VIEW_HEIGHT + self.size / 2):
            return

        now = context.current_time

        if now < self._attacking_until:
            facing = self._attack_facing or "D"
            frames = get_unit_attack_frames("Swordsman", int(self.size), tuple(self.player_color[:3]), facing=facing)
            frame_time = self._ATTACK_FRAME_TIME
        else:
            facing = self._facing_from_velocity()
            if facing != "M":
                self._last_facing = facing
            frames = get_unit_walk_frames("Swordsman", int(self.size), tuple(self.player_color[:3]), facing=facing)
            frame_time = self._WALK_FRAME_TIME

        if not frames:
            return super().draw(screen, camera_x, camera_y)

        idx = int(now / frame_time) % len(frames)
        image = frames[idx]

        x = self.pos.x - camera_x + VIEW_MARGIN_LEFT
        y = self.pos.y - camera_y + VIEW_MARGIN_TOP

        image_surface = image.copy()
        image_surface.set_alpha(self.alpha)
        rect = image_surface.get_rect(center=(int(x), int(y)))
        # Per-unit manual offset for ATTACK animations (auto-scaled by SCALE)
        if now < self._attacking_until:
            dx, dy = _scaled_attack_offset("Swordsman", facing)
            rect.move_ip(dx, dy)

        _blit_sprite_with_border(screen, image_surface, rect)

        if self.selected:
            pygame.draw.rect(screen, self.player_color, (x - self.size / 2, y - self.size / 2, self.size, self.size), 1)
        if self.should_highlight(context.current_time):
            pygame.draw.rect(screen, WHITE, (x - self.size / 2, y - self.size / 2, self.size, self.size), 1)

        # HP bar (only when not fully healed)
        self.draw_health_bar(screen, x, y)

# Bear class (melee unit, like Knight)
class Bear(Unit):
    # Temporary costs: buildable from Barn
    milk_cost = 10
    wood_cost = 10

    _WALK_FRAME_TIME = 0.5
    _ATTACK_FRAME_TIME = 0.4
    _IDLE_SPEED_EPS2 = 0.05

    def __init__(self, x, y, player_id, player_color):
        super().__init__(x, y, size=UNIT_SIZE, speed=2.5, color=BLUE, player_id=player_id, player_color=player_color)
        # Stats like Knight (for now)
        self.attack_damage = 17
        self.attack_range = 20
        self.attack_cooldown = 1.0
        self.armor = 5

        self._attacking_until = 0.0
        self._attack_facing = "D"
        self._last_facing = "D"

    def _facing_from_velocity(self) -> str:
        v = self.velocity
        if v.length_squared() < self._IDLE_SPEED_EPS2:
            return "M"

        x, y = v.x, v.y
        if abs(x) < 0.35 and y < 0: return "U"
        if abs(x) < 0.35 and y > 0: return "D"
        if abs(y) < 0.35 and x < 0: return "L"
        if abs(y) < 0.35 and x > 0: return "R"
        if x < 0 and y < 0: return "LU"
        if x < 0 and y > 0: return "LD"
        if x > 0 and y < 0: return "RU"
        if x > 0 and y > 0: return "RD"
        return "D"

    def attack(self, target, current_time):
        if not isinstance(target, Unit) or isinstance(target, Tree) or target.hp <= 0 or target not in context.all_units:
            return

        distance = (self.pos - target.pos).length()
        max_range = self.attack_range + self.size / 2 + target.size / 2
        if distance > max_range:
            return

        if context.current_time - self.last_attack_time < self.attack_cooldown:
            return

        v = (target.pos - self.pos)
        if v.length_squared() > 1e-6:
            x, y = v.x, v.y
            if abs(x) < 0.35 * abs(y) and y < 0: self._attack_facing = "U"
            elif abs(x) < 0.35 * abs(y) and y > 0: self._attack_facing = "D"
            elif abs(y) < 0.35 * abs(x) and x < 0: self._attack_facing = "L"
            elif abs(y) < 0.35 * abs(x) and x > 0: self._attack_facing = "R"
            elif x < 0 and y < 0: self._attack_facing = "LU"
            elif x < 0 and y > 0: self._attack_facing = "LD"
            elif x > 0 and y < 0: self._attack_facing = "RU"
            elif x > 0 and y > 0: self._attack_facing = "RD"
            else: self._attack_facing = "D"
        else:
            self._attack_facing = self._last_facing or "D"

        attack_frames = get_unit_attack_frames("Bear", int(self.size), tuple(self.player_color[:3]), facing=self._attack_facing)
        anim_len = (len(attack_frames) * self._ATTACK_FRAME_TIME) if attack_frames else 0.35
        self._attacking_until = max(self._attacking_until, context.current_time + anim_len)

        damage = max(0, self.attack_damage - target.armor)
        hp_before = float(getattr(target, "hp", 0))
        target.hp -= damage
        progression.award_combat_xp(self, target, damage=damage, target_hp_before=hp_before)
        self.last_attack_time = context.current_time
        print(f"{self.__class__.__name__} at {self.pos} attacked {target.__class__.__name__} at {target.pos}, dealing {damage} damage")

        if isinstance(target, (Axeman, Archer, Knight, Bear, Strzyga, Priestess, Shaman, Swordsman, Spearman)) and target.hp > 0:
            target.update_attackers(self, context.current_time)
            if not target.target or getattr(target, "autonomous_target", False):
                closest_attacker = target.get_closest_attacker()
                if closest_attacker:
                    target.target = closest_attacker
                    target.autonomous_target = True
                    target.path = []
                    target.path_index = 0

    def draw(self, screen, camera_x, camera_y):
        if (self.pos.x < camera_x - self.size / 2 or self.pos.x > camera_x + VIEW_WIDTH + self.size / 2 or
            self.pos.y < camera_y - self.size / 2 or self.pos.y > camera_y + VIEW_HEIGHT + self.size / 2):
            return

        now = context.current_time

        if now < self._attacking_until:
            facing = self._attack_facing or "D"
            frames = get_unit_attack_frames("Bear", int(self.size), tuple(self.player_color[:3]), facing=facing)
            frame_time = self._ATTACK_FRAME_TIME
        else:
            facing = self._facing_from_velocity()
            if facing != "M":
                self._last_facing = facing
            frames = get_unit_walk_frames("Bear", int(self.size), tuple(self.player_color[:3]), facing=facing)
            frame_time = self._WALK_FRAME_TIME

        if not frames:
            return super().draw(screen, camera_x, camera_y)

        idx = int(now / frame_time) % len(frames)
        image = frames[idx]

        x = self.pos.x - camera_x + VIEW_MARGIN_LEFT
        y = self.pos.y - camera_y + VIEW_MARGIN_TOP

        image_surface = image.copy()
        image_surface.set_alpha(self.alpha)
        rect = image_surface.get_rect(center=(int(x), int(y)))
        if now < self._attacking_until:
            dx, dy = _scaled_attack_offset("Bear", facing)
            rect.move_ip(dx, dy)

        _blit_sprite_with_border(screen, image_surface, rect)

        if self.selected:
            pygame.draw.rect(screen, self.player_color, (x - self.size / 2, y - self.size / 2, self.size, self.size), 1)
        if self.should_highlight(context.current_time):
            pygame.draw.rect(screen, WHITE, (x - self.size / 2, y - self.size / 2, self.size, self.size), 1)

        self.draw_health_bar(screen, x, y)


# Strzyga class (melee unit, like Knight)
class Strzyga(Unit):
    milk_cost = 10
    wood_cost = 10

    _WALK_FRAME_TIME = 0.5
    _ATTACK_FRAME_TIME = 0.4
    _IDLE_SPEED_EPS2 = 0.05

    def __init__(self, x, y, player_id, player_color):
        super().__init__(x, y, size=UNIT_SIZE, speed=2.5, color=BLUE, player_id=player_id, player_color=player_color)
        # Stats like Knight (for now)
        self.attack_damage = 17
        self.attack_range = 20
        self.attack_cooldown = 1.0
        self.armor = 5

        self._attacking_until = 0.0
        self._attack_facing = "D"
        self._last_facing = "D"

    def _facing_from_velocity(self) -> str:
        v = self.velocity
        if v.length_squared() < self._IDLE_SPEED_EPS2:
            return "M"

        x, y = v.x, v.y
        if abs(x) < 0.35 and y < 0: return "U"
        if abs(x) < 0.35 and y > 0: return "D"
        if abs(y) < 0.35 and x < 0: return "L"
        if abs(y) < 0.35 and x > 0: return "R"
        if x < 0 and y < 0: return "LU"
        if x < 0 and y > 0: return "LD"
        if x > 0 and y < 0: return "RU"
        if x > 0 and y > 0: return "RD"
        return "D"

    def attack(self, target, current_time):
        if not isinstance(target, Unit) or isinstance(target, Tree) or target.hp <= 0 or target not in context.all_units:
            return

        distance = (self.pos - target.pos).length()
        max_range = self.attack_range + self.size / 2 + target.size / 2
        if distance > max_range:
            return

        if context.current_time - self.last_attack_time < self.attack_cooldown:
            return

        v = (target.pos - self.pos)
        if v.length_squared() > 1e-6:
            x, y = v.x, v.y
            if abs(x) < 0.35 * abs(y) and y < 0: self._attack_facing = "U"
            elif abs(x) < 0.35 * abs(y) and y > 0: self._attack_facing = "D"
            elif abs(y) < 0.35 * abs(x) and x < 0: self._attack_facing = "L"
            elif abs(y) < 0.35 * abs(x) and x > 0: self._attack_facing = "R"
            elif x < 0 and y < 0: self._attack_facing = "LU"
            elif x < 0 and y > 0: self._attack_facing = "LD"
            elif x > 0 and y < 0: self._attack_facing = "RU"
            elif x > 0 and y > 0: self._attack_facing = "RD"
            else: self._attack_facing = "D"
        else:
            self._attack_facing = self._last_facing or "D"

        attack_frames = get_unit_attack_frames("Strzyga", int(self.size), tuple(self.player_color[:3]), facing=self._attack_facing)
        anim_len = (len(attack_frames) * self._ATTACK_FRAME_TIME) if attack_frames else 0.35
        self._attacking_until = max(self._attacking_until, context.current_time + anim_len)

        damage = max(0, self.attack_damage - target.armor)
        hp_before = float(getattr(target, "hp", 0))
        target.hp -= damage
        progression.award_combat_xp(self, target, damage=damage, target_hp_before=hp_before)
        self.last_attack_time = context.current_time
        print(f"{self.__class__.__name__} at {self.pos} attacked {target.__class__.__name__} at {target.pos}, dealing {damage} damage")

        if isinstance(target, (Axeman, Archer, Knight, Bear, Strzyga, Priestess, Shaman, Swordsman, Spearman)) and target.hp > 0:
            target.update_attackers(self, context.current_time)
            if not target.target or getattr(target, "autonomous_target", False):
                closest_attacker = target.get_closest_attacker()
                if closest_attacker:
                    target.target = closest_attacker
                    target.autonomous_target = True
                    target.path = []
                    target.path_index = 0

    def draw(self, screen, camera_x, camera_y):
        if (self.pos.x < camera_x - self.size / 2 or self.pos.x > camera_x + VIEW_WIDTH + self.size / 2 or
            self.pos.y < camera_y - self.size / 2 or self.pos.y > camera_y + VIEW_HEIGHT + self.size / 2):
            return

        now = context.current_time

        if now < self._attacking_until:
            facing = self._attack_facing or "D"
            frames = get_unit_attack_frames("Strzyga", int(self.size), tuple(self.player_color[:3]), facing=facing)
            frame_time = self._ATTACK_FRAME_TIME
        else:
            facing = self._facing_from_velocity()
            if facing != "M":
                self._last_facing = facing
            frames = get_unit_walk_frames("Strzyga", int(self.size), tuple(self.player_color[:3]), facing=facing)
            frame_time = self._WALK_FRAME_TIME

        if not frames:
            return super().draw(screen, camera_x, camera_y)

        idx = int(now / frame_time) % len(frames)
        image = frames[idx]

        x = self.pos.x - camera_x + VIEW_MARGIN_LEFT
        y = self.pos.y - camera_y + VIEW_MARGIN_TOP

        image_surface = image.copy()
        image_surface.set_alpha(self.alpha)
        rect = image_surface.get_rect(center=(int(x), int(y)))
        if now < self._attacking_until:
            dx, dy = _scaled_attack_offset("Strzyga", facing)
            rect.move_ip(dx, dy)

        _blit_sprite_with_border(screen, image_surface, rect)

        if self.selected:
            pygame.draw.rect(screen, self.player_color, (x - self.size / 2, y - self.size / 2, self.size, self.size), 1)
        if self.should_highlight(context.current_time):
            pygame.draw.rect(screen, WHITE, (x - self.size / 2, y - self.size / 2, self.size, self.size), 1)

        self.draw_health_bar(screen, x, y)


# Archer class
# Priestess class (melee unit, like Knight) — produced from ShamansHut
class Priestess(Unit):
    # Temporary costs: buildable from ShamansHut
    milk_cost = 10
    wood_cost = 10

    _WALK_FRAME_TIME = 0.5
    _ATTACK_FRAME_TIME = 0.4
    _IDLE_SPEED_EPS2 = 0.05

    def __init__(self, x, y, player_id, player_color):
        super().__init__(x, y, size=UNIT_SIZE, speed=2.5, color=BLUE, player_id=player_id, player_color=player_color)
        # Stats like Knight (for now)
        self.attack_damage = 17
        self.attack_range = 20
        self.attack_cooldown = 1.0
        self.armor = 5

        self._attacking_until = 0.0
        self._attack_facing = "D"
        self._last_facing = "D"

    def _facing_from_velocity(self) -> str:
        v = self.velocity
        if v.length_squared() < self._IDLE_SPEED_EPS2:
            return "M"
        x, y = v.x, v.y
        if abs(x) < 0.35 and y < 0: return "U"
        if abs(x) < 0.35 and y > 0: return "D"
        if abs(y) < 0.35 and x < 0: return "L"
        if abs(y) < 0.35 and x > 0: return "R"
        if x < 0 and y < 0: return "LU"
        if x < 0 and y > 0: return "LD"
        if x > 0 and y < 0: return "RU"
        if x > 0 and y > 0: return "RD"
        return "D"

    def attack(self, target, current_time):
        if not isinstance(target, Unit) or isinstance(target, Tree) or target.hp <= 0 or target not in context.all_units:
            return

        distance = (self.pos - target.pos).length()
        max_range = self.attack_range + self.size / 2 + target.size / 2
        if distance > max_range:
            return

        if context.current_time - self.last_attack_time < self.attack_cooldown:
            return

        v = (target.pos - self.pos)
        if v.length_squared() > 1e-6:
            x, y = v.x, v.y
            if abs(x) < 0.35 * abs(y) and y < 0: self._attack_facing = "U"
            elif abs(x) < 0.35 * abs(y) and y > 0: self._attack_facing = "D"
            elif abs(y) < 0.35 * abs(x) and x < 0: self._attack_facing = "L"
            elif abs(y) < 0.35 * abs(x) and x > 0: self._attack_facing = "R"
            elif x < 0 and y < 0: self._attack_facing = "LU"
            elif x < 0 and y > 0: self._attack_facing = "LD"
            elif x > 0 and y < 0: self._attack_facing = "RU"
            elif x > 0 and y > 0: self._attack_facing = "RD"
            else: self._attack_facing = "D"
        else:
            self._attack_facing = self._last_facing or "D"

        attack_frames = get_unit_attack_frames("Priestess", int(self.size), tuple(self.player_color[:3]), facing=self._attack_facing)
        anim_len = (len(attack_frames) * self._ATTACK_FRAME_TIME) if attack_frames else 0.35
        self._attacking_until = max(self._attacking_until, context.current_time + anim_len)

        damage = max(0, self.attack_damage - target.armor)
        hp_before = float(getattr(target, "hp", 0))
        target.hp -= damage
        progression.award_combat_xp(self, target, damage=damage, target_hp_before=hp_before)
        self.last_attack_time = context.current_time

        if isinstance(target, (Axeman, Archer, Knight, Bear, Strzyga, Priestess, Shaman, Swordsman, Spearman)) and target.hp > 0:
            target.update_attackers(self, context.current_time)
            if not target.target or getattr(target, "autonomous_target", False):
                closest_attacker = target.get_closest_attacker()
                if closest_attacker:
                    target.target = closest_attacker
                    target.autonomous_target = True
                    target.path = []
                    target.path_index = 0

    def draw(self, screen, camera_x, camera_y):
        # Match Knight/Bear animation logic: walk vs attack frames
        if (self.pos.x < camera_x - self.size / 2 or self.pos.x > camera_x + VIEW_WIDTH + self.size / 2 or
            self.pos.y < camera_y - self.size / 2 or self.pos.y > camera_y + VIEW_HEIGHT + self.size / 2):
            return

        cls_name = self.__class__.__name__

        x = self.pos.x - camera_x + VIEW_MARGIN_LEFT
        y = self.pos.y - camera_y + VIEW_MARGIN_TOP

        now = context.current_time

        # Determine facing
        facing = self._facing_from_velocity()
        if facing != "M":
            self._last_facing = facing

        # choose frames
        if now < self._attacking_until:
            facing_use = self._attack_facing or self._last_facing or "D"
            frames = get_unit_attack_frames("Priestess", int(self.size), tuple(self.player_color[:3]), facing=facing_use)
            frame_time = self._ATTACK_FRAME_TIME
            dx, dy = _scaled_attack_offset(cls_name, facing_use)
        else:
            facing_use = (self._last_facing or "D")
            frames = get_unit_walk_frames("Priestess", int(self.size), tuple(self.player_color[:3]), facing=facing_use)
            frame_time = self._WALK_FRAME_TIME
            dx, dy = (0, 0)

        if not frames:
            image = get_team_sprite(cls_name, int(self.size), tuple(self.player_color[:3]))
            if image:
                image_surface = image.copy()
                image_surface.set_alpha(self.alpha)
                rect = image_surface.get_rect(center=(int(x), int(y)))
                _blit_sprite_with_border(screen, image_surface, rect)
            return

        idx = int((now / frame_time)) % len(frames)
        image_surface = frames[idx].copy()
        image_surface.set_alpha(self.alpha)
        rect = image_surface.get_rect(center=(int(x + dx), int(y + dy)))
        _blit_sprite_with_border(screen, image_surface, rect)

        if self.selected:
            pygame.draw.rect(screen, self.player_color, (x - self.size / 2, y - self.size / 2, self.size, self.size), 1)
        if self.should_highlight(context.current_time):
            pygame.draw.rect(screen, WHITE, (x - self.size / 2, y - self.size / 2, self.size, self.size), 1)
        self.draw_health_bar(screen, x, y)


# Shaman class (melee unit, like Knight) — produced from ShamansHut
class Shaman(Unit):
    milk_cost = 10
    wood_cost = 10

    _WALK_FRAME_TIME = 0.5
    _ATTACK_FRAME_TIME = 0.4
    _IDLE_SPEED_EPS2 = 0.05

    def __init__(self, x, y, player_id, player_color):
        super().__init__(x, y, size=UNIT_SIZE, speed=2.5, color=BLUE, player_id=player_id, player_color=player_color)
        # Stats like Knight (for now)
        self.attack_damage = 17
        self.attack_range = 20
        self.attack_cooldown = 1.0
        self.armor = 5

        self._attacking_until = 0.0
        self._attack_facing = "D"
        self._last_facing = "D"

    def _facing_from_velocity(self) -> str:
        v = self.velocity
        if v.length_squared() < self._IDLE_SPEED_EPS2:
            return "M"
        x, y = v.x, v.y
        if abs(x) < 0.35 and y < 0: return "U"
        if abs(x) < 0.35 and y > 0: return "D"
        if abs(y) < 0.35 and x < 0: return "L"
        if abs(y) < 0.35 and x > 0: return "R"
        if x < 0 and y < 0: return "LU"
        if x < 0 and y > 0: return "LD"
        if x > 0 and y < 0: return "RU"
        if x > 0 and y > 0: return "RD"
        return "D"

    def attack(self, target, current_time):
        if not isinstance(target, Unit) or isinstance(target, Tree) or target.hp <= 0 or target not in context.all_units:
            return

        distance = (self.pos - target.pos).length()
        max_range = self.attack_range + self.size / 2 + target.size / 2
        if distance > max_range:
            return

        if context.current_time - self.last_attack_time < self.attack_cooldown:
            return

        v = (target.pos - self.pos)
        if v.length_squared() > 1e-6:
            x, y = v.x, v.y
            if abs(x) < 0.35 * abs(y) and y < 0: self._attack_facing = "U"
            elif abs(x) < 0.35 * abs(y) and y > 0: self._attack_facing = "D"
            elif abs(y) < 0.35 * abs(x) and x < 0: self._attack_facing = "L"
            elif abs(y) < 0.35 * abs(x) and x > 0: self._attack_facing = "R"
            elif x < 0 and y < 0: self._attack_facing = "LU"
            elif x < 0 and y > 0: self._attack_facing = "LD"
            elif x > 0 and y < 0: self._attack_facing = "RU"
            elif x > 0 and y > 0: self._attack_facing = "RD"
            else: self._attack_facing = "D"
        else:
            self._attack_facing = self._last_facing or "D"

        attack_frames = get_unit_attack_frames("Shaman", int(self.size), tuple(self.player_color[:3]), facing=self._attack_facing)
        anim_len = (len(attack_frames) * self._ATTACK_FRAME_TIME) if attack_frames else 0.35
        self._attacking_until = max(self._attacking_until, context.current_time + anim_len)

        damage = max(0, self.attack_damage - target.armor)
        hp_before = float(getattr(target, "hp", 0))
        target.hp -= damage
        progression.award_combat_xp(self, target, damage=damage, target_hp_before=hp_before)
        self.last_attack_time = context.current_time

        if isinstance(target, (Axeman, Archer, Knight, Bear, Strzyga, Priestess, Shaman, Swordsman, Spearman)) and target.hp > 0:
            target.update_attackers(self, context.current_time)
            if not target.target or getattr(target, "autonomous_target", False):
                closest_attacker = target.get_closest_attacker()
                if closest_attacker:
                    target.target = closest_attacker
                    target.autonomous_target = True
                    target.path = []
                    target.path_index = 0

    def draw(self, screen, camera_x, camera_y):
        if (self.pos.x < camera_x - self.size / 2 or self.pos.x > camera_x + VIEW_WIDTH + self.size / 2 or
            self.pos.y < camera_y - self.size / 2 or self.pos.y > camera_y + VIEW_HEIGHT + self.size / 2):
            return

        cls_name = self.__class__.__name__

        x = self.pos.x - camera_x + VIEW_MARGIN_LEFT
        y = self.pos.y - camera_y + VIEW_MARGIN_TOP

        now = context.current_time

        facing = self._facing_from_velocity()
        if facing != "M":
            self._last_facing = facing

        if now < self._attacking_until:
            facing_use = self._attack_facing or self._last_facing or "D"
            frames = get_unit_attack_frames("Shaman", int(self.size), tuple(self.player_color[:3]), facing=facing_use)
            frame_time = self._ATTACK_FRAME_TIME
            dx, dy = _scaled_attack_offset(cls_name, facing_use)
        else:
            facing_use = (self._last_facing or "D")
            frames = get_unit_walk_frames("Shaman", int(self.size), tuple(self.player_color[:3]), facing=facing_use)
            frame_time = self._WALK_FRAME_TIME
            dx, dy = (0, 0)

        if not frames:
            image = get_team_sprite(cls_name, int(self.size), tuple(self.player_color[:3]))
            if image:
                image_surface = image.copy()
                image_surface.set_alpha(self.alpha)
                rect = image_surface.get_rect(center=(int(x), int(y)))
                _blit_sprite_with_border(screen, image_surface, rect)
            return

        idx = int((now / frame_time)) % len(frames)
        image_surface = frames[idx].copy()
        image_surface.set_alpha(self.alpha)
        rect = image_surface.get_rect(center=(int(x + dx), int(y + dy)))
        _blit_sprite_with_border(screen, image_surface, rect)

        if self.selected:
            pygame.draw.rect(screen, self.player_color, (x - self.size / 2, y - self.size / 2, self.size, self.size), 1)
        if self.should_highlight(context.current_time):
            pygame.draw.rect(screen, WHITE, (x - self.size / 2, y - self.size / 2, self.size, self.size), 1)
        self.draw_health_bar(screen, x, y)


class Archer(Unit):
    milk_cost = 400
    wood_cost = 200

    _WALK_FRAME_TIME = 0.12
    _ATTACK_FRAME_TIME = 0.10
    _IDLE_SPEED_EPS2 = 0.05

    # fire roughly mid-animation
    _ARROW_FIRE_FRACTION = 0.5

    def __init__(self, x, y, player_id, player_color):
        super().__init__(x, y, size=UNIT_SIZE, speed=2.3, color=YELLOW, player_id=player_id, player_color=player_color)
        self.attack_damage = 15

        # Range is defined in *tiles* (from center of tile to center of tile)
        self.attack_range_tiles = 5
        self.attack_range = self.attack_range_tiles * TILE_SIZE  # used by generic movement stop distance

        self.attack_cooldown = 1.5
        self.armor = 0

        # Attack animation state
        self._attacking_until = 0.0
        self._attack_facing = "D"

    def _tile_center(self, p: Vector2) -> Vector2:
        tx = int(p.x // TILE_SIZE)
        ty = int(p.y // TILE_SIZE)
        return Vector2(tx * TILE_SIZE + TILE_SIZE / 2, ty * TILE_SIZE + TILE_SIZE / 2)

    def _facing_from_vector(self, v: Vector2) -> str:
        if v.length_squared() < 1e-6:
            return "D"
        x, y = v.x, v.y
        if abs(x) < 0.35 * abs(y) and y < 0: return "U"
        if abs(x) < 0.35 * abs(y) and y > 0: return "D"
        if abs(y) < 0.35 * abs(x) and x < 0: return "L"
        if abs(y) < 0.35 * abs(x) and x > 0: return "R"
        if x < 0 and y < 0: return "LU"
        if x < 0 and y > 0: return "LD"
        if x > 0 and y < 0: return "RU"
        if x > 0 and y > 0: return "RD"
        return "D"

    def _facing_from_velocity(self) -> str:
        v = self.velocity
        if v.length_squared() < self._IDLE_SPEED_EPS2:
            return "M"
        return self._facing_from_vector(v)

    def attack(self, target, current_time):
        """Override: tile-based range, attack animation, and arrow effect projectile."""
        # validate
        if not isinstance(target, Unit) or isinstance(target, Tree) or target.hp <= 0 or target not in context.all_units:
            return

        # tile-center distance check
        a = self._tile_center(self.pos)
        b = self._tile_center(target.pos)
        distance = (a - b).length()
        max_range = self.attack_range_tiles * TILE_SIZE

        if distance > max_range:
            return

        if context.current_time - self.last_attack_time < self.attack_cooldown:
            return

        # Lock facing toward the target for the attack animation
        to_target = (b - a)
        self._attack_facing = self._facing_from_vector(to_target)

        # Start attack animation window
        attack_frames = get_unit_attack_frames("Archer", int(self.size), tuple(self.player_color[:3]), facing=self._attack_facing)
        anim_len = (len(attack_frames) * self._ATTACK_FRAME_TIME) if attack_frames else 0.35
        self._attacking_until = max(self._attacking_until, context.current_time + anim_len)

        # The projectile travels to a fixed impact point (snapshot at fire time).
        # If the target moves away before arrival, it "dodges" and takes no damage.
        impact_point = Vector2(b)  # tile-center of target at the moment of firing

        def apply_damage():
            # target may already be dead/removed
            if target.hp <= 0 or target not in context.all_units:
                return

            # check hit against CURRENT target rect at impact time
            half = target.size / 2
            target_rect = pygame.Rect(
                target.pos.x - half,
                target.pos.y - half,
                target.size,
                target.size,
            )

            if not target_rect.collidepoint(impact_point.x, impact_point.y):
                # Miss: projectile doesn't collide with the moved sprite -> no damage
                # (optional debug)
                # print(f"Arrow missed {target.__class__.__name__}: impact={impact_point}, target_now={target.pos}")
                return

            hp_before = float(getattr(target, "hp", 0))
            target.hp -= damage
            # Progression XP: 1 per damaging hit, bonus on kill/destroy.
            progression.award_combat_xp(self, target, damage=damage, target_hp_before=hp_before)
            print(f"Arrow hit {target.__class__.__name__} at {target.pos}, dealing {damage} damage")

        # Spawn arrow effect roughly mid-animation
        try:
            from effects import ArrowEffect
            if not hasattr(context, "effects"):
                context.effects = []
            fire_time = context.current_time + anim_len * self._ARROW_FIRE_FRACTION
            context.effects.append(
                ArrowEffect(
                    start_pos=Vector2(self.pos),
                    end_pos=Vector2(impact_point),
                    facing=self._attack_facing,
                    size_px=int(TILE_SIZE),
                    spawn_time=fire_time,
                    on_hit=apply_damage,
                )
            )
        except Exception as e:
            # If effects system isn't available, fail gracefully (still deal damage).
            # Here we keep old behavior: apply immediately.
            print(f"ArrowEffect spawn failed: {e}")
            apply_damage()

        # Deal damage at fire time (game logic stays immediate)
        damage = max(0, self.attack_damage - target.armor)

        self.last_attack_time = context.current_time

        # Notify target for defensive behavior (same as Unit.attack)
        if isinstance(target, (Axeman, Archer, Knight, Bear, Strzyga, Priestess, Shaman, Swordsman, Spearman)) and target.hp > 0:
            target.update_attackers(self, context.current_time)
            if not target.target or getattr(target, "autonomous_target", False):
                closest_attacker = target.get_closest_attacker()
                if closest_attacker:
                    target.target = closest_attacker
                    target.autonomous_target = True
                    target.path = []
                    target.path_index = 0

    def draw(self, screen, camera_x, camera_y):
        if (self.pos.x < camera_x - self.size / 2 or self.pos.x > camera_x + VIEW_WIDTH + self.size / 2 or
                self.pos.y < camera_y - self.size / 2 or self.pos.y > camera_y + VIEW_HEIGHT + self.size / 2):
            return

        now = context.current_time

        # Choose facing + frames based on whether we're in attack animation window
        if now < self._attacking_until:
            facing = self._attack_facing or "D"
            frames = get_unit_attack_frames("Archer", int(self.size), tuple(self.player_color[:3]), facing=facing)
            frame_time = self._ATTACK_FRAME_TIME
        else:
            facing = self._facing_from_velocity()
            frames = get_unit_walk_frames("Archer", int(self.size), tuple(self.player_color[:3]), facing=facing)
            frame_time = self._WALK_FRAME_TIME

        if not frames:
            return super().draw(screen, camera_x, camera_y)

        idx = int(now / frame_time) % len(frames)
        image = frames[idx]

        x = self.pos.x - camera_x + VIEW_MARGIN_LEFT
        y = self.pos.y - camera_y + VIEW_MARGIN_TOP

        image_surface = image.copy()
        image_surface.set_alpha(self.alpha)
        rect = image_surface.get_rect(center=(int(x), int(y)))
        # Per-unit manual offset for ATTACK animations (auto-scaled by SCALE)
        if now < self._attacking_until:
            dx, dy = _scaled_attack_offset("Archer", facing)
            rect.move_ip(dx, dy)

        _blit_sprite_with_border(screen, image_surface, rect)

        if self.selected:
            pygame.draw.rect(screen, self.player_color, (x - self.size / 2, y - self.size / 2, self.size, self.size), 1)
        if self.should_highlight(context.current_time):
            pygame.draw.rect(screen, WHITE, (x - self.size / 2, y - self.size / 2, self.size, self.size), 1)

        # HP bar (only when not fully healed)
        self.draw_health_bar(screen, x, y)

class Spearman(Unit):
    milk_cost = 10
    wood_cost = 10

    _WALK_FRAME_TIME = 0.12
    _ATTACK_FRAME_TIME = 0.10
    _IDLE_SPEED_EPS2 = 0.05

    # fire roughly mid-animation
    _ARROW_FIRE_FRACTION = 0.5

    def __init__(self, x, y, player_id, player_color):
        super().__init__(x, y, size=UNIT_SIZE, speed=2.3, color=YELLOW, player_id=player_id, player_color=player_color)
        self.attack_damage = 15

        # Range is defined in *tiles* (from center of tile to center of tile)
        self.attack_range_tiles = 5
        self.attack_range = self.attack_range_tiles * TILE_SIZE  # used by generic movement stop distance

        self.attack_cooldown = 1.5
        self.armor = 0

        # Attack animation state
        self._attacking_until = 0.0
        self._attack_facing = "D"

    def _tile_center(self, p: Vector2) -> Vector2:
        tx = int(p.x // TILE_SIZE)
        ty = int(p.y // TILE_SIZE)
        return Vector2(tx * TILE_SIZE + TILE_SIZE / 2, ty * TILE_SIZE + TILE_SIZE / 2)

    def _facing_from_vector(self, v: Vector2) -> str:
        if v.length_squared() < 1e-6:
            return "D"
        x, y = v.x, v.y
        if abs(x) < 0.35 * abs(y) and y < 0: return "U"
        if abs(x) < 0.35 * abs(y) and y > 0: return "D"
        if abs(y) < 0.35 * abs(x) and x < 0: return "L"
        if abs(y) < 0.35 * abs(x) and x > 0: return "R"
        if x < 0 and y < 0: return "LU"
        if x < 0 and y > 0: return "LD"
        if x > 0 and y < 0: return "RU"
        if x > 0 and y > 0: return "RD"
        return "D"

    def _facing_from_velocity(self) -> str:
        v = self.velocity
        if v.length_squared() < self._IDLE_SPEED_EPS2:
            return "M"
        return self._facing_from_vector(v)

    def attack(self, target, current_time):
        """Override: tile-based range, attack animation, and arrow effect projectile."""
        # validate
        if not isinstance(target, Unit) or isinstance(target, Tree) or target.hp <= 0 or target not in context.all_units:
            return

        # tile-center distance check
        a = self._tile_center(self.pos)
        b = self._tile_center(target.pos)
        distance = (a - b).length()
        max_range = self.attack_range_tiles * TILE_SIZE

        if distance > max_range:
            return

        if context.current_time - self.last_attack_time < self.attack_cooldown:
            return

        # Lock facing toward the target for the attack animation
        to_target = (b - a)
        self._attack_facing = self._facing_from_vector(to_target)

        # Start attack animation window
        attack_frames = get_unit_attack_frames("Spearman", int(self.size), tuple(self.player_color[:3]), facing=self._attack_facing)
        anim_len = (len(attack_frames) * self._ATTACK_FRAME_TIME) if attack_frames else 0.35
        self._attacking_until = max(self._attacking_until, context.current_time + anim_len)

        # The projectile travels to a fixed impact point (snapshot at fire time).
        # If the target moves away before arrival, it "dodges" and takes no damage.
        impact_point = Vector2(b)  # tile-center of target at the moment of firing

        def apply_damage():
            # target may already be dead/removed
            if target.hp <= 0 or target not in context.all_units:
                return

            # check hit against CURRENT target rect at impact time
            half = target.size / 2
            target_rect = pygame.Rect(
                target.pos.x - half,
                target.pos.y - half,
                target.size,
                target.size,
            )

            if not target_rect.collidepoint(impact_point.x, impact_point.y):
                # Miss: projectile doesn't collide with the moved sprite -> no damage
                # (optional debug)
                # print(f"Arrow missed {target.__class__.__name__}: impact={impact_point}, target_now={target.pos}")
                return

            hp_before = float(getattr(target, "hp", 0))
            target.hp -= damage
            # Progression XP: 1 per damaging hit, bonus on kill/destroy.
            progression.award_combat_xp(self, target, damage=damage, target_hp_before=hp_before)
            print(f"Arrow hit {target.__class__.__name__} at {target.pos}, dealing {damage} damage")

        # Spawn arrow effect roughly mid-animation
        try:
            from effects import ArrowEffect
            if not hasattr(context, "effects"):
                context.effects = []
            fire_time = context.current_time + anim_len * self._ARROW_FIRE_FRACTION
            context.effects.append(
                ArrowEffect(
                    start_pos=Vector2(self.pos),
                    end_pos=Vector2(impact_point),
                    facing=self._attack_facing,
                    size_px=int(TILE_SIZE),
                    spawn_time=fire_time,
                    on_hit=apply_damage,
                )
            )
        except Exception as e:
            # If effects system isn't available, fail gracefully (still deal damage).
            # Here we keep old behavior: apply immediately.
            print(f"ArrowEffect spawn failed: {e}")
            apply_damage()

        # Deal damage at fire time (game logic stays immediate)
        damage = max(0, self.attack_damage - target.armor)

        self.last_attack_time = context.current_time

        # Notify target for defensive behavior (same as Unit.attack)
        if isinstance(target, (Axeman, Archer, Knight, Bear, Strzyga, Priestess, Shaman, Swordsman, Spearman)) and target.hp > 0:
            target.update_attackers(self, context.current_time)
            if not target.target or getattr(target, "autonomous_target", False):
                closest_attacker = target.get_closest_attacker()
                if closest_attacker:
                    target.target = closest_attacker
                    target.autonomous_target = True
                    target.path = []
                    target.path_index = 0

    def draw(self, screen, camera_x, camera_y):
        if (self.pos.x < camera_x - self.size / 2 or self.pos.x > camera_x + VIEW_WIDTH + self.size / 2 or
                self.pos.y < camera_y - self.size / 2 or self.pos.y > camera_y + VIEW_HEIGHT + self.size / 2):
            return

        now = context.current_time

        # Choose facing + frames based on whether we're in attack animation window
        if now < self._attacking_until:
            facing = self._attack_facing or "D"
            frames = get_unit_attack_frames("Spearman", int(self.size), tuple(self.player_color[:3]), facing=facing)
            frame_time = self._ATTACK_FRAME_TIME
        else:
            facing = self._facing_from_velocity()
            frames = get_unit_walk_frames("Spearman", int(self.size), tuple(self.player_color[:3]), facing=facing)
            frame_time = self._WALK_FRAME_TIME

        if not frames:
            return super().draw(screen, camera_x, camera_y)

        idx = int(now / frame_time) % len(frames)
        image = frames[idx]

        x = self.pos.x - camera_x + VIEW_MARGIN_LEFT
        y = self.pos.y - camera_y + VIEW_MARGIN_TOP

        image_surface = image.copy()
        image_surface.set_alpha(self.alpha)
        rect = image_surface.get_rect(center=(int(x), int(y)))
        # Per-unit manual offset for ATTACK animations (auto-scaled by SCALE)
        if now < self._attacking_until:
            dx, dy = _scaled_attack_offset("Spearman", facing)
            rect.move_ip(dx, dy)

        _blit_sprite_with_border(screen, image_surface, rect)

        if self.selected:
            pygame.draw.rect(screen, self.player_color, (x - self.size / 2, y - self.size / 2, self.size, self.size), 1)
        if self.should_highlight(context.current_time):
            pygame.draw.rect(screen, WHITE, (x - self.size / 2, y - self.size / 2, self.size, self.size), 1)

        # HP bar (only when not fully healed)
        self.draw_health_bar(screen, x, y)
# Cow class
class Cow(Unit):
    milk_cost = 400
    wood_cost = 0
    returning = False
    return_pos = None

    _ANIM_DIR = "assets/units/cow"

    _ANIM_DIR = "assets/units/cow"

    _WALK_FRAME_TIME = 0.2
    _HARVEST_FRAME_TIME = 0.5
    _IDLE_SPEED_EPS2 = 0.05

    _HARVEST_HOLD_TIME = 0.40
    _DIE_FRAME_TIME = 0.10

    def __init__(self, x, y, player_id, player_color):
        super().__init__(x, y, size=UNIT_SIZE, speed=2, color=BROWN, player_id=player_id, player_color=player_color)
        self.harvest_rate = 0.005
        self.assigned_corner = None
        self.autonomous_target = False
        self.attack_damage = 5
        self.attack_range = 20
        self.attack_cooldown = 2.0
        self.armor = 1
        # remember last movement direction so harvesting keeps orientation
        self._last_facing = "D"

        # ✅ Harvest animation timer
        self._harvesting_until = 0.0

    def _facing_from_velocity(self) -> str:
        v = self.velocity
        if v.length_squared() < self._IDLE_SPEED_EPS2:
            return "M"

        x, y = v.x, v.y

        # 4-direction thresholds + diagonals
        if abs(x) < 0.35 and y < 0: return "U"
        if abs(x) < 0.35 and y > 0: return "D"
        if abs(y) < 0.35 and x < 0: return "L"
        if abs(y) < 0.35 and x > 0: return "R"

        if x < 0 and y < 0: return "LU"
        if x < 0 and y > 0: return "LD"
        if x > 0 and y < 0: return "RU"
        if x > 0 and y > 0: return "RD"

        return "D"

    def _get_walk_frames(self, facing: str) -> List[pygame.Surface]:
        cls_name = self.__class__.__name__
        size_px = int(self.size)
        pc = tuple(self.player_color[:3])

        if facing == "M":
            p = os.path.join(self._ANIM_DIR, "walk_M.gif")
            return _get_team_anim_frames(f"{cls_name}/walk_M", p, size_px, cls_name, pc)

        # present as files
        if facing in ("D", "L", "LD", "LU", "U"):
            p = os.path.join(self._ANIM_DIR, f"walk_{facing}.gif")
            return _get_team_anim_frames(f"{cls_name}/walk_{facing}", p, size_px, cls_name, pc)

        # mirrored directions (RIGHT side)
        if facing == "R":
            p = os.path.join(self._ANIM_DIR, "walk_L.gif")
            return _get_team_anim_frames(f"{cls_name}/walk_R_from_L", p, size_px, cls_name, pc, flip_x=True)

        if facing == "RD":
            p = os.path.join(self._ANIM_DIR, "walk_LD.gif")
            return _get_team_anim_frames(f"{cls_name}/walk_RD_from_LD", p, size_px, cls_name, pc, flip_x=True)

        if facing == "RU":
            p = os.path.join(self._ANIM_DIR, "walk_LU.gif")
            return _get_team_anim_frames(f"{cls_name}/walk_RU_from_LU", p, size_px, cls_name, pc, flip_x=True)

        # fallback
        p = os.path.join(self._ANIM_DIR, "walk_M.gif")
        return _get_team_anim_frames(f"{cls_name}/walk_M", p, size_px, cls_name, pc)

    def _get_harvest_frames(self, facing: str) -> List[pygame.Surface]:
        """
        Harvest uses harvest_<facing>.gif.
        Robustness:
        - accept common typo: harverst_<facing>.gif
        - for RIGHT directions: try direct file first, else mirror LEFT
        """
        cls_name = self.__class__.__name__
        size_px = int(self.size)
        pc = tuple(self.player_color[:3])

        if facing == "M" or not facing:
            facing = "D"

        def try_load(prefix: str, key_suffix: str, flip_x: bool = False) -> List[pygame.Surface]:
            path = os.path.join(self._ANIM_DIR, f"{prefix}{key_suffix}.gif")
            frames = _get_team_anim_frames(f"{cls_name}/{prefix}{key_suffix}", path, size_px, cls_name, pc, flip_x=flip_x)
            return frames

        def try_both_prefixes(key_suffix: str, flip_x: bool = False) -> List[pygame.Surface]:
            # prefer correct name first
            fr = try_load("harvest_", key_suffix, flip_x=flip_x)
            if fr:
                return fr
            # fallback: common typo
            fr = try_load("harverst_", key_suffix, flip_x=flip_x)
            if fr:
                return fr
            return []

        # Direct (non-mirrored) directions
        if facing in ("D", "L", "LD", "LU", "U"):
            fr = try_both_prefixes(facing, flip_x=False)
            if not fr:
                print(f"[Cow] Missing harvest gif for facing={facing} in {self._ANIM_DIR} "
                      f"(expected harvest_{facing}.gif or harverst_{facing}.gif)")
            return fr

        # Right side: try direct right gif first; otherwise mirror left
        if facing == "R":
            fr = try_both_prefixes("R", flip_x=False)
            if fr:
                return fr
            fr = try_both_prefixes("L", flip_x=True)
            if not fr:
                print(f"[Cow] Missing harvest gif for facing=R (expected harvest_R.gif / harverst_R.gif "
                      f"or mirror source harvest_L.gif / harverst_L.gif) in {self._ANIM_DIR}")
            return fr

        if facing == "RD":
            fr = try_both_prefixes("RD", flip_x=False)
            if fr:
                return fr
            fr = try_both_prefixes("LD", flip_x=True)
            if not fr:
                print(f"[Cow] Missing harvest gif for facing=RD (expected harvest_RD.gif / harverst_RD.gif "
                      f"or mirror source harvest_LD.gif / harverst_LD.gif) in {self._ANIM_DIR}")
            return fr

        if facing == "RU":
            fr = try_both_prefixes("RU", flip_x=False)
            if fr:
                return fr
            fr = try_both_prefixes("LU", flip_x=True)
            if not fr:
                print(f"[Cow] Missing harvest gif for facing=RU (expected harvest_RU.gif / harverst_RU.gif "
                      f"or mirror source harvest_LU.gif / harverst_LU.gif) in {self._ANIM_DIR}")
            return fr

        # Fallback to D
        fr = try_both_prefixes("D", flip_x=False)
        if not fr:
            print(f"[Cow] Missing fallback harvest_D.gif/harverst_D.gif in {self._ANIM_DIR}")
        return fr

    def _get_die_frames(self, facing: str) -> List[pygame.Surface]:
        # simplest: one die.gif for all; or directional die_D, die_L... if you have them
        cls_name = self.__class__.__name__
        size_px = int(self.size)
        pc = tuple(self.player_color[:3])

        # Try directional first
        if facing in ("D", "L", "LD", "LU", "U", "M"):
            key = "D" if facing == "M" else facing
            p = os.path.join(self._ANIM_DIR, f"die_{key}.gif")
            fr = _get_team_anim_frames(f"{cls_name}/die_{key}", p, size_px, cls_name, pc, flip_x=False)
            if fr:
                return fr

        # Mirror for right if only left exists
        if facing == "R":
            p = os.path.join(self._ANIM_DIR, "die_L.gif")
            fr = _get_team_anim_frames(f"{cls_name}/die_R_from_L", p, size_px, cls_name, pc, flip_x=True)
            if fr:
                return fr

        if facing == "RD":
            p = os.path.join(self._ANIM_DIR, "die_LD.gif")
            fr = _get_team_anim_frames(f"{cls_name}/die_RD_from_LD", p, size_px, cls_name, pc, flip_x=True)
            if fr:
                return fr

        if facing == "RU":
            p = os.path.join(self._ANIM_DIR, "die_LU.gif")
            fr = _get_team_anim_frames(f"{cls_name}/die_RU_from_LU", p, size_px, cls_name, pc, flip_x=True)
            if fr:
                return fr

        # Fallback: single die.gif
        p = os.path.join(self._ANIM_DIR, "die.gif")
        return _get_team_anim_frames(f"{cls_name}/die", p, size_px, cls_name, pc, flip_x=False)

    def draw(self, screen, camera_x, camera_y):
        if (self.pos.x < camera_x - self.size / 2 or self.pos.x > camera_x + VIEW_WIDTH + self.size / 2 or
                self.pos.y < camera_y - self.size / 2 or self.pos.y > camera_y + VIEW_HEIGHT + self.size / 2):
            return

        raw_facing = self._facing_from_velocity()

        # update last facing only when moving
        if raw_facing != "M":
            self._last_facing = raw_facing

        is_harvesting = context.current_time <= self._harvesting_until

        if is_harvesting:
            # ✅ Harvest uses last movement direction (default D)
            facing = self._last_facing or "D"
            frames = self._get_harvest_frames(facing)
            frame_time = self._HARVEST_FRAME_TIME
        else:
            facing = raw_facing
            frames = self._get_walk_frames(facing)
            frame_time = self._WALK_FRAME_TIME

        if not frames:
            return super().draw(screen, camera_x, camera_y)

        idx = int(context.current_time / frame_time) % len(frames)
        image = frames[idx]

        x = self.pos.x - camera_x + VIEW_MARGIN_LEFT
        y = self.pos.y - camera_y + VIEW_MARGIN_TOP

        image_surface = image.copy()
        image_surface.set_alpha(self.alpha)
        rect = image_surface.get_rect(center=(int(x), int(y)))
        _blit_sprite_with_border(screen, image_surface, rect)

        if self.selected:
            pygame.draw.rect(screen, self.player_color,
                             (x - self.size / 2, y - self.size / 2, self.size, self.size), 1)
        if self.should_highlight(context.current_time):
            pygame.draw.rect(screen, WHITE,
                             (x - self.size / 2, y - self.size / 2, self.size, self.size), 1)

        # HP bar (only when not fully healed) + special bar (kept as before)
        self.draw_health_bar(screen, x, y)

        # Special bar: align with the HP bar width/position
        pad = 4
        bar_w = max(10, int(self.size) - pad * 2)
        bar_h = 5
        bar_offset = 3
        bar_x = x - bar_w / 2
        bar_y = y - self.size / 2 - bar_h - bar_offset

        if self.special > 0:
            fill_w = int(round(bar_w * max(0.0, min(1.0, self.special / 100.0))))
            pygame.draw.rect(screen, WHITE, (bar_x, bar_y + bar_h + 2, fill_w, bar_h))
            pygame.draw.rect(screen, BLACK, (bar_x, bar_y + bar_h + 2, bar_w, bar_h), 1)

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
                # print(f"Cow at {self.pos} targeting barn at {self.target}")
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

                        # ✅ trigger harvest animation, keep last movement direction
                        self._harvesting_until = context.current_time + self._HARVEST_HOLD_TIME

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
                                    # print(f"Cow at {self.pos} targeting adjacent grass tile at {self.target}")
                                    break