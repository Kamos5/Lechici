from __future__ import annotations

import os
import random
import weakref
from typing import List, Optional, Tuple

import pygame
from PIL import Image
from pygame.math import Vector2

from constants import TILE_SIZE, VIEW_MARGIN_LEFT, VIEW_MARGIN_TOP


# Cache per (folder_key, size_px, flip_x) -> frames
_EFFECT_ANIM_CACHE = {}


def _load_gif_frames(path: str) -> List[pygame.Surface]:
    """Load all frames from an animated GIF using Pillow."""
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


def _get_effect_frames(folder_key: str, path: str, desired_px: int, flip_x: bool = False) -> List[pygame.Surface]:
    key = (folder_key, int(desired_px), bool(flip_x))
    if key in _EFFECT_ANIM_CACHE:
        return _EFFECT_ANIM_CACHE[key]

    frames: List[pygame.Surface] = []
    if os.path.exists(path):
        if path.lower().endswith(".gif"):
            frames = _load_gif_frames(path)
        else:
            try:
                frames = [pygame.image.load(path).convert_alpha()]
            except Exception:
                frames = []
    else:
        frames = []

    if frames:
        # scale each frame so the arrow roughly fits in a tile unless caller overrides
        scaled = []
        for f in frames:
            scaled.append(pygame.transform.scale(f, (int(desired_px), int(desired_px))))
        frames = scaled

    if flip_x and frames:
        frames = [pygame.transform.flip(f, True, False) for f in frames]

    _EFFECT_ANIM_CACHE[key] = frames
    return frames


def _arrow_asset_path(facing: str) -> Optional[str]:
    """
    Prefer GIFs, fall back to PNGs if present.
    Expected names:
      assets/effects/arrow/arrow_<facing>.gif  (or .png)
    """
    base = os.path.join("assets", "effects", "arrow", f"arrow_{facing}")
    for ext in (".gif", ".png"):
        p = base + ext
        if os.path.exists(p):
            return p
    # Even if file doesn't exist yet, return gif path so dev sees intended name
    return base + ".gif"


class ArrowEffect:
    """
    A simple projectile effect that flies from start_pos to end_pos.
    - Uses arrow_<direction> animated asset (GIF or PNG).
    - Can be delayed via spawn_time (so it fires mid-attack animation).
    """

    _FRAME_TIME = 0.08

    def __init__(
        self,
        start_pos: Vector2,
        end_pos: Vector2,
        facing: str,
        size_px: int = TILE_SIZE,
        speed_px_per_s: float = 700.0,
        spawn_time: float = 0.0,

        on_hit=None,
    ):
        self.start_pos = Vector2(start_pos)
        self.end_pos = Vector2(end_pos)
        self.pos = Vector2(start_pos)
        self.facing = facing
        self.size_px = int(size_px)
        self.speed = float(speed_px_per_s)
        self.spawn_time = float(spawn_time)

        self.done = False
        self.on_hit = on_hit

        # pre-load frames (mirror right facings the same way units do)
        if facing in ("M", "D", "L", "LD", "LU", "U"):
            path = _arrow_asset_path(facing)
            self.frames = _get_effect_frames(f"arrow_{facing}", path, self.size_px, flip_x=False)
        elif facing == "R":
            path = _arrow_asset_path("L")
            self.frames = _get_effect_frames("arrow_L", path, self.size_px, flip_x=True)
        elif facing == "RD":
            path = _arrow_asset_path("LD")
            self.frames = _get_effect_frames("arrow_LD", path, self.size_px, flip_x=True)
        elif facing == "RU":
            path = _arrow_asset_path("LU")
            self.frames = _get_effect_frames("arrow_LU", path, self.size_px, flip_x=True)
        else:
            path = _arrow_asset_path("D")
            self.frames = _get_effect_frames("arrow_D", path, self.size_px, flip_x=False)

    def update(self, current_time: float, dt: float) -> None:
        if self.done:
            return
        if current_time < self.spawn_time:
            return

        to_end = self.end_pos - self.pos
        dist = to_end.length()
        if dist <= 1.0:
            self.pos = Vector2(self.end_pos)
            if self.on_hit:
                self.on_hit()
            self.done = True
            return

        step = self.speed * dt
        if step >= dist:
            self.pos = Vector2(self.end_pos)
            if self.on_hit:
                self.on_hit()
            self.done = True
        else:
            self.pos += to_end.normalize() * step

    def draw(self, screen: pygame.Surface, camera_x: float, camera_y: float, current_time: float) -> None:
        if self.done or current_time < self.spawn_time:
            return
        if not self.frames:
            return

        idx = int(current_time / self._FRAME_TIME) % len(self.frames)
        image = self.frames[idx]
        x = self.pos.x - camera_x + VIEW_MARGIN_LEFT
        y = self.pos.y - camera_y + VIEW_MARGIN_TOP
        rect = image.get_rect(center=(int(x), int(y)))
        screen.blit(image, rect)


# ------------------ Flame / fire effects ------------------

# Cache per (level, size_px) -> frames
_FIRE_ANIM_CACHE = {}
# Remember missing flame assets so we don't spam the console.
_MISSING_FIRE_WARNED = set()

_FIRE_FRAME_TIME = 0.12

def _fire_asset_path(level: int) -> str | None:
    """Resolve flame gif path for a given level (1..3)."""
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

def get_flame_frames(level: int, size_px: int) -> list[pygame.Surface]:
    """Load & cache flame GIF frames scaled to (size_px, size_px)."""
    key = (int(level), int(size_px))
    if key in _FIRE_ANIM_CACHE:
        return _FIRE_ANIM_CACHE[key]

    path = _fire_asset_path(level)
    if not path or not os.path.exists(path):
        _FIRE_ANIM_CACHE[key] = []
        if key not in _MISSING_FIRE_WARNED:
            _MISSING_FIRE_WARNED.add(key)
            print(f"[fire] Missing flame asset for level={level}: expected flame{level}.gif under assets/effects/flame/ (or similar).")
        return []

    frames = _load_gif_frames(path)
    out: list[pygame.Surface] = []
    for fr in frames:
        out.append(pygame.transform.scale(fr, (int(size_px), int(size_px))))
    _FIRE_ANIM_CACHE[key] = out
    return out

class FlameEffect:
    """A looping flame animation at a fixed world position.

    Can be used as a set piece by spawning into context.effects.
    """

    def __init__(
        self,
        pos: Vector2,
        *,
        level: int = 1,
        size_px: int = TILE_SIZE,
        spawn_time: float = 0.0,
        duration: float | None = None,
    ):
        self.pos = Vector2(pos)
        self.level = int(level)
        self.size_px = int(size_px)
        self.spawn_time = float(spawn_time)
        self.duration = None if duration is None else float(duration)
        self.done = False

    def update(self, current_time: float, dt: float) -> None:
        if self.done:
            return
        if current_time < self.spawn_time:
            return
        if self.duration is not None and current_time >= self.spawn_time + self.duration:
            self.done = True

    def draw(self, screen: pygame.Surface, camera_x: float, camera_y: float, current_time: float) -> None:
        if self.done or current_time < self.spawn_time:
            return
        frames = get_flame_frames(self.level, self.size_px)
        if not frames:
            return
        idx = int(current_time / _FIRE_FRAME_TIME) % len(frames)
        fr = frames[idx]
        x = self.pos.x - camera_x + VIEW_MARGIN_LEFT
        y = self.pos.y - camera_y + VIEW_MARGIN_TOP
        rect = fr.get_rect(center=(int(x), int(y)))
        screen.blit(fr, rect)

# Stable per-building overlay state without storing sprite state on units.
_BUILDING_FIRE_STATE: "weakref.WeakKeyDictionary[object, dict]" = weakref.WeakKeyDictionary()

def _compute_building_fire_state(hp: float, max_hp: float) -> tuple[int, int]:
    """Return (flame_level, flame_count) based on HP percent.
    Levels map to flame1/2/3 assets. Count increases 1→2→3 as damage worsens."""
    mx = float(max_hp or 0)
    if mx <= 0:
        return (0, 0)
    pct = max(0.0, min(1.0, float(hp or 0) / mx))
    if pct > 0.75:
        return (0, 0)
    if pct > 0.50:
        return (1, 1)
    if pct > 0.25:
        return (2, 2)
    return (3, 3)

def _ensure_building_fire_slots(building: object, tile_size: int) -> dict:
    """Ensure stable slots/jitter for this building."""
    st = _BUILDING_FIRE_STATE.get(building)
    if isinstance(st, dict) and len(st.get("slots", [])) >= 3 and len(st.get("jitter", [])) >= 3:
        return st

    candidates = [(ox, oy) for ox in (-1, 0, 1) for oy in (-1, 0, 1)]
    slots = random.sample(candidates, 3) if len(candidates) >= 3 else candidates
    j = max(1, int(round(float(tile_size) * 0.15)))
    jitter = [(random.randint(-j, j), random.randint(-j, j)) for _ in range(len(slots))]
    st = {"slots": slots, "jitter": jitter}
    _BUILDING_FIRE_STATE[building] = st
    return st

def draw_building_damage_fire(building: object, screen: pygame.Surface, x: float, y: float, *, now: float) -> None:
    """Draw health-based fire overlay for a building (no unit sprite maintenance).

    Expects:
      - building.hp, building.max_hp
      - building.SIZE_TILES (optional; defaults to 1)
      - building.alpha (optional; defaults to 255)
    """
    try:
        size_tiles = int(getattr(building, "SIZE_TILES", 1) or 1)
        if size_tiles < 3:
            return
        if int(getattr(building, "alpha", 255) or 255) < 255:
            return
        hp = float(getattr(building, "hp", 0) or 0)
        mx = float(getattr(building, "max_hp", 0) or 0)
    except Exception:
        return

    level, count = _compute_building_fire_state(hp, mx)
    if level <= 0 or count <= 0:
        return

    frames = get_flame_frames(level, int(TILE_SIZE))
    if not frames:
        return

    st = _ensure_building_fire_slots(building, int(TILE_SIZE))
    slots = st.get("slots", [])
    jitter = st.get("jitter", [])

    idx = int(float(now) / _FIRE_FRAME_TIME) % len(frames)
    fr = frames[idx]

    # Lift a bit so flames appear on the roof/top of the 3x3 sprite.
    lift = int(round(float(TILE_SIZE) * 0.35))
    n = min(int(count), len(slots))

    for i in range(n):
        ox, oy = slots[i]
        jx, jy = jitter[i] if i < len(jitter) else (0, 0)
        fx = float(x) + float(ox) * float(TILE_SIZE) + float(jx)
        fy = float(y) + float(oy) * float(TILE_SIZE) + float(jy) - float(lift)
        rect = fr.get_rect(center=(int(fx), int(fy)))
        screen.blit(fr, rect)


def update_effects(effects: list, current_time: float, dt: float) -> None:
    """Update and prune a list of effects (e.g., context.effects)."""
    for e in list(effects):
        if hasattr(e, "update"):
            e.update(current_time, dt)
    effects[:] = [e for e in effects if not getattr(e, "done", False)]


def draw_effects(effects: list, screen: pygame.Surface, camera_x: float, camera_y: float, current_time: float) -> None:
    """Draw a list of effects (e.g., context.effects)."""
    for e in effects:
        if hasattr(e, "draw"):
            e.draw(screen, camera_x, camera_y, current_time)
