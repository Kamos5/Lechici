from __future__ import annotations

import os
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
