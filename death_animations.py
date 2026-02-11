from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pygame
from PIL import Image

from constants import VIEW_MARGIN_LEFT, VIEW_MARGIN_TOP


@dataclass
class DeathAnim:
    """One-shot lingering death animation (units only)."""
    pos: pygame.math.Vector2
    size: int
    frames: List[pygame.Surface]
    start_time: float
    total_duration: float = 60.0  # seconds
    frame_duration: float = 30.0  # seconds (2 frames => 60s)

    def is_expired(self, now: float) -> bool:
        return (now - self.start_time) >= self.total_duration

    def current_frame(self, now: float) -> Optional[pygame.Surface]:
        if not self.frames:
            return None
        elapsed = max(0.0, now - self.start_time)
        # No repeat: clamp to last frame index (but object expires at total_duration)
        idx = int(elapsed // self.frame_duration)
        if idx >= len(self.frames):
            idx = len(self.frames) - 1
        return self.frames[idx]


class DeathAnimationManager:
    """Keeps and draws lingering death animations.

    Requirements implemented:
    - Only spawned for *units* (NOT buildings, NOT trees).
    - Loads one file per unit: <unit_anim_dir>/death.gif (or assets/units/<unit>/death.gif fallback)
    - Does NOT loop; shows frames slowly so the full gif duration is ~60s.
    - After ~60s, disappears.
    - Logs once per unit class if death.gif missing.
    """

    def __init__(self) -> None:
        self._anims: List[DeathAnim] = []
        self._frame_cache: Dict[Tuple[str, int], List[pygame.Surface]] = {}
        self._missing_logged: set[str] = set()

    def update(self, now: float) -> None:
        self._anims = [a for a in self._anims if not a.is_expired(now)]

    def draw(self, screen: pygame.Surface, camera_x: float, camera_y: float, now: float) -> None:
        for anim in self._anims:
            frame = anim.current_frame(now)
            if frame is None:
                continue
            x = anim.pos.x - camera_x + VIEW_MARGIN_LEFT
            y = anim.pos.y - camera_y + VIEW_MARGIN_TOP
            rect = frame.get_rect(center=(int(x), int(y)))
            screen.blit(frame, rect.topleft)

    def spawn_from_unit(self, unit, now: float) -> None:
        # Only units (not buildings/trees). We avoid importing Building/Tree here to keep the module lightweight.
        if getattr(unit, "hp", 1) > 0:
            return

        cls = unit.__class__
        cls_name = cls.__name__
        if cls_name in ("Tree",) or getattr(unit, "SIZE_TILES", None) is not None:
            # Tree: special world unit; SIZE_TILES exists on buildings (and walls).
            return

        size = int(getattr(unit, "size", 0) or 0)
        if size <= 0:
            size = 32

        frames = self._get_death_frames_for_class(cls, size)
        if not frames:
            return

        # Slow down so the whole animation takes about 60 seconds.
        total = 60.0
        per = total / max(1, len(frames))

        self._anims.append(
            DeathAnim(
                pos=pygame.math.Vector2(unit.pos),
                size=size,
                frames=frames,
                start_time=float(now),
                total_duration=total,
                frame_duration=per,
            )
        )

    def _get_death_frames_for_class(self, cls, size: int) -> List[pygame.Surface]:
        cls_name = cls.__name__
        # Resolve path: prefer unit class' _ANIM_DIR if present.
        anim_dir = getattr(cls, "_ANIM_DIR", None)
        if not anim_dir:
            anim_dir = os.path.join("assets", "units", cls_name.lower())
        path = os.path.join(anim_dir, "death.gif")

        cache_key = (path, int(size))
        if cache_key in self._frame_cache:
            return self._frame_cache[cache_key]

        if not os.path.exists(path):
            if cls_name not in self._missing_logged:
                print(f"[death] missing death.gif for unit '{cls_name}' at: {path}")
                self._missing_logged.add(cls_name)
            self._frame_cache[cache_key] = []
            return []

        try:
            img = Image.open(path)
            frames: List[pygame.Surface] = []
            for i in range(getattr(img, "n_frames", 1)):
                img.seek(i)
                frame = img.convert("RGBA")
                mode = frame.mode
                w, h = frame.size
                data = frame.tobytes()
                surf = pygame.image.fromstring(data, (w, h), mode).convert_alpha()
                if (w, h) != (size, size):
                    surf = pygame.transform.smoothscale(surf, (size, size))
                frames.append(surf)
            self._frame_cache[cache_key] = frames
            return frames
        except Exception as e:
            print(f"[death] failed to load {path}: {e}")
            self._frame_cache[cache_key] = []
            return []
