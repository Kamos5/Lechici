from __future__ import annotations

from typing import Optional, Tuple

import pygame
from pygame import Vector2

from constants import *


class SimpleTile:
    _image = None

    def __init__(self, x, y):
        self.pos = Vector2(x, y)
        self.center = Vector2(x + TILE_HALF, y + TILE_HALF)

    def draw(self, surface, camera_x, camera_y):
        if (self.pos.x < camera_x - TILE_SIZE or self.pos.x > camera_x + VIEW_WIDTH or
            self.pos.y < camera_y - TILE_SIZE or self.pos.y > camera_y + VIEW_HEIGHT):
            return
        if self._image:
            surface.blit(self._image, (self.pos.x - camera_x, self.pos.y - camera_y))
        else:
            pygame.draw.rect(surface, GRASS_BROWN, (self.pos.x - camera_x, self.pos.y - camera_y, TILE_SIZE, TILE_SIZE))

    def harvest(self, amount):
        return 0.0

    def regrow(self, amount):
        pass


class WorldObject:
    """
    Drawn ABOVE tiles but BELOW units.
    Grid-aligned: x,y are top-left tile coords like tiles.
    """
    def __init__(self, x: int, y: int, *, passable: bool = False):
        self.pos = Vector2(x, y)
        self.center = Vector2(x + TILE_HALF, y + TILE_HALF)
        self.passable = bool(passable)

    def draw(self, surface, camera_x, camera_y):
        raise NotImplementedError

# GrassTile class
class GrassTile(SimpleTile):
    _grass_image = None
    _dirt_image = None
    _surface_cache = {}

    def __init__(self, x, y):
        super().__init__(x, y)
        self.grass_level = 1.0
        if GrassTile._grass_image is None or GrassTile._dirt_image is None:
            GrassTile.load_images()

    @classmethod
    def load_images(cls):
        try:
            cls._grass_image = pygame.image.load("assets/grass.png").convert_alpha()
            cls._grass_image = pygame.transform.scale(cls._grass_image, (TILE_SIZE, TILE_SIZE))
            cls._dirt_image = pygame.image.load("assets/dirt.png").convert_alpha()
            cls._dirt_image = pygame.transform.scale(cls._dirt_image, (TILE_SIZE, TILE_SIZE))
        except (pygame.error, FileNotFoundError) as e:
            print(f"Failed to load grass.png or dirt.png: {e}")
            cls._grass_image = None
            cls._dirt_image = None

    def draw(self, surface, camera_x, camera_y):
        if (self.pos.x < camera_x - TILE_SIZE or self.pos.x > camera_x + VIEW_WIDTH or
            self.pos.y < camera_y - TILE_SIZE or self.pos.y > camera_y + VIEW_HEIGHT):
            return
        if self._grass_image and self._dirt_image:
            level_key = round(self.grass_level * 100)
            if level_key not in self._surface_cache:
                blended_surface = pygame.Surface((TILE_SIZE, TILE_SIZE), pygame.SRCALPHA)
                blended_surface.blit(self._grass_image, (0, 0))
                dirt_alpha = int((1.0 - self.grass_level) * 255)
                dirt_overlay = self._dirt_image.copy()
                dirt_overlay.set_alpha(dirt_alpha)
                blended_surface.blit(dirt_overlay, (0, 0))
                self._surface_cache[level_key] = blended_surface
            surface.blit(self._surface_cache[level_key], (self.pos.x - camera_x, self.pos.y - camera_y))
        else:
            color = (
                int(GRASS_GREEN[0] * self.grass_level + GRASS_BROWN[0] * (1 - self.grass_level)),
                int(GRASS_GREEN[1] * self.grass_level + GRASS_BROWN[1] * (1 - self.grass_level)),
                int(GRASS_GREEN[2] * self.grass_level + GRASS_BROWN[2] * (1 - self.grass_level))
            )
            pygame.draw.rect(surface, color, (self.pos.x - camera_x, self.pos.y - camera_y, TILE_SIZE, TILE_SIZE))

    def harvest(self, amount):
        old_level = self.grass_level
        self.grass_level = max(0.0, self.grass_level - amount)
        return old_level - self.grass_level

    def regrow(self, amount):
        if self.grass_level < 1.0:
            self.grass_level = min(1.0, self.grass_level + amount)

# Dirt class
class Dirt(SimpleTile):
    _image = None

    def __init__(self, x, y):
        self.grass_level = 0.0
        super().__init__(x, y)
        if Dirt._image is None:
            Dirt.load_image()

    @classmethod
    def load_image(cls):
        try:
            cls._image = pygame.image.load(f"assets/{cls.__name__.lower()}.png").convert_alpha()
            cls._image = pygame.transform.scale(cls._image, (TILE_SIZE, TILE_SIZE))
        except (pygame.error, FileNotFoundError) as e:
            print(f"Failed to load assets/{cls.__name__.lower()}.png: {e}")
            cls._image = None

# Foundation class
class Foundation(SimpleTile):
    _image = None

    def __init__(self, x, y):
        self.grass_level = 0.0
        super().__init__(x, y)
        if Foundation._image is None:
            Foundation.load_image()

    @classmethod
    def load_image(cls):
        try:
            cls._image = pygame.image.load(f"assets/{cls.__name__.lower()}.png").convert_alpha()
            cls._image = pygame.transform.scale(cls._image, (TILE_SIZE, TILE_SIZE))
        except (pygame.error, FileNotFoundError) as e:
            print(f"Failed to load assets/{cls.__name__.lower()}.png: {e}")
            cls._image = None

class Bridge(WorldObject):
    _variant_images: dict[str, pygame.Surface | None] = {}
    DEFAULT_VARIANT = "bridge5"

    def __init__(self, x, y, *, variant: str | None = None, passable: bool = True):
        super().__init__(x, y, passable=passable)
        self.variant = self._sanitize_variant(variant) or self.DEFAULT_VARIANT
        if self.variant not in Bridge._variant_images:
            Bridge._variant_images[self.variant] = Bridge._load_variant_image(self.variant)

    @staticmethod
    def _sanitize_variant(v: str | None) -> str | None:
        if not isinstance(v, str) or not v.startswith("bridge"):
            return None
        try:
            idx = int(v[len("bridge"):])
        except Exception:
            return None
        return v if 1 <= idx <= 9 else None

    @classmethod
    def _load_variant_image(cls, variant: str) -> pygame.Surface | None:
        path = f"assets/worldObjects/bridge/{variant}.png"
        try:
            img = pygame.image.load(path).convert_alpha()
            return pygame.transform.scale(img, (TILE_SIZE, TILE_SIZE))
        except (pygame.error, FileNotFoundError) as e:
            print(f"Failed to load {path}: {e}")
            return None

    def draw(self, surface, camera_x, camera_y):
        if (self.pos.x < camera_x - TILE_SIZE or self.pos.x > camera_x + VIEW_WIDTH or
            self.pos.y < camera_y - TILE_SIZE or self.pos.y > camera_y + VIEW_HEIGHT):
            return

        img = Bridge._variant_images.get(self.variant)
        if img is not None:
            surface.blit(img, (self.pos.x - camera_x, self.pos.y - camera_y))
        else:
            pygame.draw.rect(
                surface,
                (120, 90, 40),
                (self.pos.x - camera_x, self.pos.y - camera_y, TILE_SIZE, TILE_SIZE),
            )

class River(SimpleTile):
    # Cache per-variant, already scaled to TILE_SIZE
    _variant_images: dict[str, pygame.Surface | None] = {}
    DEFAULT_VARIANT = "river5"  # requested default

    def __init__(self, x, y, *, variant: str | None = None):
        super().__init__(x, y)
        self.grass_level = 0.0  # Non-harvestable
        self.variant = self._sanitize_variant(variant) or self.DEFAULT_VARIANT

        # Ensure this variant is loaded
        if self.variant not in River._variant_images:
            River._variant_images[self.variant] = River._load_variant_image(self.variant)

    @staticmethod
    def _sanitize_variant(v: str | None) -> str | None:
        if not isinstance(v, str):
            return None
        if not v.startswith("river"):
            return None
        try:
            idx = int(v[len("river"):])
        except Exception:
            return None
        return v if RIVER_VARIANT_MIN <= idx <= RIVER_VARIANT_MAX else None

    @classmethod
    def _load_variant_image(cls, variant: str) -> pygame.Surface | None:
        path = f"assets/river/{variant}.png"
        try:
            img = pygame.image.load(path).convert_alpha()
            return pygame.transform.scale(img, (TILE_SIZE, TILE_SIZE))
        except (pygame.error, FileNotFoundError) as e:
            print(f"Failed to load {path}: {e}")
            return None

    def draw(self, surface, camera_x, camera_y):
        # Same culling as SimpleTile.draw, but pick per-instance image
        if (self.pos.x < camera_x - TILE_SIZE or self.pos.x > camera_x + VIEW_WIDTH or
            self.pos.y < camera_y - TILE_SIZE or self.pos.y > camera_y + VIEW_HEIGHT):
            return

        img = River._variant_images.get(self.variant)
        if img is not None:
            surface.blit(img, (self.pos.x - camera_x, self.pos.y - camera_y))
        else:
            # fallback color if missing
            pygame.draw.rect(
                surface,
                (40, 90, 180),
                (self.pos.x - camera_x, self.pos.y - camera_y, TILE_SIZE, TILE_SIZE),
            )

    def regrow(self, rate):
        pass  # River tiles don't regrow grass

class Mountain(SimpleTile):
    # Cache per-variant, already scaled to TILE_SIZE
    _variant_images: dict[str, pygame.Surface | None] = {}
    DEFAULT_VARIANT = "mountain1"  # requested default

    def __init__(self, x, y, *, variant: str | None = None):
        super().__init__(x, y)
        self.grass_level = 0.0  # Non-harvestable
        self.variant = self._sanitize_variant(variant) or self.DEFAULT_VARIANT

        # Ensure this variant is loaded
        if self.variant not in Mountain._variant_images:
            Mountain._variant_images[self.variant] = Mountain._load_variant_image(self.variant)

    @staticmethod
    def _sanitize_variant(v: str | None) -> str | None:
        if not isinstance(v, str):
            return None
        if not v.startswith("mountain"):
            return None
        try:
            idx = int(v[len("mountain"):])
        except Exception:
            return None
        return v if MOUNTAIN_VARIANT_MIN <= idx <= MOUNTAIN_VARIANT_MAX else None

    @classmethod
    def _load_variant_image(cls, variant: str) -> pygame.Surface | None:
        path = f"assets/tiles/mountain/{variant}.png"
        try:
            img = pygame.image.load(path).convert_alpha()
            return pygame.transform.scale(img, (TILE_SIZE, TILE_SIZE))
        except (pygame.error, FileNotFoundError) as e:
            print(f"Failed to load {path}: {e}")
            return None

    def draw(self, surface, camera_x, camera_y):
        # Same culling as SimpleTile.draw, but pick per-instance image
        if (self.pos.x < camera_x - TILE_SIZE or self.pos.x > camera_x + VIEW_WIDTH or
                self.pos.y < camera_y - TILE_SIZE or self.pos.y > camera_y + VIEW_HEIGHT):
            return

        img = Mountain._variant_images.get(self.variant)
        if img is not None:
            surface.blit(img, (self.pos.x - camera_x, self.pos.y - camera_y))
        else:
            # fallback color if missing
            pygame.draw.rect(
                surface,
                (40, 90, 180),
                (self.pos.x - camera_x, self.pos.y - camera_y, TILE_SIZE, TILE_SIZE),
            )

    def regrow(self, rate):
        pass  # Mountain tiles don't regrow grass

