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

class Bridge(SimpleTile):
    _image = None

    def __init__(self, x, y):
        super().__init__(x, y)
        self.grass_level = 0.0  # No grass, like Dirt
        Bridge.load_image()
    @classmethod
    def load_image(cls):
        try:
            cls._image = pygame.image.load("assets/bridge_hor.png").convert_alpha()
            cls._image = pygame.transform.scale(cls._image, (TILE_SIZE, TILE_SIZE))
        except (pygame.error, FileNotFoundError) as e:
            print(f"Failed to load bridge_hor.png: {e}")
            cls._image = None
    def regrow(self, rate):
        pass  # Bridges don't regrow grass

class River(SimpleTile):
    _image = None

    def __init__(self, x, y):
        super().__init__(x, y)
        self.grass_level = 0.0  # Non-harvestable

        if River._image is None:
            River.load_image()

    @classmethod
    def load_image(cls):
        try:
            cls._image = pygame.image.load(f"assets/{cls.__name__.lower()}.png").convert_alpha()
            cls._image = pygame.transform.scale(cls._image, (TILE_SIZE, TILE_SIZE))
        except (pygame.error, FileNotFoundError) as e:
            print(f"Failed to load assets/{cls.__name__.lower()}.png: {e}")
            cls._image = None

    def regrow(self, rate):
        pass  # River tiles don't regrow grass

# Unit class
