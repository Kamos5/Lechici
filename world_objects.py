import pygame
from pygame import Vector2

from constants import TILE_HALF, TILE_SIZE, VIEW_WIDTH, VIEW_HEIGHT


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

class Road(WorldObject):
    """A passable, tile-aligned world object built by players.

    Notes:
      - x,y are TOP-LEFT tile coords (like tiles), not tile center.
      - Roads are passable (they don't block movement).
      - Roads have NO ownership: player_id is optional and ignored by placement/connection logic.
        Any player may connect to any existing road network.
    """

    # Build costs (used by the same UI/production system as buildings)
    milk_cost = 0
    wood_cost = 10

    _variant_images: dict[str, pygame.Surface | None] = {}

    # Default should be the "no adjacent road" tile per your rules
    DEFAULT_VARIANT = "road10"
    VARIANT_MIN = 1
    VARIANT_MAX = 15

    def __init__(self, x, y, *, variant: str | None = None, passable: bool = True, player_id: int | None = None):
        # Roads should be passable by default
        super().__init__(x, y, passable=passable)
        self.player_id = player_id
        self.variant = self._sanitize_variant(variant) or self.DEFAULT_VARIANT
        if self.variant not in Road._variant_images:
            Road._variant_images[self.variant] = Road._load_variant_image(self.variant)

    @staticmethod
    def _sanitize_variant(v: str | None) -> str | None:
        if not isinstance(v, str) or not v.startswith("road"):
            return None
        try:
            idx = int(v[len("road"):])
        except Exception:
            return None
        return v if Road.VARIANT_MIN <= idx <= Road.VARIANT_MAX else None

    def set_variant(self, variant: str) -> None:
        v = self._sanitize_variant(variant) or self.DEFAULT_VARIANT
        self.variant = v
        if v not in Road._variant_images:
            Road._variant_images[v] = Road._load_variant_image(v)

    @classmethod
    def _load_variant_image(cls, variant: str) -> pygame.Surface | None:
        path = f"assets/worldObjects/road/{variant}.png"
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

        img = Road._variant_images.get(self.variant)
        if img is not None:
            surface.blit(img, (self.pos.x - camera_x, self.pos.y - camera_y))
        else:
            pygame.draw.rect(
                surface,
                (90, 90, 90),
                (self.pos.x - camera_x, self.pos.y - camera_y, TILE_SIZE, TILE_SIZE),
            )

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

class MiscPassable(WorldObject):
    _variant_images: dict[str, pygame.Surface | None] = {}
    DEFAULT_VARIANT = "misc_pass1"

    def __init__(self, x, y, *, variant: str | None = None):
        # Walls are NEVER passable
        super().__init__(x, y, passable=True)

        self.variant = self._sanitize_variant(variant) or self.DEFAULT_VARIANT
        if self.variant not in MiscPassable._variant_images:
            MiscPassable._variant_images[self.variant] = MiscPassable._load_variant_image(self.variant)

    @staticmethod
    def _sanitize_variant(v: str | None) -> str | None:
        if not isinstance(v, str) or not v.startswith("misc_pass"):
            return None
        try:
            idx = int(v[len("misc_pass"):])
        except Exception:
            return None
        return v if 1 <= idx <= 12 else None

    @classmethod
    def _load_variant_image(cls, variant: str) -> pygame.Surface | None:
        path = f"assets/worldObjects/misc_pass/{variant}.png"
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

        img = MiscPassable._variant_images.get(self.variant)
        if img is not None:
            surface.blit(img, (self.pos.x - camera_x, self.pos.y - camera_y))
        else:
            pygame.draw.rect(
                surface,
                (60, 60, 60),
                (self.pos.x - camera_x, self.pos.y - camera_y, TILE_SIZE, TILE_SIZE),
            )


class MiscImpassable(WorldObject):
    """A tile-aligned world object that blocks movement (impassable).

    Assets live in: assets/worldObjects/misc_impa/{variant}.png
    Expected variant names: misc_impa1..misc_impa12 (adjust VARIANT_MAX if needed).
    """

    _variant_images: dict[str, pygame.Surface | None] = {}
    DEFAULT_VARIANT = "misc_impa1"
    VARIANT_MIN = 1
    VARIANT_MAX = 12

    def __init__(self, x, y, *, variant: str | None = None):
        super().__init__(x, y, passable=False)
        self.variant = self._sanitize_variant(variant) or self.DEFAULT_VARIANT
        if self.variant not in MiscImpassable._variant_images:
            MiscImpassable._variant_images[self.variant] = MiscImpassable._load_variant_image(self.variant)

    @staticmethod
    def _sanitize_variant(v: str | None) -> str | None:
        if not isinstance(v, str) or not v.startswith("misc_impa"):
            return None
        try:
            idx = int(v[len("misc_impa"):])
        except Exception:
            return None
        return v if MiscImpassable.VARIANT_MIN <= idx <= MiscImpassable.VARIANT_MAX else None

    def set_variant(self, variant: str) -> None:
        v = self._sanitize_variant(variant) or self.DEFAULT_VARIANT
        self.variant = v
        if v not in MiscImpassable._variant_images:
            MiscImpassable._variant_images[v] = MiscImpassable._load_variant_image(v)

    @classmethod
    def _load_variant_image(cls, variant: str) -> pygame.Surface | None:
        path = f"assets/worldObjects/misc_impa/{variant}.png"
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

        img = MiscImpassable._variant_images.get(self.variant)
        if img is not None:
            surface.blit(img, (self.pos.x - camera_x, self.pos.y - camera_y))
        else:
            pygame.draw.rect(
                surface,
                (60, 30, 30),
                (self.pos.x - camera_x, self.pos.y - camera_y, TILE_SIZE, TILE_SIZE),
            )

