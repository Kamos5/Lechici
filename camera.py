from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import pygame
from pygame.math import Vector2


@dataclass
class Camera:
    view_width: int
    view_height: int
    map_width: int
    map_height: int
    screen_width: int
    screen_height: int
    scroll_margin: int
    scroll_speed: int

    x: float = 0.0  # world-space left of view
    y: float = 0.0  # world-space top of view

    def _center_axis(self, view: int, world: int) -> float:
        """When the world is smaller than the view, keep it centered."""
        return -((view - world) / 2.0)

    def _clamp_axis(self, value: float, view: int, world: int) -> float:
        """Clamp camera axis when world larger than view."""
        if world <= view:
            return self._center_axis(view, world)
        return max(0.0, min(float(world - view), value))

    def update(self, mouse_pos_screen: Tuple[int, int]) -> None:
        """Edge-scroll the camera, but lock an axis if the map fits in view."""
        mx, my = mouse_pos_screen

        # X axis
        if self.map_width <= self.view_width:
            self.x = self._center_axis(self.view_width, self.map_width)
        else:
            if mx < self.scroll_margin:
                self.x -= self.scroll_speed
            elif mx > self.screen_width - self.scroll_margin:
                self.x += self.scroll_speed
            self.x = self._clamp_axis(self.x, self.view_width, self.map_width)

        # Y axis
        if self.map_height <= self.view_height:
            self.y = self._center_axis(self.view_height, self.map_height)
        else:
            if my < self.scroll_margin:
                self.y -= self.scroll_speed
            elif my > self.screen_height - self.scroll_margin:
                self.y += self.scroll_speed
            self.y = self._clamp_axis(self.y, self.view_height, self.map_height)

    def screen_to_world(self, screen_pos: Vector2, *, view_margin_left: int, view_margin_top: int) -> Vector2:
        """Convert screen coords (pixels) to world coords, considering view margins and camera offset."""
        return Vector2(screen_pos.x - view_margin_left + self.x, screen_pos.y - view_margin_top + self.y)

    def world_to_screen(self, world_pos: Vector2, *, view_margin_left: int, view_margin_top: int) -> Vector2:
        """Convert world coords (pixels) to screen coords, considering view margins and camera offset."""
        return Vector2(world_pos.x - self.x + view_margin_left, world_pos.y - self.y + view_margin_top)

    def center_on(self, world_pos: Vector2) -> None:
        """
        Center the camera on a world-space point.
        If near the map border, clamp so the view stays in-bounds
        (i.e. as close to centered as possible).
        """
        target_x = float(world_pos.x) - (self.view_width / 2.0)
        target_y = float(world_pos.y) - (self.view_height / 2.0)

        self.x = self._clamp_axis(target_x, self.view_width, self.map_width)
        self.y = self._clamp_axis(target_y, self.view_height, self.map_height)