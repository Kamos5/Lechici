"""
worldmap_module.py

Importable, parameterized clickable-region map with:
- Background image under translucent tiles
- Cached borders
- Optional initial ownership injection (initial_owner)
- Hover highlight on the region under cursor
- Click rule: you can only change a region if it borders at least one region
  owned by the selected player (territory expansion rule).

Usage:
    from worldmap_module import run_map

    final_owner, meta = run_map(
        player_count=4,
        num_regions=25,
        background_path="background.jpg",
        tile_alpha=128,
        none_alpha=204,
        initial_owner=[...],   # optional, length == num_regions
    )
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple, Dict, Any, Set

import numpy as np
import pygame

Color = Tuple[int, int, int]

DEFAULT_PLAYERS: List[Tuple[str, Color]] = [
    ("R", (255, 0, 0)),
    ("G", (0, 255, 0)),
    ("B", (0, 0, 255)),
    ("C", (0, 255, 255)),
    ("M", (255, 0, 255)),
    ("Y", (255, 255, 0)),
    ("None", (0, 0, 0)),
]


@dataclass
class MapConfig:
    # Window / layout
    width: int = 800
    height: int = 600
    bar_h: int = 80

    # Low-res generation for speed
    map_w: int = 400
    map_h: int = 300
    num_regions: Optional[int] = None
    num_regions_range: Tuple[int, int] = (12, 24)

    # Equal-area tuning knobs
    iterations: int = 35
    weight_lr: float = 0.25
    centroid_pull: float = 0.35
    do_centroids: bool = True

    # Rendering
    border_color: Color = (255, 255, 255)
    tile_alpha: int = 128             # 50% opacity
    none_alpha: int = 204             # 80% opacity for black/None
    background_path: Optional[str] = "background.jpg"
    background_fallback_color: Color = (30, 30, 30)

    # Hover highlight
    hover_alpha_ok: int = 90          # highlight opacity when click is allowed
    hover_alpha_bad: int = 110        # highlight opacity when click is NOT allowed

    # Flicker
    flicker_duration_ms: int = 2000
    flicker_hz: int = 60

    # Players / UI
    players: Optional[List[Tuple[str, Color]]] = None
    player_count: Optional[int] = None    # uses first N of DEFAULT_PLAYERS (excluding None) + None
    show_buttons: bool = True

    # Coloring mode
    # "owner"  -> region colors follow current owner (click changes ownership)
    # "values" -> region colors follow region_values
    color_mode: str = "owner"
    region_values: Optional[Sequence[Any]] = None
    value_to_color: Optional[Callable[[Any], Color]] = None

    # Optional initial ownership by region (length must equal num_regions)
    initial_owner: Optional[Sequence[int]] = None

    # Custom click handler
    # on_region_click(region_idx, selected_player, state_dict) -> None
    on_region_click: Optional[Callable[[int, int, Dict[str, Any]], None]] = None


def _build_players(cfg: MapConfig) -> List[Tuple[str, Color]]:
    if cfg.players is not None:
        return list(cfg.players)

    if cfg.player_count is not None:
        base = [p for p in DEFAULT_PLAYERS if p[0] != "None"]
        n = max(1, int(cfg.player_count))
        chosen = base[:n]
        chosen.append(("None", (0, 0, 0)))
        return chosen

    return list(DEFAULT_PLAYERS)


def _load_background(path: Optional[str], size: Tuple[int, int], fallback: Color) -> pygame.Surface:
    surf = pygame.Surface(size)
    surf.fill(fallback)

    if not path:
        return surf

    try:
        bg_raw = pygame.image.load(path).convert()
        return pygame.transform.smoothscale(bg_raw, size)
    except Exception as e:
        print(f"Could not load {path}: {e}")
        return surf


def _generate_regions(cfg: MapConfig, map_w: int, map_h: int, num_regions: int) -> np.ndarray:
    sites = np.array(
        [(random.randint(0, map_w - 1), random.randint(0, map_h - 1)) for _ in range(num_regions)],
        dtype=np.float32
    )
    weights = np.zeros(num_regions, dtype=np.float32)

    xs = np.arange(map_w, dtype=np.float32)
    ys = np.arange(map_h, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys, indexing="xy")  # (map_h, map_w)

    target_area = (map_w * map_h) / num_regions
    assign: Optional[np.ndarray] = None

    for _ in range(cfg.iterations):
        dx = X[None, :, :] - sites[:, 0][:, None, None]
        dy = Y[None, :, :] - sites[:, 1][:, None, None]
        dist2 = dx * dx + dy * dy

        score = dist2 - weights[:, None, None]
        assign = np.argmin(score, axis=0).astype(np.int32)

        areas = np.bincount(assign.ravel(), minlength=num_regions).astype(np.float32)
        err = (areas - target_area) / target_area
        weights -= cfg.weight_lr * err * (map_w * map_h / num_regions) * 0.10

        if cfg.do_centroids:
            flat = assign.ravel()
            sum_x = np.bincount(flat, weights=X.ravel(), minlength=num_regions).astype(np.float32)
            sum_y = np.bincount(flat, weights=Y.ravel(), minlength=num_regions).astype(np.float32)
            safe_areas = np.maximum(areas, 1.0)
            cx = sum_x / safe_areas
            cy = sum_y / safe_areas
            centroids = np.stack([cx, cy], axis=1)

            sites = sites + cfg.centroid_pull * (centroids - sites)
            sites[:, 0] = np.clip(sites[:, 0], 0, map_w - 1)
            sites[:, 1] = np.clip(sites[:, 1], 0, map_h - 1)

    assert assign is not None
    return assign


def _build_hidden_color_map(assign: np.ndarray, map_w: int, map_h: int, disp_w: int, disp_h: int):
    num_regions = int(assign.max()) + 1

    region_colors: List[Color] = []
    for i in range(num_regions):
        r = 20 + (i * 11) % 236
        g = 40 + (i * 17) % 216
        b = 60 + (i * 13) % 196
        region_colors.append((r, g, b))

    region_colors_arr = np.array(region_colors, dtype=np.uint8)

    rgb_low = region_colors_arr[assign]                      # (map_h, map_w, 3)
    rgb_low_bytes = np.transpose(rgb_low, (1, 0, 2)).copy()  # (map_w, map_h, 3)

    color_map_low = pygame.Surface((map_w, map_h))
    pygame.surfarray.blit_array(color_map_low, rgb_low_bytes)

    color_map = pygame.transform.smoothscale(color_map_low, (disp_w, disp_h))
    return color_map, region_colors


def _precompute_borders(color_map: pygame.Surface, disp_w: int, disp_h: int, border_color: Color) -> pygame.Surface:
    border_surf = pygame.Surface((disp_w, disp_h), flags=pygame.SRCALPHA)
    border_surf.fill((0, 0, 0, 0))

    cp = pygame.PixelArray(color_map)
    bp = pygame.PixelArray(border_surf)

    border_rgba = (*border_color, 255)
    border_px = border_surf.map_rgb(border_rgba)

    for x in range(disp_w):
        for y in range(disp_h):
            c = cp[x][y]
            if (x < disp_w - 1 and cp[x + 1][y] != c) or (y < disp_h - 1 and cp[x][y + 1] != c):
                bp[x][y] = border_px

    del cp
    del bp
    return border_surf


def _build_region_adjacency(assign: np.ndarray, num_regions: int) -> List[Set[int]]:
    """Adjacency based on 4-neighborhood borders in the low-res assign grid."""
    h, w = assign.shape
    adj: List[Set[int]] = [set() for _ in range(num_regions)]
    for y in range(h):
        for x in range(w):
            a = int(assign[y, x])
            if x < w - 1:
                b = int(assign[y, x + 1])
                if a != b:
                    adj[a].add(b)
                    adj[b].add(a)
            if y < h - 1:
                c = int(assign[y + 1, x])
                if a != c:
                    adj[a].add(c)
                    adj[c].add(a)
    return adj


def _build_region_highlight_surface(assign: np.ndarray, region_idx: int, map_w: int, map_h: int, alpha: int) -> pygame.Surface:
    """
    Builds a low-res SRCALPHA surface where the given region is filled with white at given alpha.
    """
    mask = (assign == region_idx)               # (H, W)
    mask_wh = np.transpose(mask, (1, 0))        # (W, H)

    surf = pygame.Surface((map_w, map_h), flags=pygame.SRCALPHA)
    surf.fill((0, 0, 0, 0))

    rgb = pygame.surfarray.pixels3d(surf)
    a = pygame.surfarray.pixels_alpha(surf)

    rgb[:, :, 0] = 255
    rgb[:, :, 1] = 255
    rgb[:, :, 2] = 255
    a[:, :] = (mask_wh.astype(np.uint8) * int(alpha))

    del rgb
    del a
    return surf


def run_map(**kwargs):
    """
    Run the interactive map.

    Returns:
        (owner, meta)
        owner: List[int] ownership by region (length = num_regions)
        meta:  dict with useful internals (assign array, region_colors, etc.)
    """
    cfg = MapConfig(**kwargs)

    pygame.init()

    disp_w, disp_h = cfg.width, cfg.height - cfg.bar_h
    screen = pygame.display.set_mode((cfg.width, cfg.height))
    pygame.display.set_caption("Clickable Map (importable)")

    players = _build_players(cfg)
    player_count = len(players)
    none_idx = player_count - 1

    font = pygame.font.SysFont("arial", 20)
    big_font = pygame.font.SysFont("arial", 28)

    # Loading splash
    loading_text = big_font.render("Generating map... please wait", True, (220, 220, 255))
    screen.fill((10, 10, 30))
    screen.blit(loading_text, (cfg.width // 2 - 210, cfg.height // 2 - 20))
    pygame.display.flip()

    # Regions
    if cfg.num_regions is None:
        lo, hi = cfg.num_regions_range
        num_regions = random.randint(int(lo), int(hi))
    else:
        num_regions = int(cfg.num_regions)

    assign = _generate_regions(cfg, cfg.map_w, cfg.map_h, num_regions)

    # Hidden ID color map for click detection
    color_map, region_colors = _build_hidden_color_map(assign, cfg.map_w, cfg.map_h, disp_w, disp_h)

    # Cached borders
    border_surf = _precompute_borders(color_map, disp_w, disp_h, cfg.border_color)

    # Background
    background_surf = _load_background(cfg.background_path, (disp_w, disp_h), cfg.background_fallback_color)

    # Adjacency for click rules
    adjacency = _build_region_adjacency(assign, num_regions)

    # State: owner
    owner: List[int] = [none_idx] * num_regions
    if cfg.initial_owner is not None:
        if len(cfg.initial_owner) != num_regions:
            raise ValueError(f"initial_owner length {len(cfg.initial_owner)} must equal num_regions {num_regions}")
        owner = [int(x) for x in cfg.initial_owner]
        bad = [i for i, v in enumerate(owner) if v < 0 or v >= player_count]
        if bad:
            raise ValueError(f"initial_owner contains out-of-range indices at regions: {bad[:10]}")
    else:
        # Default: ensure at least one region per player (including None)
        for p in range(player_count):
            owner[p % num_regions] = p

    # Values mode
    region_values = None
    if cfg.color_mode == "values":
        if cfg.region_values is None:
            region_values = owner[:]
        else:
            if len(cfg.region_values) != num_regions:
                raise ValueError(f"region_values length {len(cfg.region_values)} must equal num_regions {num_regions}")
            region_values = list(cfg.region_values)

    selected_player = 0

    # Flicker: region_idx -> dict
    flickers: Dict[int, Dict[str, int]] = {}
    half_period_ms = max(1, round(1000 / (cfg.flicker_hz * 2)))

    # Rendering surfaces
    fill_surf = pygame.Surface((disp_w, disp_h), flags=pygame.SRCALPHA)
    needs_redraw = True

    # Hover highlight cache
    hover_region: Optional[int] = None
    hover_ok: bool = False
    hover_hi_scaled_ok: Optional[pygame.Surface] = None
    hover_hi_scaled_bad: Optional[pygame.Surface] = None

    def compute_effective_owner(now_ms: int) -> List[int]:
        eff = owner[:]
        for ridx, st in list(flickers.items()):
            age = now_ms - st["start_ms"]
            if age >= cfg.flicker_duration_ms:
                flickers.pop(ridx, None)
                continue
            show_new = ((age // half_period_ms) % 2 == 0)
            eff[ridx] = st["new_owner"] if show_new else st["old_owner"]
        return eff

    def value_to_rgba(v: Any) -> Tuple[int, int, int, int]:
        if cfg.value_to_color is not None:
            r, g, b = cfg.value_to_color(v)
        else:
            idx = int(v)
            r, g, b = players[idx][1]

        alpha = cfg.none_alpha if (r, g, b) == (0, 0, 0) else cfg.tile_alpha
        return (int(r), int(g), int(b), int(alpha))

    def rebuild_fill_surface(now_ms: int):
        nonlocal fill_surf

        if cfg.color_mode == "owner":
            eff_owner = compute_effective_owner(now_ms)
            region_to_rgb = np.array([players[eff_owner[i]][1] for i in range(num_regions)], dtype=np.uint8)
            region_to_a = np.array(
                [cfg.none_alpha if eff_owner[i] == none_idx else cfg.tile_alpha for i in range(num_regions)],
                dtype=np.uint8
            )
        elif cfg.color_mode == "values":
            assert region_values is not None
            rgba = np.array([value_to_rgba(region_values[i]) for i in range(num_regions)], dtype=np.uint8)
            region_to_rgb = rgba[:, :3]
            region_to_a = rgba[:, 3]
        else:
            raise ValueError("color_mode must be 'owner' or 'values'")

        low_rgb = region_to_rgb[assign]  # (map_h, map_w, 3)
        low_a = region_to_a[assign]      # (map_h, map_w)

        low = pygame.Surface((cfg.map_w, cfg.map_h), flags=pygame.SRCALPHA)

        rgb_bytes = np.transpose(low_rgb, (1, 0, 2)).copy()  # (map_w, map_h, 3)
        pygame.surfarray.blit_array(low, rgb_bytes)

        a_bytes = np.transpose(low_a, (1, 0)).copy()         # (map_w, map_h)
        alpha_view = pygame.surfarray.pixels_alpha(low)
        alpha_view[:, :] = a_bytes
        del alpha_view

        fill_surf = pygame.transform.smoothscale(low, (disp_w, disp_h))

    def draw_buttons():
        if not cfg.show_buttons:
            return

        pygame.draw.rect(screen, (25, 25, 25), pygame.Rect(0, disp_h, cfg.width, cfg.bar_h))

        pad = 10
        n = player_count
        btn_w = (cfg.width - pad * (n + 1)) // n
        btn_h = cfg.bar_h - 2 * pad

        for i, (label, col) in enumerate(players):
            x = pad + i * (btn_w + pad)
            y = disp_h + pad
            rect = pygame.Rect(x, y, btn_w, btn_h)

            pygame.draw.rect(screen, col, rect)
            pygame.draw.rect(
                screen,
                (255, 255, 255) if i == selected_player else (140, 140, 140),
                rect,
                4 if i == selected_player else 2,
            )

            text_col = (0, 0, 0) if sum(col) > 380 else (255, 255, 255)
            t = font.render(label, True, text_col)
            screen.blit(t, (rect.centerx - t.get_width() // 2, rect.centery - t.get_height() // 2))

    def button_index_at(pos):
        if not cfg.show_buttons:
            return None

        x, y = pos
        if y < disp_h:
            return None

        pad = 10
        n = player_count
        btn_w = (cfg.width - pad * (n + 1)) // n
        btn_h = cfg.bar_h - 2 * pad

        for i in range(n):
            bx = pad + i * (btn_w + pad)
            by = disp_h + pad
            if pygame.Rect(bx, by, btn_w, btn_h).collidepoint(pos):
                return i
        return None

    def can_click_region(region_idx: int, selected: int) -> bool:
        # Rule: only allow if region borders at least one region owned by selected player.
        # Also allow clicking your own region (no-op or re-affirm).
        if owner[region_idx] == selected:
            return True
        for nb in adjacency[region_idx]:
            if owner[nb] == selected:
                return True
        return False

    clock = pygame.time.Clock()
    running = True

    state: Dict[str, Any] = {
        "owner": owner,
        "region_values": region_values,
        "players": players,
        "num_regions": num_regions,
    }

    while running:
        now_ms = pygame.time.get_ticks()

        # --- Hover detection & highlight caching (every frame) ---
        mx, my = pygame.mouse.get_pos()
        new_hover = None
        new_hover_ok = False

        if 0 <= mx < disp_w and 0 <= my < disp_h:
            hidden_color = color_map.get_at((mx, my))[:3]
            try:
                new_hover = region_colors.index(hidden_color)
                new_hover_ok = can_click_region(new_hover, selected_player) if new_hover is not None else False
            except ValueError:
                new_hover = None
                new_hover_ok = False

        if new_hover != hover_region:
            hover_region = new_hover
            hover_ok = new_hover_ok
            hover_hi_scaled_ok = None
            hover_hi_scaled_bad = None

            if hover_region is not None:
                low_ok = _build_region_highlight_surface(
                    assign, hover_region, cfg.map_w, cfg.map_h, alpha=cfg.hover_alpha_ok
                )
                low_bad = _build_region_highlight_surface(
                    assign, hover_region, cfg.map_w, cfg.map_h, alpha=cfg.hover_alpha_bad
                )
                hover_hi_scaled_ok = pygame.transform.smoothscale(low_ok, (disp_w, disp_h))
                hover_hi_scaled_bad = pygame.transform.smoothscale(low_bad, (disp_w, disp_h))
        else:
            hover_ok = new_hover_ok

        # --- Events ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                pos = event.pos

                bi = button_index_at(pos)
                if bi is not None:
                    selected_player = bi
                    needs_redraw = True
                else:
                    mx2, my2 = pos
                    if 0 <= mx2 < disp_w and 0 <= my2 < disp_h:
                        hidden_color = color_map.get_at((mx2, my2))[:3]
                        try:
                            region_idx = region_colors.index(hidden_color)
                        except ValueError:
                            region_idx = None

                        if region_idx is not None:
                            # Click restriction:
                            if not can_click_region(region_idx, selected_player):
                                continue

                            if cfg.on_region_click is not None:
                                cfg.on_region_click(region_idx, selected_player, state)
                                needs_redraw = True
                            else:
                                if cfg.color_mode == "owner":
                                    old_owner = owner[region_idx]
                                    new_owner = selected_player
                                    if old_owner != new_owner:
                                        owner[region_idx] = new_owner
                                        flickers[region_idx] = {
                                            "start_ms": now_ms,
                                            "old_owner": old_owner,
                                            "new_owner": new_owner,
                                        }
                                    needs_redraw = True
                                else:
                                    assert region_values is not None
                                    region_values[region_idx] = selected_player
                                    needs_redraw = True

        # Flicker requires redraw each frame while active (owner mode only)
        if cfg.color_mode == "owner" and flickers:
            needs_redraw = True

        if needs_redraw:
            rebuild_fill_surface(now_ms)
            needs_redraw = False

        # --- Draw ---
        screen.blit(background_surf, (0, 0))
        screen.blit(fill_surf, (0, 0))
        screen.blit(border_surf, (0, 0))

        # Hover highlight (white when OK, red-tinted when blocked)
        if hover_region is not None:
            if hover_ok and hover_hi_scaled_ok is not None:
                screen.blit(hover_hi_scaled_ok, (0, 0))
            elif (not hover_ok) and hover_hi_scaled_bad is not None:
                tmp = hover_hi_scaled_bad.copy()
                # tint red-ish
                tmp.fill((255, 90, 90, 0), special_flags=pygame.BLEND_RGBA_MULT)
                screen.blit(tmp, (0, 0))

        draw_buttons()
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

    meta = {
        "assign": assign,                # low-res region assignment (map_h, map_w)
        "num_regions": num_regions,
        "players": players,
        "region_colors": region_colors,  # hidden colors for click detection
        "adjacency": adjacency,
    }
    return owner, meta
