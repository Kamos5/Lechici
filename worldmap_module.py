"""
worldmap_module.py

Changes requested:
1) dummy_method now always returns True:
    def dummy_method(attacker: int, region_idx: int) -> bool:
        return random.random() < 1

2) Defend message is shown AFTER the animation of changing tile colour:
   - When a deferred grant targets a tile owned by the selected player:
       a) first play an "attack preview" animation (tile flickers to attacker)
       b) after the preview ends, show defend prompt
       c) YES => revert tile to selected player (revert flicker)
          NO  => confirm tile to attacker (confirm flicker)

Other behavior preserved.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Dict, Any, Set
from cards import pick_round_card
import numpy as np
import pygame

from main import run_game

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
    width: int = 800
    height: int = 600
    bar_h: int = 80

    map_w: int = 400
    map_h: int = 300
    num_regions: Optional[int] = None
    num_regions_range: Tuple[int, int] = (12, 24)

    iterations: int = 35
    weight_lr: float = 0.25
    centroid_pull: float = 0.35
    do_centroids: bool = True

    border_color: Color = (255, 255, 255)
    tile_alpha: int = 128
    none_alpha: int = 204
    background_path: Optional[str] = "background.jpg"
    background_fallback_color: Color = (30, 30, 30)

    hover_alpha_ok: int = 90
    hover_alpha_bad: int = 110

    flicker_duration_ms: int = 2000
    flicker_hz: int = 60

    players: Optional[List[Tuple[str, Color]]] = None
    player_count: Optional[int] = None
    show_buttons: bool = True

    initial_owner: Optional[Sequence[int]] = None

    show_arrows: bool = True
    arrow_alpha: int = 210
    arrow_width: int = 2
    arrow_curvature_px: float = 40.0
    arrow_head_len_px: float = 12.0
    arrow_head_angle_deg: float = 28.0
    arrow_segments: int = 22

    battle_delay_ms: int = 1000
    reward_extra_tiles: int = 1
    between_anims_delay_ms: int = 150
    revert_flicker_ms: int = 900

    hud_h: int = 28
    hud_bg_alpha: int = 140

    defend_box_alpha: int = 210
    defend_box_w: int = 560
    defend_box_h: int = 160

    # NEW: attack preview duration before showing defend prompt
    defend_preview_ms: int = 900
    defend_confirm_ms: int = 900


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
        dtype=np.float32,
    )
    weights = np.zeros(num_regions, dtype=np.float32)

    xs = np.arange(map_w, dtype=np.float32)
    ys = np.arange(map_h, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys, indexing="xy")

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
            safe = np.maximum(areas, 1.0)
            cx = sum_x / safe
            cy = sum_y / safe
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
    rgb_low = region_colors_arr[assign]
    rgb_low_bytes = np.transpose(rgb_low, (1, 0, 2)).copy()

    color_map_low = pygame.Surface((map_w, map_h))
    pygame.surfarray.blit_array(color_map_low, rgb_low_bytes)
    color_map = pygame.transform.smoothscale(color_map_low, (disp_w, disp_h))
    return color_map, region_colors


def _precompute_borders(color_map: pygame.Surface, disp_w: int, disp_h: int, border_color: Color) -> pygame.Surface:
    border_surf = pygame.Surface((disp_w, disp_h), flags=pygame.SRCALPHA)
    border_surf.fill((0, 0, 0, 0))

    cp = pygame.PixelArray(color_map)
    bp = pygame.PixelArray(border_surf)
    border_px = border_surf.map_rgb((*border_color, 255))

    for x in range(disp_w):
        for y in range(disp_h):
            c = cp[x][y]
            if (x < disp_w - 1 and cp[x + 1][y] != c) or (y < disp_h - 1 and cp[x][y + 1] != c):
                bp[x][y] = border_px

    del cp
    del bp
    return border_surf


def _build_region_adjacency(assign: np.ndarray, num_regions: int) -> List[Set[int]]:
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


def _compute_centroids(assign: np.ndarray, num_regions: int) -> np.ndarray:
    h, w = assign.shape
    ys, xs = np.indices((h, w))
    flat = assign.ravel()
    areas = np.bincount(flat, minlength=num_regions).astype(np.float32)
    sum_x = np.bincount(flat, weights=xs.ravel(), minlength=num_regions).astype(np.float32)
    sum_y = np.bincount(flat, weights=ys.ravel(), minlength=num_regions).astype(np.float32)
    safe = np.maximum(areas, 1.0)
    cx = sum_x / safe
    cy = sum_y / safe
    return np.stack([cx, cy], axis=1)


def _compute_contact_points(assign: np.ndarray, num_regions: int) -> Dict[Tuple[int, int], Tuple[float, float]]:
    h, w = assign.shape
    contact: Dict[Tuple[int, int], Tuple[float, float]] = {}
    for y in range(h):
        for x in range(w):
            a = int(assign[y, x])
            if x < w - 1:
                b = int(assign[y, x + 1])
                if a != b:
                    key = (a, b) if a < b else (b, a)
                    if key not in contact:
                        contact[key] = (x + 0.5, y + 0.0)
            if y < h - 1:
                c = int(assign[y + 1, x])
                if a != c:
                    key = (a, c) if a < c else (c, a)
                    if key not in contact:
                        contact[key] = (x + 0.0, y + 0.5)
    return contact


def _build_region_highlight_surface(assign: np.ndarray, region_idx: int, map_w: int, map_h: int, alpha: int) -> pygame.Surface:
    mask = (assign == region_idx)
    mask_wh = np.transpose(mask, (1, 0))
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


def _bezier_points(p0, p1, p2, n: int):
    pts = []
    for i in range(n + 1):
        t = i / n
        u = 1.0 - t
        x = u * u * p0[0] + 2 * u * t * p1[0] + t * t * p2[0]
        y = u * u * p0[1] + 2 * u * t * p1[1] + t * t * p2[1]
        pts.append((x, y))
    return pts


def _draw_curved_arrow(
    surf: pygame.Surface,
    color_rgba: Tuple[int, int, int, int],
    p0: Tuple[float, float],
    p2: Tuple[float, float],
    curvature_px: float,
    width: int,
    head_len: float,
    head_angle_deg: float,
    segments: int,
):
    dx = p2[0] - p0[0]
    dy = p2[1] - p0[1]
    dist = (dx * dx + dy * dy) ** 0.5
    if dist < 1.0:
        return

    nx = -dy / dist
    ny = dx / dist
    mx = (p0[0] + p2[0]) * 0.5
    my = (p0[1] + p2[1]) * 0.5
    p1 = (mx + nx * curvature_px, my + ny * curvature_px)

    pts = _bezier_points(p0, p1, p2, max(6, int(segments)))
    pygame.draw.aalines(surf, color_rgba, False, pts)

    if width > 1:
        mid_i = len(pts) // 2
        if mid_i >= 2:
            ddx = pts[mid_i + 1][0] - pts[mid_i - 1][0]
            ddy = pts[mid_i + 1][1] - pts[mid_i - 1][1]
            d = (ddx * ddx + ddy * ddy) ** 0.5 or 1.0
            px = -ddy / d
            py = ddx / d
        else:
            px, py = nx, ny
        for k in range(1, width):
            off = (k - (width - 1) / 2.0)
            pts2 = [(x + px * off, y + py * off) for (x, y) in pts]
            pygame.draw.aalines(surf, color_rgba, False, pts2)

    ex, ey = pts[-1]
    px_, py_ = pts[-2]
    vx = ex - px_
    vy = ey - py_
    vd = (vx * vx + vy * vy) ** 0.5 or 1.0
    vx /= vd
    vy /= vd

    import math
    ang = math.radians(head_angle_deg)
    ca, sa = math.cos(ang), math.sin(ang)

    lx = vx * ca - vy * sa
    ly = vx * sa + vy * ca
    rx = vx * ca + vy * sa
    ry = -vx * sa + vy * ca

    left = (ex - lx * head_len, ey - ly * head_len)
    right = (ex - rx * head_len, ey - ry * head_len)

    pygame.draw.aaline(surf, color_rgba, (ex, ey), left)
    pygame.draw.aaline(surf, color_rgba, (ex, ey), right)


def run_map(**kwargs):
    cfg = MapConfig(**kwargs)

    pygame.init()
    disp_w, disp_h = cfg.width, cfg.height - cfg.bar_h
    screen = pygame.display.set_mode((cfg.width, cfg.height))
    pygame.display.set_caption("Clickable Map (defend after anim)")

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
    color_map, region_colors = _build_hidden_color_map(assign, cfg.map_w, cfg.map_h, disp_w, disp_h)
    border_surf = _precompute_borders(color_map, disp_w, disp_h, cfg.border_color)
    background_surf = _load_background(cfg.background_path, (disp_w, disp_h), cfg.background_fallback_color)

    adjacency = _build_region_adjacency(assign, num_regions)
    centroids = _compute_centroids(assign, num_regions)
    contact_points = _compute_contact_points(assign, num_regions)

    # Ownership
    owner: List[int] = [none_idx] * num_regions
    if cfg.initial_owner is not None:
        if len(cfg.initial_owner) != num_regions:
            raise ValueError(f"initial_owner length {len(cfg.initial_owner)} must equal num_regions {num_regions}")
        owner = [int(x) for x in cfg.initial_owner]
        for v in owner:
            if v < 0 or v >= player_count:
                raise ValueError("initial_owner contains out-of-range player index")
    else:
        for p in range(player_count):
            owner[p % num_regions] = p

    # Flickers
    flickers: Dict[int, Dict[str, int]] = {}
    half_period_ms = max(1, round(1000 / (cfg.flicker_hz * 2)))

    fill_surf = pygame.Surface((disp_w, disp_h), flags=pygame.SRCALPHA)
    needs_redraw = True

    # Hover cache
    hover_region: Optional[int] = None
    hover_ok: bool = False
    hover_hi_scaled_ok: Optional[pygame.Surface] = None
    hover_hi_scaled_bad: Optional[pygame.Surface] = None

    # Phase machine
    PHASE_IDLE = "IDLE"
    PHASE_BATTLE_FLICKER = "BATTLE_FLICKER"
    PHASE_BATTLE_WAIT = "BATTLE_WAIT"
    PHASE_PLAY_QUEUE = "PLAY_QUEUE"
    PHASE_DEFEND_PROMPT = "DEFEND_PROMPT"
    PHASE_DEFEND_PREVIEW = "DEFEND_PREVIEW"
    phase = PHASE_IDLE

    battle_ctx: Dict[str, Any] = {}

    # Animation queue:
    # - {"kind":"region", "region":int, "to":int, "dur":int}
    # - {"kind":"deferred_grant", "player":int, "dur":int}
    anim_queue: List[Dict[str, Any]] = []
    current_anim: Optional[Dict[str, Any]] = None
    between_anim_wait_until: int = 0

    # Defend context
    defend_ctx: Optional[Dict[str, Any]] = None  # attacker, target, dur, prev_owner

    # Round counter
    round_no = 1

    # Selected player
    selected_player = 0

    # (1) requested dummy_method
    def dummy_method(attacker: int, region_idx: int) -> bool:
        return run_game() == 1
        # """Replace with your real game logic."""
        # return random.random() < 1

    def player_has_tiles(p: int) -> bool:
        return sum(1 for o in owner if o == p) > 0

    def compute_effective_owner(now_ms: int) -> List[int]:
        eff = owner[:]
        for ridx, st in list(flickers.items()):
            age = now_ms - st["start_ms"]
            if age >= st.get("duration_ms", cfg.flicker_duration_ms):
                flickers.pop(ridx, None)
                continue
            show_new = ((age // half_period_ms) % 2 == 0)
            eff[ridx] = st["new_owner"] if show_new else st["old_owner"]
        return eff

    def rebuild_fill_surface(now_ms: int):
        nonlocal fill_surf
        eff_owner = compute_effective_owner(now_ms)

        region_to_rgb = np.array([players[eff_owner[i]][1] for i in range(num_regions)], dtype=np.uint8)
        region_to_a = np.array(
            [cfg.none_alpha if eff_owner[i] == none_idx else cfg.tile_alpha for i in range(num_regions)],
            dtype=np.uint8,
        )

        low_rgb = region_to_rgb[assign]
        low_a = region_to_a[assign]

        low = pygame.Surface((cfg.map_w, cfg.map_h), flags=pygame.SRCALPHA)
        rgb_bytes = np.transpose(low_rgb, (1, 0, 2)).copy()
        pygame.surfarray.blit_array(low, rgb_bytes)

        a_bytes = np.transpose(low_a, (1, 0)).copy()
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
        if owner[region_idx] == selected:
            return True
        for nb in adjacency[region_idx]:
            if owner[nb] == selected:
                return True
        return False

    def to_disp(pt_low: Tuple[float, float]) -> Tuple[float, float]:
        return (pt_low[0] * (disp_w / cfg.map_w), pt_low[1] * (disp_h / cfg.map_h))

    def draw_arrows(now_ms: int):
        if not cfg.show_arrows:
            return

        eff_owner = compute_effective_owner(now_ms)
        targets: Dict[int, int] = {}

        cent_disp = np.zeros_like(centroids, dtype=np.float32)
        cent_disp[:, 0] = centroids[:, 0] * (disp_w / cfg.map_w)
        cent_disp[:, 1] = centroids[:, 1] * (disp_h / cfg.map_h)

        for src in range(num_regions):
            if eff_owner[src] != selected_player:
                continue
            for tgt in adjacency[src]:
                if eff_owner[tgt] == selected_player:
                    continue
                if tgt not in targets:
                    targets[tgt] = src
                else:
                    prev_src = targets[tgt]
                    dx1 = cent_disp[src, 0] - cent_disp[tgt, 0]
                    dy1 = cent_disp[src, 1] - cent_disp[tgt, 1]
                    dx2 = cent_disp[prev_src, 0] - cent_disp[tgt, 0]
                    dy2 = cent_disp[prev_src, 1] - cent_disp[tgt, 1]
                    if (dx1 * dx1 + dy1 * dy1) < (dx2 * dx2 + dy2 * dy2):
                        targets[tgt] = src

        if not targets:
            return

        arrows_surf = pygame.Surface((disp_w, disp_h), flags=pygame.SRCALPHA)
        arrows_surf.fill((0, 0, 0, 0))
        base_rgb = players[selected_player][1]
        arrow_col = (base_rgb[0], base_rgb[1], base_rgb[2], int(cfg.arrow_alpha))

        for tgt, src in targets.items():
            key = (src, tgt) if src < tgt else (tgt, src)
            if key in contact_points:
                cx, cy = contact_points[key]
                contact_disp = to_disp((cx, cy))
                sx, sy = float(cent_disp[src, 0]), float(cent_disp[src, 1])
                tx, ty = float(cent_disp[tgt, 0]), float(cent_disp[tgt, 1])
                p0 = (sx * 0.55 + contact_disp[0] * 0.45, sy * 0.55 + contact_disp[1] * 0.45)
                p2 = (tx * 0.55 + contact_disp[0] * 0.45, ty * 0.55 + contact_disp[1] * 0.45)
            else:
                p0 = (float(cent_disp[src, 0]), float(cent_disp[src, 1]))
                p2 = (float(cent_disp[tgt, 0]), float(cent_disp[tgt, 1]))

            _draw_curved_arrow(
                arrows_surf,
                arrow_col,
                p0,
                p2,
                curvature_px=float(cfg.arrow_curvature_px),
                width=int(cfg.arrow_width),
                head_len=float(cfg.arrow_head_len_px),
                head_angle_deg=float(cfg.arrow_head_angle_deg),
                segments=int(cfg.arrow_segments),
            )

        screen.blit(arrows_surf, (0, 0))

    def compute_border_candidates(snapshot_owner: List[int], player: int) -> List[int]:
        cand: Set[int] = set()
        for r in range(num_regions):
            if snapshot_owner[r] != player:
                continue
            for nb in adjacency[r]:
                if snapshot_owner[nb] != player:
                    cand.add(int(nb))
        return list(cand)

    def eligible_adjacent_tiles(player: int) -> List[int]:
        c: Set[int] = set()
        for r in range(num_regions):
            if owner[r] != player:
                continue
            for nb in adjacency[r]:
                if owner[nb] != player:
                    c.add(int(nb))
        return list(c)

    def pick_adjacent_tile_for_player(player: int) -> Optional[int]:
        candidates = eligible_adjacent_tiles(player)
        if not candidates:
            return None
        blacks = [i for i in candidates if owner[i] == none_idx]
        if blacks:
            return random.choice(blacks)
        return random.choice(candidates)

    # ---------- Queue helpers ----------
    def enqueue_anim(region: int, to_owner: int, duration_ms: int):
        anim_queue.append({"kind": "region", "region": int(region), "to": int(to_owner), "dur": int(duration_ms)})

    def enqueue_deferred_grant(for_player: int, duration_ms: int):
        anim_queue.append({"kind": "deferred_grant", "player": int(for_player), "dur": int(duration_ms)})

    def schedule_automated_player_grants(attacker: int, none_player: int):
        for p in range(player_count):
            if p == attacker or p == none_player:
                continue
            enqueue_deferred_grant(p, int(cfg.flicker_duration_ms))

    def _start_region_anim(now_ms: int, ridx: int, to_owner: int, dur: int):
        nonlocal current_anim
        current_anim = {"kind": "region", "region": int(ridx), "to": int(to_owner), "dur": int(dur), "start_ms": int(now_ms)}
        old_o = owner[ridx]
        flickers[ridx] = {
            "start_ms": int(now_ms),
            "duration_ms": int(dur),
            "old_owner": int(old_o),
            "new_owner": int(to_owner),
        }

    def start_next_anim(now_ms: int):
        """
        Deferred grants are resolved now.
        If target is owned by selected player:
          - run DEFEND_PREVIEW (tile flickers to attacker)
          - AFTER preview ends -> show defend prompt
        """
        nonlocal current_anim, phase, defend_ctx

        if current_anim is not None or phase in (PHASE_DEFEND_PROMPT, PHASE_DEFEND_PREVIEW):
            return

        while anim_queue:
            item = anim_queue.pop(0)

            if item.get("kind") == "region":
                _start_region_anim(now_ms, int(item["region"]), int(item["to"]), int(item["dur"]))
                return

            if item.get("kind") == "deferred_grant":
                attacker = int(item["player"])
                if not player_has_tiles(attacker):
                    continue

                tgt = pick_adjacent_tile_for_player(attacker)
                if tgt is None:
                    continue

                # If attacking selected player's tile, preview first then prompt
                if owner[tgt] == selected_player and attacker != selected_player:
                    defend_ctx = {
                        "attacker": attacker,
                        "target": int(tgt),
                        "prev_owner": int(owner[tgt]),
                        "preview_start_ms": int(now_ms),
                    }
                    # Start preview flicker (without committing owner change)
                    flickers[tgt] = {
                        "start_ms": int(now_ms),
                        "duration_ms": int(cfg.defend_preview_ms),
                        "old_owner": int(owner[tgt]),
                        "new_owner": int(attacker),
                    }
                    phase = PHASE_DEFEND_PREVIEW
                    return

                # normal attack
                _start_region_anim(now_ms, int(tgt), attacker, int(item["dur"]))
                return

        # none runnable

    def update_anim(now_ms: int):
        nonlocal current_anim, between_anim_wait_until, needs_redraw
        if current_anim is None:
            return
        ridx = int(current_anim["region"])
        age = now_ms - int(current_anim["start_ms"])
        if age >= int(current_anim["dur"]):
            owner[ridx] = int(current_anim["to"])
            flickers.pop(ridx, None)
            current_anim = None
            between_anim_wait_until = now_ms + int(cfg.between_anims_delay_ms)
            needs_redraw = True

    # ---------- HUD + defend prompt ----------
    def draw_hud():
        hud = pygame.Surface((cfg.width, cfg.hud_h), flags=pygame.SRCALPHA)
        hud.fill((0, 0, 0, int(cfg.hud_bg_alpha)))

        sel_label, sel_col = players[selected_player]
        text = f"Round: {round_no}    Selected: {sel_label}"
        t = font.render(text, True, (240, 240, 240))
        hud.blit(t, (10, 4))

        sw = pygame.Rect(10 + t.get_width() + 12, 6, 16, 16)
        pygame.draw.rect(hud, sel_col, sw)
        pygame.draw.rect(hud, (255, 255, 255), sw, 1)

        screen.blit(hud, (0, 0))

    def draw_defend_prompt():
        w, h = cfg.defend_box_w, cfg.defend_box_h
        box = pygame.Surface((w, h), flags=pygame.SRCALPHA)
        box.fill((0, 0, 0, int(cfg.defend_box_alpha)))
        pygame.draw.rect(box, (255, 255, 255), pygame.Rect(0, 0, w, h), 2)

        if defend_ctx is None:
            msg = "Defend this tile?"
        else:
            atk_label = players[int(defend_ctx["attacker"])][0]
            msg = f"{atk_label} attacked your tile. Defend it?"

        title = big_font.render(msg, True, (240, 240, 240))
        box.blit(title, (w // 2 - title.get_width() // 2, 20))

        btn_w, btn_h = 160, 44
        yes_local = pygame.Rect(w // 2 - btn_w - 20, h - 70, btn_w, btn_h)
        no_local = pygame.Rect(w // 2 + 20, h - 70, btn_w, btn_h)

        pygame.draw.rect(box, (30, 160, 30), yes_local)
        pygame.draw.rect(box, (200, 60, 60), no_local)
        pygame.draw.rect(box, (255, 255, 255), yes_local, 2)
        pygame.draw.rect(box, (255, 255, 255), no_local, 2)

        yes_txt = font.render("YES (Y)", True, (255, 255, 255))
        no_txt = font.render("NO (N)", True, (255, 255, 255))
        box.blit(yes_txt, (yes_local.centerx - yes_txt.get_width() // 2, yes_local.centery - yes_txt.get_height() // 2))
        box.blit(no_txt, (no_local.centerx - no_txt.get_width() // 2, no_local.centery - no_txt.get_height() // 2))

        x = cfg.width // 2 - w // 2
        y = disp_h // 2 - h // 2
        screen.blit(box, (x, y))
        return (
            pygame.Rect(x + yes_local.x, y + yes_local.y, yes_local.w, yes_local.h),
            pygame.Rect(x + no_local.x, y + no_local.y, no_local.w, no_local.h),
        )

    clock = pygame.time.Clock()
    running = True

    while running:
        now_ms = pygame.time.get_ticks()

        # Hover
        mx, my = pygame.mouse.get_pos()
        new_hover = None
        if 0 <= mx < disp_w and 0 <= my < disp_h:
            hidden_color = color_map.get_at((mx, my))[:3]
            try:
                new_hover = region_colors.index(hidden_color)
            except ValueError:
                new_hover = None

        new_hover_ok = (new_hover is not None) and (phase == PHASE_IDLE) and can_click_region(new_hover, selected_player)

        if new_hover != hover_region:
            hover_region = new_hover
            hover_ok = new_hover_ok
            hover_hi_scaled_ok = None
            hover_hi_scaled_bad = None
            if hover_region is not None:
                low_ok = _build_region_highlight_surface(assign, hover_region, cfg.map_w, cfg.map_h, cfg.hover_alpha_ok)
                low_bad = _build_region_highlight_surface(assign, hover_region, cfg.map_w, cfg.map_h, cfg.hover_alpha_bad)
                hover_hi_scaled_ok = pygame.transform.smoothscale(low_ok, (disp_w, disp_h))
                hover_hi_scaled_bad = pygame.transform.smoothscale(low_bad, (disp_w, disp_h))
        else:
            hover_ok = new_hover_ok

        # Events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # Defend prompt input
            if phase == PHASE_DEFEND_PROMPT:
                if event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_y, pygame.K_RETURN):
                        # DEFEND: revert to selected
                        if defend_ctx is not None:
                            tgt = int(defend_ctx["target"])
                            flickers[tgt] = {
                                "start_ms": int(now_ms),
                                "duration_ms": int(cfg.defend_confirm_ms),
                                "old_owner": int(defend_ctx["attacker"]),
                                "new_owner": int(selected_player),
                            }
                            owner[tgt] = int(selected_player)  # keep ownership
                        defend_ctx = None
                        phase = PHASE_PLAY_QUEUE
                        between_anim_wait_until = now_ms + int(cfg.between_anims_delay_ms)
                        needs_redraw = True

                    elif event.key in (pygame.K_n,):
                        # ALLOW: commit to attacker
                        if defend_ctx is not None:
                            tgt = int(defend_ctx["target"])
                            attacker = int(defend_ctx["attacker"])
                            flickers[tgt] = {
                                "start_ms": int(now_ms),
                                "duration_ms": int(cfg.defend_confirm_ms),
                                "old_owner": int(selected_player),
                                "new_owner": int(attacker),
                            }
                            owner[tgt] = int(attacker)
                        defend_ctx = None
                        phase = PHASE_PLAY_QUEUE
                        between_anim_wait_until = now_ms + int(cfg.between_anims_delay_ms)
                        needs_redraw = True

                    elif event.key in (pygame.K_ESCAPE,):
                        # default defend
                        if defend_ctx is not None:
                            tgt = int(defend_ctx["target"])
                            flickers[tgt] = {
                                "start_ms": int(now_ms),
                                "duration_ms": int(cfg.defend_confirm_ms),
                                "old_owner": int(defend_ctx["attacker"]),
                                "new_owner": int(selected_player),
                            }
                            owner[tgt] = int(selected_player)
                        defend_ctx = None
                        phase = PHASE_PLAY_QUEUE
                        between_anim_wait_until = now_ms + int(cfg.between_anims_delay_ms)
                        needs_redraw = True

                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    # We'll use actual rects from draw step; simplest: decide by x position
                    if event.pos[0] < cfg.width // 2:
                        # YES defend
                        if defend_ctx is not None:
                            tgt = int(defend_ctx["target"])
                            flickers[tgt] = {
                                "start_ms": int(now_ms),
                                "duration_ms": int(cfg.defend_confirm_ms),
                                "old_owner": int(defend_ctx["attacker"]),
                                "new_owner": int(selected_player),
                            }
                            owner[tgt] = int(selected_player)
                        defend_ctx = None
                    else:
                        # NO allow
                        if defend_ctx is not None:
                            tgt = int(defend_ctx["target"])
                            attacker = int(defend_ctx["attacker"])
                            flickers[tgt] = {
                                "start_ms": int(now_ms),
                                "duration_ms": int(cfg.defend_confirm_ms),
                                "old_owner": int(selected_player),
                                "new_owner": int(attacker),
                            }
                            owner[tgt] = int(attacker)
                        defend_ctx = None

                    phase = PHASE_PLAY_QUEUE
                    between_anim_wait_until = now_ms + int(cfg.between_anims_delay_ms)
                    needs_redraw = True

                continue

            # Normal UI
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                pos = event.pos

                # Buttons anytime
                bi = button_index_at(pos)
                if bi is not None:
                    selected_player = bi
                    needs_redraw = True
                    continue

                # Tile clicks only when idle
                if phase != PHASE_IDLE:
                    continue

                mx2, my2 = pos
                if not (0 <= mx2 < disp_w and 0 <= my2 < disp_h):
                    continue

                hidden_color = color_map.get_at((mx2, my2))[:3]
                try:
                    region_idx = region_colors.index(hidden_color)
                except ValueError:
                    region_idx = None
                if region_idx is None:
                    continue

                if not can_click_region(region_idx, selected_player):
                    continue

                attacker = selected_player
                old_owner = owner[region_idx]

                snapshot = owner[:]
                pre_border_cands = compute_border_candidates(snapshot, attacker)

                owner[region_idx] = attacker
                flickers[region_idx] = {
                    "start_ms": int(now_ms),
                    "duration_ms": int(cfg.flicker_duration_ms),
                    "old_owner": int(old_owner),
                    "new_owner": int(attacker),
                }

                battle_ctx = {
                    "attacker": int(attacker),
                    "region": int(region_idx),
                    "old_owner": int(old_owner),
                    "pre_border_cands": [int(x) for x in pre_border_cands],
                    "battle_start_ms": int(now_ms),
                    "wait_until_ms": 0,
                }
                phase = PHASE_BATTLE_FLICKER
                needs_redraw = True

        # Phase machine
        if phase == PHASE_DEFEND_PREVIEW:
            # Wait until preview flicker ends, then show prompt
            if defend_ctx is not None:
                tgt = int(defend_ctx["target"])
                age = now_ms - int(defend_ctx["preview_start_ms"])
                if age >= int(cfg.defend_preview_ms):
                    flickers.pop(tgt, None)  # stop preview flicker
                    phase = PHASE_DEFEND_PROMPT
                    needs_redraw = True
            else:
                phase = PHASE_PLAY_QUEUE

        elif phase == PHASE_BATTLE_FLICKER:
            age = now_ms - int(battle_ctx["battle_start_ms"])
            if age >= int(cfg.flicker_duration_ms):
                ridx = int(battle_ctx["region"])
                flickers.pop(ridx, None)
                battle_ctx["wait_until_ms"] = now_ms + int(cfg.battle_delay_ms)
                phase = PHASE_BATTLE_WAIT
                needs_redraw = True

        elif phase == PHASE_BATTLE_WAIT:
            if now_ms >= int(battle_ctx["wait_until_ms"]):
                attacker = int(battle_ctx["attacker"])
                ridx = int(battle_ctx["region"])
                old_owner = int(battle_ctx["old_owner"])

                win = bool(dummy_method(attacker, ridx))

                anim_queue.clear()
                current_anim = None
                between_anim_wait_until = now_ms

                if not win:
                    enqueue_anim(ridx, old_owner, int(cfg.revert_flicker_ms))
                else:
                    pre = [c for c in battle_ctx["pre_border_cands"] if c != ridx]
                    pre = [c for c in pre if owner[c] != attacker]
                    random.shuffle(pre)
                    for _ in range(int(cfg.reward_extra_tiles)):
                        if not pre:
                            break
                        reward = pre.pop()
                        enqueue_anim(reward, attacker, int(cfg.flicker_duration_ms))

                schedule_automated_player_grants(attacker, none_idx)
                phase = PHASE_PLAY_QUEUE
                needs_redraw = True

        elif phase == PHASE_PLAY_QUEUE:
            if current_anim is None and anim_queue and now_ms >= between_anim_wait_until:
                start_next_anim(now_ms)

            update_anim(now_ms)

            if current_anim is None and not anim_queue and phase not in (PHASE_DEFEND_PROMPT, PHASE_DEFEND_PREVIEW):
                # next round begins NOW (before re-enabling clicking)
                round_no += 1
                needs_redraw = True

                # draw + update once so the map is visible behind the overlay
                rebuild_fill_surface(now_ms)
                screen.blit(background_surf, (0, 0))
                screen.blit(fill_surf, (0, 0))
                screen.blit(border_surf, (0, 0))
                if cfg.show_arrows:
                    draw_arrows(now_ms)
                draw_buttons()
                pygame.display.flip()

                # === CARD CHOICE (modal) ===
                # picked = pick_round_card(screen)
                # TODO: apply effects later based on picked["id"]
                # print("Picked card:", picked)

                # finally re-enable play
                phase = PHASE_IDLE

        if flickers:
            needs_redraw = True
        if needs_redraw:
            rebuild_fill_surface(now_ms)
            needs_redraw = False

        # Draw
        screen.blit(background_surf, (0, 0))
        screen.blit(fill_surf, (0, 0))
        screen.blit(border_surf, (0, 0))
        draw_arrows(now_ms)

        # Hover highlight
        if hover_region is not None:
            if hover_ok and hover_hi_scaled_ok is not None:
                screen.blit(hover_hi_scaled_ok, (0, 0))
            elif hover_hi_scaled_bad is not None:
                tmp = hover_hi_scaled_bad.copy()
                tmp.fill((255, 90, 90, 0), special_flags=pygame.BLEND_RGBA_MULT)
                screen.blit(tmp, (0, 0))

        # HUD
        hud = pygame.Surface((cfg.width, cfg.hud_h), flags=pygame.SRCALPHA)
        hud.fill((0, 0, 0, int(cfg.hud_bg_alpha)))
        sel_label, sel_col = players[selected_player]
        txt = font.render(f"Round: {round_no}    Selected: {sel_label}", True, (240, 240, 240))
        hud.blit(txt, (10, 4))
        sw = pygame.Rect(10 + txt.get_width() + 12, 6, 16, 16)
        pygame.draw.rect(hud, sel_col, sw)
        pygame.draw.rect(hud, (255, 255, 255), sw, 1)
        screen.blit(hud, (0, 0))

        draw_buttons()

        if phase == PHASE_DEFEND_PROMPT:
            draw_defend_prompt()

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

    meta = {
        "assign": assign,
        "num_regions": num_regions,
        "players": players,
        "region_colors": region_colors,
        "adjacency": adjacency,
        "centroids": centroids,
        "contact_points": contact_points,
    }
    return owner, meta
