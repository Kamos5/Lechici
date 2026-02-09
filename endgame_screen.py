
"""
endgame_screen.py

End-of-mission screen (victory/defeat) with AoE-II-style stats:
- summary numbers
- MVP unit (miniature + name)
- time-series line charts + a simple bar chart for final values
- bottom buttons: Restart / Continue

This module is intentionally "dummy-friendly" and does NOT depend on the game's ui.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List

import pygame


@dataclass
class Button:
    rect: pygame.Rect
    label: str


def _draw_text(surface: pygame.Surface, font: pygame.font.Font, text: str, pos: Tuple[int, int], color=(255, 255, 255)) -> pygame.Rect:
    img = font.render(text, True, color)
    r = img.get_rect(topleft=pos)
    surface.blit(img, r)
    return r


def _draw_panel(surface: pygame.Surface, rect: pygame.Rect, *, border=(230, 230, 230), fill=(10, 10, 10)) -> None:
    pygame.draw.rect(surface, fill, rect)
    pygame.draw.rect(surface, border, rect, 2)


def _nice_ticks(max_v: float) -> float:
    # cheap "nice number" tick step
    if max_v <= 0:
        return 1.0
    mag = 10 ** int(max(0, (len(str(int(max_v))) - 1)))
    for step in (1, 2, 5, 10):
        if max_v / (mag * step) <= 5:
            return mag * step
    return mag * 10


def _draw_line_chart(
    surface: pygame.Surface,
    rect: pygame.Rect,
    title: str,
    t: List[float],
    series: List[Tuple[str, List[float], Tuple[int, int, int]]],
    font: pygame.font.Font,
    small_font: pygame.font.Font,
) -> None:
    _draw_panel(surface, rect)
    _draw_text(surface, font, title, (rect.x + 10, rect.y + 6))

    plot = pygame.Rect(rect.x + 10, rect.y + 40, rect.w - 20, rect.h - 55)
    pygame.draw.rect(surface, (20, 20, 20), plot)

    if not t or len(t) < 2:
        _draw_text(surface, small_font, "No data yet.", (plot.x + 10, plot.y + 10), (200, 200, 200))
        return

    t0, t1 = float(t[0]), float(t[-1])
    if t1 <= t0:
        t1 = t0 + 1.0

    # find global max
    max_v = 0.0
    for _, vals, _ in series:
        if vals:
            max_v = max(max_v, float(max(vals)))
    max_v = max(1.0, max_v)

    # grid lines
    step = _nice_ticks(max_v)
    y = 0.0
    while y <= max_v + 1e-6:
        yy = plot.bottom - int((y / max_v) * plot.h)
        pygame.draw.line(surface, (35, 35, 35), (plot.x, yy), (plot.right, yy), 1)
        _draw_text(surface, small_font, str(int(y)), (plot.x + 4, yy - 10), (160, 160, 160))
        y += step

    # x-axis labels (start/end)
    _draw_text(surface, small_font, f"{int(t0)}s", (plot.x, plot.bottom + 4), (160, 160, 160))
    _draw_text(surface, small_font, f"{int(t1)}s", (plot.right - 40, plot.bottom + 4), (160, 160, 160))

    def to_xy(tt: float, vv: float) -> Tuple[int, int]:
        x = plot.x + int(((tt - t0) / (t1 - t0)) * plot.w)
        y = plot.bottom - int((vv / max_v) * plot.h)
        return x, y

    # series lines + legend
    lx = plot.right - 160
    ly = rect.y + 10
    for name, vals, color in series:
        if not vals or len(vals) != len(t):
            continue
        pts = [to_xy(tt, float(vv)) for tt, vv in zip(t, vals)]
        pygame.draw.lines(surface, color, False, pts, 2)
        pygame.draw.rect(surface, color, pygame.Rect(lx, ly + 6, 10, 10))
        _draw_text(surface, small_font, name, (lx + 16, ly + 2), (230, 230, 230))
        ly += 18


def _draw_bar_chart(
    surface: pygame.Surface,
    rect: pygame.Rect,
    title: str,
    items: List[Tuple[str, float, Tuple[int, int, int]]],
    font: pygame.font.Font,
    small_font: pygame.font.Font,
) -> None:
    _draw_panel(surface, rect)
    _draw_text(surface, font, title, (rect.x + 10, rect.y + 6))

    plot = pygame.Rect(rect.x + 10, rect.y + 40, rect.w - 20, rect.h - 55)
    pygame.draw.rect(surface, (20, 20, 20), plot)

    if not items:
        _draw_text(surface, small_font, "No data.", (plot.x + 10, plot.y + 10), (200, 200, 200))
        return

    max_v = max(1.0, max(float(v) for _, v, _ in items))
    n = len(items)
    bar_w = max(8, plot.w // max(1, n * 2))
    gap = bar_w

    x = plot.x + 10
    for name, val, color in items:
        h = int((float(val) / max_v) * (plot.h - 30))
        br = pygame.Rect(x, plot.bottom - h - 20, bar_w, h)
        pygame.draw.rect(surface, color, br)
        _draw_text(surface, small_font, str(int(val)), (x, br.y - 16), (230, 230, 230))
        _draw_text(surface, small_font, name, (x - 2, plot.bottom - 18), (200, 200, 200))
        x += bar_w + gap


def _make_button(font: pygame.font.Font, label: str, x: int, y: int, w: int, h: int) -> Button:
    return Button(rect=pygame.Rect(x, y, w, h), label=label)


def _draw_button(surface: pygame.Surface, btn: Button, font: pygame.font.Font, *, enabled=True) -> None:
    fill = (40, 40, 40) if enabled else (25, 25, 25)
    border = (230, 230, 230) if enabled else (80, 80, 80)
    pygame.draw.rect(surface, fill, btn.rect, border_radius=8)
    pygame.draw.rect(surface, border, btn.rect, 2, border_radius=8)
    txt = font.render(btn.label, True, (255, 255, 255) if enabled else (140, 140, 140))
    tr = txt.get_rect(center=btn.rect.center)
    surface.blit(txt, tr)


def run_endgame_stats_screen(
    *,
    screen: pygame.Surface,
    clock: pygame.time.Clock,
    result: str,  # "victory" | "defeat"
    stats,
    fonts: dict,
) -> str:
    """
    Returns:
        "restart" | "continue" | "quit"
    """
    # fonts dict comes from ui.create_fonts()
    title_font = fonts.get("end_button_font") or fonts.get("button_font") or fonts["font"]
    font = fonts["font"]
    small_font = fonts.get("small_font") or fonts["font"]
    btn_font = fonts.get("button_font") or fonts["font"]

    # Support both single-player StatsTracker and MultiStats wrapper.
    pstats = stats.get_tracker(1) if hasattr(stats, 'get_tracker') and stats.get_tracker(1) is not None else stats

    # Simple tab state
    tabs = ["Summary", "Economy", "Military", "Production"]
    active_tab = 0

    # Colors
    C_BG = (0, 0, 0)
    C_VIC = (0, 140, 0)
    C_DEF = (160, 0, 0)
    C_ACC1 = (70, 180, 255)
    C_ACC2 = (255, 200, 80)
    C_ACC3 = (230, 120, 255)
    C_ACC4 = (200, 200, 200)

    w, h = screen.get_width(), screen.get_height()

    # Layout
    pad = 16
    header = pygame.Rect(pad, pad, w - 2 * pad, 70)
    body = pygame.Rect(pad, header.bottom + 10, w - 2 * pad, h - header.h - 120)
    footer = pygame.Rect(pad, h - 90, w - 2 * pad, 74)

    # Left/right body panels
    left = pygame.Rect(body.x, body.y, int(body.w * 0.35), body.h)
    right = pygame.Rect(left.right + 10, body.y, body.right - (left.right + 10), body.h)

    # Buttons
    restart_btn = _make_button(btn_font, "Restart", footer.x + 10, footer.y + 16, 170, 42)
    cont_btn = _make_button(btn_font, "Continue", footer.right - 180, footer.y + 16, 170, 42)

    # Build tab buttons (top of right panel)
    tab_btns = []
    tx = right.x + 10
    for tname in tabs:
        tab_btns.append(Button(pygame.Rect(tx, right.y + 10, 140, 32), tname))
        tx += 150

    # MVP snapshot
    mvp = stats.get_mvp() if hasattr(stats, "get_mvp") else None

    # Main loop
    running = True
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                return "quit"
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    return "continue"
            if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                mp = ev.pos

                # tabs
                for i, b in enumerate(tab_btns):
                    if b.rect.collidepoint(mp):
                        active_tab = i

                if restart_btn.rect.collidepoint(mp):
                    return "restart"
                if cont_btn.rect.collidepoint(mp):
                    return "continue"

        screen.fill(C_BG)

        # Header
        _draw_panel(screen, header)
        title = "Victory" if result == "victory" else "Defeat"
        tcolor = C_VIC if result == "victory" else C_DEF
        _draw_text(screen, title_font, title, (header.x + 12, header.y + 12), tcolor)
        _draw_text(screen, small_font, "End of mission statistics", (header.x + 12, header.y + 44), (210, 210, 210))

        # Body panels
        _draw_panel(screen, left)
        _draw_panel(screen, right)

        # Left: summary numbers
        y = left.y + 12
        _draw_text(screen, font, "Totals", (left.x + 12, y))
        y += 34

        rows = [
            ("Produced units", int(getattr(pstats, "produced_units", 0))),
            ("Produced buildings", int(getattr(pstats, "produced_buildings", 0))),
            ("Collected milk", int(getattr(pstats, "collected_milk", 0))),
            ("Collected wood", int(getattr(pstats, "collected_wood", 0))),
            ("Units trained", int(getattr(pstats, "units_trained", 0))),
            ("Units lost", int(getattr(pstats, "units_lost", 0))),
            ("Units killed", int(getattr(pstats, "units_killed", 0))),
        ]
        for name, val in rows:
            _draw_text(screen, small_font, name, (left.x + 12, y), (200, 200, 200))
            _draw_text(screen, small_font, str(val), (left.right - 70, y), (255, 255, 255))
            y += 22

        # Multi-player summary (excluding Gaia). Shows all players if MultiStats is used.
        if hasattr(stats, 'iter_player_ids') and hasattr(stats, 'get_tracker'):
            pids = [pid for pid in stats.iter_player_ids() if int(pid) != 0]
            if len(pids) > 1:
                y += 14
                _draw_text(screen, small_font, "Players", (left.x + 12, y))
                y += 22
                _draw_text(screen, small_font, "PID  Trn  Lst  Kll  Bld", (left.x + 12, y))
                y += 18
                for pid in pids:
                    tr = stats.get_tracker(pid)
                    if tr is None:
                        continue
                    line = f"{pid:>3}  {int(getattr(tr,'units_trained',0)):>3}  {int(getattr(tr,'units_lost',0)):>3}  {int(getattr(tr,'units_killed',0)):>3}  {int(getattr(tr,'produced_buildings',0)):>3}"
                    _draw_text(screen, small_font, line, (left.x + 12, y))
                    y += 18

        # MVP panel
        y += 12
        _draw_text(screen, font, "MVP", (left.x + 12, y))
        y += 32
        mvp_box = pygame.Rect(left.x + 12, y, left.w - 24, 90)
        pygame.draw.rect(screen, (18, 18, 18), mvp_box)
        pygame.draw.rect(screen, (60, 60, 60), mvp_box, 2)

        if mvp is None:
            _draw_text(screen, small_font, "No kills recorded.", (mvp_box.x + 10, mvp_box.y + 10), (200, 200, 200))
        else:
            # icon
            icon = getattr(mvp, "icon", None)
            if icon is None:
                icon = pygame.Surface((64, 64), pygame.SRCALPHA)
                icon.fill((60, 60, 60, 255))
                pygame.draw.rect(icon, (200, 200, 200), icon.get_rect(), 2)
            else:
                icon = pygame.transform.smoothscale(icon, (64, 64))

            screen.blit(icon, (mvp_box.x + 10, mvp_box.y + 12))
            _draw_text(screen, small_font, str(getattr(mvp, "name", "Unit")), (mvp_box.x + 86, mvp_box.y + 14))
            _draw_text(screen, small_font, f"Level: {int(getattr(mvp, 'level', 0))}", (mvp_box.x + 86, mvp_box.y + 36), (200, 200, 200))
            _draw_text(screen, small_font, f"Kills: {int(getattr(mvp, 'kills', 0))}", (mvp_box.x + 86, mvp_box.y + 58), (200, 200, 200))

        # Tabs (right panel top)
        for i, b in enumerate(tab_btns):
            on = (i == active_tab)
            fill = (40, 40, 40) if on else (22, 22, 22)
            pygame.draw.rect(screen, fill, b.rect, border_radius=6)
            pygame.draw.rect(screen, (230, 230, 230) if on else (70, 70, 70), b.rect, 2, border_radius=6)
            txt = small_font.render(b.label, True, (255, 255, 255) if on else (180, 180, 180))
            screen.blit(txt, txt.get_rect(center=b.rect.center))

        # Right: charts
        chart_area = pygame.Rect(right.x + 10, right.y + 52, right.w - 20, right.h - 62)

        t = list(getattr(pstats.series, "t", []))
        # (Name, values, color)
        if active_tab == 0:  # Summary
            _draw_line_chart(
                screen,
                pygame.Rect(chart_area.x, chart_area.y, chart_area.w, int(chart_area.h * 0.62)),
                "Resources collected over time",
                t,
                [
                    ("Milk", list(getattr(pstats.series, "milk_collected", [])), C_ACC1),
                    ("Wood", list(getattr(pstats.series, "wood_collected", [])), C_ACC2),
                ],
                font,
                small_font,
            )
            _draw_bar_chart(
                screen,
                pygame.Rect(chart_area.x, chart_area.y + int(chart_area.h * 0.62) + 10, chart_area.w, chart_area.h - int(chart_area.h * 0.62) - 10),
                "Final totals",
                [
                    ("U", float(getattr(pstats, "produced_units", 0)), C_ACC3),
                    ("B", float(getattr(pstats, "produced_buildings", 0)), C_ACC4),
                    ("M", float(getattr(pstats, "collected_milk", 0)), C_ACC1),
                    ("W", float(getattr(pstats, "collected_wood", 0)), C_ACC2),
                ],
                font,
                small_font,
            )
        elif active_tab == 1:  # Economy
            _draw_line_chart(
                screen,
                chart_area,
                "Economy (cumulative)",
                t,
                [
                    ("Milk", list(getattr(pstats.series, "milk_collected", [])), C_ACC1),
                    ("Wood", list(getattr(pstats.series, "wood_collected", [])), C_ACC2),
                ],
                font,
                small_font,
            )
        elif active_tab == 2:  # Military
            _draw_line_chart(
                screen,
                pygame.Rect(chart_area.x, chart_area.y, chart_area.w, int(chart_area.h * 0.62)),
                "Military (cumulative)",
                t,
                [
                    ("Trained", list(getattr(pstats.series, "units_trained", [])), C_ACC1),
                    ("Lost", list(getattr(pstats.series, "units_lost", [])), C_DEF),
                    ("Killed", list(getattr(pstats.series, "units_killed", [])), C_VIC),
                ],
                font,
                small_font,
            )
            _draw_bar_chart(
                screen,
                pygame.Rect(chart_area.x, chart_area.y + int(chart_area.h * 0.62) + 10, chart_area.w, chart_area.h - int(chart_area.h * 0.62) - 10),
                "Final military",
                [
                    ("Tr", float(getattr(pstats, "units_trained", 0)), C_ACC1),
                    ("Ls", float(getattr(pstats, "units_lost", 0)), C_DEF),
                    ("Kl", float(getattr(pstats, "units_killed", 0)), C_VIC),
                ],
                font,
                small_font,
            )
        else:  # Production
            _draw_line_chart(
                screen,
                chart_area,
                "Production (cumulative)",
                t,
                [
                    ("Units", list(getattr(pstats.series, "units_total", [])), C_ACC3),
                    ("Buildings", list(getattr(pstats.series, "buildings_total", [])), C_ACC4),
                ],
                font,
                small_font,
            )

        # Footer
        _draw_panel(screen, footer)
        _draw_button(screen, restart_btn, btn_font, enabled=True)
        _draw_button(screen, cont_btn, btn_font, enabled=True)

        pygame.display.flip()
        clock.tick(60)

    return "continue"
