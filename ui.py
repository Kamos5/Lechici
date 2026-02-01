import pygame
from pygame.math import Vector2

from constants import *
from units import Unit, Building, Barn, Barracks, TownCenter, Axeman, Archer, Knight, Cow, ShamansHut, KnightsEstate, WarriorsLodge, Ruin


def load_ui_icons():
    """Load and scale UI icons. Falls back to colored squares if missing."""
    try:
        wood_icon = pygame.image.load("assets/wood_icon.png").convert_alpha()
        milk_icon = pygame.image.load("assets/milk_icon.png").convert_alpha()
        # Optional icons - fallback surfaces if extensions missing
        unit_icon = (
            pygame.image.load("assets/unit_icon.png").convert_alpha()
            if pygame.image.get_extended() else pygame.Surface((20, 20))
        )
        building_icon = (
            pygame.image.load("assets/building_icon.png").convert_alpha()
            if pygame.image.get_extended() else pygame.Surface((20, 20))
        )

        totem_icon = (
            pygame.image.load("assets/totem.png").convert_alpha()
            if pygame.image.get_extended() else pygame.Surface((20, 20))
        )

        wood_icon = pygame.transform.scale(wood_icon, (20, 20))
        milk_icon = pygame.transform.scale(milk_icon, (20, 20))
        unit_icon = pygame.transform.scale(unit_icon, (20, 20))
        building_icon = pygame.transform.scale(building_icon, (20, 20))

        # Totem separator icon (scaled down, keep aspect ratio)
        w, h = totem_icon.get_size()
        target_h = 180
        scale = (target_h / h) if h else 1
        totem_icon = pygame.transform.smoothscale(
            totem_icon, (max(1, int(w * scale)), target_h)
        )

    except (pygame.error, FileNotFoundError):
        wood_icon = pygame.Surface((20, 20))
        milk_icon = pygame.Surface((20, 20))
        unit_icon = pygame.Surface((20, 20))
        building_icon = pygame.Surface((20, 20))
        wood_icon.fill(BROWN)
        milk_icon.fill(WHITE)
        unit_icon.fill(LIGHT_GRAY)
        building_icon.fill(LIGHT_GRAY)

        totem_icon = pygame.Surface((50, 100), pygame.SRCALPHA)
        totem_icon.fill(LIGHT_GRAY)
    return {
        "wood": wood_icon,
        "milk": milk_icon,
        "unit": unit_icon,
        "building": building_icon,
        "totem": totem_icon,
    }


def create_fonts():
    """Create and return common fonts."""
    return {
        "font": pygame.font.SysFont(None, 24),
        "small_font": pygame.font.SysFont(None, 16),
        "button_font": pygame.font.SysFont(None, 20),
        # Bigger fonts for the new description/tooltip panel
        "tooltip_title_font": pygame.font.SysFont(None, 36),
        "tooltip_body_font": pygame.font.SysFont(None, 30),
        "end_button_font": pygame.font.Font(None, 36),
    }



# ---------------------------------------------------------------------------
# UI helpers: hoverable buttons + tooltip panel
# ---------------------------------------------------------------------------

def wrap_text(font: pygame.font.Font, text: str, max_width: int) -> list[str]:
    """Simple word-wrapping for pygame text."""
    if not text:
        return []
    words = text.split()
    lines: list[str] = []
    cur = ""
    for w in words:
        test = (cur + " " + w).strip()
        if font.size(test)[0] <= max_width:
            cur = test
        else:
            if cur:
                lines.append(cur)
            # If a single word is longer than max_width, hard-split it.
            if font.size(w)[0] > max_width:
                chunk = ""
                for ch in w:
                    test_chunk = chunk + ch
                    if font.size(test_chunk)[0] <= max_width:
                        chunk = test_chunk
                    else:
                        if chunk:
                            lines.append(chunk)
                        chunk = ch
                cur = chunk
            else:
                cur = w
    if cur:
        lines.append(cur)
    return lines


def draw_tooltip_panel(
    screen: pygame.Surface,
    *,
    name: str,
    description: str,
    milk_cost: int | None,
    wood_cost: int | None,
    icons: dict,
    fonts: dict,
):
    """Draw bottom-left tooltip panel (20% width/height of the window)."""
    w = int(SCREEN_WIDTH * 0.20)
    h = int(SCREEN_HEIGHT * 0.20)
    # Anchor inside the main world view (avoid overlapping UI borders/panels)
    x0 = VIEW_MARGIN_LEFT + 10
    y0 = VIEW_MARGIN_TOP + VIEW_HEIGHT - h - 10
    rect = pygame.Rect(x0, y0, w, h)

    panel = pygame.Surface((rect.w, rect.h), pygame.SRCALPHA)
    panel.fill((0, 0, 0, 160))  # semi-transparent black

    pad = 8
    x = pad
    y = pad

    title_font = fonts.get("tooltip_title_font") or fonts.get("button_font") or fonts.get("font")
    body_font = fonts.get("tooltip_body_font") or fonts.get("button_font") or fonts.get("font")

    # Title
    if name:
        title_surf = title_font.render(name, True, (255, 255, 255))
        panel.blit(title_surf, (x, y))
        y += title_surf.get_height() + 6

    # Costs line (optional)
    if milk_cost is not None or wood_cost is not None:
        icon_scale = 1
        milk_icon = icons.get("milk")
        wood_icon = icons.get("wood")

        if milk_icon is not None:
            mw, mh = milk_icon.get_size()
            milk_small = pygame.transform.smoothscale(milk_icon, (int(mw * icon_scale), int(mh * icon_scale)))
        else:
            milk_small = None

        if wood_icon is not None:
            ww, wh = wood_icon.get_size()
            wood_small = pygame.transform.smoothscale(wood_icon, (int(ww * icon_scale), int(wh * icon_scale)))
        else:
            wood_small = None

        line_x = x
        if milk_cost is not None:
            if milk_small:
                panel.blit(milk_small, (line_x, y))
                line_x += milk_small.get_width() + 4
            t = body_font.render(str(milk_cost), True, (255, 255, 255))
            panel.blit(t, (line_x, y + 2))
            line_x += t.get_width() + 12

        if wood_cost is not None:
            if wood_small:
                panel.blit(wood_small, (line_x, y))
                line_x += wood_small.get_width() + 4
            t2 = body_font.render(str(wood_cost), True, (255, 255, 255))
            panel.blit(t2, (line_x, y + 2))

        y += 22  # fixed row height

    # Description (wrapped)
    max_text_w = rect.w - 2 * pad
    for line in wrap_text(body_font, description or "", max_text_w):
        if y + body_font.get_height() > rect.h - pad:
            break
        panel.blit(body_font.render(line, True, (255, 255, 255)), (x, y))
        y += body_font.get_height() + 2

    screen.blit(panel, rect.topleft)


class UIButton:
    """A minimal hoverable UI button for grids and action panels."""

    def __init__(
        self,
        rect: pygame.Rect,
        *,
        enabled: bool = True,
        label: str = "",
        hotkey: str | None = None,
        icon: pygame.Surface | None = None,
        milk_cost: int | None = None,
        wood_cost: int | None = None,
        description: str = "",
    ):
        self.rect = rect
        self.enabled = enabled
        self.label = label
        self.hotkey = hotkey
        self.icon = icon
        self.milk_cost = milk_cost
        self.wood_cost = wood_cost
        self.description = description

    def hovered(self, mouse_pos) -> bool:
        return self.rect.collidepoint(mouse_pos)

    def draw_base(self, screen: pygame.Surface, *, hovered: bool):
        base_color = HIGHLIGHT_GRAY if self.enabled else GRAY
        pygame.draw.rect(screen, base_color, self.rect)

        if hovered:
            # More visible hover highlight (brighter + thin border)
            overlay = pygame.Surface((self.rect.w, self.rect.h), pygame.SRCALPHA)
            overlay.fill((255, 255, 255, 120))
            screen.blit(overlay, self.rect.topleft)
            pygame.draw.rect(screen, WHITE, self.rect, 2)

    def draw_icon_fill(
        self,
        screen: pygame.Surface,
        *,
        icon_pad: int = 4,
    ):
        """Draw the icon centered, scaled to fill the button with a small padding border."""
        if not self.icon:
            return

        inner = self.rect.inflate(-2 * icon_pad, -2 * icon_pad)
        if inner.w <= 1 or inner.h <= 1:
            return

        iw, ih = self.icon.get_size()
        if iw <= 0 or ih <= 0:
            return

        # Scale to cover the available space (like a thumbnail)
        scale = max(inner.w / iw, inner.h / ih)
        tw, th = max(1, int(iw * scale)), max(1, int(ih * scale))
        img = pygame.transform.smoothscale(self.icon, (tw, th))

        # Center-crop into the inner rect
        sx = (tw - inner.w) // 2
        sy = (th - inner.h) // 2
        src = pygame.Rect(sx, sy, inner.w, inner.h)
        screen.blit(img, inner.topleft, src)

        # Subtle frame
        pygame.draw.rect(screen, (0, 0, 0), inner, 1)

    def draw_label(self, screen: pygame.Surface, font: pygame.font.Font):
        # Only draw label text (if any). Hotkey badge is drawn elsewhere always.
        if self.label:
            txt = font.render(self.label, True, BLACK)
            screen.blit(txt, (self.rect.x + 2, self.rect.y + 2))

    def draw_hotkey_badge(self, screen: pygame.Surface, font: pygame.font.Font):
        """Draw hotkey in the top-right corner inside [] using the provided font (size 20)."""
        if not self.hotkey:
            return
        text = f"[{self.hotkey.upper()}]"
        hk = font.render(text, True, BLACK)
        screen.blit(hk, (self.rect.right - hk.get_width() - 4, self.rect.y + 2))

    def draw_hotkey_center(self, screen: pygame.Surface, font: pygame.font.Font):
        """For icon-less buttons: draw hotkey centered as the only visible glyph."""
        if not self.hotkey:
            return
        hk = font.render(self.hotkey.upper(), True, BLACK)
        r = hk.get_rect(center=self.rect.center)
        screen.blit(hk, r.topleft)


def build_ui_layout():
    """Build and return UI rectangles (player buttons, grid buttons, quit button)."""
    button_player0_pos = (VIEW_MARGIN_LEFT + VIEW_WIDTH - BUTTON_WIDTH - 10, VIEW_MARGIN_TOP + 40)
    button_player1_pos = (
        VIEW_MARGIN_LEFT + VIEW_WIDTH - BUTTON_WIDTH - 10,
        VIEW_MARGIN_TOP + 40 + (BUTTON_HEIGHT + BUTTON_MARGIN),
    )
    button_player2_pos = (
        VIEW_MARGIN_LEFT + VIEW_WIDTH - BUTTON_WIDTH - 10,
        VIEW_MARGIN_TOP + 40 + 2 * (BUTTON_HEIGHT + BUTTON_MARGIN),
    )

    button_player0 = pygame.Rect(button_player0_pos[0], button_player0_pos[1], BUTTON_WIDTH, BUTTON_HEIGHT)
    button_player1 = pygame.Rect(button_player1_pos[0], button_player1_pos[1], BUTTON_WIDTH, BUTTON_HEIGHT)
    button_player2 = pygame.Rect(button_player2_pos[0], button_player2_pos[1], BUTTON_WIDTH, BUTTON_HEIGHT)

    # Left panel action/production grid: 4 cols x 3 rows
    GRID_COLS = 4
    GRID_ROWS = 3

    # Fit inside left panel (0..VIEW_MARGIN_LEFT)
    left_pad = 10
    top_pad = 10

    grid_total_w = VIEW_MARGIN_LEFT - 2 * left_pad
    # Use the full bottom panel height for a squarer 4x3 grid
    grid_total_h = PANEL_HEIGHT - 2 * top_pad

    grid_button_start_x = left_pad
    grid_button_start_y = PANEL_Y + top_pad  # inside the bottom panel, but left side

    # Make buttons more square-like.
    # Compute the largest square size that fits into the available region.
    MIN_BTN = 56
    max_by_w = (grid_total_w - (GRID_COLS - 1) * GRID_BUTTON_MARGIN) // GRID_COLS
    max_by_h = (grid_total_h - (GRID_ROWS - 1) * GRID_BUTTON_MARGIN) // GRID_ROWS
    btn_size = max(MIN_BTN, min(max_by_w, max_by_h))

    # Derive effective margins so the grid fits nicely.
    eff_margin_x = GRID_BUTTON_MARGIN
    eff_margin_y = GRID_BUTTON_MARGIN
    if GRID_COLS > 1:
        remaining_w = grid_total_w - GRID_COLS * btn_size
        eff_margin_x = max(2, remaining_w // (GRID_COLS - 1))
    if GRID_ROWS > 1:
        remaining_h = grid_total_h - GRID_ROWS * btn_size
        eff_margin_y = max(2, remaining_h // (GRID_ROWS - 1))

    grid_buttons = []
    for row in range(GRID_ROWS):
        row_buttons = []
        for col in range(GRID_COLS):
            x = grid_button_start_x + col * (btn_size + eff_margin_x)
            y = grid_button_start_y + row * (btn_size + eff_margin_y)
            row_buttons.append(pygame.Rect(x, y, btn_size, btn_size))
        grid_buttons.append(row_buttons)

    quit_button = pygame.Rect(
        SCREEN_WIDTH // 2 - 200 // 2,
        SCREEN_HEIGHT // 2 + 50,
        200,
        50,
    )

    return {
        "button_player0": button_player0,
        "button_player1": button_player1,
        "button_player2": button_player2,
        "grid_buttons": grid_buttons,
        "quit_button": quit_button,
        "icon_size": 32,
        "icon_margin": 5,
    }


def _get_selected_buildings(current_player):
    if not current_player:
        return None, None, None
    selected_barn = next(
        (u for u in current_player.units if isinstance(u, Barn) and u.selected and u.alpha == 255),
        None,
    )
    selected_barracks = next(
        (u for u in current_player.units if isinstance(u, Barracks) and u.selected and u.alpha == 255),
        None,
    )
    selected_town_center = next(
        (u for u in current_player.units if isinstance(u, TownCenter) and u.selected and u.alpha == 255),
        None,
    )
    selected_shamans_hut = next(
        (u for u in current_player.units if isinstance(u, ShamansHut) and u.selected and u.alpha == 255),
        None,
    )
    return selected_barn, selected_barracks, selected_town_center, selected_shamans_hut


def draw_panels(screen):
    pygame.draw.rect(screen, PANEL_COLOR, (0, 0, VIEW_MARGIN_LEFT, SCREEN_HEIGHT))
    pygame.draw.rect(screen, PANEL_COLOR, (SCREEN_WIDTH - VIEW_MARGIN_RIGHT, 0, VIEW_MARGIN_RIGHT, SCREEN_HEIGHT))
    pygame.draw.rect(screen, PANEL_COLOR, (0, 0, SCREEN_WIDTH, VIEW_MARGIN_TOP))
    pygame.draw.rect(screen, PANEL_COLOR, (VIEW_MARGIN_LEFT, PANEL_Y, VIEW_WIDTH, PANEL_HEIGHT))


def draw_grid_buttons(screen, grid_buttons, current_player, all_units, production_queues, current_time, icons, fonts):

    small_font = fonts["small_font"]
    mouse_pos = pygame.mouse.get_pos()
    hovered_tooltip = None  # (name, desc, milk_cost, wood_cost)

    # --- cached scaled icons for build grid ---
    if not hasattr(draw_grid_buttons, "_scaled_cache"):
        draw_grid_buttons._scaled_cache = {}

    def get_scaled(icon, scale):
        key = (id(icon), scale)
        if key not in draw_grid_buttons._scaled_cache:
            w, h = icon.get_size()
            draw_grid_buttons._scaled_cache[key] = pygame.transform.smoothscale(
                icon, (int(w * scale), int(h * scale))
            )
        return draw_grid_buttons._scaled_cache[key]

    milk_icon_small = get_scaled(icons["milk"], 0.5)
    wood_icon_small = get_scaled(icons["wood"], 0.5)

    """Draw the 3x1 grid and its contextual contents (unit/building actions)."""
    selected_barn, selected_barracks, selected_town_center, selected_shamans_hut = _get_selected_buildings(current_player)

    selected = [u for u in all_units if u.selected and current_player and u.player_id == current_player.player_id]
    selected_units = [u for u in selected if not isinstance(u, Building)]
    selected_buildings = [u for u in selected if isinstance(u, Building) and getattr(u, "alpha", 255) == 255]

    # Priority rule: if ANY units selected (even with buildings), show unit command buttons
    show_unit_commands = len(selected_units) > 0

    # Unit-action availability rules
    has_axeman = any(isinstance(u, Axeman) for u in selected_units)
    has_cow = any(isinstance(u, Cow) for u in selected_units)

    can_harvest = has_axeman or has_cow  # only cows + axemen
    can_repair = has_axeman  # only axemen

    # Helper: draw a labeled button (fast + simple)
    def draw_label(btn_rect, label, enabled=True, hotkey_text= None):
        txt = small_font.render(label, True, BLACK)
        screen.blit(txt, (btn_rect.x + 2, btn_rect.y + 2))

        # hotkey in top-right corner
        if hotkey_text:
            hk = small_font.render(hotkey_text.upper(), True, BLACK)
            screen.blit(hk, (btn_rect.right - hk.get_width() - 2, btn_rect.y + 2))

    # Helper: draw a unit/build icon + costs + queue progress
    def draw_costs(btn_rect, unit_cls):
        screen.blit(milk_icon_small, (btn_rect.x + 2, btn_rect.y + 14))
        screen.blit(
            small_font.render(f"{unit_cls.milk_cost}", True, BLACK),
            (btn_rect.x + 14, btn_rect.y + 16),
        )

        screen.blit(wood_icon_small, (btn_rect.x + 2, btn_rect.y + 26))
        screen.blit(
            small_font.render(f"{unit_cls.wood_cost}", True, BLACK),
            (btn_rect.x + 14, btn_rect.y + 28),
        )

    def draw_progress(btn_rect, queue_owner, unit_cls):
        if queue_owner in production_queues and production_queues[queue_owner]["unit_type"] == unit_cls:
            progress = (current_time - production_queues[queue_owner]["start_time"]) / unit_cls.production_time
            progress = max(0.0, min(1.0, progress))
            w = int(progress * (btn_rect.width - 4))
            pygame.draw.rect(screen, GREEN, (btn_rect.x + 2, btn_rect.y + 2, w, 4))

    # Fill background for all buttons first
    for row in range(len(grid_buttons)):
        for col in range(len(grid_buttons[row])):
            pygame.draw.rect(screen, LIGHT_GRAY, grid_buttons[row][col])

    if show_unit_commands:
        # Grid hotkeys (4x3): q w e r / a s d f / z x c v
        hotkeys = {
            (0, 0): "q", (0, 1): "w", (0, 2): "e", (0, 3): "r",
            (1, 0): "a", (1, 1): "s", (1, 2): "d", (1, 3): "f",
            (2, 0): "z", (2, 1): "x", (2, 2): "c", (2, 3): "v",
        }

        # Always visible unit buttons:
        # - Patrol, Move, Attack, Stop = always enabled
        # - Harvest only if Cow/Axeman selected
        # - Repair only if Axeman selected
        mapping = {
            (0, 0): ("Patrol", True),
            (0, 1): ("Move", True),
            (0, 2): ("Harvest", can_harvest),
            (0, 3): ("Repair", can_repair),
            (1, 0): ("Attack", True),
            (1, 1): ("Stop", True),
        }

        for (r, c), (label, enabled) in mapping.items():
            if r < len(grid_buttons) and c < len(grid_buttons[r]):
                btn = grid_buttons[r][c]
                ui_btn = UIButton(
                    btn,
                    enabled=enabled,
                    # Keep the grid clean: no text inside buttons.
                    # Use hotkeys + hover tooltip for details.
                    label="",
                    hotkey=hotkeys.get((r, c)),
                    description={
                        'Patrol': 'Move in formation between points (TODO).',
                        'Move': 'Issue a move order (right-click also works).',
                        'Harvest': 'Gather resources (cows/axemen only).',
                        'Repair': 'Repair buildings (axemen only).',
                        'Attack': 'Force-attack mode (TODO).',
                        'Stop': 'Cancel current orders and clear targets.',
                    }.get(label, ''),
                )
                hov = ui_btn.hovered(mouse_pos)
                ui_btn.draw_base(screen, hovered=hov)
                # For action buttons we currently have no icons: show the hotkey letter centered.
                ui_btn.draw_hotkey_badge(screen, fonts.get("button_font") or small_font)
                if hov:
                    hovered_tooltip = (label, ui_btn.description, None, None)

    else:
        # Production mode (only buildings selected)
        # We pack production options from top-left across 4 columns, then next row.
        options = []

        if current_player and selected_barn:
            options.append(("Cow", Cow, selected_barn))

        if current_player and selected_barracks:
            options.extend([
                ("Axeman", Axeman, selected_barracks),
                ("Archer", Archer, selected_barracks),
                ("Knight", Knight, selected_barracks),
            ])

        if current_player and selected_town_center:
            options.extend([
                ("Barn", Barn, selected_town_center),
                ("Barracks", Barracks, selected_town_center),
                ("TownCenter", TownCenter, selected_town_center),
                ("ShamansHut", ShamansHut, selected_town_center),
                ("KnightsEstate", KnightsEstate, selected_town_center),
                ("WarriorsLodge", WarriorsLodge, selected_town_center),
                ("Ruin", Ruin, selected_town_center),
            ])

        # Draw options into the 4x3 grid
        idx = 0
        for row in range(len(grid_buttons)):
            for col in range(len(grid_buttons[row])):
                if idx >= len(options):
                    continue

                label, cls, owner = options[idx]
                btn = grid_buttons[row][col]

                enabled = (
                        current_player.milk >= cls.milk_cost
                        and current_player.wood >= cls.wood_cost
                        and owner not in production_queues
                )

                # If placing buildings from TownCenter, also enforce building limit
                if owner == selected_town_center and issubclass(cls, Building):
                    enabled = enabled and (
                            current_player.building_limit is None
                            or current_player.building_count < current_player.building_limit
                    )

                ui_btn = UIButton(
                    btn,
                    enabled=enabled,
                    # No text on production/build buttons; icon + tooltip only.
                    label="",
                    hotkey=None,
                    icon=None,
                    milk_cost=getattr(cls, 'milk_cost', 0),
                    wood_cost=getattr(cls, 'wood_cost', 0),
                    description=getattr(cls, 'description', ''),
                )

                hov = ui_btn.hovered(mouse_pos)
                ui_btn.draw_base(screen, hovered=hov)

                # Ensure icon is loaded even if no instance exists yet
                if cls.__name__ not in Unit._unit_icons or Unit._unit_icons.get(cls.__name__) is None:
                    Unit.load_images(
                        cls.__name__,
                        BUILDING_SIZE if issubclass(cls, Building) else UNIT_SIZE,
                    )

                icon = Unit._unit_icons.get(cls.__name__)
                ui_btn.icon = icon
                # Center-fill the icon, covering the whole button with a small border.
                ui_btn.draw_icon_fill(screen, icon_pad=4)

                # Costs are shown in the tooltip panel, not on the button.
                draw_progress(btn, owner, cls)

                if hov:
                    hovered_tooltip = (label, ui_btn.description, ui_btn.milk_cost, ui_btn.wood_cost)

                idx += 1


    # Tooltip panel (bottom-left)
    if hovered_tooltip:
        name, desc, milk_cost, wood_cost = hovered_tooltip
        draw_tooltip_panel(
            screen,
            name=name,
            description=desc,
            milk_cost=milk_cost,
            wood_cost=wood_cost,
            icons=icons,
            fonts=fonts,
        )


def _hp_color(pct: float) -> tuple[int, int, int]:
    """0..1 -> red..green"""
    pct = max(0.0, min(1.0, pct))
    r = int(255 * (1.0 - pct))
    g = int(255 * pct)
    return (r, g, 0)


def draw_selected_building_panel(
    screen: pygame.Surface,
    building: Building,
    *,
    production_queues: dict,
    current_time: float,
    fonts: dict,
):
    """AoE2-like info panel for a single selected building."""
    font = fonts.get("font") or pygame.font.SysFont(None, 24)
    small_font = fonts.get("small_font") or pygame.font.SysFont(None, 16)

    # Panel area: bottom UI panel, between left build-grid area and right minimap panel.
    pad = 8
    x0 = VIEW_MARGIN_LEFT + 270
    y0 = PANEL_Y + pad
    w = VIEW_WIDTH - 500  # leave some space near right edge; minimap is in right side panel anyway
    h = PANEL_HEIGHT - 2 * pad
    rect = pygame.Rect(x0, y0, w, h)

    # Background (shade of gray for now)
    pygame.draw.rect(screen, (170, 170, 170), rect)
    pygame.draw.rect(screen, (110, 110, 110), rect, 2)

    # Layout columns
    left_w = 200
    mid_w = 170
    left = pygame.Rect(rect.x + pad, rect.y + pad, left_w - pad, rect.h - 2 * pad)
    mid = pygame.Rect(left.right + pad, rect.y + pad, mid_w - pad, rect.h - 2 * pad)
    right = pygame.Rect(mid.right + pad, rect.y + pad, rect.right - (mid.right + 2 * pad), rect.h - 2 * pad)

    # --- LEFT: name, icon, HP bar ---
    name = getattr(building, "name", "") or building.__class__.__name__
    screen.blit(font.render(name, True, BLACK), (left.x, left.y))

    # Building icon
    cls_name = building.__class__.__name__
    icon = Unit._unit_icons.get(cls_name)
    if icon is None:
        # Ensure icon exists
        Unit.load_images(cls_name, BUILDING_SIZE)
        icon = Unit._unit_icons.get(cls_name)

    icon_size = 54
    icon_rect = pygame.Rect(left.x, left.y + 28, icon_size, icon_size)
    pygame.draw.rect(screen, (60, 60, 60), icon_rect, 1)
    if icon:
        img = pygame.transform.smoothscale(icon, (icon_size, icon_size))
        screen.blit(img, icon_rect.topleft)

    # HP bar with black background + dynamic fill color
    hp = float(getattr(building, "hp", 0))
    max_hp = float(getattr(building, "max_hp", 1) or 1)
    pct = 0.0 if max_hp <= 0 else max(0.0, min(1.0, hp / max_hp))

    # HP bar: same width as icon
    bar_w = icon_rect.w
    bar_h = 10  # slightly smaller height too (optional, tweak as you like)
    bar_x = icon_rect.x
    bar_y = icon_rect.bottom + 8

    # black background
    pygame.draw.rect(screen, (0, 0, 0), (bar_x, bar_y, bar_w, bar_h))
    # colored fill
    fill_w = int((bar_w - 2) * pct)
    pygame.draw.rect(screen, _hp_color(pct), (bar_x + 1, bar_y + 1, fill_w, bar_h - 2))

    # HP text under bar
    hp_text = f"{int(hp)}/{int(max_hp)}"
    screen.blit(small_font.render(hp_text, True, BLACK), (bar_x, bar_y + bar_h + 4))

    # --- MID: queue (single slot) ---
    # Only show queue UI if producing
    if building in production_queues:
        slot = pygame.Rect(mid.x + 10, mid.y + 20, 64, 64)
        pygame.draw.rect(screen, (200, 200, 200), slot)
        pygame.draw.rect(screen, (80, 80, 80), slot, 2)

        unit_cls = production_queues[building].get("unit_type")
        start_time = production_queues[building].get("start_time", current_time)
        prod_time = float(getattr(unit_cls, "production_time", 1) or 1)
        progress = (current_time - start_time) / prod_time
        progress = max(0.0, min(1.0, progress))

        # Unit icon in slot
        if unit_cls is not None:
            u_name = unit_cls.__name__
            u_icon = Unit._unit_icons.get(u_name)
            if u_icon is None:
                Unit.load_images(u_name, UNIT_SIZE)
                u_icon = Unit._unit_icons.get(u_name)
            if u_icon:
                img = pygame.transform.smoothscale(u_icon, (slot.w, slot.h))
                screen.blit(img, slot.topleft)

        # Progress bar under slot
        pb = pygame.Rect(slot.x, slot.bottom + 8, slot.w, 10)
        pygame.draw.rect(screen, (0, 0, 0), pb)
        pygame.draw.rect(screen, (0, 200, 0), (pb.x + 1, pb.y + 1, int((pb.w - 2) * progress), pb.h - 2))

        # Percentage text
        pct_txt = f"Creating {int(progress * 100)}%"
        screen.blit(small_font.render(pct_txt, True, BLACK), (slot.x + slot.w + 10, slot.y + 6))
        if unit_cls is not None:
            screen.blit(small_font.render(unit_cls.__name__, True, BLACK), (slot.x + slot.w + 10, slot.y + 24))

    # --- RIGHT: stats ---
    stats_lines = []
    for label, attr in [
        ("Armor", "armor"),
        ("View", "view_distance"),
        ("Attack", "attack_damage"),
        ("Range", "attack_range"),
    ]:
        if hasattr(building, attr):
            stats_lines.append(f"{label}: {getattr(building, attr)}")

    stats_x = icon_rect.right + 12
    stats_y = icon_rect.y
    for ln in stats_lines:
        screen.blit(small_font.render(ln, True, BLACK), (stats_x, stats_y))
        stats_y += small_font.get_height() + 4


def draw_selected_unit_panel(
    screen: pygame.Surface,
    unit: Unit,
    *,
    fonts: dict,
):
    """AoE2-like info panel for a single selected unit."""
    font = fonts.get("font") or pygame.font.SysFont(None, 24)
    small_font = fonts.get("small_font") or pygame.font.SysFont(None, 16)

    pad = 8
    x0 = VIEW_MARGIN_LEFT + 270
    y0 = PANEL_Y + pad
    w = VIEW_WIDTH - 500
    h = PANEL_HEIGHT - 2 * pad
    rect = pygame.Rect(x0, y0, w, h)

    # Background (shade of gray for now)
    pygame.draw.rect(screen, (170, 170, 170), rect)
    pygame.draw.rect(screen, (110, 110, 110), rect, 2)

    # LEFT block (name, icon, HP)
    left_w = 260
    left = pygame.Rect(rect.x + pad, rect.y + pad, left_w - pad, rect.h - 2 * pad)
    stats = pygame.Rect(left.right + pad, rect.y + pad, rect.right - (left.right + 2 * pad), rect.h - 2 * pad)

    name = getattr(unit, "name", "") or unit.__class__.__name__
    screen.blit(font.render(name, True, BLACK), (left.x, left.y))

    cls_name = unit.__class__.__name__
    icon = Unit._unit_icons.get(cls_name)
    if icon is None:
        Unit.load_images(cls_name, UNIT_SIZE)
        icon = Unit._unit_icons.get(cls_name)

    icon_size = 54
    icon_rect = pygame.Rect(left.x, left.y + 28, icon_size, icon_size)
    pygame.draw.rect(screen, (60, 60, 60), icon_rect, 1)
    if icon:
        img = pygame.transform.smoothscale(icon, (icon_size, icon_size))
        screen.blit(img, icon_rect.topleft)

    # HP bar (only if hp fields exist)
    if hasattr(unit, "hp") and hasattr(unit, "max_hp"):
        hp = float(getattr(unit, "hp", 0))
        max_hp = float(getattr(unit, "max_hp", 1) or 1)
        pct = 0.0 if max_hp <= 0 else max(0.0, min(1.0, hp / max_hp))

        bar_w = icon_rect.w
        bar_h = 10
        bar_x = icon_rect.x
        bar_y = icon_rect.bottom + 8

        pygame.draw.rect(screen, (0, 0, 0), (bar_x, bar_y, bar_w, bar_h))
        fill_w = int((bar_w - 2) * pct)
        pygame.draw.rect(screen, _hp_color(pct), (bar_x + 1, bar_y + 1, fill_w, bar_h - 2))

        hp_text = f"{int(hp)}/{int(max_hp)}"
        screen.blit(small_font.render(hp_text, True, BLACK), (bar_x, bar_y + bar_h + 4))

    # STATS (to the right of the icon, AoE2-ish)
    stats_lines: list[str] = []

    # Common combat/movement stats (only if present)
    for label, attr in [
        ("Armor", "armor"),
        ("Attack", "attack_damage"),
        ("Range", "attack_range"),
        ("Cooldown", "attack_cooldown"),
        ("Speed", "speed"),
        ("View", "view_distance"),
    ]:
        if hasattr(unit, attr):
            stats_lines.append(f"{label}: {getattr(unit, attr)}")

    # Special resource/ability value (generic)
    if hasattr(unit, "special"):
        special_val = getattr(unit, "special")
        # Optional nicer labels for known units
        if isinstance(unit, Cow):
            stats_lines.append(f"Milk: {int(special_val)}")
        elif isinstance(unit, Axeman):
            stats_lines.append(f"Wood: {int(special_val)}")
        else:
            stats_lines.append(f"Special: {special_val}")

    # Mana (if present)
    if hasattr(unit, "mana"):
        mana = getattr(unit, "mana")
        max_mana = getattr(unit, "max_mana", None)
        if max_mana is not None:
            stats_lines.append(f"Mana: {int(mana)}/{int(max_mana)}")
        else:
            stats_lines.append(f"Mana: {int(mana)}")

    sx = icon_rect.right + 12
    sy = icon_rect.y
    for ln in stats_lines:
        screen.blit(small_font.render(ln, True, BLACK), (sx, sy))
        sy += small_font.get_height() + 4



def draw_selected_entity_panel(
    screen: pygame.Surface,
    all_units: list,
    current_player,
    production_queues: dict,
    current_time: float,
    fonts: dict,
    *,
    icon_size: int = 32,
    icon_margin: int = 5,
):
    """Draw selected info: AoE2-like building panel when one building is selected, else fallback icons."""
    if not current_player:
        return

    selected = [u for u in all_units if getattr(u, "selected", False)]
    # hide under-construction buildings
    selected = [u for u in selected if not (isinstance(u, Building) and getattr(u, "alpha", 255) < 255)]

    # If exactly one entity selected -> AoE2-like panel
    if len(selected) == 1:
        if isinstance(selected[0], Building):
            draw_selected_building_panel(
                screen,
                selected[0],
                production_queues=production_queues,
                current_time=current_time,
                fonts=fonts,
            )
            return
        else:
            draw_selected_unit_panel(
                screen,
                selected[0],
                fonts=fonts,
            )
            return

    # Fallback: original icon strip + single-unit details
    small_font = fonts["small_font"]
    icon_x = VIEW_MARGIN_LEFT + 350
    icon_y = PANEL_Y + 10

    for unit in selected:
        cls_name = unit.__class__.__name__
        unit_icon_img = Unit._unit_icons.get(cls_name)
        if unit_icon_img:
            screen.blit(unit_icon_img, (icon_x, icon_y))
        else:
            pygame.draw.rect(screen, WHITE, (icon_x, icon_y, icon_size, icon_size))

        if len(selected) == 1:
            display_text = f"{getattr(unit, 'name', '') or cls_name} - {cls_name}"
            color = getattr(unit, "player_color", BLACK)
            screen.blit(small_font.render(display_text, True, color), (icon_x, icon_y + icon_size + 5))
            if hasattr(unit, "hp") and hasattr(unit, "max_hp"):
                screen.blit(small_font.render(f"HP: {int(unit.hp)}/{int(unit.max_hp)}", True, color), (icon_x, icon_y + icon_size + 20))
            if hasattr(unit, "attack_damage"):
                screen.blit(small_font.render(f"Attack: {unit.attack_damage}", True, color), (icon_x, icon_y + icon_size + 35))
            if hasattr(unit, "armor"):
                screen.blit(small_font.render(f"Armor: {unit.armor}", True, color), (icon_x, icon_y + icon_size + 50))
            if hasattr(unit, "speed"):
                screen.blit(small_font.render(f"Speed: {unit.speed}", True, color), (icon_x, icon_y + icon_size + 65))
            if isinstance(unit, Cow):
                screen.blit(small_font.render(f"Milk: {int(unit.special)}", True, color), (icon_x, icon_y + icon_size + 80))
            if isinstance(unit, Axeman) and int(unit.special) > 0:
                screen.blit(small_font.render(f"Wood: {int(unit.special)}", True, color), (icon_x, icon_y + icon_size + 80))

        icon_x += icon_size + icon_margin


def draw_resources_and_limits(screen, current_player, icons, fonts):
    if not current_player:
        return

    font = fonts["font"]

    screen.blit(icons["milk"], (VIEW_MARGIN_LEFT + 10, 10))
    screen.blit(
        font.render(f": {current_player.milk:.0f}/{current_player.max_milk}", True, current_player.color),
        (VIEW_MARGIN_LEFT + 30, 15),
    )

    screen.blit(icons["wood"], (VIEW_MARGIN_LEFT + 120, 10))
    screen.blit(
        font.render(f": {current_player.wood:.0f}/{current_player.max_wood}", True, current_player.color),
        (VIEW_MARGIN_LEFT + 140, 15),
    )

    if current_player.unit_limit is not None:
        screen.blit(icons["unit"], (VIEW_MARGIN_LEFT + 230, 10))
        text_color = ORANGE if current_player.unit_count > current_player.unit_limit else current_player.color
        screen.blit(
            font.render(f": {current_player.unit_count}/{current_player.unit_limit}", True, text_color),
            (VIEW_MARGIN_LEFT + 250, 15),
        )

    if current_player.building_limit is not None:
        screen.blit(icons["building"], (VIEW_MARGIN_LEFT + 320, 10))
        text_color = ORANGE if current_player.building_count > current_player.building_limit else current_player.color
        screen.blit(
            font.render(f": {current_player.building_count}/{current_player.building_limit}", True, text_color),
            (VIEW_MARGIN_LEFT + 340, 15),
        )


def draw_fps(screen, fps, fonts):
    screen.blit(fonts["font"].render(f"FPS: {int(fps)}", True, WHITE), (VIEW_MARGIN_LEFT + VIEW_WIDTH - 80, VIEW_MARGIN_TOP + 10))


def draw_selected_count(screen, all_units, current_player, fonts):
    if not current_player:
        return
    selected_count = sum(1 for u in all_units if u.selected and u.player_id == current_player.player_id)
    screen.blit(
        fonts["font"].render(f"Selected Units: {selected_count}", True, BLACK),
        (VIEW_MARGIN_LEFT + 10, PANEL_Y + 135),
    )


def draw_end_screen(screen, mode_text, quit_button, fonts):
    """mode_text is ('Defeat!...', bg_color) or ('Victory!...', bg_color)."""
    text, bg = mode_text
    screen.fill(bg)
    font = fonts["font"]
    button_font = fonts["end_button_font"]

    msg = font.render(text, True, WHITE)
    msg_rect = msg.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 50))
    screen.blit(msg, msg_rect)

    pygame.draw.rect(screen, GRAY, quit_button)
    quit_text = button_font.render("Quit", True, BLACK)
    quit_rect = quit_text.get_rect(center=quit_button.center)
    screen.blit(quit_text, quit_rect)


def draw_game_ui(
    screen,
    grid_buttons,
    current_player,
    production_queues,
    current_time,
    all_units,
    icons,
    fonts,
    fps,
    *,
    grass_tiles,
    camera,
):
    draw_panels(screen)
    draw_grid_buttons(screen, grid_buttons, current_player, all_units, production_queues, current_time, icons, fonts)

    # --- NEW: minimap bottom-right (inside right panel) ---
    draw_minimap(screen, grass_tiles=grass_tiles, all_units=all_units, camera=camera, current_player=current_player)

    # Totem separator between the left build grid and the selected-unit info area
    if icons and icons.get("totem"):
        totem = icons["totem"]
        tx = VIEW_MARGIN_LEFT - (totem.get_width() // 2) + 250
        ty = PANEL_Y + (PANEL_HEIGHT - totem.get_height()) // 2
        screen.blit(totem, (tx, ty))

    draw_selected_entity_panel(
        screen,
        all_units,
        current_player,
        production_queues,
        current_time,
        fonts,
        icon_size=32,
        icon_margin=5,
    )
    draw_resources_and_limits(screen, current_player, icons, fonts)
    draw_fps(screen, fps, fonts)
    # draw_selected_count(screen, all_units, current_player, fonts)

def _minimap_rect():
    # Right panel bounds: x in [SCREEN_WIDTH - VIEW_MARGIN_RIGHT, SCREEN_WIDTH)
    pad = 10
    panel_x = SCREEN_WIDTH - VIEW_MARGIN_RIGHT
    panel_w = VIEW_MARGIN_RIGHT

    # Choose a square minimap that fits comfortably
    size = min(panel_w - 2 * pad, 220)
    size = max(180, size)  # avoid too tiny

    x = panel_x + panel_w - pad - size
    y = SCREEN_HEIGHT - pad - size
    return pygame.Rect(x, y, size, size)


def _minimap_base_surface(grass_tiles, size):
    """
    Build (and cache) a low-res terrain thumbnail.
    Rebuilt only if (a) map object changes or (b) size changes.
    """
    if not hasattr(_minimap_base_surface, "_cache"):
        _minimap_base_surface._cache = {}  # (id(grass_tiles), size) -> Surface

    key = (id(grass_tiles), size)
    if key in _minimap_base_surface._cache:
        return _minimap_base_surface._cache[key]

    # Build a small surface; we sample the world grid into this thumbnail
    surf = pygame.Surface((size, size))
    surf.fill((0, 0, 0))

    rows = len(grass_tiles)
    cols = len(grass_tiles[0]) if rows else 0
    if rows == 0 or cols == 0:
        _minimap_base_surface._cache[key] = surf
        return surf

    # Map thumbnail pixels -> world tiles (nearest sampling)
    # This is O(size^2) once, which is cheap at ~200x200.
    for py in range(size):
        row = int(py * rows / size)
        for px in range(size):
            col = int(px * cols / size)
            t = grass_tiles[row][col]

            # Try to use tile.color if it exists, else fall back by class name
            c = getattr(t, "color", None)
            if c is not None:
                rgb = c[:3]
            else:
                name = t.__class__.__name__
                if "River" in name:
                    rgb = (40, 90, 180)
                elif "Bridge" in name:
                    rgb = (120, 100, 60)
                elif "Dirt" in name:
                    rgb = (110, 80, 40)
                else:
                    rgb = (40, 140, 60)  # grass-ish fallback

            surf.set_at((px, py), rgb)

    _minimap_base_surface._cache[key] = surf
    return surf

def minimap_screen_to_world(pos) -> Vector2 | None:
    """
    If `pos` (screen pixels) is inside the minimap, return the corresponding
    world-space point (pixels). Otherwise return None.
    """
    mm = _minimap_rect()

    # Accept either tuple (x,y) or Vector2
    if hasattr(pos, "x"):
        sx, sy = int(pos.x), int(pos.y)
    else:
        sx, sy = int(pos[0]), int(pos[1])

    if not mm.collidepoint(sx, sy):
        return None

    nx = (sx - mm.x) / mm.w
    ny = (sy - mm.y) / mm.h

    # Map normalized minimap coord -> world pixels
    wx = max(0.0, min(float(MAP_WIDTH), nx * MAP_WIDTH))
    wy = max(0.0, min(float(MAP_HEIGHT), ny * MAP_HEIGHT))
    return Vector2(wx, wy)

def draw_minimap(screen, *, grass_tiles, all_units, camera, current_player):
    mm = _minimap_rect()

    # Base terrain (cached)
    base = _minimap_base_surface(grass_tiles, mm.w)
    screen.blit(base, mm.topleft)

    # -------------------------------
    # ✅ 1. Draw unit/world dots FIRST
    # -------------------------------
    for u in all_units:
        ux = mm.x + int((u.pos.x / MAP_WIDTH) * mm.w)
        uy = mm.y + int((u.pos.y / MAP_HEIGHT) * mm.h)

        if mm.collidepoint(ux, uy):

            # ✅ worldObject uses minimapColor
            if getattr(u, "worldObject", False):
                col = getattr(u, "minimapColor", (255, 255, 255))
            else:
                col = getattr(u, "player_color", (255, 255, 255))

            pygame.draw.circle(screen, col, (ux, uy), 2)

    # ---------------------------------
    # ✅ 2. Draw camera rectangle ON TOP
    # ---------------------------------
    cam_x = max(0, min(camera.x, MAP_WIDTH - VIEW_WIDTH))
    cam_y = max(0, min(camera.y, MAP_HEIGHT - VIEW_HEIGHT))

    vx = mm.x + int((cam_x / MAP_WIDTH) * mm.w)
    vy = mm.y + int((cam_y / MAP_HEIGHT) * mm.h)
    vw = max(2, int((VIEW_WIDTH / MAP_WIDTH) * mm.w))
    vh = max(2, int((VIEW_HEIGHT / MAP_HEIGHT) * mm.h))

    pygame.draw.rect(
        screen,
        (255, 255, 255),
        pygame.Rect(vx, vy, vw, vh),
        1
    )

    # ---------------------------------
    # ✅ 3. Draw minimap border LAST
    # ---------------------------------
    pygame.draw.rect(screen, (240, 240, 240), mm, 2)