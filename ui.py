import pygame
from pygame.math import Vector2

from constants import *
from units import Unit, Building, Barn, Barracks, TownCenter, Axeman, Archer, Knight, Cow, ShamansHut


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

        wood_icon = pygame.transform.scale(wood_icon, (20, 20))
        milk_icon = pygame.transform.scale(milk_icon, (20, 20))
        unit_icon = pygame.transform.scale(unit_icon, (20, 20))
        building_icon = pygame.transform.scale(building_icon, (20, 20))
    except (pygame.error, FileNotFoundError):
        wood_icon = pygame.Surface((20, 20))
        milk_icon = pygame.Surface((20, 20))
        unit_icon = pygame.Surface((20, 20))
        building_icon = pygame.Surface((20, 20))
        wood_icon.fill(BROWN)
        milk_icon.fill(WHITE)
        unit_icon.fill(LIGHT_GRAY)
        building_icon.fill(LIGHT_GRAY)

    return {
        "wood": wood_icon,
        "milk": milk_icon,
        "unit": unit_icon,
        "building": building_icon,
    }


def create_fonts():
    """Create and return common fonts."""
    return {
        "font": pygame.font.SysFont(None, 24),
        "small_font": pygame.font.SysFont(None, 16),
        "button_font": pygame.font.SysFont(None, 20),
        "end_button_font": pygame.font.Font(None, 36),
    }


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
    grid_total_h = 3 * GRID_BUTTON_HEIGHT + 2 * GRID_BUTTON_MARGIN

    grid_button_start_x = left_pad
    grid_button_start_y = PANEL_Y + top_pad  # inside the bottom panel, but left side

    # Make buttons readable: use a minimum width, then compute margin to fit
    btn_h = GRID_BUTTON_HEIGHT

    # Choose a good minimum width for text+icons+costs
    MIN_BTN_W = 70  # tweak: 64–80 usually good
    btn_w = max(MIN_BTN_W, (grid_total_w - (GRID_COLS - 1) * GRID_BUTTON_MARGIN) // GRID_COLS)

    # If 4*btn_w doesn't fit, reduce effective margin instead of shrinking buttons too much
    needed_w = GRID_COLS * btn_w
    remaining = grid_total_w - needed_w
    eff_margin = GRID_BUTTON_MARGIN
    if remaining < (GRID_COLS - 1) * eff_margin:
        eff_margin = max(2, remaining // (GRID_COLS - 1))  # squeeze margins, keep at least 2px

    grid_buttons = []
    for row in range(GRID_ROWS):
        row_buttons = []
        for col in range(GRID_COLS):
            x = grid_button_start_x + col * (btn_w + eff_margin)
            y = grid_button_start_y + row * (btn_h + eff_margin)
            row_buttons.append(pygame.Rect(x, y, btn_w, btn_h))
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


def draw_grid_buttons(screen, grid_buttons, current_player, all_units, production_queues, current_time, icons, small_font):

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
                # enabled = LIGHT_GRAY, disabled = GRAY
                pygame.draw.rect(screen, HIGHLIGHT_GRAY if enabled else GRAY, btn)
                draw_label(btn, label, hotkeys.get((r, c)))

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

                pygame.draw.rect(screen, HIGHLIGHT_GRAY if enabled else GRAY, btn)

                # Ensure icon is loaded even if no instance exists yet
                if cls.__name__ not in Unit._unit_icons or Unit._unit_icons.get(cls.__name__) is None:
                    Unit.load_images(
                        cls.__name__,
                        BUILDING_SIZE if issubclass(cls, Building) else UNIT_SIZE,
                    )

                icon = Unit._unit_icons.get(cls.__name__)

                if icon:
                    icon_75 = get_scaled(icon, 0.75)
                    screen.blit(
                        icon_75,
                        (btn.x + btn.width - icon_75.get_width() -2, btn.y + 14),
                    )

                draw_label(btn, label, enabled=True)  # label always visible
                draw_costs(btn, cls)
                draw_progress(btn, owner, cls)

                idx += 1


def draw_selected_unit_icons(screen, all_units, current_player, fonts, icon_size=32, icon_margin=5):
    """Draw icons for currently selected units + details if one unit selected."""
    if not current_player:
        return

    small_font = fonts["small_font"]
    icon_x = VIEW_MARGIN_LEFT + 350
    icon_y = PANEL_Y + 10

    selected_units = [u for u in all_units if u.selected]
    for unit in selected_units:
        # hide under-construction buildings
        if isinstance(unit, Building) and unit.alpha < 255:
            continue

        cls_name = unit.__class__.__name__
        unit_icon_img = Unit._unit_icons.get(cls_name)
        if unit_icon_img:
            screen.blit(unit_icon_img, (icon_x, icon_y))
        else:
            pygame.draw.rect(screen, WHITE, (icon_x, icon_y, icon_size, icon_size))

        if len(selected_units) == 1:
            display_text = f"{unit.name or cls_name} - {cls_name}"
            screen.blit(small_font.render(display_text, True, unit.player_color), (icon_x, icon_y + icon_size + 5))
            screen.blit(small_font.render(f"HP: {int(unit.hp)}/{int(unit.max_hp)}", True, unit.player_color), (icon_x, icon_y + icon_size + 20))
            screen.blit(small_font.render(f"Attack: {unit.attack_damage}", True, unit.player_color), (icon_x, icon_y + icon_size + 35))
            screen.blit(small_font.render(f"Armor: {unit.armor}", True, unit.player_color), (icon_x, icon_y + icon_size + 50))
            screen.blit(small_font.render(f"Speed: {unit.speed}", True, unit.player_color), (icon_x, icon_y + icon_size + 65))
            if isinstance(unit, Cow):
                screen.blit(small_font.render(f"Milk: {int(unit.special)}", True, unit.player_color), (icon_x, icon_y + icon_size + 80))
            if isinstance(unit, Axeman) and int(unit.special) > 0:
                screen.blit(small_font.render(f"Wood: {int(unit.special)}", True, unit.player_color), (icon_x, icon_y + icon_size + 80))

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


def draw_game_ui(screen, grid_buttons, current_player, production_queues, current_time, all_units, icons, fonts, fps):
    draw_panels(screen)
    draw_grid_buttons(screen, grid_buttons, current_player, all_units, production_queues, current_time, icons, fonts["small_font"])
    draw_selected_unit_icons(
        screen,
        all_units,
        current_player,
        fonts,
        icon_size=32,
        icon_margin=5,
    )
    draw_resources_and_limits(screen, current_player, icons, fonts)
    draw_fps(screen, fps, fonts)
    draw_selected_count(screen, all_units, current_player, fonts)
