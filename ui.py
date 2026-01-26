import pygame
from pygame.math import Vector2

from constants import *
from units import Unit, Building, Barn, Barracks, TownCenter, Axeman, Archer, Knight, Cow


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

    grid_button_start_x = VIEW_BOUNDS_X - GRID_BUTTON_WIDTH - BUTTON_MARGIN
    grid_button_start_y = PANEL_Y + (
        PANEL_HEIGHT - (GRID_BUTTON_ROWS * GRID_BUTTON_HEIGHT + (GRID_BUTTON_ROWS - 1) * GRID_BUTTON_MARGIN)
    ) // 2

    grid_buttons = []
    for row in range(GRID_BUTTON_ROWS):
        row_buttons = []
        for col in range(GRID_BUTTON_COLS):
            x = grid_button_start_x - col * (GRID_BUTTON_WIDTH + GRID_BUTTON_MARGIN)
            y = grid_button_start_y + row * (GRID_BUTTON_HEIGHT + GRID_BUTTON_MARGIN)
            row_buttons.append(pygame.Rect(x, y, GRID_BUTTON_WIDTH, GRID_BUTTON_HEIGHT))
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
    return selected_barn, selected_barracks, selected_town_center


def draw_panels(screen):
    pygame.draw.rect(screen, PANEL_COLOR, (0, 0, VIEW_MARGIN_LEFT, SCREEN_HEIGHT))
    pygame.draw.rect(screen, PANEL_COLOR, (SCREEN_WIDTH - VIEW_MARGIN_RIGHT, 0, VIEW_MARGIN_RIGHT, SCREEN_HEIGHT))
    pygame.draw.rect(screen, PANEL_COLOR, (0, 0, SCREEN_WIDTH, VIEW_MARGIN_TOP))
    pygame.draw.rect(screen, PANEL_COLOR, (VIEW_MARGIN_LEFT, PANEL_Y, VIEW_WIDTH, PANEL_HEIGHT))


def draw_grid_buttons(screen, grid_buttons, current_player, production_queues, current_time, icons, small_font):
    """Draw the 3x1 grid and its contextual contents (unit/building actions)."""
    selected_barn, selected_barracks, selected_town_center = _get_selected_buildings(current_player)

    for row in range(GRID_BUTTON_ROWS):
        for col in range(GRID_BUTTON_COLS):
            button_rect = grid_buttons[row][col]

            def draw_costs(unit_cls):
                screen.blit(icons["milk"], (button_rect.x + 44, button_rect.y + 2))
                screen.blit(small_font.render(f"{unit_cls.milk_cost}", True, BLACK), (button_rect.x + 70, button_rect.y + 6))
                screen.blit(icons["wood"], (button_rect.x + 44, button_rect.y + 22))
                screen.blit(small_font.render(f"{unit_cls.wood_cost}", True, BLACK), (button_rect.x + 70, button_rect.y + 26))

            def draw_progress(queue_owner, unit_cls):
                if queue_owner in production_queues and production_queues[queue_owner]["unit_type"] == unit_cls:
                    progress = (current_time - production_queues[queue_owner]["start_time"]) / unit_cls.production_time
                    progress_width = int(progress * (GRID_BUTTON_WIDTH - 4))
                    pygame.draw.rect(screen, GREEN, (button_rect.x + 2, button_rect.y + 2, progress_width, 4))

            # Barn -> Cow (row 0)
            if current_player and selected_barn and row == 0 and col == 0:
                enabled = (
                    current_player.milk >= Cow.milk_cost
                    and current_player.wood >= Cow.wood_cost
                    and selected_barn not in production_queues
                )
                pygame.draw.rect(screen, HIGHLIGHT_GRAY if enabled else GRAY, button_rect)
                screen.blit(Unit._unit_icons.get("Cow"), (button_rect.x + 8, button_rect.y + 4))
                draw_costs(Cow)
                draw_progress(selected_barn, Cow)

            # Barracks -> Axeman/Archer/Knight (rows 0..2)
            elif current_player and selected_barracks and col == 0 and row in (0, 1, 2):
                mapping = {0: Axeman, 1: Archer, 2: Knight}
                unit_cls = mapping[row]
                enabled = (
                    current_player.milk >= unit_cls.milk_cost
                    and current_player.wood >= unit_cls.wood_cost
                    and selected_barracks not in production_queues
                )
                pygame.draw.rect(screen, HIGHLIGHT_GRAY if enabled else GRAY, button_rect)
                screen.blit(Unit._unit_icons.get(unit_cls.__name__), (button_rect.x + 8, button_rect.y + 4))
                draw_costs(unit_cls)
                draw_progress(selected_barracks, unit_cls)

            # TownCenter -> place Barn/Barracks/TownCenter (rows 0..2)
            elif current_player and selected_town_center and col == 0 and row in (0, 1, 2):
                mapping = {0: Barn, 1: Barracks, 2: TownCenter}
                build_cls = mapping[row]
                enabled = (
                    current_player.milk >= build_cls.milk_cost
                    and current_player.wood >= build_cls.wood_cost
                    and (current_player.building_limit is None or current_player.building_count < current_player.building_limit)
                    and selected_town_center not in production_queues
                )
                pygame.draw.rect(screen, HIGHLIGHT_GRAY if enabled else GRAY, button_rect)
                screen.blit(Unit._unit_icons.get(build_cls.__name__), (button_rect.x + 8, button_rect.y + 4))
                draw_costs(build_cls)
                draw_progress(selected_town_center, build_cls)

            else:
                pygame.draw.rect(screen, LIGHT_GRAY, button_rect)


def draw_selected_unit_icons(screen, all_units, current_player, fonts, icon_size=32, icon_margin=5):
    """Draw icons for currently selected units + details if one unit selected."""
    if not current_player:
        return

    small_font = fonts["small_font"]
    icon_x = VIEW_MARGIN_LEFT + 10
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
    draw_grid_buttons(screen, grid_buttons, current_player, production_queues, current_time, icons, fonts["small_font"])
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
