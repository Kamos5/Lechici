# grid_actions.py
import pygame
from pygame.math import Vector2

from units import Building, Barn, Barracks, TownCenter, Axeman, Archer, Knight, Cow, ShamansHut

# 4x3 hotkeys: q w e r / a s d f / z x c v
GRID_HOTKEYS = {
    pygame.K_q: (0, 0), pygame.K_w: (0, 1), pygame.K_e: (0, 2), pygame.K_r: (0, 3),
    pygame.K_a: (1, 0), pygame.K_s: (1, 1), pygame.K_d: (1, 2), pygame.K_f: (1, 3),
    pygame.K_z: (2, 0), pygame.K_x: (2, 1), pygame.K_c: (2, 2), pygame.K_v: (2, 3),
}


def cell_from_mouse(grid_buttons, mouse_pos) -> tuple[int, int] | None:
    """Return (row, col) if mouse_pos hits a grid rect, else None."""
    for r in range(len(grid_buttons)):
        for c in range(len(grid_buttons[r])):
            if grid_buttons[r][c].collidepoint(mouse_pos):
                return (r, c)
    return None


def cell_from_key(key) -> tuple[int, int] | None:
    """Return (row, col) if key is mapped to a grid cell, else None."""
    return GRID_HOTKEYS.get(key)


def _get_selected_buildings(current_player):
    """Return (selected_barn, selected_barracks, selected_town_center) for this player."""
    if not current_player:
        return None, None, None
    selected_barn = next((u for u in current_player.units if isinstance(u, Barn) and u.selected and u.alpha == 255), None)
    selected_barracks = next((u for u in current_player.units if isinstance(u, Barracks) and u.selected and u.alpha == 255), None)
    selected_town_center = next((u for u in current_player.units if isinstance(u, TownCenter) and u.selected and u.alpha == 255), None)
    selected_shamans_hut = next((u for u in current_player.units if isinstance(u, ShamansHut) and u.selected and u.alpha == 255), None)
    return selected_barn, selected_barracks, selected_town_center, selected_shamans_hut


def execute_grid_cell(
    r: int,
    c: int,
    *,
    current_player,
    grid_buttons,
    production_queues: dict,
    current_time: float,
    placing_building: bool,
    building_to_place,
):
    """
    Executes whatever the grid cell (r,c) means, based on selection:
      - If any units selected -> unit command buttons (Patrol/Move/Harvest/Repair/Attack/Stop)
      - Else -> production options based on selected building(s)

    Returns:
      handled (bool), placing_building (bool), building_to_place (cls|None)
    """
    if not current_player:
        return False, placing_building, building_to_place

    # Which things are selected?
    selected_owned = [u for u in current_player.units if u.selected]
    selected_units = [u for u in selected_owned if not isinstance(u, Building)]
    show_unit_commands = len(selected_units) > 0

    # ---- UNIT COMMAND MODE ----
    if show_unit_commands:
        has_axeman = any(isinstance(u, Axeman) for u in selected_units)
        has_cow = any(isinstance(u, Cow) for u in selected_units)

        # Patrol (Q)
        if (r, c) == (0, 0):
            # TODO: implement patrol
            print("Patrol pressed (TODO)")
            return True, placing_building, building_to_place

        # Move (W)
        if (r, c) == (0, 1):
            # TIP: your right-click already does move/attack contextually
            print("Move pressed")
            return True, placing_building, building_to_place

        # Harvest (E) - only cows + axemen
        if (r, c) == (0, 2):
            if has_axeman or has_cow:
                # TODO: implement harvest mode
                print("Harvest pressed (TODO)")
                return True, placing_building, building_to_place
            return False, placing_building, building_to_place

        # Repair (R) - only axemen
        if (r, c) == (0, 3):
            if has_axeman:
                # TODO: implement repair mode
                print("Repair pressed (TODO)")
                return True, placing_building, building_to_place
            return False, placing_building, building_to_place

        # Attack (A)
        if (r, c) == (1, 0):
            # TODO: implement attack-only mode
            print("Attack pressed (TODO)")
            return True, placing_building, building_to_place

        # Stop (S)
        if (r, c) == (1, 1):
            for u in selected_units:
                u.target = None
                u.autonomous_target = False
                if hasattr(u, "path"):
                    u.path = []
                if hasattr(u, "path_index"):
                    u.path_index = 0
            return True, placing_building, building_to_place

        return False, placing_building, building_to_place

    # ---- PRODUCTION MODE ----
    selected_barn, selected_barracks, selected_town_center, selected_shamans_hut = _get_selected_buildings(current_player)

    options = []
    if selected_barn:
        options.append((Cow, selected_barn, "unit"))
    if selected_barracks:
        options.extend([
            (Axeman, selected_barracks, "unit"),
            (Archer, selected_barracks, "unit"),
            (Knight, selected_barracks, "unit"),
        ])
    if selected_town_center:
        options.extend([
            (Barn, selected_town_center, "building"),
            (Barracks, selected_town_center, "building"),
            (TownCenter, selected_town_center, "building"),
            (ShamansHut, selected_town_center, "building"),
        ])

    # (r,c) -> linear idx in packed grid
    idx = r * len(grid_buttons[0]) + c
    if not (0 <= idx < len(options)):
        return False, placing_building, building_to_place

    cls, owner, kind = options[idx]

    if owner in production_queues:
        print(f"{owner.__class__.__name__} at {owner.pos} is already producing")
        return True, placing_building, building_to_place

    # affordability check (for units & buildings)
    if current_player.milk < cls.milk_cost or current_player.wood < cls.wood_cost:
        print(f"Cannot queue {cls.__name__}: milk={current_player.milk}/{cls.milk_cost}, wood={current_player.wood}/{cls.wood_cost}")
        return True, placing_building, building_to_place

    # buildings from TownCenter: start placement, DO NOT deduct yet
    if kind == "building" and issubclass(cls, Building):
        if current_player.building_limit is not None and current_player.building_count >= current_player.building_limit:
            print("Cannot place building: building limit reached")
            return True, placing_building, building_to_place

        placing_building = True
        building_to_place = cls
        print(f"Initiated placement of {cls.__name__} for Player {current_player.player_id}")
        return True, placing_building, building_to_place

    # units: queue immediately and deduct immediately (as your current behavior)
    production_queues[owner] = {
        "unit_type": cls,
        "start_time": current_time,
        "player_id": current_player.player_id
    }
    current_player.milk -= cls.milk_cost
    current_player.wood -= cls.wood_cost
    print(f"Started production of {cls.__name__} at {owner.__class__.__name__} {owner.pos} for Player {current_player.player_id}")
    return True, placing_building, building_to_place
