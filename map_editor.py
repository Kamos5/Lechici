# map_editor.py
# Map editor: paint tiles + place unit SPRITES for chosen player.
# Features:
# - Clean borders only where tile types differ
# - View snapped to tile grid, no half-tile clipping
# - Camera: arrow keys only
# - Scrollbars (top + right) show camera position
# - Buildings occupy multi-tile footprint and block placement
# - Tree button cycles variants tree0..tree6 on repeated clicks
# - Tree placements store {"variant": "treeN"} and load/save it

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Type, Any

import pygame

from constants import *  # SCREEN_WIDTH/HEIGHT, TILE_SIZE, VIEW_* etc.
from tiles import GrassTile, Dirt, River, Bridge, Foundation
from units import Unit, Axeman, Knight, Archer, Cow, Tree, Barn, TownCenter, Barracks, ShamansHut, get_team_sprite

# -----------------------------
# Config
# -----------------------------
DEFAULT_SAVE_PATH = "maps/editor_map.json"

# Tree variants for the editor preview + saved metadata
_TREE_EDITOR_IMAGES: Dict[Tuple[str, int], Optional[pygame.Surface]] = {}

# River variants for the editor + saved metadata
RIVER_VARIANT_PREFIX = "river"  # assets/river/river1.png ... river9.png
_RIVER_EDITOR_IMAGES: Dict[Tuple[str, int], Optional[pygame.Surface]] = {}

# Tiles available to paint
TILE_TYPES: Dict[str, Type] = {
    "GrassTile": GrassTile,
    "Dirt": Dirt,
    "River": River,
    "Bridge": Bridge,
    "Foundation": Foundation,
}

# Units available to place (as sprites)
UNIT_TYPES: Dict[str, Type] = {
    "Axeman": Axeman,
    "Knight": Knight,
    "Archer": Archer,
    "Cow": Cow,
    "Tree": Tree,
    "Barn": Barn,
    "TownCenter": TownCenter,
    "Barracks": Barracks,
    "ShamansHut": ShamansHut,
}

# UI grouping (Tree is in bottom row with buildings)
UNIT_ROW: List[str] = ["Axeman", "Knight", "Archer", "Cow"]
BOTTOM_ROW: List[str] = ["Tree", "Barn", "TownCenter", "Barracks", "ShamansHut"]

# Editor players (id -> color)
PLAYERS: List[Tuple[str, Tuple[int, int, int]]] = [
    ("GAIA", WHITE_GRAY),
    ("P1", BLUE),
    ("P2", PURPLE),
]


# -----------------------------
# UI button
# -----------------------------
@dataclass
class Button:
    rect: pygame.Rect
    label: str
    kind: str  # "mode"|"tile"|"unit"|"player"|"action"
    value: str

    def hit(self, pos: Tuple[int, int]) -> bool:
        return self.rect.collidepoint(pos)


def ensure_dirs(path: str) -> None:
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def load_tree_editor_image(variant: str, desired_px: int) -> Optional[pygame.Surface]:
    """
    Loads assets/{variant}.png (e.g. assets/tree0.png), scaled to desired_px.
    Cached by (variant, desired_px).
    """
    key = (variant, desired_px)
    if key in _TREE_EDITOR_IMAGES:
        return _TREE_EDITOR_IMAGES[key]

    path = f"assets/{variant}.png"
    try:
        img = pygame.image.load(path).convert_alpha()
        scale = desired_px / max(img.get_width(), img.get_height())
        img = pygame.transform.smoothscale(
            img,
            (max(1, int(img.get_width() * scale)), max(1, int(img.get_height() * scale))),
        )
        _TREE_EDITOR_IMAGES[key] = img
        return img
    except Exception as e:
        print(f"[EDITOR] Failed to load {path}: {e}")
        _TREE_EDITOR_IMAGES[key] = None
        return None

def load_river_editor_image(variant: str, desired_px: int) -> Optional[pygame.Surface]:
    """
    Loads assets/river/{variant}.png (e.g. assets/river/river5.png), scaled to desired_px.
    Cached by (variant, desired_px).
    """
    key = (variant, desired_px)
    if key in _RIVER_EDITOR_IMAGES:
        return _RIVER_EDITOR_IMAGES[key]

    path = f"assets/river/{variant}.png"
    try:
        img = pygame.image.load(path).convert_alpha()
        img = pygame.transform.smoothscale(img, (desired_px, desired_px))
        _RIVER_EDITOR_IMAGES[key] = img
        return img
    except Exception as e:
        print(f"[EDITOR] Failed to load {path}: {e}")
        _RIVER_EDITOR_IMAGES[key] = None
        return None

# -----------------------------
# Map data helpers
# -----------------------------
def new_grass_map() -> List[List[object]]:
    grid: List[List[object]] = []
    for r in range(GRASS_ROWS):
        row = []
        for c in range(GRASS_COLS):
            row.append(GrassTile(c * TILE_SIZE, r * TILE_SIZE))
        grid.append(row)
    return grid


def set_tile(grid: List[List[object]], row: int, col: int, tile_name: str, variant: Optional[str] = None) -> None:
    tile_cls = TILE_TYPES[tile_name]
    if tile_name == "River":
        grid[row][col] = tile_cls(col * TILE_SIZE, row * TILE_SIZE, variant=variant)
    else:
        grid[row][col] = tile_cls(col * TILE_SIZE, row * TILE_SIZE)


def world_to_tile(world_x: int, world_y: int) -> Tuple[int, int]:
    col = int(world_x // TILE_SIZE)
    row = int(world_y // TILE_SIZE)
    return row, col


def tile_center(row: int, col: int) -> Tuple[int, int]:
    x = col * TILE_SIZE + TILE_HALF
    y = row * TILE_SIZE + TILE_HALF
    return x, y


# -----------------------------
# Sprite loading & drawing
# -----------------------------
def _ensure_unit_sprite_loaded(cls_name: str, desired_px: int, player_color=None) -> Optional[pygame.Surface]:
    img = Unit._images.get(cls_name)

    def max_dim(surf: pygame.Surface) -> int:
        w, h = surf.get_size()
        return max(w, h)

    # Load if missing
    if img is None:
        Unit.load_images(cls_name, desired_px)
        return Unit._images.get(cls_name)

    # If cached size is different, reload at correct size (overwrites cache)
    try:
        if max_dim(img) != desired_px:
            Unit.load_images(cls_name, desired_px)
    except Exception:
        pass

    return Unit._images.get(cls_name)


def draw_unit_sprite(
    surf: pygame.Surface,
    unit_type: str,
    player_color: Tuple[int, int, int],
    world_cx: int,
    world_cy: int,
    camera_x: int,
    camera_y: int,
    tree_variant: Optional[str] = None,
) -> None:
    # Buildings bigger
    if unit_type in ("Barn", "TownCenter", "Barracks", "ShamansHut"):
        desired = int(BUILDING_SIZE)
    elif unit_type == "Tree":
        desired = int(TILE_SIZE)
    else:
        desired = int(max(UNIT_SIZE, TILE_SIZE))

    if unit_type == "Tree":
        # If not provided, default to tree0
        tv = tree_variant or f"{TREE_VARIANT_PREFIX}0"
        img = load_tree_editor_image(tv, desired)
    else:
        img = _ensure_unit_sprite_loaded(unit_type, desired, player_color)

    sx = world_cx - camera_x
    sy = world_cy - camera_y

    if img is not None:
        rect = img.get_rect(center=(sx, sy))
        surf.blit(img, rect)
        # reduced border as requested earlier
        if unit_type != "Tree":
            pygame.draw.rect(surf, player_color, rect.inflate(-2, -2), 1)
    else:
        r = max(6, TILE_SIZE // 3)
        pygame.draw.circle(surf, player_color, (sx, sy), r)
        pygame.draw.circle(surf, (10, 10, 10), (sx, sy), r, 2)


# -----------------------------
# Save / Load
# -----------------------------
def save_map(grid: List[List[object]], units_by_cell: Dict[str, Dict[str, Any]], path: str) -> None:
    ensure_dirs(path)
    tiles_out = []
    for r in range(GRASS_ROWS):
        row_out = []
        for c in range(GRASS_COLS):
            t = grid[r][c]
            name = t.__class__.__name__
            if name == "River":
                row_out.append({"type": "River", "variant": getattr(t, "variant", "river5")})
            else:
                row_out.append(name)
        tiles_out.append(row_out)

    data = {
        "rows": GRASS_ROWS,
        "cols": GRASS_COLS,
        "tile_size": TILE_SIZE,
        "tiles": tiles_out,
        "units": units_by_cell,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"[EDITOR] Saved: {path}")


def load_map(path: str) -> Tuple[List[List[object]], Dict[str, Dict[str, Any]]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = int(data.get("rows", GRASS_ROWS))
    cols = int(data.get("cols", GRASS_COLS))
    if rows != GRASS_ROWS or cols != GRASS_COLS:
        raise ValueError(f"Map size mismatch. Expected {GRASS_ROWS}x{GRASS_COLS}, got {rows}x{cols}")

    tiles = data.get("tiles")
    if not tiles:
        raise ValueError("Missing tiles[] in save")

    grid = new_grass_map()
    for r in range(GRASS_ROWS):
        for c in range(GRASS_COLS):
            cell = tiles[r][c]

            # Backward compatible: old saves store strings; new saves may store dicts
            if isinstance(cell, dict):
                name = cell.get("type", "GrassTile")
                if name not in TILE_TYPES:
                    name = "GrassTile"
                if name == "River":
                    var = cell.get("variant")
                    set_tile(grid, r, c, "River", variant=var)
                else:
                    set_tile(grid, r, c, name)
            else:
                name = cell
                if name not in TILE_TYPES:
                    name = "GrassTile"
                set_tile(grid, r, c, name)

    units_raw = data.get("units", {})
    units_by_cell: Dict[str, Dict[str, Any]] = {}
    if isinstance(units_raw, dict):
        for k, v in units_raw.items():
            if not isinstance(k, str) or "," not in k or not isinstance(v, dict):
                continue

            t = v.get("type")
            p = v.get("player")
            if t not in UNIT_TYPES:
                continue

            try:
                p = int(p)
            except Exception:
                continue
            if p < 0 or p >= len(PLAYERS):
                continue

            try:
                rs, cs = k.split(",")
                rr, cc = int(rs), int(cs)
            except Exception:
                continue
            if not (0 <= rr < GRASS_ROWS and 0 <= cc < GRASS_COLS):
                continue

            entry: Dict[str, Any] = {"type": t, "player": p}

            # Preserve tree variant if present and valid
            if t == "Tree":
                var = v.get("variant")
                if isinstance(var, str) and var.startswith(TREE_VARIANT_PREFIX):
                    # accept tree0..tree6
                    try:
                        idx = int(var[len(TREE_VARIANT_PREFIX):])
                        if 0 <= idx < TREE_VARIANT_COUNT:
                            entry["variant"] = var
                    except Exception:
                        pass

            units_by_cell[k] = entry

    print(f"[EDITOR] Loaded: {path}")
    return grid, units_by_cell


# -----------------------------
# Borders / scrollbars drawing
# -----------------------------
def draw_tile_type_borders(
    surf: pygame.Surface,
    grid: List[List[object]],
    start_row: int,
    end_row: int,
    start_col: int,
    end_col: int,
    camera_x: int,
    camera_y: int,
    border_color: Tuple[int, int, int] = (0, 0, 0),
) -> None:
    """
    Draw 1px borders ONLY where adjacent tile types differ (cleaner than full grid lines).
    """
    for r in range(start_row, end_row):
        for c in range(start_col, end_col):
            tname = grid[r][c].__class__.__name__

            x0 = c * TILE_SIZE - camera_x
            y0 = r * TILE_SIZE - camera_y
            x1 = x0 + TILE_SIZE
            y1 = y0 + TILE_SIZE

            if c + 1 < GRASS_COLS:
                rname = grid[r][c + 1].__class__.__name__
                if rname != tname:
                    pygame.draw.line(surf, border_color, (x1, y0), (x1, y1), 1)

            if r + 1 < GRASS_ROWS:
                bname = grid[r + 1][c].__class__.__name__
                if bname != tname:
                    pygame.draw.line(surf, border_color, (x0, y1), (x1, y1), 1)

    pygame.draw.rect(surf, border_color, pygame.Rect(0, 0, surf.get_width(), surf.get_height()), 1)


def draw_scrollbars(
    screen: pygame.Surface,
    view_frame: pygame.Rect,
    camera_x: int,
    camera_y: int,
    view_px_w: int,
    view_px_h: int,
) -> None:
    """
    Draw a small horizontal scrollbar at the TOP of the view frame and a vertical one on the RIGHT.
    Shows where the camera is looking within the whole map.
    """
    bar_thick = 8
    track_col = (60, 60, 60)
    thumb_col = (200, 200, 200)

    # Horizontal (top)
    h_track = pygame.Rect(view_frame.x, view_frame.y - (bar_thick + 4), view_frame.w, bar_thick)
    pygame.draw.rect(screen, track_col, h_track, border_radius=4)

    denom_x = max(1, MAP_WIDTH - view_px_w)
    thumb_w = max(20, int(h_track.w * (view_px_w / MAP_WIDTH)))
    thumb_x = h_track.x + int((h_track.w - thumb_w) * (camera_x / denom_x))
    h_thumb = pygame.Rect(thumb_x, h_track.y, thumb_w, bar_thick)
    pygame.draw.rect(screen, thumb_col, h_thumb, border_radius=4)

    # Vertical (right)
    v_track = pygame.Rect(view_frame.right + 4, view_frame.y, bar_thick, view_frame.h)
    pygame.draw.rect(screen, track_col, v_track, border_radius=4)

    denom_y = max(1, MAP_HEIGHT - view_px_h)
    thumb_h = max(20, int(v_track.h * (view_px_h / MAP_HEIGHT)))
    thumb_y = v_track.y + int((v_track.h - thumb_h) * (camera_y / denom_y))
    v_thumb = pygame.Rect(v_track.x, thumb_y, bar_thick, thumb_h)
    pygame.draw.rect(screen, thumb_col, v_thumb, border_radius=4)


# -----------------------------
# Footprints / occupancy
# -----------------------------
def is_building(unit_type: str) -> bool:
    return unit_type in ("Barn", "TownCenter", "Barracks", "ShamansHut")


def footprint_cells(unit_type: str, anchor_row: int, anchor_col: int) -> List[Tuple[int, int]]:
    """
    Returns list of (row,col) tiles occupied by this placed object.
    - Normal units: 1 tile
    - Buildings: NxN tiles where N = ceil(BUILDING_SIZE / TILE_SIZE), centered on anchor tile
    """
    if not is_building(unit_type):
        return [(anchor_row, anchor_col)]

    n = int(math.ceil(BUILDING_SIZE / float(TILE_SIZE)))
    half = n // 2

    r0 = anchor_row - half
    c0 = anchor_col - half

    cells: List[Tuple[int, int]] = []
    for rr in range(r0, r0 + n):
        for cc in range(c0, c0 + n):
            if 0 <= rr < GRASS_ROWS and 0 <= cc < GRASS_COLS:
                cells.append((rr, cc))
    return cells


def build_occupancy(units_by_cell: Dict[str, Dict[str, Any]]) -> Dict[Tuple[int, int], str]:
    """
    Returns mapping: occupied_tile -> anchor_key ("r,c" where the unit was placed).
    """
    occ: Dict[Tuple[int, int], str] = {}
    for anchor_key, data in units_by_cell.items():
        try:
            rs, cs = anchor_key.split(",")
            ar, ac = int(rs), int(cs)
        except Exception:
            continue
        ut = str(data.get("type", ""))
        for cell in footprint_cells(ut, ar, ac):
            occ[cell] = anchor_key
    return occ


# -----------------------------
# Main editor
# -----------------------------
def main() -> None:
    if not pygame.get_init():
        pygame.init()
    if not pygame.display.get_init():
        pygame.display.init()
    if not pygame.font.get_init():
        pygame.font.init()

    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Map Editor (Clean Borders + Scrollbars)")
    clock = pygame.time.Clock()

    font = pygame.font.SysFont("arial", 16)
    font_big = pygame.font.SysFont("arial", 18)

    # ---- IMPORTANT: make view surface an exact multiple of TILE_SIZE ----
    view_tiles_w = max(1, VIEW_WIDTH // TILE_SIZE)
    view_tiles_h = max(1, VIEW_HEIGHT // TILE_SIZE)
    view_px_w = view_tiles_w * TILE_SIZE
    view_px_h = view_tiles_h * TILE_SIZE

    # Center that tile-aligned view inside the available view frame
    view_x = VIEW_MARGIN_LEFT + (VIEW_WIDTH - view_px_w) // 2
    view_y = VIEW_MARGIN_TOP + (VIEW_HEIGHT - view_px_h) // 2
    view_frame = pygame.Rect(view_x, view_y, view_px_w, view_px_h)

    view_surf = pygame.Surface((view_px_w, view_px_h))

    grid = new_grass_map()
    units_by_cell: Dict[str, Dict[str, Any]] = {}

    mode = "tile"  # "tile" | "unit" | "erase"
    selected_tile = "Dirt"
    selected_unit = "Axeman"
    selected_player = 1

    # Tree selection state (cycles on repeated clicks)
    selected_tree_index = 0  # 0..7

    # River selection state (cycles on repeated clicks of River tile button)
    selected_river_index = RIVER_DEFAULT_INDEX  # 1..9

    # Camera is always snapped to tile grid
    camera_x = 0
    camera_y = 0
    CAM_STEP = TILE_SIZE

    panel_rect = pygame.Rect(0, PANEL_Y, SCREEN_WIDTH, PANEL_HEIGHT)
    buttons: List[Button] = []

    BTN_H = 32
    BTN_W = 130
    BTN_GAP = 10

    def current_tree_variant() -> str:
        return f"{TREE_VARIANT_PREFIX}{selected_tree_index}"

    def current_river_variant() -> str:
        return f"{RIVER_VARIANT_PREFIX}{selected_river_index}"

    def rebuild_ui() -> None:
        buttons.clear()

        # Row 1: Modes + actions + players
        x = 10
        y = PANEL_Y + 10

        for label, val in [("Tile", "tile"), ("Unit", "unit"), ("Erase", "erase")]:
            buttons.append(Button(pygame.Rect(x, y, 90, BTN_H), label, "mode", val))
            x += 90 + 8

        x += 20
        buttons.append(Button(pygame.Rect(x, y, 100, BTN_H), "Save", "action", "save"))
        x += 100 + 8
        buttons.append(Button(pygame.Rect(x, y, 100, BTN_H), "Load", "action", "load"))
        x += 100 + 8
        buttons.append(Button(pygame.Rect(x, y, 100, BTN_H), "Clear", "action", "clear"))

        # Players on right: same height as others
        px = SCREEN_WIDTH - 10 - (len(PLAYERS) * (70 + 6))
        px = max(px, x + 20)
        for i, (plabel, _col) in enumerate(PLAYERS):
            buttons.append(Button(pygame.Rect(px + i * (70 + 6), y, 70, BTN_H), plabel, "player", str(i)))

        # Row 2: tiles + units
        x2 = 10
        y2 = PANEL_Y + 10 + BTN_H + 10

        for name in TILE_TYPES.keys():
            lab = name.replace("Tile", "")
            buttons.append(Button(pygame.Rect(x2, y2, BTN_W, BTN_H), lab, "tile", name))
            x2 += BTN_W + BTN_GAP

        x2 += 20
        for name in UNIT_ROW:
            if name not in UNIT_TYPES:
                continue
            buttons.append(Button(pygame.Rect(x2, y2, BTN_W, BTN_H), name, "unit", name))
            x2 += BTN_W + BTN_GAP

        # Row 3: bottom row (Tree + buildings)
        x3 = 10
        y3 = y2 + BTN_H + 10
        for name in BOTTOM_ROW:
            if name not in UNIT_TYPES:
                continue
            # NOTE: label for Tree will be drawn dynamically (TreeN)
            buttons.append(Button(pygame.Rect(x3, y3, BTN_W, BTN_H), name, "unit", name))
            x3 += BTN_W + BTN_GAP

    rebuild_ui()

    def snap_to_tile(n: int) -> int:
        return (n // TILE_SIZE) * TILE_SIZE

    def clamp_camera() -> None:
        nonlocal camera_x, camera_y
        max_x = max(0, MAP_WIDTH - view_px_w)
        max_y = max(0, MAP_HEIGHT - view_px_h)
        camera_x = max(0, min(camera_x, max_x))
        camera_y = max(0, min(camera_y, max_y))
        camera_x = snap_to_tile(camera_x)
        camera_y = snap_to_tile(camera_y)

    def screen_to_world(mx: int, my: int) -> Optional[Tuple[int, int]]:
        if not view_frame.collidepoint(mx, my):
            return None
        vx = mx - view_frame.x
        vy = my - view_frame.y
        return vx + camera_x, vy + camera_y

    def paint_at(mx: int, my: int) -> None:
        nonlocal grid, units_by_cell

        wp = screen_to_world(mx, my)
        if wp is None:
            return
        wx, wy = wp
        row, col = world_to_tile(wx, wy)
        if not (0 <= row < GRASS_ROWS and 0 <= col < GRASS_COLS):
            return

        occ = build_occupancy(units_by_cell)
        clicked_cell = (row, col)

        if mode == "tile":
            if clicked_cell in occ:
                return
            if selected_tile == "River":
                set_tile(grid, row, col, "River", variant=current_river_variant())
            else:
                set_tile(grid, row, col, selected_tile)

        elif mode == "unit":
            fp = footprint_cells(selected_unit, row, col)
            if any(cell in occ for cell in fp):
                return

            entry: Dict[str, Any] = {"type": selected_unit, "player": selected_player}
            if selected_unit == "Tree":
                entry["variant"] = current_tree_variant()

            units_by_cell[f"{row},{col}"] = entry

        elif mode == "erase":
            if clicked_cell in occ:
                anchor_key = occ[clicked_cell]
                units_by_cell.pop(anchor_key, None)
                return
            set_tile(grid, row, col, "GrassTile")

    def get_button_icon(kind: str, value: str, desired_px: int) -> Optional[pygame.Surface]:
        """
        Returns a small sprite/icon for a button.
        - Tiles: prefer their real tile asset (River uses current river variant).
        - Units: uses unit sprite (Tree uses current tree variant).
        """
        # Tiles
        if kind == "tile":
            if value == "River":
                return load_river_editor_image(current_river_variant(), desired_px)

            # Try to load a tile asset based on class naming convention:
            # assets/{classname_lower}.png
            # Example: Dirt -> assets/dirt.png, Bridge -> assets/bridge.png
            # GrassTile -> assets/grasstile.png
            path = f"assets/{value.lower()}.png"
            try:
                img = pygame.image.load(path).convert_alpha()
                img = pygame.transform.smoothscale(img, (desired_px, desired_px))
                return img
            except Exception:
                return None

        # Units
        if kind == "unit":
            if value == "Tree":
                return load_tree_editor_image(current_tree_variant(), desired_px)

            _, pcol = PLAYERS[selected_player]

            # IMPORTANT: don't load Unit._images at icon size; load at real unit draw size first
            big_px = int(max(UNIT_SIZE, TILE_SIZE))
            big_img = _ensure_unit_sprite_loaded(value, big_px, pcol)
            if big_img is None:
                return None

            # Then scale down just for the button icon
            try:
                return pygame.transform.smoothscale(big_img, (desired_px, desired_px))
            except Exception:
                return big_img

        return None

    def draw_button(b: Button) -> None:
        active = False
        if b.kind == "mode" and b.value == mode:
            active = True
        elif b.kind == "tile" and (mode == "tile") and b.value == selected_tile:
            active = True
        elif b.kind == "unit" and (mode == "unit") and b.value == selected_unit:
            active = True
        elif b.kind == "player" and int(b.value) == selected_player:
            active = True

        # Player buttons
        if b.kind == "player":
            pid = int(b.value)
            plabel, pcol = PLAYERS[pid]
            pygame.draw.rect(screen, pcol, b.rect, border_radius=6)
            pygame.draw.rect(screen, (255, 255, 255) if active else (80, 80, 80), b.rect, 3, border_radius=6)
            txt_col = (0, 0, 0) if sum(pcol) > 380 else (255, 255, 255)
            t = font.render(plabel, True, txt_col)
            screen.blit(t, (b.rect.centerx - t.get_width() // 2, b.rect.centery - t.get_height() // 2))
            return

        # Normal buttons
        pygame.draw.rect(screen, (40, 40, 40), b.rect, border_radius=6)
        pygame.draw.rect(screen, (255, 255, 255) if active else (120, 120, 120), b.rect, 2, border_radius=6)

        label = b.label

        # Dynamic label for Tree: show current selected variant index
        if b.kind == "unit" and b.value == "Tree":
            label = f"Tree{selected_tree_index}"

        # Dynamic label for River tile: show current river variant
        if b.kind == "tile" and b.value == "River":
            label = f"River{selected_river_index}"

        # ---- draw small icon for tiles/units ----
        PAD = 6
        ICON_PX = min(22, b.rect.h - 2 * PAD)  # fits inside button height
        icon = None
        if b.kind in ("tile", "unit"):
            icon = get_button_icon(b.kind, b.value, ICON_PX)

        text_color = (240, 240, 240)
        t = font.render(label, True, text_color)

        # Layout:
        # [ icon ] [ gap ] [ text ]
        gap = 6
        x = b.rect.x + PAD
        y_center = b.rect.centery

        if icon is not None:
            icon_rect = icon.get_rect()
            icon_rect.left = x
            icon_rect.centery = y_center
            screen.blit(icon, icon_rect)
            x = icon_rect.right + gap  # text starts after icon
        else:
            # if no icon, keep some left padding so text isn't glued to border
            x = b.rect.x + PAD

        # Vertically center text, but align left (so it doesn't overlap icon)
        screen.blit(t, (x, y_center - t.get_height() // 2))

    painting = False
    running = True
    while running:
        clock.tick(FPS)

        # Camera: ONLY arrow keys
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            camera_x -= CAM_STEP
        if keys[pygame.K_RIGHT]:
            camera_x += CAM_STEP
        if keys[pygame.K_UP]:
            camera_y -= CAM_STEP
        if keys[pygame.K_DOWN]:
            camera_y += CAM_STEP
        clamp_camera()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                hit_ui = False
                for b in buttons:
                    if b.hit(event.pos):
                        hit_ui = True

                        if b.kind == "mode":
                            mode = b.value

                        elif b.kind == "tile":
                            # Special behavior: repeated clicks on River cycle river1..river9
                            if b.value == "River":
                                if selected_tile == "River" and mode == "tile":
                                    selected_river_index += 1
                                    if selected_river_index > RIVER_VARIANT_MAX:
                                        selected_river_index = RIVER_VARIANT_MIN
                                selected_tile = "River"
                                mode = "tile"
                                # preload preview image so it feels instant
                                _ = load_river_editor_image(current_river_variant(), int(TILE_SIZE))
                            else:
                                selected_tile = b.value
                                mode = "tile"

                        elif b.kind == "unit":
                            # Special behavior: repeated clicks on Tree cycle tree0..tree6
                            if b.value == "Tree":
                                if selected_unit == "Tree":
                                    selected_tree_index = (selected_tree_index + 1) % TREE_VARIANT_COUNT
                                else:
                                    selected_unit = "Tree"
                                    mode = "unit"
                                # preload current preview image so it feels instant
                                _ = load_tree_editor_image(current_tree_variant(), int(TILE_SIZE))
                            else:
                                selected_unit = b.value
                                mode = "unit"

                        elif b.kind == "player":
                            selected_player = int(b.value)

                        elif b.kind == "action":
                            if b.value == "save":
                                save_map(grid, units_by_cell, DEFAULT_SAVE_PATH)
                            elif b.value == "load":
                                try:
                                    grid, units_by_cell = load_map(DEFAULT_SAVE_PATH)
                                except Exception as e:
                                    print(f"[EDITOR] Load failed: {e}")
                            elif b.value == "clear":
                                grid = new_grass_map()
                                units_by_cell = {}
                                print("[EDITOR] Cleared.")
                        break

                if not hit_ui:
                    painting = True
                    paint_at(*event.pos)

            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                painting = False

            elif event.type == pygame.MOUSEMOTION and painting:
                paint_at(*event.pos)

        # -------- DRAW --------
        screen.fill(BLACK)
        view_surf.fill((0, 0, 0))

        start_col = max(0, camera_x // TILE_SIZE)
        end_col = min(GRASS_COLS, (camera_x + view_px_w) // TILE_SIZE + 1)
        start_row = max(0, camera_y // TILE_SIZE)
        end_row = min(GRASS_ROWS, (camera_y + view_px_h) // TILE_SIZE + 1)

        # Tiles
        for r in range(start_row, end_row):
            for c in range(start_col, end_col):
                grid[r][c].draw(view_surf, camera_x, camera_y)

        # Borders ONLY between different tile types
        draw_tile_type_borders(
            view_surf,
            grid,
            start_row=start_row,
            end_row=end_row,
            start_col=start_col,
            end_col=end_col,
            camera_x=camera_x,
            camera_y=camera_y,
            border_color=(0, 0, 0),
        )

        # Units on top (use stored tree variant if present)
        for k, v in units_by_cell.items():
            try:
                rs, cs = k.split(",")
                r, c = int(rs), int(cs)
            except Exception:
                continue
            if r < start_row or r >= end_row or c < start_col or c >= end_col:
                continue

            unit_type = str(v.get("type"))
            pid = int(v.get("player", 0))
            pid = max(0, min(pid, len(PLAYERS) - 1))
            _, pcol = PLAYERS[pid]
            cx, cy = tile_center(r, c)

            if unit_type == "Tree":
                tv = v.get("variant")
                if not isinstance(tv, str):
                    tv = None
                draw_unit_sprite(view_surf, unit_type, pcol, cx, cy, camera_x, camera_y, tree_variant=tv)
            else:
                draw_unit_sprite(view_surf, unit_type, pcol, cx, cy, camera_x, camera_y)

        # Blit view aligned
        screen.blit(view_surf, (view_frame.x, view_frame.y))

        # Scrollbars (top + right)
        draw_scrollbars(
            screen,
            view_frame=view_frame,
            camera_x=camera_x,
            camera_y=camera_y,
            view_px_w=view_px_w,
            view_px_h=view_px_h,
        )

        # Panel
        pygame.draw.rect(screen, PANEL_COLOR, panel_rect)

        header = (
            f"Mode: {mode.upper()} | Tile: {selected_tile} | Unit: {selected_unit} | "
            f"TreeVar: {current_tree_variant()} | Player: {PLAYERS[selected_player][0]} | "
            f"Camera: Arrow Keys | View: {view_tiles_w}x{view_tiles_h} tiles"
        )
        screen.blit(font_big.render(header, True, WHITE), (10, PANEL_Y - 26))

        for b in buttons:
            draw_button(b)

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
