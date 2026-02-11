import sys
import math
import pygame
from typing import Optional, List, Dict, Tuple

# -------------------------------
# Config
# -------------------------------
WIDTH, HEIGHT = 1100, 760
FPS = 60

TILE_W = 128
TILE_H = 64
HALF_W = TILE_W // 2
HALF_H = TILE_H // 2
Z_STEP = 32  # vertical pixels per height level

TRANSPARENCY = 0.5            # 50% transparent (≈128 alpha) for occluding walls
OCCLUDE_RADIUS_TILES = 2      # only fade occluders within 2 tiles of player

MAP_W, MAP_H = 12, 12
BG_COLOR = (20, 24, 28)
FLOOR_COLOR = (100, 170, 210)
GRID_COLOR = (50, 90, 120)
TOP_COLOR = (180, 210, 230)
LEFT_FACE = (150, 170, 190)
RIGHT_FACE = (120, 150, 180)
HIGHLIGHT = (255, 230, 120)

PLAYER_COLOR = (255, 100, 80)
PLAYER_RADIUS = 10

# Inventory UI
INV_COLS, INV_ROWS = 6, 2
INV_SLOT_W, INV_SLOT_H = 64, 64
INV_MARGIN = 12
INV_LEFT = 14
INV_TOP = HEIGHT - (INV_ROWS * INV_SLOT_H + (INV_ROWS - 1) * 8) - 14

# Container UI
CONT_COLS, CONT_ROWS = 4, 2
CONT_SLOT_W, CONT_SLOT_H = 64, 64
CONT_MARGIN = 12
CONT_RIGHT = WIDTH - (CONT_COLS * CONT_SLOT_W + (CONT_COLS - 1) * 8) - 14
CONT_TOP = INV_TOP

# -------------------------------
# World data
# -------------------------------
# Simple height map: 0 = floor, 1..N = raised block (wall)
HEIGHT_MAP = [[0]*MAP_W for _ in range(MAP_H)]
for y in range(MAP_H):
    for x in range(MAP_W):
        if (x in (4, 5, 6) and y in (2, 3, 4)) or (x in (2, 9) and 6 <= y <= 8):
            HEIGHT_MAP[y][x] = 2  # walls
        if (x, y) in {(8, 2), (7, 5), (3, 9)}:
            HEIGHT_MAP[y][x] = 1

# Items (simple dicts)
def make_item(name: str, qty: int, color: Tuple[int,int,int], max_stack: int = 99) -> Dict:
    return {"name": name, "qty": qty, "color": color, "max_stack": max_stack}

# Items on the ground
WORLD_ITEMS: List[Dict] = [
    {"pos": (1, 1), "item": make_item("Coin", 12, (250, 220, 70), 99)},
    {"pos": (6, 2), "item": make_item("Key", 1, (250, 250, 250), 1)},
    {"pos": (9, 7), "item": make_item("Potion", 3, (180, 50, 200), 10)},
]

# Containers on the map (not walls; separate interactive objects)
CONTAINERS: List[Dict] = [
    {"pos": (2, 5), "name": "Crate A",
     "slots": [make_item("Coin", 8, (250, 220, 70), 99), None, None, make_item("Bandage", 2, (220, 220, 220), 5),
               None, None, None, None]},
    {"pos": (10, 4), "name": "Crate B",
     "slots": [make_item("Potion", 2, (180, 50, 200), 10), make_item("Gem", 1, (80, 220, 200), 5)] + [None]*6},
]

# Player inventory (12 slots)
INVENTORY: List[Optional[Dict]] = [None] * (INV_COLS * INV_ROWS)

# Cursor-held stack (picked with mouse)
CURSOR_ITEM: Optional[Dict] = None

OPEN_CONTAINER_INDEX: Optional[int] = None  # index into CONTAINERS when open

# -------------------------------
# Helpers (geometry)
# -------------------------------
def iso_to_screen(wx: float, wy: float, wz: float = 0.0):
    sx = (wx - wy) * HALF_W
    sy = (wx + wy) * HALF_H - wz * Z_STEP
    return sx, sy

def screen_to_tile(sx: float, sy: float, camx: float, camy: float):
    sx -= camx
    sy -= camy
    wx = (sx / HALF_W + sy / HALF_H) * 0.5
    wy = (sy / HALF_H - sx / HALF_W) * 0.5
    return wx, wy

def tile_poly_center(x, y, z=0):
    cx, cy = iso_to_screen(x, y, z)
    top    = (cx,          cy - HALF_H)
    right  = (cx + HALF_W, cy)
    bottom = (cx,          cy + HALF_H)
    left   = (cx - HALF_W, cy)
    return top, right, bottom, left

def point_in_poly(pt, poly):
    x, y = pt
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if ((y1 > y) != (y2 > y)):
            xinters = (x2 - x1) * (y - y1) / (y2 - y1 + 1e-9) + x1
            if x < xinters:
                inside = not inside
    return inside

# -------------------------------
# Rendering
# -------------------------------
def draw_ground(surface, x, y, cam):
    t0, r0, b0, l0 = tile_poly_center(x, y, 0)
    pts = [t0, r0, b0, l0]
    pts = [(px + cam[0] + WIDTH//2, py + cam[1] + HEIGHT//2) for px, py in pts]
    pygame.draw.polygon(surface, FLOOR_COLOR, pts)
    pygame.draw.lines(surface, GRID_COLOR, True, pts, 1)

def draw_wall(surface, x, y, h, cam, alpha_override=None):
    if h <= 0:
        return
    cxh, cyh = iso_to_screen(x, y, h)

    def S(pt):
        px, py = pt
        return (px + cam[0] + WIDTH//2, py + cam[1] + HEIGHT//2)

    t0, r0, b0, l0 = tile_poly_center(x, y, 0)
    t1, r1, b1, l1 = tile_poly_center(x, y, h)
    right_face = list(map(S, [r1, r0, b0, b1]))
    left_face  = list(map(S, [l1, l0, b0, b1]))
    top_poly   = list(map(S, [t1, r1, b1, l1]))

    height_px = int(h * Z_STEP + TILE_H)
    top_left = (int(cxh - HALF_W + cam[0] + WIDTH//2),
                int(cyh - HALF_H + cam[1] + HEIGHT//2))
    wall_surf = pygame.Surface((TILE_W, height_px), pygame.SRCALPHA)

    def to_local(poly): return [(x - top_left[0], y - top_left[1]) for x, y in poly]
    rf_local = to_local(right_face)
    lf_local = to_local(left_face)
    tp_local = to_local(top_poly)

    pygame.draw.polygon(wall_surf, RIGHT_FACE, rf_local)
    pygame.draw.polygon(wall_surf, LEFT_FACE,  lf_local)
    pygame.draw.polygon(wall_surf, TOP_COLOR,  tp_local)
    pygame.draw.lines(wall_surf, GRID_COLOR, True, tp_local, 1)

    if alpha_override is not None and 0 <= alpha_override < 255:
        wall_surf.set_alpha(alpha_override)
    surface.blit(wall_surf, top_left)

def draw_player(surface, px, py, cam):
    tx, ty = int(round(px)), int(round(py))
    h = HEIGHT_MAP[ty][tx] if 0 <= tx < MAP_W and 0 <= ty < MAP_H else 0
    sx, sy = iso_to_screen(px, py, h)
    sx += cam[0] + WIDTH//2
    sy += cam[1] + HEIGHT//2 - HALF_H + 6
    pygame.draw.circle(surface, PLAYER_COLOR, (int(sx), int(sy)), PLAYER_RADIUS)
    return (int(sx), int(sy))

def draw_item_on_ground(surface, x, y, cam, item):
    # draw a small diamond with the item color + label above ground
    cx, cy = iso_to_screen(x, y, 0)
    cx += cam[0] + WIDTH//2
    cy += cam[1] + HEIGHT//2
    diamond = [(cx, cy - 8), (cx + 12, cy), (cx, cy + 8), (cx - 12, cy)]
    pygame.draw.polygon(surface, item["color"], diamond)
    pygame.draw.lines(surface, (20, 20, 20), True, diamond, 1)

# -------------------------------
# Inventory / Container logic
# -------------------------------
def can_stack(a: Dict, b: Dict) -> bool:
    return a and b and a["name"] == b["name"] and a["max_stack"] == b["max_stack"]

def add_to_slots(slots: List[Optional[Dict]], itm: Dict) -> Dict:
    """Try to insert `itm` into slots. Returns leftover dict if not fully placed (qty>0)."""
    if itm["qty"] <= 0: return None
    # 1) merge with same stacks
    for s in slots:
        if s and can_stack(s, itm) and s["qty"] < s["max_stack"]:
            take = min(itm["qty"], s["max_stack"] - s["qty"])
            s["qty"] += take
            itm["qty"] -= take
            if itm["qty"] <= 0: return None
    # 2) use empty slots
    for i in range(len(slots)):
        if slots[i] is None:
            put = min(itm["qty"], itm["max_stack"])
            slots[i] = {"name": itm["name"], "qty": put, "color": itm["color"], "max_stack": itm["max_stack"]}
            itm["qty"] -= put
            if itm["qty"] <= 0: return None
    return itm  # leftover

def slots_rects(cols, rows, slot_w, slot_h, left, top) -> List[pygame.Rect]:
    rects = []
    for r in range(rows):
        for c in range(cols):
            x = left + c * (slot_w + 8)
            y = top + r * (slot_h + 8)
            rects.append(pygame.Rect(x, y, slot_w, slot_h))
    return rects

def draw_slots(surface, font, rects: List[pygame.Rect], slots: List[Optional[Dict]], title: str):
    # title
    if title:
        label = font.render(title, True, (230, 240, 255))
        surface.blit(label, (rects[0].x, rects[0].y - 20))
    # slots
    for i, rc in enumerate(rects):
        pygame.draw.rect(surface, (40, 48, 56), rc, border_radius=8)
        pygame.draw.rect(surface, (90, 110, 140), rc, width=2, border_radius=8)
        itm = slots[i] if i < len(slots) else None
        if itm:
            # colored chip
            pygame.draw.rect(surface, itm["color"], rc.inflate(-16, -16), border_radius=8)
            # qty & name
            qty_surf = font.render(str(itm["qty"]), True, (15, 20, 25))
            surface.blit(qty_surf, (rc.right - qty_surf.get_width() - 8, rc.bottom - qty_surf.get_height() - 6))
            name_surf = font.render(itm["name"], True, (230, 240, 255))
            surface.blit(name_surf, (rc.x + 8, rc.y + 6))

def click_slot_transfer(idx_from: int, slots_from: List[Optional[Dict]],
                        idx_to: int, slots_to: List[Optional[Dict]]):
    """Move/merge stack from one slot array to another."""
    if idx_from is None or idx_to is None: return
    if idx_from >= len(slots_from) or idx_to >= len(slots_to): return
    a = slots_from[idx_from]
    b = slots_to[idx_to]
    if not a: return
    if b is None:
        slots_to[idx_to] = a
        slots_from[idx_from] = None
        return
    if can_stack(a, b):
        free = b["max_stack"] - b["qty"]
        take = min(free, a["qty"])
        b["qty"] += take
        a["qty"] -= take
        if a["qty"] <= 0:
            slots_from[idx_from] = None
    else:
        # swap
        slots_from[idx_from], slots_to[idx_to] = b, a

# -------------------------------
# Main
# -------------------------------
def main():
    global OPEN_CONTAINER_INDEX
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Isometric + Inventory/Containers — Walls fade when occluding")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 20)
    font_big = pygame.font.SysFont(None, 24)

    # Precompute UI rects
    inv_rects = slots_rects(INV_COLS, INV_ROWS, INV_SLOT_W, INV_SLOT_H, INV_LEFT, INV_TOP)

    px, py = 2.5, 2.5
    SPEED = 3.0  # tiles/sec

    # For container UI rects we recreate each frame (same layout), but could cache
    dragging_from = None  # ("inv" or "cont", index)
    dragging_item = None  # not used now (we do direct click-to-move)

    while True:
        dt = clock.tick(FPS) / 1000.0

        # Input
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit(); sys.exit()

            # Mouse clicks: move stacks between slots
            if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                mx, my = ev.pos
                # Inventory first
                hit_inv = None
                for i, rc in enumerate(inv_rects):
                    if rc.collidepoint(mx, my):
                        hit_inv = i; break

                open_slots = None
                cont_rects = []
                if OPEN_CONTAINER_INDEX is not None:
                    cont_rects = slots_rects(CONT_COLS, CONT_ROWS, CONT_SLOT_W, CONT_SLOT_H,
                                             CONT_RIGHT, CONT_TOP)
                    open_slots = CONTAINERS[OPEN_CONTAINER_INDEX]["slots"]
                hit_cont = None
                for i, rc in enumerate(cont_rects):
                    if rc.collidepoint(mx, my):
                        hit_cont = i; break

                # Transfer logic: click on a stack in one panel, then click target in the other
                # If both clicks are within the same panel, it swaps/merges in that panel.
                if hit_inv is not None and open_slots is not None and hit_cont is None:
                    # inventory → inventory (self) click: pick the other inv slot? we'll wait second click
                    dragging_from = ("inv", hit_inv)
                elif hit_cont is not None and open_slots is not None and hit_inv is None:
                    dragging_from = ("cont", hit_cont)
                elif hit_inv is not None and hit_cont is not None and open_slots is not None:
                    # direct cross-panel move: pick inventory slot, drop to container or vice versa
                    # prefer "from where there IS an item"
                    if INVENTORY[hit_inv] and not open_slots[hit_cont]:
                        click_slot_transfer(hit_inv, INVENTORY, hit_cont, open_slots)
                    elif open_slots[hit_cont] and not INVENTORY[hit_inv]:
                        click_slot_transfer(hit_cont, open_slots, hit_inv, INVENTORY)
                    else:
                        # try both directions (merge/swap)
                        click_slot_transfer(hit_inv, INVENTORY, hit_cont, open_slots)
                else:
                    # Single panel ops (inv only)
                    if hit_inv is not None:
                        if dragging_from and dragging_from[0] == "inv":
                            click_slot_transfer(dragging_from[1], INVENTORY, hit_inv, INVENTORY)
                        dragging_from = ("inv", hit_inv)
                    # container only
                    if hit_cont is not None and open_slots is not None:
                        if dragging_from and dragging_from[0] == "cont":
                            click_slot_transfer(dragging_from[1], open_slots, hit_cont, open_slots)
                        dragging_from = ("cont", hit_cont)

            if ev.type == pygame.KEYDOWN:
                if ev.key in (pygame.K_ESCAPE,):
                    pygame.quit(); sys.exit()
                # Pick up items on current tile
                if ev.key == pygame.K_SPACE:
                    ptx, pty = int(round(px)), int(round(py))
                    to_remove = []
                    for i, wi in enumerate(WORLD_ITEMS):
                        if wi["pos"] == (ptx, pty):
                            leftover = add_to_slots(INVENTORY, wi["item"])
                            if leftover is None or leftover["qty"] <= 0:
                                to_remove.append(i)
                            else:
                                wi["item"] = leftover  # partially picked
                    # remove consumed
                    for idx in reversed(to_remove):
                        WORLD_ITEMS.pop(idx)
                # Open/close nearest container (<= 1 tile away)
                if ev.key == pygame.K_e:
                    ptx, pty = int(round(px)), int(round(py))
                    nearest = None; bestd = 999
                    for idx, c in enumerate(CONTAINERS):
                        tx, ty = c["pos"]
                        d = max(abs(tx - ptx), abs(ty - pty))
                        if d <= 1 and d < bestd:
                            nearest = idx; bestd = d
                    if nearest is not None:
                        if OPEN_CONTAINER_INDEX == nearest:
                            OPEN_CONTAINER_INDEX = None
                        else:
                            OPEN_CONTAINER_INDEX = nearest

        # Movement
        keys = pygame.key.get_pressed()
        dx = dy = 0.0
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            dx -= SPEED * dt
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            dx += SPEED * dt
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            dy -= SPEED * dt
        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            dy += SPEED * dt

        def clamp_player(px, py):
            px = max(0.0, min(MAP_W - 1e-4, px))
            py = max(0.0, min(MAP_H - 1e-4, py))
            return px, py

        def is_blocked(nx, ny):
            ix, iy = int(round(nx)), int(round(ny))
            if ix < 0 or iy < 0 or ix >= MAP_W or iy >= MAP_H:
                return True
            return HEIGHT_MAP[iy][ix] > 0

        nx, ny = clamp_player(px + dx, py)
        if not is_blocked(nx, py): px = nx
        nx, ny = clamp_player(px, py + dy)
        if not is_blocked(px, ny): py = ny

        # Camera centers on player (ground height)
        pcx, pcy = iso_to_screen(px, py, 0)
        camx = -pcx
        camy = -pcy

        # -------- Render world --------
        screen.fill(BG_COLOR)

        # Ground
        for ty in range(MAP_H):
            for tx in range(MAP_W):
                draw_ground(screen, tx, ty, (camx, camy))

        # Items on ground (draw after ground)
        for wi in WORLD_ITEMS:
            draw_item_on_ground(screen, wi["pos"][0], wi["pos"][1], (camx, camy), wi["item"])

        # Draw containers as small boxes on floor
        for c in CONTAINERS:
            tx, ty = c["pos"]
            cx, cy = iso_to_screen(tx, ty, 0)
            cx += camx + WIDTH//2; cy += camy + HEIGHT//2
            # simple box
            body = [(cx-10, cy+2), (cx+10, cy+2), (cx+8, cy+16), (cx-8, cy+16)]
            pygame.draw.polygon(screen, (140, 100, 60), body)
            pygame.draw.lines(screen, (60,40,25), True, body, 2)
            lid = [(cx-10, cy+2), (cx, cy-10), (cx+10, cy+2)]
            pygame.draw.polygon(screen, (170, 130, 80), lid)
            pygame.draw.lines(screen, (80,50,30), True, lid, 2)

        # Prepare actors for painter's algorithm
        ptx, pty = int(round(px)), int(round(py))
        player_depth = px + py + 0.3
        player_screen = None

        actors = []
        for ty in range(MAP_H):
            for tx in range(MAP_W):
                h = HEIGHT_MAP[ty][tx]
                if h > 0:
                    actors.append(("wall", tx + ty + 0.7, (tx, ty, h)))
        actors.append(("player", player_depth, None))
        actors.sort(key=lambda a: a[1])

        def wall_faces_screen(tx, ty, h):
            t0, r0, b0, l0 = tile_poly_center(tx, ty, 0)
            t1, r1, b1, l1 = tile_poly_center(tx, ty, h)
            def S(pt):
                x, y = pt
                return (x + camx + WIDTH//2, y + camy + HEIGHT//2)
            right_face = list(map(S, [r1, r0, b0, b1]))
            left_face  = list(map(S, [l1, l0, b0, b1]))
            return left_face, right_face

        # Draw walls/player with occlusion-based transparency
        for kind, depth, payload in actors:
            if kind == "wall":
                tx, ty, h = payload
                alpha = 255
                if player_screen is not None and depth > player_depth:
                    grid_dist = max(abs(tx - ptx), abs(ty - pty))
                    if grid_dist <= OCCLUDE_RADIUS_TILES:
                        lf, rf = wall_faces_screen(tx, ty, h)
                        if point_in_poly(player_screen, lf) or point_in_poly(player_screen, rf):
                            alpha = int(255 * (1.0 - TRANSPARENCY))
                draw_wall(screen, tx, ty, h, (camx, camy), alpha_override=alpha)
            else:
                player_screen = draw_player(screen, px, py, (camx, camy))

        # -------- UI --------
        # Inventory
        draw_slots(screen, font, inv_rects, INVENTORY, "Inventory")

        # Open container panel (if any)
        if OPEN_CONTAINER_INDEX is not None:
            cont_rects = slots_rects(CONT_COLS, CONT_ROWS, CONT_SLOT_W, CONT_SLOT_H, CONT_RIGHT, CONT_TOP)
            slots = CONTAINERS[OPEN_CONTAINER_INDEX]["slots"]
            title = f'Container: {CONTAINERS[OPEN_CONTAINER_INDEX]["name"]}'
            draw_slots(screen, font, cont_rects, slots, title)
        else:
            cont_rects = []

        # Interaction prompt
        ptx, pty = int(round(px)), int(round(py))
        prompt = None
        # Check items at feet
        if any(wi["pos"] == (ptx, pty) for wi in WORLD_ITEMS):
            prompt = "SPACE: pick up items"
        # Nearby container prompt
        for idx, c in enumerate(CONTAINERS):
            tx, ty = c["pos"]
            d = max(abs(tx - ptx), abs(ty - pty))
            if d <= 1:
                prompt = "E: open container" if OPEN_CONTAINER_INDEX != idx else "E: close container"

        if prompt:
            p = font_big.render(prompt, True, (255, 255, 255))
            screen.blit(p, (14, INV_TOP - 36))

        # Mouse highlight (optional)
        mx, my = pygame.mouse.get_pos()
        wx, wy = screen_to_tile(mx - WIDTH//2, my - HEIGHT//2, camx, camy)
        hx, hy = int(round(wx)), int(round(wy))
        if 0 <= hx < MAP_W and 0 <= hy < MAP_H:
            h = HEIGHT_MAP[hy][hx]
            t, r, b, l = tile_poly_center(hx, hy, h)
            outline = [(x + camx + WIDTH//2, y + camy + HEIGHT//2) for (x, y) in (t, r, b, l)]
            pygame.draw.lines(screen, HIGHLIGHT, True, outline, 3)

        # HUD help
        help_text = "Move: WASD/Arrows • SPACE: pick up • E: open/close container • Click: move/merge stacks"
        info = font.render(help_text, True, (230, 240, 255))
        screen.blit(info, (14, 10))

        pygame.display.flip()

if __name__ == "__main__":
    main()
