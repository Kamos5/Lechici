import pygame
import random
import numpy as np

pygame.init()

WIDTH, HEIGHT = 800, 600
BAR_H = 80
MAP_DISP_W, MAP_DISP_H = WIDTH, HEIGHT - BAR_H

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Clickable Map + Player Paint + Flicker (Smooth FPS)")

# Low-res generation for speed
MAP_W, MAP_H = 400, 300
num_regions = random.randint(20, 25)

# ---------- Equal-area tuning knobs ----------
ITERATIONS = 35
WEIGHT_LR = 0.25
CENTROID_PULL = 0.35
DO_CENTROIDS = True
# --------------------------------------------

BORDER_COLOR = (255, 255, 255)

# Players: r,g,b,c,m,y + none(black)
PLAYERS = [
    ("R", (255, 0, 0)),
    ("G", (0, 255, 0)),
    ("B", (0, 0, 255)),
    ("C", (0, 255, 255)),
    ("M", (255, 0, 255)),
    ("Y", (255, 255, 0)),
    ("None", (0, 0, 0)),
]
PLAYER_COUNT = len(PLAYERS)

TILE_ALPHA = 128      # 50% transparency
NONE_ALPHA = 204      # 80% opacity (i.e., 20% transparency)
NONE_IDX = PLAYER_COUNT - 1  # "None" is last in PLAYERS

# Flicker settings (consistent square wave)
FLICKER_DURATION_MS = 2000
FLICKER_HZ = 60
HALF_PERIOD_MS = max(1, round(1000 / (FLICKER_HZ * 2)))

font = pygame.font.SysFont("arial", 20)
big_font = pygame.font.SysFont("arial", 28)

# --- loading ---
loading_text = big_font.render("Generating map... please wait", True, (220, 220, 255))
screen.fill((10, 10, 30))
screen.blit(loading_text, (WIDTH // 2 - 210, HEIGHT // 2 - 20))
pygame.display.flip()

# --- background image (only under the map area) ---
try:
    bg_raw = pygame.image.load("background.png").convert()
    background_surf = pygame.transform.smoothscale(bg_raw, (MAP_DISP_W, MAP_DISP_H))
except Exception as e:
    print("Could not load background.png:", e)
    background_surf = pygame.Surface((MAP_DISP_W, MAP_DISP_H))
    background_surf.fill((30, 30, 30))

# -------------------- Generate equal-ish regions --------------------
sites = np.array(
    [(random.randint(0, MAP_W - 1), random.randint(0, MAP_H - 1)) for _ in range(num_regions)],
    dtype=np.float32
)

weights = np.zeros(num_regions, dtype=np.float32)

xs = np.arange(MAP_W, dtype=np.float32)
ys = np.arange(MAP_H, dtype=np.float32)
X, Y = np.meshgrid(xs, ys, indexing="xy")  # (MAP_H, MAP_W)

target_area = (MAP_W * MAP_H) / num_regions
assign = None

for _ in range(ITERATIONS):
    dx = X[None, :, :] - sites[:, 0][:, None, None]
    dy = Y[None, :, :] - sites[:, 1][:, None, None]
    dist2 = dx * dx + dy * dy

    score = dist2 - weights[:, None, None]
    assign = np.argmin(score, axis=0).astype(np.int32)

    areas = np.bincount(assign.ravel(), minlength=num_regions).astype(np.float32)
    err = (areas - target_area) / target_area
    weights -= WEIGHT_LR * err * (MAP_W * MAP_H / num_regions) * 0.10

    if DO_CENTROIDS:
        flat = assign.ravel()
        sum_x = np.bincount(flat, weights=X.ravel(), minlength=num_regions).astype(np.float32)
        sum_y = np.bincount(flat, weights=Y.ravel(), minlength=num_regions).astype(np.float32)
        safe_areas = np.maximum(areas, 1.0)
        cx = sum_x / safe_areas
        cy = sum_y / safe_areas
        centroids = np.stack([cx, cy], axis=1)

        sites = sites + CENTROID_PULL * (centroids - sites)
        sites[:, 0] = np.clip(sites[:, 0], 0, MAP_W - 1)
        sites[:, 1] = np.clip(sites[:, 1], 0, MAP_H - 1)

# -------------------- Hidden ID color map for click detection --------------------
region_colors = []
for i in range(num_regions):
    r = 20 + (i * 11) % 236
    g = 40 + (i * 17) % 216
    b = 60 + (i * 13) % 196
    region_colors.append((r, g, b))
region_colors_arr = np.array(region_colors, dtype=np.uint8)

rgb_low = region_colors_arr[assign]                       # (MAP_H, MAP_W, 3)
rgb_low_bytes = np.transpose(rgb_low, (1, 0, 2)).copy()   # (MAP_W, MAP_H, 3)

color_map_low = pygame.Surface((MAP_W, MAP_H))
pygame.surfarray.blit_array(color_map_low, rgb_low_bytes)

# Scale up hidden map to map display size (no button bar)
color_map = pygame.transform.smoothscale(color_map_low, (MAP_DISP_W, MAP_DISP_H))

# -------------------- Precompute border layer ONCE --------------------
border_surf = pygame.Surface((MAP_DISP_W, MAP_DISP_H), flags=pygame.SRCALPHA)
border_surf.fill((0, 0, 0, 0))  # transparent

cp = pygame.PixelArray(color_map)
bp = pygame.PixelArray(border_surf)

# set only border pixels to opaque white
white = border_surf.map_rgb((255, 255, 255, 255))
for x in range(MAP_DISP_W):
    for y in range(MAP_DISP_H):
        c = cp[x][y]
        if (x < MAP_DISP_W - 1 and cp[x + 1][y] != c) or (y < MAP_DISP_H - 1 and cp[x][y + 1] != c):
            bp[x][y] = white

del cp
del bp

# -------------------- Ownership (ensure at least one per player) --------------------
owner = [PLAYER_COUNT - 1] * num_regions  # default None/black
for p in range(PLAYER_COUNT):
    owner[p % num_regions] = p

selected_player = 0

# Flicker state: region -> {start_ms, old_owner, new_owner}
flickers = {}

# We only rebuild the fill surface when it actually changes
needs_redraw = True
fill_surf = pygame.Surface((MAP_DISP_W, MAP_DISP_H))


def compute_effective_owner(now_ms):
    eff = owner[:]
    for ridx, st in list(flickers.items()):
        age = now_ms - st["start_ms"]
        if age >= FLICKER_DURATION_MS:
            flickers.pop(ridx, None)
            continue
        show_new = ((age // HALF_PERIOD_MS) % 2 == 0)
        eff[ridx] = st["new_owner"] if show_new else st["old_owner"]
    return eff


def rebuild_fill_surface(now_ms):
    global fill_surf
    eff_owner = compute_effective_owner(now_ms)

    # region -> RGB
    region_to_rgb = np.array([PLAYERS[eff_owner[i]][1] for i in range(num_regions)], dtype=np.uint8)
    low_rgb = region_to_rgb[assign]  # (MAP_H, MAP_W, 3)

    # region -> alpha (None gets different alpha)
    region_to_a = np.array(
        [NONE_ALPHA if eff_owner[i] == NONE_IDX else TILE_ALPHA for i in range(num_regions)],
        dtype=np.uint8
    )
    low_a = region_to_a[assign]  # (MAP_H, MAP_W)

    # Build a low-res SRCALPHA surface
    low = pygame.Surface((MAP_W, MAP_H), flags=pygame.SRCALPHA)

    # surfarray expects (W, H, 3) for RGB
    rgb_bytes = np.transpose(low_rgb, (1, 0, 2)).copy()  # (MAP_W, MAP_H, 3)
    pygame.surfarray.blit_array(low, rgb_bytes)

    # alpha expects (W, H)
    a_bytes = np.transpose(low_a, (1, 0)).copy()  # (MAP_W, MAP_H)
    alpha_view = pygame.surfarray.pixels_alpha(low)
    alpha_view[:, :] = a_bytes
    del alpha_view

    fill_surf = pygame.transform.smoothscale(low, (MAP_DISP_W, MAP_DISP_H))


def draw_buttons():
    pygame.draw.rect(screen, (25, 25, 25), pygame.Rect(0, MAP_DISP_H, WIDTH, BAR_H))

    pad = 10
    n = PLAYER_COUNT
    btn_w = (WIDTH - pad * (n + 1)) // n
    btn_h = BAR_H - 2 * pad

    for i, (label, col) in enumerate(PLAYERS):
        x = pad + i * (btn_w + pad)
        y = MAP_DISP_H + pad
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
    x, y = pos
    if y < MAP_DISP_H:
        return None

    pad = 10
    n = PLAYER_COUNT
    btn_w = (WIDTH - pad * (n + 1)) // n
    btn_h = BAR_H - 2 * pad

    for i in range(n):
        bx = pad + i * (btn_w + pad)
        by = MAP_DISP_H + pad
        if pygame.Rect(bx, by, btn_w, btn_h).collidepoint(pos):
            return i
    return None


clock = pygame.time.Clock()
running = True

while running:
    now_ms = pygame.time.get_ticks()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            pos = event.pos

            bi = button_index_at(pos)
            if bi is not None:
                selected_player = bi
            else:
                mx, my = pos
                if 0 <= mx < MAP_DISP_W and 0 <= my < MAP_DISP_H:
                    hidden_color = color_map.get_at((mx, my))[:3]
                    try:
                        region_idx = region_colors.index(hidden_color)

                        old_owner = owner[region_idx]
                        new_owner = selected_player

                        owner[region_idx] = new_owner
                        flickers[region_idx] = {
                            "start_ms": now_ms,
                            "old_owner": old_owner,
                            "new_owner": new_owner,
                        }
                        needs_redraw = True
                    except ValueError:
                        pass

    # If any flicker is active, we must redraw fills each frame (but borders are cached)
    if flickers:
        needs_redraw = True

    if needs_redraw:
        rebuild_fill_surface(now_ms)
        needs_redraw = False

    # Draw
    screen.blit(background_surf, (0, 0))  # background first
    screen.blit(fill_surf, (0, 0))  # tiles (with per-pixel alpha)
    screen.blit(border_surf, (0, 0))  # borders on top
    draw_buttons()
    pygame.display.flip()
    clock.tick(60)

pygame.quit()
