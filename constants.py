# constants.py (generated from contants.py)
import pygame

# -----------------------------
# Screen / Window
# -----------------------------
SCREEN_WIDTH = 1500
SCREEN_HEIGHT = 1200
SCREEN_WIDTH_BIG = 1500
SCREEN_HEIGHT_BIG = 1500
FPS = 60

# -----------------------------
# Scaling
# -----------------------------
SCALE = 4

# -----------------------------
# Tile / Map
# -----------------------------
TILE_SIZE = 20 * SCALE
TILE_HALF = TILE_SIZE // 2
TILE_QUARTER = TILE_SIZE // 4

GRASS_ROWS = 60
GRASS_COLS = 60

MAP_WIDTH = GRASS_COLS * TILE_SIZE
MAP_HEIGHT = GRASS_ROWS * TILE_SIZE

BUILDING_SIZE = 60 * SCALE
UNIT_SIZE = 16 * SCALE

# -----------------------------
# View / UI Layout
# -----------------------------
VIEW_MARGIN_LEFT = 10
VIEW_MARGIN_RIGHT = 10
VIEW_MARGIN_TOP = 40
VIEW_MARGIN_BOTTOM = 0

PANEL_HEIGHT = 150

VIEW_WIDTH = SCREEN_WIDTH - (VIEW_MARGIN_LEFT + VIEW_MARGIN_RIGHT)
VIEW_HEIGHT = SCREEN_HEIGHT - (VIEW_MARGIN_TOP + VIEW_MARGIN_BOTTOM + PANEL_HEIGHT)

PANEL_Y = SCREEN_HEIGHT - PANEL_HEIGHT - VIEW_MARGIN_BOTTOM

VIEW_BOUNDS_X = VIEW_MARGIN_LEFT + VIEW_WIDTH
VIEW_BOUNDS_Y = VIEW_MARGIN_TOP + VIEW_HEIGHT

# -----------------------------
# Buttons
# -----------------------------
BUTTON_WIDTH = 100
BUTTON_HEIGHT = 30
BUTTON_MARGIN = 5

GRID_BUTTON_ROWS = 3
GRID_BUTTON_COLS = 2
GRID_BUTTON_WIDTH = 110
GRID_BUTTON_HEIGHT = 40
GRID_BUTTON_MARGIN = 5

# -----------------------------
# Camera
# -----------------------------
SCROLL_SPEED = 10
SCROLL_MARGIN = 20

# -----------------------------
# Icons
# -----------------------------
ICON_SIZE = 32
ICON_MARGIN = 5

# -----------------------------
# Spatial Grid
# -----------------------------
GRID_CELL_SIZE = 60 * SCALE

# -----------------------------
# Gameplay tuning
# -----------------------------
REGROWTH_RATE = 0.0001

# -----------------------------
# Colors
# -----------------------------
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
PURPLE = (128, 0, 128)
ORANGE = (255, 165, 0)

BROWN = (75, 25, 0)

GRASS_GREEN = (0, 100, 0)
GRASS_BROWN = (139, 69, 19)
TREE_GREEN = (0, 50, 0)

DARK_GRAY = (50, 50, 50, 128)
GRAY = (128, 128, 128, 128)
LIGHT_GRAY = (150, 150, 150)
WHITE_GRAY = (200, 200, 200, 128)

TOWN_CENTER_GRAY = (100, 100, 100, 128)

BORDER_OUTER = (50, 50, 50)
BORDER_INNER = (200, 200, 200)
PANEL_COLOR = (100, 100, 100)
HIGHLIGHT_GRAY = (200, 200, 200)

# List of unique names for units
UNIQUE_MALE_NAMES = [
    "Bolesław", "Mieszko", "Władysław", "Kazimierz", "Dobrosław", "Mirosław", "Sławomir", "Wojciech",
    "Stanisław", "Zbigniew", "Bogumił", "Jaromir", "Witosław", "Czesław", "Leszek", "Radosław",
    "Witold", "Ziemowit", "Przemysław", "Bohusław", "Lubomir", "Wacław", "Zdzisław", "Mieczysław",
    "Radomił", "Świętosław", "Bronisław", "Gniewomir", "Siemowit", "Bohdan", "Jarosław", "Krzysztof",
    "Władimir", "Domarad", "Sulimir", "Bezprym", "Sambor", "Rostysław", "Wyszesław", "Bogusław"
]

FEMALE_NAMES = [
    "Milena", "Zofia", "Wanda", "Dobrawa", "Ludmiła", "Bronisława", "Jadwiga", "Elżbieta",
    "Radosława", "Bogumiła", "Stanisława", "Wiesława", "Mirosława", "Sławomira", "Zdzisława",
    "Czesława", "Jaromira", "Witosława", "Przemysława", "Bohusława", "Lubomira", "Wacława",
    "Radomiła", "Świętosława", "Dobrosława", "Gosława", "Mieczysława", "Władysława", "Kazimiera",
    "Ziemowita", "Bogna", "Danuta", "Halina", "Irena", "Krystyna", "Aldona", "Jolanta",
    "Beata", "Agnieszka", "Ewa", "Moolisa", "Twoolisa"
]
