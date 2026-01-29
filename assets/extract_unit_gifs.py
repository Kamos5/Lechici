#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from PIL import Image

# =========================
# USTAWIENIA
# =========================
TILE_W = 16
TILE_H = 14

GIF_DURATION_MS = 110
GIF_LOOP = 0  # 0 = w kółko

# kolejność jak chcesz:
DIR_ORDER = ["LU", "U", "L", "M", "LD", "D"]

# =========================
# NARZĘDZIA
# =========================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def load_png(path: str) -> Image.Image:
    img = Image.open(path)
    return img.convert("RGBA") if img.mode != "RGBA" else img

def crop_tile(sheet: Image.Image, col: int, row: int) -> Image.Image:
    """col,row 0-based w siatce 16x14."""
    x1 = col * TILE_W
    y1 = row * TILE_H
    return sheet.crop((x1, y1, x1 + TILE_W, y1 + TILE_H))

def apply_colorkey_transparency(frame: Image.Image, key_rgb: Tuple[int, int, int]) -> Image.Image:
    """
    Ustaw alpha=0 dla pikseli dokładnie równych key_rgb.
    """
    if frame.mode != "RGBA":
        frame = frame.convert("RGBA")
    pix = frame.load()
    w, h = frame.size
    r0, g0, b0 = key_rgb
    for y in range(h):
        for x in range(w):
            r, g, b, a = pix[x, y]
            if (r, g, b) == (r0, g0, b0):
                pix[x, y] = (r, g, b, 0)
    return frame

def save_gif_rgba(frames: List[Image.Image], out_path: str) -> None:
    """
    Zapis GIF z przezroczystością.
    Pillow zwykle poprawnie mapuje alpha->transparent color.
    """
    if not frames:
        return
    ensure_dir(os.path.dirname(out_path))

    # upewnij się, że wszystkie są RGBA
    frames = [f.convert("RGBA") for f in frames]

    # zapis
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=GIF_DURATION_MS,
        loop=GIF_LOOP,
        disposal=2,
        optimize=False
    )

# =========================
# KONFIG
# =========================

@dataclass
class SheetSpec:
    filename: str
    base_type: int   # typ bazowy, np. 0 dla 04_19, 4 dla 05_20
    types: List[int] # które typy są w tym sheetcie

@dataclass
class Unit:
    typ: int
    name: str
    sheet_file: str
    local_index: int  # 0..3 w obrębie sheeta

# =========================
# MAPOWANIE JEDNOSTEK
# =========================

SHEETS = [
    SheetSpec(filename="combined_04_19.png", base_type=0, types=[0, 1, 2, 3]),
    SheetSpec(filename="combined_05_20.png", base_type=4, types=[4, 5, 6, 7]),
]

# Nazwy (możesz zmienić jak chcesz)
TYPE_NAMES = {
    0: "typ0_krowa",
    1: "typ1_topornik",
    2: "typ2",
    3: "typ3",
    4: "typ4_kaplan",
    5: "typ5_miecznik",
    6: "typ6_lucznik",
    7: "typ7_bohater",
}

def build_units() -> List[Unit]:
    units: List[Unit] = []
    for s in SHEETS:
        for t in s.types:
            local = t - s.base_type  # 0..3
            units.append(Unit(
                typ=t,
                name=TYPE_NAMES.get(t, f"typ{t}"),
                sheet_file=s.filename,
                local_index=local
            ))
    return units

# =========================
# WYCIĄGANIE ANIMACJI
# =========================

def extract_walk_6dirs_interleaved(sheet: Image.Image, local_index: int, col_start_1based: int = 1) -> Dict[str, List[Image.Image]]:
    """
    Układ zgodny z kodem (faza*32 + x*16, y*14 + typ*42):
      - 3 wiersze (y=0..2)
      - 6 kolumn (3 fazy * 2 kolumny x=0/1)
    Kierunki jak chcesz:
      y=0: x=0->LU, x=1->U
      y=1: x=0->L,  x=1->M
      y=2: x=0->LD, x=1->D

    Klatki:
      x=0 -> kolumny 1,3,5
      x=1 -> kolumny 2,4,6
    """
    # start wiersza w siatce:
    # typ zajmuje 42 px = 3 kafelki po 14, więc:
    # row_start_1based = 1 + local_index * 3
    row_start_1based = 1 + local_index * 3

    r0 = row_start_1based - 1
    c0 = col_start_1based - 1

    # wytnij 3x6 kafelków
    tiles = [[crop_tile(sheet, c0 + c, r0 + r) for c in range(6)] for r in range(3)]

    cols_x0 = [0, 2, 4]  # 1,3,5
    cols_x1 = [1, 3, 5]  # 2,4,6

    out: Dict[str, List[Image.Image]] = {
        "LU": [tiles[0][i] for i in cols_x0],
        "U":  [tiles[0][i] for i in cols_x1],
        "L":  [tiles[1][i] for i in cols_x0],
        "M":  [tiles[1][i] for i in cols_x1],
        "LD": [tiles[2][i] for i in cols_x0],
        "D":  [tiles[2][i] for i in cols_x1],
    }
    return out

def make_transparent(frames_by_dir: Dict[str, List[Image.Image]], key_mode: str = "topleft") -> Dict[str, List[Image.Image]]:
    """
    key_mode:
      - "topleft": kolor klucza = RGB lewego-górnego piksela danej klatki
                  (działa dobrze, gdy tło jest jednolite)
    """
    out: Dict[str, List[Image.Image]] = {}
    for d, frames in frames_by_dir.items():
        new_frames: List[Image.Image] = []
        for f in frames:
            if key_mode == "topleft":
                r, g, b, a = f.convert("RGBA").getpixel((0, 0))
                f2 = apply_colorkey_transparency(f, (r, g, b))
            else:
                f2 = f.convert("RGBA")
            new_frames.append(f2)
        out[d] = new_frames
    return out

# =========================
# MAIN
# =========================

def main(input_dir: str, output_dir: str) -> None:
    ensure_dir(output_dir)

    available = {os.path.basename(p): p for p in glob.glob(os.path.join(input_dir, "*.png"))}
    if not available:
        raise SystemExit(f"Brak PNG w folderze: {input_dir}")

    units = build_units()

    # wczytaj sheety tylko raz
    sheet_cache: Dict[str, Image.Image] = {}
    for u in units:
        if u.sheet_file not in available:
            print(f"[WARN] Brak {u.sheet_file} w {input_dir} (dla {u.name})")
            continue
        if u.sheet_file not in sheet_cache:
            sheet_cache[u.sheet_file] = load_png(available[u.sheet_file])

    for u in units:
        if u.sheet_file not in sheet_cache:
            continue

        sheet = sheet_cache[u.sheet_file]
        frames_by_dir = extract_walk_6dirs_interleaved(sheet, local_index=u.local_index, col_start_1based=1)

        # (1) przezroczyste tło
        frames_by_dir = make_transparent(frames_by_dir, key_mode="topleft")

        unit_dir = os.path.join(output_dir, u.name)
        ensure_dir(unit_dir)

        # zapis GIF-ów w kolejności DIR_ORDER
        for d in DIR_ORDER:
            frames = frames_by_dir.get(d, [])
            out_path = os.path.join(unit_dir, f"walk_{d}.gif")
            save_gif_rgba(frames, out_path)

        print(f"[OK] {u.name} (typ={u.typ}, sheet={u.sheet_file}, local={u.local_index}): zapisano GIF-y walk ({', '.join(DIR_ORDER)})")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Wyciąga animacje jednostek z połączonych sheetów i robi GIF-y (z przezroczystością).")
    ap.add_argument("--in", dest="input_dir", default="out_png/combined", help="Folder z combined_XX_YY.png")
    ap.add_argument("--out", dest="output_dir", default="unit_gifs", help="Folder wyjściowy")
    args = ap.parse_args()
    main(args.input_dir, args.output_dir)
