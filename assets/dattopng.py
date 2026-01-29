#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import struct
import argparse
from typing import List, Optional, Tuple
from PIL import Image

CHUNK_SIZE = 33000
PALETTE_BYTES = 768  # 256 * 3

# Pary "góra + dół" wyczytane z kodu (ShowPicture(x,0) i ShowPicture(y,100))
STACK_PAIRS = [
    (0, 15),
    (1, 16),
    (2, 17),
    (3, 18),
    (4, 19),
    (5, 20),
    (7, 22),
    (8, 23),
    (9, 25),
    (10, 24),
    (11, 26),
    (14, 29),
]

def clamp_u8(x: float) -> int:
    return 0 if x < 0 else 255 if x > 255 else int(round(x))

def vga63_to_u8(v: int) -> int:
    # DOS/VGA DAC: 0..63 -> 0..255
    return int(round(v * 255.0 / 63.0))

def apply_gamma_u8(v: int, gamma: float) -> int:
    if gamma is None or abs(gamma - 1.0) < 1e-9:
        return v
    x = v / 255.0
    y = pow(x, 1.0 / gamma)
    return clamp_u8(y * 255.0)

def load_palette(
    pal_path: str,
    pal_index: int,
    gamma: float = 2.2,
    channel_order: str = "RGB",
    auto_scale: bool = True
) -> List[int]:
    """
    Wczytuje paletę z pal.dat:
      - auto_scale: jeśli max<=63, traktuje jako VGA 0..63
      - gamma: żeby kolory wyglądały sensownie w nowym systemie
      - channel_order: RGB albo BGR
    Zwraca listę 768 wartości (R,G,B)*256 w zakresie 0..255.
    """
    with open(pal_path, "rb") as f:
        f.seek(pal_index * PALETTE_BYTES)
        raw = f.read(PALETTE_BYTES)
        if len(raw) != PALETTE_BYTES:
            raise ValueError(f"Nie mogę wczytać palety #{pal_index} z {pal_path} (za krótki plik).")

    maxv = max(raw) if raw else 0
    use_vga63 = (maxv <= 63) if auto_scale else True

    pal: List[int] = []
    for i in range(0, PALETTE_BYTES, 3):
        r, g, b = raw[i], raw[i+1], raw[i+2]
        if channel_order.upper() == "BGR":
            r, b = b, r

        if use_vga63:
            r = vga63_to_u8(r)
            g = vga63_to_u8(g)
            b = vga63_to_u8(b)
        else:
            r, g, b = int(r), int(g), int(b)

        r = apply_gamma_u8(r, gamma)
        g = apply_gamma_u8(g, gamma)
        b = apply_gamma_u8(b, gamma)

        pal.extend([r, g, b])

    return pal

def decode_graf_chunk(chunk: bytes) -> Tuple[int, int, bytes]:
    """
    Blok: 3x uint16 LE (pierwsze zwykle ID), potem indeksy pikseli.
    """
    if len(chunk) != CHUNK_SIZE:
        raise ValueError("Zły rozmiar chunka (nie 33000).")

    _a, w, h = struct.unpack_from("<3H", chunk, 0)
    pixels_len = w * h
    pixels = chunk[6:6 + pixels_len]
    if len(pixels) != pixels_len:
        raise ValueError(f"Za mało danych pixeli: oczekiwano {pixels_len}, jest {len(pixels)}.")
    return w, h, pixels

def read_all_chunks(graf_path: str) -> List[Tuple[int, int, bytes]]:
    with open(graf_path, "rb") as f:
        data = f.read()

    if len(data) % CHUNK_SIZE != 0:
        raise ValueError(f"graf.dat ma rozmiar {len(data)} niepodzielny przez {CHUNK_SIZE}.")

    n = len(data) // CHUNK_SIZE
    chunks = []
    for i in range(n):
        chunk = data[i * CHUNK_SIZE:(i + 1) * CHUNK_SIZE]
        chunks.append(decode_graf_chunk(chunk))
    return chunks

def sheet_to_rgba(w: int, h: int, pixels: bytes, pal: List[int], transparent_index: Optional[int]) -> Image.Image:
    """
    Konwertuje indeksy + paletę do RGBA.
    """
    # LUT 256
    lut = [(pal[i*3], pal[i*3+1], pal[i*3+2], 255) for i in range(256)]
    if transparent_index is not None and 0 <= transparent_index < 256:
        r, g, b, _ = lut[transparent_index]
        lut[transparent_index] = (r, g, b, 0)

    out = bytearray(w * h * 4)
    for p, idx in enumerate(pixels):
        r, g, b, a = lut[idx]
        o = p * 4
        out[o:o+4] = bytes((r, g, b, a))
    return Image.frombytes("RGBA", (w, h), bytes(out))

def pad_to_width(img: Image.Image, target_w: int) -> Image.Image:
    """
    Jeśli img ma mniej niż target_w, dopadkuj po prawej kopiując ostatnią kolumnę.
    """
    w, h = img.size
    if w == target_w:
        return img
    if w > target_w:
        return img.crop((0, 0, target_w, h))

    padded = Image.new("RGBA", (target_w, h), (0, 0, 0, 0))
    padded.paste(img, (0, 0))

    # wypełnij brakujące kolumny ostatnią kolumną z img (ładniej niż transparent/czarny)
    last_col = img.crop((w - 1, 0, w, h))
    for x in range(w, target_w):
        padded.paste(last_col, (x, 0))
    return padded

def save_individual_sheets(chunks, pal, out_dir, transparent_index):
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for i, (w, h, pixels) in enumerate(chunks):
        img = sheet_to_rgba(w, h, pixels, pal, transparent_index)
        out = os.path.join(out_dir, f"sheet_{i:02d}.png")
        img.save(out)
        paths.append(out)
    return paths

def combine_pairs(chunks, pal, pairs, out_dir, transparent_index, target_w=320, target_h=200, y_split=100):
    """
    Składa (top, bottom) w 320x200:
      top na y=0
      bottom na y=100
    """
    os.makedirs(out_dir, exist_ok=True)
    outputs = []

    for top_i, bottom_i in pairs:
        if top_i < 0 or top_i >= len(chunks) or bottom_i < 0 or bottom_i >= len(chunks):
            print(f"[WARN] Pomijam parę ({top_i},{bottom_i}) – poza zakresem.")
            continue

        tw, th, tpix = chunks[top_i]
        bw, bh, bpix = chunks[bottom_i]

        top_img = sheet_to_rgba(tw, th, tpix, pal, transparent_index)
        bot_img = sheet_to_rgba(bw, bh, bpix, pal, transparent_index)

        # dopasuj szerokość do 320
        top_img = pad_to_width(top_img, target_w)
        bot_img = pad_to_width(bot_img, target_w)

        canvas = Image.new("RGBA", (target_w, target_h), (0, 0, 0, 0))
        canvas.paste(top_img, (0, 0))
        canvas.paste(bot_img, (0, y_split))

        out = os.path.join(out_dir, f"combined_{top_i:02d}_{bottom_i:02d}.png")
        canvas.save(out)
        outputs.append(out)

    return outputs

def main():
    ap = argparse.ArgumentParser(description="Wyciąga sheety z graf.dat i składa pary (góra+dół) w 320x200 PNG.")
    ap.add_argument("--graf", default="graf.dat", help="Ścieżka do graf.dat")
    ap.add_argument("--pal", default="pal.dat", help="Ścieżka do pal.dat")
    ap.add_argument("--pal-index", type=int, default=4, help="Indeks palety (domyślnie 4 = pal[4])")
    ap.add_argument("--out", default="out_png", help="Folder wyjściowy")
    ap.add_argument("--gamma", type=float, default=2.2, help="Gamma dla palety (typowo 1.8–2.2)")
    ap.add_argument("--channel-order", choices=["RGB", "BGR"], default="RGB", help="Jeśli kanały zamienione, użyj BGR")
    ap.add_argument("--no-auto-scale", action="store_true", help="Wyłącz auto-wykrywanie skali (wymuś VGA 0..63)")
    ap.add_argument("--transparent-index", type=int, default=None, help="Indeks palety jako przezroczysty (np. 0)")

    ap.add_argument("--save-sheets", action="store_true", help="Zapisz też pojedyncze sheety jako PNG")
    ap.add_argument("--combine", action="store_true", help="Złóż pary w 320x200 PNG (domyślnie włączone jeśli nic nie podasz)")

    args = ap.parse_args()

    # jeśli user nie podał ani --save-sheets ani --combine, to zrób combine (bo o to prosisz)
    do_combine = args.combine or (not args.save_sheets and not args.combine)

    pal = load_palette(
        pal_path=args.pal,
        pal_index=args.pal_index,
        gamma=args.gamma,
        channel_order=args.channel_order,
        auto_scale=not args.no_auto_scale
    )

    chunks = read_all_chunks(args.graf)
    print(f"Wczytano {len(chunks)} chunków z graf.dat")

    if args.save_sheets:
        sheet_dir = os.path.join(args.out, "sheets")
        save_individual_sheets(chunks, pal, sheet_dir, args.transparent_index)
        print(f"Pojedyncze sheety zapisane do: {sheet_dir}")

    if do_combine:
        comb_dir = os.path.join(args.out, "combined")
        outs = combine_pairs(
            chunks=chunks,
            pal=pal,
            pairs=STACK_PAIRS,
            out_dir=comb_dir,
            transparent_index=args.transparent_index,
            target_w=320,
            target_h=200,
            y_split=100
        )
        print(f"Złożone obrazy zapisane do: {comb_dir} ({len(outs)} plików)")

if __name__ == "__main__":
    main()
