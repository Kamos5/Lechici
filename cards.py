"""
round_card_choice.py

A tiny "pick 1 of 3 cards" UI meant to be called right after a new round starts.

- Shows 3 cards:
    1) Free cow
    2) Additional Axeman
    3) Fortification
  each with a short description and an image (PNG) if found.

- Returns a dict describing the selected card.

How to use from your world map loop (at start of next round):
    from round_card_choice import pick_round_card

    # inside your code, when a new round begins:
    picked = pick_round_card(screen)
    print("Picked:", picked["title"])

Images:
Place these PNGs next to this file (optional):
    free_cow.png
    additional_axeman.png
    fortification.png

Controls:
- Click a card to pick it
- Or press 1 / 2 / 3
- ESC closes and returns None
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Tuple, List

import pygame


def _load_png_or_placeholder(path: str, size: Tuple[int, int], label: str) -> pygame.Surface:
    surf = pygame.Surface(size, pygame.SRCALPHA)
    if os.path.exists(path):
        try:
            img = pygame.image.load(path).convert_alpha()
            img = pygame.transform.smoothscale(img, size)
            return img
        except Exception:
            pass

    # Placeholder
    surf.fill((35, 35, 35, 255))
    pygame.draw.rect(surf, (220, 220, 220), surf.get_rect(), 2)
    font = pygame.font.SysFont("arial", 16, bold=True)
    t = font.render(label, True, (240, 240, 240))
    surf.blit(t, (size[0] // 2 - t.get_width() // 2, size[1] // 2 - t.get_height() // 2))
    return surf


def pick_round_card(
    screen: pygame.Surface,
    *,
    cards: Optional[List[Dict]] = None,
    dim_alpha: int = 170,
) -> Optional[Dict]:
    """
    Modal overlay: draw 3 cards and force player to pick one.
    Returns the chosen card dict, or None if cancelled (ESC/close).
    """
    if not pygame.get_init():
        pygame.init()

    W, H = screen.get_size()
    clock = pygame.time.Clock()

    title_font = pygame.font.SysFont("arial", 30, bold=True)
    card_title_font = pygame.font.SysFont("arial", 22, bold=True)
    body_font = pygame.font.SysFont("arial", 18)

    if cards is None:
        cards = [
            {
                "id": "free_cow",
                "title": "Free cow",
                "desc": "Gain a cow for free.\n(+food / economy boost later)",
                "image": "free_cow.png",
            },
            {
                "id": "additional_axeman",
                "title": "Additional Axeman",
                "desc": "Recruit an extra axeman.\n(+combat advantage later)",
                "image": "additional_axeman.png",
            },
            {
                "id": "fortification",
                "title": "Fortification",
                "desc": "Build defenses on your land.\n(+defense chance later)",
                "image": "fortification.png",
            },
        ]

    # Layout
    pad = 24
    top_y = 70
    card_w = (W - pad * 4) // 3
    card_h = 360
    img_h = 150

    card_rects = []
    for i in range(3):
        x = pad + i * (card_w + pad)
        y = top_y
        card_rects.append(pygame.Rect(x, y, card_w, card_h))

    # Preload images
    images = []
    for c in cards:
        images.append(_load_png_or_placeholder(c["image"], (card_w - 24, img_h), c["title"]))

    def draw_wrapped_text(surf: pygame.Surface, text: str, font: pygame.font.Font, color, rect: pygame.Rect, line_gap=4):
        lines = []
        for paragraph in text.split("\n"):
            words = paragraph.split(" ")
            cur = ""
            for w in words:
                test = (cur + " " + w).strip()
                if font.size(test)[0] <= rect.w:
                    cur = test
                else:
                    if cur:
                        lines.append(cur)
                    cur = w
            if cur:
                lines.append(cur)
            lines.append("")  # paragraph gap
        if lines and lines[-1] == "":
            lines.pop()

        y = rect.y
        for line in lines:
            if line == "":
                y += font.get_height() // 2
                continue
            r = font.render(line, True, color)
            surf.blit(r, (rect.x, y))
            y += r.get_height() + line_gap
            if y > rect.bottom:
                break

    hovered = -1
    running = True

    while running:
        mx, my = pygame.mouse.get_pos()
        hovered = -1
        for i, r in enumerate(card_rects):
            if r.collidepoint(mx, my):
                hovered = i
                break

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return None
                if event.key == pygame.K_1:
                    return cards[0]
                if event.key == pygame.K_2:
                    return cards[1]
                if event.key == pygame.K_3:
                    return cards[2]

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                for i, r in enumerate(card_rects):
                    if r.collidepoint(event.pos):
                        return cards[i]

        # Dim background overlay
        dim = pygame.Surface((W, H), pygame.SRCALPHA)
        dim.fill((0, 0, 0, dim_alpha))
        screen.blit(dim, (0, 0))

        # Header
        header = title_font.render("Choose one card", True, (245, 245, 245))
        screen.blit(header, (W // 2 - header.get_width() // 2, 18))

        hint = body_font.render("Click a card or press 1/2/3", True, (230, 230, 230))
        screen.blit(hint, (W // 2 - hint.get_width() // 2, 48))

        # Cards
        for i, r in enumerate(card_rects):
            # Card panel
            base_col = (40, 40, 40)
            border_col = (200, 200, 200)
            if i == hovered:
                base_col = (55, 55, 55)
                border_col = (255, 255, 255)

            panel = pygame.Surface((r.w, r.h), pygame.SRCALPHA)
            panel.fill((*base_col, 245))
            pygame.draw.rect(panel, border_col, panel.get_rect(), 2)

            # Title
            t = card_title_font.render(cards[i]["title"], True, (245, 245, 245))
            panel.blit(t, (12, 10))

            # Image
            img = images[i]
            panel.blit(img, (12, 46))

            # Description
            text_rect = pygame.Rect(12, 46 + img_h + 12, r.w - 24, r.h - (46 + img_h + 24))
            draw_wrapped_text(panel, cards[i]["desc"], body_font, (235, 235, 235), text_rect)

            # Bottom key hint
            key_hint = body_font.render(f"[{i+1}]", True, (210, 210, 210))
            panel.blit(key_hint, (r.w - key_hint.get_width() - 10, r.h - key_hint.get_height() - 10))

            screen.blit(panel, (r.x, r.y))

        pygame.display.flip()
        clock.tick(60)

    return None


# Standalone test
if __name__ == "__main__":
    pygame.init()
    screen = pygame.display.set_mode((900, 600))
    pygame.display.set_caption("Card Choice Test")

    # Background test
    bg = pygame.Surface(screen.get_size())
    bg.fill((20, 30, 40))

    screen.blit(bg, (0, 0))
    pygame.display.flip()

    picked = pick_round_card(screen)
    print("Picked:", picked)

    # show result for a moment
    if picked:
        font = pygame.font.SysFont("arial", 30, bold=True)
        t = font.render(f"You picked: {picked['title']}", True, (255, 255, 255))
        screen.blit(bg, (0, 0))
        screen.blit(t, (screen.get_width() // 2 - t.get_width() // 2, screen.get_height() // 2 - t.get_height() // 2))
        pygame.display.flip()
        pygame.time.wait(1200)

    pygame.quit()
