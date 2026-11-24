#!/usr/bin/env python3
"""
Render a 1024x1024 illustrated broadsheet front page summarizing the world news
of Monday, 23 November 2026 for make.fun bounty news-image-20261123.

Rendered at 2x supersampling (2048x2048) and downsampled with Lanczos filtering
for crisp typography, anti-aliased editorial vector art, and legibility at 300px.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Iterable, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Canvas dimensions (1x design space -> 2x render space)
SCALE = 2
W1, H1 = 1024, 1024
W, H = W1 * SCALE, H1 * SCALE

# Editorial broadsheet palette
PAPER = (246, 241, 231)
CARD_BG = (253, 250, 244)
INK = (19, 27, 42)
INK_MUTED = (62, 72, 88)
NAVY = (18, 29, 48)
NAVY_LIGHT = (31, 47, 74)
RULE_DARK = (32, 42, 58)
RULE_LIGHT = (206, 198, 184)
GOLD = (244, 196, 78)
GOLD_DEEP = (196, 138, 24)
CRIMSON = (194, 44, 38)
CRIMSON_LIGHT = (232, 86, 62)
EMERALD = (18, 126, 76)
EMERALD_BRIGHT = (52, 211, 122)
TEAL = (24, 102, 118)
STEEL = (29, 86, 136)
AMBER = (194, 104, 16)
INDIGO = (52, 68, 128)
PLUM = (138, 48, 66)
WHITE = (255, 255, 255)
CREAM = (250, 246, 238)

FONT_SERIF_BOLD = "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf"
FONT_SERIF_REG = "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf"
FONT_SANS_BOLD = "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"
FONT_SANS_REG = "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"


def s(v: float) -> int:
    return int(round(v * SCALE))


def pt(x: float, y: float) -> Tuple[int, int]:
    return (s(x), s(y))


def box(x0: float, y0: float, x1: float, y1: float) -> Tuple[int, int, int, int]:
    return (s(x0), s(y0), s(x1), s(y1))


def load_font(path: str, size_1x: float) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, s(size_1x))


def draw_vertical_gradient(
    img: Image.Image,
    rect_1x: Tuple[float, float, float, float],
    top_rgb: Tuple[int, int, int],
    bot_rgb: Tuple[int, int, int],
) -> None:
    x0, y0, x1, y1 = box(*rect_1x)
    h = max(1, y1 - y0)
    w = max(1, x1 - x0)
    t = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None, None]
    c0 = np.array(top_rgb, dtype=np.float32)[None, None, :]
    c1 = np.array(bot_rgb, dtype=np.float32)[None, None, :]
    grad = np.clip(c0 * (1.0 - t) + c1 * t, 0, 255).astype(np.uint8)
    grad = np.broadcast_to(grad, (h, w, 3)).copy()
    patch = Image.fromarray(grad, mode="RGB")
    img.paste(patch, (x0, y0))


def add_subtle_paper_grain(img: Image.Image, strength: float = 3.2) -> None:
    rng = np.random.default_rng(20261123)
    arr = np.asarray(img, dtype=np.int16)
    noise = rng.normal(0.0, strength, size=(arr.shape[0], arr.shape[1], 1)).astype(np.int16)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    img.paste(Image.fromarray(arr, mode="RGB"))


def measure_pill(
    draw: ImageDraw.ImageDraw,
    text: str,
    font_size: float = 16.5,
    pad_x: float = 9.0,
    pad_y: float = 4.0,
) -> Tuple[float, float]:
    font = load_font(FONT_SANS_BOLD, font_size)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = (bbox[2] - bbox[0]) / SCALE
    th = (bbox[3] - bbox[1]) / SCALE
    return (tw + 2 * pad_x, th + 2 * pad_y)


def draw_pill(
    draw: ImageDraw.ImageDraw,
    x: float,
    y: float,
    text: str,
    bg: Tuple[int, int, int],
    fg: Tuple[int, int, int] = WHITE,
    font_size: float = 16.5,
    pad_x: float = 9.0,
    pad_y: float = 4.0,
    radius: float = 4.0,
    align_right: bool = False,
) -> Tuple[float, float]:
    font = load_font(FONT_SANS_BOLD, font_size)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = (bbox[2] - bbox[0]) / SCALE
    th = (bbox[3] - bbox[1]) / SCALE
    w = tw + 2 * pad_x
    h = th + 2 * pad_y
    if align_right:
        x = x - w
    draw.rounded_rectangle(box(x, y, x + w, y + h), radius=s(radius), fill=bg)
    tx = x + pad_x - bbox[0] / SCALE
    ty = y + pad_y - bbox[1] / SCALE
    draw.text(pt(tx, ty), text, font=font, fill=fg)
    return (w, h)


def draw_centered_text(
    draw: ImageDraw.ImageDraw,
    cx: float,
    cy: float,
    text: str,
    font: ImageFont.FreeTypeFont,
    fill: Tuple[int, int, int],
) -> None:
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    draw.text((s(cx) - tw // 2 - bbox[0], s(cy) - th // 2 - bbox[1]), text, font=font, fill=fill)


def draw_card_frame(
    draw: ImageDraw.ImageDraw,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    accent: Tuple[int, int, int],
) -> None:
    # Subtle drop border + clean paper card
    draw.rounded_rectangle(box(x0 + 2, y0 + 2, x1 + 2, y1 + 2), radius=s(6), fill=(224, 217, 204))
    draw.rounded_rectangle(
        box(x0, y0, x1, y1),
        radius=s(6),
        fill=CARD_BG,
        outline=RULE_LIGHT,
        width=s(1.5),
    )
    # Crisp top editorial accent rule
    draw.rectangle(box(x0 + 1, y0 + 1, x1 - 1, y0 + 5), fill=accent)


def draw_masthead(img: Image.Image, draw: ImageDraw.ImageDraw) -> None:
    # Top classic broadsheet rules
    x0, y0, x1, y1 = 20, 18, 1004, 116
    draw_vertical_gradient(img, (x0, y0, x1, y1), (16, 26, 44), (24, 38, 62))
    draw.rectangle(box(x0, y0, x1, y0 + 5), fill=CRIMSON)
    draw.rectangle(box(x0, y1 - 4, x1, y1), fill=GOLD)

    # Stylized globe emblem on left
    gcx, gcy, gr = 64, 67, 29
    draw.ellipse(box(gcx - gr, gcy - gr, gcx + gr, gcy + gr), fill=(28, 66, 108), outline=GOLD, width=s(2.5))
    # Latitude/longitude arcs
    for rx in (11, 21):
        draw.ellipse(box(gcx - rx, gcy - gr, gcx + rx, gcy + gr), outline=(140, 186, 224), width=s(1.4))
    for dy in (-13, 0, 13):
        half_w = math.sqrt(max(0.0, gr * gr - dy * dy))
        draw.line([pt(gcx - half_w, gcy + dy), pt(gcx + half_w, gcy + dy)], fill=(140, 186, 224), width=s(1.4))
    # Subtle continent highlights
    draw.polygon([pt(48, 56), pt(60, 50), pt(64, 62), pt(54, 70), pt(46, 64)], fill=(96, 186, 128))
    draw.polygon([pt(68, 52), pt(82, 56), pt(80, 74), pt(70, 78), pt(66, 64)], fill=(96, 186, 128))

    # Main masthead title
    f_title = load_font(FONT_SERIF_BOLD, 54)
    draw.text(pt(108, 28), "WORLD NEWS", font=f_title, fill=CREAM)
    f_sub = load_font(FONT_SANS_BOLD, 16.5)
    draw.text(pt(110, 88), "ONE-DAY GLOBAL BRIEFING  ·  SIX CONTINENTS", font=f_sub, fill=(198, 212, 232))

    # Prominent Date Block on right
    dx0, dy0, dx1, dy1 = 556, 28, 990, 106
    draw.rounded_rectangle(box(dx0, dy0, dx1, dy1), radius=s(6), fill=(11, 18, 32), outline=GOLD, width=s(2))
    draw.rectangle(box(dx0, dy0, dx0 + 8, dy1), fill=CRIMSON)
    f_day = load_font(FONT_SANS_BOLD, 19.5)
    f_date = load_font(FONT_SANS_BOLD, 37)
    draw_centered_text(draw, (dx0 + dx1) / 2 + 4, dy0 + 19, "MONDAY", f_day, (220, 232, 248))
    draw_centered_text(draw, (dx0 + dx1) / 2 + 4, dy0 + 53, "23 NOVEMBER 2026", f_date, GOLD)


# ---------------------------------------------------------------------------
# Card 1 Illustration: Nvidia Reinstates CEO Jensen Huang (+31.5%, Nasdaq +6.5%)
# ---------------------------------------------------------------------------
def draw_art_nvidia(img: Image.Image, draw: ImageDraw.ImageDraw, rect: Tuple[float, float, float, float]) -> None:
    x0, y0, x1, y1 = rect
    w, h = x1 - x0, y1 - y0
    draw_vertical_gradient(img, rect, (11, 32, 26), (18, 52, 42))

    # Subtle chart gridlines
    for i in range(1, 5):
        gy = y0 + i * h / 5.0
        draw.line([pt(x0 + 8, gy), pt(x1 - 8, gy)], fill=(28, 72, 58), width=s(1))
    for i in range(1, 8):
        gx = x0 + i * w / 8.0
        draw.line([pt(gx, y0 + 8), pt(gx, y1 - 8)], fill=(24, 64, 52), width=s(1))

    # Subtle PCB circuit traces radiating from chip
    cx, cy = x0 + 108, y0 + 108
    for ty in (cy - 28, cy, cy + 28):
        draw.line([pt(x0 + 12, ty), pt(cx - 58, ty)], fill=(36, 98, 76), width=s(2))
        draw.ellipse(box(x0 + 10, ty - 2.5, x0 + 15, ty + 2.5), fill=EMERALD_BRIGHT)

    # Gold pins around substrate
    for offset in range(-42, 46, 12):
        draw.rectangle(box(cx + offset - 2, cy - 56, cx + offset + 2, cy - 47), fill=GOLD)
        draw.rectangle(box(cx + offset - 2, cy + 47, cx + offset + 2, cy + 56), fill=GOLD)
        draw.rectangle(box(cx - 56, cy + offset - 2, cx - 47, cy + offset + 2), fill=GOLD)
        draw.rectangle(box(cx + 47, cy + offset - 2, cx + 56, cy + offset + 2), fill=GOLD)
    # Dark emerald PCB substrate with subtle corner bevels
    draw.rounded_rectangle(box(cx - 48, cy - 48, cx + 48, cy + 48), radius=s(7), fill=(18, 76, 52), outline=EMERALD_BRIGHT, width=s(2.5))
    # Metallic heat spreader / silicon die
    draw.rounded_rectangle(box(cx - 32, cy - 32, cx + 32, cy + 32), radius=s(5), fill=(32, 48, 56), outline=(176, 224, 200), width=s(2))
    draw.rounded_rectangle(box(cx - 23, cy - 23, cx + 23, cy + 23), radius=s(3), fill=(18, 138, 80))
    draw_centered_text(draw, cx, cy, "NVDA", load_font(FONT_SANS_BOLD, 16.5), WHITE)

    # Stock chart surge curve on right
    pts_1x = [
        (x0 + 180, y1 - 34),
        (x0 + 214, y1 - 44),
        (x0 + 242, y1 - 26),
        (x0 + 270, y1 - 32),
        (x0 + 298, y1 - 76),
        (x0 + 332, y1 - 112),
        (x0 + 374, y1 - 134),
        (x0 + 422, y1 - 152),
    ]
    area_pts = [pt(pts_1x[0][0], y1 - 8)] + [pt(px, py) for px, py in pts_1x] + [pt(pts_1x[-1][0], y1 - 8)]
    draw.polygon(area_pts, fill=(24, 94, 62))

    # Vertical volume bars along bottom right
    bar_heights = [16, 22, 18, 28, 42, 54, 64, 76]
    for i, bh in enumerate(bar_heights):
        bx = x0 + 188 + i * 32
        col = (44, 156, 100) if i >= 4 else (32, 96, 70)
        draw.rectangle(box(bx, y1 - 8 - bh, bx + 16, y1 - 8), fill=col)

    # Crisp glowing line + nodes + arrowhead
    for a, b in zip(pts_1x[:-1], pts_1x[1:]):
        draw.line([pt(*a), pt(*b)], fill=EMERALD_BRIGHT, width=s(5))
    for px, py in pts_1x[3:]:
        draw.ellipse(box(px - 4.5, py - 4.5, px + 4.5, py + 4.5), fill=WHITE, outline=EMERALD, width=s(2))
    ax, ay = pts_1x[-1]
    draw.polygon([pt(ax + 12, ay - 6), pt(ax - 12, ay - 10), pt(ax - 2, ay + 12)], fill=EMERALD_BRIGHT)

    # Bold data callout pills (all safely inside viewport)
    draw_pill(draw, x0 + 12, y0 + 10, "NVDA +31.5%", (20, 158, 86), WHITE, font_size=19, pad_x=9, pad_y=4)
    draw_pill(draw, x0 + 182, y0 + 10, "NASDAQ +6.5%", (14, 24, 38), GOLD, font_size=15.5, pad_x=8, pad_y=5)
    draw_pill(draw, x1 - 12, y1 - 38, "+$952B VALUE", (14, 24, 38), EMERALD_BRIGHT, font_size=15.5, pad_x=8, pad_y=4, align_right=True)


# ---------------------------------------------------------------------------
# Card 2 Illustration: 3 Tankers Hit Mines in Strait of Hormuz
# ---------------------------------------------------------------------------
def draw_art_hormuz(img: Image.Image, draw: ImageDraw.ImageDraw, rect: Tuple[float, float, float, float]) -> None:
    x0, y0, x1, y1 = rect
    draw_vertical_gradient(img, rect, (15, 48, 78), (22, 74, 110))

    # Top Iranian coastline & bottom Arabian/Oman peninsula forming the Strait
    top_coast = [
        pt(x0, y0), pt(x1, y0), pt(x1, y0 + 36), pt(x1 - 80, y0 + 44),
        pt(x0 + 250, y0 + 54), pt(x0 + 160, y0 + 36), pt(x0 + 60, y0 + 42), pt(x0, y0 + 28)
    ]
    draw.polygon(top_coast, fill=(210, 174, 126))
    draw.line(top_coast[2:], fill=(158, 122, 78), width=s(2.5))

    bot_coast = [
        pt(x0, y1), pt(x1, y1), pt(x1, y1 - 24), pt(x1 - 110, y1 - 32),
        pt(x0 + 275, y1 - 46), pt(x0 + 215, y1 - 34), pt(x0 + 110, y1 - 18), pt(x0, y1 - 24)
    ]
    draw.polygon(bot_coast, fill=(218, 184, 136))
    draw.line(bot_coast[2:], fill=(158, 122, 78), width=s(2.5))

    # Water ripples
    for wx, wy in [(x0 + 28, y0 + 96), (x0 + 148, y0 + 128), (x0 + 345, y0 + 106), (x0 + 392, y0 + 78)]:
        draw.line([pt(wx, wy), pt(wx + 32, wy)], fill=(68, 136, 178), width=s(2))

    # Helper to draw burning oil tanker
    def draw_tanker(tx: float, ty: float, sc: float) -> None:
        # Subtle orange fire reflection on water below hull
        draw.ellipse(box(tx - 18 * sc, ty + 15 * sc, tx + 28 * sc, ty + 21 * sc), fill=(116, 62, 48))
        # Smoke plume rising up-right
        for idx, (ox, oy, rr) in enumerate([(8, -22, 13), (20, -34, 16), (34, -44, 18), (48, -52, 19)]):
            c = 44 + idx * 14
            draw.ellipse(box(tx + ox * sc - rr * sc, ty + oy * sc - rr * sc, tx + ox * sc + rr * sc, ty + oy * sc + rr * sc), fill=(c, c + 4, c + 10))
        # Hull
        hull = [
            pt(tx - 48 * sc, ty - 4 * sc), pt(tx + 46 * sc, ty - 4 * sc),
            pt(tx + 38 * sc, ty + 14 * sc), pt(tx - 40 * sc, ty + 14 * sc)
        ]
        draw.polygon(hull, fill=(32, 42, 56))
        # Red waterline
        draw.polygon([pt(tx - 42 * sc, ty + 8 * sc), pt(tx + 40 * sc, ty + 8 * sc), pt(tx + 38 * sc, ty + 14 * sc), pt(tx - 40 * sc, ty + 14 * sc)], fill=CRIMSON)
        # Deck pipes / cargo tanks
        draw.line([pt(tx - 22 * sc, ty - 6 * sc), pt(tx + 36 * sc, ty - 6 * sc)], fill=(154, 166, 180), width=max(1, s(2 * sc)))
        # Aft bridge
        draw.rectangle(box(tx - 42 * sc, ty - 22 * sc, tx - 24 * sc, ty - 4 * sc), fill=CREAM)
        draw.rectangle(box(tx - 39 * sc, ty - 18 * sc, tx - 27 * sc, ty - 12 * sc), fill=NAVY)
        # Vivid flame bursts on mid-deck
        draw.polygon([pt(tx - 8 * sc, ty - 4 * sc), pt(tx + 4 * sc, ty - 34 * sc), pt(tx + 18 * sc, ty - 4 * sc)], fill=(238, 72, 26))
        draw.polygon([pt(tx + 4 * sc, ty - 4 * sc), pt(tx + 18 * sc, ty - 28 * sc), pt(tx + 30 * sc, ty - 4 * sc)], fill=(250, 126, 24))
        draw.polygon([pt(tx - 1 * sc, ty - 4 * sc), pt(tx + 8 * sc, ty - 20 * sc), pt(tx + 16 * sc, ty - 4 * sc)], fill=(255, 224, 92))

    # Draw 3 tankers (foreground main + two mid/distant in the strait)
    draw_tanker(x0 + 366, y0 + 82, 0.56)
    draw_tanker(x0 + 102, y0 + 84, 0.65)
    draw_tanker(x0 + 224, y0 + 108, 1.05)

    # Spiked naval sea mines in foreground water
    for mx, my, mr in [(x0 + 114, y0 + 144, 10.5), (x0 + 334, y0 + 132, 9.5)]:
        draw.ellipse(box(mx - mr - 7, my - mr - 7, mx + mr + 7, my + mr + 7), outline=(248, 120, 72), width=s(2))
        for deg in range(0, 360, 45):
            rad = math.radians(deg)
            draw.line([pt(mx, my), pt(mx + math.cos(rad) * (mr + 5.5), my + math.sin(rad) * (mr + 5.5))], fill=(26, 32, 42), width=s(3))
        draw.ellipse(box(mx - mr, my - mr, mx + mr, my + mr), fill=(36, 44, 56), outline=(120, 134, 152), width=s(1.5))
        draw.ellipse(box(mx - 3, my - 3, mx + 3, my + 3), fill=CRIMSON_LIGHT)

    # Crisp right-aligned and left-aligned pills inside viewport
    draw_pill(draw, x0 + 10, y0 + 10, "STRAIT OF HORMUZ", NAVY, GOLD, font_size=15, pad_x=8, pad_y=4)
    draw_pill(draw, x1 - 10, y0 + 10, "BRENT $89.4 (-3.5%)", CRIMSON, WHITE, font_size=14.5, pad_x=8, pad_y=4, align_right=True)


# ---------------------------------------------------------------------------
# Card 3 Illustration: Ukraine Sinks Russian Corvette Merkury in Libya
# ---------------------------------------------------------------------------
def draw_art_ukraine_libya(img: Image.Image, draw: ImageDraw.ImageDraw, rect: Tuple[float, float, float, float]) -> None:
    x0, y0, x1, y1 = rect
    draw_vertical_gradient(img, rect, (18, 56, 92), (12, 38, 66))

    # Libyan coastline across bottom
    coast = [pt(x0, y1), pt(x1, y1), pt(x1, y1 - 24), pt(x0 + 210, y1 - 34), pt(x0 + 120, y1 - 20), pt(x0, y1 - 28)]
    draw.polygon(coast, fill=(210, 182, 136))
    draw.line(coast[2:], fill=(154, 126, 84), width=s(2))

    # Subtle dashed trajectory arc from upper-center (Ukraine 2,400 km north) to target
    arc_pts = [(x0 + 190, y0 + 10), (x0 + 186, y0 + 26), (x0 + 180, y0 + 42), (x0 + 174, y0 + 58), (x0 + 168, y0 + 74)]
    for i in range(0, len(arc_pts) - 1):
        p0, p1 = arc_pts[i], arc_pts[i + 1]
        mid_p = ((p0[0] + p1[0]) * 0.5, (p0[1] + p1[1]) * 0.5)
        draw.line([pt(*p0), pt(*mid_p)], fill=GOLD, width=s(2.5))

    # Listing/sinking Russian corvette Merkury on right
    hull = [
        pt(x0 + 148, y0 + 82), pt(x0 + 278, y0 + 64),
        pt(x0 + 270, y0 + 92), pt(x0 + 154, y0 + 102)
    ]
    draw.polygon(hull, fill=(96, 108, 124), outline=(44, 54, 68), width=s(2))
    # Darker lower hull shading
    draw.polygon([pt(x0 + 151, y0 + 92), pt(x0 + 274, y0 + 78), pt(x0 + 270, y0 + 92), pt(x0 + 154, y0 + 102)], fill=(66, 78, 92))
    # Stealth superstructure, deck gun, and radar mast tilted
    draw.polygon([pt(x0 + 186, y0 + 76), pt(x0 + 240, y0 + 68), pt(x0 + 234, y0 + 48), pt(x0 + 194, y0 + 53)], fill=(134, 146, 160))
    draw.polygon([pt(x0 + 206, y0 + 51), pt(x0 + 219, y0 + 49), pt(x0 + 212, y0 + 26)], fill=(172, 184, 198))
    # Deck gun turret on bow
    draw.line([pt(x0 + 252, y0 + 66), pt(x0 + 268, y0 + 56)], fill=(182, 194, 206), width=s(2.5))

    # Explosion flash + smoke plume at waterline
    bx, by = x0 + 166, y0 + 88
    for ox, oy, rr, col in [(16, -24, 14, (64, 70, 80)), (32, -38, 17, (86, 92, 102)), (48, -50, 15, (108, 114, 124))]:
        draw.ellipse(box(bx + ox - rr, by + oy - rr, bx + ox + rr, by + oy + rr), fill=col)
    star = []
    for k in range(12):
        ang = math.radians(k * 30)
        rad = 22 if k % 2 == 0 else 10
        star.append(pt(bx + math.cos(ang) * rad, by + math.sin(ang) * rad))
    draw.polygon(star, fill=(242, 92, 32))
    draw.ellipse(box(bx - 9, by - 9, bx + 9, by + 9), fill=(255, 232, 108))

    # 3 sleek Ukrainian naval drones approaching with white V-wakes
    for dx, dy in [(x0 + 44, y0 + 76), (x0 + 74, y0 + 102), (x0 + 106, y0 + 86)]:
        draw.line([pt(dx - 26, dy - 7), pt(dx - 4, dy)], fill=(168, 218, 246), width=s(2))
        draw.line([pt(dx - 26, dy + 7), pt(dx - 4, dy)], fill=(168, 218, 246), width=s(2))
        draw.polygon([pt(dx - 10, dy - 3.2), pt(dx + 12, dy), pt(dx - 10, dy + 3.2)], fill=(20, 26, 36), outline=WHITE, width=s(1.2))

    draw_pill(draw, x0 + 10, y0 + 10, "BENGHAZI · LIBYA", NAVY, GOLD, font_size=14, pad_x=7, pad_y=3.5)
    draw_pill(draw, x1 - 10, y1 - 28, "MERKURY", CRIMSON, WHITE, font_size=13.5, pad_x=7, pad_y=3, align_right=True)


# ---------------------------------------------------------------------------
# Card 4 Illustration: U.S. Blacklists Megaspeed Cloud (SE Asia -> China)
# ---------------------------------------------------------------------------
def draw_art_megaspeed(img: Image.Image, draw: ImageDraw.ImageDraw, rect: Tuple[float, float, float, float]) -> None:
    x0, y0, x1, y1 = rect
    draw_vertical_gradient(img, rect, (26, 30, 44), (44, 34, 38))

    for gy in (y0 + 36, y0 + 76, y0 + 114):
        draw.line([pt(x0 + 12, gy), pt(x1 - 12, gy)], fill=(54, 62, 82), width=s(1.2))

    # Server rack / AI cloud cluster on left
    rx0, ry0, rx1, ry1 = x0 + 24, y0 + 38, x0 + 124, y0 + 134
    draw.rounded_rectangle(box(rx0, ry0, rx1, ry1), radius=s(6), fill=(36, 48, 68), outline=(104, 134, 174), width=s(2))
    for i in range(4):
        sy = ry0 + 9 + i * 22
        draw.rounded_rectangle(box(rx0 + 8, sy, rx1 - 8, sy + 15), radius=s(2), fill=(20, 28, 42))
        draw.ellipse(box(rx0 + 14, sy + 4, rx0 + 21, sy + 11), fill=(58, 206, 198))
        draw.rectangle(box(rx0 + 28, sy + 5, rx1 - 14, sy + 10), fill=(72, 108, 148))

    # Severed connection lines to SE Asia nodes on right
    draw.line([pt(rx1 + 4, y0 + 86), pt(x0 + 216, y0 + 52)], fill=GOLD, width=s(3))
    draw.line([pt(rx1 + 4, y0 + 86), pt(x0 + 224, y0 + 86)], fill=GOLD, width=s(3))
    draw.line([pt(rx1 + 4, y0 + 86), pt(x0 + 216, y0 + 120)], fill=GOLD, width=s(3))

    for nx, ny, lbl in [(x0 + 246, y0 + 50, "SG"), (x0 + 254, y0 + 86, "MY"), (x0 + 246, y0 + 120, "TH")]:
        draw.ellipse(box(nx - 18, ny - 15, nx + 18, ny + 15), fill=(46, 62, 88), outline=GOLD, width=s(2))
        draw_centered_text(draw, nx, ny, lbl, load_font(FONT_SANS_BOLD, 14), WHITE)

    # Prohibition / blacklist seal cleanly over link
    cx, cy, r = x0 + 156, y0 + 86, 33
    draw.ellipse(box(cx - r, cy - r, cx + r, cy + r), fill=(46, 18, 22), outline=CRIMSON_LIGHT, width=s(6))
    d = r * 0.68
    draw.line([pt(cx - d, cy - d), pt(cx + d, cy + d)], fill=CRIMSON_LIGHT, width=s(6))

    draw_pill(draw, x0 + 10, y0 + 10, "ENTITY LIST · 7 UNITS", CRIMSON, WHITE, font_size=14, pad_x=8, pad_y=3.5)


# ---------------------------------------------------------------------------
# Card 5 Illustration: COP31 Talks End Without a Deal (Antalya)
# ---------------------------------------------------------------------------
def draw_art_cop31(img: Image.Image, draw: ImageDraw.ImageDraw, rect: Tuple[float, float, float, float]) -> None:
    x0, y0, x1, y1 = rect
    mid = (x0 + x1) / 2.0
    # Split backdrop: clean green-sky left vs smoggy warm-ochre right
    draw_vertical_gradient(img, (x0, y0, mid, y1), (206, 236, 224), (168, 212, 196))
    draw_vertical_gradient(img, (mid, y0, x1, y1), (234, 206, 178), (194, 150, 120))

    # Split Earth globe centered cleanly below the top corner pills
    gcx, gcy, gr = mid, y0 + 88, 45
    # Left half of globe
    draw.pieslice(box(gcx - 5 - gr, gcy - gr, gcx - 5 + gr, gcy + gr), start=90, end=270, fill=(36, 128, 168), outline=INK, width=s(2.5))
    draw.polygon([pt(gcx - 38, gcy - 14), pt(gcx - 16, gcy - 26), pt(gcx - 10, gcy + 10), pt(gcx - 32, gcy + 20)], fill=(44, 162, 96))

    # Sleek tapered wind turbine on left
    tx, ty = x0 + 46, y0 + 82
    draw.polygon([pt(tx - 3, y1 - 10), pt(tx + 3, y1 - 10), pt(tx + 1.5, ty), pt(tx - 1.5, ty)], fill=INK)
    for ang_deg in (-90, 30, 150):
        ang = math.radians(ang_deg)
        perp = ang + math.pi / 2.0
        bx, by = tx + math.cos(ang) * 28, ty + math.sin(ang) * 28
        blade = [
            pt(tx + math.cos(perp) * 3.2, ty + math.sin(perp) * 3.2),
            pt(bx, by),
            pt(tx - math.cos(perp) * 2.2, ty - math.sin(perp) * 2.2),
        ]
        draw.polygon(blade, fill=WHITE, outline=INK, width=s(1.3))
    draw.ellipse(box(tx - 4, ty - 4, tx + 4, ty + 4), fill=INK)

    # Right half of globe
    draw.pieslice(box(gcx + 5 - gr, gcy - gr, gcx + 5 + gr, gcy + gr), start=-90, end=90, fill=(96, 86, 84), outline=INK, width=s(2.5))
    draw.polygon([pt(gcx + 12, gcy - 20), pt(gcx + 38, gcy - 10), pt(gcx + 32, gcy + 22), pt(gcx + 14, gcy + 12)], fill=(142, 92, 62))

    # Smokestacks / fossil-fuel plant on right
    sx = x1 - 58
    draw.rectangle(box(sx, y0 + 78, sx + 14, y1 - 10), fill=(52, 48, 54))
    draw.rectangle(box(sx + 20, y0 + 90, sx + 32, y1 - 10), fill=(68, 62, 68))
    draw.rectangle(box(sx - 4, y1 - 24, sx + 38, y1 - 10), fill=(44, 40, 46))
    for ox, oy, rr in [(8, -10, 8.5), (18, -20, 11), (30, -28, 12)]:
        draw.ellipse(box(sx + ox - rr, y0 + 78 + oy - rr, sx + ox + rr, y0 + 78 + oy + rr), fill=(116, 108, 106))

    # Jagged fault crack cleanly between the two halves
    crack = [pt(mid, y0 + 36), pt(mid - 4.5, y0 + 62), pt(mid + 5.5, y0 + 88), pt(mid - 4.5, y0 + 114), pt(mid + 2, y0 + 140)]
    draw.line(crack, fill=CRIMSON, width=s(4))

    draw_pill(draw, x0 + 8, y0 + 9, "COP31 ANTALYA", NAVY, WHITE, font_size=13.5, pad_x=7, pad_y=3.5)
    draw_pill(draw, x1 - 8, y0 + 9, "NO DEAL", CRIMSON, WHITE, font_size=13.5, pad_x=8, pad_y=3.5, align_right=True)


# ---------------------------------------------------------------------------
# Card 6 Illustration: FBI Probes NCIC Crime Database Breach (297 x 82)
# ---------------------------------------------------------------------------
def draw_art_fbi_cyber(img: Image.Image, draw: ImageDraw.ImageDraw, rect: Tuple[float, float, float, float]) -> None:
    x0, y0, x1, y1 = rect
    draw_vertical_gradient(img, rect, (20, 28, 56), (34, 44, 82))

    for gx in range(int(x0 + 16), int(x1 - 16), 24):
        draw.line([pt(gx, y0 + 8), pt(gx, y1 - 8)], fill=(44, 58, 98), width=s(1))

    # Database cylinder stack cleanly clear of the NCIC pill
    dx, dy = x0 + 84, y0 + 46
    for oy in (14, 0, -14):
        draw.rectangle(box(dx - 24, dy + oy - 5, dx + 24, dy + oy + 7), fill=(52, 86, 134), outline=(146, 190, 236), width=s(1.5))
        draw.ellipse(box(dx - 24, dy + oy - 10, dx + 24, dy + oy), fill=(74, 118, 174), outline=(146, 190, 236), width=s(1.5))

    # Law enforcement shield in center (text clean and unobstructed)
    scx, scy = x0 + 160, y0 + 41
    shield = [
        pt(scx - 24, scy - 24), pt(scx, scy - 30), pt(scx + 24, scy - 24),
        pt(scx + 22, scy + 8), pt(scx, scy + 28), pt(scx - 22, scy + 8)
    ]
    draw.polygon(shield, fill=NAVY, outline=GOLD, width=s(3))
    draw_centered_text(draw, scx, scy - 2, "FBI", load_font(FONT_SANS_BOLD, 16), GOLD)

    # Red breach lightning between database/shield and alert triangle
    draw.line([pt(scx + 24, scy - 8), pt(scx + 34, scy + 1), pt(scx + 29, scy + 5), pt(scx + 41, scy + 13)], fill=CRIMSON_LIGHT, width=s(3))

    # Warning triangle on right
    wx, wy = x0 + 242, y0 + 44
    draw.polygon([pt(wx, wy - 23), pt(wx + 25, wy + 20), pt(wx - 25, wy + 20)], fill=CRIMSON, outline=GOLD, width=s(2.5))
    draw_centered_text(draw, wx, wy + 4, "!", load_font(FONT_SANS_BOLD, 23), WHITE)

    draw_pill(draw, x0 + 8, y0 + 8, "NCIC", CRIMSON, WHITE, font_size=13, pad_x=6.5, pad_y=2.5)


# ---------------------------------------------------------------------------
# Card 7 Illustration: Drone Strike Hits Police Base in Cúcuta, Colombia (297 x 82)
# ---------------------------------------------------------------------------
def draw_art_colombia(img: Image.Image, draw: ImageDraw.ImageDraw, rect: Tuple[float, float, float, float]) -> None:
    x0, y0, x1, y1 = rect
    draw_vertical_gradient(img, rect, (44, 34, 54), (124, 64, 54))

    # Distant Andean hills
    hills = [pt(x0, y1), pt(x0, y0 + 50), pt(x0 + 75, y0 + 34), pt(x0 + 165, y0 + 46), pt(x0 + 240, y0 + 30), pt(x1, y0 + 44), pt(x1, y1)]
    draw.polygon(hills, fill=(58, 42, 54))

    # City rooftops silhouette (Cúcuta residential neighborhood + police base tower)
    buildings = [
        (10, 58, 44), (48, 48, 78), (84, 42, 120), (126, 56, 160),
        (166, 58, 204), (210, 52, 248), (254, 58, 288)
    ]
    for bx0, by0, bx1 in buildings:
        col = (34, 52, 82) if bx0 == 84 else (30, 28, 38)
        draw.rectangle(box(x0 + bx0, y0 + by0, x0 + bx1, y1), fill=col)
    # Lit police base windows
    for wy in (y0 + 50, y0 + 62):
        for wx in (x0 + 91, x0 + 101, x0 + 111):
            draw.rectangle(box(wx, wy, wx + 5, wy + 5), fill=(118, 204, 246))

    # Signal jamming arcs staying cleanly inside the top border
    jx, jy = x0 + 102, y0 + 41
    for rr in (10, 18, 26):
        draw.arc(box(jx - rr, jy - rr, jx + rr, jy + rr), start=215, end=335, fill=(104, 212, 248), width=s(2))

    # Incoming quadcopter drone + blast over adjacent homes
    dx, dy = x0 + 188, y0 + 26
    draw.line([pt(dx - 14, dy), pt(dx + 14, dy)], fill=WHITE, width=s(2.5))
    draw.ellipse(box(dx - 5, dy - 4, dx + 5, dy + 4), fill=INK, outline=WHITE, width=s(1.5))
    draw.line([pt(dx - 18, dy - 4), pt(dx - 10, dy - 4)], fill=GOLD, width=s(2))
    draw.line([pt(dx + 10, dy - 4), pt(dx + 18, dy - 4)], fill=GOLD, width=s(2))

    # Impact flash on residential rooftops
    fx, fy = x0 + 226, y0 + 52
    star = []
    for k in range(10):
        ang = math.radians(k * 36)
        rad = 17 if k % 2 == 0 else 7.5
        star.append(pt(fx + math.cos(ang) * rad, fy + math.sin(ang) * rad))
    draw.polygon(star, fill=(246, 102, 34))
    draw.ellipse(box(fx - 6, fy - 6, fx + 6, fy + 6), fill=(255, 232, 114))

    draw_pill(draw, x0 + 8, y0 + 8, "CÚCUTA", NAVY, GOLD, font_size=13, pad_x=7, pad_y=2.5)


# ---------------------------------------------------------------------------
# Card 8 Illustration: M6.5 Deep Quake Off Vanuatu (298 x 82)
# ---------------------------------------------------------------------------
def draw_art_vanuatu(img: Image.Image, draw: ImageDraw.ImageDraw, rect: Tuple[float, float, float, float]) -> None:
    x0, y0, x1, y1 = rect
    draw_vertical_gradient(img, rect, (12, 52, 74), (16, 76, 98))

    # Concentric seismic wave rings cleanly inside the ocean panel
    ecx, ecy = x0 + 104, y0 + 46
    for rr, col in [(28, (214, 92, 62)), (19, (238, 128, 62)), (11, (252, 184, 72))]:
        draw.ellipse(box(ecx - rr, ecy - rr, ecx + rr, ecy + rr), outline=col, width=s(2.4))
    draw.ellipse(box(ecx - 5, ecy - 5, ecx + 5, ecy + 5), fill=CRIMSON_LIGHT, outline=WHITE, width=s(1.5))

    # Vanuatu volcanic island arc silhouettes neatly clear of M6.5 pill
    for ix, iy, irx, iry in [(x0 + 56, y0 + 40, 7, 5), (x0 + 66, y0 + 53, 5.5, 8), (x0 + 78, y0 + 66, 6.5, 4.5), (x0 + 91, y0 + 72, 4.5, 3.5)]:
        draw.ellipse(box(ix - irx, iy - iry, ix + irx, iy + iry), fill=(76, 174, 110))

    # Crisp seismograph waveform on right
    wave_x = np.linspace(x0 + 142, x1 - 10, 36)
    offsets = [0, 1, -1, 2, -2, 3, -5, 8, -13, 20, -23, 24, -20, 16, -13, 10, -8, 6, -4, 3, -3, 2, -2, 2, -1, 1, 0, 1, -1, 1, 0, 0, 0, 0, 0, 0]
    wave_pts = [pt(float(wx), y0 + 48 + float(dy)) for wx, dy in zip(wave_x, offsets)]
    draw.line(wave_pts, fill=(116, 234, 214), width=s(2.5))

    draw_pill(draw, x0 + 8, y0 + 8, "M6.5", CRIMSON, WHITE, font_size=13.5, pad_x=7, pad_y=2.5)
    draw_pill(draw, x1 - 8, y0 + 8, "601 KM DEEP", NAVY, GOLD, font_size=12.5, pad_x=6, pad_y=2.5, align_right=True)


def draw_story_card(
    img: Image.Image,
    draw: ImageDraw.ImageDraw,
    rect: Tuple[float, float, float, float],
    kicker: str,
    accent: Tuple[int, int, int],
    art_fn,
    art_h: float,
    headline_lines: Sequence[str],
    deck_lines: Sequence[str],
    headline_size: float,
    deck_size: float,
    kicker_size: float = 15.5,
) -> None:
    x0, y0, x1, y1 = rect
    draw_card_frame(draw, x0, y0, x1, y1, accent)

    pad = 10.0
    # Kicker pill
    _, kh = draw_pill(draw, x0 + pad, y0 + 11, kicker, accent, WHITE, font_size=kicker_size, pad_x=8, pad_y=3.5, radius=3.5)

    # Art viewport
    ay0 = y0 + 11 + kh + 7
    ay1 = ay0 + art_h
    art_rect = (x0 + pad, ay0, x1 - pad, ay1)
    art_fn(img, draw, art_rect)
    draw.rectangle(box(*art_rect), outline=(182, 174, 160), width=s(1.2))

    # Headline (bold high-contrast serif)
    f_head = load_font(FONT_SERIF_BOLD, headline_size)
    hy = ay1 + 8
    line_step = headline_size * 1.12
    for line in headline_lines:
        draw.text(pt(x0 + pad, hy), line, font=f_head, fill=INK)
        hy += line_step

    # Crisp deck sub-label
    f_deck = load_font(FONT_SANS_BOLD, deck_size)
    dy = hy + 4
    deck_step = deck_size * 1.24
    for line in deck_lines:
        draw.text(pt(x0 + pad, dy), line, font=f_deck, fill=INK_MUTED)
        dy += deck_step


def render(out_path: Path, preview_300_path: Path | None = None) -> None:
    img = Image.new("RGB", (W, H), PAPER)
    draw = ImageDraw.Draw(img)

    # Masthead: y = 18..116
    draw_masthead(img, draw)

    # Row 1: Two Hero Lead Stories (y = 128..466, height = 338)
    draw_story_card(
        img, draw,
        rect=(20, 128, 504, 466),
        kicker="TECH & MARKETS · USA",
        accent=EMERALD,
        art_fn=draw_art_nvidia,
        art_h=186,
        headline_lines=["Nvidia Reinstates", "CEO Jensen Huang"],
        deck_lines=["Shares soar 31.5% (+$952B) in Wall St rally"],
        headline_size=31,
        deck_size=19,
        kicker_size=16,
    )

    draw_story_card(
        img, draw,
        rect=(520, 128, 1004, 466),
        kicker="MIDDLE EAST · ENERGY",
        accent=CRIMSON,
        art_fn=draw_art_hormuz,
        art_h=186,
        headline_lines=["3 Tankers Hit Mines", "in Strait of Hormuz"],
        deck_lines=["Iran says ships ablaze; Brent dips to $89.4"],
        headline_size=31,
        deck_size=19,
        kicker_size=16,
    )

    # Row 2: Three Feature Stories (y = 480..786, height = 306)
    draw_story_card(
        img, draw,
        rect=(20, 480, 337, 786),
        kicker="EUROPE · LIBYA",
        accent=STEEL,
        art_fn=draw_art_ukraine_libya,
        art_h=148,
        headline_lines=["Ukraine Sinks", "Warship in Libya"],
        deck_lines=["7 sea drones hit Russian", "corvette at Benghazi"],
        headline_size=25,
        deck_size=17.5,
        kicker_size=15,
    )

    draw_story_card(
        img, draw,
        rect=(353, 480, 670, 786),
        kicker="ASIA · U.S. SANCTIONS",
        accent=AMBER,
        art_fn=draw_art_megaspeed,
        art_h=148,
        headline_lines=["U.S. Blacklists", "Megaspeed Cloud"],
        deck_lines=["7 SE Asia units barred", "over AI chip diversion"],
        headline_size=25,
        deck_size=17.5,
        kicker_size=15,
    )

    draw_story_card(
        img, draw,
        rect=(686, 480, 1004, 786),
        kicker="CLIMATE · TURKEY",
        accent=TEAL,
        art_fn=draw_art_cop31,
        art_h=148,
        headline_lines=["COP31 Talks End", "Without a Deal"],
        deck_lines=["Antalya summit breaks", "down over fossil fuels"],
        headline_size=25,
        deck_size=17.5,
        kicker_size=15,
    )

    # Row 3: Three World Dispatches (y = 800..1006, height = 206)
    draw_story_card(
        img, draw,
        rect=(20, 800, 337, 1006),
        kicker="U.S. · CYBERSECURITY",
        accent=INDIGO,
        art_fn=draw_art_fbi_cyber,
        art_h=82,
        headline_lines=["FBI Probes NCIC", "Database Breach"],
        deck_lines=["Apparent hack hits police files"],
        headline_size=22.5,
        deck_size=16.5,
        kicker_size=14,
    )

    draw_story_card(
        img, draw,
        rect=(353, 800, 670, 1006),
        kicker="COLOMBIA · SECURITY",
        accent=PLUM,
        art_fn=draw_art_colombia,
        art_h=82,
        headline_lines=["Drone Strike Hits", "Cúcuta Police Base"],
        deck_lines=["3 hurt, 60 homes damaged"],
        headline_size=22.5,
        deck_size=16.5,
        kicker_size=14,
    )

    draw_story_card(
        img, draw,
        rect=(686, 800, 1004, 1006),
        kicker="PACIFIC · EARTHQUAKE",
        accent=(26, 110, 128),
        art_fn=draw_art_vanuatu,
        art_h=82,
        headline_lines=["M6.5 Deep Quake", "Strikes Off Vanuatu"],
        deck_lines=["601 km deep; no tsunami threat"],
        headline_size=22.5,
        deck_size=16.5,
        kicker_size=14,
    )

    # Downsample 2x -> 1x with Lanczos for crisp anti-aliasing
    final_img = img.resize((W1, H1), Image.Resampling.LANCZOS)
    final_img.save(out_path, format="PNG", optimize=True, compress_level=9)

    if preview_300_path is not None:
        preview = final_img.resize((300, 300), Image.Resampling.LANCZOS)
        preview.save(preview_300_path, format="PNG", optimize=True)


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parent
    output_png = repo_root / "entry.png"
    preview_env = os.environ.get("PREVIEW_300_PATH")
    preview_path = Path(preview_env) if preview_env else None
    render(output_png, preview_path)
    size_bytes = output_png.stat().st_size
    print(f"Rendered {output_png} ({W1}x{H1}, {size_bytes:,} bytes / {size_bytes / 1024:.1f} KiB)")
