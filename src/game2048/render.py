"""Dependency-free RGB rendering of a 2048 board (for videos and notebooks)."""

from __future__ import annotations

import numpy as np

# 3x5 bitmap font for digits 0-9.
_GLYPHS = {
    "0": ("111", "101", "101", "101", "111"),
    "1": ("010", "110", "010", "010", "111"),
    "2": ("111", "001", "111", "100", "111"),
    "3": ("111", "001", "111", "001", "111"),
    "4": ("101", "101", "111", "001", "001"),
    "5": ("111", "100", "111", "001", "111"),
    "6": ("111", "100", "111", "101", "111"),
    "7": ("111", "001", "001", "001", "001"),
    "8": ("111", "101", "111", "101", "111"),
    "9": ("111", "101", "111", "001", "111"),
}
_GLYPH_MASKS = {
    ch: np.array([[c == "1" for c in line] for line in rows], dtype=bool)
    for ch, rows in _GLYPHS.items()
}

# Classic 2048 palette, indexed by exponent; tiles beyond the list reuse the last colour.
_TILE_COLOURS = [
    (205, 193, 180),  # empty
    (238, 228, 218),  # 2
    (237, 224, 200),  # 4
    (242, 177, 121),  # 8
    (245, 149, 99),  # 16
    (246, 124, 95),  # 32
    (246, 94, 59),  # 64
    (237, 207, 114),  # 128
    (237, 204, 97),  # 256
    (237, 200, 80),  # 512
    (237, 197, 63),  # 1024
    (237, 194, 46),  # 2048
    (60, 58, 50),  # 4096+
]
_BACKGROUND = (187, 173, 160)
_DARK_TEXT = (119, 110, 101)
_LIGHT_TEXT = (249, 246, 242)


def _draw_number(img: np.ndarray, value: int, y0: int, x0: int, tile: int, colour) -> None:
    text = str(value)
    pad = tile // 8
    avail = tile - 2 * pad
    # Each glyph is 3 units wide with 1 unit spacing; 5 units tall.
    scale = max(1, min(avail // (4 * len(text) - 1), avail // 2 // 5))
    text_w = (4 * len(text) - 1) * scale
    text_h = 5 * scale
    y = y0 + (tile - text_h) // 2
    x = x0 + (tile - text_w) // 2
    for ch in text:
        mask = np.kron(_GLYPH_MASKS[ch], np.ones((scale, scale), dtype=bool))
        img[y : y + text_h, x : x + 3 * scale][mask] = colour
        x += 4 * scale


def render_rgb(board: np.ndarray, tile: int = 100, gap: int = 10) -> np.ndarray:
    """Render an exponent board as an ``(H, W, 3)`` ``uint8`` image."""
    size = board.shape[0]
    side = size * tile + (size + 1) * gap
    img = np.empty((side, side, 3), dtype=np.uint8)
    img[:] = _BACKGROUND
    for r in range(size):
        for c in range(size):
            exp = int(board[r, c])
            y0 = gap + r * (tile + gap)
            x0 = gap + c * (tile + gap)
            img[y0 : y0 + tile, x0 : x0 + tile] = _TILE_COLOURS[min(exp, len(_TILE_COLOURS) - 1)]
            if exp:
                colour = _DARK_TEXT if exp <= 2 else _LIGHT_TEXT
                _draw_number(img, 1 << exp, y0, x0, tile, colour)
    return img
