"""Small RGB PNG encoder for environments without Pillow or imageio."""

import struct
import zlib

import numpy as np


def _chunk(tag: bytes, payload: bytes) -> bytes:
    return (struct.pack(">I", len(payload)) + tag + payload
            + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF))


def rgb_png(rgb: np.ndarray) -> bytes:
    """Encode an HxWx3 uint8 array as PNG."""
    height, width, channels = rgb.shape
    if channels != 3 or rgb.dtype != np.uint8:
        raise ValueError("expected HxWx3 uint8 RGB")
    scanlines = b"".join(b"\x00" + row.tobytes() for row in rgb)
    return (b"\x89PNG\r\n\x1a\n"
            + _chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + _chunk(b"IDAT", zlib.compress(scanlines, level=3))
            + _chunk(b"IEND", b""))
