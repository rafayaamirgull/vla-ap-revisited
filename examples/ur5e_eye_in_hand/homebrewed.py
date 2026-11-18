"""Convert HomebrewedDB colored PLY objects to textured MuJoCo OBJ assets.

MuJoCo does not read PLY vertex colors. Each input triangle gets a tiny texture
tile whose colors are interpolated from its three original PLY vertex colors.
The conversion is cached and leaves the source dataset untouched.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from png_io import rgb_png


CACHE_VERSION = 1
TILE = 4


@dataclass(frozen=True)
class PreparedObject:
    object_id: int
    mesh_path: Path
    texture_path: Path
    size_m: tuple[float, float, float]
    face_count: int


def model_dir(root: Path) -> Path:
    """Accept the dataset root or its models directory."""
    if (root / "models_info.json").is_file():
        return root
    return root / "hb_models" / "models"


def available_objects(root: Path, include_large: bool = False) -> list[int]:
    directory = model_dir(root)
    info_path = directory / "models_info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"HomebrewedDB models_info.json not found under {root}")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    ids = []
    for key, size in info.items():
        object_id = int(key)
        if not (directory / f"obj_{object_id:06d}.ply").is_file():
            continue
        if not include_large and (
            max(size["size_x"], size["size_y"]) > 220.0 or size["size_z"] > 180.0
        ):
            continue
        ids.append(object_id)
    if len(ids) < 3:
        raise ValueError(f"Fewer than three usable HomebrewedDB objects in {directory}")
    return sorted(ids)


def _read_ply(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with path.open("r", encoding="ascii") as handle:
        if handle.readline().strip() != "ply" or handle.readline().strip() != "format ascii 1.0":
            raise ValueError(f"Expected ASCII PLY: {path}")
        vertex_count = face_count = None
        properties: list[str] = []
        section = ""
        for raw in handle:
            line = raw.strip()
            if line == "end_header":
                break
            parts = line.split()
            if len(parts) == 3 and parts[0] == "element":
                section = parts[1]
                if section == "vertex":
                    vertex_count = int(parts[2])
                elif section == "face":
                    face_count = int(parts[2])
            elif section == "vertex" and len(parts) == 3 and parts[0] == "property":
                properties.append(parts[2])
        else:
            raise ValueError(f"Missing PLY end_header: {path}")
        if vertex_count is None or face_count is None:
            raise ValueError(f"Missing vertex or face count: {path}")
        required = ("x", "y", "z", "nx", "ny", "nz", "red", "green", "blue")
        if not all(name in properties for name in required):
            raise ValueError(f"Missing geometry or RGB fields: {path}")
        vertex_data = np.loadtxt(handle, dtype=np.float32, max_rows=vertex_count, ndmin=2)
        face_data = np.loadtxt(handle, dtype=np.int32, max_rows=face_count, ndmin=2)
    if vertex_data.shape != (vertex_count, len(properties)):
        raise ValueError(f"Unexpected PLY vertex shape: {path}")
    if face_data.shape != (face_count, 4) or not np.all(face_data[:, 0] == 3):
        raise ValueError(f"Expected triangular PLY faces: {path}")
    faces = face_data[:, 1:4]
    if np.any(faces < 0) or np.any(faces >= vertex_count):
        raise ValueError(f"Invalid PLY face index: {path}")
    xyz = vertex_data[:, [properties.index(name) for name in ("x", "y", "z")]]
    normals = vertex_data[:, [properties.index(name) for name in ("nx", "ny", "nz")]]
    rgb = np.clip(
        vertex_data[:, [properties.index(name) for name in ("red", "green", "blue")]],
        0, 255,
    ).astype(np.uint8)
    return xyz, normals, rgb, faces


def _texture(rgb: np.ndarray, faces: np.ndarray, columns: int, rows: int) -> np.ndarray:
    face_count = len(faces)
    tiles = np.zeros((rows * columns, TILE, TILE, 3), dtype=np.uint8)
    y, x = np.mgrid[:TILE, :TILE]
    # A, B, C lie at pixel centers (0.5,0.5), (2.5,0.5), (0.5,2.5).
    b = np.clip(x / 2.0, 0.0, 1.0)
    c = np.clip(y / 2.0, 0.0, 1.0)
    a = np.clip(1.0 - b - c, 0.0, 1.0)
    weights = np.stack((a, b, c), axis=-1).astype(np.float32)
    weights /= weights.sum(axis=-1, keepdims=True)
    for start in range(0, face_count, 8192):
        stop = min(start + 8192, face_count)
        colors = rgb[faces[start:stop]].astype(np.float32)
        pixels = np.einsum("xyk,nkc->nxyc", weights, colors)
        tiles[start:stop] = np.rint(pixels).astype(np.uint8)
    return tiles.reshape(rows, columns, TILE, TILE, 3).transpose(0, 2, 1, 3, 4).reshape(
        rows * TILE, columns * TILE, 3
    )


def _write_obj(path: Path, xyz: np.ndarray, normals: np.ndarray, faces: np.ndarray,
               columns: int, rows: int) -> None:
    count = len(faces)
    face_ids = np.arange(count, dtype=np.int32)
    tile_x = face_ids % columns
    tile_y = face_ids // columns
    width = columns * TILE
    height = rows * TILE
    uv = np.empty((count * 3, 2), dtype=np.float32)
    uv[0::3, 0] = (tile_x * TILE + 0.5) / width
    uv[0::3, 1] = 1.0 - (tile_y * TILE + 0.5) / height
    uv[1::3, 0] = (tile_x * TILE + 2.5) / width
    uv[1::3, 1] = uv[0::3, 1]
    uv[2::3, 0] = uv[0::3, 0]
    uv[2::3, 1] = 1.0 - (tile_y * TILE + 2.5) / height
    with path.open("w", encoding="ascii") as handle:
        handle.write("# HomebrewedDB PLY vertex colors baked into a texture atlas\n")
        np.savetxt(handle, xyz, fmt="v %.7f %.7f %.7f")
        np.savetxt(handle, normals, fmt="vn %.6f %.6f %.6f")
        np.savetxt(handle, uv, fmt="vt %.8f %.8f")
        for face_id, (a, b, c) in enumerate(faces):
            t = face_id * 3 + 1
            handle.write(f"f {a+1}/{t}/{a+1} {b+1}/{t+1}/{b+1} {c+1}/{t+2}/{c+1}\n")


def prepare_object(root: Path, object_id: int, cache_dir: Path) -> PreparedObject:
    source = model_dir(root) / f"obj_{object_id:06d}.ply"
    if not source.is_file():
        raise FileNotFoundError(source)
    cache_dir.mkdir(parents=True, exist_ok=True)
    stem = source.stem
    obj_path = cache_dir / f"{stem}.obj"
    png_path = cache_dir / f"{stem}.png"
    info_path = cache_dir / f"{stem}.json"
    source_stat = source.stat()
    if obj_path.is_file() and png_path.is_file() and info_path.is_file():
        try:
            cached = json.loads(info_path.read_text(encoding="utf-8"))
            size_m = cached["size_m"]
            face_count = cached["face_count"]
            if (cached.get("version") == CACHE_VERSION and
                    cached.get("source_size") == source_stat.st_size and
                    cached.get("source_mtime_ns") == source_stat.st_mtime_ns and
                    isinstance(size_m, list) and len(size_m) == 3 and
                    all(type(size) in (int, float) and math.isfinite(size) and size > 0
                        for size in size_m) and
                    type(face_count) is int and face_count > 0):
                return PreparedObject(object_id, obj_path, png_path,
                                      tuple(size_m), face_count)
        except (OSError, UnicodeError, ValueError, KeyError, TypeError, AttributeError):
            pass

    print(f"Converting {source.name} with its original vertex colors...", flush=True)
    xyz_mm, normals, rgb, faces = _read_ply(source)
    bounds_min = xyz_mm.min(axis=0)
    bounds_max = xyz_mm.max(axis=0)
    origin = np.array([(bounds_min[0] + bounds_max[0]) / 2.0,
                       (bounds_min[1] + bounds_max[1]) / 2.0,
                       bounds_min[2]], dtype=np.float32)
    xyz_m = (xyz_mm - origin) * 0.001
    size_m = tuple(float(x) for x in (bounds_max - bounds_min) * 0.001)
    columns = math.ceil(math.sqrt(len(faces)))
    rows = math.ceil(len(faces) / columns)
    atlas = _texture(rgb, faces, columns, rows)
    obj_tmp = obj_path.with_name(f"{obj_path.name}.tmp.{os.getpid()}")
    png_tmp = png_path.with_name(f"{png_path.name}.tmp.{os.getpid()}")
    _write_obj(obj_tmp, xyz_m, normals, faces, columns, rows)
    png_tmp.write_bytes(rgb_png(atlas))
    obj_tmp.replace(obj_path)
    png_tmp.replace(png_path)
    cache_data = {
        "version": CACHE_VERSION,
        "source_size": source_stat.st_size,
        "source_mtime_ns": source_stat.st_mtime_ns,
        "size_m": size_m,
        "face_count": len(faces),
    }
    info_tmp = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=cache_dir,
                                         prefix=f"{stem}.json.tmp.", delete=False) as handle:
            info_tmp = Path(handle.name)
            handle.write(json.dumps(cache_data, indent=2) + "\n")
        info_tmp.replace(info_path)
    finally:
        if info_tmp is not None:
            info_tmp.unlink(missing_ok=True)
    return PreparedObject(object_id, obj_path, png_path, size_m, len(faces))
