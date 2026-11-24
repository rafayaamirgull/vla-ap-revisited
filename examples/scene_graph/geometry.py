"""Metric geometry and evidence-aware spatial relations for one RGB-D frame."""

from __future__ import annotations

import math

import numpy as np
from scipy.spatial import cKDTree


def checked_transform(value, name: str) -> np.ndarray:
    transform = np.asarray(value, dtype=np.float64)
    if (transform.shape != (4, 4) or not np.isfinite(transform).all() or
            not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-6) or
            not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-3) or
            not np.isclose(np.linalg.det(transform[:3, :3]), 1.0, atol=1e-3)):
        raise ValueError(f"Invalid rigid transform: {name}")
    return transform


def world_from_optical(metadata: dict) -> tuple[np.ndarray, str]:
    direct = checked_transform(metadata["world_from_camera_optical"],
                               "world_from_camera_optical")
    chain = metadata.get("kinematic_chain")
    if chain is None:
        return direct, "recorded_world_camera_pose"
    world_base = checked_transform(chain["world_from_base"], "world_from_base")
    base_ee = checked_transform(chain["base_from_ee"], "base_from_ee")
    ee_camera = checked_transform(chain["ee_from_camera_optical"],
                                  "ee_from_camera_optical")
    composed = world_base @ base_ee @ ee_camera
    if not np.allclose(composed, direct, atol=1e-5):
        raise ValueError("Recorded camera pose disagrees with the base-to-wrist-to-camera chain")
    return composed, "world_base_x_base_ee_x_ee_camera_optical"


def backproject_mask(depth: np.ndarray, mask: np.ndarray, intrinsics: dict,
                     min_depth: float, max_depth: float, max_pixels: int
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Return optical XYZ and corresponding (row, column) for valid pixel centers."""
    if depth.ndim != 2 or mask.shape != depth.shape:
        raise ValueError("Depth and mask must have the same height and width")
    fx, fy = float(intrinsics["fx"]), float(intrinsics["fy"])
    cx, cy = float(intrinsics["cx"]), float(intrinsics["cy"])
    if not all(math.isfinite(v) for v in (fx, fy, cx, cy)) or fx <= 0 or fy <= 0:
        raise ValueError("Invalid camera intrinsics")
    valid = np.asarray(mask, dtype=bool) & np.isfinite(depth)
    valid &= (depth >= min_depth) & (depth <= max_depth)
    rows, columns = np.nonzero(valid)
    if len(rows) > max_pixels:
        stride = math.ceil(len(rows) / max_pixels)
        rows, columns = rows[::stride], columns[::stride]
    z = depth[rows, columns].astype(np.float64)
    points = np.column_stack(((columns + 0.5 - cx) * z / fx,
                              (rows + 0.5 - cy) * z / fy, z))
    return points, np.column_stack((rows, columns))


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]


def dense_component_pixels(depth: np.ndarray, mask: np.ndarray, intrinsics: dict,
                           world_from_camera: np.ndarray, cluster_world: np.ndarray,
                           min_depth: float, max_depth: float, radius: float) -> np.ndarray:
    """Recover full-resolution mask pixels close to the retained 3D component."""
    optical, pixels = backproject_mask(depth, mask, intrinsics, min_depth,
                                       max_depth, mask.size)
    if len(pixels) == 0:
        return pixels
    distances, _ = cKDTree(cluster_world).query(transform_points(optical, world_from_camera),
                                               distance_upper_bound=radius)
    return pixels[np.isfinite(distances)]


def largest_metric_cluster(points: np.ndarray, voxel_size: float, radius: float,
                           min_points: int) -> tuple[np.ndarray, dict]:
    """Join occupied metric voxels and keep the component with most source points."""
    if len(points) < min_points:
        return np.empty(0, dtype=int), {"components": 0, "largest_points": 0}
    voxels, inverse = np.unique(np.floor(points / voxel_size).astype(np.int64),
                                axis=0, return_inverse=True)
    counts = np.bincount(inverse)
    centers = np.zeros((len(voxels), 3), dtype=np.float64)
    np.add.at(centers, inverse, points)
    centers /= counts[:, None]
    tree = cKDTree(centers)
    labels = np.full(len(voxels), -1, dtype=int)
    component_sizes = []
    for seed in range(len(voxels)):
        if labels[seed] != -1:
            continue
        component = len(component_sizes)
        labels[seed] = component
        stack = [seed]
        size = 0
        while stack:
            current = stack.pop()
            size += int(counts[current])
            for neighbor in tree.query_ball_point(centers[current], radius):
                if labels[neighbor] == -1:
                    labels[neighbor] = component
                    stack.append(neighbor)
        component_sizes.append(size)
    best = int(np.argmax(component_sizes))
    if component_sizes[best] < min_points:
        return np.empty(0, dtype=int), {"components": len(component_sizes),
                                        "largest_points": component_sizes[best]}
    retained = np.flatnonzero(labels[inverse] == best)
    return retained, {"components": len(component_sizes),
                      "largest_points": len(retained)}


def observed_aabb(points: np.ndarray) -> dict:
    lower, upper = np.percentile(points, [2, 98], axis=0)
    return {"min": lower.tolist(), "max": upper.tolist()}


def contact_patch(points: np.ndarray, support_top: float, band: float,
                  voxel_size: float, radius: float) -> dict:
    """Find connected visible points close to a known horizontal support plane."""
    near = points[(points[:, 2] >= support_top - 0.01) &
                  (points[:, 2] <= support_top + band)]
    retained, _ = largest_metric_cluster(near, voxel_size, radius, 1)
    return {
        "band_above_support_m": band,
        "points_in_band": len(near),
        "largest_connected_patch_points": len(retained),
        "nearest_observed_gap_m": float(near[:, 2].min() - support_top) if len(near) else None,
        "patch_centroid_world_m": np.median(near[retained], axis=0).tolist()
        if len(retained) else None,
    }


def xy_overlap_fraction(a: dict, b: dict) -> float:
    a_min, a_max = np.asarray(a["min"]), np.asarray(a["max"])
    b_min, b_max = np.asarray(b["min"]), np.asarray(b["max"])
    intersection = np.maximum(0.0, np.minimum(a_max[:2], b_max[:2]) -
                               np.maximum(a_min[:2], b_min[:2]))
    smaller_area = min(np.prod(a_max[:2] - a_min[:2]),
                       np.prod(b_max[:2] - b_min[:2]))
    return float(np.prod(intersection) / smaller_area) if smaller_area > 0 else 0.0


def xy_clearance(a: dict, b: dict) -> float:
    a_min, a_max = np.asarray(a["min"]), np.asarray(a["max"])
    b_min, b_max = np.asarray(b["min"]), np.asarray(b["max"])
    gap = np.maximum(0.0, np.maximum(a_min[:2] - b_max[:2], b_min[:2] - a_max[:2]))
    return float(np.linalg.norm(gap))


def pairwise_geometry(objects: list[dict]) -> list[dict]:
    """Measure unordered object pairs from visible-surface centroids and bounds."""
    pairs = []
    for index, source in enumerate(objects):
        for target in objects[index + 1:]:
            a, b = source["world_aabb_m"], target["world_aabb_m"]
            a_min, a_max = np.asarray(a["min"]), np.asarray(a["max"])
            b_min, b_max = np.asarray(b["min"]), np.asarray(b["max"])
            delta = (np.asarray(target["centroid_world_m"]) -
                     np.asarray(source["centroid_world_m"]))
            intersection_size = np.maximum(0.0, np.minimum(a_max, b_max) -
                                           np.maximum(a_min, b_min))
            intersection = float(np.prod(intersection_size))
            union = (float(np.prod(a_max - a_min)) +
                     float(np.prod(b_max - b_min)) - intersection)
            pairs.append({
                "source": source["id"], "target": target["id"],
                "frame": "workspace_world", "geometry_source": "observed_masked_depth",
                "centroid_delta_target_minus_source_world_m": delta.tolist(),
                "centroid_distance_m": float(np.linalg.norm(delta)),
                "centroid_xy_distance_m": float(np.linalg.norm(delta[:2])),
                "xy_clearance_m": xy_clearance(a, b),
                "xy_overlap_fraction_of_smaller": xy_overlap_fraction(a, b),
                "observed_aabb_iou_3d": intersection / union if union > 0 else 0.0,
            })
    return pairs


def infer_directional_relations(objects: list[dict], deadband: float = 0.04,
                                min_vertical_xy_overlap: float = 0.15) -> list[dict]:
    """Order visible objects in the workspace: +X right, +Y front, +Z up.

    XY relations use observed centroids with a deadband. Vertical relations
    require separated observed Z intervals and overlapping XY footprints.
    Every asserted directed relation has its inverse in the output.
    """
    if deadband < 0 or not 0 <= min_vertical_xy_overlap <= 1:
        raise ValueError("Invalid directional relation limits")
    edges = []
    for index, a in enumerate(objects):
        for b in objects[index + 1:]:
            delta = np.asarray(b["centroid_world_m"]) - np.asarray(a["centroid_world_m"])
            for axis, negative, positive in ((0, "left_of", "right_of"),
                                             (1, "behind", "in_front_of")):
                if abs(delta[axis]) <= deadband:
                    continue
                a_relation, b_relation = ((negative, positive) if delta[axis] > 0
                                          else (positive, negative))
                evidence = {"method": "observed_centroid_axis_order",
                            "axis": "xyz"[axis], "deadband_m": deadband}
                edges.extend((
                    {"source": a["id"], "target": b["id"], "relation": a_relation,
                     "frame": "workspace_world",
                     "evidence": {**evidence, "target_minus_source_m": float(delta[axis])}},
                    {"source": b["id"], "target": a["id"], "relation": b_relation,
                     "frame": "workspace_world",
                     "evidence": {**evidence, "target_minus_source_m": -float(delta[axis])}},
                ))
            overlap = xy_overlap_fraction(a["world_aabb_m"], b["world_aabb_m"])
            a_bounds, b_bounds = a["world_aabb_m"], b["world_aabb_m"]
            a_above_gap = float(a_bounds["min"][2] - b_bounds["max"][2])
            b_above_gap = float(b_bounds["min"][2] - a_bounds["max"][2])
            if overlap < min_vertical_xy_overlap or max(a_above_gap, b_above_gap) < 0:
                continue
            upper, lower = (a, b) if a_above_gap >= 0 else (b, a)
            evidence = {"method": "separated_observed_z_intervals",
                        "observed_vertical_clearance_m": max(a_above_gap, b_above_gap),
                        "xy_overlap_fraction": overlap,
                        "min_xy_overlap_fraction": min_vertical_xy_overlap}
            edges.extend((
                {"source": upper["id"], "target": lower["id"],
                 "relation": "above", "frame": "workspace_world", "evidence": evidence},
                {"source": lower["id"], "target": upper["id"],
                 "relation": "below", "frame": "workspace_world", "evidence": evidence},
            ))
    return edges


def infer_between_relations(objects: list[dict], max_perpendicular: float = 0.06,
                            min_anchor_distance: float = 0.15,
                            max_height_spread: float = 0.12) -> list[dict]:
    """Find tabletop A between anchors B and C; retain the two anchors as a pair."""
    if max_perpendicular < 0 or min_anchor_distance <= 0 or max_height_spread < 0:
        raise ValueError("Invalid between relation limits")
    relations = []
    for subject in objects:
        anchors = sorted((node for node in objects if node["id"] != subject["id"]),
                         key=lambda node: node["id"])
        for index, first in enumerate(anchors):
            for second in anchors[index + 1:]:
                center = np.asarray(subject["centroid_world_m"])
                start = np.asarray(first["centroid_world_m"])
                end = np.asarray(second["centroid_world_m"])
                baseline = end[:2] - start[:2]
                separation = float(np.linalg.norm(baseline))
                if separation < min_anchor_distance:
                    continue
                fraction = float(np.dot(center[:2] - start[:2], baseline) /
                                 separation ** 2)
                perpendicular = float(np.linalg.norm(
                    center[:2] - (start[:2] + fraction * baseline)))
                height_spread = float(np.ptp([center[2], start[2], end[2]]))
                if (0.15 <= fraction <= 0.85 and perpendicular <= max_perpendicular and
                        height_spread <= max_height_spread):
                    relations.append({
                        "subject": subject["id"],
                        "references": [first["id"], second["id"]],
                        "relation": "between", "frame": "workspace_world_xy",
                        "evidence": {"method": "observed_centroid_segment_projection",
                                     "segment_fraction": fraction,
                                     "perpendicular_distance_m": perpendicular,
                                     "anchor_separation_m": separation,
                                     "centroid_height_spread_m": height_spread},
                    })
    return relations


def infer_relations(objects: list[dict], table: dict | None,
                    interiors: dict[str, dict] | None = None,
                    support_tolerance: float = 0.025,
                    neighbor_distance: float = 0.08,
                    min_contact_points: int = 20) -> list[dict]:
    """Produce geometric hypotheses; no edge is a calibrated probability."""
    edges = []
    supports = objects + ([table] if table is not None else [])
    for subject in objects:
        a = subject["world_aabb_m"]
        for target in supports:
            if target["id"] == subject["id"]:
                continue
            b = target["world_aabb_m"]
            gap = float(a["min"][2] - b["max"][2])
            overlap = xy_overlap_fraction(a, b)
            contact = subject.get("table_contact") if target["id"] == "table" else None
            has_contact_patch = (contact is not None and
                                 contact["largest_connected_patch_points"] >= min_contact_points)
            ordered_above_object = (a["min"][2] > b["min"][2] and
                                    a["max"][2] > b["max"][2])
            ordered_above_support = (subject["centroid_world_m"][2] > b["min"][2]
                                     if target["id"] == "table" else ordered_above_object)
            if ((-0.01 <= gap <= support_tolerance or has_contact_patch) and
                    overlap >= 0.3 and ordered_above_support):
                evidence = {"observed_bottom_gap_m": gap,
                            "xy_overlap_fraction": overlap,
                            "method": "connected_visible_contact_patch" if has_contact_patch
                            else "observed_support_heuristic"}
                if contact is not None:
                    evidence["table_contact"] = contact
                edges.append({"source": subject["id"], "target": target["id"],
                              "relation": "on", "evidence": evidence})

    for index, left in enumerate(objects):
        for right in objects[index + 1:]:
            a, b = left["world_aabb_m"], right["world_aabb_m"]
            clearance = xy_clearance(a, b)
            vertical = abs(left["centroid_world_m"][2] - right["centroid_world_m"][2])
            if (clearance <= neighbor_distance and vertical <= 0.15 and
                    xy_overlap_fraction(a, b) < 0.1 and
                    not any(e["relation"] == "on" and
                            {e["source"], e["target"]} == {left["id"], right["id"]}
                            for e in edges)):
                edges.append({"source": left["id"], "target": right["id"],
                              "relation": "next_to", "symmetric": True,
                              "evidence": {"xy_clearance_m": clearance,
                                           "centroid_height_difference_m": vertical,
                                           "method": "observed_aabb_heuristic"}})

    if interiors is not None and not isinstance(interiors, dict):
        raise ValueError("Interiors must map container object IDs to world AABBs")
    for container_id, interior in (interiors or {}).items():
        container = next((node for node in objects if node["id"] == container_id), None)
        if container is None:
            raise ValueError(f"Interior supplied for unknown object ID: {container_id}")
        lower, upper = np.asarray(interior["min"], dtype=float), np.asarray(interior["max"], dtype=float)
        if (lower.shape != (3,) or upper.shape != (3,) or
                not np.isfinite(lower).all() or not np.isfinite(upper).all() or
                not np.all(lower < upper)):
            raise ValueError(f"Invalid interior world AABB for {container_id}")
        for subject in objects:
            if subject["id"] == container_id:
                continue
            a = subject["world_aabb_m"]
            if np.all(np.asarray(a["min"]) >= lower) and np.all(np.asarray(a["max"]) <= upper):
                edges.append({"source": subject["id"], "target": container_id,
                              "relation": "inside", "evidence": {
                                  "method": "user_supplied_interior_world_aabb",
                                  "interior_world_aabb_m": interior}})
    return edges
