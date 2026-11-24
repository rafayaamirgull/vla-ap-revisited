"""Build a metric, instance-level scene graph from aligned wrist RGB-D and masks."""

from __future__ import annotations

import argparse
import hashlib
import json
import textwrap
from pathlib import Path

import cv2
import numpy as np

from geometry import (backproject_mask, contact_patch, dense_component_pixels, infer_relations,
                      largest_metric_cluster, observed_aabb, transform_points,
                      world_from_optical)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT = ROOT / "outputs/open_vocab_pipeline/frame_0000_detections.json"
DEFAULT_OUTPUT = ROOT / "outputs/scene_graph"


def report_stem(path: Path) -> str:
    return path.stem[:-len("_detections")] if path.stem.endswith("_detections") else path.stem


def sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def load_aligned_inputs(report_path: Path, report: dict) -> tuple[np.ndarray, np.ndarray]:
    inputs = report.get("inputs")
    if not isinstance(inputs, dict) or not inputs.get("depth"):
        raise ValueError(f"{report_path}: aligned RGB-D fingerprints are missing. "
                         "Re-run the open-vocabulary pipeline on a simulator RGB-D capture.")
    rgb_info, depth_info = inputs["rgb"], inputs["depth"]
    rgb_path = (report_path.parent / rgb_info["path"]).resolve()
    depth_path = (report_path.parent / depth_info["path"]).resolve()
    bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"RGB image not found or unreadable: {rgb_path}")
    depth = np.load(depth_path, allow_pickle=False)
    if (depth.ndim != 2 or depth.shape != bgr.shape[:2] or
            depth.dtype.kind != "f" or depth.dtype.itemsize != 4):
        raise ValueError("Expected aligned float32 depth with the RGB dimensions")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    depth = np.asarray(depth, dtype="<f4", order="C")
    if (rgb_info.get("encoding") != "rgb8" or
            rgb_info.get("shape") != list(rgb.shape) or
            depth_info.get("encoding") != "float32_le_meters" or
            depth_info.get("shape") != list(depth.shape)):
        raise ValueError("RGB-D input shape or encoding disagrees with the detection report")
    if sha256(rgb) != rgb_info["sha256"]:
        raise ValueError(f"RGB fingerprint mismatch for {rgb_path}; the frame was replaced")
    if sha256(depth) != depth_info["sha256"]:
        raise ValueError(f"Depth fingerprint mismatch for {depth_path}; the frame was replaced")
    return rgb, depth


def load_mask(report_path: Path, detection: dict, shape: tuple[int, int]) -> np.ndarray:
    mask_path = report_path.parent / detection["mask_file"]
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None or mask.shape != shape:
        raise ValueError(f"Mask missing or wrong size: {mask_path}")
    binary = (mask > 127).astype(np.uint8)
    expected = detection.get("mask_sha256")
    if expected is None or sha256(binary) != expected:
        raise ValueError(f"Mask fingerprint missing or mismatched: {mask_path}")
    return binary.astype(bool)


def clip_crop(rgb: np.ndarray, pixels: np.ndarray) -> np.ndarray:
    """Square crop from full-resolution pixels of the dominant 3D component."""
    rows, columns = pixels[:, 0], pixels[:, 1]
    pad = 8
    y1, y2 = max(0, int(rows.min()) - pad), min(rgb.shape[0], int(rows.max()) + pad + 1)
    x1, x2 = max(0, int(columns.min()) - pad), min(rgb.shape[1], int(columns.max()) + pad + 1)
    side = max(y2 - y1, x2 - x1)
    top, left = (side - (y2 - y1)) // 2, (side - (x2 - x1)) // 2
    crop = np.full((side, side, 3), 128, dtype=np.uint8)
    crop[rows - y1 + top, columns - x1 + left] = rgb[rows, columns]
    return crop


class ClipEncoder:
    def __init__(self, device: str) -> None:
        self.requested_device = device
        self.model = None

    def encode(self, crops: list[np.ndarray], prompts: list[str]) -> tuple[np.ndarray, np.ndarray]:
        try:
            import clip
            import torch
        except ImportError as exc:
            raise SystemExit("Install examples/open_vocab_pipeline/requirements.txt for CLIP") from exc
        from PIL import Image

        if self.model is None:
            self.device = (("cuda" if torch.cuda.is_available() else "cpu")
                           if self.requested_device == "auto" else self.requested_device)
            self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
            self.model.eval()
        images = torch.stack([self.preprocess(Image.fromarray(crop)) for crop in crops]).to(self.device)
        tokens = clip.tokenize(prompts).to(self.device)
        with torch.inference_mode():
            image_features = self.model.encode_image(images).float()
            text_features = self.model.encode_text(tokens).float()
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarities = image_features @ text_features.T
        return image_features.cpu().numpy(), similarities.cpu().numpy()


def table_node(metadata: dict) -> dict | None:
    for item in metadata.get("static_geometry", []):
        if item["name"] == "table":
            aabb = item["world_aabb_m"]
            minimum, maximum = np.asarray(aabb["min"]), np.asarray(aabb["max"])
            return {"id": "table", "type": "support_surface", "label": "table",
                    "geometry_source": "simulator_static_geometry", "world_aabb_m": aabb,
                    "centroid_world_m": ((minimum + maximum) / 2).tolist()}
    return None


def make_graph(report_path: Path, encoder: ClipEncoder, args: argparse.Namespace
               ) -> tuple[dict, dict[str, np.ndarray]]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    rgb, depth = load_aligned_inputs(report_path, report)
    metadata = report.get("camera_metadata")
    if not isinstance(metadata, dict):
        raise ValueError("A simulator capture with camera metadata is required")
    if metadata.get("depth_encoding") != "float32_le_meters":
        raise ValueError("Expected axial camera depth in float32 meters")
    if (metadata["height"], metadata["width"]) != depth.shape:
        raise ValueError("Camera metadata dimensions do not match RGB-D")
    world_camera, pose_source = world_from_optical(metadata)
    capture_hashes = metadata.get("capture_fingerprints")
    if not isinstance(capture_hashes, dict):
        raise ValueError("Camera metadata lacks RGB-D capture fingerprints. "
                         "Re-capture the simulator frame and rerun perception.")
    if (capture_hashes.get("rgb_sha256") != report["inputs"]["rgb"]["sha256"] or
            capture_hashes.get("depth_sha256") != report["inputs"]["depth"]["sha256"]):
        raise ValueError("Camera metadata fingerprints disagree with the RGB-D frame")
    table = table_node(metadata)
    nodes = [table] if table is not None else []
    objects, crops, pointclouds, discarded = [], [], {}, []

    for index, detection in enumerate(report["detections"]):
        mask = load_mask(report_path, detection, depth.shape)
        if not args.allow_border_masks and (mask[0].any() or mask[-1].any() or
                                            mask[:, 0].any() or mask[:, -1].any()):
            discarded.append({"mask_index": index, "reason": "truncated_at_image_border"})
            continue
        optical, pixels = backproject_mask(depth, mask, metadata["intrinsics_opencv"],
                                           args.min_depth, args.max_depth, args.max_pixels)
        if len(optical) < args.min_points:
            discarded.append({"mask_index": index, "reason": "insufficient_valid_depth",
                              "valid_points": len(optical)})
            continue
        world = transform_points(optical, world_camera)
        retained, cluster = largest_metric_cluster(world, args.voxel_size,
                                                    args.cluster_radius, args.min_points)
        if not len(retained):
            discarded.append({"mask_index": index, "reason": "no_large_3d_cluster",
                              "valid_points": len(world), **cluster})
            continue
        selected = world[retained]
        selected_pixels = pixels[retained]
        appearance_pixels = dense_component_pixels(
            depth, mask, metadata["intrinsics_opencv"], world_camera, selected,
            args.min_depth, args.max_depth, args.cluster_radius)
        object_id = f"object_{index:02d}"
        pointclouds[f"{object_id}_world_xyz_m"] = selected.astype(np.float32)
        pointclouds[f"{object_id}_rgb8"] = rgb[selected_pixels[:, 0], selected_pixels[:, 1]]
        node = {
            "id": object_id, "type": "object", "label": detection["label"],
            "geometry_source": "masked_depth_largest_metric_cluster",
            "detection_confidence": float(detection["confidence"]),
            "mask_index": index, "mask_file": detection["mask_file"],
            "centroid_world_m": np.median(selected, axis=0).tolist(),
            "world_aabb_m": observed_aabb(selected),
            "mask_pixels": int(np.count_nonzero(mask)),
            "valid_depth_points": len(world), "cluster_points": len(selected),
            "appearance_pixels": len(appearance_pixels),
            "cluster_components": cluster["components"],
            "pointcloud_key": f"{object_id}_world_xyz_m",
        }
        if table is not None:
            node["table_contact"] = contact_patch(
                selected, table["world_aabb_m"]["max"][2], args.contact_band,
                args.voxel_size, args.cluster_radius)
        objects.append(node)
        crops.append(clip_crop(rgb, appearance_pixels))

    if objects:
        embeddings, similarities = encoder.encode(crops, report["prompts"])
        for node, crop, vector, scores in zip(objects, crops, embeddings, similarities):
            ranked = np.argsort(scores)[::-1]
            top_prompt = report["prompts"][int(ranked[0])]
            margin = float(scores[ranked[0]] - scores[ranked[1]]) if len(ranked) > 1 else None
            node["semantic_status"] = (
                "distinct_within_prompts" if top_prompt == node["label"] and
                margin is not None and margin >= args.clip_margin
                else "ambiguous_within_prompts")
            crop_name = f"{report_stem(report_path)}_{node['id']}_clip_crop.png"
            crop_path = args.output_dir / crop_name
            args.output_dir.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(crop_path), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)):
                raise OSError(f"Could not write {crop_path}")
            node["clip"] = {
                "model": "ViT-B/32", "crop_file": crop_name,
                "crop_source": "full_resolution_pixels_near_retained_3d_cluster",
                "image_embedding_l2": [round(float(value), 7) for value in vector],
                "top_prompt": top_prompt,
                "margin_to_second_prompt": round(margin, 5) if margin is not None else None,
                "text_cosine": {prompt: round(float(value), 5)
                                for prompt, value in zip(report["prompts"], scores)},
            }
    nodes.extend(objects)
    interiors = (json.loads(args.interiors.read_text(encoding="utf-8"))
                 if args.interiors is not None else None)
    edges = infer_relations(objects, table, interiors, args.support_tolerance,
                            args.neighbor_distance, args.min_contact_points)
    graph = {
        "schema_version": 1, "coordinate_frame": "MuJoCo world, meters",
        "depth_convention": "camera optical axial Z, meters",
        "frame": metadata.get("frame"), "simulation_time_s": metadata.get("simulation_time_s"),
        "camera": {"intrinsics_opencv": metadata["intrinsics_opencv"],
                   "world_from_camera_optical": world_camera.tolist(),
                   "pose_source": pose_source,
                   "kinematic_chain": metadata.get("kinematic_chain")},
        "provenance": {"detections_report": str(report_path),
                       "rgb_sha256": report["inputs"]["rgb"]["sha256"],
                       "depth_sha256": report["inputs"]["depth"]["sha256"],
                       "detector_model": report["models"]["detector"],
                       "segmenter_model": report["models"]["segmenter"],
                       "clip_model": "ViT-B/32"},
        "parameters": {"min_depth_m": args.min_depth, "max_depth_m": args.max_depth,
                       "max_mask_pixels": args.max_pixels, "voxel_size_m": args.voxel_size,
                       "cluster_radius_m": args.cluster_radius,
                       "min_cluster_points": args.min_points,
                       "support_tolerance_m": args.support_tolerance,
                       "contact_band_m": args.contact_band,
                       "min_contact_patch_points": args.min_contact_points,
                       "neighbor_distance_m": args.neighbor_distance,
                       "clip_prompt_margin": args.clip_margin,
                       "allow_border_masks": args.allow_border_masks},
        "nodes": nodes, "edges": edges, "discarded_detections": discarded,
    }
    return graph, pointclouds


def plot_graph(graph: dict, clouds: dict[str, np.ndarray], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig = plt.figure(figsize=(12, 5.5), dpi=140)
    ax = fig.add_subplot(121, projection="3d")
    relations_ax = fig.add_subplot(122)
    objects = [node for node in graph["nodes"] if node["type"] == "object"]
    table = next((node for node in graph["nodes"] if node["id"] == "table"), None)
    all_points = [clouds[node["pointcloud_key"]] for node in objects]
    if all_points:
        bounds = np.vstack(all_points)
        x0, y0 = np.min(bounds[:, :2], axis=0)
        x1, y1 = np.max(bounds[:, :2], axis=0)
        margin = max(0.08, 0.15 * max(x1 - x0, y1 - y0))
        x0, x1, y0, y1 = x0 - margin, x1 + margin, y0 - margin, y1 + margin
    elif table is not None:
        x0, y0 = table["world_aabb_m"]["min"][:2]
        x1, y1 = table["world_aabb_m"]["max"][:2]
    if table is not None:
        aabb = table["world_aabb_m"]
        z = aabb["max"][2]
        xx, yy = np.meshgrid([max(x0, aabb["min"][0]), min(x1, aabb["max"][0])],
                             [max(y0, aabb["min"][1]), min(y1, aabb["max"][1])])
        ax.plot_surface(xx, yy, np.full_like(xx, z), alpha=0.23, color="#b37c45")
    for node in objects:
        points = clouds[node["pointcloud_key"]]
        stride = max(1, len(points) // 1800)
        rgb = clouds[f"{node['id']}_rgb8"][::stride].astype(float) / 255.0
        points = points[::stride]
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=2,
                   c=rgb, alpha=0.8, depthshade=False)
        center = node["centroid_world_m"]
        ax.text(center[0], center[1], float(points[:, 2].max()) + 0.012,
                node["id"], fontsize=8,
                bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 1})
    if all_points:
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        z_min = table["world_aabb_m"]["max"][2] if table else float(bounds[:, 2].min())
        z_max = float(bounds[:, 2].max()) + 0.035
        ax.set_zlim(z_min - 0.005, z_max)
        ax.set_box_aspect((x1 - x0, y1 - y0, z_max - z_min + 0.005))
    ax.set(xlabel="world X (m)", ylabel="world Y (m)", zlabel="world Z (m)")
    ax.set_title("Observed 3D points and table plane")
    ax.view_init(elev=26, azim=-55)
    relations_ax.set(xlim=(0, 1), ylim=(0, 1), title="Instance relations")
    relations_ax.axis("off")
    ordered = sorted(objects, key=lambda node: node["centroid_world_m"][0])
    object_x = [0.5] if len(ordered) == 1 else np.linspace(0.12, 0.88, len(ordered))
    positions = {node["id"]: (x, 0.72) for node, x in zip(ordered, object_x)}
    if table is not None:
        positions["table"] = (0.5, 0.18)
    for edge in graph["edges"]:
        start, end = positions[edge["source"]], positions[edge["target"]]
        lateral = edge["relation"] in ("next_to", "inside")
        relations_ax.add_patch(FancyArrowPatch(start, end,
            arrowstyle="<->" if edge.get("symmetric") else "-|>",
            mutation_scale=13, linewidth=1.6, color="#465569",
            connectionstyle="arc3,rad=-0.5" if lateral else "arc3,rad=0.08",
            shrinkA=33, shrinkB=33))
        midpoint = ((start[0] + end[0]) / 2,
                    0.89 if lateral else (start[1] + end[1]) / 2)
        relations_ax.text(*midpoint, edge["relation"], ha="center", va="center",
                          fontsize=9, bbox={"facecolor": "white", "edgecolor": "none", "pad": 2})
    for node in ordered:
        x, y = positions[node["id"]]
        relations_ax.add_patch(FancyBboxPatch((x - 0.09, y - 0.09), 0.18, 0.18,
            boxstyle="round,pad=0.015", facecolor="#e9f0fa", edgecolor="#4b6c93"))
        uncertain = "?" if node.get("semantic_status") == "ambiguous_within_prompts" else ""
        display_label = "\n".join(textwrap.wrap(node["label"] + uncertain, width=13))
        relations_ax.text(x, y, f"{node['id']}\n{display_label}",
                          ha="center", va="center", fontsize=7.5)
    if table is not None:
        relations_ax.add_patch(FancyBboxPatch((0.38, 0.13), 0.24, 0.10,
            boxstyle="round,pad=0.015", facecolor="#f1e2d2", edgecolor="#a56d39"))
        relations_ax.text(0.5, 0.18, "table", ha="center", va="center", fontsize=9)
    if not graph["edges"]:
        relations_ax.text(0.5, 0.42, "No relation passed the geometric thresholds",
                          ha="center", fontsize=9, color="#666666")
    relations_ax.text(0.5, 0.03, "Left to right follows world X · ? marks an ambiguous prompt",
                      ha="center", fontsize=8, color="#666666")
    fig.suptitle(f"Wrist RGB-D scene graph · frame {graph['frame']}", fontsize=13)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def report_paths(source: Path, limit: int):
    if source.is_file():
        yield source
    elif source.is_dir():
        paths = sorted(source.glob("*_detections.json"))
        if not paths:
            raise FileNotFoundError(f"No detection reports in {source}")
        yield from paths if limit == 0 else paths[:limit]
    else:
        raise FileNotFoundError(source)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detections", type=Path, default=DEFAULT_REPORT,
                        help="one detection JSON or a directory of reports")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--frames", type=int, default=1, help="0 processes every report in a directory")
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda")
    parser.add_argument("--min-depth", type=float, default=0.15)
    parser.add_argument("--max-depth", type=float, default=2.0)
    parser.add_argument("--max-pixels", type=int, default=30000)
    parser.add_argument("--voxel-size", type=float, default=0.008)
    parser.add_argument("--cluster-radius", type=float, default=0.025)
    parser.add_argument("--min-points", type=int, default=80)
    parser.add_argument("--support-tolerance", type=float, default=0.025)
    parser.add_argument("--contact-band", type=float, default=0.03,
                        help="height above table plane for a visible contact patch")
    parser.add_argument("--min-contact-points", type=int, default=20)
    parser.add_argument("--neighbor-distance", type=float, default=0.08)
    parser.add_argument("--clip-margin", type=float, default=0.04,
                        help="minimum top-vs-second text cosine gap for a distinct prompt")
    parser.add_argument("--allow-border-masks", action="store_true")
    parser.add_argument("--interiors", type=Path,
                        help="JSON mapping object IDs to calibrated world-frame interior AABBs")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    if (args.frames < 0 or not 0 < args.min_depth < args.max_depth or
            args.max_pixels < 1 or args.voxel_size <= 0 or args.cluster_radius <= 0 or
            args.min_points < 1 or args.support_tolerance < 0 or args.contact_band <= 0 or
            args.min_contact_points < 1 or args.neighbor_distance < 0 or
            not 0 <= args.clip_margin <= 2):
        parser.error("Invalid frame, depth, cluster, or relation limits")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    encoder = ClipEncoder(args.device)
    for path in report_paths(args.detections, args.frames):
        graph, clouds = make_graph(path, encoder, args)
        stem = report_stem(path)
        graph["pointcloud_file"] = f"{stem}_points.npz"
        np.savez_compressed(args.output_dir / graph["pointcloud_file"], **clouds)
        (args.output_dir / f"{stem}_scene_graph.json").write_text(
            json.dumps(graph, indent=2) + "\n", encoding="utf-8")
        if not args.no_plot:
            plot_graph(graph, clouds, args.output_dir / f"{stem}_scene_graph.png")
        print(f"{stem}: {sum(node['type'] == 'object' for node in graph['nodes'])} objects, "
              f"{len(graph['edges'])} relations, {len(graph['discarded_detections'])} rejected masks "
              f"-> {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
