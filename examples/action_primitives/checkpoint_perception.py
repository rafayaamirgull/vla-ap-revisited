"""Run the existing wrist perception pipeline on action checkpoints.

Every result here is a *new observation*. The instance IDs assigned by the
scene-graph pipeline are frame-local and are not a tracking result.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def _checkpoint_names(output_dir: Path) -> dict[str, str]:
    manifest = output_dir / "checkpoints" / "manifest.json"
    if not manifest.is_file():
        return {}
    entries = json.loads(manifest.read_text(encoding="utf-8")).get("checkpoints", [])
    names = {}
    for item in entries:
        stem = item.get("capture_stem")
        if stem is None:
            stem = f"frame_{int(item['frame']):04d}"
        names[stem] = item["name"]
    return names


def _fit_image(image: np.ndarray | None, width: int, height: int) -> np.ndarray:
    canvas = np.full((height, width, 3), (38, 42, 49), dtype=np.uint8)
    if image is None:
        cv2.putText(canvas, "Image unavailable", (28, height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (225, 225, 225), 2, cv2.LINE_AA)
        return canvas
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(image, (max(1, round(image.shape[1] * scale)),
                                 max(1, round(image.shape[0] * scale))),
                         interpolation=cv2.INTER_AREA)
    top = (height - resized.shape[0]) // 2
    left = (width - resized.shape[1]) // 2
    canvas[top:top + resized.shape[0], left:left + resized.shape[1]] = resized
    return canvas


def _comparison_image(frames: list[dict], output_dir: Path) -> Path:
    rows = []
    for frame in frames:
        row = np.full((440, 1280, 3), (25, 28, 33), dtype=np.uint8)
        status = (f"{frame['checkpoint']}  |  observed objects: "
                  f"{len(frame['observed_objects'])}  |  relations: {len(frame['edges'])}")
        cv2.putText(row, status[:95], (14, 26), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (245, 245, 245), 2, cv2.LINE_AA)
        if frame["observed_objects"]:
            detail = "Observed wrist graph; node IDs are local to this frame"
        elif frame["filtered_candidates"]:
            candidate = frame["filtered_candidates"][0]
            detail = ("UNOBSERVED in wrist graph; "
                      f"{candidate.get('label', 'candidate')} filtered: "
                      f"{candidate.get('reason', 'unknown reason')}")
        else:
            detail = "UNOBSERVED in wrist graph; no accepted object detection"
        cv2.putText(row, detail[:110], (14, 48), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, (190, 215, 240), 1, cv2.LINE_AA)
        overlay = cv2.imread(str(output_dir / frame["overlay_file"]), cv2.IMREAD_COLOR)
        graph = cv2.imread(str(output_dir / frame["graph_image_file"]), cv2.IMREAD_COLOR)
        row[54:440, :640] = _fit_image(overlay, 640, 386)
        row[54:440, 640:] = _fit_image(graph, 640, 386)
        rows.append(row)
    title = np.full((48, 1280, 3), (14, 17, 22), dtype=np.uint8)
    cv2.putText(title, "OBSERVED WRIST GRAPH  |  frame-local IDs, no tracking",
                (14, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.82,
                (245, 245, 245), 2, cv2.LINE_AA)
    image = np.vstack([title, *rows])
    destination = output_dir / "checkpoints" / "perceived_checkpoints.png"
    if not cv2.imwrite(str(destination), image):
        raise OSError(f"Could not write {destination}")
    return destination


def process_checkpoints(capture_dir: Path, output_dir: Path, prompts: str,
                        confidence: float) -> dict:
    """Build independent observed graphs for saved RGB-D action checkpoints.

    Returns paths to a compact JSON comparison and side-by-side image. Empty
    object lists mean no object was accepted from that wrist view; they do not
    prove an object is absent from the world.
    """
    capture_dir = Path(capture_dir)
    output_dir = Path(output_dir)
    if not prompts.strip() or not 0 <= confidence <= 1:
        raise ValueError("Prompts must be nonempty and confidence must be in [0, 1]")
    names = _checkpoint_names(output_dir)
    images = ([capture_dir / f"{stem}_rgb.png" for stem in sorted(names)] if names else
              sorted(capture_dir.glob("frame_[0-9][0-9][0-9][0-9]_rgb.png")))
    if not images:
        raise ValueError(f"No RGB-D action checkpoints found in {capture_dir}")
    if any(not image.is_file() for image in images):
        raise FileNotFoundError("Checkpoint manifest refers to a missing RGB image")
    stems = [path.name[:-len("_rgb.png")] for path in images]
    if stems != [f"frame_{index:04d}" for index in range(len(stems))]:
        raise ValueError("Action checkpoints must use consecutive frame numbers from zero")
    for stem in stems:
        if not (capture_dir / f"{stem}_depth.npy").is_file() or not (
                capture_dir / f"{stem}_camera.json").is_file():
            raise FileNotFoundError(f"Checkpoint {stem} lacks depth or camera metadata")

    checkpoint_root = output_dir / "checkpoints"
    perception_dir = checkpoint_root / "perception"
    graph_dir = checkpoint_root / "perceived_graph"
    perception_dir.mkdir(parents=True, exist_ok=True)
    graph_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        sys.executable, str(ROOT / "examples" / "open_vocab_pipeline" / "run.py"),
        "--input", str(capture_dir), "--frames", str(len(stems)), "--prompts", prompts,
        "--confidence", str(confidence), "--output-dir", str(perception_dir),
    ], check=True)
    subprocess.run([
        sys.executable, str(ROOT / "examples" / "scene_graph" / "run.py"),
        "--detections", str(perception_dir), "--frames", str(len(stems)),
        "--output-dir", str(graph_dir),
    ], check=True)

    frames = []
    for stem in stems:
        report_path = perception_dir / f"{stem}_detections.json"
        graph_path = graph_dir / f"{stem}_scene_graph.json"
        if not report_path.is_file() or not graph_path.is_file():
            raise FileNotFoundError(f"Perception produced no report or graph for {stem}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        graph = json.loads(graph_path.read_text(encoding="utf-8"))
        nodes = [
            {"frame_local_id": node["id"], "label": node["label"],
             "centroid_world_m": node["centroid_world_m"],
             "world_aabb_m": node["world_aabb_m"],
             "touches_image_border": node.get("visibility", {}).get(
                 "touches_image_border", False)}
            for node in graph["nodes"] if node["type"] == "object"
        ]
        edges = [{"source_frame_local_id": edge["source"],
                  "target_frame_local_id": edge["target"],
                  "relation": edge["relation"]} for edge in graph["edges"]]
        rejected = [
            {key: candidate.get(key) for key in (
                "label", "reason", "touches_image_border", "valid_depth_fraction")}
            for candidate in report.get("filtered_candidates", [])
        ]
        frames.append({
            "checkpoint": names.get(stem, stem), "capture_stem": stem,
            "simulation_time_s": graph.get("simulation_time_s"),
            "visibility_status": ("observed_objects" if nodes else
                                  "no_accepted_object_nodes"),
            "observed_objects": nodes, "edges": edges,
            "filtered_candidates": rejected,
            "discarded_3d_masks": graph.get("discarded_detections", []),
            "interpretation": ("Observed nodes are independent detections in this frame."
                               if nodes else "No object was accepted from this wrist view; "
                               "object absence is not established."),
            "report_file": str(report_path.relative_to(output_dir)),
            "graph_file": str(graph_path.relative_to(output_dir)),
            "overlay_file": str((perception_dir / f"{stem}_overlay.png").relative_to(
                output_dir)),
            "graph_image_file": str((graph_dir / f"{stem}_scene_graph.png").relative_to(
                output_dir)),
        })

    comparison = {
        "source": "wrist_rgbd_yolo_world_sam2_clip_scene_graph",
        "identity": "Instance IDs are assigned independently in each checkpoint; no tracking.",
        "absence_policy": "A missing node means unobserved from this camera, not absent.",
        "frames": frames,
    }
    summary_path = checkpoint_root / "perceived_change.json"
    summary_path.write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    image_path = _comparison_image(frames, output_dir)
    return {"summary_file": str(summary_path),
            "comparison_image_file": str(image_path),
            "frames": len(frames),
            "observed_objects_per_frame": [len(frame["observed_objects"]) for frame in frames]}
