"""Detect prompted objects in wrist RGB frames with YOLO-World and SAM2 Tiny."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FRAME = ROOT / "outputs/ur5e_eye_in_hand/frame_0000_rgb.png"
DEFAULT_OUTPUT = ROOT / "outputs/open_vocab_pipeline"
COLORS = ((255, 120, 70), (80, 200, 80), (70, 160, 255), (200, 120, 220), (80, 220, 220))


def parse_prompts(value: str) -> list[str]:
    prompts = [part.strip() for part in value.split(",")]
    if not prompts or any(not prompt for prompt in prompts):
        raise argparse.ArgumentTypeError("provide comma-separated, nonempty object names")
    return prompts


def weight_path(model_dir: Path, name: str) -> Path:
    path = Path(name)
    return path if path.is_absolute() else model_dir / path


def input_record(path: Path, output_dir: Path, pixels: np.ndarray, encoding: str) -> dict:
    """Hash decoded pixels, independent of PNG/NPY file headers and compression."""
    return {
        "path": Path(os.path.relpath(path.resolve(), output_dir.resolve())).as_posix(),
        "sha256": hashlib.sha256(pixels.tobytes(order="C")).hexdigest(),
        "shape": list(pixels.shape),
        "encoding": encoding,
    }


def read_depth(path: Path, image_shape: tuple[int, int]) -> np.ndarray:
    depth = np.load(path, allow_pickle=False)
    if depth.shape != image_shape or depth.dtype.kind != "f" or depth.dtype.itemsize != 4:
        raise ValueError(f"Expected float32 depth with shape {image_shape}: {path}")
    return np.ascontiguousarray(depth, dtype="<f4")


def check_capture_fingerprints(metadata: dict | None, rgb: np.ndarray,
                               depth: np.ndarray | None) -> None:
    """Catch a camera JSON paired with pixels from another capture."""
    fingerprints = metadata.get("capture_fingerprints") if metadata is not None else None
    if fingerprints is None:
        return  # Older simulator captures have no embedded fingerprints.
    if fingerprints.get("rgb_sha256") != hashlib.sha256(rgb.tobytes()).hexdigest():
        raise ValueError("Camera metadata RGB fingerprint does not match the frame")
    if depth is None or fingerprints.get("depth_sha256") != hashlib.sha256(depth.tobytes()).hexdigest():
        raise ValueError("Camera metadata depth fingerprint does not match the frame")


class OpenVocabPipeline:
    def __init__(self, prompts: list[str], detector_weight: Path, segmenter_weight: Path,
                 device: str, confidence: float, max_detections: int) -> None:
        try:
            import torch
            from ultralytics import SAM, YOLOWorld
        except ImportError as exc:
            raise SystemExit("Install the example dependencies from requirements.txt first") from exc
        self.prompts = prompts
        self.device = ("0" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        self.confidence = confidence
        self.max_detections = max_detections
        self.detector_weight = detector_weight
        self.segmenter_weight = segmenter_weight
        print(f"Loading YOLO-World and SAM2 Tiny on {self.device}...", flush=True)
        self.detector = YOLOWorld(str(detector_weight))
        self.detector.set_classes(prompts)
        self.segmenter = SAM(str(segmenter_weight))

    def infer(self, bgr: np.ndarray) -> tuple[list[dict], float, float]:
        started = time.perf_counter()
        result = self.detector.predict(
            source=bgr, imgsz=640, conf=self.confidence, max_det=self.max_detections,
            device=self.device, verbose=False,
        )[0]
        detect_ms = (time.perf_counter() - started) * 1000
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return [], detect_ms, 0.0

        xyxy = boxes.xyxy.cpu().numpy()
        scores = boxes.conf.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy().astype(int)
        started = time.perf_counter()
        segmented = self.segmenter.predict(
            source=bgr, bboxes=xyxy.tolist(), conf=0.0, device=self.device, verbose=False,
        )[0]
        segment_ms = (time.perf_counter() - started) * 1000
        masks = segmented.masks
        if masks is None or len(masks.data) != len(xyxy):
            raise RuntimeError("SAM2 returned a different mask count than YOLO-World boxes; "
                               "refusing to pair masks with the wrong labels")

        height, width = bgr.shape[:2]
        detections = []
        for index, (box, score, class_id, mask_tensor) in enumerate(
            zip(xyxy, scores, class_ids, masks.data)
        ):
            mask = mask_tensor.cpu().numpy().astype(np.uint8)
            if mask.shape != (height, width):
                mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
            detections.append({
                "label": self.prompts[class_id],
                "class_id": int(class_id),
                "confidence": float(score),
                "box_xyxy": [float(value) for value in box],
                "mask": mask.astype(bool),
                "mask_index": index,
            })
        return detections, detect_ms, segment_ms


def save_frame(bgr: np.ndarray, detections: list[dict], source: str, stem: str,
               camera_metadata: dict | None, pipeline: OpenVocabPipeline,
               detect_ms: float, segment_ms: float, output_dir: Path,
               rgb_path: Path, depth_path: Path | None, depth: np.ndarray | None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    inputs = {
        "rgb": input_record(rgb_path, output_dir, rgb, "rgb8"),
        "depth": input_record(depth_path, output_dir, depth, "float32_le_meters")
        if depth_path is not None and depth is not None else None,
    }
    overlay = bgr.copy()
    records = []
    for detection in detections:
        index = detection["mask_index"]
        color = COLORS[index % len(COLORS)]
        mask = detection["mask"]
        overlay[mask] = (0.55 * overlay[mask] + 0.45 * np.asarray(color)).astype(np.uint8)
        x1, y1, x2, y2 = (round(value) for value in detection["box_xyxy"])
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        caption = f"{detection['label']} {detection['confidence']:.2f}"
        (text_width, text_height), _ = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        text_x = min(max(x1, 0), max(0, bgr.shape[1] - text_width - 4))
        text_y = max(text_height + 4, y1 - 7)
        cv2.rectangle(overlay, (text_x, text_y - text_height - 3),
                      (text_x + text_width + 4, text_y + 3), color, -1)
        cv2.putText(overlay, caption, (text_x + 2, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 1, cv2.LINE_AA)
        mask_name = f"{stem}_mask_{index:02d}.png"
        mask_uint8 = np.ascontiguousarray(mask, dtype=np.uint8)
        if not cv2.imwrite(str(output_dir / mask_name), mask_uint8 * 255):
            raise OSError(f"Could not save {mask_name}")
        records.append({key: value for key, value in detection.items() if key != "mask"} |
                       {"mask_file": mask_name,
                        "mask_sha256": hashlib.sha256(mask_uint8.tobytes(order="C")).hexdigest(),
                        "mask_area_px": int(np.count_nonzero(mask_uint8))})

    overlay_path = output_dir / f"{stem}_overlay.png"
    if not cv2.imwrite(str(overlay_path), overlay):
        raise OSError(f"Could not save {overlay_path}")
    report = {
        "source": source,
        "inputs": inputs,
        "image_size": {"width": bgr.shape[1], "height": bgr.shape[0]},
        "prompts": pipeline.prompts,
        "models": {"detector": str(pipeline.detector_weight),
                   "segmenter": str(pipeline.segmenter_weight)},
        "timing_ms": {"detector": round(detect_ms, 1), "segmenter": round(segment_ms, 1)},
        "camera_metadata": camera_metadata,
        "detections": records,
    }
    (output_dir / f"{stem}_detections.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(f"{stem}: {len(records)} detections, YOLO {detect_ms:.0f} ms, "
          f"SAM2 {segment_ms:.0f} ms -> {overlay_path}", flush=True)


def image_paths(source: Path, limit: int):
    if source.is_file():
        yield source
    elif source.is_dir():
        paths = sorted(source.glob("frame_*_rgb.png"))
        if not paths:
            raise FileNotFoundError(f"No frame_*_rgb.png images in {source}")
        yield from paths if limit == 0 else paths[:limit]
    else:
        raise FileNotFoundError(f"Wrist RGB frame not found: {source}. Run the MuJoCo capture first.")


def run_saved(source: Path, limit: int, pipeline: OpenVocabPipeline, output_dir: Path) -> None:
    for path in image_paths(source, limit):
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"Cannot read image: {path}")
        stem = path.stem.removesuffix("_rgb")
        metadata_path = path.with_name(f"{stem}_camera.json")
        depth_path = path.with_name(f"{stem}_depth.npy")
        depth = read_depth(depth_path, bgr.shape[:2]) if depth_path.is_file() else None
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else None
        check_capture_fingerprints(metadata, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), depth)
        detections, detect_ms, segment_ms = pipeline.infer(bgr)
        save_frame(bgr, detections, str(path), stem, metadata, pipeline,
                   detect_ms, segment_ms, output_dir, path,
                   depth_path if depth is not None else None, depth)


def run_stream(endpoint: str, limit: int, pipeline: OpenVocabPipeline, output_dir: Path) -> None:
    try:
        import zmq
    except ImportError as exc:
        raise SystemExit("ZMQ input needs pyzmq; install requirements.txt") from exc
    with zmq.Context() as context, context.socket(zmq.SUB) as subscriber:
        subscriber.setsockopt(zmq.RCVHWM, 1)
        subscriber.setsockopt(zmq.SUBSCRIBE, b"rgbd")
        subscriber.connect(endpoint)
        print(f"Listening for wrist RGB frames on {endpoint}", flush=True)
        count = 0
        try:
            while limit == 0 or count < limit:
                topic, header, rgb_bytes, depth_bytes = subscriber.recv_multipart()
                if topic != b"rgbd":
                    continue
                metadata = json.loads(header)
                height, width = metadata["height"], metadata["width"]
                if (metadata.get("rgb_encoding") != "rgb8" or
                        metadata.get("depth_encoding") != "float32_le_meters"):
                    raise ValueError("Expected rgb8 and float32_le_meters ZMQ frame encodings")
                rgb = np.frombuffer(rgb_bytes, dtype=np.uint8).reshape(height, width, 3)
                depth = np.frombuffer(depth_bytes, dtype="<f4").reshape(height, width)
                check_capture_fingerprints(metadata, rgb, depth)
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                stem = f"frame_{metadata['frame']:04d}"
                output_dir.mkdir(parents=True, exist_ok=True)
                rgb_path = output_dir / f"{stem}_rgb.png"
                depth_path = output_dir / f"{stem}_depth.npy"
                if not cv2.imwrite(str(rgb_path), bgr):
                    raise OSError(f"Could not save {rgb_path}")
                np.save(depth_path, depth)
                detections, detect_ms, segment_ms = pipeline.infer(bgr)
                save_frame(bgr, detections, endpoint, stem, metadata, pipeline,
                           detect_ms, segment_ms, output_dir, rgb_path, depth_path, depth)
                count += 1
        except KeyboardInterrupt:
            print("Stopped by user", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--input", type=Path, help="one saved wrist PNG or its output directory")
    source.add_argument("--connect", metavar="ENDPOINT", help="subscribe to run.py's RGB-D ZMQ stream")
    parser.add_argument("--prompts", type=parse_prompts,
                        default=parse_prompts("red cube, green bell pepper, toy car"),
                        help="comma-separated object names (default: red cube, green bell pepper, toy car)")
    parser.add_argument("--frames", type=int, default=1, help="number of frames; 0 means all/continuous")
    parser.add_argument("--confidence", type=float, default=0.15, help="YOLO detection threshold")
    parser.add_argument("--max-detections", type=int, default=5)
    parser.add_argument("--device", default="auto", help="auto, cpu, or a CUDA device such as 0")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_OUTPUT / "models")
    parser.add_argument("--detector-model", default="yolov8s-worldv2.pt")
    parser.add_argument("--segmenter-model", default="sam2_t.pt")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.frames < 0 or not 0 <= args.confidence <= 1 or args.max_detections < 1:
        parser.error("--frames must be >= 0, --confidence in [0, 1], and --max-detections >= 1")
    args.model_dir.mkdir(parents=True, exist_ok=True)
    pipeline = OpenVocabPipeline(
        args.prompts, weight_path(args.model_dir, args.detector_model),
        weight_path(args.model_dir, args.segmenter_model), args.device,
        args.confidence, args.max_detections,
    )
    if args.connect:
        run_stream(args.connect, args.frames, pipeline, args.output_dir)
    else:
        run_saved(args.input or DEFAULT_FRAME, args.frames, pipeline, args.output_dir)


if __name__ == "__main__":
    main()
