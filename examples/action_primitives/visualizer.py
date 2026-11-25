"""Live MuJoCo action display, MP4 recording, and aligned RGB-D checkpoints."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import cv2
import mujoco
import numpy as np

from live_graph import LiveSceneGraph, graph_delta, render_panel


VIEW_WIDTH = 640
VIEW_HEIGHT = 480
HEADER_HEIGHT = 44
WINDOW_TITLE = "UR5e action | MuJoCo + wrist RGB + live graph"
RUN_OUTPUTS = ("action_result.json", "trajectory.csv", "action_visualization.mp4",
               "preview_start.png", "preview_end.png", "live_graph_timeline.jsonl",
               "checkpoints")
RESERVED_MARKER = ".action_run_reserved"


def reserve_output_dir(requested: Path) -> Path:
    """Allocate a fresh run directory without overwriting earlier results."""
    requested = Path(requested)
    if requested.exists() and not requested.is_dir():
        raise FileExistsError(f"Action output path is not a directory: {requested}")
    for run_number in range(10000):
        candidate = (requested if run_number == 0 else requested.with_name(
            f"{requested.name}_run_{run_number:03d}"))
        try:
            candidate.mkdir(parents=True, exist_ok=True)
        except FileExistsError:
            continue
        if any((candidate / name).exists() for name in RUN_OUTPUTS):
            continue
        try:
            marker = os.open(candidate / RESERVED_MARKER,
                             os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            continue
        os.close(marker)
        return candidate
    raise FileExistsError(f"No free action run directory beside {requested}")


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


class ActionVisualizer:
    """Render action motion and the oracle graph on the same simulation clock.

    The live graph reads MuJoCo state. Saved RGB-D checkpoints can be passed to
    the existing YOLO-World, SAM2, and scene-graph scripts separately.
    """

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, scene: Any,
                 source_graph: dict, output_dir: Path, scene_module: Any,
                 show: bool = False, record: bool = True, fps: float = 10.0,
                 playback_speed: float = 1.0, native_viewer: bool = False,
                 focus_id: str | None = None) -> None:
        if fps <= 0 or playback_speed <= 0:
            raise ValueError("Video FPS and playback speed must be positive")
        self.model, self.data, self.scene = model, data, scene
        self.source_graph, self.scene_module = source_graph, scene_module
        self.output_dir = Path(output_dir)
        if any((self.output_dir / name).exists() for name in RUN_OUTPUTS):
            raise FileExistsError(
                f"Action outputs already exist in {self.output_dir}; "
                "reserve a fresh run directory before visualizing")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.capture_dir = self.output_dir / "checkpoints" / "capture"
        self.capture_dir.mkdir(parents=True, exist_ok=True)
        self.graph = LiveSceneGraph(model, data, scene.objects, source_graph,
                                    focus_id=focus_id)
        self.renderer = mujoco.Renderer(model, width=VIEW_WIDTH, height=VIEW_HEIGHT)
        self.overview = mujoco.MjvCamera()
        self.overview.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.overview.lookat[:] = [0.0, 0.34, 0.32]
        self.overview.distance = 1.75
        self.overview.azimuth = 150.0
        self.overview.elevation = -25.0
        self.fps = float(fps)
        self.playback_speed = float(playback_speed)
        self.start_sim_time = float(data.time)
        self.start_wall_time = time.monotonic()
        self.next_frame_time = self.start_sim_time
        self.phase = "ready"
        self.show = bool(show)
        self.viewer = None
        self._last_frame_bgr: np.ndarray | None = None
        self._first_frame_saved = False
        self._closed = False
        self.checkpoints: list[dict] = []
        self._checkpoint_graphs: list[dict] = []
        self.simulator_changes: list[dict] = []
        self.timeline = (self.output_dir / "live_graph_timeline.jsonl").open(
            "w", encoding="utf-8")
        self.video_path = self.output_dir / "action_visualization.mp4" if record else None
        self.writer = None
        if record:
            frame_size = (3 * VIEW_WIDTH, VIEW_HEIGHT + HEADER_HEIGHT)
            self.writer = cv2.VideoWriter(
                str(self.video_path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps,
                frame_size)
            if not self.writer.isOpened():
                self.timeline.close()
                self.renderer.close()
                raise OSError(f"Cannot open MP4 writer: {self.video_path}")
        if self.show:
            cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WINDOW_TITLE, 1440, 393)
        if native_viewer:
            from mujoco import viewer as mj_viewer

            self.viewer = mj_viewer.launch_passive(model, data)
            with self.viewer.lock():
                self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                self.viewer.cam.lookat[:] = self.overview.lookat
                self.viewer.cam.distance = self.overview.distance
                self.viewer.cam.azimuth = self.overview.azimuth
                self.viewer.cam.elevation = self.overview.elevation
            self.viewer.sync()

    def set_phase(self, phase: str) -> None:
        self.phase = phase

    def _frame(self, data: mujoco.MjData) -> tuple[np.ndarray, dict]:
        mujoco.mj_forward(self.model, data)
        self.renderer.disable_depth_rendering()
        self.renderer.update_scene(data, camera=self.overview)
        overview = self.renderer.render().copy()
        self.renderer.update_scene(data, camera=self.scene_module.CAMERA_NAME)
        wrist = self.renderer.render().copy()
        graph_snapshot = self.graph.snapshot(data)
        graph_rgb = render_panel(graph_snapshot, VIEW_WIDTH, VIEW_HEIGHT)
        if (overview.shape != (VIEW_HEIGHT, VIEW_WIDTH, 3) or
                wrist.shape != overview.shape or graph_rgb.shape != overview.shape):
            raise ValueError("The overview, wrist, and graph panels must be 640x480 RGB")
        body = np.concatenate((overview, wrist, graph_rgb), axis=1)
        header = np.full((HEADER_HEIGHT, body.shape[1], 3), (28, 35, 45),
                         dtype=np.uint8)
        cv2.putText(header, "MuJoCo overview", (18, 29), cv2.FONT_HERSHEY_SIMPLEX,
                    0.68, (246, 248, 250), 2, cv2.LINE_AA)
        cv2.putText(header, "RealSense wrist RGB", (VIEW_WIDTH + 18, 29),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.68, (246, 248, 250), 2, cv2.LINE_AA)
        cv2.putText(header, f"{self.phase}  |  t={data.time:.2f}s",
                    (2 * VIEW_WIDTH + 18, 29), cv2.FONT_HERSHEY_SIMPLEX,
                    0.62, (246, 248, 250), 2, cv2.LINE_AA)
        return cv2.cvtColor(np.concatenate((header, body), axis=0),
                            cv2.COLOR_RGB2BGR), graph_snapshot

    def _display_and_record(self, frame_bgr: np.ndarray, graph_snapshot: dict) -> None:
        if not self._first_frame_saved:
            cv2.imwrite(str(self.output_dir / "preview_start.png"), frame_bgr)
            self._first_frame_saved = True
        self._last_frame_bgr = frame_bgr
        self.timeline.write(json.dumps(graph_snapshot, default=_json_default) + "\n")
        if self.writer is not None:
            self.writer.write(frame_bgr)
        if self.viewer is not None and self.viewer.is_running():
            self.viewer.sync()
        if self.show:
            try:
                cv2.imshow(WINDOW_TITLE, frame_bgr)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")) or cv2.getWindowProperty(
                        WINDOW_TITLE, cv2.WND_PROP_VISIBLE) < 1:
                    self.show = False
                    cv2.destroyWindow(WINDOW_TITLE)
            except cv2.error:
                self.show = False
        if self.show or self.viewer is not None:
            deadline = (self.start_wall_time +
                        (graph_snapshot["simulation_time_s"] - self.start_sim_time) /
                        self.playback_speed)
            time.sleep(max(0.0, deadline - time.monotonic()))

    def on_step(self, data: mujoco.MjData) -> None:
        """Pass as the primitive controller's on_step callback."""
        if self._closed or data.time + 1e-9 < self.next_frame_time:
            return
        while self.next_frame_time <= data.time + 1e-9:
            self.next_frame_time += 1.0 / self.fps
        frame, graph_snapshot = self._frame(data)
        self._display_and_record(frame, graph_snapshot)

    def checkpoint(self, name: str) -> dict:
        """Save aligned wrist RGB-D and a separate simulator-state graph JSON."""
        if self._closed:
            raise RuntimeError("Visualizer is already closed")
        frame_index = len(self.checkpoints)
        stem = f"frame_{frame_index:04d}"
        mujoco.mj_forward(self.model, self.data)
        rgb, depth = self.scene_module.capture(self.renderer, self.data)
        metadata = self.scene_module.camera_metadata(
            self.model, self.data, frame_index, self.scene)
        metadata["capture_fingerprints"] = {
            "rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
            "depth_sha256": hashlib.sha256(depth.tobytes()).hexdigest(),
        }
        (self.capture_dir / f"{stem}_rgb.png").write_bytes(
            self.scene_module.rgb_png(rgb))
        np.save(self.capture_dir / f"{stem}_depth.npy", depth)
        (self.capture_dir / f"{stem}_camera.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        graph_snapshot = self.graph.snapshot(self.data)
        (self.capture_dir / f"{stem}_sim_graph.json").write_text(
            json.dumps(graph_snapshot, indent=2, default=_json_default) + "\n",
            encoding="utf-8")
        if self._checkpoint_graphs:
            self.simulator_changes.append({
                "from": self.checkpoints[-1]["name"], "to": name,
                "delta": graph_delta(self._checkpoint_graphs[-1], graph_snapshot),
            })
        self._checkpoint_graphs.append(graph_snapshot)
        (self.output_dir / "checkpoints" / "simulator_changes.json").write_text(
            json.dumps({"source": "simulator_state_oracle",
                        "changes": self.simulator_changes}, indent=2) + "\n",
            encoding="utf-8")
        composite, _ = self._frame(self.data)
        cv2.imwrite(str(self.capture_dir / f"{stem}_composite.png"), composite)
        entry = {"frame": frame_index, "name": name,
                 "simulation_time_s": float(self.data.time),
                 "capture_stem": stem,
                 "graph_source": "simulator_state_oracle"}
        self.checkpoints.append(entry)
        (self.output_dir / "checkpoints" / "manifest.json").write_text(
            json.dumps({"checkpoints": self.checkpoints}, indent=2) + "\n",
            encoding="utf-8")
        self.renderer.disable_depth_rendering()
        self._display_and_record(composite, graph_snapshot)
        return entry

    def hold_end(self, seconds: float = 2.0) -> None:
        """Keep the final live frame visible briefly without advancing physics."""
        if seconds <= 0:
            return
        if self.show:
            cv2.waitKey(round(seconds * 1000))
        elif self.viewer is not None and self.viewer.is_running():
            time.sleep(seconds)

    def close(self) -> dict:
        if self._closed:
            return self.summary()
        self._closed = True
        if self._last_frame_bgr is not None:
            cv2.imwrite(str(self.output_dir / "preview_end.png"),
                        self._last_frame_bgr)
        self.timeline.close()
        if self.writer is not None:
            self.writer.release()
        if self.viewer is not None:
            self.viewer.close()
            # The passive viewer owns a GL context on a daemon thread.
            time.sleep(0.5)
        self.renderer.close()
        if self.show:
            cv2.destroyWindow(WINDOW_TITLE)
        return self.summary()

    def summary(self) -> dict:
        return {"video_path": str(self.video_path) if self.video_path else None,
                "checkpoint_manifest": str(self.output_dir / "checkpoints" /
                                           "manifest.json"),
                "simulator_changes": str(self.output_dir / "checkpoints" /
                                         "simulator_changes.json"),
                "checkpoint_count": len(self.checkpoints),
                "preview_start": str(self.output_dir / "preview_start.png"),
                "preview_end": str(self.output_dir / "preview_end.png")}

    def __enter__(self) -> "ActionVisualizer":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
