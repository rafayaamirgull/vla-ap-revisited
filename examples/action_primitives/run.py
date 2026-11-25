"""Run contact-checked UR5e IK primitives in the tabletop MuJoCo scene."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np

from primitives import ActionError, ActionPrimitives, HandleSpec, TargetRegion


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples" / "ur5e_eye_in_hand"))
scene_path = ROOT / "examples" / "ur5e_eye_in_hand" / "run.py"
scene_spec = importlib.util.spec_from_file_location("_ur5e_scene_builder", scene_path)
if scene_spec is None or scene_spec.loader is None:
    raise ImportError(f"Cannot load scene builder: {scene_path}")
scene_module = importlib.util.module_from_spec(scene_spec)
sys.modules[scene_spec.name] = scene_module
scene_spec.loader.exec_module(scene_module)
HOME = scene_module.HOME
SceneBuild = scene_module.SceneBuild
build_scene = scene_module.build_scene


def initialize(scene: SceneBuild, stop_time: float = 0.5,
               gripper_command: float = 0.0) -> mujoco.MjData:
    model = scene.model
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    data.qpos[:6] = HOME
    data.ctrl[:6] = HOME
    data.ctrl[model.actuator("gripper/fingers_actuator").id] = gripper_command
    mujoco.mj_forward(model, data)
    if stop_time < 0.5 or stop_time > 10:
        raise ValueError("Capture time must be between 0.5 and 10 seconds")
    for _ in range(round(stop_time / model.opt.timestep)):
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)
    return data


def fixture_graph(scene: SceneBuild, data: mujoco.MjData) -> dict:
    """A simulator-labeled cube fixture for testing control without perception."""
    model = scene.model
    table_id = model.geom("table").id
    table_center = data.geom_xpos[table_id]
    table_extent = np.abs(data.geom_xmat[table_id].reshape(3, 3)) @ model.geom_size[table_id, :3]
    cube_id = model.geom("red_cube_geom").id
    cube_center = data.geom_xpos[cube_id]
    cube_extent = np.abs(data.geom_xmat[cube_id].reshape(3, 3)) @ model.geom_size[cube_id, :3]

    def bounds(center: np.ndarray, extent: np.ndarray) -> dict:
        return {"min": (center - extent).tolist(), "max": (center + extent).tolist()}

    return {"schema_version": 2, "coordinate_frame": "MuJoCo world, meters",
            "geometry_source": "simulator_fixture_oracle",
            "nodes": [
                {"id": "table", "type": "support_surface", "world_aabb_m":
                 bounds(table_center, table_extent)},
                {"id": "object_00", "type": "object", "label": "red cube",
                 "geometry_source": "simulator_fixture_oracle",
                 "world_aabb_m": bounds(cube_center, cube_extent)},
            ]}


def replay_graph(graph_path: Path) -> tuple[SceneBuild, mujoco.MjData, dict]:
    """Recreate a frame-0 scene; reject a graph that does not match its capture."""
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    if graph.get("coordinate_frame") != "MuJoCo world, meters" or graph.get("frame") != 0:
        raise ValueError("Only world-frame graphs from capture frame 0 can be replayed")
    provenance = graph.get("provenance", {})
    report_path = Path(provenance.get("detections_report", ""))
    if not report_path.is_file():
        raise FileNotFoundError(f"Detection report from graph provenance is unavailable: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    metadata = report.get("camera_metadata", {})
    hashes = metadata.get("capture_fingerprints", {})
    if (hashes.get("rgb_sha256") != provenance.get("rgb_sha256") or
            hashes.get("depth_sha256") != provenance.get("depth_sha256")):
        raise ValueError("Graph and detection report refer to different RGB-D captures")
    expected = metadata.get("objects", [])
    if not expected or (metadata.get("scene_seed") is None and len(expected) > 1):
        raise ValueError("Capture lacks the seed and object list needed for scene replay")
    hb_count = len(expected) - 1
    with_drawer = bool(metadata.get("with_drawer", False))
    if hb_count == 0:
        options = [(None, False, True)]
    else:
        if scene_module.resolve_hb_root(None) is None:
            raise FileNotFoundError("Captured scene needs HomebrewedDB mounted at "
                                    "/datasets/homebrewed")
        generation = metadata.get("scene_generation")
        if isinstance(generation, dict) and "include_large" in generation:
            options = [(generation.get("requested_hb_count"),
                        bool(generation["include_large"]), False)]
        else:
            # Older captures did not record whether the count was random.
            # The random-count draw consumes one RNG value before placement.
            options = [(None, False, False), (hb_count, False, False),
                       (None, True, False), (hb_count, True, False)]
    first_error = None
    for requested_count, include_large, cube_only in options:
        try:
            scene = build_scene(seed=int(metadata["scene_seed"])
                                if not cube_only else None,
                                hb_count=requested_count,
                                include_large=include_large,
                                with_drawer=with_drawer, cube_only=cube_only)
        except (FileNotFoundError, ValueError) as exc:
            if first_error is None:
                first_error = exc
            continue
        if [(item["name"], item["object_id"]) for item in scene.objects] != [
            (item["name"], item["object_id"]) for item in expected
        ]:
            continue
        data = initialize(scene, float(metadata["simulation_time_s"]),
                          float(metadata.get("gripper_command_0_to_255", 0)))
        errors = [np.linalg.norm(data.xpos[scene.model.body(item["name"]).id] -
                                 item["world_position_m"]) for item in expected]
        orientation_matches = True
        for item in expected:
            saved_quat = item.get("world_quaternion_wxyz")
            if saved_quat is None:  # Older capture metadata may lack orientation.
                continue
            saved_quat = np.asarray(saved_quat, dtype=float)
            if (saved_quat.shape != (4,) or not np.isfinite(saved_quat).all() or
                    np.linalg.norm(saved_quat) < 1e-8):
                raise ValueError("Capture has invalid object orientation metadata")
            current_quat = data.xquat[scene.model.body(item["name"]).id]
            cosine = np.clip(abs(np.dot(saved_quat, current_quat)) /
                             np.linalg.norm(saved_quat), 0.0, 1.0)
            if 2 * np.arccos(cosine) > np.deg2rad(5):
                orientation_matches = False
                break
        if errors and max(errors) < 0.015 and orientation_matches:
            return scene, data, graph
    raise ValueError("Rebuilt scene differs from the saved capture; recapture and rebuild "
                     "the graph before commanding a graph node") from first_error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("grasp", "grasp-place", "push",
                                             "pull", "pull-push"),
                        default="grasp-place")
    parser.add_argument("--scene-graph", type=Path,
                        help="replay a fresh frame-0 graph; omit for the cube control fixture")
    parser.add_argument("--node-id", help="scene-graph object ID; fixture cube is object_00")
    parser.add_argument("--handle-id", default="drawer_handle")
    parser.add_argument("--target-xy", nargs=2, type=float,
                        metavar=("X", "Y"),
                        help="absolute place/push target center in world meters")
    parser.add_argument("--region-half-size", nargs=2, type=float,
                        metavar=("HALF_X", "HALF_Y"))
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "outputs" / "action_primitives")
    parser.add_argument("--visualize", action="store_true",
                        help="show overview, wrist RGB and changing scene graph live; save MP4")
    parser.add_argument("--record-video", action="store_true",
                        help="save the same MP4 without opening a window")
    parser.add_argument("--native-viewer", action="store_true",
                        help="also open the mouse-controlled MuJoCo 3D viewer")
    parser.add_argument("--video-fps", type=float, default=10.0)
    parser.add_argument("--playback-speed", type=float, default=1.0,
                        help="live playback speed; 1.0 follows simulation time")
    parser.add_argument("--perception-checkpoints", action="store_true",
                        help="run YOLO-World/SAM2 and scene graph on action checkpoints")
    parser.add_argument("--prompts", default="red cube, robot gripper, table",
                        help="YOLO-World prompts for --perception-checkpoints")
    parser.add_argument("--confidence", type=float, default=0.003,
                        help="YOLO-World score threshold for checkpoints")
    args = parser.parse_args()
    drawer_action = args.action in ("pull", "pull-push")
    if drawer_action and args.scene_graph is not None:
        parser.error("The current scene graphs have no observed articulated handle; "
                     "run the drawer action with the explicit fixture")
    if args.scene_graph is not None and args.node_id is None:
        parser.error("--node-id is required with --scene-graph; IDs are frame-local")
    if args.native_viewer and not args.visualize:
        parser.error("--native-viewer requires --visualize")
    if args.video_fps <= 0 or args.playback_speed <= 0:
        parser.error("--video-fps and --playback-speed must be positive")
    if not 0 <= args.confidence <= 1:
        parser.error("--confidence must be in [0, 1]")
    if args.action == "push" and args.target_xy is None:
        parser.error("--action push requires --target-xy X Y")

    try:
        if args.scene_graph is None:
            scene = build_scene(with_drawer=drawer_action, cube_only=True)
            data = initialize(scene)
            graph = fixture_graph(scene, data)
            graph_source = "simulator_fixture_oracle"
        else:
            scene, data, graph = replay_graph(args.scene_graph)
            graph_source = str(args.scene_graph)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    model = scene.model
    trace: list[dict] = []
    last_sample = -1.0
    site_id = model.site("gripper/pinch").id
    cube_id = model.body("red_cube").id
    drawer_joint = model.joint("drawer_slide") if drawer_action else None
    visualizer = None

    def sample(state: mujoco.MjData) -> None:
        nonlocal last_sample
        if state.time - last_sample < 0.05:
            return
        last_sample = float(state.time)
        trace.append({"time_s": float(state.time),
                      "pinch_xyz_m": state.site_xpos[site_id].tolist(),
                      "cube_xyz_m": state.xpos[cube_id].tolist(),
                      "drawer_slide_m": (float(state.qpos[model.jnt_qposadr[drawer_joint.id]])
                                         if drawer_joint is not None else "")})

    def on_step(state: mujoco.MjData) -> None:
        sample(state)
        if visualizer is not None:
            visualizer.on_step(state)

    handles = ({"drawer_handle": HandleSpec("drawer_handle", "drawer_handle_geom",
                                             "drawer_slide",
                                             (0, -1, 0), 0.08)}
               if drawer_action else {})
    controller = ActionPrimitives(model, data, graph, scene.objects, handles, on_step)
    node_id = args.node_id or "object_00"
    region = None
    try:
        if args.action in ("grasp", "grasp-place"):
            controller.validate_grasp(node_id)
        if args.action in ("grasp-place", "push"):
            lower, upper = controller._node_bounds(node_id)
            footprint = upper[:2] - lower[:2]
            target_xy = args.target_xy or (-0.16, 0.62)
            half_size = (tuple(args.region_half_size) if args.region_half_size is not None
                         else tuple(float(value) for value in
                                    np.ceil(np.maximum(0.06, footprint / 2 + 0.035) *
                                            100 - 1e-9) / 100))
            region = TargetRegion(tuple(target_xy), half_size,
                                  controller.table_top)
            region.validate(controller.table_bounds)
            if np.any(footprint > 2 * np.asarray(region.half_size_xy_m)):
                required = tuple(np.ceil((footprint / 2 + 0.01) * 100) / 100)
                raise ValueError(
                    f"{node_id} observed footprint is "
                    f"{footprint[0] * 100:.1f} x {footprint[1] * 100:.1f} cm; "
                    f"increase --region-half-size to at least "
                    f"{required[0]:.2f} {required[1]:.2f} m")
            # Simulation-only clearance check; the graph may miss an occluded
            # object, so avoid commanding a place/push into a known body.
            from live_graph import LiveSceneGraph
            simulator_graph = LiveSceneGraph(model, data, scene.objects, graph).snapshot(data)
            region_min = np.asarray(region.center_xy_m) - region.half_size_xy_m
            region_max = np.asarray(region.center_xy_m) + region.half_size_xy_m
            for other in simulator_graph["nodes"]:
                if other["type"] != "object" or other["id"] == node_id:
                    continue
                bounds = other["world_aabb_m"]
                other_min = np.asarray(bounds["min"][:2])
                other_max = np.asarray(bounds["max"][:2])
                if np.all(region_min < other_max + 0.015) and np.all(
                        region_max > other_min - 0.015):
                    raise ValueError(
                        f"Target region overlaps {other['simulator_body']} in the "
                        "replayed scene; choose a clear --target-xy")
            if args.action == "push":
                controller.plan_push(node_id, region)
    except (ActionError, ValueError) as exc:
        parser.error(str(exc))

    from visualizer import ActionVisualizer, reserve_output_dir
    requested_output_dir = args.output_dir
    try:
        args.output_dir = reserve_output_dir(requested_output_dir)
    except OSError as exc:
        parser.error(f"Cannot create output directory: {exc}")
    if args.output_dir != requested_output_dir:
        print(f"Output directory already contains a run; writing to "
              f"{args.output_dir}", file=sys.stderr, flush=True)

    if args.visualize or args.record_video or args.perception_checkpoints:
        visualizer = ActionVisualizer(
            model, data, scene, graph, args.output_dir, scene_module=scene_module,
            show=args.visualize, record=args.visualize or args.record_video,
            fps=args.video_fps, playback_speed=args.playback_speed,
            native_viewer=args.native_viewer,
            focus_id=args.handle_id if drawer_action else node_id)
    results = []
    error = None
    visualization_info = None
    try:
        if visualizer is not None:
            visualizer.checkpoint("before_action")
        if drawer_action:
            if visualizer is not None:
                visualizer.set_phase("grasp handle + pull")
            results.append(controller.pull(args.handle_id).as_dict())
            if visualizer is not None:
                visualizer.checkpoint("after_pull")
            if args.action == "pull-push":
                if visualizer is not None:
                    visualizer.set_phase("push drawer closed")
                results.append(controller.push_drawer(args.handle_id).as_dict())
                if visualizer is not None:
                    visualizer.checkpoint("after_push")
        elif args.action == "push":
            if visualizer is not None:
                visualizer.set_phase("push tabletop object")
            results.append(controller.push(node_id, region).as_dict())
            if visualizer is not None:
                visualizer.checkpoint("after_push")
        else:
            if visualizer is not None:
                visualizer.set_phase("approach + grasp")
            results.append(controller.grasp(node_id).as_dict())
            if visualizer is not None:
                visualizer.checkpoint("after_grasp")
            if args.action == "grasp-place":
                if visualizer is not None:
                    visualizer.set_phase("transfer + place")
                results.append(controller.place(node_id, region).as_dict())
                if visualizer is not None:
                    visualizer.checkpoint("after_place")
    except (ActionError, ValueError) as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        if visualizer is not None:
            if args.visualize:
                visualizer.hold_end(2.0)
            visualization_info = visualizer.close()

    perception_info = None
    perception_error = None
    if (args.perception_checkpoints and visualization_info is not None and
            visualization_info["checkpoint_count"] > 0):
        from checkpoint_perception import process_checkpoints
        try:
            perception_info = process_checkpoints(
                args.output_dir / "checkpoints" / "capture", args.output_dir,
                args.prompts, args.confidence)
        except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
            perception_error = f"{type(exc).__name__}: {exc}"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {"action": args.action, "status": "failed" if error else "succeeded",
              "output_dir": str(args.output_dir),
              "graph_source": graph_source, "node_id": node_id if not drawer_action else None,
              "handle_id": args.handle_id if drawer_action else None,
              "results": results, "error": error,
              "final_simulation_time_s": float(data.time)}
    if visualization_info is not None:
        report["visualization"] = visualization_info
    if args.perception_checkpoints:
        report["perception_checkpoints"] = perception_info
        report["perception_error"] = perception_error
    (args.output_dir / "action_result.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with (args.output_dir / "trajectory.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(("time_s", "pinch_x_m", "pinch_y_m", "pinch_z_m",
                         "cube_x_m", "cube_y_m", "cube_z_m", "drawer_slide_m"))
        for row in trace:
            writer.writerow((row["time_s"], *row["pinch_xyz_m"],
                             *row["cube_xyz_m"], row["drawer_slide_m"]))
    print(json.dumps(report, indent=2), flush=True)
    if error or perception_error:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
