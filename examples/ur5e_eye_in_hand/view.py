"""Open the interactive MuJoCo viewer for the UR5e tabletop scene."""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import mujoco
import mujoco.viewer

from run import CAMERA_NAME, GRIPPER_ACTUATOR_NAME, HOME, build_scene


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--camera", choices=("overview", "wrist", "realsense"), default="overview",
        help="initial viewer camera",
    )
    parser.add_argument(
        "--sweep-radians", type=float, default=0.0,
        help="gently sweep the shoulder-pan joint to show the moving wrist view",
    )
    parser.add_argument(
        "--seconds", type=float, default=0.0,
        help="close automatically after this many seconds; 0 waits for the viewer to close",
    )
    parser.add_argument("--gripper-command", type=float, default=0.0,
                        help="2F-85 command from 0 (open) to 255 (closed)")
    parser.add_argument("--gripper-demo", action="store_true",
                        help="repeatedly open and close the gripper")
    parser.add_argument("--hb-root", type=Path, help="HomebrewedDB root; auto-detected at /datasets/homebrewed")
    parser.add_argument("--hb-count", type=int, choices=(2, 3), help="otherwise randomly choose 2 or 3 objects")
    parser.add_argument("--seed", type=int, help="repeatable object choice and placement")
    parser.add_argument("--include-large", action="store_true", help="allow all 33 models, including large ones")
    args = parser.parse_args()
    if args.seconds < 0 or not 0 <= args.sweep_radians <= 0.25:
        parser.error("--seconds must be >= 0 and --sweep-radians must be in [0, 0.25]")
    if not 0.0 <= args.gripper_command <= 255.0:
        parser.error("--gripper-command must be in [0, 255]")
    if args.gripper_demo and args.gripper_command:
        parser.error("use either --gripper-demo or --gripper-command")

    scene = build_scene(args.hb_root, args.seed, args.hb_count, args.include_large)
    model = scene.model
    if scene.seed is not None:
        selected = [item["object_id"] for item in scene.objects if item["object_id"] is not None]
        print(f"HomebrewedDB seed={scene.seed}, objects={selected}", flush=True)
    else:
        print("HomebrewedDB not mounted; showing the cube only.", flush=True)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    data.qpos[:6] = HOME
    data.ctrl[:6] = HOME
    gripper_actuator_id = model.actuator(GRIPPER_ACTUATOR_NAME).id
    data.ctrl[gripper_actuator_id] = args.gripper_command
    mujoco.mj_forward(model, data)
    for _ in range(round(0.5 / model.opt.timestep)):
        mujoco.mj_step(model, data)
    settled_time = data.time

    with mujoco.viewer.launch_passive(model, data) as viewer:
        with viewer.lock():
            if args.camera == "wrist":
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                viewer.cam.fixedcamid = model.camera(CAMERA_NAME).id
            elif args.camera == "realsense":
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                viewer.cam.lookat[:] = (
                    data.xpos[model.body("realsense/d435i").id]
                    + data.xpos[model.body("gripper/base").id]
                ) / 2.0
                viewer.cam.distance = 0.56
                viewer.cam.azimuth = 210.0
                viewer.cam.elevation = -20.0
            else:
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                viewer.cam.lookat[:] = [0.0, 0.34, 0.32]
                viewer.cam.distance = 1.75
                viewer.cam.azimuth = 150.0
                viewer.cam.elevation = -25.0
        viewer.sync()
        print(f"MuJoCo viewer open ({args.camera}). Close the window or press Ctrl+C to stop.", flush=True)
        started = time.monotonic()
        try:
            while viewer.is_running() and (args.seconds == 0 or time.monotonic() - started < args.seconds):
                step_started = time.monotonic()
                if args.sweep_radians:
                    data.ctrl[0] = HOME[0] + args.sweep_radians * math.sin(
                        2.0 * math.pi * (data.time - settled_time) / 4.0
                    )
                if args.gripper_demo:
                    data.ctrl[gripper_actuator_id] = (
                        255.0 if (data.time - settled_time) % 4.0 < 2.0 else 0.0
                    )
                mujoco.mj_step(model, data)
                viewer.sync()
                time.sleep(max(0.0, model.opt.timestep - (time.monotonic() - step_started)))
        except KeyboardInterrupt:
            pass
    # MuJoCo's passive viewer runs on a daemon thread. Give its GL context
    # time to close before Python tears down the process.
    time.sleep(0.5)


if __name__ == "__main__":
    main()
