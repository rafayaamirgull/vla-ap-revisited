"""UR5e tabletop simulation with a 2F-85 gripper and side-mounted D435i.

The scene is assembled with MjSpec so the upstream UR5e model stays untouched.
Tested with MuJoCo 4.0.1 and NumPy. pyzmq is optional.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from homebrewed import available_objects, model_dir, prepare_object
from png_io import rgb_png


WIDTH = 640
HEIGHT = 480
FOVY_DEG = 80.0
CAMERA_NAME = "realsense/rgbd"
GRIPPER_ACTUATOR_NAME = "gripper/fingers_actuator"
GRIPPER_PINCH_SITE_NAME = "gripper/pinch"
HOME = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0])
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MENAGERIE_ROOT = PROJECT_ROOT / "thirdparty" / "mujoco_menagerie"
ROBOT_XML = MENAGERIE_ROOT / "universal_robots_ur5e" / "ur5e.xml"
GRIPPER_XML = MENAGERIE_ROOT / "robotiq_2f85" / "2f85.xml"
REALSENSE_XML = MENAGERIE_ROOT / "realsense_d435i" / "d435i.xml"
HB_CACHE = PROJECT_ROOT / "outputs" / "ur5e_eye_in_hand" / "hb_cache"
DEFAULT_HB_ROOT = Path("/datasets/homebrewed")


@dataclass(frozen=True)
class SceneBuild:
    model: mujoco.MjModel
    objects: list[dict]
    seed: int | None


def resolve_hb_root(requested: Path | None) -> Path | None:
    if requested is not None:
        if not requested.is_dir():
            raise FileNotFoundError(f"HomebrewedDB directory not found: {requested}")
        return requested
    if (DEFAULT_HB_ROOT / "hb_models" / "models" / "models_info.json").is_file():
        return DEFAULT_HB_ROOT
    return None


def _place_objects(spec: mujoco.MjSpec, hb_root: Path, seed: int, count: int | None,
                   include_large: bool) -> list[dict]:
    rng = random.Random(seed)
    candidate_ids = available_objects(hb_root, include_large=include_large)
    count = count if count is not None else rng.choice((2, 3))
    model_info = json.loads((model_dir(hb_root) / "models_info.json").read_text(encoding="utf-8"))
    for _ in range(100):
        selected_ids = rng.sample(candidate_ids, count)
        occupied = [(-0.34, 0.39, 0.05)]  # cube plus a little clearance
        placements = []
        for index, object_id in enumerate(selected_ids):
            info = model_info[str(object_id)]
            sx, sy = info["size_x"] * 0.001, info["size_y"] * 0.001
            radius = math.hypot(sx, sy) / 2.0 + 0.025
            for attempt in range(4000):
                if index == 0 and attempt < 200:
                    x = rng.uniform(-0.20, 0.03)
                    y = rng.uniform(0.40, 0.61)
                else:
                    x = rng.uniform(-0.61 + radius, 0.61 - radius)
                    y = rng.uniform(0.21 + radius, 0.89 - radius)
                if all(math.hypot(x - ox, y - oy) >= radius + other_radius
                       for ox, oy, other_radius in occupied):
                    break
            else:
                break
            occupied.append((x, y, radius))
            placements.append((object_id, x, y, rng.uniform(-math.pi, math.pi)))
        if len(placements) == count:
            break
    else:
        raise RuntimeError("Could not fit the selected HomebrewedDB objects on the table")

    objects = []
    for object_id, x, y, yaw in placements:
        asset = prepare_object(hb_root, object_id, HB_CACHE)
        sx, sy, sz = asset.size_m
        mesh_name = f"hb_mesh_{object_id:06d}"
        texture_name = f"hb_texture_{object_id:06d}"
        material_name = f"hb_material_{object_id:06d}"
        body_name = f"hb_{object_id:06d}"
        spec.add_mesh(name=mesh_name, file=str(asset.mesh_path))
        spec.add_texture(name=texture_name, type=mujoco.mjtTexture.mjTEXTURE_2D,
                         file=str(asset.texture_path))
        material = spec.add_material(name=material_name, rgba=[1.0, 1.0, 1.0, 1.0])
        material.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = texture_name
        body = spec.worldbody.add_body(name=body_name, pos=[x, y, 0.27],
                                       euler=[0.0, 0.0, yaw])
        body.add_freejoint()
        body.add_geom(name=f"{body_name}_visual", type=mujoco.mjtGeom.mjGEOM_MESH,
                      meshname=mesh_name, material=material_name,
                      contype=0, conaffinity=0, mass=0.0, group=2)
        # A box approximates contact while the detailed mesh supplies RGB-D appearance.
        body.add_geom(name=f"{body_name}_collision", type=mujoco.mjtGeom.mjGEOM_BOX,
                      pos=[0.0, 0.0, sz / 2.0], size=[sx / 2.0, sy / 2.0, sz / 2.0],
                      density=400.0, rgba=[0.0, 0.0, 0.0, 0.0], group=3)
        objects.append({"name": body_name, "dataset": "HomebrewedDB",
                        "object_id": object_id, "size_m": asset.size_m})
    return objects


def build_scene(hb_root: Path | None = None, seed: int | None = None,
                hb_count: int | None = None, include_large: bool = False) -> SceneBuild:
    hb_root = resolve_hb_root(hb_root)
    spec = mujoco.MjSpec.from_file(str(ROBOT_XML))
    spec.modelname = "UR5e 2F-85 side-camera tabletop"
    spec.visual.global_.offwidth = WIDTH
    spec.visual.global_.offheight = HEIGHT
    spec.visual.headlight.ambient[:] = [0.35, 0.35, 0.35]
    # Put the robot on a stand level with the table. This gives the wrist
    # enough clearance to view household objects at their original scale.
    spec.body("base").pos[:] = [0.0, 0.0, 0.25]

    # Both attachments use the flange site's frame. The 2F-85's local +Z is
    # its approach direction, so its fingers and the camera look down together.
    gripper = mujoco.MjSpec.from_file(str(GRIPPER_XML))
    spec.option.cone = gripper.option.cone
    spec.option.impratio = gripper.option.impratio
    spec.attach(gripper, prefix="gripper/", site=spec.site("attachment_site"))

    # The D435i mesh has a mounting feature at the center of its local -Y
    # side. Roll the housing 180 degrees around its viewing axis so that side
    # faces the flange, while the lenses still point along tool +Z.
    realsense = mujoco.MjSpec.from_file(str(REALSENSE_XML))
    camera_body = realsense.body("d435i")
    camera_body.pos[:] = [0.0, -0.11, 0.015]
    camera_body.quat[:] = [0.0, 0.0, 0.0, 1.0]
    camera_body.add_camera(
        name="rgbd",
        pos=[0.0325, 0.0, 0.008],
        # Counter-roll the virtual sensor so the landscape feed stays upright.
        # MuJoCo looks along camera -Z; this points through the RGB lens.
        quat=[0.0, 0.0, 1.0, 0.0],
        fovy=FOVY_DEG,
        resolution=[WIDTH, HEIGHT],
    )
    spec.attach(realsense, prefix="realsense/", site=spec.site("attachment_site"))

    # A separate, non-colliding holder reaches the modeled mounting point.
    # The camera is 110 mm to tool -Y; its near side is at -97.5 mm and the
    # mounting feature's center is near tool Z = -3 mm.
    holder_spec = mujoco.MjSpec()
    holder = holder_spec.worldbody.add_body(name="camera_holder")
    holder.add_geom(
        name="base_plate", type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[0.0, -0.02, -0.003], size=[0.024, 0.018, 0.009],
        contype=0, conaffinity=0, mass=0.0, group=2,
        rgba=[0.88, 0.88, 0.84, 1.0],
    )
    holder.add_geom(
        name="support_arm", type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[0.0, -0.064, -0.003], size=[0.011, 0.031, 0.006],
        contype=0, conaffinity=0, mass=0.0, group=2,
        rgba=[0.88, 0.88, 0.84, 1.0],
    )
    holder.add_geom(
        name="bolt_head", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        pos=[0.0, -0.093, -0.003], quat=[math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0],
        size=[0.005, 0.002, 0.0], contype=0, conaffinity=0, mass=0.0, group=2,
        rgba=[0.62, 0.64, 0.66, 1.0],
    )
    holder.add_geom(
        name="mounting_bolt", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        pos=[0.0, -0.097, -0.003], quat=[math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0],
        size=[0.0025, 0.004, 0.0], contype=0, conaffinity=0, mass=0.0, group=2,
        rgba=[0.62, 0.64, 0.66, 1.0],
    )
    spec.attach(holder_spec, prefix="camera_holder/", site=spec.site("attachment_site"))

    world = spec.worldbody
    world.add_light(
        name="table_light", pos=[0.0, 0.3, 1.5], dir=[0.0, 0.0, -1.0],
        castshadow=0, ambient=[0.3, 0.3, 0.3], diffuse=[0.8, 0.8, 0.8],
    )
    world.add_geom(
        name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[0.0, 0.0, 0.05], rgba=[0.35, 0.38, 0.42, 1.0],
    )
    world.add_geom(
        name="robot_stand", type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[0.0, 0.0, 0.125], size=[0.11, 0.11, 0.125],
        rgba=[0.30, 0.33, 0.36, 1.0],
    )
    world.add_geom(
        name="table", type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[0.0, 0.55, 0.20], size=[0.65, 0.38, 0.05],
        rgba=[0.55, 0.42, 0.28, 1.0],
    )

    cube = world.add_body(name="red_cube", pos=[-0.34, 0.39, 0.29])
    cube.add_freejoint()
    cube.add_geom(name="red_cube_geom", type=mujoco.mjtGeom.mjGEOM_BOX,
                  size=[0.025, 0.025, 0.025], rgba=[0.9, 0.08, 0.06, 1.0])
    objects = [{"name": "red_cube", "dataset": "primitive", "object_id": None,
                "size_m": [0.05, 0.05, 0.05]}]
    if hb_root is not None:
        if seed is None:
            seed = random.SystemRandom().randrange(2**32)
        objects.extend(_place_objects(spec, hb_root, seed, hb_count, include_large))

    return SceneBuild(spec.compile(), objects, seed if hb_root is not None else None)


def camera_metadata(model: mujoco.MjModel, data: mujoco.MjData, frame: int,
                    scene: SceneBuild) -> dict:
    def world_pose(position: np.ndarray, rotation: np.ndarray) -> np.ndarray:
        transform = np.eye(4)
        transform[:3, :3] = rotation.reshape(3, 3)
        transform[:3, 3] = position
        return transform

    def inverse_pose(transform: np.ndarray) -> np.ndarray:
        inverse = np.eye(4)
        inverse[:3, :3] = transform[:3, :3].T
        inverse[:3, 3] = -inverse[:3, :3] @ transform[:3, 3]
        return inverse

    camera_id = model.camera(CAMERA_NAME).id
    fy = HEIGHT / (2.0 * math.tan(math.radians(FOVY_DEG) / 2.0))
    world_from_mujoco_camera = world_pose(data.cam_xpos[camera_id], data.cam_xmat[camera_id])
    # OpenCV optical coordinates: +X right, +Y down, +Z forward.
    world_from_optical = world_from_mujoco_camera @ np.diag([1.0, -1.0, -1.0, 1.0])
    base_id = model.body("base").id
    ee_id = model.site("attachment_site").id
    world_from_base = world_pose(data.xpos[base_id], data.xmat[base_id])
    world_from_ee = world_pose(data.site_xpos[ee_id], data.site_xmat[ee_id])
    base_from_ee = inverse_pose(world_from_base) @ world_from_ee
    ee_from_camera_optical = inverse_pose(world_from_ee) @ world_from_optical

    table_id = model.geom("table").id
    table_center = data.geom_xpos[table_id]
    table_rotation = data.geom_xmat[table_id].reshape(3, 3)
    table_extent = np.abs(table_rotation) @ model.geom_size[table_id, :3]
    return {
        "frame": frame,
        "simulation_time_s": float(data.time),
        "camera": CAMERA_NAME,
        "width": WIDTH,
        "height": HEIGHT,
        "rgb_encoding": "rgb8",
        "depth_encoding": "float32_le_meters",
        "intrinsics_opencv": {
            "fx": fy, "fy": fy, "cx": WIDTH / 2.0, "cy": HEIGHT / 2.0,
        },
        "world_from_camera_optical": world_from_optical.tolist(),
        "kinematic_chain": {
            "world_from_base": world_from_base.tolist(),
            "base_from_ee": base_from_ee.tolist(),
            "ee_from_camera_optical": ee_from_camera_optical.tolist(),
        },
        "static_geometry": [{
            "name": "table",
            "world_aabb_m": {
                "min": (table_center - table_extent).tolist(),
                "max": (table_center + table_extent).tolist(),
            },
        }],
        "joint_positions_rad": {
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i): float(data.qpos[i])
            for i in range(6)
        },
        "gripper_command_0_to_255": float(data.ctrl[model.actuator(GRIPPER_ACTUATOR_NAME).id]),
        "gripper_driver_positions_rad": {
            side: float(data.qpos[model.jnt_qposadr[model.joint(f"gripper/{side}_driver_joint").id]])
            for side in ("left", "right")
        },
        "gripper_pinch_world_position_m": data.site_xpos[model.site(GRIPPER_PINCH_SITE_NAME).id].tolist(),
        "scene_seed": scene.seed,
        "objects": [
            {**item,
             "world_position_m": data.xpos[model.body(item["name"]).id].tolist(),
             "world_quaternion_wxyz": data.xquat[model.body(item["name"]).id].tolist()}
            for item in scene.objects
        ],
    }


def capture(renderer: mujoco.Renderer, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
    renderer.disable_depth_rendering()
    renderer.update_scene(data, camera=CAMERA_NAME)
    rgb = renderer.render().copy()
    renderer.enable_depth_rendering()
    renderer.update_scene(data, camera=CAMERA_NAME)
    depth = renderer.render().astype("<f4", copy=True)
    return rgb, depth


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=1, help="0 streams forever")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--output-dir", type=Path,
                        default=PROJECT_ROOT / "outputs" / "ur5e_eye_in_hand")
    parser.add_argument("--publish", metavar="ENDPOINT", help="optional ZMQ PUB bind, e.g. tcp://*:5555")
    parser.add_argument("--no-save", action="store_true", help="publish without writing files")
    parser.add_argument("--hb-root", type=Path, help="HomebrewedDB root; auto-detected at /datasets/homebrewed")
    parser.add_argument("--hb-count", type=int, choices=(2, 3), help="otherwise randomly choose 2 or 3 objects")
    parser.add_argument("--seed", type=int, help="repeatable object choice and placement")
    parser.add_argument("--include-large", action="store_true", help="allow all 33 models, including large ones")
    parser.add_argument("--gripper-command", type=float, default=0.0,
                        help="2F-85 command from 0 (open) to 255 (closed)")
    args = parser.parse_args()
    if args.frames < 0 or args.fps <= 0:
        parser.error("--frames must be >= 0 and --fps must be > 0")
    if args.no_save and not args.publish:
        parser.error("--no-save requires --publish")
    if not 0.0 <= args.gripper_command <= 255.0:
        parser.error("--gripper-command must be in [0, 255]")

    publisher = context = None
    if args.publish:
        try:
            import zmq
        except ImportError as exc:
            raise SystemExit("ZMQ publishing needs pyzmq: /root/miniconda3/bin/pip install pyzmq") from exc
        context = zmq.Context()
        publisher = context.socket(zmq.PUB)
        publisher.setsockopt(zmq.SNDHWM, 2)
        publisher.bind(args.publish)
        print(f"Publishing RGB-D on {args.publish}", flush=True)

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
    data.ctrl[model.actuator(GRIPPER_ACTUATOR_NAME).id] = args.gripper_command
    mujoco.mj_forward(model, data)

    # Let the two blocks settle while the arm holds its home configuration.
    for _ in range(round(0.5 / model.opt.timestep)):
        mujoco.mj_step(model, data)

    if not args.no_save:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    steps_per_frame = max(1, round(1.0 / (args.fps * model.opt.timestep)))
    wall_start = time.monotonic()
    with mujoco.Renderer(model, width=WIDTH, height=HEIGHT) as renderer:
        frame = 0
        try:
            while args.frames == 0 or frame < args.frames:
                if frame:
                    for _ in range(steps_per_frame):
                        mujoco.mj_step(model, data)
                if publisher is not None:
                    time.sleep(max(0.0, wall_start + frame / args.fps - time.monotonic()))
                mujoco.mj_forward(model, data)
                rgb, depth = capture(renderer, data)
                meta = camera_metadata(model, data, frame, scene)
                meta["capture_fingerprints"] = {
                    "rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
                    "depth_sha256": hashlib.sha256(depth.tobytes()).hexdigest(),
                }
                stem = f"frame_{frame:04d}"
                if not args.no_save:
                    (args.output_dir / f"{stem}_rgb.png").write_bytes(rgb_png(rgb))
                    np.save(args.output_dir / f"{stem}_depth.npy", depth)
                    (args.output_dir / f"{stem}_camera.json").write_text(
                        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
                    )
                if publisher is not None:
                    publisher.send_multipart([
                        b"rgbd", json.dumps(meta).encode("utf-8"),
                        rgb.tobytes(), depth.tobytes(),
                    ])
                destination = "ZMQ" if args.no_save else str(args.output_dir)
                print(f"{stem}: t={data.time:.3f}s, RGB {rgb.shape}, "
                      f"depth {depth.shape} m -> {destination}", flush=True)
                frame += 1
        except KeyboardInterrupt:
            pass
        finally:
            if publisher is not None:
                publisher.close(linger=0)
                context.term()


if __name__ == "__main__":
    main()
