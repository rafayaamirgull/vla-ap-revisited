"""Jacobian IK and contact-checked UR5e tabletop action primitives."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Callable

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


ARM_JOINTS = (
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
)
ARM_ACTUATORS = (
    "shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3",
)
PINCH_SITE = "gripper/pinch"
GRIPPER_ACTUATOR = "gripper/fingers_actuator"
GRIPPER_OPENING_M = 0.085


class ActionError(RuntimeError):
    """An action was unsafe, unreachable, or did not achieve its measured outcome."""


@dataclass(frozen=True)
class TargetRegion:
    center_xy_m: tuple[float, float]
    half_size_xy_m: tuple[float, float]
    support_z_m: float

    def validate(self, table_bounds: dict) -> None:
        center = np.asarray(self.center_xy_m, dtype=float)
        half = np.asarray(self.half_size_xy_m, dtype=float)
        lower = np.asarray(table_bounds["min"], dtype=float)
        upper = np.asarray(table_bounds["max"], dtype=float)
        if (center.shape != (2,) or half.shape != (2,) or
                not np.isfinite([*center, *half, self.support_z_m]).all() or
                np.any(half <= 0) or np.any(center - half < lower[:2]) or
                np.any(center + half > upper[:2]) or
                abs(self.support_z_m - upper[2]) > 0.02):
            raise ValueError("Target region must fit on the calibrated table surface")


@dataclass(frozen=True)
class HandleSpec:
    site_name: str
    contact_geom_name: str
    slide_joint_name: str
    pull_direction_world: tuple[float, float, float]
    pull_distance_m: float


@dataclass(frozen=True)
class ActionResult:
    action: str
    target_id: str
    success: bool
    simulation_time_s: float
    evidence: dict

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class PushPlan:
    """Geometry and initial reachability checked before a tabletop push."""

    footprint: np.ndarray
    target_xy: np.ndarray
    direction: np.ndarray
    axis: int
    desired_distance: float
    approach: np.ndarray
    end: np.ndarray
    body_id: int


class ArmIK:
    """Damped least-squares IK driving the existing joint-position servos."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData,
                 on_step: Callable[[mujoco.MjData], None] | None = None) -> None:
        self.model, self.data, self.on_step = model, data, on_step
        self.site_id = model.site(PINCH_SITE).id
        self.joint_ids = [model.joint(name).id for name in ARM_JOINTS]
        self.qpos_ids = np.asarray([model.jnt_qposadr[i] for i in self.joint_ids], dtype=int)
        self.dof_ids = np.asarray([model.jnt_dofadr[i] for i in self.joint_ids], dtype=int)
        self.actuator_ids = np.asarray([model.actuator(name).id for name in ARM_ACTUATORS],
                                       dtype=int)
        self.lower = np.maximum(model.jnt_range[self.joint_ids, 0],
                                model.actuator_ctrlrange[self.actuator_ids, 0]) + 1e-4
        self.upper = np.minimum(model.jnt_range[self.joint_ids, 1],
                                model.actuator_ctrlrange[self.actuator_ids, 1]) - 1e-4
        self.scratch = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        self.tool_rotation = data.site_xmat[self.site_id].reshape(3, 3).copy()

    def pose(self) -> tuple[np.ndarray, np.ndarray]:
        mujoco.mj_forward(self.model, self.data)
        return (self.data.site_xpos[self.site_id].copy(),
                self.data.site_xmat[self.site_id].reshape(3, 3).copy())

    def solve(self, target_position: np.ndarray, target_rotation: np.ndarray | None = None,
              max_iterations: int = 200) -> np.ndarray:
        target = np.asarray(target_position, dtype=float)
        rotation = self.tool_rotation if target_rotation is None else np.asarray(
            target_rotation, dtype=float)
        if (target.shape != (3,) or rotation.shape != (3, 3) or
                not np.isfinite(target).all() or not np.isfinite(rotation).all()):
            raise ValueError("IK requires a finite 3D position and 3x3 rotation")
        self.scratch.qpos[:] = self.data.qpos
        self.scratch.qvel[:] = 0
        q = self.data.qpos[self.qpos_ids].copy()
        for _ in range(max_iterations):
            self.scratch.qpos[self.qpos_ids] = q
            mujoco.mj_forward(self.model, self.scratch)
            current = self.scratch.site_xpos[self.site_id].copy()
            current_rotation = self.scratch.site_xmat[self.site_id].reshape(3, 3)
            position_error = target - current
            rotation_error = Rotation.from_matrix(rotation @ current_rotation.T).as_rotvec()
            if (np.linalg.norm(position_error) < 0.002 and
                    np.linalg.norm(rotation_error) < 0.02):
                return q
            jacp = np.zeros((3, self.model.nv))
            jacr = np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(self.model, self.scratch, jacp, jacr, self.site_id)
            jacobian = np.vstack((jacp[:, self.dof_ids], jacr[:, self.dof_ids]))
            error = np.concatenate((position_error, rotation_error))
            step = jacobian.T @ np.linalg.solve(
                jacobian @ jacobian.T + 0.035**2 * np.eye(6), error)
            length = np.linalg.norm(step)
            if length > 0.12:
                step *= 0.12 / length
            q = np.clip(q + step, self.lower, self.upper)
        raise ActionError(f"IK did not reach {target.tolist()} from the current arm state")

    def step_for(self, seconds: float) -> None:
        steps = max(1, round(seconds / self.model.opt.timestep))
        for _ in range(steps):
            mujoco.mj_step(self.model, self.data)
            if self.on_step is not None:
                self.on_step(self.data)

    def _servo(self, q_target: np.ndarray, seconds: float) -> None:
        q_start = self.data.ctrl[self.actuator_ids].copy()
        steps = max(1, round(seconds / self.model.opt.timestep))
        for index in range(steps):
            fraction = (index + 1) / steps
            smooth = fraction * fraction * (3 - 2 * fraction)
            self.data.ctrl[self.actuator_ids] = q_start + (q_target - q_start) * smooth
            mujoco.mj_step(self.model, self.data)
            if self.on_step is not None:
                self.on_step(self.data)
        self.data.ctrl[self.actuator_ids] = q_target

    def move_linear(self, target_position: np.ndarray, speed_m_s: float = 0.13,
                    final_tolerance_m: float = 0.025) -> None:
        if speed_m_s <= 0 or final_tolerance_m <= 0:
            raise ValueError("Cartesian speed and final tolerance must be positive")
        start, _ = self.pose()
        target = np.asarray(target_position, dtype=float)
        distance = float(np.linalg.norm(target - start))
        waypoints = max(1, math.ceil(distance / 0.04))
        previous = start
        for index in range(1, waypoints + 1):
            waypoint = start + (target - start) * index / waypoints
            q = self.solve(waypoint)
            duration = max(0.20, float(np.linalg.norm(waypoint - previous)) / speed_m_s)
            self._servo(q, duration)
            previous = waypoint
        self.step_for(0.35)
        actual, _ = self.pose()
        if np.linalg.norm(actual - target) > final_tolerance_m:
            raise ActionError(f"Arm stopped {np.linalg.norm(actual - target):.3f} m from target")

    def preview_linear(self, target_position: np.ndarray) -> None:
        """Check the free-space waypoints without advancing physics or controls."""
        start, _ = self.pose()
        target = np.asarray(target_position, dtype=float)
        waypoints = max(1, math.ceil(float(np.linalg.norm(target - start)) / 0.04))
        for index in range(1, waypoints + 1):
            waypoint = start + (target - start) * index / waypoints
            try:
                self.solve(waypoint)
            except ActionError as exc:
                raise ActionError(
                    f"IK cannot reach approach waypoint {np.round(waypoint, 3).tolist()}") from exc


class ActionPrimitives:
    """Operate from graph nodes; use simulator contacts only to check outcomes."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, graph: dict,
                 scene_objects: list[dict], handles: dict[str, HandleSpec] | None = None,
                 on_step: Callable[[mujoco.MjData], None] | None = None) -> None:
        self.model, self.data, self.graph = model, data, graph
        self.arm = ArmIK(model, data, on_step)
        self.gripper_id = model.actuator(GRIPPER_ACTUATOR).id
        self.object_body_ids = {model.body(item["name"]).id for item in scene_objects}
        self.handles = handles or {}
        table = next((node for node in graph["nodes"] if node["id"] == "table"), None)
        if table is None:
            raise ValueError("Graph needs a calibrated table node for manipulation")
        self.table_bounds = table["world_aabb_m"]
        self.table_top = float(self.table_bounds["max"][2])
        self.nodes = {node["id"]: node for node in graph["nodes"] if node["type"] == "object"}
        self.held_node_id: str | None = None
        self.held_body_id: int | None = None

    def _node_bounds(self, node_id: str) -> tuple[np.ndarray, np.ndarray]:
        if node_id not in self.nodes:
            raise ValueError(f"Unknown scene-graph object node: {node_id}")
        bounds = self.nodes[node_id]["world_aabb_m"]
        lower, upper = np.asarray(bounds["min"], dtype=float), np.asarray(bounds["max"], dtype=float)
        if (lower.shape != (3,) or upper.shape != (3,) or
                not np.isfinite(lower).all() or not np.isfinite(upper).all() or
                np.any(lower >= upper)):
            raise ValueError(f"Invalid observed bounds for {node_id}")
        return lower, upper

    def validate_grasp(self, node_id: str) -> None:
        """Reject objects wider than this fixed top-down gripper approach."""
        lower, upper = self._node_bounds(node_id)
        mujoco.mj_forward(self.model, self.data)
        left = self.data.geom_xpos[self.model.geom("gripper/left_pad1").id]
        right = self.data.geom_xpos[self.model.geom("gripper/right_pad1").id]
        opening_axis = (right - left) / np.linalg.norm(right - left)
        observed_width = float(np.abs(opening_axis) @ (upper - lower))
        if observed_width > GRIPPER_OPENING_M:
            raise ActionError(
                f"{node_id} spans {observed_width * 100:.1f} cm across the "
                f"current top-down gripper opening; the 2F-85 opens "
                f"{GRIPPER_OPENING_M * 100:.1f} cm. This grasp pose is unsupported."
            )

    def _gripper(self, command: float, seconds: float) -> None:
        self.data.ctrl[self.gripper_id] = command
        self.arm.step_for(seconds)
        mujoco.mj_forward(self.model, self.data)

    def _pad_contacts(self, body_id: int, geom_id: int | None = None) -> set[str]:
        sides = set()
        for contact in self.data.contact[:self.data.ncon]:
            first, second = int(contact.geom1), int(contact.geom2)
            for pad_geom, object_geom in ((first, second), (second, first)):
                if self.model.geom_bodyid[object_geom] != body_id:
                    continue
                if geom_id is not None and object_geom != geom_id:
                    continue
                name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, pad_geom) or ""
                if name.startswith("gripper/left_pad"):
                    sides.add("left")
                elif name.startswith("gripper/right_pad"):
                    sides.add("right")
        return sides

    def _touching_geom(self, body_id: int, geom_id: int) -> bool:
        return any(
            (int(contact.geom1) == geom_id and
             int(self.model.geom_bodyid[contact.geom2]) == body_id) or
            (int(contact.geom2) == geom_id and
             int(self.model.geom_bodyid[contact.geom1]) == body_id)
            for contact in self.data.contact[:self.data.ncon]
        )

    def _tool_touching_body(self, body_id: int) -> bool:
        for contact in self.data.contact[:self.data.ncon]:
            for tool_geom, other_geom in ((int(contact.geom1), int(contact.geom2)),
                                          (int(contact.geom2), int(contact.geom1))):
                if self.model.geom_bodyid[other_geom] != body_id:
                    continue
                name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM,
                                         tool_geom) or ""
                if name.startswith("gripper/"):
                    return True
        return False

    def grasp(self, node_id: str) -> ActionResult:
        if self.held_node_id is not None:
            raise ActionError(f"Already holding {self.held_node_id}")
        self.validate_grasp(node_id)
        lower, upper = self._node_bounds(node_id)
        center = (lower + upper) / 2
        if center[2] < self.table_top + 0.01:
            raise ActionError("Object grasp center is below the tabletop clearance")
        approach = center.copy()
        approach[2] = max(float(upper[2] + 0.12), float(center[2] + 0.16))
        self._gripper(0, 0.4)
        self.arm.move_linear(approach)
        self.arm.move_linear(center)
        before = {body: self.data.xpos[body].copy() for body in self.object_body_ids}
        self._gripper(255, 1.2)
        candidates = [body for body in self.object_body_ids
                      if self._pad_contacts(body) == {"left", "right"}]
        candidates.sort(key=lambda body: np.linalg.norm(before[body][:2] - center[:2]))
        if not candidates or np.linalg.norm(before[candidates[0]][:2] - center[:2]) > 0.10:
            self._gripper(0, 0.2)
            self.arm.move_linear(approach)
            raise ActionError(f"No two-sided gripper contact with {node_id}")
        body = candidates[0]
        lift = center.copy()
        lift[2] += 0.17
        self.arm.move_linear(lift)
        mujoco.mj_forward(self.model, self.data)
        rise = float(self.data.xpos[body][2] - before[body][2])
        still_grasped = self._pad_contacts(body) == {"left", "right"}
        if rise < 0.06 or not still_grasped:
            self._gripper(0, 0.2)
            raise ActionError(f"{node_id} did not remain grasped after lifting "
                              f"({rise:.3f} m, two-sided contact={still_grasped})")
        self.held_node_id, self.held_body_id = node_id, body
        return ActionResult("grasp", node_id, True, float(self.data.time), {
            "target_center_world_m": center.tolist(),
            "target_geometry_source": self.nodes[node_id].get("geometry_source", "unknown"),
            "simulator_contact_body": mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_BODY, body),
            "two_sided_pad_contact_at_close": True,
            "two_sided_pad_contact_after_lift": still_grasped,
            "measured_lift_m": rise,
            "body_source": "simulator_outcome_check_only",
        })

    def place(self, node_id: str, target_region: TargetRegion) -> ActionResult:
        if self.held_node_id != node_id or self.held_body_id is None:
            raise ActionError(f"Cannot place {node_id}: it is not held")
        target_region.validate(self.table_bounds)
        lower, upper = self._node_bounds(node_id)
        object_height = float(upper[2] - lower[2])
        if np.any(upper[:2] - lower[:2] > 2 * np.asarray(target_region.half_size_xy_m)):
            raise ValueError("Observed object footprint does not fit in target region")
        transfer_z = max(float(self.arm.pose()[0][2]), self.table_top + 0.20)
        current = self.arm.pose()[0]
        if current[2] < transfer_z - 0.01:
            self.arm.move_linear([current[0], current[1], transfer_z])
        x, y = target_region.center_xy_m
        self.arm.move_linear([x, y, transfer_z])
        self.arm.move_linear([x, y, target_region.support_z_m + object_height / 2])
        self._gripper(0, 0.8)
        self.arm.move_linear([x, y, transfer_z])
        mujoco.mj_forward(self.model, self.data)
        body_position = self.data.xpos[self.held_body_id].copy()
        inside_xy = bool(np.all(np.abs(body_position[:2] - [x, y]) <=
                                np.asarray(target_region.half_size_xy_m)))
        on_support = abs(float(body_position[2]) -
                         (target_region.support_z_m + object_height / 2)) < 0.045
        table_contact = self._touching_geom(self.held_body_id,
                                            self.model.geom("table").id)
        body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY,
                                      self.held_body_id)
        self.held_node_id, self.held_body_id = None, None
        if not inside_xy or not on_support or not table_contact:
            raise ActionError(f"Placed body did not settle in region on table: {body_position}")
        return ActionResult("place", node_id, True, float(self.data.time), {
            "target_region": asdict(target_region),
            "simulator_contact_body": body_name,
            "final_body_world_m": body_position.tolist(),
            "table_contact_at_rest": table_contact,
            "body_source": "simulator_outcome_check_only",
        })

    def plan_push(self, node_id: str, target_region: TargetRegion) -> PushPlan:
        """Check a push target and initial arm approach without moving the scene."""
        if self.held_node_id is not None:
            raise ActionError("Release the held object before pushing")
        target_region.validate(self.table_bounds)
        lower, upper = self._node_bounds(node_id)
        footprint = upper[:2] - lower[:2]
        half_region = np.asarray(target_region.half_size_xy_m)
        if np.any(footprint > 2 * half_region):
            raise ValueError("Observed object footprint does not fit in target region")
        center = (lower + upper) / 2
        target_xy = np.asarray(target_region.center_xy_m, dtype=float)
        travel_xy = target_xy - center[:2]
        axis = int(np.argmax(np.abs(travel_xy)))
        other_axis = 1 - axis
        desired_distance = float(abs(travel_xy[axis]))
        if desired_distance < 0.08:
            raise ValueError("Push target must be at least 8 cm from the observed object")
        if abs(travel_xy[other_axis]) > 0.03:
            raise ValueError(
                f"{node_id} is centered at world X={center[0]:.3f}, Y={center[1]:.3f} m. "
                "This push policy moves along one world axis: keep X near "
                f"{center[0]:.3f} m for a Y push or Y near {center[1]:.3f} m "
                "for an X push (within 0.03 m). Choose a target for this object.")
        direction = np.zeros(2)
        direction[axis] = math.copysign(1.0, travel_xy[axis])
        half_along = float(np.abs(direction) @ (footprint / 2))
        start_xy = center[:2] - direction * (half_along + 0.12)
        table_min = np.asarray(self.table_bounds["min"][:2])
        table_max = np.asarray(self.table_bounds["max"][:2])
        if np.any(start_xy < table_min) or np.any(start_xy > table_max):
            raise ActionError("No tabletop clearance behind the object for this push")
        push_z = max(self.table_top + 0.035,
                     min(float(center[2]), self.table_top + 0.10))
        approach = np.array([*start_xy, push_z])
        end = np.array([*(center[:2] + direction *
                           (desired_distance - half_along)), push_z])
        before = {body: self.data.xpos[body].copy() for body in self.object_body_ids}
        ordered = sorted(self.object_body_ids,
                         key=lambda body: np.linalg.norm(before[body][:2] - center[:2]))
        if not ordered or np.linalg.norm(before[ordered[0]][:2] - center[:2]) > 0.12:
            raise ActionError(f"No simulator body near observed node {node_id}")
        try:
            self.arm.preview_linear(approach)
        except ActionError as exc:
            raise ActionError(
                f"Push approach for {node_id} is unreachable with this wrist pose; "
                "choose a different world-axis target. " + str(exc)) from exc
        return PushPlan(footprint, target_xy, direction, axis, desired_distance,
                        approach, end, ordered[0])

    def push(self, node_id: str, target_region: TargetRegion) -> ActionResult:
        """Slide a tabletop object with the closed gripper along the table."""
        plan = self.plan_push(node_id, target_region)
        footprint = plan.footprint
        half_region = np.asarray(target_region.half_size_xy_m)
        target_xy = plan.target_xy
        direction = plan.direction
        axis = plan.axis
        desired_distance = plan.desired_distance
        body = plan.body_id
        before = {candidate: self.data.xpos[candidate].copy()
                  for candidate in self.object_body_ids}
        self._gripper(255, 1.0)
        self.arm.move_linear(plan.approach)
        contact_seen = False
        previous_callback = self.arm.on_step

        def observe_contact(data: mujoco.MjData) -> None:
            nonlocal contact_seen
            if previous_callback is not None:
                previous_callback(data)
            contact_seen |= self._tool_touching_body(body)

        self.arm.on_step = observe_contact
        try:
            # The object can block the tool short of its free-space endpoint.
            # Object motion and contact determine success after retreat.
            self.arm.move_linear(plan.end, speed_m_s=0.09, final_tolerance_m=0.20)
        finally:
            self.arm.on_step = previous_callback
        retreat = self.arm.pose()[0].copy()
        retreat[:2] -= direction * 0.20
        self.arm.move_linear(retreat, final_tolerance_m=0.05)
        self.arm.move_linear(retreat + [0, 0, 0.15], final_tolerance_m=0.05)
        self.arm.step_for(0.5)
        mujoco.mj_forward(self.model, self.data)
        final_position = self.data.xpos[body].copy()
        displacement = final_position[:2] - before[body][:2]
        progress = float(displacement @ direction)
        inside = bool(np.all(np.abs(final_position[:2] - target_xy) <=
                             half_region - footprint / 2))
        table_contact = self._touching_geom(body, self.model.geom("table").id)
        if not contact_seen or progress < min(0.05, desired_distance * 0.5) or not inside or not table_contact:
            raise ActionError(f"Push of {node_id} did not settle in target region "
                              f"(progress={progress:.3f} m, contact={contact_seen}, "
                              f"on_table={table_contact})")
        return ActionResult("push", node_id, True, float(self.data.time), {
            "target_region": asdict(target_region),
            "simulator_contact_body": mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_BODY, body),
            "tool_contact_during_push": contact_seen,
            "push_axis_world": ("+" if direction[axis] > 0 else "-") + "XY"[axis],
            "progress_along_push_m": progress,
            "final_body_world_m": final_position.tolist(),
            "table_contact_at_rest": table_contact,
            "body_source": "simulator_outcome_check_only",
        })

    def pull(self, handle_id: str) -> ActionResult:
        if self.held_node_id is not None:
            raise ActionError("Release the held object before pulling a handle")
        if handle_id not in self.handles:
            raise ValueError(f"Unknown registered handle: {handle_id}")
        spec = self.handles[handle_id]
        site_id = self.model.site(spec.site_name).id
        joint_id = self.model.joint(spec.slide_joint_name).id
        if self.model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_SLIDE:
            raise ValueError("Registered handle must use a slide joint")
        qpos_id = self.model.jnt_qposadr[joint_id]
        direction = np.asarray(spec.pull_direction_world, dtype=float)
        if (direction.shape != (3,) or not np.isfinite(direction).all() or
                not np.isclose(np.linalg.norm(direction), 1.0)):
            raise ValueError("Pull direction must be a unit vector")
        if not np.isfinite(spec.pull_distance_m) or spec.pull_distance_m <= 0:
            raise ValueError("Pull distance must be finite and positive")
        mujoco.mj_forward(self.model, self.data)
        handle = self.data.site_xpos[site_id].copy()
        body_id = int(self.model.site_bodyid[site_id])
        if int(self.model.jnt_bodyid[joint_id]) != body_id:
            raise ValueError("Handle site and slide joint must belong to the same body")
        slide_axis_world = (self.data.xmat[body_id].reshape(3, 3) @
                            self.model.jnt_axis[joint_id])
        if np.dot(direction, slide_axis_world) < 0.98:
            raise ValueError("Pull direction must align with the slide axis")
        if (self.model.jnt_limited[joint_id] and
                self.data.qpos[qpos_id] + spec.pull_distance_m >
                self.model.jnt_range[joint_id, 1] + 0.005):
            raise ValueError("Requested pull exceeds the remaining slide travel")
        handle_geom_id = self.model.geom(spec.contact_geom_name).id
        if int(self.model.geom_bodyid[handle_geom_id]) != body_id:
            raise ValueError("Handle contact geom and site must belong to the same body")
        approach = handle.copy()
        approach[2] += 0.15
        self._gripper(0, 0.4)
        self.arm.move_linear(approach)
        self.arm.move_linear(handle)
        self._gripper(255, 1.0)
        if self._pad_contacts(body_id, handle_geom_id) != {"left", "right"}:
            self._gripper(0, 0.2)
            self.arm.move_linear(approach)
            raise ActionError(f"No two-sided contact with {handle_id}")
        before = float(self.data.qpos[qpos_id])
        self.arm.move_linear(handle + direction * spec.pull_distance_m, speed_m_s=0.08)
        after = float(self.data.qpos[qpos_id])
        stayed_grasped = self._pad_contacts(body_id, handle_geom_id) == {"left", "right"}
        self._gripper(0, 0.4)
        final_handle = self.data.site_xpos[site_id].copy()
        self.arm.move_linear(final_handle + [0, 0, 0.15])
        travel = float(self.data.qpos[qpos_id]) - before
        if (after - before < spec.pull_distance_m * 0.8 or
                travel < spec.pull_distance_m * 0.8 or not stayed_grasped):
            raise ActionError(f"{handle_id} settled after {travel:.3f} m; "
                              f"two-sided handle contact after pull={stayed_grasped}")
        return ActionResult("pull", handle_id, True, float(self.data.time), {
            "joint": spec.slide_joint_name,
            "joint_travel_m": travel,
            "two_sided_pad_contact_at_start": True,
            "two_sided_pad_contact_after_pull": stayed_grasped,
            "evidence_source": "simulator_joint_and_contact",
        })

    def push_drawer(self, handle_id: str) -> ActionResult:
        """Regrasp an open sliding drawer and push its handle to the closed stop."""
        if self.held_node_id is not None:
            raise ActionError("Release the held object before pushing a drawer")
        if handle_id not in self.handles:
            raise ValueError(f"Unknown registered handle: {handle_id}")
        spec = self.handles[handle_id]
        site_id = self.model.site(spec.site_name).id
        joint_id = self.model.joint(spec.slide_joint_name).id
        if self.model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_SLIDE:
            raise ValueError("Registered handle must use a slide joint")
        qpos_id = self.model.jnt_qposadr[joint_id]
        direction = np.asarray(spec.pull_direction_world, dtype=float)
        if (direction.shape != (3,) or not np.isfinite(direction).all() or
                not np.isclose(np.linalg.norm(direction), 1.0)):
            raise ValueError("Pull direction must be a unit vector")
        mujoco.mj_forward(self.model, self.data)
        body_id = int(self.model.site_bodyid[site_id])
        if int(self.model.jnt_bodyid[joint_id]) != body_id:
            raise ValueError("Handle site and slide joint must belong to the same body")
        slide_axis_world = (self.data.xmat[body_id].reshape(3, 3) @
                            self.model.jnt_axis[joint_id])
        if np.dot(direction, slide_axis_world) < 0.98:
            raise ValueError("Pull direction must align with the slide axis")
        handle_geom_id = self.model.geom(spec.contact_geom_name).id
        if int(self.model.geom_bodyid[handle_geom_id]) != body_id:
            raise ValueError("Handle contact geom and site must belong to the same body")
        closed_q = float(self.model.jnt_range[joint_id, 0])
        start_q = float(self.data.qpos[qpos_id])
        if start_q - closed_q < 0.02:
            raise ActionError(f"{handle_id} is already closed")

        handle = self.data.site_xpos[site_id].copy()
        approach = handle.copy()
        approach[2] += 0.15
        self._gripper(0, 0.4)
        self.arm.move_linear(approach)
        # The tray can shift slightly during approach; use its current site.
        self.arm.move_linear(self.data.site_xpos[site_id].copy())
        self._gripper(255, 1.0)
        if self._pad_contacts(body_id, handle_geom_id) != {"left", "right"}:
            self._gripper(0, 0.2)
            self.arm.move_linear(approach)
            raise ActionError(f"No two-sided contact with {handle_id}")

        push_start_q = float(self.data.qpos[qpos_id])
        current_handle = self.data.site_xpos[site_id].copy()
        push_target = current_handle - direction * max(0.0, push_start_q - closed_q)
        self.arm.move_linear(push_target, speed_m_s=0.08)
        stayed_grasped = self._pad_contacts(body_id, handle_geom_id) == {"left", "right"}
        self._gripper(0, 0.4)
        final_handle = self.data.site_xpos[site_id].copy()
        self.arm.move_linear(final_handle + [0, 0, 0.15])
        final_q = float(self.data.qpos[qpos_id])
        if final_q - closed_q > 0.01 or not stayed_grasped:
            raise ActionError(f"{handle_id} remains {final_q - closed_q:.3f} m open; "
                              f"two-sided handle contact during push={stayed_grasped}")
        return ActionResult("push_drawer", handle_id, True, float(self.data.time), {
            "joint": spec.slide_joint_name,
            "joint_travel_m": start_q - final_q,
            "final_opening_m": final_q - closed_q,
            "two_sided_pad_contact_at_start": True,
            "two_sided_pad_contact_after_push": stayed_grasped,
            "evidence_source": "simulator_joint_and_contact",
        })
