"""Live, explicitly oracle-labeled MuJoCo scene graph for action visualization.

This module reads simulator contacts and poses. It is a diagnostic view of the
world state, not a fresh YOLO/SAM2/CLIP inference or a new observed scene graph.
"""

from __future__ import annotations

from typing import Any

import cv2
import mujoco
import numpy as np


def _geom_aabb(model: mujoco.MjModel, data: mujoco.MjData,
               geom_id: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Bounds for the primitive collision geoms used by this tabletop scene."""
    geom_type = int(model.geom_type[geom_id])
    size = model.geom_size[geom_id]
    if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
        extent = size[:3]
    elif geom_type in (mujoco.mjtGeom.mjGEOM_CYLINDER,
                       mujoco.mjtGeom.mjGEOM_CAPSULE):
        extent = np.array([size[0], size[0], size[1] +
                           (size[0] if geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE else 0)])
    elif geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
        extent = np.repeat(size[0], 3)
    else:
        return None
    rotation = data.geom_xmat[geom_id].reshape(3, 3)
    half = np.abs(rotation) @ extent
    center = data.geom_xpos[geom_id]
    return center - half, center + half


def _body_aabb(model: mujoco.MjModel, data: mujoco.MjData,
               body_id: int, size_m: list[float] | None) -> tuple[np.ndarray, np.ndarray]:
    boxes = []
    for geom_id in np.flatnonzero(model.geom_bodyid == body_id):
        geom_id = int(geom_id)
        # HomebrewedDB visual meshes do not collide; their box proxy does.
        if not (model.geom_contype[geom_id] or model.geom_conaffinity[geom_id]):
            continue
        box = _geom_aabb(model, data, geom_id)
        if box is not None:
            boxes.append(box)
    if boxes:
        return (np.min([box[0] for box in boxes], axis=0),
                np.max([box[1] for box in boxes], axis=0))
    center = data.xpos[body_id]
    half = np.asarray(size_m if size_m is not None else [0.02] * 3) / 2
    return center - half, center + half


def _aabb_dict(lower: np.ndarray, upper: np.ndarray) -> dict[str, list[float]]:
    return {"min": lower.tolist(), "max": upper.tolist()}


def _center(node: dict) -> np.ndarray:
    if "centroid_world_m" in node:
        return np.asarray(node["centroid_world_m"], dtype=float)
    bounds = node["world_aabb_m"]
    return (np.asarray(bounds["min"], dtype=float) +
            np.asarray(bounds["max"], dtype=float)) / 2


class LiveSceneGraph:
    """Keep capture labels stable while updating geometry from MuJoCo state."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData,
                 scene_objects: list[dict], source_graph: dict | None = None,
                 focus_id: str | None = None) -> None:
        self.model = model
        self.scene_objects = list(scene_objects)
        self.focus_id = focus_id
        self.table_geom_id = model.geom("table").id
        self.gripper_site_id = model.site("gripper/pinch").id
        self.object_body_ids = {item["name"]: model.body(item["name"]).id
                                for item in scene_objects}
        self.geom_names = {
            geom_id: (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or "")
            for geom_id in range(model.ngeom)
        }
        self.drawer_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                                  "drawer_slide")
        self.drawer_handle_geom_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, "drawer_handle_geom")
        self.drawer_handle_site_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, "drawer_handle")
        self._labels = self._match_labels(data, source_graph)

    def _match_labels(self, data: mujoco.MjData,
                      source_graph: dict | None) -> dict[str, tuple[str, str]]:
        labels = {item["name"]: (item["name"], item["name"])
                  for item in self.scene_objects}
        if source_graph is None:
            return labels
        mujoco.mj_forward(self.model, data)
        candidates = [node for node in source_graph.get("nodes", [])
                      if node.get("type") == "object"]
        if not candidates:
            return labels
        distances: dict[tuple[str, str], float] = {}
        for item in self.scene_objects:
            name = item["name"]
            body_xy = data.xpos[self.object_body_ids[name], :2]
            for node in candidates:
                try:
                    location = _center(node)
                except (KeyError, TypeError, ValueError):
                    continue
                if location.shape == (3,) and np.isfinite(location).all():
                    distances[(name, node["id"])] = float(np.linalg.norm(
                        location[:2] - body_xy))
        used_names, used_nodes = set(), set()
        for (name, node_id), distance in sorted(distances.items(), key=lambda entry: entry[1]):
            if name in used_names or node_id in used_nodes or distance > 0.10:
                continue
            # A close second object makes a detection-to-body association unsafe.
            other_distances = [d for (other_name, candidate_id), d in distances.items()
                               if candidate_id == node_id and other_name != name]
            competing_nodes = [d for (candidate_name, other_id), d in distances.items()
                               if candidate_name == name and other_id != node_id]
            if ((other_distances and min(other_distances) - distance < 0.025) or
                    (competing_nodes and min(competing_nodes) - distance < 0.025)):
                continue
            node = next(node for node in candidates if node["id"] == node_id)
            labels[name] = (node_id, str(node.get("label", name)))
            used_names.add(name)
            used_nodes.add(node_id)
        return labels

    def snapshot(self, data: mujoco.MjData) -> dict:
        model = self.model
        mujoco.mj_forward(model, data)
        table_lower, table_upper = _geom_aabb(model, data, self.table_geom_id)
        nodes: list[dict[str, Any]] = [{
            "id": "table", "type": "support_surface", "label": "table",
            "world_aabb_m": _aabb_dict(table_lower, table_upper),
            "centroid_world_m": ((table_lower + table_upper) / 2).tolist(),
        }]
        gripper = data.site_xpos[self.gripper_site_id].copy()
        nodes.append({"id": "gripper", "type": "end_effector", "label": "2F-85 gripper",
                      "centroid_world_m": gripper.tolist()})
        contacts: dict[int, set[str]] = {}
        table_touch: set[int] = set()
        handle_sides: set[str] = set()
        for contact in data.contact[:data.ncon]:
            first, second = int(contact.geom1), int(contact.geom2)
            for pad_geom, other_geom in ((first, second), (second, first)):
                side = ("left" if self.geom_names[pad_geom].startswith("gripper/left_pad")
                        else "right" if self.geom_names[pad_geom].startswith(
                            "gripper/right_pad") else None)
                if side is not None:
                    body_id = int(model.geom_bodyid[other_geom])
                    contacts.setdefault(body_id, set()).add(side)
                    if other_geom == self.drawer_handle_geom_id:
                        handle_sides.add(side)
            if first == self.table_geom_id:
                table_touch.add(int(model.geom_bodyid[second]))
            elif second == self.table_geom_id:
                table_touch.add(int(model.geom_bodyid[first]))

        edges: list[dict[str, Any]] = []
        object_nodes = []
        for item in self.scene_objects:
            name = item["name"]
            body_id = self.object_body_ids[name]
            lower, upper = _body_aabb(model, data, body_id, item.get("size_m"))
            node_id, label = self._labels[name]
            sides = contacts.get(body_id, set())
            node = {
                "id": node_id, "type": "object", "label": label,
                "simulator_body": name, "label_source": (
                    "captured_scene_graph" if node_id != name else "simulator_body_name"),
                "centroid_world_m": ((lower + upper) / 2).tolist(),
                "world_aabb_m": _aabb_dict(lower, upper),
                "table_contact": body_id in table_touch,
                "gripper_pad_contacts": sorted(sides),
            }
            nodes.append(node)
            object_nodes.append(node)
            if body_id in table_touch:
                edges.append({"source": node_id, "target": "table", "relation": "on",
                              "evidence": "MuJoCo table contact"})
            if sides == {"left", "right"}:
                edges.append({"source": node_id, "target": "gripper",
                              "relation": "held_by", "evidence": "both finger-pad contacts"})
            elif sides:
                edges.append({"source": node_id, "target": "gripper",
                              "relation": "touching", "evidence": "one finger-pad contact"})

        for index, a in enumerate(object_nodes):
            a_xyz = np.asarray(a["centroid_world_m"])
            for b in object_nodes[index + 1:]:
                b_xyz = np.asarray(b["centroid_world_m"])
                dx, dy = b_xyz[:2] - a_xyz[:2]
                if abs(dx) > 0.06:
                    source, target = (a, b) if dx > 0 else (b, a)
                    edges.append({"source": source["id"], "target": target["id"],
                                  "relation": "left_of", "evidence": "world X centers"})
                if abs(dy) > 0.06:
                    source, target = (b, a) if dy > 0 else (a, b)
                    edges.append({"source": source["id"], "target": target["id"],
                                  "relation": "in_front_of", "evidence": "world Y centers"})
                if np.linalg.norm(b_xyz - a_xyz) < 0.20:
                    edges.append({"source": a["id"], "target": b["id"],
                                  "relation": "next_to", "evidence": "3D center distance < 0.20 m"})

        drawer = None
        if self.drawer_joint_id >= 0:
            joint_id = self.drawer_joint_id
            displacement = float(data.qpos[model.jnt_qposadr[joint_id]])
            max_travel = float(model.jnt_range[joint_id, 1])
            tray_id = model.body("drawer_tray").id
            nodes.append({"id": "drawer_tray", "type": "articulated_object",
                          "label": "drawer tray", "centroid_world_m":
                          data.xpos[tray_id].tolist(), "slide_displacement_m": displacement,
                          "open_fraction": displacement / max_travel if max_travel > 0 else 0})
            handle_xyz = data.site_xpos[self.drawer_handle_site_id].tolist()
            nodes.append({"id": "drawer_handle", "type": "handle",
                          "label": "drawer handle", "centroid_world_m": handle_xyz})
            edges.append({"source": "drawer_handle", "target": "drawer_tray",
                          "relation": "attached_to", "evidence": "MuJoCo fixture"})
            if handle_sides == {"left", "right"}:
                edges.append({"source": "drawer_handle", "target": "gripper",
                              "relation": "held_by", "evidence": "both finger-pad contacts"})
            elif handle_sides:
                edges.append({"source": "drawer_handle", "target": "gripper",
                              "relation": "touching", "evidence": "one finger-pad contact"})
            drawer = {"slide_displacement_m": displacement,
                      "open_fraction": displacement / max_travel if max_travel > 0 else 0,
                      "state": "open" if displacement > 0.02 else "closed"}

        focus = self.focus_id
        if focus is None and object_nodes:
            focus = next((node["id"] for node in object_nodes
                          if node["simulator_body"] == "red_cube"), object_nodes[0]["id"])
        return {
            "schema_version": 1, "source": "simulator_state_oracle",
            "coordinate_frame": "MuJoCo world, meters",
            "simulation_time_s": float(data.time), "focus_id": focus,
            "nodes": nodes, "edges": edges, "drawer": drawer,
        }


def snapshot(model: mujoco.MjModel, data: mujoco.MjData,
             scene_objects: list[dict], source_graph: dict | None = None) -> dict:
    """One-shot convenience wrapper; keep a LiveSceneGraph for stable live IDs."""
    return LiveSceneGraph(model, data, scene_objects, source_graph).snapshot(data)


def render_panel(graph: dict, width: int = 640, height: int = 480) -> np.ndarray:
    """Render a compact RGB graph panel suitable for a live video composite."""
    if width < 440 or height < 320:
        raise ValueError("Graph panel needs at least 440x320 pixels")
    image = np.full((height, width, 3), 250, dtype=np.uint8)

    def text(label: str, x: int, y: int, scale: float = 0.46,
             color: tuple[int, int, int] = (36, 49, 65), thick: int = 1) -> None:
        cv2.putText(image, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    color, thick, cv2.LINE_AA)

    text("LIVE 3D SCENE GRAPH", 18, 30, 0.65, (22, 36, 54), 2)
    text("SIMULATOR STATE (oracle)  |  world XYZ in meters", 18, 50,
         0.39, (129, 75, 27))
    text(f"t = {graph['simulation_time_s']:.2f} s", width - 112, 30,
         0.42, (76, 88, 105))
    split = int(width * 0.65)
    cv2.line(image, (split, 65), (split, height - 25), (215, 221, 228), 1)
    table = next(node for node in graph["nodes"] if node["id"] == "table")
    table_bounds = table["world_aabb_m"]
    xmin, ymin = table_bounds["min"][:2]
    xmax, ymax = table_bounds["max"][:2]
    plot_left, plot_right = 35, split - 20
    plot_top, plot_bottom = 93, height - 56
    ratio = min((plot_right - plot_left) / (xmax - xmin),
                (plot_bottom - plot_top) / (ymax - ymin))
    map_width, map_height = (xmax - xmin) * ratio, (ymax - ymin) * ratio
    x0 = int((plot_left + plot_right - map_width) / 2)
    y0 = int((plot_top + plot_bottom - map_height) / 2)
    x1, y1 = int(x0 + map_width), int(y0 + map_height)
    cv2.rectangle(image, (x0, y0), (x1, y1), (235, 223, 206), -1)
    cv2.rectangle(image, (x0, y0), (x1, y1), (172, 145, 116), 1)

    def xy(position: list[float]) -> tuple[int, int]:
        return (int(x0 + (position[0] - xmin) * ratio),
                int(y1 - (position[1] - ymin) * ratio))

    nodes_by_id = {node["id"]: node for node in graph["nodes"]}
    object_nodes = [node for node in graph["nodes"] if node["type"] == "object"]
    focus_id = graph.get("focus_id")
    gripper = nodes_by_id["gripper"]
    gripper_xy = xy(gripper["centroid_world_m"])
    for edge in graph["edges"]:
        if edge["relation"] == "held_by" and edge["source"] in nodes_by_id:
            source = nodes_by_id[edge["source"]]
            cv2.line(image, xy(source["centroid_world_m"]), gripper_xy,
                     (44, 142, 93), 2, cv2.LINE_AA)

    drawer_node = nodes_by_id.get("drawer_tray")
    if drawer_node is not None:
        drawer_xy = xy(drawer_node["centroid_world_m"])
        cv2.rectangle(image, (drawer_xy[0] - 9, drawer_xy[1] - 9),
                      (drawer_xy[0] + 9, drawer_xy[1] + 9), (92, 105, 121), 2)
        text("drawer", drawer_xy[0] + 13, drawer_xy[1] + 3, 0.36)
        handle_node = nodes_by_id["drawer_handle"]
        handle_xy = xy(handle_node["centroid_world_m"])
        cv2.circle(image, handle_xy, 6, (132, 80, 152), -1, cv2.LINE_AA)
        text("handle", min(handle_xy[0] + 8, split - 65),
             handle_xy[1] - 8, 0.34, (95, 60, 115))

    for node in object_nodes:
        point = xy(node["centroid_world_m"])
        held = any(edge["source"] == node["id"] and edge["relation"] == "held_by"
                   for edge in graph["edges"])
        color = ((39, 140, 86) if held else
                 (184, 91, 43) if node["table_contact"] else (113, 111, 158))
        cv2.circle(image, point, 9 if node["id"] == focus_id else 7,
                   color, -1, cv2.LINE_AA)
        cv2.circle(image, point, 10 if node["id"] == focus_id else 8,
                   (255, 255, 255), 1, cv2.LINE_AA)
        label = node["id"]
        x_text = min(point[0] + 11, split - 88)
        text(label, x_text, point[1] - 4, 0.36)
        text(f"z {node['centroid_world_m'][2]:.2f}", x_text, point[1] + 10,
             0.32, (89, 99, 110))
    cv2.circle(image, gripper_xy, 7, (46, 100, 173), 2, cv2.LINE_AA)
    cv2.line(image, (gripper_xy[0] - 11, gripper_xy[1]),
             (gripper_xy[0] + 11, gripper_xy[1]), (46, 100, 173), 1)
    cv2.line(image, (gripper_xy[0], gripper_xy[1] - 11),
             (gripper_xy[0], gripper_xy[1] + 11), (46, 100, 173), 1)
    overlaps_object = any(np.linalg.norm(np.asarray(gripper_xy) -
                                  np.asarray(xy(node["centroid_world_m"]))) < 27
                          for node in object_nodes)
    if overlaps_object:
        text("gripper", max(gripper_xy[0] - 62, 4), gripper_xy[1] - 16,
             0.34, (46, 100, 173))
    else:
        text("gripper", min(gripper_xy[0] + 12, split - 75),
             gripper_xy[1] + 15, 0.34, (46, 100, 173))
    text("world +X >", x0, height - 32, 0.38)
    text("world +Y ^", x1 - 92, height - 32, 0.38)

    sidebar_x = split + 14
    text("CURRENT RELATIONS", sidebar_x, 87, 0.49, (22, 36, 54), 1)
    focus_node = nodes_by_id.get(focus_id)
    line_y = 111
    if focus_node is not None:
        text(f"{focus_id}: {focus_node['label'][:20]}", sidebar_x, line_y,
             0.39, (37, 50, 66))
        line_y += 22
        focus_edges = [edge for edge in graph["edges"]
                       if edge["source"] == focus_id or edge["target"] == focus_id]
        if not focus_edges:
            text("no active relation", sidebar_x, line_y, 0.38, (107, 115, 122))
            line_y += 21
        else:
            for edge in focus_edges[:7]:
                other = edge["target"] if edge["source"] == focus_id else edge["source"]
                relation = edge["relation"]
                if edge["target"] == focus_id and relation == "left_of":
                    relation = "right_of"
                elif edge["target"] == focus_id and relation == "in_front_of":
                    relation = "behind"
                text(f"{relation} {other}"[:27], sidebar_x, line_y, 0.37,
                     (56, 76, 92))
                line_y += 20
    if graph.get("drawer") is not None:
        line_y = max(line_y + 16, height - 100)
        drawer = graph["drawer"]
        text("DRAWER", sidebar_x, line_y, 0.46, (22, 36, 54))
        text(f"{drawer['state']}  {drawer['slide_displacement_m'] * 100:.1f} cm",
             sidebar_x, line_y + 20, 0.40, (56, 76, 92))
    text("orange: on table   green: held   blue: tool", 18, height - 10,
         0.32, (99, 109, 119))
    return image


def graph_delta(before: dict, after: dict, move_threshold_m: float = 0.02) -> dict:
    """Summarize relation changes and object motion between two oracle snapshots."""
    def edge_key(edge: dict) -> tuple[str, str, str]:
        return edge["source"], edge["relation"], edge["target"]

    old_edges = {edge_key(edge) for edge in before["edges"]}
    new_edges = {edge_key(edge) for edge in after["edges"]}
    old_nodes = {node["id"]: node for node in before["nodes"]}
    moved = []
    for node in after["nodes"]:
        if node["type"] != "object" or node["id"] not in old_nodes:
            continue
        start = np.asarray(old_nodes[node["id"]]["centroid_world_m"])
        end = np.asarray(node["centroid_world_m"])
        distance = float(np.linalg.norm(end - start))
        if distance >= move_threshold_m:
            moved.append({"node_id": node["id"], "distance_m": distance,
                          "from_world_m": start.tolist(), "to_world_m": end.tolist()})
    before_drawer, after_drawer = before.get("drawer"), after.get("drawer")
    return {"added_edges": [list(edge) for edge in sorted(new_edges - old_edges)],
            "removed_edges": [list(edge) for edge in sorted(old_edges - new_edges)],
            "moved_objects": moved,
            "drawer_travel_delta_m": ((after_drawer["slide_displacement_m"] -
                                       before_drawer["slide_displacement_m"])
                                      if before_drawer and after_drawer else None),
            "drawer_state_change": ({"from": before_drawer["state"],
                                     "to": after_drawer["state"]}
                                    if before_drawer and after_drawer and
                                    before_drawer["state"] != after_drawer["state"]
                                    else None)}
