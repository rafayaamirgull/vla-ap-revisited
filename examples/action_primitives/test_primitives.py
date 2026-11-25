"""Physics integration checks for UR5e tabletop action primitives."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import mujoco
import numpy as np

from live_graph import LiveSceneGraph, graph_delta
from primitives import ActionPrimitives, HandleSpec, TargetRegion
from run import build_scene, fixture_graph, initialize, replay_graph, scene_module
from visualizer import reserve_output_dir


class ActionPrimitivesPhysicsTest(unittest.TestCase):
    def test_cube_is_lifted_and_released_in_place_region(self) -> None:
        scene = build_scene(cube_only=True)
        data = initialize(scene)
        controller = ActionPrimitives(scene.model, data, fixture_graph(scene, data),
                                      scene.objects)
        live_graph = LiveSceneGraph(scene.model, data, scene.objects,
                                    fixture_graph(scene, data))
        before_graph = live_graph.snapshot(data)
        cube_id = scene.model.body("red_cube").id
        start_z = float(data.xpos[cube_id, 2])

        grasp = controller.grasp("object_00")
        self.assertTrue(grasp.success)
        self.assertGreater(float(data.xpos[cube_id, 2]) - start_z, 0.10)
        self.assertEqual(controller._pad_contacts(cube_id), {"left", "right"})
        held_graph = live_graph.snapshot(data)
        lifted_change = graph_delta(before_graph, held_graph)
        self.assertIn(["object_00", "held_by", "gripper"],
                      lifted_change["added_edges"])
        self.assertIn(["object_00", "on", "table"],
                      lifted_change["removed_edges"])

        too_small = TargetRegion((-0.16, 0.62), (0.01, 0.01), controller.table_top)
        paused_at = float(data.time)
        with self.assertRaisesRegex(ValueError, "footprint"):
            controller.place("object_00", too_small)
        self.assertEqual(float(data.time), paused_at)

        region = TargetRegion((-0.16, 0.62), (0.06, 0.06), controller.table_top)
        placed = controller.place("object_00", region)
        mujoco.mj_forward(scene.model, data)
        cube = data.xpos[cube_id].copy()
        self.assertTrue(placed.success)
        self.assertLess(float(np.linalg.norm(cube[:2] - region.center_xy_m)), 0.025)
        self.assertAlmostEqual(float(cube[2]), 0.275, delta=0.015)
        self.assertIsNone(controller.held_node_id)
        placed_change = graph_delta(held_graph, live_graph.snapshot(data))
        self.assertIn(["object_00", "on", "table"],
                      placed_change["added_edges"])
        self.assertIn(["object_00", "held_by", "gripper"],
                      placed_change["removed_edges"])

    def test_handle_contact_moves_drawer_joint(self) -> None:
        scene = build_scene(cube_only=True, with_drawer=True)
        data = initialize(scene)
        bad = HandleSpec("drawer_handle", "drawer_handle_geom", "drawer_slide",
                         (0, -1, 0), -0.08)
        controller = ActionPrimitives(scene.model, data, fixture_graph(scene, data),
                                      scene.objects, {"drawer_handle": bad})
        live_graph = LiveSceneGraph(scene.model, data, scene.objects,
                                    fixture_graph(scene, data), focus_id="drawer_handle")
        before_graph = live_graph.snapshot(data)
        joint_id = scene.model.joint("drawer_slide").id
        qpos_id = scene.model.jnt_qposadr[joint_id]
        paused_at = float(data.time)
        with self.assertRaisesRegex(ValueError, "positive"):
            controller.pull("drawer_handle")
        self.assertEqual(float(data.time), paused_at)
        self.assertAlmostEqual(float(data.qpos[qpos_id]), 0.0, delta=0.005)

        controller.handles["drawer_handle"] = HandleSpec(
            "drawer_handle", "drawer_handle_geom", "drawer_slide", (0, -1, 0), 0.08)
        pulled = controller.pull("drawer_handle")
        self.assertTrue(pulled.success)
        self.assertGreater(float(pulled.evidence["joint_travel_m"]), 0.05)
        self.assertGreater(float(data.qpos[qpos_id]), 0.05)
        drawer_change = graph_delta(before_graph, live_graph.snapshot(data))
        self.assertEqual(drawer_change["drawer_state_change"],
                         {"from": "closed", "to": "open"})
        open_graph = live_graph.snapshot(data)
        closed = controller.push_drawer("drawer_handle")
        self.assertTrue(closed.success)
        self.assertLess(float(data.qpos[qpos_id]), 0.01)
        self.assertEqual(graph_delta(open_graph, live_graph.snapshot(data))[
            "drawer_state_change"], {"from": "open", "to": "closed"})

    def test_contact_push_moves_cube_and_updates_graph(self) -> None:
        scene = build_scene(cube_only=True)
        data = initialize(scene)
        graph = fixture_graph(scene, data)
        controller = ActionPrimitives(scene.model, data, graph, scene.objects)
        live_graph = LiveSceneGraph(scene.model, data, scene.objects, graph)
        before = live_graph.snapshot(data)
        result = controller.push("object_00", TargetRegion(
            (-0.34, 0.60), (0.07, 0.07), controller.table_top))
        change = graph_delta(before, live_graph.snapshot(data))
        self.assertTrue(result.evidence["tool_contact_during_push"])
        self.assertGreater(result.evidence["progress_along_push_m"], 0.15)
        self.assertGreater(change["moved_objects"][0]["distance_m"], 0.15)

    def test_legacy_auto_count_capture_replays(self) -> None:
        if scene_module.resolve_hb_root(None) is None:
            self.skipTest("HomebrewedDB is not mounted")
        scene = build_scene(seed=3, hb_count=None)
        data = initialize(scene)
        metadata = scene_module.camera_metadata(scene.model, data, 0, scene)
        metadata.pop("scene_generation")  # legacy captures did not store the RNG recipe
        metadata["capture_fingerprints"] = {"rgb_sha256": "rgb", "depth_sha256": "depth"}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = root / "frame_0000_detections.json"
            report.write_text(json.dumps({"camera_metadata": metadata}), encoding="utf-8")
            graph = root / "frame_0000_scene_graph.json"
            graph.write_text(json.dumps({
                "coordinate_frame": "MuJoCo world, meters", "frame": 0,
                "provenance": {"detections_report": str(report),
                               "rgb_sha256": "rgb", "depth_sha256": "depth"},
            }), encoding="utf-8")
            rebuilt, replay_data, _ = replay_graph(graph)
            metadata["objects"][0]["world_quaternion_wxyz"] = [0, 1, 0, 0]
            report.write_text(json.dumps({"camera_metadata": metadata}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Rebuilt scene differs"):
                replay_graph(graph)
        self.assertEqual([item["object_id"] for item in scene.objects],
                         [item["object_id"] for item in rebuilt.objects])
        self.assertAlmostEqual(float(replay_data.time), 0.5, delta=0.002)

    def test_repeated_run_preserves_previous_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            requested = Path(directory) / "demo"
            first = reserve_output_dir(requested)
            original_result = first / "action_result.json"
            original_result.write_text("successful run", encoding="utf-8")
            second = reserve_output_dir(requested)
            self.assertEqual(first, requested)
            self.assertEqual(second, requested.with_name("demo_run_001"))
            self.assertEqual(original_result.read_text(encoding="utf-8"),
                             "successful run")


if __name__ == "__main__":
    unittest.main()
