"""Focused geometry and frame-pairing checks; run with unittest discover."""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from geometry import (backproject_mask, contact_patch, dense_component_pixels,
                      infer_between_relations, infer_directional_relations,
                      infer_relations, largest_metric_cluster, observed_aabb,
                      pairwise_geometry, transform_points, world_from_optical)
from run import clip_crop, load_aligned_inputs, main, make_graph, sha256


def node(identifier, minimum, maximum):
    return {"id": identifier, "type": "object", "world_aabb_m": {
        "min": minimum, "max": maximum},
        "centroid_world_m": ((np.asarray(minimum) + maximum) / 2).tolist()}


class GeometryTests(unittest.TestCase):
    def test_pixel_centers_and_nontrivial_base_transform(self):
        depth = np.full((2, 2), 2.0, dtype=np.float32)
        mask = np.array([[True, False], [False, False]])
        optical, pixels = backproject_mask(depth, mask,
                                           {"fx": 2, "fy": 2, "cx": 1, "cy": 1},
                                           0.1, 3.0, 100)
        np.testing.assert_array_equal(pixels, [[0, 0]])
        np.testing.assert_allclose(optical, [[-0.5, -0.5, 2.0]])
        world_base = np.array([[-1, 0, 0, 0], [0, -1, 0, 0],
                               [0, 0, 1, 0.25], [0, 0, 0, 1]], dtype=float)
        base_ee = np.eye(4)
        base_ee[:3, 3] = [0.3, 0, 0]
        ee_camera = np.eye(4)
        ee_camera[:3, 3] = [0, 0.1, 0]
        expected = world_base @ base_ee @ ee_camera
        metadata = {"world_from_camera_optical": expected.tolist(),
                    "kinematic_chain": {"world_from_base": world_base.tolist(),
                                        "base_from_ee": base_ee.tolist(),
                                        "ee_from_camera_optical": ee_camera.tolist()}}
        actual, provenance = world_from_optical(metadata)
        self.assertIn("world_base", provenance)
        np.testing.assert_allclose(transform_points(optical, actual), [[0.2, 0.4, 2.25]])
        metadata["world_from_camera_optical"][0][3] += 0.1
        with self.assertRaisesRegex(ValueError, "disagrees"):
            world_from_optical(metadata)

    def test_largest_cluster_drops_separate_points(self):
        small = np.array([[x, y, 0] for x in (0, 0.003, 0.006)
                          for y in (0, 0.003, 0.006)])
        large = np.array([[1 + x, y, 0] for x in (0, 0.003, 0.006, 0.009)
                          for y in (0, 0.003, 0.006, 0.009)])
        points = np.vstack((small, large))
        retained, stats = largest_metric_cluster(points, 0.004, 0.007, 10)
        self.assertEqual(len(retained), 16)
        self.assertEqual(stats["components"], 2)
        self.assertTrue(np.all(points[retained, 0] > 0.9))

    def test_full_resolution_appearance_excludes_wrong_depth(self):
        depth = np.ones((5, 5), dtype=np.float32)
        depth[2, 2] = 0.5
        mask = np.zeros((5, 5), dtype=bool)
        mask[1:4, 1:4] = True
        intrinsics = {"fx": 100, "fy": 100, "cx": 2.5, "cy": 2.5}
        object_mask = mask.copy()
        object_mask[2, 2] = False
        selected, _ = backproject_mask(depth, object_mask, intrinsics, 0.1, 2.0, 25)
        pixels = dense_component_pixels(depth, mask, intrinsics, np.eye(4), selected,
                                        0.1, 2.0, 0.03)
        self.assertEqual(len(pixels), 8)
        self.assertFalse(np.any(np.all(pixels == [2, 2], axis=1)))
        rgb = np.full((5, 5, 3), [255, 0, 0], dtype=np.uint8)
        rgb[2, 2] = [0, 255, 0]
        crop = clip_crop(rgb, pixels)
        np.testing.assert_array_equal(crop[2, 2], [128, 128, 128])

    def test_support_neighbor_and_calibrated_interior(self):
        table = node("table", [-1, -1, -0.1], [1, 1, 0])
        a = node("object_00", [0, 0, 0.005], [0.05, 0.05, 0.055])
        b = node("object_01", [0.09, 0, 0.005], [0.14, 0.05, 0.055])
        container = node("object_02", [0.4, 0.4, 0], [0.6, 0.6, 0.3])
        child = node("object_03", [0.45, 0.45, 0.08], [0.49, 0.49, 0.12])
        edges = infer_relations([a, b, container, child], table, {
            "object_02": {"min": [0.42, 0.42, 0.02], "max": [0.58, 0.58, 0.28]}})
        self.assertTrue(any(e["relation"] == "on" and e["source"] == "object_00"
                            and e["target"] == "table" for e in edges))
        self.assertTrue(any(e["relation"] == "next_to" and
                            {e["source"], e["target"]} == {"object_00", "object_01"}
                            for e in edges))
        self.assertTrue(any(e["relation"] == "inside" and e["source"] == "object_03"
                            and e["target"] == "object_02" for e in edges))

    def test_sparse_visible_contact_patch_recovers_table_support(self):
        table = node("table", [-1, -1, -0.1], [1, 1, 0])
        low = np.array([[x, y, 0.028] for x in np.linspace(0, 0.04, 5)
                        for y in np.linspace(0, 0.04, 5)])
        high = np.array([[x, y, 0.1] for x in np.linspace(0, 0.04, 45)
                         for y in np.linspace(0, 0.04, 45)])
        cloud = np.vstack((low, high))
        subject = node("object_00", [0, 0, 0.05], [0.04, 0.04, 0.1])
        subject["world_aabb_m"] = observed_aabb(cloud)
        subject["table_contact"] = contact_patch(cloud, 0, 0.03, 0.008, 0.025)
        self.assertGreater(subject["world_aabb_m"]["min"][2], 0.025)
        self.assertEqual(subject["table_contact"]["largest_connected_patch_points"], 25)
        edges = infer_relations([subject], table)
        self.assertEqual([(edge["relation"], edge["target"]) for edge in edges],
                         [("on", "table")])
        self.assertEqual(edges[0]["evidence"]["method"], "connected_visible_contact_patch")

    def test_object_support_requires_consistent_vertical_order(self):
        first = node("first", [0, 0, 0.10], [0.05, 0.05, 0.105])
        second = node("second", [0, 0, 0.10], [0.05, 0.05, 0.105])
        self.assertFalse(any(edge["relation"] == "on" for edge in
                             infer_relations([first, second], None)))

        lower = node("lower", [0, 0, 0], [0.05, 0.05, 0.10])
        upper = node("upper", [0, 0, 0.095], [0.05, 0.05, 0.15])
        on_edges = [edge for edge in infer_relations([lower, upper], None)
                    if edge["relation"] == "on"]
        self.assertEqual([(edge["source"], edge["target"]) for edge in on_edges],
                         [("upper", "lower")])

    def test_world_directions_have_inverse_edges_and_abstain_on_ties(self):
        left = node("left", [0, 0, 0], [0.04, 0.04, 0.04])
        right = node("right", [0.2, 0.2, 0], [0.24, 0.24, 0.04])
        edges = infer_directional_relations([left, right])
        triples = {(e["source"], e["relation"], e["target"]) for e in edges}
        self.assertEqual(triples, {("left", "left_of", "right"),
                                   ("right", "right_of", "left"),
                                   ("left", "behind", "right"),
                                   ("right", "in_front_of", "left")})
        self.assertTrue(all(e["frame"] == "workspace_world" for e in edges))
        lateral = [e for e in edges if e["relation"] in ("left_of", "right_of")]
        self.assertAlmostEqual(sum(e["evidence"]["target_minus_source_m"]
                                   for e in lateral), 0.0)
        tie = node("tie", [0.03, 0, 0], [0.07, 0.04, 0.04])
        self.assertEqual(infer_directional_relations([left, tie]), [])

        lower = node("lower", [0, 0, 0], [0.1, 0.1, 0.1])
        upper = node("upper", [0, 0, 0.11], [0.1, 0.1, 0.2])
        vertical = infer_directional_relations([lower, upper])
        self.assertEqual({e["relation"] for e in vertical}, {"above", "below"})
        upper["world_aabb_m"]["min"][:2] = [0.2, 0.2]
        upper["world_aabb_m"]["max"][:2] = [0.3, 0.3]
        self.assertFalse(any(e["relation"] in ("above", "below") for e in
                             infer_directional_relations([lower, upper])))

    def test_pair_metrics_and_ternary_between_keep_geometry_explicit(self):
        first = node("first", [-0.02, -0.02, 0], [0.02, 0.02, 0.04])
        middle = node("middle", [0.09, 0.0, 0], [0.13, 0.04, 0.04])
        last = node("last", [0.20, -0.02, 0], [0.24, 0.02, 0.04])
        metrics = pairwise_geometry([first, last])
        self.assertEqual(len(metrics), 1)
        self.assertAlmostEqual(metrics[0]["centroid_distance_m"], 0.22)
        self.assertEqual(metrics[0]["centroid_delta_target_minus_source_world_m"],
                         [0.22, 0.0, 0.0])
        self.assertAlmostEqual(metrics[0]["xy_clearance_m"], 0.18)
        self.assertEqual(metrics[0]["observed_aabb_iou_3d"], 0.0)
        between = infer_between_relations([first, middle, last])
        self.assertEqual([(r["subject"], r["references"]) for r in between],
                         [("middle", ["first", "last"])])
        self.assertEqual(between[0]["frame"], "workspace_world_xy")
        shuffled = infer_between_relations([last, middle, first])
        self.assertEqual(shuffled[0]["references"], between[0]["references"])
        self.assertAlmostEqual(shuffled[0]["evidence"]["segment_fraction"],
                               between[0]["evidence"]["segment_fraction"])
        middle["centroid_world_m"][1] = 0.15
        self.assertEqual(infer_between_relations([first, middle, last]), [])

    def test_replaced_rgb_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rgb_path, depth_path = root / "frame_rgb.png", root / "frame_depth.npy"
            rgb = np.full((3, 4, 3), 42, dtype=np.uint8)
            depth = np.ones((3, 4), dtype="<f4")
            self.assertTrue(cv2.imwrite(str(rgb_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)))
            np.save(depth_path, depth)
            report = {"inputs": {"rgb": {"path": rgb_path.name, "sha256": sha256(rgb),
                                         "shape": list(rgb.shape), "encoding": "rgb8"},
                                 "depth": {"path": depth_path.name, "sha256": sha256(depth),
                                           "shape": list(depth.shape),
                                           "encoding": "float32_le_meters"}}}
            loaded_rgb, loaded_depth = load_aligned_inputs(root / "frame_detections.json", report)
            np.testing.assert_array_equal(loaded_rgb, rgb)
            np.testing.assert_array_equal(loaded_depth, depth)
            rgb[0, 0, 0] = 43
            self.assertTrue(cv2.imwrite(str(rgb_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)))
            with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
                load_aligned_inputs(root / "frame_detections.json", report)

    def test_one_interior_file_cannot_calibrate_multiple_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reports = root / "reports"
            reports.mkdir()
            for index in range(2):
                (reports / f"frame_{index:04d}_detections.json").write_text("{}")
            interiors = root / "interiors.json"
            interiors.write_text("{}")
            output = root / "graph"
            argv = ["run.py", "--detections", str(reports), "--frames", "0",
                    "--interiors", str(interiors), "--output-dir", str(output)]
            stderr = io.StringIO()
            with patch("sys.argv", argv), contextlib.redirect_stderr(stderr):
                with self.assertRaises(SystemExit) as error:
                    main()
            self.assertEqual(error.exception.code, 2)
            self.assertIn("per-frame object IDs", stderr.getvalue())
            self.assertFalse(output.exists())

    def test_graph_from_aligned_masked_depth(self):
        class FakeEncoder:
            def encode(self, crops, prompts):
                self_size = crops[0].shape[:2]
                if self_size[0] != self_size[1]:
                    raise AssertionError("CLIP crop was not square")
                if np.count_nonzero(np.all(crops[0] == [255, 0, 0], axis=-1)) < 20:
                    raise AssertionError("CLIP crop lost dense RGB appearance")
                embedding = np.zeros((len(crops), 512), dtype=np.float32)
                embedding[:, 0] = 1
                return embedding, np.full((len(crops), len(prompts)), 0.7)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rgb = np.full((20, 20, 3), [120, 120, 120], dtype=np.uint8)
            rgb[4:8, 6:12] = [255, 0, 0]
            depth = np.ones((20, 20), dtype="<f4")
            mask = np.zeros((20, 20), dtype=np.uint8)
            mask[4:8, 6:12] = 1
            cv2.imwrite(str(root / "frame_rgb.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(root / "frame_mask_00.png"), mask * 255)
            np.save(root / "frame_depth.npy", depth)
            pose = np.eye(4)
            pose[2, 3] = -0.95
            report = {
                "inputs": {
                    "rgb": {"path": "frame_rgb.png", "sha256": sha256(rgb),
                            "shape": list(rgb.shape), "encoding": "rgb8"},
                    "depth": {"path": "frame_depth.npy", "sha256": sha256(depth),
                              "shape": list(depth.shape), "encoding": "float32_le_meters"},
                },
                "prompts": ["red cube"],
                "models": {"detector": "test", "segmenter": "test"},
                "camera_metadata": {
                    "frame": 0, "simulation_time_s": 0.5, "height": 20, "width": 20,
                    "depth_encoding": "float32_le_meters",
                    "capture_fingerprints": {"rgb_sha256": sha256(rgb),
                                             "depth_sha256": sha256(depth)},
                    "intrinsics_opencv": {"fx": 100, "fy": 100, "cx": 10, "cy": 10},
                    "world_from_camera_optical": pose.tolist(),
                    "kinematic_chain": {"world_from_base": np.eye(4).tolist(),
                                        "base_from_ee": np.eye(4).tolist(),
                                        "ee_from_camera_optical": pose.tolist()},
                    "static_geometry": [{"name": "table", "world_aabb_m": {
                        "min": [-1, -1, -0.1], "max": [1, 1, 0.04]}}],
                },
                "detections": [{"label": "red cube", "confidence": 0.8,
                                "mask_file": "frame_mask_00.png", "mask_sha256": sha256(mask)}],
            }
            report_path = root / "frame_detections.json"
            report_path.write_text(json.dumps(report))
            args = Namespace(min_depth=0.1, max_depth=2.0, max_pixels=10,
                             min_points=8, voxel_size=0.008, cluster_radius=0.04,
                             support_tolerance=0.025, contact_band=0.03,
                             min_contact_points=20, neighbor_distance=0.08,
                             direction_deadband=0.04, vertical_xy_overlap=0.15,
                             between_max_perpendicular=0.06,
                             between_min_anchor_distance=0.15,
                             between_max_height_spread=0.12,
                             clip_margin=0.04, allow_border_masks=False,
                             interiors=None, output_dir=root / "graph")
            graph, clouds = make_graph(report_path, FakeEncoder(), args)
            self.assertEqual(graph["schema_version"], 2)
            self.assertEqual(graph["pairwise_geometry"], [])
            self.assertEqual(graph["ternary_relations"], [])
            self.assertEqual(len(graph["nodes"]), 2)
            cube = graph["nodes"][1]
            self.assertEqual(cube["visibility"]["visible_mask_pixels"], 24)
            self.assertEqual(cube["visibility"]["valid_depth_fraction"], 1.0)
            self.assertFalse(cube["visibility"]["touches_image_border"])
            self.assertIsNone(cube["visibility"]["occluded_fraction"])
            self.assertAlmostEqual(cube["centroid_world_m"][2], 0.05)
            self.assertEqual(cube["clip"]["image_embedding_l2"][0], 1.0)
            self.assertEqual(len(clouds[cube["pointcloud_key"]]), 8)
            self.assertEqual([(edge["source"], edge["relation"], edge["target"])
                              for edge in graph["edges"]], [("object_00", "on", "table")])
            report["camera_metadata"]["capture_fingerprints"]["depth_sha256"] = "wrong"
            report_path.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, "fingerprints disagree"):
                make_graph(report_path, FakeEncoder(), args)


if __name__ == "__main__":
    unittest.main()
