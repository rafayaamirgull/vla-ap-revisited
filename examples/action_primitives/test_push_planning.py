"""Push target validation must leave the simulated scene untouched."""

from __future__ import annotations

import unittest

import numpy as np

from primitives import ActionError, ActionPrimitives, TargetRegion
from run import build_scene, fixture_graph, initialize


class PushPlanningTest(unittest.TestCase):
    def test_invalid_axis_and_reachability_preview_do_not_advance_physics(self) -> None:
        scene = build_scene(cube_only=True)
        data = initialize(scene)
        controller = ActionPrimitives(scene.model, data,
                                      fixture_graph(scene, data), scene.objects)
        before_time = float(data.time)
        before_qpos = data.qpos.copy()
        before_ctrl = data.ctrl.copy()

        with self.assertRaisesRegex(ValueError, "object_00 is centered at world"):
            controller.plan_push("object_00", TargetRegion(
                (-0.14, 0.71), (0.06, 0.06), controller.table_top))
        with self.assertRaisesRegex(ActionError, "IK cannot reach approach waypoint"):
            controller.arm.preview_linear(np.array([-0.643, 0.687, 0.339]))
        controller.plan_push("object_00", TargetRegion(
            (-0.34, 0.60), (0.07, 0.07), controller.table_top))

        self.assertEqual(float(data.time), before_time)
        np.testing.assert_array_equal(data.qpos, before_qpos)
        np.testing.assert_array_equal(data.ctrl, before_ctrl)


if __name__ == "__main__":
    unittest.main()
