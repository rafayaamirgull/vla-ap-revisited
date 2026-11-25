# Phase 2, part 1: UR5e action primitives

This example implements `grasp(node_id)`, `place(node_id, target_region)`,
`push(node_id, target_region)`, `pull(handle_id)`, and drawer closure in the
MuJoCo tabletop scene. The arm uses damped least-squares
Jacobian IK at the gripper pinch site, short Cartesian waypoints, and the UR5e's
existing joint-position actuators. The 2F-85 gripper closes through its modeled
actuator. Objects and the drawer move through simulated contact; the code does
not move their poses directly.

## Run the control fixtures

Use the Docker container from the [main README](../../README.md). A cube-only
fixture makes the first two actions repeatable even when HomebrewedDB is not
mounted:

```bash
docker exec vla_scenegraph_env /root/miniconda3/bin/python \
  /workspace/examples/action_primitives/run.py \
  --action grasp-place --target-xy -0.16 0.62 \
  --output-dir /workspace/outputs/action_primitives/cube_demo
```

Use `--action grasp` to stop after lifting the cube. The place target is a
rectangle on the table, specified by `--target-xy X Y` and
`--region-half-size HALF_X HALF_Y` in **MuJoCo world meters**. When omitted,
the region starts at 6 cm half-size and grows to fit the selected object's
observed footprint with clearance. It must remain inside the table bounds.
For a random scene, choose a free `--target-xy`; the runner checks the replayed
simulator bodies and rejects a target region that overlaps another object
before moving the robot.

For a tabletop push, provide a destination center. The closed gripper moves
along one world X or Y axis, physically contacts the object, then retreats:

```bash
docker exec vla_scenegraph_env /root/miniconda3/bin/python \
  /workspace/examples/action_primitives/run.py --action push \
  --target-xy -0.34 0.60 \
  --output-dir /workspace/outputs/action_primitives/cube_push
```

`--target-xy` is an absolute world position. Changing `--node-id` while keeping
the same target requests a different motion; choose a clear, reachable target
for each object. A push must move at least 8 cm and keep either X or Y aligned
with the selected object's observed center (within 3 cm).

The optional drawer fixture has a named sliding joint, handle site, and handle
contact geom. Open it with the handle:

```bash
docker exec vla_scenegraph_env /root/miniconda3/bin/python \
  /workspace/examples/action_primitives/run.py --action pull \
  --handle-id drawer_handle \
  --output-dir /workspace/outputs/action_primitives/drawer_demo
```

Use `--action pull-push` to pull the drawer open and then physically push it
closed. The graph records the slide state at all three checkpoints:

```bash
docker exec -it vla_scenegraph_env /root/miniconda3/bin/python \
  /workspace/examples/action_primitives/run.py --action pull-push \
  --visualize --native-viewer \
  --output-dir /workspace/outputs/action_primitives/live_pull_push
```

Each action run that starts writes `action_result.json` and `trajectory.csv`. The JSON records
the action outcome, target source, measured lift or drawer travel, and contact
evidence. The CSV samples the pinch site, cube, and drawer joint during the run.
An action that fails after starting writes a failed result and exits with a
nonzero status.
If the requested output directory already contains an action run, the runner
creates a numbered sibling directory and prints its actual path. Earlier videos
and checkpoints remain available. Invalid targets caught before motion produce
an error without replacing an earlier run.

## Watch the robot and the graph change

The same action runner can open a live three-panel view: MuJoCo overview, wrist
RGB, and a scene graph whose positions and relations update as contact changes.
It saves a video and RGB-D snapshots at the start, after grasp, and after place
or pull:

```bash
docker exec -it vla_scenegraph_env /root/miniconda3/bin/python \
  /workspace/examples/action_primitives/run.py --action grasp-place \
  --visualize --perception-checkpoints \
  --output-dir /workspace/outputs/action_primitives/live_pick
```

For the sliding drawer, use `--action pull --visualize` and a different
`--output-dir`. Add `--native-viewer` if you want a second MuJoCo window whose
3D camera you can rotate with the mouse. Use `--record-video` in place of
`--visualize` to create the MP4 without opening a window. Live playback follows
simulation time by default; `--playback-speed 2` runs twice as fast.

Open the saved results from the repository root after the command finishes:
If the runner printed a numbered output directory, substitute that directory
in these commands.

```bash
xdg-open outputs/action_primitives/live_pick/action_visualization.mp4
xdg-open outputs/action_primitives/live_pick/checkpoints/perceived_checkpoints.png
```

`preview_start.png` and `preview_end.png` are quick stills. The
`checkpoints/simulator_changes.json` file spells out which relation edges were
added or removed and how far each object or drawer moved. The individual
RGB-D captures, detection overlays, and 3D graph plots are under `checkpoints/`.

The **live** graph is built from MuJoCo body poses, table contacts, finger-pad
contacts, and drawer joint travel. It carries the original observed node labels
when they can be matched unambiguously, but its geometry and changing edges are
clearly marked **simulator state (oracle)**. This makes `on table` disappear on
lift, `held_by gripper` appear during contact, and `on table` return after
placement. A push moves an object node while its `on table` edge remains. The
drawer node moves from closed to open during `pull`, then back during
`pull-push`.

`--perception-checkpoints` additionally runs the existing YOLO-World/SAM2 and
3D scene-graph stages on saved wrist RGB-D frames **after** the live action.
These are separate observed graphs with frame-local IDs. With the default
camera and filtering, the cube is detected before grasp and after place, but
may be missing while held: it reaches the image border and is closer than the
default 15 cm valid-depth limit. A missing observed node at that checkpoint
means **unobserved from the wrist**, not removed from the simulation.

## Use an observed 3D scene-graph node

The control fixture above gets the cube bounds from MuJoCo and labels that
source `simulator_fixture_oracle`. To drive the arm from masked depth instead,
capture a fresh frame and run the existing perception and scene-graph stages:

```bash
docker exec vla_scenegraph_env /root/miniconda3/bin/python \
  /workspace/examples/ur5e_eye_in_hand/run.py --frames 1 --cube-only \
  --output-dir /workspace/outputs/action_primitives/perception_capture

docker exec vla_scenegraph_env /root/miniconda3/bin/python \
  /workspace/examples/open_vocab_pipeline/run.py \
  --input /workspace/outputs/action_primitives/perception_capture/frame_0000_rgb.png \
  --prompts 'red cube, robot gripper, table' --confidence 0.003 \
  --output-dir /workspace/outputs/action_primitives/perception

docker exec vla_scenegraph_env /root/miniconda3/bin/python \
  /workspace/examples/scene_graph/run.py \
  --detections /workspace/outputs/action_primitives/perception/frame_0000_detections.json \
  --output-dir /workspace/outputs/action_primitives/graph

docker exec vla_scenegraph_env /root/miniconda3/bin/python \
  /workspace/examples/action_primitives/run.py --action grasp-place \
  --scene-graph /workspace/outputs/action_primitives/graph/frame_0000_scene_graph.json \
  --node-id object_00 \
  --output-dir /workspace/outputs/action_primitives/graph_demo
```

Check the generated graph and detection overlay before selecting `--node-id`:
`object_00` is an example for this one-object capture, and graph IDs change
between frames. The action runner accepts only frame 0, checks the capture
fingerprints, rebuilds the scene from its recorded seed and object list, and
rejects a replay whose initial object poses differ. A HomebrewedDB graph needs
the same dataset mounted at `/datasets/homebrewed` for this check.

**Capture seeds belong only to the simulator capture command.** Perception and
action replay read the seed and the exact object-count mode from capture
metadata. Use a separate capture, perception, and graph output directory for
each experiment; reusing `frame_0000` for a different scene after building a
graph invalidates that graph's RGB-D fingerprints.

For the saved seed-1 scene with three HomebrewedDB objects, the red cube is
`object_02` in that **specific frame**. This command records a pick and place
into a free table region while keeping the other objects visible in the live
simulator graph:

```bash
docker exec -it vla_scenegraph_env /root/miniconda3/bin/python \
  /workspace/examples/action_primitives/run.py --action grasp-place \
  --scene-graph /workspace/outputs/seed1_graph/frame_0000_scene_graph.json \
  --node-id object_02 --target-xy 0.30 0.40 \
  --visualize --perception-checkpoints \
  --prompts 'red cube, green frog, toy hippopotamus, cardboard box' \
  --confidence 0.001 \
  --output-dir /workspace/outputs/action_primitives/live_seed1_pick
```

The scene graph, detection report, captured RGB-D files, and HomebrewedDB
mount must still match for replay to pass. In this view, the refined prompt
set produced four observed objects before the action, none at the held wrist
checkpoint, and one red cube after placement. The live simulator graph keeps
showing all four objects and the cube's physical relations throughout.

The fixed top-down 2F-85 grasp opens 8.5 cm along its current jaw axis. In the
saved seed-1 graph, the other three objects' observed widths along that axis
are larger, so `--action grasp` and `grasp-place` reject them before moving.
They can be pushed to a clear region with the following **seed-1-only** node
IDs and tested targets:

| Seed-1 node | Object candidate | Push command arguments |
| --- | --- | --- |
| `object_00` | toy hippopotamus | `--action push --node-id object_00 --target-xy -0.44 0.83` |
| `object_01` | green frog | `--action push --node-id object_01 --target-xy -0.14 0.71` |
| `object_02` | red cube | `--action push --node-id object_02 --target-xy -0.34 0.49` |
| `object_03` | cardboard box | `--action push --node-id object_03 --target-xy 0.30 0.67 --region-half-size 0.15 0.13` |

Add `--scene-graph /workspace/outputs/seed1_graph/frame_0000_scene_graph.json`
and a fresh `--output-dir` to each command. These are axis-aligned pushes with
the current object collision proxies; their final displacement is measured
from simulation contacts, not assumed from the arm target. A different random
scene requires inspecting its graph and choosing a free destination.

To watch the frog push with wrist perception checkpoints:

```bash
docker exec -it vla_scenegraph_env /root/miniconda3/bin/python \
  /workspace/examples/action_primitives/run.py --action push \
  --scene-graph /workspace/outputs/seed1_graph/frame_0000_scene_graph.json \
  --node-id object_01 --target-xy -0.14 0.71 \
  --visualize --perception-checkpoints \
  --prompts 'red cube, green frog, toy hippopotamus, cardboard box' \
  --confidence 0.001 \
  --output-dir /workspace/outputs/action_primitives/live_seed1_frog_push
```

## Controller and outcome checks

- `grasp` moves above the selected graph bounds, descends to their center,
  closes the gripper, and lifts. Both finger pads must touch the same nearby
  simulator body at closure and after lift; that body must rise at least 6 cm.
- `place` transfers the held object above a table region, descends, opens the
  gripper, and retreats. The released body must settle inside the region, at
  table height, with actual table contact.
- `pull` approaches the registered handle from above, closes on its named
  contact geom, and moves along the slide axis. It checks handle contact at the
  start and end of the pull and measures the joint travel.
- `push` moves the closed gripper from behind a tabletop object along one world
  axis. It checks tool contact, object movement into the target region, and
  final table contact. Diagonal target offsets above 3 cm are rejected.
- `pull-push` runs `pull`, then regrips and pushes the drawer handle to within
  1 cm of closed.

The graph bounds guide motion. Simulator body poses, contacts, and joint travel
are used for **outcome checks**. The drawer handle registry is a fixture ID,
not yet a detected scene-graph node. These are top-down tabletop primitives:
they do not plan around clutter, choose alternate grasp orientations, track
nodes across frames, or provide the later action graph and VLM planner.
Irregular or large objects may require a side grasp or a different push pose.

Run the physics integration tests in the same container:

```bash
docker exec vla_scenegraph_env /root/miniconda3/bin/python \
  -m unittest discover -s /workspace/examples/action_primitives -p 'test_*.py'
```
