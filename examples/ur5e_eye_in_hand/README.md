# UR5e tabletop with 2F-85 gripper and side-mounted RealSense

This example loads Menagerie's UR5e, adds a table, one red cube, and two or
three randomly selected HomebrewedDB objects. A Robotiq 2F-85 gripper is
attached at the UR5e flange. Menagerie's RealSense D435i is mounted beside
the gripper on a simple white bracket with no cables. The camera housing is
rolled 180 degrees so its modeled mounting feature faces the robot. A single
support and short bolt meet that feature, with no visible gap. Its lenses
still point along the gripper's approach direction. The virtual 640x480
landscape camera sits at the model's RGB lens and captures RGB and metric
depth. The script writes
PNG, NumPy, and JSON files; it can also publish frames over ZMQ for a later
perception process. It uses the existing MuJoCo model and does not require
ROS 2.
The three upstream robot and camera models are stored under
`thirdparty/mujoco_menagerie/` at the project root.

## Run in the existing Docker container

From the repository directory on the host, start your image with the dataset
mounted read-only (omit this if the container already has that mount):

```bash
HB_ROOT='/media/rafay/Windows-SSD/other/data/open_source/HB_HomebrewedDB_TUM'
docker run -it --rm --name vla_scenegraph_env --gpus all --ipc=host --network=host \
  -e DISPLAY="$DISPLAY" \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,display \
  -v /tmp/.X11-unix:/tmp/.X11-unix -v "$PWD":/workspace \
  -v "$HB_ROOT":/datasets/homebrewed:ro -w /workspace \
  -d vla_scenegraph_env:v01
```

Without that mount, the scripts show the cube alone. The first time an object
is selected, its colored PLY is converted to an OBJ mesh and a texture under
`outputs/ur5e_eye_in_hand/hb_cache/`; later runs reuse that cache. The source
dataset is never edited. The cache can be removed to reclaim space. MuJoCo
supports OBJ textures but does not load PLY vertex colors directly; the
conversion interpolates the source colors into small texture tiles for each
triangle. Physical contact uses simple box proxies, so grasping research will
need more accurate collision shapes.

Capture one frame with the gripper open:

```bash
docker exec vla_scenegraph_env /root/miniconda3/bin/python \
  /workspace/examples/ur5e_eye_in_hand/run.py --frames 1
```

Each run chooses two or three distinct tabletop-sized models from 22 eligible
objects and prints a seed. Use the same seed in `run.py` and `view.py` to see
the same selection and placement. For example:

```bash
docker exec vla_scenegraph_env /root/miniconda3/bin/python \
  /workspace/examples/ur5e_eye_in_hand/run.py --frames 1 --seed 7 --hb-count 3
```

Add `--include-large` to sample from all 33 models. Some of the large models
may be partly outside the wrist camera's field of view, but remain visible in
the overview viewer.

The files appear in `outputs/ur5e_eye_in_hand/` on the host. You can run
`--frames 10 --fps 5` for a short sequence. `--frames 0` runs until Ctrl+C;
each frame is saved, so choose a finite frame count for a quick check.
To close the gripper before capture, add `--gripper-command 255`. Commands
range from `0` (open) to `255` (closed). The JSON metadata records the
gripper command, both driver joint angles, and the pinch site's world position.

The container's `/usr/bin/python3` does not have MuJoCo installed. Use the
Conda Python path shown above. On this image, offscreen rendering works with
the X11 flags in the Docker command; no MuJoCo viewer window is required.

## See the whole robot and table in MuJoCo

Run this on the host while the container is running:

```bash
docker exec -it vla_scenegraph_env /root/miniconda3/bin/python \
  /workspace/examples/ur5e_eye_in_hand/view.py
```

The interactive MuJoCo window opens with an overview of the arm, gripper,
RealSense D435i, table, cube, and selected objects. Drag with the mouse to
orbit, and scroll to zoom.

To inspect the gripper, D435i, and mount close up:

```bash
docker exec -it vla_scenegraph_env /root/miniconda3/bin/python \
  /workspace/examples/ur5e_eye_in_hand/view.py --camera realsense
```
Add `--gripper-demo` to repeatedly open and close the fingers, or use
`--gripper-command 255` to hold them closed.
Close the window or press Ctrl+C in the terminal to stop.

To look through the wrist camera instead, start the viewer with:

```bash
docker exec -it vla_scenegraph_env /root/miniconda3/bin/python \
  /workspace/examples/ur5e_eye_in_hand/view.py --camera wrist
```

Add `--sweep-radians 0.1` to either command to move the arm gently and see the
camera follow the gripper. The RGB-D recording script above still saves the
virtual camera's 640x480 images and depth arrays. MuJoCo uses an idealized
pinhole RGB-D render here; the D435i's physical sensor noise and calibrated
intrinsics are not modeled.
Use the same `--seed` (and `--hb-count`, if specified) in both viewer commands
to compare the same object arrangement.

## Optional ZMQ stream

Install `pyzmq` in the running container if it is missing:

```bash
docker exec vla_scenegraph_env /root/miniconda3/bin/pip install pyzmq
```

Start the publisher and subscriber in separate terminals:

```bash
docker exec vla_scenegraph_env /root/miniconda3/bin/python \
  /workspace/examples/ur5e_eye_in_hand/run.py --frames 0 --fps 10 \
  --publish 'tcp://*:5555' --no-save
```

```bash
docker exec vla_scenegraph_env /root/miniconda3/bin/python \
  /workspace/examples/ur5e_eye_in_hand/subscribe.py
```

Each ZMQ message has four parts: topic `rgbd`, UTF-8 JSON metadata, raw
`uint8` RGB in HxWx3 order, and little-endian `float32` depth in meters in
HxW order. Files use the same pixel layout. The JSON provides camera
intrinsics and `world_from_camera_optical`, with OpenCV axes: +X right, +Y
down, +Z forward. The camera pose follows the wrist as the robot moves.
With `--publish`, the publisher aims for the requested wall-clock frame rate;
without it, `--fps` controls only the spacing in simulated time.
