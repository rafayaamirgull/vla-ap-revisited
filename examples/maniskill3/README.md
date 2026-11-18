# ManiSkill PickCube example

Run a short, headless check in the container's ManiSkill Conda environment:

```bash
docker exec -e PYTHONPATH=/workspace/thirdparty/maniskill3 vla_scenegraph_env \
  /root/miniconda3/envs/maniskill/bin/python \
  /workspace/examples/maniskill3/parallel_in_single_scene.py --num-envs 2 --steps 2
```

For an interactive viewer, run:

```bash
docker exec -it -e PYTHONPATH=/workspace/thirdparty/maniskill3 vla_scenegraph_env \
  /root/miniconda3/envs/maniskill/bin/python \
  /workspace/examples/maniskill3/parallel_in_single_scene.py --render --steps 0
```

Headless mode defaults to four environments; viewer mode defaults to one. The viewer shows the first environment if you request more than one. `--num-envs`, `--steps`, and `--seed` are configurable. The two existing benchmark videos are preserved in `videos/ms3_benchmark/`.

This example uses state observations. The headless path uses `parallel_in_single_scene=True`; viewer mode uses separate scenes because ManiSkill 3.0.1's CPU viewer fails with the parallel scene layout. RGB-D observations also require `parallel_in_single_scene=False`. For RGB-D experiments, create the environment with `obs_mode="rgbd"` and `parallel_in_single_scene=False`; a one-environment RGB-D render was verified with the vendored source.
