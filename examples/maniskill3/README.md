# ManiSkill PickCube example

Run a short, headless check in the container's ManiSkill Conda environment:

```bash
docker exec -e PYTHONPATH=/workspace/thirdparty/maniskill3 vla_scenegraph_env \
  /root/miniconda3/envs/maniskill/bin/python \
  /workspace/examples/maniskill3/parallel_in_single_scene.py --num-envs 2 --steps 2
```

For an interactive viewer, run with `-it` and add `--render --steps 0`. The default is four environments and 100 steps. `--num-envs`, `--steps`, and `--seed` are configurable. The two existing benchmark videos are preserved in `videos/ms3_benchmark/`.

This example uses state observations. In ManiSkill 3.0.1, `parallel_in_single_scene=True` is incompatible with RGB-D observations. For RGB-D experiments, create the environment with `obs_mode="rgbd"` and `parallel_in_single_scene=False`; a one-environment RGB-D render was verified with the vendored source.
