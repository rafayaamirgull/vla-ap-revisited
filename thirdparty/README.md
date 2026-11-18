# Third-party sources

- `mujoco_menagerie/` contains the three MuJoCo Menagerie models used by the UR5e tabletop scene, including their original model licenses.
- `maniskill3/` contains the `mani_skill` 3.0.1 Python package source and bundled assets from its published wheel.

The MuJoCo Python runtime is installed in the Docker image and is not copied here because its compiled binaries and Python dependencies belong to that environment. The image also has a separate Conda environment at `/root/miniconda3/envs/maniskill` with ManiSkill 3.0.1 and its dependencies. Set `PYTHONPATH=/workspace/thirdparty/maniskill3` to run the example against this source copy.

See each directory's `UPSTREAM.md` for provenance and license information.
