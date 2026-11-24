# Open-vocabulary wrist-camera perception

This example runs [YOLO-World](https://docs.ultralytics.com/models/yolo-world/) on the MuJoCo wrist RGB frame with your text prompts, then uses each detected box to prompt [SAM2 Tiny](https://docs.ultralytics.com/models/sam-2/). It saves an annotated image, one binary PNG mask per detection, and a JSON report with labels, scores, boxes, timing, and the original camera metadata when available.

The example uses the smallest YOLO-World v2 checkpoint (`yolov8s-worldv2.pt`) and SAM2 Tiny (`sam2_t.pt`). Their weights download on first use to `outputs/open_vocab_pipeline/models/`; they are not stored in Git. YOLO-World also downloads a CLIP text encoder to the container's `/root/.cache/clip/`. That CLIP cache is lost if the container is removed unless you mount it from the host. Ultralytics distributes its code and models under [AGPL-3.0](https://www.ultralytics.com/license).

## Install in the running container

```bash
docker exec vla_scenegraph_env /root/miniconda3/bin/pip install \
  -r /workspace/examples/open_vocab_pipeline/requirements.txt
```

The image already has CUDA PyTorch. You can also set `--device cpu`, though SAM2 will be slower. The default processes one frame at a time and caps detections at five to suit a 6 GB GPU.

If you recreate the Docker container often, add `-v "$PWD/outputs/open_vocab_pipeline/clip_cache":/root/.cache/clip` to the `docker run` command from the repository root. Create that host directory first with `mkdir -p outputs/open_vocab_pipeline/clip_cache`. This keeps the approximately 338 MB CLIP checkpoint between container runs.

## Saved wrist frame

First capture a frame with the [UR5e example](../ur5e_eye_in_hand/README.md):

```bash
docker exec vla_scenegraph_env /root/miniconda3/bin/python \
  /workspace/examples/ur5e_eye_in_hand/run.py --frames 1 --seed 7 --hb-count 3
```

Then detect and segment objects visible in that image:

```bash
docker exec -it vla_scenegraph_env /root/miniconda3/bin/python \
  /workspace/examples/open_vocab_pipeline/run.py \
  --input /workspace/outputs/ur5e_eye_in_hand/frame_0000_rgb.png \
  --prompts "red cube, green bell pepper, toy car" --confidence 0.005
```

To process a capture directory, pass the directory to `--input` and use `--frames 0` for every `frame_*_rgb.png`. Change prompts freely, for example `--prompts "red mug, drawer handle, screwdriver"`. A prompt for an object absent from the scene may produce zero detections; the JSON and overlay are still written. The default confidence threshold is 0.15. The lower value in the example reveals small simulated objects with scores around 0.01, so inspect the overlay and treat those labels as tentative. The overlay prints five decimal places because rounding these scores to `0.00` hides useful differences; they are model scores, not calibrated probabilities.

For a seed 1 capture containing four tabletop objects, prompts limited to `red cube, green rabbit, white and black cow, box` led to two animal masks being labeled `green rabbit`. Trying `green frog` and `toy hippopotamus` as additional candidates produced much stronger matching proposals on that frame. Prompt coverage matters: neither YOLO-World nor a CLIP comparison among the supplied prompts can identify a class omitted from the candidate list. Keep alternative object names in the prompt set, inspect the masks, and use a separate `--output-dir` for each prompt experiment.

For that saved seed 1 frame, this broader prompt set and a 0.003 YOLO threshold retained the four visible masks while filtering one gripper proposal:

```bash
docker exec vla_scenegraph_env /root/miniconda3/bin/python \
  /workspace/examples/open_vocab_pipeline/run.py \
  --input /workspace/outputs/ur5e_eye_in_hand/frame_0000_rgb.png \
  --prompts "red cube, green rabbit, green frog, toy cow, toy hippopotamus, cardboard box" \
  --confidence 0.003 --output-dir /workspace/outputs/seed1_perception
```

With aligned depth available, the pipeline asks YOLO-World for three times the requested final detection count, segments those candidates, then filters masks that have too little valid depth, are fragmented or unusually large, duplicate a stronger mask, or touch the image border. The JSON stores each removed proposal and its reason in `filtered_candidates`, plus the thresholds in `post_filter`. This removes gripper and table-edge fragments in the seed 1 capture while preserving the four full object masks. RGB-only input skips the depth tests. A real object cut off by the image edge is also filtered by default; use `--allow-border-masks` to retain it. The depth and mask thresholds are exposed as CLI options for other camera viewpoints and object sizes.

Results appear under `outputs/open_vocab_pipeline/`. `frame_0000_overlay.png` shows the boxes and masks; `frame_0000_mask_00.png` is a single-object binary mask; `frame_0000_detections.json` records the results and, for simulator captures, the camera pose and intrinsics. Its `inputs` field names the RGB image and optional aligned depth array using paths relative to the JSON file. The SHA-256 values cover decoded RGB pixels in H×W×3 `uint8` order and metric depth pixels in H×W little-endian `float32` order, not PNG or NPY file bytes. Each detection also has `mask_sha256` for the H×W binary mask in `uint8` 0/1 order. Fresh simulator captures additionally carry matching RGB-D fingerprints in the camera metadata; the 3D graph requires them so it cannot silently use a stale pose. A regular RGB image without a sibling `_depth.npy` records `"depth": null`; the 3D graph step requires depth.

## Live frames from MuJoCo

Start the perception subscriber first in one terminal so model loading finishes before publishing:

```bash
docker exec -it vla_scenegraph_env /root/miniconda3/bin/python \
  /workspace/examples/open_vocab_pipeline/run.py \
  --connect tcp://127.0.0.1:5555 --frames 0 \
  --prompts "red cube, green bell pepper, toy car" --confidence 0.005
```

Then start the simulator publisher in a second terminal:

```bash
docker exec -it vla_scenegraph_env /root/miniconda3/bin/python \
  /workspace/examples/ur5e_eye_in_hand/run.py --frames 0 --fps 2 \
  --publish 'tcp://*:5555' --no-save --seed 7 --hb-count 3
```

Stop either process with Ctrl+C. For each received frame, the subscriber also saves `frame_XXXX_rgb.png` and `frame_XXXX_depth.npy` beside its detection JSON so the 3D graph step can load the exact inputs. The subscriber keeps only a small input queue, so it may drop frames if inference takes longer than the simulator's frame interval. It runs YOLO-World and SAM2 independently on each received frame; it does not track object identity across frames.
