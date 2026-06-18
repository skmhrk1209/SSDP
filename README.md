# Stochastic Signed Distance Processes

The official implementation of "Stochastic Signed Distance Processes".

## Installation

1. Install [uv](https://github.com/astral-sh/uv).

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Install this project.

Our project is built on [Nerfstudio](https://github.com/nerfstudio-project/nerfstudio).

```bash
uv sync --frozen && uv sync --frozen --group ext
```

## Datasets

### [DTU](https://roboimagedata.compute.dtu.dk/?page_id=36)

1. Download the dataset via Nerfstudio.

```bash
uv run ns-download-data sdfstudio \
    --dataset-name=dtu \
    --save-dir=datasets/nerfstudio \
```

2. Download [SampleSet](http://roboimagedata2.compute.dtu.dk/data/MVS/SampleSet.zip) and [Points](http://roboimagedata2.compute.dtu.dk/data/MVS/Points.zip) from the [DTU official website](https://roboimagedata.compute.dtu.dk/?page_id=36).

3. Organize the dataset in the following directory structure.

```bash
datasets
├── dtu
│   └── SampleSet
│       └── MVS Data    # <--- <DTU_DIR>
│           ├── ObsMask
│           └── Points
└── nerfstudio
    └── sdfstudio
        └── dtu         # <--- <ROOT_DIR>
            ├── scan24
            ├── ...
            ├── scanXXX # <--- <SCENE_DIR>
            ├── ...
            └── scan122
```

4. Create the metadata for high-resolution images.

Since the original metadata loads the downsampled low-resolution (384x384) images, we modify it to load the original high-resolution (1200x1600) images.

```bash
uv run python tools/dataset/dtu/restore_image_size.py \
    --root-dir=${ROOT_DIR} \
```

5. (Optional) Perform Poisson surface reconstruction to obtain the GT meshes.

GT meshes are needed for evaluating the learned first-passage distribution for each ray with the uncertainty quantification metrics reported in the paper.

```bash
uv run python tools/evaluation/reconstruct_surface.py \
    --point-file=${DTU_DIR}/Points/stl/stl${SCENE_ID}_total.ply \
    --output-file=${SCENE_DIR}/mesh/gt_mesh.ply \
```

### [MobileBrick](https://github.com/ActiveVisionLab/MobileBrick)

1. Download the dataset from the [official repository](https://github.com/ActiveVisionLab/MobileBrick).

2. Organize the dataset in the following directory structure.

```bash
datasets
├── nerfstudio
    └── sdfstudio
        └── lego            # <--- <ROOT_DIR>
            └── test
                ├── aston
                ├── ...
                ├── XXX     # <--- <SCENE_DIR>
                ├── ...
                └── space_shuttle
```

3. Create the metadata for Nerfstudio.

```bash
uv run python tools/dataset/lego/create_meta_data.py \
    --root-dir=${ROOT_DIR} \
```

## Training

Run the following command to train  *SSDP-Facto*, which is an SSDP variant of NeuS-Facto.
Use `-h` to see all available command-line arguments.

```bash
MODEL_NAME=ssdp-facto
uv run ns-train ${MODEL_NAME} \
    --output-dir=${OUTPUT_DIR} \
    --project-name=${PROJECT_NAME} \
    --experiment-name=${EXPERIMENT_NAME} \
    sdfstudio-data \
        --data=${SCENE_DIR} \
```

All artifacts are saved to `${OUTPUT_DIR}/${PROJECT_NAME}/${MODEL_NAME}/${EXPERIMENT_NAME}/${TIMESTAMP}` =: `EXPERIMENT_DIR`.

## Mesh Extraction

1. Perform Marching Cubes to extract the mesh from the optimized mean field.

```bash
uv run ns-export marching-cubes \
    --load-config=${EXPERIMENT_DIR}/nerfstudio/config.yml \
    --output-dir=${EXPERIMENT_DIR}/nerfstudio/export \
    --resolution=512 \
```

2. Transform the reconstructed mesh from the normalized space to the metric space, and filter out vertices outside the union of viewing frustums (*unmasked* protocol) or the visual hull (*masked* protocol).

- Unmasked Protocol

```bash
uv run python tools/evaluation/postprocess_mesh.py \
    --meta-file=${SCENE_DIR}/meta_data.json \
    --mesh-file=${EXPERIMENT_DIR}/nerfstudio/export/sdf_marching_cubes_mesh.ply \
    --config-file=${EXPERIMENT_DIR}/nerfstudio/config.yml \
    --output-file=${EXPERIMENT_DIR}/nerfstudio/export/mesh_unmasked.ply \
```

- Masked Protocol

```bash
uv run python tools/evaluation/postprocess_mesh.py \
    --meta-file=${SCENE_DIR}/meta_data.json \
    --mesh-file=${EXPERIMENT_DIR}/nerfstudio/export/sdf_marching_cubes_mesh.ply \
    --config-file=${EXPERIMENT_DIR}/nerfstudio/config.yml \
    --output-file=${EXPERIMENT_DIR}/nerfstudio/export/mesh_masked.ply \
    --use-foreground-masks \
    --keep-largest-component \
```

## Evaluation

1. Evaluate the reconstructed mesh on the official evaluation protocol.

- DTU

```bash
uv run python external/dtu_eval/eval.py \
    --data=${EXPERIMENT_DIR}/nerfstudio/export/mesh_(unmasked|masked).ply \
    --dataset_dir=${DTU_DIR} \
    --scan=${SCENE_ID} \
```

- MobileBrick

For MobileBrick, the pinned evaluation script does not expose a scene-wise command-line interface for this workflow, so we intentionally omit a concrete command from this README. Local adaptation may be necessary depending on your workflow.

```bash
uv run python external/lego_eval/evaluations/evaluate_3d.py ...
```

2. (Optional) Evaluate the learned first-passage distribution for each ray with the uncertainty quantification metrics reported in the paper.

- DTU

```bash
uv run python tools/evaluation/evaluate_uq_metrics.py \
    --meta-file=${SCENE_DIR}/meta_data.json \
    --target-mesh-file=${SCENE_DIR}/mesh/gt_mesh.ply \
    --config-file=${EXPERIMENT_DIR}/nerfstudio/config.yml \
    --output-file=${EXPERIMENT_DIR}/nerfstudio/evaluation/depth/metrics.json \
    --ray-interval=5.0 \ # in millimeters
```

- MobileBrick

```bash
uv run python tools/evaluation/evaluate_uq_metrics.py \
    --meta-file=${SCENE_DIR}/meta_data.json \
    --target-mesh-file=${SCENE_DIR}/mesh/gt_mesh.ply \
    --config-file=${EXPERIMENT_DIR}/nerfstudio/config.yml \
    --output-file=${EXPERIMENT_DIR}/nerfstudio/evaluation/depth/metrics.json \
    --ray-interval=0.005 \ # in meters
```

## Citation

Coming soon.
