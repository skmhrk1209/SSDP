# Stochastic Signed Distance Processes

Hiroki Sakuma and Masatoshi Okutomi <br>
**Transactions on Machine Learning Research (TMLR), 2026** <br>
[arXiv](https://arxiv.org/abs/2606.20856) | [OpenReview](https://openreview.net/forum?id=JSftvBX0lg) | [Bibliography](https://jmlr.org/tmlr/papers/bib/JSftvBX0lg.bib)

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
    --save-dir=datasets/nerfstudio
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
    --root-dir=${ROOT_DIR}
```

5. (Optional) Perform Poisson surface reconstruction to obtain the GT meshes.

GT meshes are needed for evaluating the learned first-passage distribution for each ray with the uncertainty quantification metrics reported in the paper.

```bash
uv run python tools/evaluation/reconstruct_surface.py \
    --point-file=${DTU_DIR}/Points/stl/stl${SCENE_ID}_total.ply \
    --output-file=${SCENE_DIR}/mesh/gt_mesh.ply
```

### [MobileBrick](https://github.com/ActiveVisionLab/MobileBrick)

1. Download the dataset from the [official repository](https://github.com/ActiveVisionLab/MobileBrick).

2. Organize the dataset in the following directory structure.

```bash
datasets
└── nerfstudio
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
    --root-dir=${ROOT_DIR}
```

## Training

Run the following command to train *SSDP-Facto*, which is an SSDP variant of NeuS-Facto.
Use `-h` to see all available command-line arguments.

```bash
MODEL_NAME=ssdp-facto
uv run train ${MODEL_NAME} \
    --output-dir=${OUTPUT_DIR} \
    --project-name=${PROJECT_NAME} \
    --experiment-name=${EXPERIMENT_NAME} \
    custom-sdfstudio-data \
        --data=${SCENE_DIR}
```

All artifacts are saved to `${OUTPUT_DIR}/${PROJECT_NAME}/${MODEL_NAME}/${EXPERIMENT_NAME}/${TIMESTAMP}` =: `EXPERIMENT_DIR`.

The models reported in the paper are summarized as follows.

<table>
  <tr>
    <th align="center">Model</th>
    <th align="center"><code>MODEL_NAME</code></th>
    <th align="center">Renderer</th>
    <th align="center">Sampler</th>
  </tr>
  <tr>
    <td align="center">SSDP&#8209;Facto</td>
    <td align="center"><code>ssdp-facto</code></td>
    <td align="center"><a href="https://arxiv.org/abs/2606.20856">SSDP</a></td>
    <td align="center"><a href="https://arxiv.org/abs/2111.12077">Proposal</a></td>
  </tr>
  <tr>
    <td align="center">NeuS</td>
    <td align="center"><code>neus</code></td>
    <td align="center"><a href="https://arxiv.org/abs/2106.10689">NeuS</a></td>
    <td align="center"><a href="https://arxiv.org/abs/2106.10689">NeuS</a></td>
  </tr>
  <tr>
    <td align="center">NeuS&#8209;Facto</td>
    <td align="center"><code>neus-facto</code></td>
    <td align="center"><a href="https://arxiv.org/abs/2106.10689">NeuS</a></td>
    <td align="center"><a href="https://arxiv.org/abs/2111.12077">Proposal</a></td>
  </tr>
  <tr>
    <td align="center">OaV&#8209;Facto</td>
    <td align="center"><code>oav-facto</code></td>
    <td align="center"><a href="https://arxiv.org/abs/2312.15406">OaV</a></td>
    <td align="center"><a href="https://arxiv.org/abs/2111.12077">Proposal</a></td>
  </tr>
  <tr>
    <td align="center">UNIS&#8209;Facto (C1)</td>
    <td align="center"><code>unis-c1-facto</code></td>
    <td align="center"><a href="https://openaccess.thecvf.com/content/ICCV2025/html/Deng_UNIS_A_Unified_Framework_for_Achieving_Unbiased_Neural_Implicit_Surfaces_ICCV_2025_paper.html">UNIS (C1)</a></td>
    <td align="center"><a href="https://arxiv.org/abs/2111.12077">Proposal</a></td>
  </tr>
  <tr>
    <td align="center">UNIS&#8209;Facto (C2)</td>
    <td align="center"><code>unis-c2-facto</code></td>
    <td align="center"><a href="https://openaccess.thecvf.com/content/ICCV2025/html/Deng_UNIS_A_Unified_Framework_for_Achieving_Unbiased_Neural_Implicit_Surfaces_ICCV_2025_paper.html">UNIS (C2)</a></td>
    <td align="center"><a href="https://arxiv.org/abs/2111.12077">Proposal</a></td>
  </tr>
  <tr>
    <td align="center">UNIS&#8209;Facto (C3)</td>
    <td align="center"><code>unis-c3-facto</code></td>
    <td align="center"><a href="https://openaccess.thecvf.com/content/ICCV2025/html/Deng_UNIS_A_Unified_Framework_for_Achieving_Unbiased_Neural_Implicit_Surfaces_ICCV_2025_paper.html">UNIS (C3)</a></td>
    <td align="center"><a href="https://arxiv.org/abs/2111.12077">Proposal</a></td>
  </tr>
</table>

Add `--schedule-scale-factor=4` for the 4x training schedule reported in the paper.

## Mesh Extraction

1. Perform Marching Cubes to extract the mesh from the optimized mean field.

```bash
uv run export \
    --config-file=${EXPERIMENT_DIR}/nerfstudio/config.yml \
    --output-file=${EXPERIMENT_DIR}/nerfstudio/export/mesh.ply \
    --resolution=512
```

2. Transform the reconstructed mesh from the normalized space to the metric space, and filter out vertices outside the union of viewing frustums (*unmasked* protocol) or the visual hull (*masked* protocol).

- Unmasked Protocol

```bash
uv run python tools/evaluation/postprocess_mesh.py \
    --meta-file=${SCENE_DIR}/meta_data.json \
    --mesh-file=${EXPERIMENT_DIR}/nerfstudio/export/mesh.ply \
    --config-file=${EXPERIMENT_DIR}/nerfstudio/config.yml \
    --output-file=${EXPERIMENT_DIR}/nerfstudio/export/mesh_unmasked.ply
```

- Masked Protocol

```bash
uv run python tools/evaluation/postprocess_mesh.py \
    --meta-file=${SCENE_DIR}/meta_data.json \
    --mesh-file=${EXPERIMENT_DIR}/nerfstudio/export/mesh.ply \
    --config-file=${EXPERIMENT_DIR}/nerfstudio/config.yml \
    --output-file=${EXPERIMENT_DIR}/nerfstudio/export/mesh_masked.ply \
    --use-foreground-masks \
    --keep-largest-component
```

## Evaluation

1. Evaluate the reconstructed mesh on the official evaluation protocol.

- DTU

```bash
uv run python external/dtu_eval/eval.py \
    --data=${EXPERIMENT_DIR}/nerfstudio/export/mesh_unmasked.ply \
    --dataset_dir=${DTU_DIR} \
    --scan=${SCENE_ID}
```

- MobileBrick

For MobileBrick, the pinned evaluation script does not expose a scene-wise command-line interface for this workflow, so we intentionally omit a concrete command from this README. Local adaptation may be necessary depending on your workflow.

```bash
uv run python external/lego_eval/evaluations/evaluate_3d.py ...
```

2. (Optional) Evaluate the learned first-passage distribution for each ray with the uncertainty quantification metrics reported in the paper.

- DTU (in millimeters)

```bash
uv run python tools/evaluation/evaluate_uq_metrics.py \
    --meta-file=${SCENE_DIR}/meta_data.json \
    --target-mesh-file=${SCENE_DIR}/mesh/gt_mesh.ply \
    --config-file=${EXPERIMENT_DIR}/nerfstudio/config.yml \
    --output-file=${EXPERIMENT_DIR}/nerfstudio/evaluation/depth/metrics.json \
    --ray-interval=5.0
```

- MobileBrick (in meters)

```bash
uv run python tools/evaluation/evaluate_uq_metrics.py \
    --meta-file=${SCENE_DIR}/meta_data.json \
    --target-mesh-file=${SCENE_DIR}/mesh/gt_mesh.ply \
    --config-file=${EXPERIMENT_DIR}/nerfstudio/config.yml \
    --output-file=${EXPERIMENT_DIR}/nerfstudio/evaluation/depth/metrics.json \
    --ray-interval=0.005
```

## Checkpoints

The checkpoints of the models reported in the paper are available on [Hugging Face Hub](https://huggingface.co/skmhrk1209/SSDP).
They were produced with [v1.0.0](https://github.com/skmhrk1209/SSDP/tree/v1.0.0) of this repository.
*Schedule* is the length of the training schedule relative to that of NeuS-Facto.

<table>
  <tr>
    <th align="center">Dataset</th>
    <th align="center">Schedule</th>
    <th align="center">SSDP&#8209;Facto</th>
    <th align="center">NeuS</th>
    <th align="center">NeuS&#8209;Facto</th>
    <th align="center">OaV&#8209;Facto</th>
    <th align="center">UNIS&#8209;Facto<br>(C1)</th>
    <th align="center">UNIS&#8209;Facto<br>(C2)</th>
    <th align="center">UNIS&#8209;Facto<br>(C3)</th>
  </tr>
  <tr>
    <td align="center" rowspan="2">DTU</td>
    <td align="center">1x</td>
    <td align="center"><a href="https://huggingface.co/skmhrk1209/SSDP/tree/v1.0.0/dtu-1x/ssdp-facto">Link</a></td>
    <td align="center"><a href="https://huggingface.co/skmhrk1209/SSDP/tree/v1.0.0/dtu-1x/neus">Link</a></td>
    <td align="center"><a href="https://huggingface.co/skmhrk1209/SSDP/tree/v1.0.0/dtu-1x/neus-facto">Link</a></td>
    <td align="center"><a href="https://huggingface.co/skmhrk1209/SSDP/tree/v1.0.0/dtu-1x/oav-facto">Link</a></td>
    <td align="center"><a href="https://huggingface.co/skmhrk1209/SSDP/tree/v1.0.0/dtu-1x/unis-c1-facto">Link</a></td>
    <td align="center"><a href="https://huggingface.co/skmhrk1209/SSDP/tree/v1.0.0/dtu-1x/unis-c2-facto">Link</a></td>
    <td align="center"><a href="https://huggingface.co/skmhrk1209/SSDP/tree/v1.0.0/dtu-1x/unis-c3-facto">Link</a></td>
  </tr>
  <tr>
    <td align="center">4x</td>
    <td align="center"><a href="https://huggingface.co/skmhrk1209/SSDP/tree/v1.0.0/dtu-4x/ssdp-facto">Link</a></td>
    <td align="center"><a href="https://huggingface.co/skmhrk1209/SSDP/tree/v1.0.0/dtu-4x/neus">Link</a></td>
    <td align="center"><a href="https://huggingface.co/skmhrk1209/SSDP/tree/v1.0.0/dtu-4x/neus-facto">Link</a></td>
    <td align="center"><a href="https://huggingface.co/skmhrk1209/SSDP/tree/v1.0.0/dtu-4x/oav-facto">Link</a></td>
    <td align="center"><a href="https://huggingface.co/skmhrk1209/SSDP/tree/v1.0.0/dtu-4x/unis-c1-facto">Link</a></td>
    <td align="center"><a href="https://huggingface.co/skmhrk1209/SSDP/tree/v1.0.0/dtu-4x/unis-c2-facto">Link</a></td>
    <td align="center"><a href="https://huggingface.co/skmhrk1209/SSDP/tree/v1.0.0/dtu-4x/unis-c3-facto">Link</a></td>
  </tr>
  <tr>
    <td align="center">MobileBrick</td>
    <td align="center">1x</td>
    <td align="center"><a href="https://huggingface.co/skmhrk1209/SSDP/tree/v1.0.0/lego-1x/ssdp-facto">Link</a></td>
    <td align="center"><a href="https://huggingface.co/skmhrk1209/SSDP/tree/v1.0.0/lego-1x/neus">Link</a></td>
    <td align="center"><a href="https://huggingface.co/skmhrk1209/SSDP/tree/v1.0.0/lego-1x/neus-facto">Link</a></td>
    <td align="center"><a href="https://huggingface.co/skmhrk1209/SSDP/tree/v1.0.0/lego-1x/oav-facto">Link</a></td>
    <td align="center"><a href="https://huggingface.co/skmhrk1209/SSDP/tree/v1.0.0/lego-1x/unis-c1-facto">Link</a></td>
    <td align="center"><a href="https://huggingface.co/skmhrk1209/SSDP/tree/v1.0.0/lego-1x/unis-c2-facto">Link</a></td>
    <td align="center"><a href="https://huggingface.co/skmhrk1209/SSDP/tree/v1.0.0/lego-1x/unis-c3-facto">Link</a></td>
  </tr>
</table>

### Usage

Download the checkpoints to `OUTPUT_DIR`.
They are organized in the same way as in [Training](#training), where `PROJECT_NAME` is the dataset name with the training schedule (e.g., `dtu-1x`) and `EXPERIMENT_NAME` is the scene name (e.g., `scan24`).

```bash
uv run hf download skmhrk1209/SSDP \
    --revision=v1.0.0 \
    --include="${PROJECT_NAME}/${MODEL_NAME}/${EXPERIMENT_NAME}/*" \
    --local-dir=${OUTPUT_DIR}
```

The downloaded checkpoints can be used for rendering, evaluation, and so on through the Nerfstudio API.

```python
from pathlib import Path

from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.utils.eval_utils import eval_setup

SCENE_DIR = Path(...)
OUTPUT_DIR = Path(...)
EXPERIMENT_DIR = Path(...)


def update_config(config: TrainerConfig) -> TrainerConfig:
    config.output_dir = OUTPUT_DIR
    config.pipeline.datamanager.dataparser.data = SCENE_DIR
    return config


config, pipeline, checkpoint_file, step = eval_setup(
    config_path=EXPERIMENT_DIR / "nerfstudio" / "config.yml",
    update_config_callback=update_config,
)
```

## Citation

```bibtex
@article{
    anonymous2026stochastic,
    title={Stochastic Signed Distance Processes},
    author={Anonymous},
    journal={Submitted to Transactions on Machine Learning Research},
    year={2026},
    url={https://openreview.net/forum?id=JSftvBX0lg},
    note={Under review}
}
```
