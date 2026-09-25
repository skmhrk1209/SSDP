DATA_DIR=outputs/TMLR-toy-analysis/data
FIGS_DIR=outputs/TMLR-toy-analysis/figures

# NOTE: The color scale is common to all the slices: symmetric about zero, ending at the larger magnitude of
# the 5% and the 95% points of the learned SDF over all the slices.
SDF_RANGE=$(uv run python -c "
from pathlib import Path
from tools.analysis.plot_sdf_error import get_sdf_range
print(get_sdf_range(tuple(sorted(Path('$DATA_DIR/sdf').glob('*.npz'))), (0.05, 0.95)))
")

# NOTE: The slice of the learned SDF of every run of `evaluate_sdf_error.sh`, in the folders of the scenes
# (`K-<number of the slabs>/W-<width of the slab>`) as the figures of `plot_approx_error.sh`.
for SCENE_ID in ${1:-k1_w0.05 k1_w0.025}; do
    FOLDER=K-${SCENE_ID:1:1}/W-${SCENE_ID#*_w}

    for VARIANT in ${2:-na na_up bf bf_up}; do
        uv run python -m tools.analysis.plot_sdf_error \
            --input-dir $DATA_DIR/sdf \
            --output-dir $FIGS_DIR/sdf/$FOLDER \
            --scene-id $SCENE_ID \
            --sdf-range $SDF_RANGE \
            --variant $(tr a-z A-Z <<< $VARIANT)
    done
done
