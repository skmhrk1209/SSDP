DATA_DIR=outputs/TMLR-toy-analysis/data/approx-errors
RUNS_DIR=outputs/TMLR-toy-analysis/ssdp-facto-half
DATASET_DIR=datasets/nerfstudio/sdfstudio/toy/cuboid

# NOTE: The scenes, the renderers and the slices of the checkpoints (<start>-<stop>, ten checkpoints per slice
# so that a job runs within an hour) can be given as arguments, e.g., for evaluating them in parallel.
# The slices of a run are merged by `merge_training_trajectory.sh`.
for SCENE_ID in ${1:-k1_w0.05 k3_w0.05 k1_w0.025 k3_w0.025}; do
    for VARIANT in ${2:-na na_up bf bf_up}; do
        RUN_ID=$(tr a-z_ A-Z- <<< $VARIANT)

        for SLICE in ${3:-0-10 10-20 20-30 30-40 40-50 50-60 60-70 70-80 80-90 90-100}; do
            uv run python -m tools.analysis.evaluate_training_trajectory \
                --config-file $RUNS_DIR/ssdp-facto-half-$SCENE_ID-$RUN_ID/202604/nerfstudio/config.yml \
                --target-mesh-file $DATASET_DIR/$SCENE_ID/mesh.ply \
                --output-file $DATA_DIR/trajectories/slices/${SCENE_ID}_${VARIANT}_$SLICE.json \
                --checkpoint-slice ${SLICE/-/ }
        done
    done
done
