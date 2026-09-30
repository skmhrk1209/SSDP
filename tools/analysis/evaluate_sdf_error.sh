DATA_DIR=outputs/TMLR-toy-analysis/data
RUNS_DIR=outputs/TMLR-toy-analysis/ssdp-facto-half

# NOTE: <scene ID> <number of the slabs> <range of their centers> <half width of the slabs>: the scenes of
# `evaluate_approx_error.sh`. The scenes and the renderers can be given as arguments, e.g., for evaluating them in
# parallel.
SCENES=(
    "k1_w0.05 1 0.0 0.0 0.025"
    "k3_w0.05 3 -0.5 0.5 0.025"
    "k1_w0.025 1 0.0 0.0 0.0125"
    "k3_w0.025 3 -0.5 0.5 0.0125"
)

for SCENE in "${SCENES[@]}"; do
    read -r SCENE_ID COUNT LOWER UPPER RADIUS <<< "$SCENE"

    [[ -n "$1" && ! " $1 " =~ " $SCENE_ID " ]] && continue

    for VARIANT in ${2:-na na_up bf bf_up}; do
        RUN_ID=$(tr a-z_ A-Z- <<< $VARIANT)

        uv run python -m tools.analysis.evaluate_sdf_error \
            --config-file $RUNS_DIR/ssdp-facto-half-$SCENE_ID-$RUN_ID/202604/nerfstudio/config.yml \
            --output-file $DATA_DIR/sdf/${SCENE_ID}_$VARIANT.json \
            --cuboid-config.radii $RADIUS 0.5 0.5 \
            --cuboid-config.range $LOWER $UPPER \
            --cuboid-config.count $COUNT
    done
done
