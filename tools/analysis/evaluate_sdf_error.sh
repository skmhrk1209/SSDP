DATA_DIR=outputs/TMLR-toy-analysis/data
RUNS_DIR=outputs/TMLR-toy-analysis/ssdp-facto-half

# NOTE: <scene ID> <half width of the slab>: the single-slab scenes of `evaluate_approx_error.sh`. The scenes and
# the renderers can be given as arguments, e.g., for evaluating them in parallel.
SCENES=(
    "k1_w0.05 0.025"
    "k1_w0.025 0.0125"
)

for SCENE in "${SCENES[@]}"; do
    read -r SCENE_ID RADIUS <<< "$SCENE"

    [[ -n "$1" && ! " $1 " =~ " $SCENE_ID " ]] && continue

    for VARIANT in ${2:-na na_up bf bf_up}; do
        RUN_ID=$(tr a-z_ A-Z- <<< $VARIANT)

        uv run python -m tools.analysis.evaluate_sdf_error \
            --config-file $RUNS_DIR/ssdp-facto-half-$SCENE_ID-$RUN_ID/202604/nerfstudio/config.yml \
            --output-file $DATA_DIR/sdf/${SCENE_ID}_$VARIANT.json \
            --cuboid-config.radii $RADIUS 0.5 0.5 \
            --cuboid-config.range 0.0 0.0 \
            --cuboid-config.count 1
    done
done
