DATA_DIR=outputs/TMLR-toy-analysis/data/approx-errors

# NOTE: <scene ID> <number of the slabs> <range of their centers> <half width of the slabs>
SCENES=(
    "k1_w0.05 1 0.0 0.0 0.025"
    "k3_w0.05 3 -0.5 0.5 0.025"
    "k1_w0.025 1 0.0 0.0 0.0125"
    "k3_w0.025 3 -0.5 0.5 0.0125"
)

# NOTE: The phases of the slabs relative to the sample points (`ApproxErrorEvaluator.phase`): at 0.0 the centers of
# the slabs are at sample points, at 0.5 in the middle of a sampling interval.
# The phases and the scenes can be given as arguments, e.g., for evaluating them in parallel.
for PHASE in ${1:-0.0 0.5}; do
    for SCENE in "${SCENES[@]}"; do
        read -r SCENE_ID COUNT LOWER UPPER RADIUS <<< "$SCENE"

        [[ -n "$2" && ! " $2 " =~ " $SCENE_ID " ]] && continue

        uv run python -m tools.analysis.evaluate_approx_error \
            --output-file $DATA_DIR/maps/phase-$PHASE/$SCENE_ID.json \
            --cuboid-config.radii $RADIUS 0.5 0.5 \
            --cuboid-config.range $LOWER $UPPER \
            --cuboid-config.count $COUNT \
            --phase $PHASE
    done
done
