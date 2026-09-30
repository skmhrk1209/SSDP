DATA_DIR=outputs/TMLR-toy-analysis/data/ray
FIGS_DIR=outputs/TMLR-toy-analysis/figures/ray

# NOTE: <folder> <renderer without the approximation> <renderer with it> <scene>:<subfolder>...: the panels of
# the paper, one pair of renderers and two scenes per approximation, with the renderer of the paper (NA) or the one
# without approximation (BF_UP) in the pair; the subfolder of a scene is named after what varies between the scenes.
# approx-A: the approximation of the survival conditioning, for one and three slabs of the width 0.05.
# approx-B: the omission of the up-crossing term, for one slab of each width.
# approx-A+B: both of them, i.e., the renderer of the paper against the one without approximation, for one thick slab
# and for three thin ones.
PANELS=(
    "approx-A BF_UP NA_UP k1_w0.05:K-1 k3_w0.05:K-3"
    "approx-B BF_UP BF k1_w0.05:W-0.05 k1_w0.025:W-0.025"
    "approx-A+B BF_UP NA k1_w0.05:K-1_W-0.05 k3_w0.025:K-3_W-0.025"
)
# NOTE: The other pairs of renderers of each approximation, drawn when "all" is given as the third argument.
OPTIONAL_PANELS=(
    "approx-A BF NA k1_w0.05:K-1 k3_w0.05:K-3"
    "approx-B NA_UP NA k1_w0.05:W-0.05 k1_w0.025:W-0.025"
)
[[ "$3" == all ]] && PANELS+=("${OPTIONAL_PANELS[@]}")

# NOTE: The color scale of the maps is common to all the figures of a metric: it ends at the largest quantile of
# the level below of a map (over both phases, all the scenes and the renderers). The vertical axis of the plots along
# training is common to the figures of a panel (computed in the loop below).
METRICS=(
    CONDITIONAL_CRAMER_DISTANCE
    SQUARED_DISTANCE
)
MAP_QUANTILE=0.75
# NOTE: The arrows and the labels of the steps on the learned trajectories of the maps are drawn up to this step
# (given by hand: the trajectories have settled by then; the lines are drawn to the end).
ANNOTATED_STEP=400
declare -A MAX_DISTANCE
for METRIC_NAME in "${METRICS[@]}"; do
    MAX_DISTANCE[$METRIC_NAME]=$(uv run python -c "
from pathlib import Path
from tools.analysis.plot_approx_error import Metric, get_max_distance
print(get_max_distance(tuple(sorted(Path('$DATA_DIR/maps').glob('phase-*/*.json'))), Metric.$METRIC_NAME, $MAP_QUANTILE))
")
done

# NOTE: All the panels, for each scene: the plots along training in the folder of the scene (they do not depend on
# the phase) and the maps of each phase of `evaluate_approx_error.sh` in its subfolder. The phases and the folders of
# the panels (a regular expression) can be given as arguments, e.g., for drawing them in parallel; a blank list of
# phases (" ") skips the maps.
for PANEL in "${PANELS[@]}"; do
    read -r FOLDER VARIANT_1 VARIANT_2 SCENE_SPECS <<< "$PANEL"
    FOLDER=$FOLDER/$(tr _ - <<< $VARIANT_1)_vs_$(tr _ - <<< $VARIANT_2)

    [[ -n "$2" && ! "$FOLDER" =~ $2 ]] && continue

    SCENE_IDS=()
    OUTPUT_DIRS=()
    for SCENE_SPEC in $SCENE_SPECS; do
        SCENE_IDS+=(${SCENE_SPEC%%:*})
        OUTPUT_DIRS+=($FIGS_DIR/$FOLDER/${SCENE_SPEC#*:})
    done

    for METRIC_NAME in "${METRICS[@]}"; do
        # NOTE: The vertical axis of the plots along training, the main axis and the inset alike, ends at the largest
        # upper quartile over the runs of the two renderers on the scenes of the panel (all the checkpoints): common to
        # the figures of the panel, without the space that the other renderers and scenes would take.
        MAX_ERROR=$(uv run python -c "
from pathlib import Path
from tools.analysis.plot_approx_error import Metric
from tools.analysis.plot_training_trajectory import get_max_error
files = [Path('$DATA_DIR/trajectories') / f'{s}_{v.lower()}.json' for s in '${SCENE_IDS[*]}'.split() for v in ('$VARIANT_1', '$VARIANT_2')]
print(get_max_error(tuple(files), Metric.$METRIC_NAME))
")
        # NOTE: The error of the renderer used in training against Monte Carlo on the learned fields, along training.
        # It is evaluated on the camera rays, so that it does not depend on the phase.
        uv run python -m tools.analysis.plot_training_trajectory \
            --trajectory-dir $DATA_DIR/trajectories \
            --scene-ids "${SCENE_IDS[@]}" \
            --output-dirs "${OUTPUT_DIRS[@]}" \
            --variants $VARIANT_1 $VARIANT_2 \
            --metric $METRIC_NAME \
            --max-error $MAX_ERROR \
            --max-inset-error $MAX_ERROR

        for PHASE in ${1:-0.0 0.5}; do
            MAP_DIRS=()
            for OUTPUT_DIR in "${OUTPUT_DIRS[@]}"; do
                MAP_DIRS+=($OUTPUT_DIR/phase-$PHASE)
            done

            uv run python -m tools.analysis.plot_approx_error \
                --sweep-dir $DATA_DIR/maps/phase-$PHASE \
                --scene-ids "${SCENE_IDS[@]}" \
                --output-dirs "${MAP_DIRS[@]}" \
                --variants $VARIANT_1 $VARIANT_2 \
                --metric $METRIC_NAME \
                --max-distance ${MAX_DISTANCE[$METRIC_NAME]} \
                --trajectory-dir $DATA_DIR/trajectories \
                --annotated-step $ANNOTATED_STEP
        done
    done
done
