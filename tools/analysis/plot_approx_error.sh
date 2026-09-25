DATA_DIR=outputs/TMLR-toy-analysis/data/approx-errors
FIGS_DIR=outputs/TMLR-toy-analysis/figures/ray

# NOTE: <folder> <renderer without the approximation> <renderer with it>
# approx-A: the approximation of the survival conditioning, for each width of the slabs and then each number of them.
# approx-B: the omission of the up-crossing term, for each number of the slabs and then each width of them.
# approx-A+B: both of them, i.e., the renderer of the paper against the one without approximation.
PANELS=(
    "approx-A BF_UP NA_UP"
    "approx-A BF NA"
    "approx-B BF_UP BF"
    "approx-B NA_UP NA"
    "approx-A+B BF_UP NA"
)

# NOTE: The ranges are common to all the figures of a metric: the color scale of the maps ends at the largest
# quantile of the level below of a map (over both phases, all the scenes and the renderers), and the vertical axis
# of the plots along training at the largest upper quartile over all the runs.
METRICS=(
    CONDITIONAL_CRAMER_DISTANCE
    SQUARED_DISTANCE
)
MAP_QUANTILE=0.75
declare -A MAX_DISTANCE MAX_ERROR
for METRIC_NAME in "${METRICS[@]}"; do
    MAX_DISTANCE[$METRIC_NAME]=$(uv run python -c "
from pathlib import Path
from tools.analysis.plot_approx_error import Metric, get_max_distance
print(get_max_distance(tuple(sorted(Path('$DATA_DIR/maps').glob('phase-*/*.json'))), Metric.$METRIC_NAME, $MAP_QUANTILE))
")
    MAX_ERROR[$METRIC_NAME]=$(uv run python -c "
from pathlib import Path
from tools.analysis.plot_approx_error import Metric
from tools.analysis.plot_training_trajectory import get_max_error
print(get_max_error(tuple(sorted(Path('$DATA_DIR/trajectories').glob('*.json'))), Metric.$METRIC_NAME))
")
done

# NOTE: All the panels are drawn for each phase of `evaluate_approx_error.sh`. The phases and the folders (a regular
# expression) can be given as arguments, e.g., for drawing them in parallel.
for PHASE in ${1:-0.0 0.5}; do
    for PANEL in "${PANELS[@]}"; do
        read -r FOLDER VARIANT_1 VARIANT_2 <<< "$PANEL"
        FOLDER=phase-$PHASE/$FOLDER/$(tr _ - <<< $VARIANT_1)_vs_$(tr _ - <<< $VARIANT_2)

        [[ -n "$2" && ! "$FOLDER" =~ $2 ]] && continue

        SCENE_IDS=()
        OUTPUT_DIRS=()
        for COUNT in 1 3; do
            for WIDTH in 0.05 0.025; do
                SCENE_IDS+=(k${COUNT}_w$WIDTH)
                if [[ "$FOLDER" == */approx-A/* ]]; then
                    OUTPUT_DIRS+=($FIGS_DIR/$FOLDER/W-$WIDTH/K-$COUNT)
                else
                    OUTPUT_DIRS+=($FIGS_DIR/$FOLDER/K-$COUNT/W-$WIDTH)
                fi
            done
        done

        for METRIC_NAME in "${METRICS[@]}"; do
            uv run python -m tools.analysis.plot_approx_error \
                --sweep-dir $DATA_DIR/maps/phase-$PHASE \
                --scene-ids "${SCENE_IDS[@]}" \
                --output-dirs "${OUTPUT_DIRS[@]}" \
                --variants $VARIANT_1 $VARIANT_2 \
                --metric $METRIC_NAME \
                --max-distance ${MAX_DISTANCE[$METRIC_NAME]} \
                --trajectory-dir $DATA_DIR/trajectories

            # NOTE: The error of the renderer used in training against Monte Carlo on the learned fields, along training.
            # It is evaluated on the camera rays, so that it does not depend on the phase.
            uv run python -m tools.analysis.plot_training_trajectory \
                --trajectory-dir $DATA_DIR/trajectories \
                --scene-ids "${SCENE_IDS[@]}" \
                --output-dirs "${OUTPUT_DIRS[@]}" \
                --variants $VARIANT_1 $VARIANT_2 \
                --metric $METRIC_NAME \
                --max-error ${MAX_ERROR[$METRIC_NAME]}
        done
    done
done
