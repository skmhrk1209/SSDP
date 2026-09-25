DATA_DIR=outputs/TMLR-toy-analysis/data/approx-errors

# NOTE: The standard form (kappa = 1 and tau^2 = 2, the defaults), for which mu_0, sigma, a and b are the dimensionless
# groups themselves. Two sweeps of the length of the interval, each uniform in log on its own axis of
# `plot_up_cross_prob.py`, which draws each axis from its own sweep: the normalized quadratic variation
# Omega / sigma_st^2 (passed to the evaluator as kappa dt = log(1 + Omega / sigma_st^2) / 2 in double precision), and
# the normalized sampling interval kappa dt. The axis and its values can be given as arguments, e.g., for evaluating
# them in parallel: <axis> [<values>].
VALUES=$(uv run python -c "import numpy as np; print(*(f'{x:.4g}' for x in np.logspace(-1.0, 1.0, 11)))")

for AXIS in ${1:-normalized_quadratic_variation normalized_sampling_interval}; do
    for VALUE in ${2:-$VALUES}; do
        case $AXIS in
            normalized_quadratic_variation) NORMALIZED_SAMPLING_INTERVAL=$(uv run python -c "import math; print(repr(math.log1p($VALUE) / 2.0))") ;;
            normalized_sampling_interval) NORMALIZED_SAMPLING_INTERVAL=$VALUE ;;
        esac

        uv run python -m tools.analysis.evaluate_up_cross_prob \
            --output-file $DATA_DIR/up_cross_prob/$AXIS/$VALUE.json \
            --normalized-sampling-intervals $NORMALIZED_SAMPLING_INTERVAL
    done
done
