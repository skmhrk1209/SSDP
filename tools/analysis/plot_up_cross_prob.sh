DATA_DIR=outputs/TMLR-toy-analysis/data/interval
FIGS_DIR=outputs/TMLR-toy-analysis/figures/interval

# NOTE: Proposition 3.1 alone on a single interval, against Monte Carlo as a function of the normalized sampling
# interval, from its sweep (`evaluate_up_cross_prob.sh`). The top of the absolute error is given by hand (the rule
# gives 10^0).
MAX_ABSOLUTE_ERROR=10

uv run python -m tools.analysis.plot_up_cross_prob \
    --input-dir $DATA_DIR \
    --output-dir $FIGS_DIR \
    --max-absolute-error $MAX_ABSOLUTE_ERROR
