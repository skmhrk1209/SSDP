DATA_DIR=outputs/TMLR-toy-analysis/data/approx-errors
FIGS_DIR=outputs/TMLR-toy-analysis/figures/interval

# NOTE: Proposition 3.1 alone on a single interval, against Monte Carlo as a function of the length of the interval,
# on each axis from its own sweep (`evaluate_up_cross_prob.sh`), with the vertical range of an error common to
# the axes.
uv run python -m tools.analysis.plot_up_cross_prob \
    --input-dir $DATA_DIR/up_cross_prob \
    --output-dir $FIGS_DIR
