DATA_DIR=outputs/TMLR-toy-analysis/data/approx-errors

# NOTE: The numbers behind the figures of the three experiments and the checks on the data of exp 1, from
# the data of `evaluate_up_cross_prob.sh`, `evaluate_approx_error.sh` and `evaluate_training_trajectory.sh`
# (merged by `merge_training_trajectory.sh`).
uv run python -m tools.analysis.summarize_approx_error \
    --data-dir $DATA_DIR \
    --output-file $DATA_DIR/summary.log
