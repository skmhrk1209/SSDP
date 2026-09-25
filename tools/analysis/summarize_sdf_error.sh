DATA_DIR=outputs/TMLR-toy-analysis/data

# NOTE: The table of the reconstructed geometry from the data of `evaluate_sdf_error.sh`.
uv run python -m tools.analysis.summarize_sdf_error \
    --data-dir $DATA_DIR/sdf \
    --output-file $DATA_DIR/sdf/summary.log
