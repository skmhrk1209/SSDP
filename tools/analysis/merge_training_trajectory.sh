DATA_DIR=outputs/TMLR-toy-analysis/data/approx-errors

# NOTE: The slices of the checkpoints of each run (`evaluate_training_trajectory.sh`) are concatenated in the order
# of the steps into one file per run. The pattern of the slices excludes those of the `*_up` runs from the run
# without `_up`.
for SLICE_FILE in $DATA_DIR/trajectories/slices/*_0-10.json; do
    RUN=$(basename $SLICE_FILE _0-10.json)

    uv run python -c "
import json, sys
from pathlib import Path
records = [record for path in sorted(Path(sys.argv[1]).glob(sys.argv[2] + '_[0-9]*-[0-9]*.json')) for record in json.loads(path.read_text())]
records.sort(key=lambda record: record['config']['step'])
assert len(records) == 100 and len({record['config']['step'] for record in records}) == 100, len(records)
Path(sys.argv[3]).write_text(json.dumps(records, indent=4))
print(sys.argv[3])
" $DATA_DIR/trajectories/slices $RUN $DATA_DIR/trajectories/$RUN.json
done
