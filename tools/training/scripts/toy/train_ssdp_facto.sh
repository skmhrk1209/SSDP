uv run python tools/training/launch.py -m \
    experiment=toy_ssdp \
    project_name=TMLR-toy-benchmark \
    method_name=ssdp-facto-base \
    experiment_name='${method_name}-${scheduler.params.scene_id}-${scheduler.params.random_seed}' \
    scheduler.walltime=01:15:00 \
    scheduler.params.scene_id=k1_w0.05,k2_w0.05,k3_w0.05,k1_w0.025,k2_w0.025,k3_w0.025,k1_w0.0125,k2_w0.0125,k3_w0.0125 \
    scheduler.params.deterministic_end_ratio=0.25 \
    scheduler.params.uniform_sampling=true \
    scheduler.params.random_seed=42,43,44 \
    timestamp=202604 \
