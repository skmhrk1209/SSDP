uv run python tools/training/launch.py -m \
    experiment=toy_ssdp \
    project_name=TMLR-toy-analysis \
    method_name=ssdp-facto-half \
    experiment_name='${method_name}-${scheduler.params.scene_id}-NA-UP' \
    scheduler.walltime=01:00:00 \
    scheduler.params.scene_id=k1_w0.05,k3_w0.05,k1_w0.025,k3_w0.025 \
    scheduler.params.steps_per_save=100 \
    scheduler.params.save_only_latest=false \
    scheduler.params.survival_approx_end_ratio=1.0 \
    scheduler.params.up_cross_approx_end_ratio=0.0 \
    scheduler.params.up_cross_anneal_end_ratio=0.0 \
    timestamp=202604 \
