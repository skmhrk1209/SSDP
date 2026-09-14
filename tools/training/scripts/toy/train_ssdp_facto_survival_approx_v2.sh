uv run python tools/training/launch.py -m \
    experiment=toy_ssdp \
    project_name=TMLR-toy-ablation-survival-approx \
    method_name=ssdp-facto-base \
    experiment_name='${method_name}-${scheduler.params.scene_id}' \
    scheduler.walltime=01:30:00 \
    scheduler.params.scene_id=k1_w0.05,k2_w0.05,k3_w0.05,k1_w0.025,k2_w0.025,k3_w0.025,k1_w0.0125,k2_w0.0125,k3_w0.0125 \
    scheduler.params.deterministic_end_ratio=0.25 \
    scheduler.params.survival_approx_end_ratio=0.5 \
    scheduler.params.up_cross_approx_end_ratio=0.5 \
    scheduler.params.up_cross_anneal_end_ratio=0.75 \
    scheduler.params.uniform_sampling=true \
    timestamp=202604 \
