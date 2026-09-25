uv run python tools/dataset/toy/render_mesh.py \
    --mesh-file tools/dataset/toy/meshes/cuboids_k1_w0.05.ply \
    --output-dir datasets/nerfstudio/sdfstudio/toy/cuboid/k1_w0.05 \

uv run python tools/dataset/toy/render_mesh.py \
    --mesh-file tools/dataset/toy/meshes/cuboids_k3_w0.05.ply \
    --output-dir datasets/nerfstudio/sdfstudio/toy/cuboid/k3_w0.05 \

uv run python tools/dataset/toy/render_mesh.py \
    --mesh-file tools/dataset/toy/meshes/cuboids_k1_w0.025.ply \
    --output-dir datasets/nerfstudio/sdfstudio/toy/cuboid/k1_w0.025 \

uv run python tools/dataset/toy/render_mesh.py \
    --mesh-file tools/dataset/toy/meshes/cuboids_k3_w0.025.ply \
    --output-dir datasets/nerfstudio/sdfstudio/toy/cuboid/k3_w0.025 \
