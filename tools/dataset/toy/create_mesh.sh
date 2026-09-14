uv run python tools/dataset/toy/create_mesh.py \
    --output-file tools/dataset/toy/meshes/cuboids_k3_w0.05.ply \
    cuboid \
        --colors 1.0 0.25 0.5 0.25 0.5 1.0 1.0 0.75 0.25 \
        --radii 0.025 0.5 0.5 \
        --range -0.5 0.5 \
        --count 3 \

uv run python tools/dataset/toy/create_mesh.py \
    --output-file tools/dataset/toy/meshes/cuboids_k3_w0.025.ply \
    cuboid \
        --colors 1.0 0.25 0.5 0.25 0.5 1.0 1.0 0.75 0.25 \
        --radii 0.0125 0.5 0.5 \
        --range -0.5 0.5 \
        --count 3 \

uv run python tools/dataset/toy/create_mesh.py \
    --output-file tools/dataset/toy/meshes/cuboids_k3_w0.0125.ply \
    cuboid \
        --colors 1.0 0.25 0.5 0.25 0.5 1.0 1.0 0.75 0.25 \
        --radii 0.00625 0.5 0.5 \
        --range -0.5 0.5 \
        --count 3 \

uv run python tools/dataset/toy/create_mesh.py \
    --output-file tools/dataset/toy/meshes/cuboids_k2_w0.05.ply \
    cuboid \
        --colors 1.0 0.25 0.5 0.25 0.5 1.0 \
        --radii 0.025 0.5 0.5 \
        --range -0.25 0.25 \
        --count 2 \

uv run python tools/dataset/toy/create_mesh.py \
    --output-file tools/dataset/toy/meshes/cuboids_k2_w0.025.ply \
    cuboid \
        --colors 1.0 0.25 0.5 0.25 0.5 1.0 \
        --radii 0.0125 0.5 0.5 \
        --range -0.25 0.25 \
        --count 2 \

uv run python tools/dataset/toy/create_mesh.py \
    --output-file tools/dataset/toy/meshes/cuboids_k2_w0.0125.ply \
    cuboid \
        --colors 1.0 0.25 0.5 0.25 0.5 1.0 \
        --radii 0.00625 0.5 0.5 \
        --range -0.25 0.25 \
        --count 2 \

uv run python tools/dataset/toy/create_mesh.py \
    --output-file tools/dataset/toy/meshes/cuboids_k1_w0.05.ply \
    cuboid \
        --colors 1.0 0.25 0.5 \
        --radii 0.025 0.5 0.5 \
        --range -0.0 0.0 \
        --count 1 \

uv run python tools/dataset/toy/create_mesh.py \
    --output-file tools/dataset/toy/meshes/cuboids_k1_w0.025.ply \
    cuboid \
        --colors 1.0 0.25 0.5 \
        --radii 0.0125 0.5 0.5 \
        --range -0.0 0.0 \
        --count 1 \

uv run python tools/dataset/toy/create_mesh.py \
    --output-file tools/dataset/toy/meshes/cuboids_k1_w0.0125.ply \
    cuboid \
        --colors 1.0 0.25 0.5 \
        --radii 0.00625 0.5 0.5 \
        --range -0.0 0.0 \
        --count 1 \
