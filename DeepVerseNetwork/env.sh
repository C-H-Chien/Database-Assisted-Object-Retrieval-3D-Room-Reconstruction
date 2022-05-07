# System config
export OMP_NUM_THREADS=1
export NUM_WORKERS=4
export SEED=2021

# NOTE: Change the data config based on your detup!
# JSON files
export DATA_DIR=/home/chchien/BrownU/courses/Deep-Learning-2022/Final-Project/roca-datasets/Data/Dataset
# Resized images with intrinsics and poses
export IMAGE_ROOT=/home/chchien/BrownU/courses/Deep-Learning-2022/Final-Project/roca-datasets/Data/Images
# Depths and instances rendered over images
export RENDERING_ROOT=/home/chchien/BrownU/courses/Deep-Learning-2022/Final-Project/roca-datasets/Data/Rendering
# Scan2CAD Full Annotations
export FULL_ANNOT=/home/chchien/BrownU/courses/Deep-Learning-2022/Final-Project/roca-datasets/Data/Scan2CAD/full_annotations.json

# Model configurations
export RETRIEVAL_MODE=point_point+image+comp
export E2E=0
export NOC_WEIGHTS=1

# Train and test behavior
export EVAL_ONLY=1
export CHECKPOINT=/home/chchien/BrownU/courses/Deep-Learning-2022/Final-Project/roca-datasets/Data/model_best.pth  # "none"
export RESUME=0  # This means from last checkpoint
export OUTPUT_DIR=output
