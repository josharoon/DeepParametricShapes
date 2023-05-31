# random_search.py

import random
import argparse
from train_2d_roto import main as train

# Define the range of your hyperparameters
w_surface_range = [0.01, 5]
w_alignment_range = [0.01, 1]
w_template_range = [0.01, 20.0]

for i in range(20):  # Perform 20 iterations
    # Generate random hyperparameters within the specified ranges
    w_surface = random.uniform(*w_surface_range)
    w_alignment = random.uniform(*w_alignment_range)
    w_template = random.uniform(*w_template_range)

    # Create an argparse.Namespace object with the hyperparameters
    args = argparse.Namespace(
        checkpoint_dir="D:\DeepParametricShapes\scripts\checkpoints",
        num_epochs=1,
        w_surface=w_surface,
        w_alignment=w_alignment,
        w_template=w_template,
        eps=0.04,
        max_stroke=0.00,
        n_samples_per_curve=120,
        chamfer=False,
        simple_templates=False,
        sample_percentage=0.9,
        dataset_type="surgery",
        canvas_size=224,
        png_dir=r"D:\pyG\data\points\transform_test\combMatte",
        architectures="unet",
        resnet_depth=50,
        num_worker_threads=0,
        bs=4,
        lr=1e-1,
        cuda=True,
        # include all other arguments needed for training with their default values
    )

    # Call the training function with the generated hyperparameters
    try:
        print(
            f"Starting training run {i + 1} with w_surface={w_surface}, w_alignment={w_alignment}, w_template={w_template}")
        train(args)
        print(f"Finished training run {i + 1}")
    except Exception as e:
        print(f"Error during training run {i + 1}: {e}")
