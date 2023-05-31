# random_search.py
import os
import pickle
import random
import argparse
import shutil

from train_2d_roto import main as train


def copy_and_rename_model(args, checkpoint_name="training_end.pth"):
    old_checkpoint_path = os.path.join(args.checkpoint_dir, checkpoint_name)
    new_checkpoint_name = f"{checkpoint_name.rstrip('.pth')}_w_surface_{args.w_surface}_w_alignment_{args.w_alignment}_w_template_{args.w_template}_architecture_{args.architectures}"
    if args.architectures == "resnet":
        new_checkpoint_name += f"_resnet_depth_{args.resnet_depth}"
    new_checkpoint_name += f"_lr_{args.lr}.pth"
    new_checkpoint_path = os.path.join(args.checkpoint_dir, new_checkpoint_name)
    shutil.copy(old_checkpoint_path, new_checkpoint_path)

architecture_options = {
    "resnet": [18, 34, 50, 101, 152],
    "unet": [None],  # None means that no depth parameters for Unet
}


# Define the range of your hyperparameters
w_surface_range = [0.01, 5]
w_alignment_range = [0.01, 1]
w_template_range = [0.01, 20.0]
lr_range = [1e-6, 1e-1]

SearchSize = 100

# Checkpoint file path
checkpoint_path = "checkpoints/random_search_checkpoint.pkl"

# Load the latest checkpoint if it exists
if os.path.exists(checkpoint_path):
    with open(checkpoint_path, "rb") as file:
        state = pickle.load(file)
    i_start = state["i"]
    tested_hyperparameters = state["tested_hyperparameters"]
else:
    i_start = 0
    tested_hyperparameters = []





for i in range(i_start, SearchSize):  # Perform 20 iterations
    # Generate random hyperparameters within the specified ranges
    w_surface = random.uniform(*w_surface_range)
    w_alignment = random.uniform(*w_alignment_range)
    w_template = random.uniform(*w_template_range)
    lr = random.uniform(*lr_range)

    # Randomly choose an architecture and its corresponding depth
    architecture = random.choice(list(architecture_options.keys()))
    depth = random.choice(architecture_options[architecture])

    # Create an argparse.Namespace object with the hyperparameters
    args = argparse.Namespace(
        checkpoint_dir="D:\DeepParametricShapes\scripts\checkpoints",
        num_epochs=5,
        w_surface=w_surface,
        w_alignment=w_alignment,
        w_template=w_template,
        eps=0.04,
        start_epoch=0,
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
        lr=lr,
        cuda=True,
        # include all other arguments needed for training with their default values
    )

    # Call the training function with the generated hyperparameters
    try:
        print(
            f"Starting training run {i + 1} with w_surface={w_surface}, w_alignment={w_alignment}, w_template={w_template}, lr={lr}, architecture={architecture}, depth={depth}")
        train(args)
        print(f"Finished training run {i + 1}")
        copy_and_rename_model(args)
        tested_hyperparameters.append((w_surface, w_alignment, w_template, lr, architecture, depth))
        state = {"i": i + 1, "tested_hyperparameters": tested_hyperparameters}
        with open(checkpoint_path, "wb") as file:
            pickle.dump(state, file)


    except Exception as e:
        print(f"Error during training run {i + 1}: {e}")
