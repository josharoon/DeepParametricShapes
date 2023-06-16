
import os
import pickle
import random
import argparse
import shutil
import glob
from train_2d_roto import main as train

TRAINING_PATH = r"D:\DeepParametricShapes\scripts\checkpoints"

def copy_and_rename_model(args, checkpoint_name="training_end.pth"):
    old_checkpoint_path = os.path.join(args.checkpoint_dir, checkpoint_name)
    new_checkpoint_name = f"{checkpoint_name.rstrip('.pth')}_w_surface_{format(args.w_surface, '.3f')}_w_alignment_{format(args.w_alignment, '.3f')}_w_template_{format(args.w_template, '.3f')}_architecture_{args.architectures}"
    if args.architectures == "resnet":
        new_checkpoint_name += f"_resnet_depth_{args.resnet_depth}"
    new_checkpoint_name += f"_lr_{format(args.lr, '.3f')}.pth"
    new_checkpoint_path = os.path.join(args.checkpoint_dir, new_checkpoint_name)
    shutil.copy(old_checkpoint_path, new_checkpoint_path)

    # create new directory for next training run
    new_directory_path = os.path.join(args.checkpoint_dir, new_checkpoint_name.rstrip('.pth'))
    os.makedirs(new_directory_path, exist_ok=True)
    # copy all .pth files into the new directory
    pth_files = glob.glob(os.path.join(args.checkpoint_dir, '*.pth'))
    for pth_file in pth_files:
        shutil.move(pth_file, new_directory_path)



architecture_options = {
    "resnet": [18, 34, 50, 101, 152],
    # "resnet": [50],
    # "unet": [None],  # None means that no depth parameters for Unet
}


# Define the range of your hyperparameters
w_surface_range = [0.0, 2]
w_alignment_range = [0.000, 0.2]
w_template_range = [0, 10]
lr_range = [1e-4, 1e-2]
chamfer_range = [0.0, 2]

SearchSize = 20

# Checkpoint file path
checkpoint_path = r"D:\DeepParametricShapes\checkpoints\random_search_checkpoint.pkl"

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
    w_chamfer = random.uniform(*chamfer_range)

    # Randomly choose an architecture and its corresponding depth
    architecture = random.choice(list(architecture_options.keys()))
    depth = random.choice(architecture_options[architecture])

    # Create an argparse.Namespace object with the hyperparameters

    args = argparse.Namespace(val_data=None,
                              config=None,
                              checkpoint_dir='D:\\DeepParametricShapes\\scripts\\checkpoints',
                              init_from=None,
                              lr=lr,
                              bs=16,
                              num_epochs=5,
                              num_worker_threads=0,
                              cuda=True,
                              server=None,
                              base_url='/',
                              env='main',
                              port=8097,
                              debug=False,
                              w_surface=w_surface,
                              w_alignment=w_alignment,
                              w_template=w_template,
                              w_chamfer=w_chamfer,
                              eps=0.04,
                              max_stroke=0.6,
                              n_samples_per_curve=19,
                              chamfer=False,
                              simple_templates=False,
                              sample_percentage=0.9,
                              dataset_type='surgery',
                              canvas_size=224, png_dir=r"D:\pyG\data\points\transform_test\instrumentMatte",
                              template_idx=1,
                              architectures=architecture,
                              resnet_depth=depth,
                              im_fr_main_root=True,
                              start_epoch=None,
                              start_pth=None)

    if args.pth_file_path:
        shutil.copy(args.pth_file_path, TRAINING_PATH)


    # Call the training function with the generated hyperparameters
    try:
        print(
            f"Starting training run {i + 1} with w_surface={w_surface}, w_alignment={w_alignment}, w_template={w_template}, lr={lr}, architecture={architecture}, depth={depth}")
        print(args)
        train(args)
        print(f"Finished training run {i + 1}")
        copy_and_rename_model(args)
        tested_hyperparameters.append((w_surface, w_alignment, w_template, lr, architecture, depth))
        state = {"i": i + 1, "tested_hyperparameters": tested_hyperparameters}
        with open(checkpoint_path, "wb") as file:
            pickle.dump(state, file)


    except Exception as e:
        print(f"Error during training run {i + 1}: {e}")
