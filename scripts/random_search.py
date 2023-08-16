import json
import os
import pickle
import random
import argparse
import shutil
import glob
from train_2d_roto import main as train
from datetime import datetime
TRAINING_PATH = r"D:\DeepParametricShapes\scripts\checkpoints"

def copy_and_rename_model(args, checkpoint_name="training_end.pth"):
    now = datetime.now()
    date_time_str = now.strftime("%Y-%m-%d_%H-%M-%S")
    old_checkpoint_path = os.path.join(args.checkpoint_dir, checkpoint_name)
    new_checkpoint_name = f"{checkpoint_name.rstrip('.pth')}_w_surface_{format(args.w_surface, '.3f')}_w_alignment_{format(args.w_alignment, '.3f')}_w_template_{format(args.w_template, '.3f')}_architecture_{args.architectures}_date_{date_time_str}"
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
    args_dict = vars(args)
    args_json_path = os.path.join(new_directory_path, "args.json")
    with open(args_json_path, "w") as f:
        json.dump(args_dict, f, indent=4)



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
curve_range=[0,1]

SearchSize = 20




# Define the path of the JSON file containing the predefined hyperparameters
predefined_hyperparameters_path = r"D:\DeepParametricShapes\scripts\hyperparameters.json"

# Load the predefined hyperparameters if the file exists
predefined_hyperparameters = []
if os.path.exists(predefined_hyperparameters_path):
    with open(predefined_hyperparameters_path, "r") as file:
        predefined_hyperparameters = json.load(file)
    print(f"Loaded {len(predefined_hyperparameters)} predefined hyperparameter sets")


def run_train(i, w_surface, w_alignment, w_template, lr, w_chamfer, w_curve, architecture, depth,
              png_dir, template_idx, im_fr_main_root,
              dataset_type, num_epochs,pth_file_path,loops):



    args = argparse.Namespace(val_data=None,
                              config=None,
                              checkpoint_dir='D:\\DeepParametricShapes\\scripts\\checkpoints',
                              init_from=None,
                              lr=lr,
                              bs=4,
                              num_epochs=num_epochs,
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
                              w_curve=w_curve,
                              eps=0.04,
                              max_stroke=0.6,
                              n_samples_per_curve=19,
                              chamfer=False,
                              simple_templates=False,
                              sample_percentage=0.9,
                              dataset_type=dataset_type,
                              canvas_size=224, png_dir=png_dir,
                              template_idx=template_idx,
                              architectures=architecture,
                              resnet_depth=depth,
                              im_fr_main_root=im_fr_main_root,
                              start_epoch=None,
                              loops=loops)
    if pth_file_path:
        shutil.copy(pth_file_path, TRAINING_PATH)
    # Call the training function with the generated hyperparameters
    try:
        print(
            f"Starting training run {i + 1} with w_surface={w_surface}, w_alignment={w_alignment}, w_template={w_template}, lr={lr}, architecture={architecture}, depth={depth}")
        print(args)
        train(args)
        print(f"Finished training run {i + 1}")
        copy_and_rename_model(args)



    except Exception as e:
        print(f"Error during training run {i + 1}: {e}")


if predefined_hyperparameters:
    # Use predefined hyperparameters
    SearchSize = len(predefined_hyperparameters)  # Adjust the search size to the number of predefined sets
    for i in range(0, SearchSize):
        hyperparameters = predefined_hyperparameters[i]
        if hyperparameters["ran"]==False:
            run_train(i, hyperparameters["w_surface"], hyperparameters["w_alignment"], hyperparameters["w_template"], hyperparameters["lr"], hyperparameters["w_chamfer"], hyperparameters["w_curve"], hyperparameters["architecture"], hyperparameters["depth"],hyperparameters["png_dir"], hyperparameters["template_idx"], hyperparameters["im_fr_main_root"],
                  hyperparameters["dataset_type"], hyperparameters["num_epochs"],hyperparameters["pth_file_path"],hyperparameters["loops"])
        else:
            print(f"Skipping predefined hyperparameters {i} because they were already used")
        # Mark the hyperparameters as used
        predefined_hyperparameters[i]["ran"] = True
        # Save the updated list of predefined hyperparameters
        print(f"Saving updated list of predefined hyperparameters to {predefined_hyperparameters_path}")
        with open(predefined_hyperparameters_path, "w") as file:
            json.dump(predefined_hyperparameters, file, indent=4)


else:
    # Perform random searcch
    for i in range(0, SearchSize):
        w_surface = random.uniform(*w_surface_range)
        w_alignment = random.uniform(*w_alignment_range)
        w_template = random.uniform(*w_template_range)
        lr = random.uniform(*lr_range)
        w_chamfer = random.uniform(*chamfer_range)
        w_curve = random.uniform(*curve_range)
        architecture = random.choice(list(architecture_options.keys()))
        depth = random.choice(architecture_options[architecture])
        run_train(i, w_surface, w_alignment, w_template, lr, w_chamfer, w_curve, architecture, depth,png_dir=None, template_idx=None, im_fr_main_root=None,
              dataset_type=None, num_epochs=None)

