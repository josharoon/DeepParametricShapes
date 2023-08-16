import json
import os

import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import glob

def calculate_iou(sequence1, sequence2):
    # Convert lists to tensors
    sequence1 = torch.stack(sequence1)
    sequence2 = torch.stack(sequence2)

    # Calculate intersection and union
    intersection = torch.sum(sequence1 * sequence2)
    union = torch.sum((sequence1 + sequence2) > 0)

    iou = intersection / union

    return iou.item()

def load_image_as_tensor(file, resize=False):
    # Load image file using PIL
    img = Image.open(file).convert('1') # Convert image to binary
    # Convert the image to a PyTorch tensor
    img_transforms = transforms.Compose([
        transforms.Resize((224, 224)) if resize else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
    ])
    img_tensor = img_transforms(img)
    return img_tensor

def main():
    gtStems=['pupil_matte','instrument_matte']
    ground_truth_path = r'D:\ThesisData\testset' # Supply the path to your groundtruth image sequence

    # Load groundtruth sequence

    #load a gt sequence with each stem
    ground_truth_sequences = []
    for stem in gtStems:
        print(f"Loading ground truth sequence for {stem}")
        ground_truth_sequences.append([load_image_as_tensor(file,resize=True) for file in sorted(glob.glob(os.path.join(ground_truth_path, stem, '*.*.png')))])


    with open(r'D:\DeepParametricShapes\scripts\checkpoints\rendered_directories.txt', 'r') as f:
        directories = f.read().splitlines()

    mean_iou_dict = {}
    for directory in directories:
        # get path 1 folder up from directory
        argsDirectory = os.path.dirname(directory)

        args_file = os.path.join(argsDirectory, "args.json")
        # Load args.json file
        if os.path.isfile(args_file):
            with open(args_file, 'r') as f:
                data = json.load(f)
            if data['template_idx'] == 0:
                gt=1
            else:
                if data['template_idx'] == 2:
                    gt=0

        else:   print(f"args.json file not found in {directory}"
                      f"Skipping this directory")

        # Load image sequence from directory
        print(f"Loading image sequence from {directory}")
        image_sequence = [load_image_as_tensor(file) for file in sorted(glob.glob(directory + '/*mask*.png'))]

        # Check to make sure we have the same number of images
        if len(image_sequence) != len(ground_truth_sequences[gt]):
            print(f"Image sequences in {directory} and ground truth have different lengths.")
            continue

        # Calculate and store mean IoU for this sequence
        mean_iou = calculate_iou(image_sequence, ground_truth_sequences[gt])
        mean_iou_dict[directory] = {
            'mean_iou': mean_iou,
            'num_epochs': data['num_epochs'],
            'w_surface': data['w_surface'],
            'w_alignment': data['w_alignment'],
            'w_template': data['w_template'],
            'w_chamfer': data['w_chamfer'],
            'w_curve': data['w_curve'],
            'png_dir': data['png_dir'],
            'template_idx': data['template_idx'],
            'resnet_depth': data['resnet_depth'],
            'im_fr_main_root': data['im_fr_main_root'],
            'loops': data['loops']
        }
        #save dictionary to file
    print(f"Mean IoU values for each image sequence: {mean_iou_dict}")
    dictPath = r'D:\DeepParametricShapes\scripts\checkpoints\mean_iou_dict.json'
    with open(r'%s' % dictPath, 'w') as f:
        print(f" saving mean IoU values for each image sequence: {mean_iou_dict} to {dictPath}")
        f.write(json.dumps(mean_iou_dict))

    # Now you have a dictionary of mean IoU values for each image sequence
    # You can process it further as needed

if __name__ == '__main__':
    main()
