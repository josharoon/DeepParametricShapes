import os
import glob
import json
import argparse
import subprocess

def main():
    checkpoint_folder = r"D:\DeepParametricShapes\scripts\checkpoints"
    checkpoint_folders = glob.glob(os.path.join(checkpoint_folder, "training_end_*"))

    for folder in checkpoint_folders:
        args_file = os.path.join(folder, "args.json")

        if os.path.isfile(args_file):
            with open(args_file, 'r') as f:
                data = json.load(f)

            out_folder = os.path.join(folder, "results")
            os.makedirs(out_folder, exist_ok=True)

            # create the arguments for your testing script
            args = argparse.Namespace()
            args.input_folder = r"D:\ThesisData\testImages\PreppedSequences\testSeq_v01"
            args.file_pattern = "{name}.*.png"
            args.n_loops = 1
            args.skip = 0
            args.template_idx = data['template_idx']  # read from args.json
            args.out = out_folder
            args.cuda = True
            args.checkpoint_path = folder

            # modify this line with your testing script name and correct arguments
            subprocess.run(['python', 'your_testing_script.py',
                            '--input_folder', args.input_folder,
                            '--file_pattern', args.file_pattern,
                            '--n_loops', str(args.n_loops),
                            '--skip', str(args.skip),
                            '--template_idx', str(args.template_idx),
                            '--out', args.out,
                            '--cuda', str(args.cuda),
                            '--checkpoint_path', args.checkpoint_path])

if __name__ == '__main__':
    main()
