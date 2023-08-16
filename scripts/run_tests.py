import os
import glob
import json
import argparse
import subprocess

inRoot = r"D:\ThesisData\testset"
stems = ["pupil_matte", "rgb", "instrument_matte"]


def main():
    checkpoint_folder = r"D:\DeepParametricShapes\scripts\checkpoints"
    checkpoint_folders = glob.glob(os.path.join(checkpoint_folder, "training_end_*"))

    rendered_directories = []

    for folder in checkpoint_folders:
        args_file = os.path.join(folder, "args.json")

        if os.path.isfile(args_file):
            with open(args_file, 'r') as f:
                data = json.load(f)

            if data['im_fr_main_root'] == True:
                stem = stems[1]
            else:
                if data['png_dir'].split("\\")[-1] == "instrumentMatte":
                    stem = stems[2]
                else:
                    if data['png_dir'].split("\\")[-1] == "pupilMatte":
                        stem = stems[0]

            out_folder = os.path.join(folder, "results")
            os.makedirs(out_folder, exist_ok=True)
            rendered_directories.append(out_folder)
            args = argparse.Namespace()
            args.input_folder = os.path.join(inRoot, stem)
            args.file_pattern = "{name}.*.png"
            args.n_loops = data["loops"]
            args.skip = 0
            args.template_idx = data['template_idx']  # read from args.json
            args.out = out_folder
            args.cuda = True
            args.checkpoint_path = folder

            # modify this line with your testing script name and correct arguments
            subprocess.run(['python', r'D:\DeepParametricShapes\scripts\run_2d_seq.py',
                            '--input_folder', args.input_folder,
                            '--file_pattern', args.file_pattern,
                            '--n_loops', str(args.n_loops),
                            '--skip', str(args.skip),
                            '--template_idx', str(args.template_idx),
                            '--out', args.out,
                            '--checkpoint_path', args.checkpoint_path])
            print(f"Finished rendering for checkpoint {folder} with stem {stem}")

    # write list to file
    with open(r'D:\DeepParametricShapes\scripts\checkpoints\rendered_directories.txt', 'w') as f:
        for directory in rendered_directories:
            f.write("%s\n" % directory)


if __name__ == '__main__':
    main()

