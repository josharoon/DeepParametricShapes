import argparse
import string
import os
import glob
import tempfile
import cairo
import torch
from PIL import Image, ImageOps
from torchvision.transforms.functional import to_tensor
import torch as th
import ttools
from simplification.cutil import simplify_coords
import pyGutils.viz
import utils
from dps_2d import  templates
#from dps_2d.models import CurvesModel
from dps_2d.models_3chan import CurvesModel
from dps_2d.viz import draw_curves, draw_curves_mask




def main(args):
    device = "cuda" if th.cuda.is_available() and args.cuda else "cpu"

    model = CurvesModel(sum(templates.topology))

    model.to(device)
    model.eval()
    checkpointer = ttools.Checkpointer(args.checkpoint_path, model)
    extras, _ = checkpointer.load_latest()
    if extras is not None:
        print(f"Loaded checkpoint (epoch {extras['epoch']})")
    else:
        print("Unable to load checkpoint")

    input_files = sorted(glob.glob(os.path.join(args.input_folder, args.file_pattern.format(name="testSeq_v01"))))


    if args.skip > 0:
        input_files = input_files[::args.skip + 1]
    curvesDict= {}
    for img_path in input_files:
        im = to_tensor(Image.open(img_path).resize((224, 224))).to(device)
        # z = th.zeros(len(string.ascii_uppercase)).scatter_(0,
        #                                                    th.tensor(string.ascii_uppercase.index(args.letter)), 1).to(
        #     device)

        # o is the index of 'p' or pupil in n_loops eye. 1 is the instrument 2 is both
        n_loops = args.n_loops
        template_idx = args.template_idx
        z = th.zeros(len(string.ascii_uppercase)).scatter_(0
                                                           , th.tensor(template_idx), 1).to(device)


        print(f"Processing image {img_path}")

        curves = model(im[None], z[None])['curves'][0].detach().cpu()

        render_image(args, curves, img_path, n_loops)
        render_image_mask(args, curves, img_path, n_loops)
        extract_RotoCurves(curves, n_loops, img_path)



def extract_RotoCurves(curves,n_loops,img_path):
    curves = th.cat(utils.unroll_curves(curves[None], templates.topology), dim=1).squeeze(0)
    curves = curves[:sum(templates.topology[:n_loops])]
    torch.save(curves,os.path.join(args.out,img_path.split(".")[-2]+".pt"))


def render_image(args, curves, img_path, n_loops):
    output_path = os.path.join(args.out, os.path.basename(img_path))
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 224, 224)
    ctx = cairo.Context(surface)
    ctx.scale(224, 224)
    ctx.rectangle(0, 0, 1, 1)
    ctx.set_source_rgb(1, 1, 1)
    ctx.fill()
    ctx.save()
    # Scale the input image to 224x224 using PIL
    input_image = Image.open(img_path)
    scaled_image = input_image.resize((224, 224), Image.ANTIALIAS)
    scaled_image_path = os.path.join(args.out, 'temp_scaled_image.png')
    scaled_image.save(scaled_image_path)
    im = cairo.ImageSurface.create_from_png(scaled_image_path)
    ctx.scale(1 / 224, 1 / 224)
    ctx.set_source_surface(im)
    ctx.paint()
    ctx.restore()
    draw_curves(curves, n_loops, ctx)
    surface.write_to_png(output_path)
    os.remove(scaled_image_path)
    print(f"Output saved to {output_path}")

def render_image_mask(args, curves, img_path, n_loops):
    output_path = os.path.join(args.out, os.path.basename(img_path.split(".")[-3]+"_mask."+img_path.split(".")[-2]+".png"))
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 224, 224)
    ctx = cairo.Context(surface)
    ctx.scale(224, 224)
    ctx.rectangle(0, 0, 1, 1)
    ctx.set_source_rgb(0, 0, 0)  # Set the background to black
    ctx.fill()
    draw_curves_mask(curves, n_loops, ctx)
    surface.write_to_png(output_path)
    print(f"Output saved to {output_path}")




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", type=str, default=r"D:\ThesisData\testImages\PreppedSequences\testSeq_v01")
    parser.add_argument("--file_pattern", type=str, default="{name}.*.png")
    parser.add_argument("--n_loops", type=int, default=1)
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument("--template_idx", type=int, default=0)
    parser.add_argument("--out", type=str, default=r"D:\ThesisData\testOuts", metavar="OUTPUT")
    parser.add_argument("--cuda", dest='cuda', action='store_true')
    parser.add_argument("--no_cuda", dest='cuda', action='store_false')
    parser.add_argument("--checkpoint_path", type=str, default=r"D:\DeepParametricShapes\scripts\checkpoints\training_end_w_surface_0.000_w_alignment_0.000_w_template_10.000_architecture_resnet_date_2023-06-30_16-01-10_resnet_depth_18_lr_0.001")
    parser.set_defaults(cuda=True)
    args = parser.parse_args()
    main(args)
