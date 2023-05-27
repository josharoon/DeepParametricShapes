import datetime
import os
import logging

import torch as th
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import ttools

from dps_2d import callbacks, datasets, templates
from dps_2d.interfaces import VectorizerInterface
from dps_2d.models_3chan import CurvesModel


LOG = logging.getLogger(__name__)

th.manual_seed(123)
th.backends.cudnn.deterministic = True
np.random.seed(123)


def _worker_init_fn(worker_id):
    np.random.seed(worker_id)

def create_dataset(dataset_type, *args, **kwargs):
    png_dir = kwargs.pop('png_dir', None)  # Extract the png_dir from kwargs
    if dataset_type == "fonts":
        data_path = r"D:\DeepParametricShapes\data\fonts"
        canvas_size = 128
        return datasets.FontsDataset(data_path,args[0],args[1], **kwargs),canvas_size
    elif dataset_type == "roto":
        data_path = r"D:\pyG\data\points\120423_183451_rev\processed"
        canvas_size = 224
        return datasets.RotoDataset(data_path,args[0],args[1], **kwargs), canvas_size
    elif dataset_type == "surgery":
        data_path=r"D:\pyG\data\points\transform_test\processed"
        canvas_size = 224
        if png_dir is None:
            use_png = False
        else:
            use_png = True
        return datasets.esDataset(data_path,args[0],args[1],use_png=use_png, png_root=png_dir, **kwargs), canvas_size
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")





def main(args):
    hparams = {
        "w_surface": args.w_surface,
        "w_alignment": args.w_alignment,
        "w_template": args.w_template,
        "eps": args.eps,
        "max_stroke": args.max_stroke,
        "n_samples_per_curve": args.n_samples_per_curve,
        "chamfer": args.chamfer,
        "simple_templates": args.simple_templates,
        "sample_percentage": args.sample_percentage,
        "dataset_type": args.dataset_type,
        "canvas_size": args.canvas_size,
        "learning_rate": args.lr,
        "batch_size": args.bs,
        "num_worker_threads": args.num_worker_threads
    }



    data, args.canvas_size = create_dataset(args.dataset_type, args.chamfer,
                                            args.n_samples_per_curve, png_dir=args.png_dir)




    dataloader = DataLoader(data, batch_size=args.bs, num_workers=args.num_worker_threads,
                            worker_init_fn=_worker_init_fn, shuffle=True, drop_last=True)

    # Subsampling validation dataset
    val_data, _ = create_dataset(args.dataset_type, args.chamfer, args.n_samples_per_curve, val=True)


    val_dataloader = DataLoader(val_data,batch_size=args.bs)



    model = CurvesModel(n_curves=sum(templates.topology),depth=args.resnet_depth,model_type=args.architectures)






    interface = VectorizerInterface(model, args.simple_templates, args.lr, args.max_stroke, args.canvas_size,
                                    args.chamfer, args.n_samples_per_curve, args.w_surface, args.w_template,
                                    args.w_alignment, cuda=args.cuda)


    checkpointer = ttools.Checkpointer(args.checkpoint_dir, model,optimizers=interface.optimizer)
    extras, meta = checkpointer.load_latest()
    print("Loaded checkpoint with extras: {},meta:{}".format(extras, meta))
    starting_epoch = extras['epoch'] if extras is not None else None

    keys = ['loss', 'chamferloss', 'templateloss'] if args.chamfer \
        else ['loss', 'surfaceloss', 'alignmentloss', 'templateloss']

    train_run_name = datetime.datetime.now().strftime('train-%m%d%y-%H%M%S')
    writer = SummaryWriter(os.path.join(args.checkpoint_dir, 'summaries',
                                        train_run_name), flush_secs=1)

    writer.add_hparams(hparams, {}, run_name=train_run_name)

    val_writer = SummaryWriter(os.path.join(args.checkpoint_dir, 'summaries',
                                            datetime.datetime.now().strftime('val-%m%d%y-%H%M%S')), flush_secs=1)

    trainer = ttools.Trainer(interface)
    trainer.add_callback(ttools.callbacks.TensorBoardLoggingCallback(keys=keys, writer=writer,
                                                                     val_writer=val_writer, frequency=5))
    trainer.add_callback(callbacks.InputImageCallback(writer=writer, val_writer=val_writer, frequency=100))
    trainer.add_callback(callbacks.InputDistanceFieldCallback(writer=writer, val_writer=val_writer, frequency=100))
    trainer.add_callback(callbacks.InputDistanceFieldCallbackComp(writer=writer, val_writer=val_writer, frequency=100))
    trainer.add_callback(callbacks.CurvesCallback(writer=writer, val_writer=val_writer, frequency=100))
    trainer.add_callback(callbacks.CurvesCallbackComp(writer=writer, val_writer=val_writer, frequency=100))
    if not args.chamfer:
        trainer.add_callback(callbacks.RenderingCallback(writer=writer, val_writer=val_writer, frequency=100))
        trainer.add_callback(callbacks.RenderingCompCallback(writer=writer, val_writer=val_writer, frequency=100))
    trainer.add_callback(ttools.callbacks.ProgressBarCallback(keys=keys))
    trainer.add_callback(ttools.callbacks.CheckpointingCallback(checkpointer, interval=None, max_epochs=2))
    print("Starting training")
    trainer.train(dataloader, num_epochs=args.num_epochs, val_dataloader=val_dataloader, starting_epoch=starting_epoch)


if __name__ == '__main__':
    parser = ttools.BasicArgumentParser()
    parser.add_argument("--w_surface", type=float, default=10)
    parser.add_argument("--w_alignment", type=float, default=0.0001)
    parser.add_argument("--w_template", type=float, default=0.1)#10
    parser.add_argument("--eps", type=float, default=0.04)
    parser.add_argument("--max_stroke", type=float, default=0.00)
    #parser.add_argument("--canvas_size", type=int, default=128)
    parser.add_argument("--n_samples_per_curve", type=int, default=120)
    parser.add_argument("--chamfer", default=False, dest='chamfer', action='store_true')
    parser.add_argument("--simple_templates", default=False, dest='simple_templates', action='store_true')
    parser.add_argument('--sample_percentage',
                      help='Percentage of the dataset to use for training and testing.',
                      type=float,
                      default=0.9)
    parser.add_argument("--dataset_type", type=str, choices=["fonts", "roto","surgery"], default="surgery",
                        help="Dataset type: 'fonts' or 'roto'")

    parser.add_argument("--canvas_size", type=int, default=224)
    parser.add_argument("--png_dir", type=str, default=None, help="path to the PNG images.")
    parser.add_argument("--architectures", type=str, choices=["unet", "resnet"], default="unet", help="Model architecture")
    parser.add_argument("--resnet_depth", type=int,choices=[18, 34, 50, 101, 152], default=50, help="ResNet depth")
    #parser.add_argument("--data", default=r"D:\DeepParametricShapes\data\fonts", help="path to the training data.")

    parser.set_defaults(num_worker_threads=0, bs=16, lr=1e-4)
    args = parser.parse_args()
    ttools.set_logger(args.debug)
    main(args)
