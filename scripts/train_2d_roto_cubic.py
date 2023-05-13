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
from dps_2d.models_3chan import CurvesModel,CurvesModelCubic


LOG = logging.getLogger(__name__)

th.manual_seed(123)
th.backends.cudnn.deterministic = True
np.random.seed(123)


def _worker_init_fn(worker_id):
    np.random.seed(worker_id)

def create_dataset(dataset_type, *args, **kwargs):
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
        return datasets.esDataset(data_path,args[0],args[1], **kwargs), canvas_size
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")




def main(args):
    data, args.canvas_size = create_dataset(args.dataset_type, args.chamfer,
                                            args.n_samples_per_curve)




    dataloader = DataLoader(data, batch_size=args.bs, num_workers=args.num_worker_threads,
                            worker_init_fn=_worker_init_fn, shuffle=True, drop_last=True)

    # Subsampling validation dataset
    val_data, _ = create_dataset(args.dataset_type, args.chamfer, args.n_samples_per_curve, val=True)


    val_dataloader = DataLoader(val_data)



    model = CurvesModelCubic(n_curves=sum(templates.topology))

    checkpointer = ttools.Checkpointer(args.checkpoint_dir, model)
    extras, meta = checkpointer.load_latest()
    starting_epoch = extras['epoch'] if extras is not None else None

    interface = VectorizerInterface(model, args.simple_templates, args.lr, args.max_stroke, args.canvas_size,
                                    args.chamfer, args.n_samples_per_curve, args.w_surface, args.w_template,
                                    args.w_alignment, cuda=args.cuda)

    keys = ['loss', 'chamferloss', 'templateloss'] if args.chamfer \
        else ['loss', 'surfaceloss', 'alignmentloss', 'templateloss']

    writer = SummaryWriter(os.path.join(args.checkpoint_dir, 'summaries',
                                        datetime.datetime.now().strftime('train-%m%d%y-%H%M%S')), flush_secs=1)
    val_writer = SummaryWriter(os.path.join(args.checkpoint_dir, 'summaries',
                                            datetime.datetime.now().strftime('val-%m%d%y-%H%M%S')), flush_secs=1)

    trainer = ttools.Trainer(interface)
    trainer.add_callback(ttools.callbacks.TensorBoardLoggingCallback(keys=keys, writer=writer,
                                                                     val_writer=val_writer, frequency=5))
    trainer.add_callback(callbacks.InputImageCallback(writer=writer, val_writer=val_writer, frequency=100))
    trainer.add_callback(callbacks.CurvesCallback(writer=writer, val_writer=val_writer, frequency=100))
    if not args.chamfer:
        trainer.add_callback(callbacks.RenderingCallback(writer=writer, val_writer=val_writer, frequency=100))
    trainer.add_callback(ttools.callbacks.ProgressBarCallback(keys=keys))
    trainer.add_callback(ttools.callbacks.CheckpointingCallback(checkpointer, interval=None, max_epochs=2))
    trainer.train(dataloader, num_epochs=args.num_epochs, val_dataloader=val_dataloader, starting_epoch=starting_epoch)


if __name__ == '__main__':
    parser = ttools.BasicArgumentParser()
    parser.add_argument("--w_surface", type=float, default=5)
    parser.add_argument("--w_alignment", type=float, default=0.05)
    parser.add_argument("--w_template", type=float, default=0.00)#10
    parser.add_argument("--eps", type=float, default=0.04)
    parser.add_argument("--max_stroke", type=float, default=0.04)
    #parser.add_argument("--canvas_size", type=int, default=128)
    parser.add_argument("--n_samples_per_curve", type=int, default=120)
    parser.add_argument("--chamfer", default=False, dest='chamfer', action='store_true')
    parser.add_argument("--simple_templates", default=True, dest='simple_templates', action='store_true')
    parser.add_argument('--sample_percentage',
                      help='Percentage of the dataset to use for training and testing.',
                      type=float,
                      default=0.9)
    parser.add_argument("--dataset_type", type=str, choices=["fonts", "roto","surgery"], default="surgery",
                        help="Dataset type: 'fonts' or 'roto'")

    parser.add_argument("--canvas_size", type=int, default=128)
    #parser.add_argument("--data", default=r"D:\DeepParametricShapes\data\fonts", help="path to the training data.")

    parser.set_defaults(num_worker_threads=0, bs=10, lr=1e-4)
    args = parser.parse_args()
    ttools.set_logger(args.debug)
    main(args)
