import datetime
import os
import logging
import json
import torch as th
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams
import ttools

from dps_2d import callbacks, datasets, templates
from dps_2d.interfaces import VectorizerInterface
from dps_2d.models_3chan import CurvesModel
from dps_2d.models import CurvesModel as FontCurvesModel
from templates import topology

LOG = logging.getLogger(__name__)

th.manual_seed(123)
th.backends.cudnn.deterministic = True
np.random.seed(123)

# th.autograd.set_detect_anomaly(True)
def _worker_init_fn(worker_id):
    np.random.seed(worker_id)

def save_args(args, filename):
    args_dict = vars(args)
    with open(filename, 'w') as f:
        json.dump(args_dict, f, indent=4)

def create_dataset(dataset_type, *args, **kwargs):
    png_dir = kwargs.pop('png_dir', None)  # Extract the png_dir from kwargs
    template_idx = kwargs.pop('template_idx', None)
    im_fr_main_root = kwargs.pop('im_fr_main_root', None)
    loops=kwargs.pop('loops',1)
    if dataset_type == "fonts":
        data_path = r"D:\DeepParametricShapes\data\fonts"
        canvas_size = 128
        return datasets.FontsDataset(data_path,args[0],args[1], **kwargs),canvas_size,datasets.FontsDataset(data_path,args[0],args[1],val=True, **kwargs)
    elif dataset_type == "roto":
        data_path = r"D:\ThesisData\data\points\rotoshapes\processed"
        canvas_size = 224
        return datasets.RotoDataset(data_path, args[0], args[1], **kwargs), canvas_size, datasets.RotoDataset(
            data_path, args[0], args[1], val=True, **kwargs)
    elif dataset_type == "surgery":
        data_path=r"D:\ThesisData\data\points\transform_test\processed"
        canvas_size = 224
        if png_dir is None:
            use_png = False
        else:
            use_png = True
        return datasets.esDataset(data_path,args[0],args[1],use_png=use_png, png_root=png_dir,im_fr_main_root=im_fr_main_root,template_idx=template_idx,loops=loops, **kwargs), canvas_size,datasets.esDataset(data_path,args[0],args[1],
            use_png=use_png, png_root=png_dir,val=True,im_fr_main_root=im_fr_main_root,template_idx=template_idx,loops=loops, **kwargs)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")





def main(args):
    hparamsDict = {
        "w_surface": args.w_surface,
        "w_alignment": args.w_alignment,
        "w_template": args.w_template,
        "w_chamfer": args.w_chamfer,
        "w_curve": args.w_curve,
        "dataset_type": args.dataset_type,
        "learning_rate": args.lr,
        "batch_size": args.bs,
        "architecture": args.architectures,
        "resnet_depth": args.resnet_depth,

    }
    hyperparams =hparams (hparamsDict, {"loss": 0,"templateloss":0,"chamferloss":0,"surfaceloss":0,"alignmentloss":0,"curve":0})


    data, args.canvas_size,val_data = create_dataset(args.dataset_type, args.chamfer,
                                            args.n_samples_per_curve, png_dir=args.png_dir,template_idx=args.template_idx,im_fr_main_root=args.im_fr_main_root, loops=args.loops)




    dataloader = DataLoader(data, batch_size=args.bs, num_workers=args.num_worker_threads,
                            worker_init_fn=_worker_init_fn, shuffle=True, drop_last=True)


    # val_data, _ = create_dataset(args.dataset_typgse, ar.chamfer, args.n_samples_per_curve, val=True)


    val_dataloader = DataLoader(val_data,batch_size=args.bs)


    if args.dataset_type == "fonts":
        model = FontCurvesModel(n_curves=sum(templates.topology))
    else:
        model = CurvesModel(n_curves=sum(templates.topology),depth=args.resnet_depth,model_type=args.architectures)






    interface = VectorizerInterface(model, args.simple_templates, args.lr, args.max_stroke, args.canvas_size,
                                    args.chamfer, args.n_samples_per_curve, args.w_surface, args.w_template,
                                    args.w_alignment,args.w_chamfer,args.w_curve,  cuda=args.cuda, dataset=args.dataset_type
                                    ,templates_topology=topology)


    checkpointer = ttools.Checkpointer(args.checkpoint_dir, model,optimizers=interface.optimizer)
    extras, meta = checkpointer.load_latest()
    if args.start_epoch is not None:
        starting_epoch = args.start_epoch
    else:
        print("Loaded checkpoint with extras: {},meta:{}".format(extras, meta))
        starting_epoch = extras['epoch'] if extras is not None else None

    keys = ['loss', 'chamferloss', 'templateloss','curveloss'] if args.chamfer \
        else ['loss', 'surfaceloss', 'alignmentloss', 'templateloss','chamferloss','curveloss']

    train_run_name = datetime.datetime.now().strftime('train-%m%d%y-%H%M%S')
    writer = SummaryWriter(os.path.join(args.checkpoint_dir, 'summaries',
                                        train_run_name), flush_secs=1)

    #writer.add_hparams(hparams, {}, run_name=train_run_name+"-hparams")

    val_writer = SummaryWriter(os.path.join(args.checkpoint_dir, 'summaries',
                                            datetime.datetime.now().strftime('val-%m%d%y-%H%M%S')), flush_secs=1)
    val_writer.file_writer.add_summary(hyperparams[0])
    val_writer.file_writer.add_summary(hyperparams[1])
    val_writer.file_writer.add_summary(hyperparams[2])


    #log argumments
    args_file_path = os.path.join(args.checkpoint_dir, 'summaries', train_run_name, 'args.json')
    save_args(args, args_file_path)


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
    trainer.add_callback(ttools.callbacks.CheckpointingCallback(checkpointer, interval=None, max_epochs=10))
    #hparam_callback = HyperparamLoggingCallback(writer, val_writer, keys=keys, hparams=hparams)
    trainer.add_callback(callbacks.HyperparamLoggingCallback(writer, val_writer, keys=keys, hparams=hparamsDict))

    print("Starting training")
    trainer.train(dataloader, num_epochs=args.num_epochs, val_dataloader=val_dataloader, starting_epoch=starting_epoch)


if __name__ == '__main__':
    parser = ttools.BasicArgumentParser()
    parser.add_argument("--w_surface", type=float, default=1)
    parser.add_argument("--w_alignment", type=float, default=0.01)
    parser.add_argument("--w_template", type=float, default=10)#10
    parser.add_argument("--w_chamfer", type=float, default=0.01)
    parser.add_argument("--w_curve", type=float, default=0)
    parser.add_argument("--eps", type=float, default=0.04)
    parser.add_argument("--max_stroke", type=float, default=0.04)
    #parser.add_argument("--canvas_size", type=int, default=128)
    parser.add_argument("--n_samples_per_curve", type=int, default=19)
    parser.add_argument("--chamfer", default=False, dest='chamfer', action='store_true')
    parser.add_argument("--simple_templates", default=False, dest='simple_templates', action='store_true')
    parser.add_argument('--sample_percentage',
                      help='Percentage of the dataset to use for training and testing.',
                      type=float,
                      default=0.95)
    parser.add_argument("--dataset_type", type=str, choices=["fonts", "roto","surgery"], default="surgery",
                        help="Dataset type: 'fonts' or 'roto'")

    parser.add_argument("--canvas_size", type=int, default=224)
    # parser.add_argument("--png_dir", type=str, default=None, help="path to the PNG images.")
    parser.add_argument("--png_dir", type=str, default=r"D:\ThesisData\data\points\transform_test\instrumentMatte", help="path to the PNG images.")
    parser.add_argument("--architectures", type=str, choices=["unet", "resnet"], default="resnet", help="Model architecture")
    parser.add_argument("--resnet_depth", type=int,choices=[18, 34, 50, 101, 152], default=18, help="ResNet depth")
    parser.add_argument("--start_epoch", type=int, default=None)
    parser.add_argument("--template_idx", type=int, default=0)
    parser.add_argument("--im_fr_main_root", type=bool, default=True)
    parser.add_argument("--loops", type=int, default=1)

    parser.set_defaults(num_worker_threads=0, bs=4, lr=1e-4)
    args = parser.parse_args()
    ttools.set_logger(args.debug)
    main(args)
