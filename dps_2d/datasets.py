import concurrent
import glob
import json
import os
import re
import string
from pathlib import Path

import numpy as np
from PIL import Image
import torch as th
from matplotlib import pyplot as plt
from torch_geometric.data import Data
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm

from pyGutils.cubicCurvesUtil import convert_to_cubic_control_points, create_grid_points
from pyGutils.viz import plot_distance_field,plotCubicSpline
#from . import utils, templates
import utils, templates


class FontsDataset(th.utils.data.Dataset):
    def __init__(self, root, chamfer, n_samples_per_curve, val=False):
        self.root = root
        self.chamfer = chamfer
        self.n_samples_per_curve = n_samples_per_curve
        self.files = [f[:-4] for f in os.listdir(os.path.join(self.root, 'pngs')) if f.endswith('.png')]
        np.random.shuffle(self.files)
        cutoff = int(0.9*len(self.files))
        if val:
            self.files = self.files[cutoff:]
        else:
            self.files = self.files[:cutoff]
        self.n_loops_dict = templates.n_loops

    def __repr__(self):
        return "FontsDataset | {} entries".format(len(self))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        im = Image.open(os.path.join(self.root, 'pngs', fname + '.png')).convert('L')
        distance_fields = th.from_numpy(
                np.load(os.path.join(self.root, 'distances', fname + '.npy'))[31:-31,31:-31].astype(np.float32)) ** 2
        alignment_fields = utils.compute_alignment_fields(distance_fields)
        distance_fields = distance_fields[1:-1,1:-1]
        occupancy_fields = utils.compute_occupancy_fields(distance_fields)
        points = th.Tensor([])
        if self.chamfer:
            points = th.from_numpy(np.load(os.path.join(self.root, 'points', fname + '.npy')).astype(np.float32))
            points = points[:self.n_samples_per_curve*sum(templates.topology)]

        return {
            'fname': fname,
            'im': to_tensor(im),
            'distance_fields': distance_fields,
            'alignment_fields': alignment_fields,
            'occupancy_fields': occupancy_fields,
            'points': points,
            'letter_idx': string.ascii_uppercase.index(fname[0]),
            'n_loops': self.n_loops_dict[fname[0]]
        }


class RotoDataset(th.utils.data.Dataset):
    def __init__(self, root, chamfer, n_samples_per_curve, val=False):
        self.root = root
        self.chamfer = chamfer
        self.n_samples_per_curve = n_samples_per_curve
        self.filesIndicies=self.sortFiles()

        np.random.shuffle(self.filesIndicies)
        cutoff = int(0.9*len(self.filesIndicies))
        if val:
            self.files = self.filesIndicies[cutoff:]
        else:
            self.files = self.filesIndicies[:cutoff]
        self.n_loops_dict = templates.n_loops

    def __repr__(self):
        return "FontsDataset | {} entries".format(len(self))

    def __len__(self):
        return len(self.filesIndicies)

    def sortFiles(self):
        #grab the number before the pt e.g distance_field.00001.pt => 1 there will be duplicate numbers so produce a set
        #then sort the set and return the sorted list
        #get all files in the directory ending with pt
        files = [f for f in os.listdir(self.root) if f.endswith('.pt')]
        #remove any files that dont start with distance_field or spoints
        files = [f for f in files if f.startswith('distance_field') or f.startswith('spoints')]
        #now get the digits from the file name
        files = [f.split('.')[-2] for f in files]
        #remove duplicates
        files = list(set(files))
        #sort the list
        files.sort()
        return files


    def __getitem__(self, idx):
        fname = f"spoints.{self.filesIndicies[idx]}.png"
        dfname = f"distance_field.{self.filesIndicies[idx]}.pt"
        im = Image.open(os.path.join('\\'.join(self.root.split("\\")[:-1]),fname))

        #im = th.load(os.path.join(self.root, fname)).y
        #make a 1 channel image by taking the first channel (to match the fonts dataset)
        #im = im[0,:,:].unsqueeze(0)
        #im= Image.open(os.path.join(self.root,dfname)).convert('L')
        distance_fields = th.load(os.path.join(self.root, dfname))
        distance_fields = th.flip(distance_fields,dims=[0])

        #add padding of 2 to the distance fields
        distance_fields_pad= th.nn.functional.pad(distance_fields,(1,1,1,1))
        alignment_fields = utils.compute_alignment_fields(distance_fields_pad)
        #distance_fields = distance_fields[1:-1,1:-1]
        occupancy_fields = utils.compute_occupancy_fields(distance_fields)
        points = th.Tensor([])
        if self.chamfer:
            points = th.from_numpy(np.load(os.path.join(self.root, 'points', fname + '.npy')).astype(np.float32))
            points = points[:self.n_samples_per_curve*sum(templates.topology)]

        return {
            'fname': fname,
            'im': to_tensor(im),
            'distance_fields': distance_fields,
            'alignment_fields': alignment_fields,
            'occupancy_fields': occupancy_fields,
            'points': points,
            'letter_idx': 23, # string.ascii_uppercase.index(fname[0]),
            'n_loops': 1  # self.n_loops_dict[fname[0]]
        }


class esDataset(th.utils.data.Dataset):
    def __init__(self, root, chamfer, n_samples_per_curve, val=False):
        self.root = root
        self.chamfer = chamfer
        self.n_samples_per_curve = n_samples_per_curve
        self.filesIndicies=self.sortFiles()

        np.random.shuffle(self.filesIndicies)
        cutoff = int(0.9*len(self.filesIndicies))
        if val:
            self.files = self.filesIndicies[cutoff:]
        else:
            self.files = self.filesIndicies[:cutoff]
        self.n_loops_dict = templates.n_loops

    def __repr__(self):
        return "FontsDataset | {} entries".format(len(self))

    def __len__(self):
        return len(self.filesIndicies)

    def sortFiles(self):
        #grab the number before the pt e.g distance_field.00001.pt => 1 there will be duplicate numbers so produce a set
        #then sort the set and return the sorted list
        #get all files in the directory ending with pt
        files = [f for f in os.listdir(self.root) if f.endswith('.pt')]
        #remove any files that dont start with distance_field or spoints
        files = [f for f in files if f.startswith('distance_field') or f.startswith('spoints')]
        #now get the digits from the file name
        files = [f.split('.')[-2] for f in files]
        #remove duplicates
        files = list(set(files))
        #sort the list
        files.sort()
        return files


    def __getitem__(self, idx):
        fname = f"spoints.{self.filesIndicies[idx]}.pt"
        dfname = f"distance_field.{self.filesIndicies[idx]}.pt"
        # im = Image.open(os.path.join('\\'.join(self.root.split("\\")[:-1]),fname))
        # #resize the image to 224x224
        # im = im.resize((224,224))

        im = th.load(os.path.join(self.root, fname)).y
        #resize the image to 3x224x224


        distance_fields = th.load(os.path.join(self.root, dfname))
        #distance_fields = th.flip(distance_fields,dims=[0])
        #distance_fields = th.flip(distance_fields, (1, 0))

        #add padding of 2 to the distance fields
        #resize the distance fields to 224x224
        distance_fields = th.nn.functional.interpolate(distance_fields.unsqueeze(0).unsqueeze(0),size=(224,224),mode='bilinear').squeeze(0).squeeze(0)
        distance_fields_pad= th.nn.functional.pad(distance_fields,(1,1,1,1))
        alignment_fields = utils.compute_alignment_fields(distance_fields_pad)
        #distance_fields = distance_fields[1:-1,1:-1]
        occupancy_fields = utils.compute_occupancy_fields(distance_fields)
        points = th.Tensor([])
        if self.chamfer:
            points = th.from_numpy(np.load(os.path.join(self.root, 'points', fname + '.npy')).astype(np.float32))
            points = points[:self.n_samples_per_curve*sum(templates.topology)]
        # get number from file name
        frNum = int(fname.split('.')[-2])
        if(frNum>7800):
            letter_idx=1
        else:
            letter_idx=0
        return {
            'fname': fname,
            'im': im,
            'distance_fields': distance_fields,
            'alignment_fields': alignment_fields,
            'occupancy_fields': occupancy_fields,
            'points': letter_idx,
            'letter_idx': 0, # string.ascii_uppercase.index(fname[0]),
            'n_loops': 1  # self.n_loops_dict[fname[0]]
        }

# class FieldProcess():
#
#     def __init__(self,root,processed_dir,labelsFile):
#         self.root=root
#         self.processed_dir=processed_dir
#         self.labelsFile=labelsFile
#         self.labels=None
#         self.labelsDict=None
#         self.process()
#     def loadLabelsJson(self):
#         """load the json file containing the labels"""
#         with open(self.labels) as f:
#             data = json.loads(f.read())
#         return data
#
#     def process(self):
#         self.labels = Path(self.root).joinpath(self.labelsFile)
#         self.labelsDict = self.loadLabelsJson()
#
#         num_workers = 36  # Adjust this according to your system's resources
#
#         # Find all processed data files using glob
#         processed_files = set(glob.glob(os.path.join(self.processed_dir, '*.pt')))
#         pattern = re.compile(r'.*\.(\d+)\.pt')
#         processed_indices = {int(pattern.match(os.path.basename(f)).group(1)) for f in processed_files}
#         #we need to subtract 1 from each index because the indices in the json file start at 1
#         processed_indices={x-1 for x in processed_indices}
#         print(f"Found {len(processed_files)} processed files.")
#
#         print(f"Found {len(processed_indices)} processed indices.")
#
#         # Calculate unprocessed indices
#         print("Calculating unprocessed indices...")
#         unprocessed_indices = [i for i in range(len(self)) if i not in processed_indices]
#         print(f"Found {len(unprocessed_indices)} unprocessed files.")
#         with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
#             list(tqdm(executor.map(self.process_image, unprocessed_indices), total=len(unprocessed_indices),
#                       desc="Processing",
#                       unit="image"))
#
#     def process_image(self, index):
#         image, label = self.get(index)
#         image = self.normalize_image(image)
#         labelGraph = ShapeGraph(self.getPoints2DList(label))
#         #we scale everything to 224x224 for the model
#         original_height, original_width = image.shape[1:]
#         if original_height != 224 or original_width != 224:
#             # Calculate scaling factors
#             height_scale_factor = 224 / original_height
#             width_scale_factor = 224 / original_width
#             # Resize the image
#             image = th.nn.functional.interpolate(
#                 image.unsqueeze(0), size=(224, 224), mode="bilinear"
#             ).squeeze(0)
#             # Scale the control points
#             for point in labelGraph.points:
#                point.scalePoints(width_scale_factor,height_scale_factor)
#             labelGraph.x = []
#             labelGraph.createX()
#             labelGraph.createData()
#
#
#
#         data = Data(x=labelGraph.x, edge_index=labelGraph.edge_index, y=image)
#         control_points = convert_to_cubic_control_points(data.x[None, :]).to(th.float64)
#         grid_size = data.y.shape[1]
#         source_points = create_grid_points(grid_size, 0, 250, 0, 250)
#
#         device = 'cuda' if torch.cuda.is_available() else 'cpu'
#         control_points = control_points.to(device)
#         source_points = source_points.to(device)
#
#         distance_field = distance_to_curves(source_points, control_points, grid_size).view(grid_size, grid_size)
#         distance_field = th.flip(distance_field, (1,))
#         distance_field = self.normalize_distance_field(distance_field)
#
#         torch.save(data, self.processed_paths[index])
#         torch.save(distance_field, Path(self.processed_dir).joinpath(f'distance_field.{index+1:04d}.pt'))




def distToPng(dist,path):
    """convert distance field to png image"""
    dist = th.stack([dist, dist, dist]).permute(1, 2, 0)
    dist.permute(1, 2, 0)
    numpy_image = dist.cpu().numpy()

    rescaled_image = numpy_image * 255
    rescaled_image = rescaled_image.astype(np.uint8)
    image = Image.fromarray(rescaled_image)
    image.save(path)
   # print(f"saved {path}")
def distFieldsToPngSeq(folder):
    """convert all distance fields in folder to png images"""
    files=glob.glob(os.path.join(folder,"distance_field.*.pt"))
    for file in files:
        dist=th.load(file)
        distToPng(dist,file.replace(".pt",".png"))


if __name__ == '__main__':
    #distFieldsToPngSeq(r"D:\pyG\data\points\transform_test\processed")

    # root=r"D:\pyG\data\points\120423_183451_rev\processed"
    # root2=r"D:\DeepParametricShapes\data\fonts"
    root3=r"D:\pyG\data\points\transform_test\processed"
    # dataset1=RotoDataset(root=root,chamfer=False,n_samples_per_curve=100,val=False)
    # dataset2=FontsDataset(root=root2,chamfer=False,n_samples_per_curve=100,val=False)
    dataset3=esDataset(root=root3,chamfer=False,n_samples_per_curve=100,val=False)
    # data2=dataset2[0]
    # data=dataset1[0]
    data3=dataset3[0]
    # #plot images and distance fields from each data object
    # fig,axs=plt.subplots(4,2)
    # axs[0,0].imshow(data['im'].cpu().numpy().transpose(1,2,0))
    # axs[0,1].imshow(data['distance_fields'].cpu().numpy())
    # axs[1,0].imshow(data2['im'].cpu().numpy().transpose(1,2,0))
    # axs[1,1].imshow(data2['distance_fields'].cpu().numpy())
    # axs[2,0].imshow(data3['im'].cpu().numpy().transpose(1,2,0))
    # axs[2,1].imshow(data3['distance_fields'].cpu().numpy())
    #
    # # Blend images and distance fields
    #
    # dist=data3['distance_fields'].cpu()
    # #make 3 channels
    # dist=th.stack([dist,dist,dist])
    #
    # blend_im = 0.5*data3['im'].cpu() + 0.5*dist
    #
    #
    # axs[3,0].imshow(blend_im.numpy().transpose(1,2,0))
    # plt.show()



