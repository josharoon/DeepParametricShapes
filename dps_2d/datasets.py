import os
import string

import numpy as np
from PIL import Image
import torch as th
from matplotlib import pyplot as plt
from torchvision.transforms.functional import to_tensor
from pyGutils.cubicCurvesUtil import convert_to_cubic_control_points
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
        distance_fields = th.flip(distance_fields,dims=[0])

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

        return {
            'fname': fname,
            'im': im,
            'distance_fields': distance_fields,
            'alignment_fields': alignment_fields,
            'occupancy_fields': occupancy_fields,
            'points': points,
            'letter_idx': 23, # string.ascii_uppercase.index(fname[0]),
            'n_loops': 1  # self.n_loops_dict[fname[0]]
        }

if __name__ == '__main__':

    root=r"D:\pyG\data\points\120423_183451_rev\processed"
    root2=r"D:\DeepParametricShapes\data\fonts"
    root3=r"D:\pyG\data\points\transform_test\processed"
    dataset1=RotoDataset(root=root,chamfer=False,n_samples_per_curve=100,val=False)
    dataset2=FontsDataset(root=root2,chamfer=False,n_samples_per_curve=100,val=False)
    dataset3=esDataset(root=root3,chamfer=False,n_samples_per_curve=100,val=False)
    data2=dataset2[0]
    data=dataset1[0]
    data3=dataset3[0]
    #plot images and distance fields from each data object
    fig,axs=plt.subplots(3,2)
    axs[0,0].imshow(data['im'].cpu().numpy().transpose(1,2,0))
    axs[0,1].imshow(data['distance_fields'].cpu().numpy())
    axs[1,0].imshow(data2['im'].cpu().numpy().transpose(1,2,0))
    axs[1,1].imshow(data2['distance_fields'].cpu().numpy())
    axs[2,0].imshow(data3['im'].cpu().numpy().transpose(1,2,0))
    axs[2,1].imshow(data3['distance_fields'].cpu().numpy())
    plt.show()
