import concurrent
import glob
import json
import os
import re
import shutil
import string
from pathlib import Path

import bezier
import torch
from matplotlib.pyplot import imshow
from torchvision.io import read_image
import numpy as np
from PIL import Image
import torch as th
from matplotlib import pyplot as plt
from torch_geometric.data import Data
from torchvision.transforms import transforms
from torchvision.transforms.functional import to_tensor
from torch_geometric.data import  Dataset

from pyGutils.viz import  plotCubicSpline
from pyGutils.cubicCurvesUtil import *
from pyGutils.viz import plot_distance_field, plotCubicSpline
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from pyGutils.cubicCurvesUtil import convert_to_cubic_control_points, create_grid_points
from pyGutils.viz import plot_distance_field,plotCubicSpline,visualize_vector_field

import dps_2d.utils, dps_2d.templates as templates


class point2D:
    def __init__(self, vertex, lftTang=None, rhtTang=None):
        self.vertex = vertex[:2]
        if lftTang is None:
            self.lftTang = None
        else:
            self.lftTang = lftTang[:2]
        if rhtTang is None:
            self.rhtTang = None
        else:
            self.rhtTang = rhtTang[:2]

    def normalize(self, max_value=224.0):
        self.vertex = [coord / max_value for coord in self.vertex]
        if self.lftTang is not None:
            self.lftTang = [coord / max_value for coord in self.lftTang]
        if self.rhtTang is not None:
            self.rhtTang = [coord / max_value for coord in self.rhtTang]

    def scalePoints(self, width,height):
        self.vertex[0]*=width
        self.vertex[1]*=height
        if self.lftTang is not None:
            self.lftTang[0]*=width
            self.lftTang[1]*=height
        if self.rhtTang is not None:
            self.rhtTang[0]*=width
            self.rhtTang[1]*=height



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
        alignment_fields = dps_2d.utils.compute_alignment_fields(distance_fields)
        distance_fields = distance_fields[1:-1,1:-1]
        occupancy_fields = dps_2d.utils.compute_occupancy_fields(distance_fields)
        points = th.Tensor([])
        try:
            points = th.from_numpy(np.load(os.path.join(self.root, 'points', fname + '.npy')).astype(np.float32))
            points = points[:self.n_samples_per_curve * sum(templates.topology)]
            mean_value = torch.nanmean(points)
            points = torch.where(torch.isnan(points), mean_value, points)

        except:
            pass
            # print("Error loading points for {}".format(fname))
        if self.chamfer:
            points = th.from_numpy(np.load(os.path.join(self.root, 'points', fname + '.npy')).astype(np.float32))
            points = points[:self.n_samples_per_curve*sum(templates.topology)]
            mean_value = torch.nanmean(points)
            points = torch.where(torch.isnan(points), mean_value, points)

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


# class RotoDataset(th.utils.data.Dataset):
#     def __init__(self, root, chamfer, n_samples_per_curve, val=False,template_idx=0):
#         self.root = root
#         self.chamfer = chamfer
#         self.n_samples_per_curve = n_samples_per_curve
#         self.nloops = 1
#         self.template_idx = template_idx
#         self.filesIndicies=self.sortFiles()
#
#
#         np.random.shuffle(self.filesIndicies)
#         cutoff = int(0.9*len(self.filesIndicies))
#         if val:
#             self.files = self.filesIndicies[cutoff:]
#         else:
#             self.files = self.filesIndicies[:cutoff]
#         self.n_loops_dict = templates.n_loops
#
#     def __repr__(self):
#         return "FontsDataset | {} entries".format(len(self))
#
#     def __len__(self):
#         return len(self.filesIndicies)
#
#     def sortFiles(self):
#         #grab the number before the pt e.g distance_field.00001.pt => 1 there will be duplicate numbers so produce a set
#         #then sort the set and return the sorted list
#         #get all files in the directory ending with pt
#         files = [f for f in os.listdir(self.root) if f.endswith('.pt')]
#         #remove any files that dont start with distance_field or spoints
#         files = [f for f in files if f.startswith('distance_field') or f.startswith('spoints')]
#         #now get the digits from the file name
#         files = [f.split('.')[-2] for f in files]
#         #remove duplicates
#         files = list(set(files))
#         #sort the list
#         files.sort()
#         return files
#
#
#     def __getitem__(self, idx):
#         fname = f"spoints.{self.filesIndicies[idx]}.png"
#         dfname = f"distance_field.{self.filesIndicies[idx]}.pt"
#         pfname = f"sampled_points_{self.filesIndicies[idx]}.npy"
#         im = Image.open(os.path.join('\\'.join(self.root.split("\\")[:-1]),fname))
#         distance_fields = th.load(os.path.join(self.root, dfname))**2
#         distance_fields = th.flip(distance_fields, (0,))[55:-55,55:-55] #we introduce a crop to match workflow from DPS
#         alignment_fields = utils.compute_alignment_fields(distance_fields)
#         distance_fields = distance_fields[1:-1, 1:-1]
#         occupancy_fields = utils.compute_occupancy_fields(distance_fields)
#         points = th.Tensor([])
#         if self.chamfer:
#             points = th.from_numpy(np.load(os.path.join(self.root, 'points', fname + '.npy')).astype(np.float32))
#             points = points[:self.n_samples_per_curve*sum(templates.topology)]
#         points = th.from_numpy(np.load(os.path.join(self.root , pfname)).astype(np.float32))
#         npoints = points.shape[0]
#         # desired_npoints = 437  # replace with the desired number of points
#         maxPoints = self.n_samples_per_curve * sum(templates.topology)
#         indices = np.linspace(0, npoints - 1, maxPoints,
#                               dtype=int)  # generates evenly spaced desired_npoints between 0 and npoints-1
#
#         # Convert points tensor to numpy, perform indexing, and convert back to tensor
#         new_points = torch.from_numpy(points.numpy()[indices])
#
#         # points = points[:maxPoints]#points_cutoff
#         mean_value = torch.nanmean(new_points)
#         points = torch.where(torch.isnan(new_points), mean_value, new_points)
#         letter_idx = self.template_idx
#         return {
#             'fname': fname,
#             'im': im,
#             'distance_fields': distance_fields,
#             'alignment_fields': alignment_fields,
#             'occupancy_fields': occupancy_fields,
#             'points': points,
#             'letter_idx': letter_idx,  # string.ascii_uppercase.index(fname[0]),
#             'n_loops': self.nloops,  # self.n_loops_dict[fname[0]]
#             'targetCurves': None
#         }

class esDataset(th.utils.data.Dataset):
    def __init__(self, root, chamfer, n_samples_per_curve, png_root=None, use_png=False, val=False, im_fr_main_root=False, template_idx=7,
                 sample=0.9, loops=1,pointLoss=False):
        self.nloops = loops
        self.template_idx = template_idx
        self.root = root
        self.chamfer = chamfer
        self.png_root = png_root
        self.use_png = use_png
        self.im_fr_main_root = im_fr_main_root
        self.n_samples_per_curve = n_samples_per_curve
        self.filesIndicies=self.sortFiles()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()

        ])
        train_files, val_files = train_test_split(self.filesIndicies, test_size=0.1, random_state=42)
        self.files = val_files if val else train_files
        self.filesIndicies = self.files
        self.n_loops_dict = templates.n_loops_eye
        self.pointLoss=pointLoss

    def __repr__(self):
        return "esDataset | {} entries".format(len(self))

    def __len__(self):
        return len(self.files)

    def sortFiles(self):
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

        if self.use_png: #png indices start from 0 instead of 1 so subtract 1
            fname = f"spoints.{str(int(self.filesIndicies[idx])).zfill(4)}.png"
            if self.im_fr_main_root:
                fname = f"spoints.{str(int(self.filesIndicies[idx])-1).zfill(4)}.png"
                im_path = os.path.join("\\".join((self.root.split("\\")[:-1])), fname)
            else:
                im_path = os.path.join(self.png_root, fname)
            im = Image.open(im_path)
            im = self.transform(im)




        else:
            fname = f"spoints.{self.filesIndicies[idx]}.pt"
            im = th.load(os.path.join(self.root, fname))#.y

        dfname = f"distance_field.{self.filesIndicies[idx]}.pt"
        pfname = f"sampled_points_{self.filesIndicies[idx]}.npy"
        cfname = f"curves{self.filesIndicies[idx]}.npy"



        # distance_fields = th.load(os.path.join(self.root, dfname))**2
        distance_fields = th.load(os.path.join(self.png_root+"\\processed\\", dfname))**2
        distance_fields = th.flip(distance_fields, (0,))[55:-55,55:-55] #we introduce a crop to match workflow from DPS
        alignment_fields = dps_2d.utils.compute_alignment_fields(distance_fields)
        distance_fields = distance_fields[1:-1, 1:-1]
        occupancy_fields = dps_2d.utils.compute_occupancy_fields(distance_fields)
        points = th.Tensor([])
        if self.pointLoss:
            if not self.use_png:
                targetCurves=th.from_numpy(np.load(os.path.join(self.root, cfname)).astype(np.float32))
            else:
                targetCurves = th.from_numpy(
                    np.load(os.path.join(self.png_root + "\\processed\\", cfname)).astype(np.float32))
        maxPoints = self.n_samples_per_curve * sum(templates.topology)
        try:
            if not self.use_png:
                points = th.from_numpy(np.load(os.path.join(self.root, pfname)).astype(np.float32))
            else:
                points = th.from_numpy(np.load(os.path.join(self.png_root+"\\processed\\", pfname)).astype(np.float32))

            #shuffle points
            # points = self.shuffle_points(points)
            #instead of shuffling points we remove points at even intervals so we can also calculate point matching loss.
            step_size = points.shape[0] // maxPoints
            npoints = points.shape[0]
            # desired_npoints = 437  # replace with the desired number of points

            indices = np.linspace(0, npoints - 1, maxPoints,
                                  dtype=int)  # generates evenly spaced desired_npoints between 0 and npoints-1

            # Convert points tensor to numpy, perform indexing, and convert back to tensor
            new_points = torch.from_numpy(points.numpy()[indices])

            # points = points[:maxPoints]#points_cutoff
            mean_value = torch.nanmean(new_points)
            points = torch.where(torch.isnan(new_points), mean_value, new_points)
        except:
            pass
            # print("Error loading points for {}".format(fname))
        if self.chamfer:
            points = th.from_numpy(np.load(os.path.join(self.root, pfname)).astype(np.float32))
            # points = points[torch.randperm(points.shape[0])]
            points = points[:maxPoints]
        # get number from file name
        # frNum = int(fname.split('.')[-2])
        # if(frNum>7800):
        #     letter_idx=1
        # else:
        #     letter_idx=0
        letter_idx= self.template_idx  # multi curve template
        return {
            'fname': fname,
            'im': im,
            'distance_fields': distance_fields,
            'alignment_fields': alignment_fields,
            'occupancy_fields': occupancy_fields,
            'points': points,
            'letter_idx': letter_idx, # string.ascii_uppercase.index(fname[0]),
            'n_loops': self.nloops,  # self.n_loops_dict[fname[0]]
            # 'targetCurves':targetCurves
        }

    def shuffle_points(self, points):
        npoints = points.shape[0]
        shuffle_indices = torch.randperm(npoints)
        points = points[shuffle_indices]
        return points
class RotoDataset(esDataset):
    def __init__(self, root, chamfer, n_samples_per_curve, val=False, template_idx=0,
                 sample=0.9, loops=1,pointLoss=True):
        super().__init__(root, chamfer, n_samples_per_curve, png_root=None, use_png=False, val=val, im_fr_main_root=False, template_idx=template_idx,
                 sample=sample, loops=loops,pointLoss=pointLoss)


    def __getitem__(self, idx):
            fname = f"spoints.{self.filesIndicies[idx]}.pt"
            try:
                im = th.load(os.path.join(self.root, fname))
            except RuntimeError as e:
                print(f"Error loading file {fname}: {str(e)}")
                im=None
            dfname = f"distance_field.{self.filesIndicies[idx]}.pt"
            pfname = f"sampled_points_{self.filesIndicies[idx]}.npy"
            distance_fields = th.load(os.path.join(self.root, dfname)) ** 2
            distance_fields = th.flip(distance_fields, (0,))[55:-55,55:-55]  # we introduce a crop to match workflow fromPS
            alignment_fields = dps_2d.utils.compute_alignment_fields(distance_fields)
            distance_fields = distance_fields[1:-1, 1:-1]
            occupancy_fields = dps_2d.utils.compute_occupancy_fields(distance_fields)
            points = th.Tensor([])
            maxPoints = self.n_samples_per_curve * sum(templates.topology)
            points = th.from_numpy(np.load(os.path.join(self.root, pfname)).astype(np.float32))


            npoints = points.shape[0]
            # desired_npoints = 437  # replace with the desired number of points
            indices = np.linspace(0, npoints - 1, maxPoints,
                                  dtype=int)  # generates evenly spaced desired_npoints between 0 and npoints-1
            # Convert points tensor to numpy, perform indexing, and convert back to tensor
            new_points = torch.from_numpy(points.numpy()[indices])
            # points = points[:maxPoints]#points_cutoff
            mean_value = torch.nanmean(new_points)
            points = torch.where(torch.isnan(new_points), mean_value, new_points)

                # print("Error loading points for {}".format(fname))
            if self.chamfer:
                points = th.from_numpy(np.load(os.path.join(self.root, pfname)).astype(np.float32))
                # points = points[torch.randperm(points.shape[0])]
                points = points[:maxPoints]
            # get number from file name
            # frNum = int(fname.split('.')[-2])
            # if(frNum>7800):
            #     letter_idx=1
            # else:
            #     letter_idx=0
            letter_idx = self.template_idx  # multi curve template
            return {
                'fname': fname,
                'im': im,
                'distance_fields': distance_fields,
                'alignment_fields': alignment_fields,
                'occupancy_fields': occupancy_fields,
                'points': points,
                'letter_idx': letter_idx,  # string.ascii_uppercase.index(fname[0]),
                'n_loops': self.nloops,  # self.n_loops_dict[fname[0]]

            }


class MultiFieldProcess(Dataset):

    def __init__(self, root, labelsFiles=None, transform=None, pre_transform=None, pre_filter=None, preprocess=True,
                 proc_path="processed", grid_expansion_ratio=1.5):
        self.grid_epansion_ratio = grid_expansion_ratio
        self.processed_subpath = proc_path
        self.root=root
        # self.processed_dir=os.path.join(self.root,self.processed_subpath)
        self.labelsFiles=labelsFiles
        self.labels=None
        self.labelsDictList=[]
        super().__init__(root, transform, pre_transform, pre_filter)
    def loadLabelsJson(self):
        """load the json file containing the labels"""
        with open(self.labels) as f:
            data = json.loads(f.read())
        return data


    @property
    def raw_file_names(self):
        #get a list of all the files in the root directory with the extension .png
        paths=[f for f in Path(self.root).iterdir() if f.suffix=='.png']
        #makes sure the files are sorted by name by number e.g. spoints.0001.png, spoints.0002.png take into account varying number of digits
        paths.sort(key=lambda f: int(''.join(filter(str.isdigit, f.name))))
        return paths



    @property
    def processed_file_names(self):
        #processed files are saved as .pt files in the processed directory and
        #are named spoints.0001.pt, spoints.0002.pt etc
        return [f'spoints.{i:04d}.pt' for i in range(1,len(self.raw_file_names)+1)]

    def len(self):
        """len is number of entries in the labelsDict"""
        return len(self.processed_file_names)

    def Shape2Point2D(self,shape):
        """convert an json point to a point2D object e.g {'center': { 362, 240, 1 }, 'leftTangent': { 9, 4, 1 }, 'rightTangent': { 9, -2, 1 }"""
        #if string change property names to double quotes and convert back to dict
        #comvert shape string to dict
        if isinstance(shape,str):
            shape=shape.replace("'",'"')
            shape=json.loads(shape)
        #remove { from values convert to list of floats e.g. { 362, 240, 1 } to [362.0, 240.0, 1.0]
        shape['center']=shape['center'].replace('{','').replace('}','').split(',')
        shape['center']=[float(x) for x in shape['center']]
        shape['leftTangent']=shape['leftTangent'].replace('{','').replace('}','').split(',')
        shape['leftTangent']=[float(x) for x in shape['leftTangent']]
        shape['rightTangent']=shape['rightTangent'].replace('{','').replace('}','').split(',')
        shape['rightTangent']=[float(x) for x in shape['rightTangent']]
        #convert to point2D object
        return point2D(shape['center'],shape['leftTangent'],shape['rightTangent'])
    def getPoints2DList(self, pointList):
        """convert a list of json points to a list of point2D objects"""
        points2D = [self.Shape2Point2D(shape) for shape in pointList]
        # for point in points2D:
        #     point.normalize
        #add center point to tangent handels in each point
        for point in points2D:
            point.lftTang[0]=point.lftTang[0]=point.lftTang[0]+point.vertex[0]
            point.lftTang[1]=point.lftTang[1]=point.lftTang[1]+point.vertex[1]
            point.rhtTang[0]=point.rhtTang[0]=point.rhtTang[0]+point.vertex[0]
            point.rhtTang[1]=point.rhtTang[1]=point.rhtTang[1]+point.vertex[1]

        return points2D

    def process(self):
        #if lables isn't list make it a list
        if not isinstance(self.labelsFiles, list):
            self.labelsFiles = [self.labelsFiles]
        for labelFile in self.labelsFiles:
            #get the path to the labels file
            self.labels = Path(self.root).joinpath(labelFile)
            self.labelsDictList.append(self.loadLabelsJson())



        num_workers = 1  # Adjust this according to your system's resources

        # Find all processed data files using glob
        processed_files = set(glob.glob(os.path.join(self.processed_dir, '*.pt')))
        pattern = re.compile(r'.*\.(\d+)\.pt')
        processed_indices = {int(pattern.match(os.path.basename(f)).group(1)) for f in processed_files if pattern.match(os.path.basename(f))}
        #we need to subtract 1 from each index because the indices in the json file start at 1
        processed_indices={x-1 for x in processed_indices}
        print(f"Found {len(processed_files)} processed files.")

        print(f"Found {len(processed_indices)} processed indices.")

        # Calculate unprocessed indices
        print("Calculating unprocessed indices...")
        unprocessed_indices = [i for i in range(len(self)) if i not in processed_indices]
        print(f"Found {len(unprocessed_indices)} unprocessed files.")
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            list(tqdm(executor.map(self.process_frame, unprocessed_indices), total=len(unprocessed_indices),
                      desc="Processing",
                      unit="image"))
    def get(self, idx):
        """get the data from the .pt files in the processed directory if file exists otherwise get the data from the raw directory"""
        # if Path(self.processed_dir).joinpath(self.processed_file_names[idx]).exists():
        #     data = torch.load(self.processed_paths[idx])
        #
        #     df = torch.load(Path(self.processed_dir).joinpath(f'distance_field.{idx+1:04d}.pt'))
        #     data, df
        # else:
        #     # get the image
        image = read_image(self.raw_paths[idx])
        # get the label
        labels=[]
        for labeldict in self.labelsDictList:
            labels.append(labeldict[str(idx + 1)])
        return image, labels


    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, self.processed_subpath)

    def normalize_image(self, image):
        return image / 255.0

    def normalize_distance_field(self, distance_field):
        d_max = torch.max(distance_field)
        return distance_field / d_max


    def process_frame(self, index):
        canvas_size = 224
        grid_size = canvas_size
        image, labels = self.get(index)
        image = self.normalize_image(image)
        curvepoints=[self.getPoints2DList(label) for label in labels]
        # add items in curvepoints list
        allpoints=[]
        for curve in curvepoints:
                allpoints+=curve
        #we scale everything to 224x224 for the model
        original_height, original_width = image.shape[1:]
        if original_height != canvas_size or original_width != canvas_size:
            # Calculate scaling factors
            height_scale_factor = canvas_size / original_height
            width_scale_factor = canvas_size / original_width
            # Resize the image
            image = th.nn.functional.interpolate(
                image.unsqueeze(0), size=(canvas_size, canvas_size), mode="bilinear"
            ).squeeze(0)
            # Scale the control points
            for point in allpoints:
               point.scalePoints(width_scale_factor,height_scale_factor)
        #convert allpoints to torch tensor
        allpointsTensor=th.zeros(len(allpoints),6)
        idx=0
        for point in allpoints:
            allpointsTensor[idx]=th.tensor(point.vertex+point.lftTang+point.rhtTang)
            idx+=1
        #split allpointsTensor into curves based on len of each list in curvepoints
        #clamp allpoints values to 0 to 224 range
        # allpointsTensor=allpointsTensor.clamp(11,213)


        curvesTensor=th.split(allpointsTensor,[len(curve) for curve in curvepoints])
        controlPointsList=[convert_to_cubic_control_points(cp[None, :]).to(th.float64) for cp in curvesTensor]
        #plot control points in controlPointsList
        control_points = th.cat(controlPointsList, dim=0)

        exp_grid_size = int(grid_size * self.grid_epansion_ratio)
        extra_grid = int((exp_grid_size - grid_size) / 2)
        # add offset to x on control points to account for the extra grid
        control_points[:, :, 0] += extra_grid*2
        source_points = create_grid_points(exp_grid_size, 0 - extra_grid, canvas_size+extra_grid, 0-extra_grid, canvas_size+extra_grid)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        control_points = control_points.to(device)
        source_points = source_points.to(device)
        # grid_pts = th.stack(th.meshgrid([th.linspace(-1 / (canvas_size - 1), 1 + 1 / (canvas_size - 1), canvas_size + 2)] * 2),
        #     dim=-1).permute(1, 0, 2).reshape(-1, 2).to(device)
        # distance_field = distance_to_curves(grid_pts, control_points, grid_size).view(canvas_size+2, canvas_size+2)
        distance_field = distance_to_curves(source_points, control_points, exp_grid_size).view(exp_grid_size, exp_grid_size)
        distance_field = th.flip(distance_field, (1,))
        distance_field = self.normalize_distance_field(distance_field)

        torch.save(image, self.processed_paths[index])
        torch.save(distance_field, Path(self.processed_dir).joinpath(f'distance_field.{index+1:04d}.pt'))

class MultiPointProcess(MultiFieldProcess):
    def __init__(self, root, labelsFiles, num_workers=0, proc_path='/', curves=False, width=1920, height=1080):
        self.width = width
        self.height = height
        self.points_fn = r"sampled_points_"
        self.curves_fn=r"curves"
        self.curves= curves
        super().__init__(root,labelsFiles, num_workers,proc_path=proc_path)

    def sample_bezier_curve(self, incurve, num_points):
        """Sample a given bezier curve at fixed intervals."""
        nodes = incurve.transpose(1, 0)

        curve = bezier.Curve(nodes, degree=3)
        s_vals = np.linspace(0.0, 1.0, num_points)
        sampled_points = curve.evaluate_multi(s_vals)

        return sampled_points

    def get(self, idx):
        """get the data from the .npy files in the processed directory if file exists otherwise get the data from the raw directory"""
        processed_path = Path(self.processed_dir).joinpath(f'{idx + 1:04d}.npy') #TODO: this section is redundant as we are only intereseted in processing frames it also returns an  incorrect name.
        #image=torch.load(self.processed_paths[idx])
        if processed_path.exists():
            data = np.load(processed_path)
            return data
        else:
            # Get the label
            # image = read_image(self.raw_paths[idx])
            labels = []
            for labeldict in self.labelsDictList:
                labels.append(labeldict[str(idx + 1)])
            return labels

    def process(self):
        # if lables isn't list make it a list
        if not isinstance(self.labelsFiles, list):
            self.labelsFiles = [self.labelsFiles]
        for labelFile in self.labelsFiles:
            # get the path to the labels file
            self.labels = Path(self.root).joinpath(labelFile)
            self.labelsDictList.append(self.loadLabelsJson())

        num_workers = 1  # Adjust this according to your system's resources

        # Find all processed data files using glob
        processed_files = set(glob.glob(os.path.join(self.processed_dir, '*.npy')))

        if self.curves:
            fn=self.curves_fn
        else:
            fn=self.points_fn
        pattern = re.compile(re.escape(fn) + r'_\.(\d+)\.npy')
        processed_indices = {int(pattern.match(os.path.basename(f)).group(1)) for f in processed_files if
                             pattern.match(os.path.basename(f))}

        # we need to subtract 1 from each index because the indices in the json file start at 1
        processed_indices = {x - 1 for x in processed_indices}
        print(f"Found {len(processed_files)} processed files.")

        print(f"Found {len(processed_indices)} processed indices.")

        # Calculate unprocessed indices
        print("Calculating unprocessed indices...")
        unprocessed_indices = [i for i in range(len(self)) if i not in processed_indices]
        print(f"Found {len(unprocessed_indices)} unprocessed files.")
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            list(tqdm(executor.map(self.process_frame, unprocessed_indices), total=len(unprocessed_indices),
                      desc="Processing",
                      unit="image"))
    def convert_to_bezier_format(self,point_array):
        bezier_curves = []
        n = len(point_array)
        for i in range(n):
            start_point = point_array[i][1]  # center of current curve
            control_point1 = start_point + point_array[i][2]  # right tangent
            control_point2 = point_array[(i + 1) % n][1] + point_array[(i + 1) % n][0]  # left tangent of next curve
            end_point = point_array[(i + 1) % n][1]  # center of next curve
            bezier_curves.append(np.vstack((start_point, control_point1, control_point2, end_point)))
        return bezier_curves

    def reduceDegree(self,point_array):
        """Convert from cubic bezier to quadratic bezier"""
        bezier_curves=self.convert_to_bezier_format(point_array)
        reduced_curves = []
        for curve in bezier_curves:

            b_curve=bezier.Curve(curve.T, degree=3)
            deg2Curve = b_curve.reduce_()
            reduced_curves.append(deg2Curve)

        return reduced_curves

    def subdivideCurves(self, curves, nCurves=8):
        while len(curves) < nCurves:
            # subdivide the curves till we have 15
            # get longest curve
            max_length = 0
            max_index = 0
            for i, curve in enumerate(curves):

                length = curve.length
                if length > max_length:
                    max_length = length
                    max_index = i
            longest_curve = curves[max_index]
            left, right = longest_curve.subdivide()

            # remove the longest curve
            curves.pop(max_index)

            # insert the subdivided curves at the position of the removed curve
            curves.insert(max_index, left)
            curves.insert(max_index + 1, right)
        return curves





    def process_frame(self, index):
        labels  = self.get(index)
        # image = image.numpy().transpose(1, 2, 0)


        sampled_points_list = []
        sampleRate=100
        height_scale_factor = 1 / self.height
        width_scale_factor = 1 / self.width

        for label in labels:
              curvepoints = self.getPoints2DList(label)
              for point in curvepoints:
                  point.lftTang[1] = self.height - point.lftTang[1]
                  point.rhtTang[1] = self.height - point.rhtTang[1]
                  point.vertex[1] = self.height - point.vertex[1]
              for point in curvepoints:
                point.scalePoints(width_scale_factor,height_scale_factor)
              for point in curvepoints:
                point.lftTang[0]=point.lftTang[0]-point.vertex[0]
                point.lftTang[1]=point.lftTang[1]-point.vertex[1]
                point.rhtTang[0]=point.rhtTang[0]-point.vertex[0]
                point.rhtTang[1]=point.rhtTang[1]-point.vertex[1]
             #now we need to invert the y axis on all points

              npPoints=[np.array([point.lftTang,point.vertex,point.rhtTang]) for point in curvepoints]
                #TODO: curves should not remain a fixed number. we also need to account for multiple shapes.
              nCurves=15
              if self.curves:
                  curves=self.reduceDegree(npPoints)
                  curves=self.subdivideCurves(curves,nCurves=nCurves)
                  assert len(curves)<=nCurves
                  tensor = torch.zeros(len(curves), 3, 2)
                  for i in range(len(curves)):
                      tensor[i] = torch.from_numpy(curves[i].nodes.T)
                  tensor = self.ensure_continuous(tensor)
                  np.save(f'{self.processed_dir}/{self.curves_fn}{index + 1:04d}.npy', tensor.numpy())
              else:
                      curves=self.convert_to_bezier_format(npPoints)
                      tensor = torch.zeros(len(curves), 4, 2)
                      for i in range(len(curves)):
                          tensor[i] = torch.from_numpy(curves[i])
                      # imshow(image)
                      # plotCubicSpline(tensor)
                      allCurves = []
                      for curve in curves:
                          allCurves.append(curve)
                      for curve in allCurves:
                          sampled_points = self.sample_bezier_curve(curve, num_points=sampleRate) # num_points can be adjusted
                          sampled_points_list.append(sampled_points)



        sampled_points_list = np.array(sampled_points_list)
        sampled_points_list =sampled_points_list.transpose(0,2, 1).reshape(-1,2)
        np.save(f'{self.processed_dir}/{self.points_fn}{index + 1:04d}.npy', sampled_points_list)

    def ensure_continuous(self,curves):
        # The curves parameter is assumed to be a numpy array of shape (n_curves, 3, 2)
        # where n_curves is the number of curves.
        n_curves = curves.shape[0]
        for i in range(1, n_curves):
            # Get the end point of the previous curve and the start point of the current curve
            prev_end = curves[i - 1, -1, :]
            curr_start = curves[i, 0, :]
            # Compute the midpoint
            midpoint = (prev_end + curr_start) / 2
            # Update the end point of the previous curve and the start point of the current curve
            curves[i - 1, -1, :] = midpoint
            curves[i, 0, :] = midpoint
        # Make sure the end point of the last curve matches the start point of the first curve
        curves[-1, -1, :] = curves[0, 0, :]
        return curves





def compute_mean_std(dataset, percentage=0.1):
    num_samples = int(len(dataset) * percentage)
    indices = torch.randperm(len(dataset))[:num_samples]  # shuffle and take the first num_samples
    sampler = torch.utils.data.SubsetRandomSampler(indices)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0, sampler=sampler)

    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs in tqdm(dataloader):
        for i in range(3):  # 3 channels: R, G, B
            mean[i] += inputs['im'].squeeze()[i, :, :].mean()
            std[i] += inputs['im'].squeeze()[i, :, :].std()
    mean.div_(len(indices))
    std.div_(len(indices))
    return mean, std

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
def distFieldsToPngSeq(folder, name="distance_field", ext="pt"):
    """convert all distance fields in folder to png images"""
    files=glob.glob(os.path.join(folder, "%s.*.%s" % (name, ext)))

    for file in files:
        if ext=="pt":
            dist=th.load(file)
        elif ext=="npy":
            dist=np.load(file)
            dist=th.from_numpy(dist)
        distToPng(dist, file.replace(".%s" % ext, ".png"))

def delFilesbyExtention(folder, ext="pt", name="*"):
    files=glob.glob(os.path.join(folder, "{}*.{}".format(name,ext)))
    for file in files:
        os.remove(file)

def copyFilesbyExtension(infolder,outfolder,ext="pt",name="*",move=False):
    files = glob.glob(os.path.join(infolder, "*.{}".format(ext)))
    for file in files:
        if move:
            shutil.move(file, os.path.join(outfolder, file.split("\\")[-1]))
        else:

            shutil.copy(file,os.path.join(outfolder,file.split("\\")[-1]))



if __name__ == '__main__':
    # delFilesbyExtention(r"D:\ThesisData\data\points\transform_test\pupilMatte\processed",ext="pt",name="*" )
    # delFilesbyExtention(r"D:\ThesisData\data\points\transform_test\pupilMatte\processed",ext="png",name="*" )
    # distFieldsToPngSeq(r"D:\ThesisData\data\points\transform_test\pupilMatte\processed",name="distance_field",ext="pt"  )


    # #
    # processPoints=MultiPointProcess(root=r"D:\ThesisData\data\points\transform_test",
    #                                  labelsFiles=["pointstransform_test_instruments.json","pointstransform_test_pupil.json"])
    # processPoints.process()

    #
    # RotoshapesRoot= r"D:\ThesisData\data\points\rotoshapes"
    # root2=r"D:\ThesisData\fonts"
    # eyeSurgeryRoot= r"D:\\ThesisData\\data\\points\\transform_test\\processed"
    # dataset1=RotoDataset(root=eyeSurgeryRoot,chamfer=False,n_samples_per_curve=100,val=False)
    # dataset2=FontsDataset(root=root2,chamfer=True,n_samples_per_curve=100,val=False)
    #dataset2=esDataset(root=root3,chamfer=False,n_samples_per_curve=100,val=False,use_png=True,png_root=r"D:\ThesisData\data\points\transform_test\combMatte")
    # dataset3=esDataset(root=eyeSurgeryRoot,chamfer=False,n_samples_per_curve=100,val=False,template_idx=1,use_png=True,  png_root=r"D:\ThesisData\data\points\transform_test\pupilMatte", im_fr_main_root=True)
    # # # # #
    #
    # for i in range(0,len(dataset3)):
    #     print(dataset3[i]['points'].shape[0])

    # b=dataset3[100]
    # # print(f"a: {a} /n b: {b}")
    #
    # # mean, std = compute_mean_std(dataset3, percentage=0.1)  # compute mean and std over 10% of the data
    # # print(f'Mean: {mean}')
    # # print(f'Std: {std}')
    # # dataset4 = esDataset(root=root3, chamfer=False, n_samples_per_curve=100, val=False,use_png=True,png_root=r"D:\ThesisData\data\points\transform_test\combMatte")
    # # mean, std = compute_mean_std(dataset4, percentage=0.1)
    # # print(f'Mean: {mean}')
    # # print(f'Std: {std}')
    #
    # data=dataset1[0]
    # dataAlignment=data['alignment_fields']
    # data2=dataset2[0]
    # data2points=data2['points']
    # data2Alignment=data2['alignment_fields']
    # image_shape2 = data2['im'].cpu().numpy().shape[1:3]  # (height, width)
    # scaled_points2 = data2points * np.array(image_shape2)[::-1]  # multiply by (width, height)
    #
    # data3=dataset3[100]
    # data3points=data3['points']
    # data3occupancy=data3['occupancy_fields']
    # data3Alignment=data3['alignment_fields']
    # image_shape3 = data3['im'].cpu().numpy().shape[1:3]  # (height, width)
    # scaled_points3 = data3points * np.array(image_shape3)[::-1]  # multiply by (width, height)
    #
    # # # #plot images and distance fields from each data object
    # fig,axs=plt.subplots(4,2)
    # # axs[0,0].imshow(data['im'].cpu().numpy().transpose(1,2,0))
    # # axs[0,1].imshow(data['distance_fields'].cpu().numpy())
    # axs[1,0].imshow(data2['im'].cpu().numpy().transpose(1,2,0))
    # axs[1, 0].scatter(scaled_points2[:, 0], scaled_points2[:, 1], s=1, c='r')
    # axs[1,1].imshow(data2['distance_fields'].cpu().numpy())
    # axs[2,0].imshow(data3['im'].cpu().numpy().transpose(1,2,0))
    # axs[2, 0].scatter(scaled_points3[:, 0], scaled_points3[:, 1], s=1, c='r')
    # axs[2,1].imshow(data3['occupancy_fields'].cpu().numpy())
    # # #
    # # # # Blend images and distance fields
    # # #
    # # dist=data3['distance_fields'].cpu()
    # # # #make 3 channels
    # # dist=th.stack([dist,dist,dist])
    # # # just plot distance field in it's own window
    # # plt.imshow(dist.numpy().transpose(1,2,0))
    # # plt.show()
    # # #
    # # blend_im = 0.5*data3['im'].cpu() + 0.5*dist
    # # #
    # # #
    # # axs[3,0].imshow(blend_im.numpy().transpose(1,2,0))
    # plt.show()
    # #
    # visualize_vector_field(data3Alignment,scale=0.0001,subsample=6)
    # visualize_vector_field(dataAlignment,scale=0.0000001,subsample=2)
    #
    # processFields=MultiFieldProcess(root=r"D:\ThesisData\data\points\transform_test",
    #                                  labelsFiles=["pointstransform_test_instruments.json","pointstransform_test_pupil.json"])

    # delFilesbyExtention(os.path.join(eyeSurgeryRoot, "instrumentMatte/processed"), ext="pt")
    # processFields=MultiFieldProcess(root=r"D:\ThesisData\data\points\transform_test",
    #                                  labelsFiles=["pointstransform_test_instruments.json"],proc_path="instrumentMatte/processed")
    #
    # processFields.process()

    # processFields=MultiFieldProcess(root=RotoshapesRoot,
    #                                  labelsFiles=["points120423_183451_rev.json"],proc_path="processed")
    #
    # processFields.process()

    # distFieldsToPngSeq(r"D:\ThesisData\data\points\transform_test\instrumentMatte\processed_ylimit_test")
    # delFilesbyExtention(r"D:\ThesisData\data\points\transform_test\instrumentMatte\processed")
    # delFilesbyExtention(os.path.join(RotoshapesRoot,"process"), ext="npy",name="sampled_points")
    processPoints=MultiPointProcess(root=r"D:\ThesisData\data\points\transform_test",
                                     labelsFiles=["pointstransform_test_instruments.json","pointstransform_test_pupil.json"],proc_path="processed",width=224,height=224)
    processPoints.process()


    # copyFilesbyExtension(r"D:\ThesisData\data\points\transform_test\instrumentMatte\tmp",r"D:\ThesisData\data\points\transform_test\instrumentMatte\processed",ext="npy",move=True)
    # processFields=MultiFieldProcess(root=r"D:\ThesisData\data\points\transform_test",
    #                                  labelsFiles=["pointstransform_test_instruments.json"],proc_path="instrumentMatte/processed")
    # #
    # processFields.process()
