import concurrent
import glob
import json
import os
import re
import string
from pathlib import Path

import torch
from torchvision.io import read_image
import numpy as np
from PIL import Image
import torch as th
from matplotlib import pyplot as plt
from torch_geometric.data import Data
from torchvision.transforms import transforms
from torchvision.transforms.functional import to_tensor
from torch_geometric.data import  Dataset
from pyGutils.cubicCurvesUtil import *
from pyGutils.viz import plot_distance_field, plotCubicSpline
from tqdm import tqdm

from pyGutils.cubicCurvesUtil import convert_to_cubic_control_points, create_grid_points
from pyGutils.viz import plot_distance_field,plotCubicSpline
#from . import utils, templates
import utils, templates


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
    def __init__(self, root, chamfer, n_samples_per_curve, png_root=None, use_png=False, val=False):
        self.root = root
        self.chamfer = chamfer
        self.png_root = png_root
        self.use_png = use_png
        self.n_samples_per_curve = n_samples_per_curve
        self.filesIndicies=self.sortFiles()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()

        ])

        np.random.shuffle(self.filesIndicies)
        cutoff = int(0.9*len(self.filesIndicies))
        if val:
            self.files = self.filesIndicies[cutoff:]
        else:
            self.files = self.filesIndicies[:cutoff]
        self.n_loops_dict = templates.n_loops_eye

    def __repr__(self):
        return "esDataset | {} entries".format(len(self))

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

        if self.use_png:
            fname = f"spoints.{str(int(self.filesIndicies[idx])).zfill(4)}.png"
            im_path = os.path.join(self.png_root, fname)
            im = Image.open(im_path)
            im = self.transform(im)




        else:
            fname = f"spoints.{self.filesIndicies[idx]}.pt"
            im = th.load(os.path.join(self.root, fname))#.y
        dfname = f"distance_field.{self.filesIndicies[idx]}.pt"
        # im = Image.open(os.path.join('\\'.join(self.root.split("\\")[:-1]),fname))
        # #resize the image to 224x224
        # im = im.resize((224,224))

        #resize the image to 3x224x224


        distance_fields = th.load(os.path.join(self.root, dfname))
        #distance_fields = th.flip(distance_fields,dims=[0])
        distance_fields = th.flip(distance_fields, (0,))

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
        # frNum = int(fname.split('.')[-2])
        # if(frNum>7800):
        #     letter_idx=1
        # else:
        #     letter_idx=0
        letter_idx=7 # multi curve template
        return {
            'fname': fname,
            'im': im,
            'distance_fields': distance_fields,
            'alignment_fields': alignment_fields,
            'occupancy_fields': occupancy_fields,
            'points': points,
            'letter_idx': letter_idx, # string.ascii_uppercase.index(fname[0]),
            'n_loops': 2  # self.n_loops_dict[fname[0]]
        }

class MultiFieldProcess(Dataset):

    def __init__(self,root,labelsFiles=None,transform=None, pre_transform=None,pre_filter=None, preprocess=True):
        self.root=root
        #self.processed_dir=processed_dir
        self.labelsFiles=labelsFiles
        self.labels=None
        self.labelsDictList=[]
        super().__init__(root, transform, pre_transform, pre_filter)
        #self.process()
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
        processed_indices = {int(pattern.match(os.path.basename(f)).group(1)) for f in processed_files}
        #we need to subtract 1 from each index because the indices in the json file start at 1
        processed_indices={x-1 for x in processed_indices}
        print(f"Found {len(processed_files)} processed files.")

        print(f"Found {len(processed_indices)} processed indices.")

        # Calculate unprocessed indices
        print("Calculating unprocessed indices...")
        unprocessed_indices = [i for i in range(len(self)) if i not in processed_indices]
        print(f"Found {len(unprocessed_indices)} unprocessed files.")
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            list(tqdm(executor.map(self.process_image, unprocessed_indices), total=len(unprocessed_indices),
                      desc="Processing",
                      unit="image"))
    def get(self, idx):
        """get the data from the .pt files in the processed directory if file exists otherwise get the data from the raw directory"""
        if Path(self.processed_dir).joinpath(self.processed_file_names[idx]).exists():
            data = torch.load(self.processed_paths[idx])
            df = torch.load(Path(self.processed_dir).joinpath(f'distance_field.{idx+1:04d}.pt'))
            return data.y, data.x, self.processed_paths[idx], df
        else:
            # get the image
            image = read_image(self.raw_paths[idx])
            # get the label
            labels=[]
            for labeldict in self.labelsDictList:
                labels.append(labeldict[str(idx + 1)])
            return image, labels

    def normalize_image(self, image):
        return image / 255.0

    def normalize_distance_field(self, distance_field):
        d_max = torch.max(distance_field)
        return distance_field / d_max


    def process_image(self, index):
        grid_size = 224
        image, labels = self.get(index)
        image = self.normalize_image(image)
        curvepoints=[self.getPoints2DList(label) for label in labels]
        # add items in curvepoints list
        allpoints=[]
        for curve in curvepoints:
                allpoints+=curve
        #we scale everything to 224x224 for the model
        original_height, original_width = image.shape[1:]
        if original_height != 224 or original_width != 224:
            # Calculate scaling factors
            height_scale_factor = 224 / original_height
            width_scale_factor = 224 / original_width
            # Resize the image
            image = th.nn.functional.interpolate(
                image.unsqueeze(0), size=(224, 224), mode="bilinear"
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
        curvesTensor=th.split(allpointsTensor,[len(curve) for curve in curvepoints])
        controlPointsList=[convert_to_cubic_control_points(cp[None, :]).to(th.float64) for cp in curvesTensor]
        #plot control points in controlPointsList
        control_points = th.cat(controlPointsList, dim=0)
        source_points = create_grid_points(grid_size, 0, 224, 0, 224)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        control_points = control_points.to(device)
        source_points = source_points.to(device)

        distance_field = distance_to_curves(source_points, control_points, grid_size).view(grid_size, grid_size)
        distance_field = th.flip(distance_field, (1,))
        distance_field = self.normalize_distance_field(distance_field)
        # for cp in controlPointsList:
        #     distance_field =distance_to_curves(source_points, cp, grid_size).view(grid_size, grid_size)
        #     distance_field = th.flip(distance_field, (1,))
        #     distance_field = self.normalize_distance_field(distance_field)
        #     plot_distance_field(distance_field, vmax=1, title="Distance field")
        #     plotCubicSpline(cp,image=image.permute(1,2,0).cpu().numpy())
        #plot_distance_field(distance_field,vmax=1, title="Distance field")
        torch.save(image, self.processed_paths[index])
        torch.save(distance_field, Path(self.processed_dir).joinpath(f'distance_field.{index+1:04d}.pt'))


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
def distFieldsToPngSeq(folder):
    """convert all distance fields in folder to png images"""
    files=glob.glob(os.path.join(folder,"distance_field.*.pt"))
    for file in files:
        dist=th.load(file)
        distToPng(dist,file.replace(".pt",".png"))


if __name__ == '__main__':
    distFieldsToPngSeq(r"D:\pyG\data\points\transform_test\processed")




    #
    # root=r"D:\pyG\data\points\120423_183451_rev\processed"
    # root2=r"D:\DeepParametricShapes\data\fonts"
    # root3=r"D:\pyG\data\points\transform_test\processed"
    # dataset1=RotoDataset(root=root,chamfer=False,n_samples_per_curve=100,val=False)
    # #dataset2=FontsDataset(root=root2,chamfer=False,n_samples_per_curve=100,val=False)
    # dataset2=esDataset(root=root3,chamfer=False,n_samples_per_curve=100,val=False,use_png=True,png_root=r"D:\pyG\data\points\transform_test\combMatte")
    # dataset3=esDataset(root=root3,chamfer=False,n_samples_per_curve=100,val=False)

    # mean, std = compute_mean_std(dataset3, percentage=0.1)  # compute mean and std over 10% of the data
    # print(f'Mean: {mean}')
    # print(f'Std: {std}')
    # dataset4 = esDataset(root=root3, chamfer=False, n_samples_per_curve=100, val=False,use_png=True,png_root=r"D:\pyG\data\points\transform_test\combMatte")
    # mean, std = compute_mean_std(dataset4, percentage=0.1)
    # print(f'Mean: {mean}')
    # print(f'Std: {std}')

    # data2=dataset2[0]
    # data=dataset1[0]
    # data3=dataset3[0]
    # # #plot images and distance fields from each data object
    # fig,axs=plt.subplots(4,2)
    # axs[0,0].imshow(data['im'].cpu().numpy().transpose(1,2,0))
    # axs[0,1].imshow(data['distance_fields'].cpu().numpy())
    # axs[1,0].imshow(data2['im'].cpu().numpy().transpose(1,2,0))
    # axs[1,1].imshow(data2['distance_fields'].cpu().numpy())
    # axs[2,0].imshow(data3['im'].cpu().numpy().transpose(1,2,0))
    # axs[2,1].imshow(data3['distance_fields'].cpu().numpy())
    # #
    # # # Blend images and distance fields
    # #
    # dist=data3['distance_fields'].cpu()
    # # #make 3 channels
    # dist=th.stack([dist,dist,dist])
    # # just plot distance field in it's own window
    # plt.imshow(dist.numpy().transpose(1,2,0))
    # plt.show()
    # #
    # blend_im = 0.5*data3['im'].cpu() + 0.5*dist
    # #
    # #
    # axs[3,0].imshow(blend_im.numpy().transpose(1,2,0))
    # plt.show()

    # processFields=MultiFieldProcess(root=r"D:\pyG\data\points\transform_test",
    #                                  labelsFiles=["pointstransform_test_instruments.json","pointstransform_test_pupil.json"])



