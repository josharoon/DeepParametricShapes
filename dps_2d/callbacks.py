import cairo
import numpy as np
import torch as th
from torchvision.transforms.functional import to_tensor
from torchvision.utils import make_grid
import ttools.callbacks as cb
import torch.nn.functional as F
from . import viz


class InputImageCallback(cb.TensorBoardImageDisplayCallback):
    def tag(self):
        return 'input'

    def visualized_image(self, batch, fwd_result):
        return batch['im']


class InputDistanceFieldCallback(cb.TensorBoardImageDisplayCallback):
    def tag(self):
        return 'df'

    def visualized_image(self, batch, fwd_result):
        df=batch['distance_fields']
        df=df.unsqueeze(1)
        return df




class InputDistanceFieldCallbackComp(cb.TensorBoardImageDisplayCallback):
    def tag(self):
        return 'df comp'

    def visualized_image(self, batch, fwd_result):
        threshold = 0.03
        df = batch['distance_fields']
        df = df.unsqueeze(1)
        df = (df.abs() < threshold).float()

        image = batch['im']
        if df.is_cuda:
            image = image.cuda()
        image+=df

        return image




class RenderingCallback(cb.TensorBoardImageDisplayCallback):
    def tag(self):
        return 'rendering'

    def visualized_image(self, batch, fwd_result):
        return fwd_result['occupancy_fields'].unsqueeze(1)


class RenderingCompCallback(cb.TensorBoardImageDisplayCallback):

    def tag(self):
        return 'rendering comp'

    def visualized_image(self, batch, fwd_result):
        occField=fwd_result['occupancy_fields'].unsqueeze(1)
        occField_Target=batch['occupancy_fields'].unsqueeze(1)
        targetPoints=batch['points']
        image=batch['im']
        if occField.is_cuda:
            image=image.cuda()
        # image+=occField

        # Replace positive occlusion field values with red pixels in each channel
        image1=image.clone()
        positive_mask = occField > 0
        image1[:,[0],:,:]=th.where(positive_mask, occField, image[:,[0],:,:])
        image1[:,[1],:,:]=th.where(positive_mask, 0, image[:,[1],:,:])
        image1[:,[2],:,:]=th.where(positive_mask, 0, image[:,[2],:,:])

        # Replace positive target occlusion field values with green pixels in each channel
        image2=image.clone()
        positive_mask_target = occField_Target > 0
        image2[:,[1],:,:]=th.where(positive_mask_target, occField_Target.float(), image[:,[1],:,:])
        image2[:,[0],:,:]=th.where(positive_mask_target, 0,image[:,[0],:,:])
        image2[:,[2],:,:]=th.where(positive_mask_target, 0,image[:,[2],:,:])
        image3=(image1+image2)/2

        # Mark target points with blue color
        targetPoints *= image.shape[2]
        image4 = image3.clone()
        for b in range(targetPoints.shape[0]):  # loop through batch
            for i in range(targetPoints.shape[1]):  # loop through points
                y, x = targetPoints[b, i]
                # Ensure the coordinates are integers and within the image dimensions
                x,y = int(x), int(y)
                if 0 <= x < image4.shape[2] and 0 <= y < image4.shape[3]:
                    image4[b, [0], x, y] = 0
                    image4[b, [1], x, y] = 0
                    image4[b, [2], x, y] = 1

        return image4



class CurvesCallback(cb.TensorBoardImageDisplayCallback):
    def tag(self):
        return 'curves'

    def visualized_image(self, batch, fwd_result):
        with th.no_grad():
            curves = fwd_result['curves'].cpu()
            n_loops = batch['n_loops'].cpu()
        data = []
        for curve, n_loop in zip(curves, n_loops):
            surface = cairo.ImageSurface(cairo.Format.RGB24, 128, 128)
            ctx = cairo.Context(surface)
            ctx.scale(128, 128)
            ctx.rectangle(0, 0, 1, 1)
            ctx.set_source_rgb(1, 1, 1)
            ctx.fill()
            viz.draw_curves(curve, n_loop, ctx)

            buf = surface.get_data()
            data.append(to_tensor(np.frombuffer(buf, np.uint8).reshape(128, 128, 4))[:3])
        return th.stack(data)

class CurvesCallbackComp(cb.TensorBoardImageDisplayCallback):
    def tag(self):
        return 'curves comp'

    def visualized_image(self, batch, fwd_result):
        with th.no_grad():
            curves = fwd_result['curves'].cpu()
            n_loops = batch['n_loops'].cpu()
        images = batch['im'].cpu()
        data = []
        for image, curve, n_loop in zip(images, curves, n_loops):
            surface = cairo.ImageSurface(cairo.Format.RGB24, 128, 128)
            ctx = cairo.Context(surface)
            ctx.scale(128, 128)
            ctx.rectangle(0, 0, 1, 1)
            ctx.set_source_rgb(0, 0, 0)
            ctx.fill()
            viz.draw_curves(curve, n_loop, ctx)

            buf = surface.get_data()
            Curveimage = to_tensor(np.frombuffer(buf, np.uint8).reshape(128, 128, 4))[:3]
            # Assume 'image' is your 3x224x224 image tensor
            rescaled_image = F.interpolate(image.unsqueeze(0), size=(128, 128), mode='bilinear', align_corners=False)

            # Squeeze to remove the batch dimension
            rescaled_image = rescaled_image.squeeze(0)
            Curveimage+=rescaled_image
            data.append(image)

        return th.stack(data)

class HyperparamLoggingCallback(cb.TensorBoardLoggingCallback):
    def __init__(self, writer, val_writer, keys=None, val_keys=None, hparams=None,
                 frequency=100, summary_type='scalar'):
        super().__init__(writer, val_writer, keys, val_keys, frequency, summary_type)
        self.hparams = hparams or {}
        self.metrics = {k: 0 for k in (keys or []) + (val_keys or [])}

    # def validation_end(self, val_data):
    #     super().validation_end(val_data)
    #     for k in self.val_keys:
    #         if self.summary_type == 'scalar':
    #             if type(val_data[k]) == float:
    #                 self.metrics[k] = val_data[k]
    #     self._writer.add_hparams(self.hparams, self.metrics, run_name=f'val_step')
    #
    #
    def validation_end(self, val_data):
        super(HyperparamLoggingCallback, self).validation_end(val_data)
        t = self.datasize * (self.epoch+1)
        for k in self.val_keys:
            if self.summary_type == 'scalar':
                if type(val_data[k]) == float:
                    self._val_writer.add_scalar(k, val_data[k], global_step=t)



    # def training_end(self):
    #     self._writer.add_hparams(self.hparams, self.metrics, run_name=f'hparams_end')
