import string

import torch as th
import torch.nn as nn

from torchvision.models.resnet import conv3x3, conv1x1, BasicBlock


class CurvesModel(nn.Module):
    # def __init__(self, n_curves, depth=18):
    #     class CurvesModel(nn.Module):
    def __init__(self, n_curves, depth=18, model_type='resnet'):
        super(CurvesModel, self).__init__()
        if model_type == 'resnet':
            depth_dict = {18: [2, 2, 2, 2],
                          34: [3, 4, 6, 3],
                          50: [3, 4, 6, 3],
                          101: [3, 4, 23, 3],
                          152: [3, 8, 36, 3]}
            self.model = ResNet(BasicBlock, depth_dict[depth], num_classes=256, n_z=len(string.ascii_uppercase))
            #self.resnet18 = ResNet(BasicBlock, depth_dict[depth], num_classes=256, n_z=len(string.ascii_uppercase))


        elif model_type == 'unet':
            self.model = UNet(in_channels=3, n_z=len(string.ascii_uppercase), out_channels=256)

        self.curves = nn.Sequential(
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_curves * 4),
            nn.Sigmoid()
        )
        self.strokes = nn.Sequential(
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_curves),
            nn.Sigmoid())


    def forward(self, x, z=None):
        code = self.model(x, z)
        return { 'curves': self.curves(code), 'strokes': self.strokes(code) }
class CurvesModelCubic(nn.Module):
    def __init__(self, n_curves):
        super(CurvesModelCubic, self).__init__()

        self.resnet18 = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=256, n_z=len(string.ascii_uppercase))
        self.curves = nn.Sequential(
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_curves * 2*3),
            nn.Sigmoid()
        )
        self.strokes = nn.Sequential(
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_curves),
            nn.Sigmoid()
        )


    def forward(self, x, z=None):
        code = self.resnet18(x, z)
        return { 'curves': self.curves(code), 'strokes': self.strokes(code) }


class ResNet(nn.Module):
    # modification of torchvision.models.resnet.ResNet to support z conditioning and single channel input

    def __init__(self, block, layers, num_classes=1000, groups=1, width_per_group=64, n_z=0):
        super(ResNet, self).__init__()
        self._norm_layer = nn.BatchNorm2d

        self.inplanes = 64+n_z
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3+n_z, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = self._norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        for m in self.modules():
            if isinstance(m, BasicBlock):
                nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, downsample=downsample, groups=self.groups,
                            base_width=self.base_width, norm_layer=norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x, z=None):
        x = add_z(x, z)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = th.flatten(x, 1)
        x = self.fc(x)

        return x


class UNet(nn.Module):
    def __init__(self, in_channels=3, n_z=0, out_channels=64):
        super(UNet, self).__init__()

        def block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )

        self.encode1 = block(in_channels + n_z, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encode2 = block(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encode3 = block(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encode4 = block(256, 512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = block(512, 1024)

        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decode4 = block(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decode3 = block(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decode2 = block(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        #self.decode1 = block(128, 64)
        self.decode1 = block(128, 512)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 , 256)



    def forward(self, x, z=None):
        x = add_z(x, z)

        enc1 = self.encode1(x)
        enc2 = self.encode2(self.pool1(enc1))
        enc3 = self.encode3(self.pool2(enc2))
        enc4 = self.encode4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = th.cat((dec4, enc4), dim=1)
        dec4 = self.decode4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = th.cat((dec3, enc3), dim=1)
        dec3 = self.decode3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = th.cat((dec2, enc2), dim=1)
        dec2 = self.decode2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = th.cat((dec1, enc1), dim=1)
        dec1 = self.decode1(dec1)
        dec1 = self.avgpool(dec1)
        dec1 = th.flatten(dec1, 1)
        return self.fc(dec1)


def add_z(x, z):
    if z is not None:
        z = z[:,:,None,None].expand(z.size(0), z.size(1), x.size(2), x.size(3))
        x = th.cat([x, z], dim=1)
    return x
