import torch
import torch.nn as nn
from configs import config_image
import torchvision.models as models
from torchvision.models import resnet34, ResNet34_Weights
from torchvision.models import efficientnet_b0,EfficientNet_B0_Weights


class Residuablock(nn.Module):
    def __init__(self,in_channels,out_channels, downsample=True):
        super().__init__()

        # cnn path
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(num_features=out_channels)
        )

        #skip connection
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()

        # add activation
        self.activation = nn.LeakyReLU()

        #reduce size
        self.pool = nn.MaxPool2d(2) if downsample else nn.Identity()

    def forward(self,x):
        identity = self.skip(x)
        out = self.conv(x)
        out = out + identity
        out = self.activation(out)
        out = self.pool(out)

        return out


class DiabeticRetinopathy(nn.Module):
    def __init__(self,num_classes = len(config_image.categorys)):
        super().__init__()
        self.block1 = Residuablock(3, 32)
        self.block2 = Residuablock(32, 64)
        self.block3 = Residuablock(64, 128)
        self.block4 = Residuablock(128, 256)
        self.block5 = Residuablock(256, 512, downsample=False)

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self,x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.global_pool(x)

        x = torch.flatten(x, 1)

        x = self.fc(x)
        return x



class DiabeticRetinopathyPretrain(nn.Module):

    def __init__(self,num_classes=len(config_image.categorys)):
        super().__init__()

        self.model = resnet34(
            weights=ResNet34_Weights.IMAGENET1K_V1
        )

        in_features = self.model.fc.in_features

        self.model.fc = nn.Sequential(
            nn.Linear(in_features,256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256,num_classes)
        )

    def forward(self,x):
        return self.model(x)

# class DiabeticRetinopathyPretrain(nn.Module):
#
#     def __init__(self, num_classes=len(config.categorys)):
#         super().__init__()
#
#         self.model = efficientnet_b0(
#             weights=EfficientNet_B0_Weights.IMAGENET1K_V1
#         )
#
#         in_features = self.model.classifier[1].in_features
#
#         self.model.classifier = nn.Sequential(
#             nn.Dropout(0.3),
#             nn.Linear(in_features, num_classes)
#         )
#
#     def forward(self, x):
#         return self.model(x)