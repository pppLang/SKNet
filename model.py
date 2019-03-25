import torch
from torch import nn
from untils import SKUnit, ResNeXtUnit


class ResNeXt(nn.Module):
    def __init__(self, class_num):
        super(ResNeXt, self).__init__()
        self.features = nn.Sequential(
            ResNeXtUnit(3, 64), #32x32
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            ResNeXtUnit(64, 128), #16x16
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.AvgPool2d(16)
        )
        self.classifier = nn.Sequential(
            nn.Linear(16*16, 1024),
            nn.Linear(10.24, class_num)
        )
    
    def forward(self, x):
        fea = self.features(x)
        fea = self.classifier(fea)
        return fea