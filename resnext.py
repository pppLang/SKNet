import torch
from torch import nn


class ResNeXtUnit(nn.Module):
    def __init__(self, in_features, out_features, mid_features=None, stride=1, groups=32):
        super(ResNeXtUnit, self).__init__()
        if mid_features is None:
            mid_features = int(out_features/2)
        self.feas = nn.Sequential(
            nn.Conv2d(in_features, mid_features, 1, stride=1),
            nn.BatchNorm2d(mid_features),
            nn.Conv2d(mid_features, mid_features, 3, stride=stride, padding=1, groups=groups),
            nn.BatchNorm2d(mid_features),
            nn.Conv2d(mid_features, out_features, 1, stride=1),
            nn.BatchNorm2d(out_features)
        )
        if in_features == out_features: # when dim not change, in could be added diectly to out
            self.shortcut = nn.Sequential()
        else: # when dim not change, in should also change dim to be added to out
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_features, out_features, 1, stride=stride),
                nn.BatchNorm2d(out_features)
            )
    
    def forward(self, x):
        fea = self.feas(x)
        return fea + self.shortcut(x)


class ResNeXt(nn.Module):
    def __init__(self, class_num):
        super(ResNeXt, self).__init__()
        self.basic_conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64)
        ) # 32x32
        self.stage_1 = nn.Sequential(
            ResNeXtUnit(64, 256, mid_features=128),
            nn.ReLU(),
            ResNeXtUnit(256, 256),
            nn.ReLU(),
            ResNeXtUnit(256, 256),
            nn.ReLU()
        ) # 32x32
        self.stage_2 = nn.Sequential(
            ResNeXtUnit(256, 512, stride=2),
            nn.ReLU(),
            ResNeXtUnit(512, 512),
            nn.ReLU(),
            ResNeXtUnit(512, 512),
            nn.ReLU()
        ) # 16x16
        self.stage_3 = nn.Sequential(
            ResNeXtUnit(512, 1024, stride=2),
            nn.ReLU(),
            ResNeXtUnit(1024, 1024),
            nn.ReLU(),
            ResNeXtUnit(1024, 1024),
            nn.ReLU()
        ) # 8x8
        self.pool = nn.AvgPool2d(8)
        self.classifier = nn.Sequential(
            nn.Linear(1024, class_num),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        fea = self.basic_conv(x)
        fea = self.stage_1(fea)
        fea = self.stage_2(fea)
        fea = self.stage_3(fea)
        fea = self.pool(fea)
        fea = torch.squeeze(fea)
        fea = self.classifier(fea)
        return fea

    
if __name__=='__main__':
    x = torch.rand(8,3,32,32)
    net = ResNeXt(10)
    out = net(x)
    print(out.shape)