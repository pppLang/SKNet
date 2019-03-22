import torch
from torch import nn

class SKConv(nn.Module):
    def __init__(self, input_features, out_features, M, G, r, L=32):
        super(SKConv, self).__init__()
        d = max(input_features/r, L)
        self.convs = nn.ModuleList()
        for i in range(M):
            self.convs.add_module(nn.Sequential(
                nn.Conv2d(input_features, input_features, kernel_size=3+i*2, stride=1, padding=1+i*2),
                nn.BatchNorm2d(input_features),
                nn.ReLU(inplace=True)
            ))