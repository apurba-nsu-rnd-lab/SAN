import torch
import torch.nn as nn
from torch.nn import Conv2d, ReLU, Dropout, BatchNorm2d, MaxPool2d, Linear


class EquivalentNetMNIST(torch.nn.Module):
    
    def __init__(self):
        super(EquivalentNetMNIST, self).__init__()
        
        self.backbone = nn.Sequential(
            Conv2d(1, 16, kernel_size=(3, 3)),
            ReLU(inplace=True),
        )
        
        self.features = nn.Sequential(
            Conv2d(16, 42, kernel_size=(4, 4), stride=(1, 1), padding=(1, 1)),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(2,2))
        
        self.classifier = nn.Sequential(
            Linear(in_features=42*2*2, out_features=72, bias=True),
            ReLU(inplace=True),
            Dropout(p=0.5, inplace=False),
            Linear(in_features=72, out_features=36, bias=True),
            ReLU(inplace=True),
            Dropout(p=0.5, inplace=False),
            Linear(in_features=36, out_features=2, bias=True)
        )
        
        
    def forward(self, x):
        x = x.view_as(x)
        h = self.backbone(x)
        h = self.features(h)
        h = self.avgpool(h)
        h = h.view(x.size(0), -1)
        h = self.classifier(h)
        
        return h
