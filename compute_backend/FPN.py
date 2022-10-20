import torch
from torch import nn
from collections import OrderedDict

class mobileBlock(nn.Module):
    def __init__(self,
                 in_channels: int, mid_channels: int, out_channels: int,
                 residual=True, downscale=False,
                ):
        super().__init__()
        self.a = nn.Conv2d(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=1,
            bias=False
        )
        self.a_bn = nn.BatchNorm2d(mid_channels, momentum=0.9)
        self.b = nn.Conv2d(
            in_channels=mid_channels,
            out_channels=mid_channels,
            kernel_size=3,
            padding=1,
            groups=mid_channels,
            stride=2 if downscale else 1,
            bias=False,
        )
        self.b_bn = nn.BatchNorm2d(mid_channels, momentum=0.9)
        self.c = nn.Conv2d(
            in_channels=mid_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=False
        )
        self.c_bn = torch.nn.BatchNorm2d(out_channels, momentum=0.9)
        self.residual = residual

    def forward(self, x):
        out = self.a_bn(self.a(x)).relu()
        out = self.b_bn(self.b(out)).relu()
        out = self.c_bn(self.c(out))
        if self.residual:
            out = out + x
        return out

class FPN(nn.Module):
    def __init__(self, out_channels=24):
        super().__init__()
        self.L1 = torch.nn.Sequential(OrderedDict([
            ('1', nn.Conv2d(3,32,3, padding=1, stride=2, bias=False)),
            ('1_bn', nn.BatchNorm2d(32, momentum=0.9)),
            ('1_relu', nn.ReLU()),
            ('2', nn.Conv2d(32,32,3, padding=1, groups=32, bias=False)),
            ('2_bn', nn.BatchNorm2d(32, momentum=0.9)),
            ('2_relu', nn.ReLU()),
            ('3', nn.Conv2d(32,16,1, padding=0, bias=False)),
            ('3_bn', nn.BatchNorm2d(16, momentum=0.9)),
        ]))
        self.L2 = nn.Sequential(OrderedDict([
            ('1', mobileBlock(16,96,24, residual=False, downscale=True)),
            ('2', mobileBlock(24,144,24, residual=True, downscale=False)),
        ]))
        self.L3 = nn.Sequential(OrderedDict([
            ('1', mobileBlock(24,144,32, residual=False, downscale=True)),
            ('2', mobileBlock(32,192,32, residual=True, downscale=False)),
            ('3', mobileBlock(32,192,32, residual=True, downscale=False)),
        ]))
        self.L3down = nn.Sequential(OrderedDict([
            ('1', nn.ConvTranspose2d(out_channels,out_channels,2, stride=2, bias=False)),
            ('1_bn', nn.BatchNorm2d(out_channels, momentum=0.9, eps=0.001)),
            ('1_relu', nn.ReLU()),
        ]))
        self.L4 = nn.Sequential(OrderedDict([
            ('1', mobileBlock(32,192,64, residual=False, downscale=True)),
            ('2', mobileBlock(64,384,64, residual=True, downscale=False)),
            ('3', mobileBlock(64,384,64, residual=True, downscale=False)),
            ('4', mobileBlock(64,384,64, residual=True, downscale=False)),
            ('5', mobileBlock(64,384,96, residual=False, downscale=False)),
            ('6', mobileBlock(96,576,96, residual=True, downscale=False)),
            ('7', mobileBlock(96,576,96, residual=True, downscale=False)),
        ]))
        self.L4down = nn.Sequential(OrderedDict([
            ('1', nn.ConvTranspose2d(out_channels,out_channels,2, stride=2, bias=False)),
            ('1_bn', nn.BatchNorm2d(out_channels, momentum=0.9, eps=0.001)),
            ('1_relu', nn.ReLU()),
        ]))
        self.L5 = nn.Sequential(OrderedDict([
            ('1', mobileBlock(96,576,160, residual=False, downscale=True)),
            ('2', mobileBlock(160,960,160, residual=True, downscale=False)),
            ('3', mobileBlock(160,960,160, residual=True, downscale=False)),
            ('4', mobileBlock(160,960,320, residual=False, downscale=False)),
            ('tip', nn.Conv2d(320,out_channels,1, bias=False)),
            ('tip_bn', nn.BatchNorm2d(out_channels, momentum=0.9, eps=0.001)),
            ('tip_relu', nn.ReLU()),
        ]))
        self.L5down = nn.Sequential(OrderedDict([
            ('1', nn.ConvTranspose2d(out_channels,out_channels,2, stride=2, bias=False)),
            ('1_bn', nn.BatchNorm2d(out_channels, momentum=0.9, eps=0.001)),
            ('1_relu', nn.ReLU()),
        ]))
        self.L45lateral = nn.Sequential(OrderedDict([
            ('1', nn.Conv2d(96,out_channels,1, bias=False)),
            ('1_bn', nn.BatchNorm2d(out_channels, momentum=0.9, eps=0.001)),
            ('1_relu', nn.ReLU()),
        ]))
        self.L34lateral = nn.Sequential(OrderedDict([
            ('1', nn.Conv2d(32,out_channels,1, bias=False)),
            ('1_bn', nn.BatchNorm2d(out_channels, momentum=0.9, eps=0.001)),
            ('1_relu', nn.ReLU()),
        ]))
        self.L23lateral = nn.Sequential(OrderedDict([
            ('1', nn.Conv2d(24,out_channels,1, bias=False)),
            ('1_bn', nn.BatchNorm2d(out_channels, momentum=0.9, eps=0.001)),
            ('1_relu', nn.ReLU()),
        ]))
        self.projector = nn.Sequential(OrderedDict([
            ('1', nn.Conv2d(24,24,3,padding=1, bias=False)),
            ('1_bn', nn.BatchNorm2d(out_channels, momentum=0.99)),
            ('1_relu', nn.ReLU()),
        ]))
        self.out1 = nn.Conv2d(24,1, 1)
        self.out2 = nn.Conv2d(24,2, 1)
        self.out3 = nn.Conv2d(24,2, 1)
        self.out4 = nn.Conv2d(24,10,1)
        
    def forward(self, x):
        x = self.L1(x)
        x = self.L2(x)
        l23 = x
        x = self.L3(x)
        l34 = x
        x = self.L4(x)
        l45 = x
        x = self.L5(x)
        x = self.L5down(x) + self.L45lateral(l45)
        x = self.L4down(x) + self.L34lateral(l34)
        x = self.L3down(x) + self.L23lateral(l23)
        x = self.projector(x)
        return self.out1(x).sigmoid(), self.out2(x), self.out3(x), self.out4(x)

