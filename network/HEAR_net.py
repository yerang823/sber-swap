#worked by jinwon kim for fun
import torch
import torch.nn as nn
import torch.nn.functional as F
from .AADLayer import *

def conv3x3(in_c, out_c, norm=nn.BatchNorm2d):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=3, stride=2, padding=1, bias=False),
        norm(out_c),
        nn.LeakyReLU(0.1, inplace=True)
    )

def conv4x4(in_c, out_c, norm=nn.BatchNorm2d):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=4, stride=2, padding=1, bias=False),
        norm(out_c),
        nn.LeakyReLU(0.1, inplace=True)
    )

class deconv4x4(nn.Module):
    def __init__(self, in_c, out_c, norm=nn.BatchNorm2d):
        super(deconv4x4, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels=in_c, out_channels=out_c, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn = norm(out_c)
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, input, skip):
        x = self.deconv(input)
        x = self.bn(x)
        x = self.lrelu(x)
        return torch.cat((x, skip), dim=1)

    
class HEAR_Net(nn.Module):
    def __init__(self):
        super(HEAR_Net, self).__init__()
        self.conv1 = conv4x4(6, 64)
        self.conv2 = conv4x4(64, 128)
        self.conv3 = conv4x4(128, 256)
        self.conv4 = conv4x4(256, 512)
        self.conv5 = conv4x4(512, 512)
        
        self.deconv1 = deconv4x4(512, 512)
        self.deconv2 = deconv4x4(512, 256)
        self.deconv3 = deconv4x4(256, 128)
        self.deconv4 = deconv4x4(128, 64)
        self.conv6 = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1, bias=False)
    
    def forward(self, Y_hot_t,):
        feat1 = self.conv1(Y_hot_t)
        #64x128x128
        feat2 = self.conv2(feat1)
        #128x64x64
        feat3 = self.conv3(feat2)
        #256x32x32
        feat4 = self.conv4(feat3)
        #512x16x16
        feat5 = self.conv5(feat4)
        #512x8x8
        feat6 = self.deconv1(feat5,feat4)
        #512x16x16
        feat7 = self.deconv2(feat6,feat3)
        #256x32x32
        feat8 = self.deconv3(feat7, feat2)
        #128x64x64
        feat9 = self.deconv4(feat8, feat1)
        #64x128x128
        out = self.conv6(F.interpolate(feat9, scale_factor=2, mode='bilinear', align_corners=True))

        return out
        