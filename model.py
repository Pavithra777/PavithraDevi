import matplotlib.pyplot as plt
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import torch
import torch.optim as optim
from torchsummary import summary
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import datasets, transforms
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        #ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        
       
            
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
       # residual = self.shortcut(x)
       # out += residual
        
        out = self.relu2(out)
        return out

class CustomResNet(nn.Module):
    def __init__(self):
      super(CustomResNet, self).__init__()
      #PrepLayer - Conv 3x3 s1, p1) >> BN >> RELU [64k]

      self.prep_layer = nn.Sequential(
                  nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False), 
                  nn.BatchNorm2d(64),
                  nn.ReLU()
              )

      #Layer1 -
      #X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [128k]
      #R1 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [128k] 
      #Add(X, R1)
      self.layer1Conv = nn.Sequential(
                  nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
                  nn.MaxPool2d(kernel_size=2, stride=2),
                  nn.BatchNorm2d(128),
                  nn.ReLU(),
                  
              )
      self.layer1Res = nn.Sequential(
                  ResBlock(128, 128),
              )
    
      #Layer 2 -
      #	1. Conv 3x3 [256k]
      #	2. MaxPooling2D
      #	3. BN
      #	4. ReLU		
          
          
      self.layer2 = nn.Sequential(
                  nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
                  nn.MaxPool2d(kernel_size=2, stride=2),
                  nn.BatchNorm2d(256),
                  nn.ReLU()
              )

      #Layer 3 -
      #	1.	X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [512k]
      #	2. R2 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [512k]
      #	3. Add(X, R2)

      self.layer3Conv = nn.Sequential(
                  nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False),
                  nn.MaxPool2d(kernel_size=2, stride=2),
                  nn.BatchNorm2d(512),
                  nn.ReLU(),
              )

      self.layer3Res = nn.Sequential(
                  ResBlock(512, 512),
              )	


      #MaxPooling with Kernel Size 4
      self.maxpool = nn.MaxPool2d(kernel_size=4,stride=4)        
      #FC Layer 
      self.fc = nn.Linear(512, 10)	

    def forward(self, x):
      out = self.prep_layer(x)
      #out=self.tranblock1(out)
      l1_conv_out = self.layer1Conv(out)
      l1_res_out = self.layer1Res(l1_conv_out)
      #out=self.tranblock2(out)
      #print("l1_conv_out shape: ",l1_conv_out.shape)
      #print("l1_res_out shape: ",l1_res_out.shape)
      l1_out = l1_conv_out + l1_res_out
      #print("l1_out shape: ",l1_out.shape)
      l2_out = self.layer2(l1_out)
      #print("l2_out shape: ",l2_out.shape)
      l3_conv_out = self.layer3Conv(l2_out)
      l3_res_out = self.layer3Res(l3_conv_out)
      #print("l3_conv_out shape: ",l3_conv_out.shape)
      #print("l3_res_out shape: ",l3_res_out.shape)
      l3_out = l3_conv_out + l3_res_out
      #print("l3_out shape: ",l3_out.shape)
      out = self.maxpool(l3_out)
      #print("out shape: ",out.shape)
      out = out.view(out.size(0), -1)
      #print("out shape: ",out.shape)
      out = self.fc(out)
      #print("out shape: ",out.shape)
      out =F.log_softmax(out, dim=-1)	
      #print("out shape: ",out.shape)
      return out
