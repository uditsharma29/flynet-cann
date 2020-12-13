# -*- coding: utf-8 -*-
"""

@author: udits
"""
from __future__ import print_function, division

import cv2
from skimage import io
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from cann import cann
import torchvision.models as models
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout

class FlyNet(nn.Module):
    def __init__(self, args, input_size=input_size, hidden_size=args.hidden_size, num_classes=args.num_classes):
        super(FlyNet, self).__init__()
        self.sampling_ratio = 0.1 # 10% random, sparse connectivity
        self.wta_length = int(hidden_size/2) # 50% WTA
        self.fna_weight = torch.Tensor(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes, bias=False)
        self.reset_fna_weight()
        self.args = args
        
    def reset_fna_weight(self, args):
        # Defining W: binary, sparse matrix
        self.fna_weight = (nn.init.sparse_(self.fna_weight,
                                           sparsity=1-self.sampling_ratio)!=0).float().to(args.device)
    def fna(self, x, args):
        firing_rates = torch.matmul(x,self.fna_weight)
        wta_threshold = torch.topk(firing_rates, self.wta_length, dim=1)[0][:,-1].reshape(args.num_classes,1)
        return (firing_rates>=wta_threshold).float()
        
    def forward(self, x):
        out = self.fna(x)
        out = self.fc(out)
        return out
        
class CNNs(nn.Module):
  def __init__(self, args, n_classes):
    super().__init__()
    self.base_model = models.vgg16(pretrained=True)  # take the model without classifier
    modules = list(self.base_model.children())[:-1]      # delete the last fc layer.
    self.base_model = nn.Sequential(*modules)
    
    self.fc1 = nn.Linear(25088, 4096)
    self.fc2 = nn.Linear(4096, 1000)
    self.fc3 = nn.Linear(1000, n_classes)
    
    self.cnn_layers = Sequential(
    # Defining a 2D convolution layer
    Conv2d(3, 4, kernel_size=3, stride=1, padding=1),
    BatchNorm2d(4),
    ReLU(inplace=True),
    MaxPool2d(kernel_size=2, stride=2),
   
    Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
    BatchNorm2d(4),
    ReLU(inplace=True),
    MaxPool2d(kernel_size=2, stride=2),
    )
    
    self.linear_layers = Sequential(
    Linear(512, 100),               #25088

    )             #4*7*7
    
  def forward(self, x):

    x = x.view(100,3,32,64)

    x = self.cnn_layers(x)
    x = x.view(x.size(0), -1)
    #print(x.shape)
    x = self.linear_layers(x)
    return x

class FC(nn.Module):
  def __init__(self, args, n_classes):
    super().__init__()
    
    self.fc1 = nn.Linear(25088, 4096)
    self.fc2 = nn.Linear(4096, 1000)
    self.fc3 = nn.Linear(1000, n_classes)
    
    self.linear_layers = Sequential(
    Linear(6144, 2048),
    Linear(2048, 1000)          

    )        
    
  def forward(self, x):

    x = x.view(100,3,32,64)

    x = x.view(x.size(0), -1)
    #print(x.shape)
    x = self.linear_layers(x)
    return x



class FC_dropout(nn.Module):
  def __init__(self, args, n_classes):
    super().__init__()
    
    self.fc1 = nn.Linear(25088, 4096)
    self.fc2 = nn.Linear(4096, 1000)
    self.fc3 = nn.Linear(1000, n_classes)
    
    self.linear_layers = Sequential(
    Linear(6144, 2048),
    nn.Dropout(p=0.4),
    Linear(2048, 1000)          

    )        
    
  def forward(self, x):

    x = x.view(100,3,32,64)

    x = x.view(x.size(0), -1)
    #print(x.shape)
    x = self.linear_layers(x)
    return x