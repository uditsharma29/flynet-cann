# -*- coding: utf-8 -*-
"""
@author: udits
"""
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

from dataloader import NordlandDataset

from models import FlyNet, CNNs, FC, FC_dropout
from metrics import plot_precision_recall, plot_accuracy


# Test the model

def eval_model(model,data_loader, args):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in data_loader:
            images = images.view(images.size(0), -1).to(args.device)
            labels = labels.to(args.device)
            outputs = model(images.float())
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            #correct = np.sum(np.equal(predictions, labels.cpu().numpy()))
        #print('Test Accuracy: {} %'.format(100 * correct / total))
    return outputs.data.cpu().numpy().argmax(axis=1), correct / total, outputs.cpu().detach().numpy()
    
def calculate_accuracy(data_loader, predictions, args):
    total = 0
    correct = 0
    for _, labels in data_loader:
        labels = labels.to(args.device)
        #pdb.set_trace()
        #_, predicted = np.max(outputs)
        total += labels.size(0)
        #correct += np.sum(predictions == labels)
        correct += np.sum(np.equal(predictions, labels.cpu().numpy()))
        #pdb.set_trace()
    return correct / total