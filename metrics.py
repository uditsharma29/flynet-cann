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
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix

from validate import eval_model


# Plotting the precision recall curve
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1) # only difference

def plot_precision_recall(dataloader, device, model):
    y0, acc_train, outs_train = eval_model(model, dataloader)
    #y1, acc_fall, outs_fall = eval_model(model, test_loader)
    #y2, acc_winter, outs_winter = eval_model(model, test_loader2)
    for image, labels in dataloader:
        labels = labels.to(device)
    outs_train = softmax(outs_train)
    sum = 0
    for i in range(0,100):
        sum = sum + outs_train[0][i]
    results = []
    for i in range(0,100):
        results.append(outs_train[i][i])
    labels = labels.cpu().numpy()
    label = []
    for i in range(0,100):
        if labels[i] == y0[i]:
            label.append(1)
        else :
            label.append(0)
    lr_precision, lr_recall, _ = precision_recall_curve(label, results)
    plt.plot(lr_recall, lr_precision, marker='.', label='PrecisionRecall')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.savefig('results/PrecisionRecall.png')
   
def plot_accuracy(model, dataloader, accs):
# Plotting the accuracy
    _, acc_summer, _ = eval_model(model, dataloader)
    plt.plot(accs, 'g', linewidth =2, label = 'summer')
    #plt.plot(fall, 'b', linewidth =2, label = 'fall')
    #plt.plot(winter,color='orange', linewidth =2, label = 'winter')
    plt.grid(True)
    plt.ylabel('Accuracy', fontsize=18)
    plt.title('Flynet', fontsize=20)
    # x = [0, 50, 100, 200]
    # # create an index for each tick position
    # xi = list(range(len(x)))
    # plt.xticks(xi, x)
    plt.legend()
    plt.savefig('results/accuracy.png')
