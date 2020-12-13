# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 19:39:35 2020

@author: udits
"""

import cv2
from skimage import io
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class NordlandDataset(Dataset):

    
    def __init__(self, data):
        """
        Args:
            data (string): Directory with all the images.
        """
        self.num_images = data.shape[0]
        self.data = data
    
    def __len__(self):
        return self.num_images
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = self.data[idx]
        ids = np.array(idx)
        image = image.transpose((2, 0, 1))
        #print(image.shape)
        return torch.from_numpy(image), torch.from_numpy(ids)