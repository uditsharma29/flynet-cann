
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 20:54:17 2020

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

from load_data import NordlandDataset

from validate import eval_model, calculate_accuracy
from models import FlyNet, CNNs, FC, FC_dropout
from metrics import plot_precision_recall, plot_accuracy

import argparse

import pdb


def get_images(img_dir, num_imgs, args):
    print('Loading images...')
    x = []
    for i in range(num_imgs):
        img = cv2.imread(img_dir+'/'+str(i)+'.png')
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #img = cv2.resize(img, (args.img_w,args.img_h),interpolation = cv2.INTER_AREA)
        #x += [img.reshape(args.img_h,args.img_w,args.img_c)]
    #return np.asarray(x)/255
    return img

    
def do_train(model, args):
    
    x_train = get_images(train_dir, args.num_classes, args)
    x_test = get_images(test_dir, args.num_classes, args)
    x_test2 = get_images(test_dir2, args.num_classes, args)
    
    test_dataset = NordlandDataset(data=x_test)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    
    test_dataset2 = NordlandDataset(data=x_test2)
    test_loader2 = DataLoader(test_dataset2, batch_size=args.batch_size,
                             shuffle=True, num_workers=args.num_workers)
    
    train_dataset = NordlandDataset(data=x_train)
    dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                            shuffle=True, num_workers=args.num_workers)
    
    summer = []
    fall = []
    winter = []
    
    criterion = nn.CrossEntropyLoss()  
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    for epoch in range(args.num_epochs):
        for batch in dataloader:
            images = batch[0].to(args.device)
            labels = batch[1].to(args.device)

            outputs = model(images.float())
            loss = criterion(outputs, labels)
              
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            _, argmax = torch.max(outputs, 1)
            accuracy = (labels == argmax.squeeze()).float().mean()
            
        _, acc_summer, _ = eval_model(model, dataloader, args)
        _, acc_fall, _ = eval_model(model, test_loader, args)
        _, acc_winter, _ = eval_model(model, test_loader2, args)
        
        summer.append(acc_summer)
        fall.append(acc_fall)
        winter.append(acc_winter)
        
        if (epoch+1) % args.iter_display == 0:
            print ('Epoch [{}/{}], train_loss: {:.4f}, train_acc: {:.2f}'
                    .format(epoch+1, args.num_epochs, loss.item(), accuracy.item()))

    #Validate
    y0, acc_train, outs_train = eval_model(model, dataloader, args)
    y1, acc_fall, outs_fall = eval_model(model, test_loader, args)
    y2, acc_winter, outs_winter = eval_model(model, test_loader2, args)
    
    print("Training accuracy without CANN: {}".format(acc_train))
    print("Testing accuracy (Fall) without CANN: {}".format(acc_fall))
    print("Testing accuracy (Winter) without CANN: {}".format(acc_winter))
    
    if args.CANN:
        out_train_CANN = cann(outs_train)
        out_fall_CANN = cann(outs_fall)
        out_winter_CANN =  cann(outs_winter)
        #np.savetxt("outs_fall.csv", outs_fall, delimiter=",")
        #np.savetxt("outs_winter.csv", outs_winter, delimiter=",")
        
        acc_train = calculate_accuracy(dataloader, out_train_CANN, args)
        acc_fall = calculate_accuracy(test_loader, out_fall_CANN, args)
        acc_winter = calculate_accuracy(test_loader2, out_winter_CANN, args)
        
        print("Training accuracy with CANN: {}".format(acc_train))
        print("Testing accuracy (Fall) with CANN: {}".format(acc_fall))
        print("Testing accuracy (Winter) with CANN: {}".format(acc_winter))
        
    plot_precision_recall(dataloader, args.device, model)
    
    plot_accuracy(model, dataloader, summer)
    plot_accuracy(model, dataloader, fall)
    plot_accuracy(model, dataloader, winter)
    


if __name__ == "__main__":

   
    data_dir = 'dataset/Nordland/'

    train_dir = data_dir + 'summer/' # train
    test_dir = data_dir + 'fall/' # test
    test_dir2 = data_dir + 'winter/' # test_2

    
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("model_choice", type=str, choices={'flynet', 'CNN', 'FC', 'FC+Dropout'}, default = "flynet",
                        help = "Chhose the type of model you want to run the model on")
    parser.add_argument("CANN", type=bool, default=True,
                        help = "Should CANN be included for inference")
    parser.add_argument("num_epochs", type=int, default=200,
                        help = "Number of epochs")
    parser.add_argument("lr", type=float, default=0.001,
                        help = "Specify learning rate")
    parser.add_argument("batch_size", type=int, default=100,
                        help = "Specify batch size")
    parser.add_argument("num_classes", type=int, default=1000,
                        help = "Specify number of classes")
    parser.add_argument("num_workers", type=int, default=4,
                        help = "Specify number of classes")
    parser.add_argument("display_iter", type=int, default=20,
                        help = "Specify the frequency of console output during training")
                        
    parser.add_argument("hidden_size_flynet", type=int, default=64,
                        help = "Specify hidden size of flynet")
    parser.add_argument("sampling_ratio", type=float, default=0.1,
                        help = "Sampling ratio for random projection matrix")
    parser.add_argument("WTA", type=float, default=0.1,
                        help = "Specify winner takes all parameter")
                        
    parser.add_argument("image_w", type=int, default=64,
                        help = "Specify image width")
    parser.add_argument("image_h", type=int, default=32,
                        help = "Specify image height")
    parser.add_argument("image_c", type=int, default=3,
                        help = "Specify image depth")
    
             
    args = parser.parse_args()
    
    args.input_size = args.img_w*args.img_h
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.iter_display = 20
                            
    if args.model_choice == "flynet":
        model = FlyNet(args).to(args.device)
    elif args.model_choice == "CNN":
        model = CNNs(args, n_classes = args.num_classes).to(args.device)
    elif args.model_choice == "FC":
        model = FC(args, n_classes = args.num_classes).to(args.device)
    elif args.model_choice == 'FC+Dropout':
        model = FC_dropout(args, n_classes = args.num_classes).to(args.device)
    
    do_train(model, args)
    
    
                        
