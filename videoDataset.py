#!/usr/bin/python
# encoding: utf-8

import os
import random
from PIL import Image
import numpy as np
from image import *
import torch

from torch.utils.data import Dataset
from utils import read_truths_args, read_truths, get_all_files
import cv2

class listDataset(Dataset):

    def __init__(self, streamImage, shape=None, shuffle=True, transform=None, target_transform=None, train=False, seen=0, batch_size=64, num_workers=4, bg_file_names=None):
       self.transform        = transform
       self.target_transform = target_transform
       self.train            = train
       self.shape            = shape
       self.seen             = seen
       self.batch_size       = batch_size
       self.num_workers      = num_workers
       self.bg_file_names    = bg_file_names
       self.streamImage      = streamImage

    def __len__(self):
        return 1

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        imgpath = "LINEMOD/holepuncherTest/JPEGImages/000000.jpg"
        imgpath = imgpath.rstrip()

        if self.train and index % 32== 0:
            if self.seen < 400*32:
               width = 13*32
               self.shape = (width, width)
            elif self.seen < 800*32:
               width = (random.randint(0,7) + 13)*32
               self.shape = (width, width)
            elif self.seen < 1200*32:
               width = (random.randint(0,9) + 12)*32
               self.shape = (width, width)
            elif self.seen < 1600*32:
               width = (random.randint(0,11) + 11)*32
               self.shape = (width, width)
            elif self.seen < 2000*32:
               width = (random.randint(0,13) + 10)*32
               self.shape = (width, width)
            elif self.seen < 2400*32:
               width = (random.randint(0,15) + 9)*32
               self.shape = (width, width)
            elif self.seen < 3000*32:
               width = (random.randint(0,17) + 8)*32
               self.shape = (width, width)
            else: # self.seen < 20000*64:
               width = (random.randint(0,19) + 7)*32
               self.shape = (width, width)
        if self.train:
            jitter = 0.2
            hue = 0.1
            saturation = 1.5 
            exposure = 1.5

            # Get background image path
            random_bg_index = random.randint(0, len(self.bg_file_names) - 1)
            bgpath = self.bg_file_names[random_bg_index]

            img, label = load_data_detection(imgpath, self.shape, jitter, hue, saturation, exposure, bgpath)
            label = torch.from_numpy(label)
        else:
            img = self.streamImage
            if self.shape:
                img = cv2.resize(img,self.shape)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        self.seen = self.seen + self.num_workers
        return img
