import os
import pandas as pd
import numpy as np
import gkernel
from torch.utils.data import Dataset
from torchvision.io import read_image
import ctypes
import multiprocessing as mp
from PIL import Image

from imresize import *

from utils import functions

class SuperResolutionDataset(Dataset):
    def __init__(self, high_resolution_dir, low_resolution_dir, transform=None, target_transform=None):
        self.high_resolution_dir = high_resolution_dir
        self.low_resolution_dir = low_resolution_dir
        self.transform = transform
        self.target_transform = target_transform
        self.files = [name for name in os.listdir(high_resolution_dir) if name.endswith('.png')]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        lr_path = os.path.join(self.low_resolution_dir, self.files[item])
        lr_image = read_image(lr_path)

        hr_path = os.path.join(self.low_resolution_dir, self.files[item])
        hr_image = read_image(hr_path)
        if self.transform:
            lr_image = self.transform(lr_image)
        if self.target_transform:
            hr_image = self.transform(hr_image)

        return lr_image, hr_image


class DIV2K(SuperResolutionDataset):
    def __init__(self, transform=None, target_transform=None):
        super().__init__('datasets/DIV2K/DIV2K/DIV2K_train_HR/',
                         'datasets/DIV2K/DIV2K/DIV2K_train_LR_unknown/X2/',
                         transform=transform, target_transform=target_transform)


class MZSRDataset(Dataset):
    def __init__(self, high_resolution, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        file = open(high_resolution, 'rb')

        file.seek(0, 2)
        file_size = file.tell() - 3
        file.seek(0)

        self.height = int.from_bytes(file.read(1), "big")
        self.width = int.from_bytes(file.read(1), "big")
        self.channels = int.from_bytes(file.read(1), "big")
        self.image_size = self.height * self.width * self.channels
        data = np.frombuffer(file.read(), dtype=np.uint8)
        file.close()
        count = data.shape[0] // self.image_size
        
        print(count)
        self.images = data.reshape((count, self.width, self.height, self.channels))
        print(self.images.shape)

    def __len__(self):
        return self.images.shape[0]

class MZSRPreTrain(MZSRDataset):
    def __init__(self, high_resolution, transform=None, target_transform=None):
        super().__init__(high_resolution, transform, target_transform)

    def __getitem__(self, item):
        hr_image = self.images[item].copy()
        width, height, chanels = hr_image.shape
        lr_image = resample[range(0, width, 2)][:, range(0, height, 2)]
        lr_image = imresize(lr_image, scale=2, kernel='cubic') / 255
        lr_image = lr_image.astype(np.float32)

        if self.transform:
            lr_image = self.transform(lr_image)
        if self.target_transform:
            hr_image = self.transform(hr_image)

        return lr_image, hr_image

class MZSRMetaTrain(MZSRDataset):
    def __init__(self, high_resolution, scaling_factor=2, transform=None, target_transform=None):
        super().__init__(high_resolution, transform, target_transform)
        self.scaling_factor = scaling_factor
        self.kernel = None

        self.regenerate_kernel()

    def regenerate_kernel(self):
        self.kernel = gkernel.generate_kernel(2.5 * self.scaling_factor)

    def __getitem__(self, item):
        hr_image = self.images[item].copy()

        lr_image = imresize(hr_image, scale=1/2, kernel=self.kernel) / 255
        lr_image = imresize(lr_image, scale=2, kernel='cubic').astype(np.float32)

        if self.transform:
            lr_image = self.transform(lr_image)
        if self.target_transform:
            hr_image = self.transform(hr_image)

        return lr_image, hr_image


class MZSRMetaTest(Dataset):
    def __init__(self, low_resolution, kernel, transform=None, target_transform=None):
        step = 32 if low_resolution.shape[0] < 1000 else 64
        
        self.low_lr_patches = functions.create_image_patches(low_resolution, (64, 64), step)
        self.kernel = kernel
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, item):
        lr_image = self.low_lr_patches[item]
        
        lr_son = imresize(lr_image, scale=1/2, kernel=self.kernel)
        lr_son = imresize(lr_son, scale=2, kernel='cubic').astype(np.float32)

        if self.transform:
            lr_son = self.transform(lr_son)
        if self.target_transform:
            lr_image = self.transform(lr_image)

        return lr_son, lr_image

    def __len__(self):
        return len(self.low_lr_patches)
