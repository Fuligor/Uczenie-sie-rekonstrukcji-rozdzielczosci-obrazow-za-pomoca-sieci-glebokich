import os
import pandas as pd
import numpy as np
import math
import gkernel
import torch
from torchvision.io import read_image
from torch.utils.data import *
import ctypes
import multiprocessing as mp
from PIL import Image

from image_resize import image_resize

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

        block_size = 512 * 1024
        count = file_size // self.image_size

        self.images = np.zeros(file_size, dtype=np.uint8)
        print(count)

        for i in range(0, file_size, block_size):
            read_size = np.minimum(file_size - i, block_size)
            self.images[i:i+read_size] = np.frombuffer(file.read(block_size), dtype=np.uint8)

        file.close()

        self.images = self.images.reshape((count, self.width, self.height, self.channels))
        print(self.images.shape)

    def __len__(self):
        return self.images.shape[0]

class MZSRPreTrain(Dataset):
    def __init__(self, path, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        file = open(path, 'rb')

        file.seek(0, 2)
        file_size = file.tell() - 4
        file.seek(0)

        self.hr_height = int.from_bytes(file.read(1), "big")
        self.hr_width = int.from_bytes(file.read(1), "big")
        self.channels = int.from_bytes(file.read(1), "big")
        self.scale = int.from_bytes(file.read(1), "big")
        self.lr_height = self.hr_height // self.scale
        self.lr_width = self.hr_width // self.scale
        self.hr_image_size = self.hr_height * self.hr_width * self.channels
        self.lr_image_size = self.lr_height * self.lr_width * self.channels
        self.record_size = self.hr_image_size + self.lr_image_size

        block_size = 512 * 1024
        count = file_size // self.record_size

        images = np.zeros(file_size, dtype=np.uint8)
        print(count)

        for i in range(0, file_size, block_size):
            read_size = min(file_size - i, block_size)
            images[i:i+read_size] = np.frombuffer(file.read(read_size), dtype=np.uint8)

        file.close()

        self.lr_images = np.zeros(count * self.lr_image_size, dtype=np.uint8)
        self.hr_images = np.zeros(count * self.hr_image_size, dtype=np.uint8)

        actual = 0
        for i in range(count):
            self.lr_images[i * self.lr_image_size:(i + 1) * self.lr_image_size] = images[actual:actual+self.lr_image_size]
            actual += self.lr_image_size
            self.hr_images[i * self.hr_image_size:(i + 1) * self.hr_image_size] = images[actual:actual+self.hr_image_size]
            actual += self.hr_image_size
        del images

        self.lr_images = self.lr_images.reshape((count, self.lr_width, self.lr_height, self.channels))
        self.hr_images = self.hr_images.reshape((count, self.hr_width, self.hr_height, self.channels))
        print(self.hr_images.shape)
        print(self.lr_images.shape)

    def __getitem__(self, item):
        hr_image = self.hr_images[item].copy().astype(np.float32) / 255
        lr_image = self.lr_images[item].copy()
        lr_image = image_resize(lr_image, scale=2, kernel='cubic').astype(np.float32) / 255

        if self.transform:
            lr_image = self.transform(lr_image)
        if self.target_transform:
            hr_image = self.transform(hr_image)

        return lr_image, hr_image

    def __len__(self):
        return self.hr_images.shape[0]

class MZSRMetaTrain(MZSRDataset):
    def __init__(self, high_resolution, scaling_factor=2, transform=None, target_transform=None):
        super().__init__(high_resolution, transform, target_transform)
        self.scaling_factor = scaling_factor
        self.kernel = None

        self.regenerate_kernel()

    def regenerate_kernel(self):
        self.kernel = gkernel.generate_kernel(2.5 * self.scaling_factor)

    def __getitem__(self, item):
        hr_image = self.images[item]

        lr_image = image_resize(hr_image, scale=1/2, kernel=self.kernel)
        lr_image = image_resize(lr_image, scale=2, kernel='cubic').astype(np.float32) / 255

        if self.transform:
            lr_image = self.transform(lr_image)
        if self.target_transform:
            hr_image = self.transform(hr_image)

        return lr_image, hr_image


class MZSRMetaTest(Dataset):
    _patch_size = 64

    def __init__(self, low_resolution, kernel, transform=None, target_transform=None):
        step = math.sqrt((low_resolution.shape[0] - self._patch_size) * (low_resolution.shape[1] - self._patch_size) // 500)
        step = max(1, int(step))

        self.low_lr_patches = functions.create_image_patches(low_resolution, (self._patch_size, self._patch_size), step)

        self.kernel = kernel
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, item):
        lr_image = self.low_lr_patches[item]
        
        lr_son = image_resize(lr_image, scale=1/2, kernel=self.kernel)
        lr_son = image_resize(lr_son, scale=2, kernel='cubic').astype(np.float32)

        if self.transform:
            lr_son = self.transform(lr_son)
        if self.target_transform:
            lr_image = self.transform(lr_image)

        return lr_son, lr_image

    def __len__(self):
        return len(self.low_lr_patches)
