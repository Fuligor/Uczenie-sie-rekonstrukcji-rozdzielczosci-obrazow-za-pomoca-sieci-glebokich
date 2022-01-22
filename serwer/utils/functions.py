import numpy as np
from scipy import signal
from PIL import Image

def resample(hr_image):
    return hr_image[range(0, hr_image.shape[0], 2)][:, range(0, hr_image.shape[1], 2)]

def downsample(hr_image, kernel):
    temp = np.zeros_like(hr_image)

    for i in range(temp.shape[2]):
        temp[:, :, i] = signal.convolve2d(hr_image[:, :, i], kernel, mode="same", boundary="symm")
        
    if temp.dtype != np.uint8:
        temp = (temp * 255).astype(np.uint8)
        
    temp = Image.fromarray(temp, mode='RGB')

    size = (hr_image.shape[1]//2, hr_image.shape[0]//2)
    lr_image = temp.resize(size=size, resample=Image.BOX)
    lr_image = lr_image.resize(size=(hr_image.shape[1], hr_image.shape[0]), resample=Image.NEAREST)

    return (np.array(lr_image) / 255).astype(np.float32)


def create_image_patches(image, patch_size, step):
    patches = []

    for i in range(0, image.shape[0] - patch_size[0], step):
        for j in range(0, image.shape[1] - patch_size[1], step):
            patch = image[i:i+patch_size[0], j:j+patch_size[1]]

            patches.append(patch)

    return patches
