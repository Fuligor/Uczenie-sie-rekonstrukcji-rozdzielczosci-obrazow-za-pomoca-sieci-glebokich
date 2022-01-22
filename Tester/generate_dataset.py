# Adapted from: https://github.com/JWSoh/MainSR
import imageio
import os
import glob
import numpy as np
from argparse import ArgumentParser
from PIL import Image

import gkernel
from image_resize import image_resize


def imread(path):
    img = imageio.imread(path)
    return img.astype(np.float32) / 255


def generate_dataset(label_path, save_path, method):
    label_list = np.sort(np.asarray(glob.glob(label_path)))

    os.makedirs(save_path, exist_ok=True)

    fileNum = len(label_list)

    kernel = 'cubic'

    for n in range(fileNum):
        if method != "bicubic":
            kernel = gkernel.generate_kernel(2, ksize=15)

        fileName = label_list[n].split('\\')[-1]
        print('[*] Image number: %d/%d: %s' % ((n + 1), fileNum, fileName))
        label = imread(label_list[n])
        lr_image = image_resize(label, scale=0.5, kernel=kernel).clip(0, 1)
        print(lr_image.shape)
        lr_image = Image.fromarray((lr_image * 255).astype(np.uint8))
        lr_image.save(f'{save_path}/{fileName}')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input', dest='input', help='Path to HR images')
    parser.add_argument('--output', dest='output', help='Save path')
    parser.add_argument('--method', dest='method', help='Save path')
    options = parser.parse_args()

    input_dir = os.path.join(options.input, '*.png')

    generate_dataset(input_dir, options.output, options.method)
    print('Done')
