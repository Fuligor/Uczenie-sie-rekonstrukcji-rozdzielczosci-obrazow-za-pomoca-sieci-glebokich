import numpy as np
from skimage.metrics import structural_similarity as ssim

def get_luminance(image):
    coeff = np.array([[0.299, 0.587, 0.114]]).T
    return np.squeeze(np.matmul(image, coeff))

def PSNR(pred, target):
    mse = np.mean((pred - target) ** 2)
    psnr = np.max(target) ** 2 / mse
    return 10 * np.log10(psnr)


def SSIM(pred, target):
    pred = get_luminance(pred)
    target = get_luminance(target)
    
    return ssim(pred, target)