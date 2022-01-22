import numpy as np
from skimage.metrics import structural_similarity as ssim
from matplotlib import pyplot as plt


def norm_image(image):
    if np.max(image) > 1:
        image = image.astype(np.float64) / 255
    return np.round(image * 255).astype(np.float64)


def get_luminance(image):
    return 65.481 / 255. * image[:, :, 0] + 128.553 / 255. * image[:, :, 1] + 24.966 / 255. * image[:, :, 2] + 16 / 255


def norm_pred(pred, target):
    n_pred = get_luminance(pred)
    n_target = get_luminance(target)

    mu_pred = np.mean(n_pred)
    mu_target = np.mean(n_target)

    diff = mu_target - mu_pred

    n_pred = norm_image(n_pred + diff)
    n_target = norm_image(n_target)

    return np.clip(n_pred + diff, 0, 255), n_target


def PSNR(pred, target):
    n_pred, n_target = norm_pred(pred, target)

    n_pred = n_pred[2:-2, 2:-2]
    n_target = n_target[2:-2, 2:-2]
    mse = np.mean((n_pred - n_target) ** 2)
    psnr = 255 ** 2 / mse
    return 10 * np.log10(psnr)


def SSIM(pred, target):
    n_pred, n_target = norm_pred(pred, target)

    return ssim(n_pred / 255, n_target / 255)
