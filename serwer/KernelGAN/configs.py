import types
import torch
import os


# noinspection PyPep8
class Config:
    def __init__(self, device):
        self.conf = types.SimpleNamespace()

        # Sizes
        self.conf.input_crop_size = 64
        self.conf.scale_factor = 0.5

        # Network architecture
        self.conf.G_chan = 64
        self.conf.D_chan = 64
        self.conf.G_kernel_size = 13
        self.conf.D_n_layers = 7
        self.conf.D_kernel_size = 7

        # Iterations
        self.conf.max_iters = 3000

        # Optimization hyper-parameters
        self.conf.g_lr = 2e-4
        self.conf.d_lr = 2e-4
        self.conf.beta1 = 0.5

        # GPU
        self.conf.gpu_id = 0
        self.conf.real_image = False

        # Kernel post processing
        self.conf.n_filtering = 40

        # ZSSR configuration
        self.conf.noise_scale = 1.
        self.conf.device = device
        self.conf.G_structure = [7, 5, 3, 1, 1, 1]

    def parse(self):
        """Parse the configuration"""
        return self.conf
