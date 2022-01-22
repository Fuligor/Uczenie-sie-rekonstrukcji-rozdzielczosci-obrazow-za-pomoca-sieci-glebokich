import fastai
from fastai.vision import *
from fastai.callbacks import *
from multiprocessing import Pool
import matplotlib.pyplot as plt
import numpy as np
import PIL
import torch
import torchvision
from torchvision.models import vgg16_bn
from skimage.metrics import structural_similarity as ssim
import os
import sys

from scipy import ndimage
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.io import read_image, ImageReadMode
from torch.utils.data import Dataset
from torch import is_tensor, FloatTensor,tensor

sys.path.append('MZSR')
sys.path.append('KernelGAN')

from utils import datasets
from image_resize import image_resize

from configs import Config
from data import DataGenerator
from kernelGAN import KernelGAN
from learner import Learner
import torch


device = "cuda" if torch.cuda.is_available() else "cpu"


class KPNLPnetwork(nn.Module):
    def __init__(self):
        super(KPNLPnetwork, self).__init__()
        self.kernel = (1.0/100)*torch.tensor([[[[1, 4, 6, 4, 1],[4, 16, 24, 16, 4],[6, 24, 36, 24, 6], [4, 16, 24, 16, 4],[1, 4, 6, 4, 1]]]])
        self.downsample = nn.PixelUnshuffle(4)
        self.conv1a = nn.Conv2d(16 , 64 , 3 , padding=1)
        self.conv1b = nn.Conv2d(64, 64, 3, padding=1)
        self.conv1qa = nn.Conv2d(64, 64, 3, padding=1)
        self.conv1qb = nn.Conv2d(64, 64, 3, padding=1)
        self.conv1ha = nn.Conv2d(16, 64, 3, padding=1)
        self.conv1hb = nn.Conv2d(64, 64, 3, padding=1)
        self.conv1fa = nn.Conv2d(4, 64, 3, padding=1)
        self.conv1fb = nn.Conv2d(64, 64, 3, padding=1)
        self.relu = nn.LeakyReLU()
        self.stack = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, 3, padding=1),

            nn.Conv2d(64, 64, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, 3, padding=1),

            nn.Conv2d(64, 64, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, 3, padding=1),

            nn.Conv2d(64, 64, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, 3, padding=1)
        )
        self.upsample2 = nn.PixelShuffle(2)
        self.upsample4 = nn.PixelShuffle(4)
        self.conv2q = nn.Conv2d(64, 25 , 3 , padding=1)
        self.conv2h = nn.Conv2d(64, 25, 3, padding=1)
        self.conv2f = nn.Conv2d(64, 25, 3, padding=1)
        self.conv3q = nn.Conv2d(25 , 1 , 5, padding='same')
        self.conv3h = nn.Conv2d(25, 1, 5, padding='same')
        self.conv3f = nn.Conv2d(25, 1, 5, padding='same')

        self.pyrConv = nn.Conv2d(1 ,1 ,5 , padding="same" , bias=False)

        self.pyrConv.weight = nn.Parameter(self.kernel)

        self.normalUp = nn.Upsample(scale_factor  = 2 , mode='bicubic')
        self.padLayer = nn.ZeroPad2d(2)

    def forward(self, x):
        #print(x.shape , x.dtype)
        common = self.downsample(x)
        #print(common.shape , common.dtype)
        common = self.conv1a(common)
        common = self.relu(common)
        #print(common.shape , common.dtype)
        common = self.stack(common)
        common = self.conv1b(common)
        common = self.relu(common)
        quarter = common
        quarter = self.conv1qa(quarter)
        quarter = self.relu(quarter)
        quarter = self.conv1qb(quarter)
        quarter = self.relu(quarter)
        quarter = self.conv2q(quarter)
        quarter = self.relu(quarter)

        half = self.upsample2(common)
        full = self.upsample4(common)

        half = self.conv1ha(half)
        half = self.relu(half)
        half = self.conv1hb(half)
        half = self.relu(half)
        half = self.conv2h(half)
        half = self.relu(half)


        full = self.conv1fa(full)
        full = self.relu(full)
        full = self.conv1fb(full)
        full = self.relu(full)
        full = self.conv2f(full)
        full = self.relu(full)
        h = x.shape[2]
        w = x.shape[3]
        padded = self.padLayer(x).to(device)
        #padded = nn.functional.pad(x , (2,2,2,2) )
        nq = torch.empty(x.shape[0] , 25, h//4, w//4).to(device)
        nh = torch.empty(x.shape[0] , 25, h//2, w//2).to(device)
        c = torch.empty(x.shape[0] , 25, h, w ).to(device)
        for i in range(h):
            for j in range(w):
                #temp = padded[... , 0, i:i+5 , j:j+5]
                c[...,:,i,j] = torch.flatten(padded[... , 0, i:i+5 , j:j+5] , start_dim=1)
        d = full*c
        e = torch.sum(d , 1, keepdim  = True)

        for i in range(h//2):
            pom_i = i*2
            for j in range(w//2):
                pom_j = j*2
                nh[...,:,i,j] = torch.flatten(padded[... , 0, pom_i:pom_i+5 , pom_j:pom_j+5] , start_dim=1)
        dh = half*nh
        eh = torch.sum(dh , 1, keepdim  = True)


        for i in range(h//4):
            pom_i = i*4
            for j in range(w//4):
                pom_j = j*4
                nq[...,:,i,j] = torch.flatten(padded[... , 0, pom_i:pom_i+5 , pom_j:pom_j+5] , start_dim=1)
        dq = quarter*nq
        eq = torch.sum(dq , 1, keepdim  = True)

        eq = self.normalUp(eq)
        eq = self.pyrConv(eq)  #zakomentowane od (2)
        eh = eh+ eq
        eh = self.normalUp(eh)
        eh = self.pyrConv(eh)  #zakomentowane od (2)
        e = eh+ e

        e = self.normalUp(e)
        #e = self.pyrConv(e)       #zakomentowane od (2)
        #c.detach()
        #eh.detach()
        #eq.detach()
        #padded.detach()
        return e


base_loss = F.l1_loss

def gram_matrix(x):
    n,c,h,w = x.size()
    x = x.view(n, c, -1)
    return (x @ x.transpose(1,2))/(c*h*w)


class FeatureLoss(nn.Module):
    def __init__(self, m_feat, layer_ids, layer_wgts):
        super().__init__()
        self.m_feat = m_feat
        self.loss_features = [self.m_feat[i] for i in layer_ids]
        self.hooks = hook_outputs(self.loss_features, detach=False)
        self.wgts = layer_wgts
        self.metric_names = ['pixel',] + [f'feat_{i}' for i in range(len(layer_ids))
                                          ] + [f'gram_{i}' for i in range(len(layer_ids))]

    def make_features(self, x, clone=False):
        self.m_feat(x)
        return [(o.clone() if clone else o) for o in self.hooks.stored]

    def forward(self, input, target):
        out_feat = self.make_features(target, clone=True)
        in_feat = self.make_features(input)
        self.feat_losses = [base_loss(input,target)]
        self.feat_losses += [base_loss(f_in, f_out)*w
                             for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)]
        self.feat_losses += [base_loss(gram_matrix(f_in), gram_matrix(f_out))*w**2 * 5e3
                             for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)]
        self.metrics = dict(zip(self.metric_names, self.feat_losses))
        return sum(self.feat_losses)

    def __del__(self): self.hooks.remove()


class MZSRNetwork(nn.Module):
    def __init__(self):
        super(MZSRNetwork, self).__init__()
        self.stack = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(64, 3, 3, padding='same')
        )

    def forward(self, x):
        pred = self.stack(x)
        return x + pred



class AbstractModel:
    def __init__(self):
        self.lr_image = None
        self.result = None

    def predict(self):
        raise NotImplementedError()

    def get_result(self) -> np.array:
        raise NotImplementedError()

    def set_input(self, lr_image: PIL.Image):
        self.lr_image = np.array(lr_image)


class IdentityModel(AbstractModel):
    def __init__(self):
        super().__init__()

    def predict(self):
        self.result = self.lr_image

    def get_result(self) -> np.array:
        return self.result


class UNetModel(AbstractModel):
    def __init__(self):
        super().__init__()
        self.model = load_learner('models')

    def predict(self):
        patch_size = 256
        x = 8
        image = self.lr_image
        temp = np.zeros([int(np.ceil((image.shape[0])/(patch_size-2*x)))*patch_size, int(np.ceil(image.shape[1]/(patch_size-2*x)))*patch_size, 3])
        result = np.zeros((temp.shape[0], temp.shape[1], 3))
        temp[x:image.shape[0]+x, x:image.shape[1]+x] = image
        for i in range(0, temp.shape[0] - patch_size + x +1, patch_size - 2*x):
            for j in range(0, temp.shape[1] - patch_size + x + 1 , patch_size - 2*x):

                j_end = j + patch_size
                im = temp[i:i+patch_size, j:j+patch_size]
                imx = pil2tensor(im ,np.float32)
                pred = self.model.predict(Image(imx))
                pred = np.moveaxis(pred[2].numpy(),0,-1)/255
                pred = np.clip(pred, 0, 1)
                result[i:(i+patch_size-2*x), j:(j+patch_size-2*x)] = pred[x:-x, x:-x]

        self.result =  result[:image.shape[0], :image.shape[1]]

    def get_result(self) -> np.array:
        return self.result

    def set_input(self, lr_image: PIL.Image):
        super().set_input(lr_image.resize((lr_image.size[0]*2, lr_image.size[1]*2), resample=PIL.Image.BILINEAR).convert('RGB'))

class KPNLPModel(AbstractModel):
    def __init__(self):
        super().__init__()
        self.model = KPNLPnetwork().to(device)
        self.model.load_state_dict(torch.load('models/KPNLP2.model' , map_location=torch.device(device)))

    def predict(self):
        img = np.moveaxis(self.lr_image, -1, 0)
        #print(img.shape)
        lr_image = FloatTensor(img)[None,:]
        lr_image_y = (16+ lr_image[..., 0, :, :]*0.25679 + lr_image[..., 1, :, :]*0.504 + lr_image[..., 2, :, :]*0.09791)/255
        lr_image_y = lr_image_y[None , :, :].to(device)
        lr_image_cb = (128 - 37.945*lr_image[..., 0, :, :]/256 - 74.494*lr_image[..., 1, :, :]/256 + 112.439*lr_image[..., 2, :, :]/256)
        lr_image_cr = (128 + 112.439*lr_image[..., 0, :, :]/256 - 94.154*lr_image[..., 1, :, :]/256 - 18.285*lr_image[..., 2, :, :]/256)
        hr_cb = nn.functional.interpolate(lr_image_cb[None , :,:],scale_factor = 2 , mode='bicubic').detach().numpy()[0,0]
        hr_cr = nn.functional.interpolate(lr_image_cr[None , :,:],scale_factor = 2 , mode='bicubic').detach().numpy()[0,0]
        pom = self.model(lr_image_y)
        pom2 = pom.to('cpu').detach().numpy()[0,0]
        pom2 *= 255
        pom2 = np.clip(pom2 , 0, 255)
        hr_cr = np.clip(hr_cr, 0, 255)
        hr_cb = np.clip(hr_cb , 0, 255)
        r = pom2 + 1.402 *(hr_cr - 128)
        r = r.clip(0, 255)
        g = pom2 - 0.344136*(hr_cb - 128) - 0.714136 *(hr_cr-128)
        g = g.clip(0, 255)
        b = pom2 + 1.772* (hr_cb - 128)
        b = b.clip(0, 255)
        self.result = np.dstack((r,g,b)).astype(np.uint8)



    def get_result(self) -> np.array:
        return self.result

    def set_input(self, lr_image: PIL.Image):
        super().set_input(lr_image)
        #print(self.lr_image.shape)
        h_pad = (4 - self.lr_image.shape[0] % 4)%4
        w_pad = (4- self.lr_image.shape[1] % 4)%4
        self.lr_image = np.pad(self.lr_image , ((0, h_pad), (0, w_pad) , (0 ,0)) )



class MZSRModel(AbstractModel):
    def __init__(self, bicubic=False):
        super().__init__()
        self.bicubic = bicubic
        self.model = MZSRNetwork().to(device)
        self.conf = Config(device).parse()

    def predict(self):
        loss_fn = nn.L1Loss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=2.5e-4)
        self.model.load_state_dict(torch.load('models/MZSR.model', map_location=torch.device(device)))

        if self.bicubic:
            kernel = 'cubic'
        else:
            gan = KernelGAN(self.conf)
            learner = Learner()
            data = DataGenerator(self.conf, gan, self.lr_image, device)

            for iteration in range(3000):
                [g_in, d_in] = data.__getitem__(iteration)
                gan.train(g_in, d_in)
                learner.update(iteration, gan)

            kernel = gan.finish()

        training_data = datasets.MZSRMetaTest(
            self.lr_image,
            kernel,
            transform=ToTensor(),
            target_transform=ToTensor()
        )
        dataloader = DataLoader(training_data, batch_size=32, shuffle=True)
        lr_image = image_resize(self.lr_image, scale=2, kernel='cubic').astype(np.float32)
        lr_image = ToTensor()(lr_image)
        lr_image = lr_image[None]

        self.model.train()
        size = len(dataloader.dataset)
        for i in range(5):
            for batch, (X, y) in enumerate(dataloader):
                X, y = X.to(device), y.to(device)

                pred = self.model(X)
                loss = loss_fn(pred, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if batch % 100 == 0:
                    loss, current = loss.item(), batch * len(X)
                    #print(f"loss: {loss:>7f}  [{current:>6d}/{size:>6d}]")

        self.model.eval()
        with torch.no_grad():
            X = lr_image.to(device)
            pred = self.model(X)

            pred = pred.cpu().detach().numpy()[0]
            self.result = np.moveaxis(pred, 0, -1).clip(0, 1)


    def get_result(self) -> np.array:
        return self.result

    def set_input(self, lr_image: np.array):
        super().set_input(lr_image)
        self.lr_image = self.lr_image.astype(np.float32) / 255