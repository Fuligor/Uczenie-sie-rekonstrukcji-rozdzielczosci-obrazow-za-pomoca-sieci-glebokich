import io
import PIL
import base64
import cv2
import numpy as np
from matplotlib import pyplot as plt
import cherrypy
from scipy.signal import convolve2d

from modele import *
from image_resize import image_resize


def calc_means(image, kernel_size=7):
    kernel = np.ones((kernel_size, kernel_size)) / kernel_size ** 2
    return convolve2d(image, kernel, mode='same')


def color_equalize(y_sr, y_lr):
    temp = image_resize(y_sr, scale=1/2, kernel='cubic')
    temp = image_resize(temp, scale=2, kernel='cubic')

    for i in range(3):
        mean_sr = calc_means(temp[:, :, i])
        mean_lr = calc_means(y_lr[:, :, i])
        diff = mean_lr - mean_sr
        y_sr[:, :, i] = np.clip(y_sr[:, :, i] + diff, 0, 1)

    return y_sr


class Server(object):
    methods = {'UNet': UNetModel(),
                'KPNLP': KPNLPModel(),
                'MZSR_bicubic': MZSRModel(bicubic=True),
                'MZSR_kernelGAN': MZSRModel()}

    @cherrypy.expose
    def index(self):
        receivedImg = cherrypy.request.body.read()
        header = cherrypy.request.headers
        model = IdentityModel()
        if 'Model' in header:
            if header['Model'] in self.methods:
                model = self.methods[header['Model']]
        print(header)
        decoded_data = base64.b64decode(receivedImg)
        npImg = np.frombuffer(decoded_data, np.uint8)
        image = cv2.imdecode(npImg, cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        model.set_input(PIL.Image.fromarray(image))
        model.predict()
        result = model.get_result()

        for i in range(3):
            result[:, :, i] = result[:, :, i].clip(0, 1)

        image = image_resize(image, scale=2, kernel='cubic').astype(np.float32)/255
        image = color_equalize(result, image)

        print('sending data back to the client')

        pill_im = PIL.Image.fromarray(np.uint8(image * 255))
        buff = io.BytesIO()
        pill_im.save(buff, format="PNG")
        encoded_data = base64.b64encode(buff.getvalue())
        return encoded_data



if __name__ == '__main__':
    cherrypy.server.socket_host = '0.0.0.0'
    cherrypy.quickstart(Server())

