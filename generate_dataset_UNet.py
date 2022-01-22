import imageio
import os
import glob
import numpy as np
from argparse import ArgumentParser

def imread(path):
    img = imageio.imread(path)
    return img

def gradients(x):
    return np.mean(((x[:-1, :-1, :] - x[1:, :-1, :]) ** 2 + (x[:-1, :-1, :] - x[:-1, 1:, :]) ** 2))

def modcrop(imgs, modulo):
    sz=imgs.shape
    sz=np.asarray(sz)
    if len(sz)==2:
        sz = sz - sz% modulo
        out = imgs[0:sz[0], 0:sz[1]]
    elif len(sz)==3:
        szt = sz[0:2]
        szt = szt - szt % modulo
        out = imgs[0:szt[0], 0:szt[1],:]

    return out

def generate_TFRecord(data_path,label_path,save_path,patch_h,patch_w,stride):
    label_list=np.sort(np.asarray(glob.glob(label_path)))
    img_list = np.sort(np.asarray(glob.glob(os.path.join(data_path, 'X' +  str(scale) + '/*.png'))))

    offset=0

    fileNum=len(label_list)

    file = open(save_path, 'wb')
    write_buf = [patch_h, patch_w, 3, scale]
    count = 0

    for n in range(fileNum):
        print('[*] Image number: %d/%d' % ((n+1), fileNum))
        img=imread(img_list[n])
        label=imread(label_list[n])

        assert os.path.basename(img_list[n])[:-6] == os.path.basename(label_list[n])[:-4]

        img=modcrop(img,scale)
        label=modcrop(label,scale)

        x, y, ch = label.shape
        for i in range(0+offset,x-patch_h+1,stride):
            for j in range(0+offset,y-patch_w+1,stride):
                patch_d = img[i // scale:i // scale + patch_h // scale, j // scale:j // scale + patch_w // scale]
                patch_l = label[i:i + patch_h, j:j + patch_w]

                if np.log(gradients(patch_l.astype(np.float64)/255.)+1e-10) >= -6.0:
                    write_buf += patch_d.flatten().tolist()
                    write_buf += patch_l.flatten().tolist()
                    count += 1

                if count % 25 == 0:
                    write_buf = bytes(write_buf)
                    file.write(write_buf)
                    write_buf = []

    write_buf = bytes(write_buf)
    file.write(write_buf)
    del write_buf

    print('Num of patches:', count)
    print('Shape: [%d, %d, %d]' % (patch_h, patch_w, ch))

    file.close()

if __name__=='__main__':
    parser=ArgumentParser()
    parser.add_argument('--scale', dest='scale', help='Scaling Factor for Super-Resolution', type=int, default=2)
    parser.add_argument('--labelpath', dest='labelpath', help='Path to HR images (./DIV2K_train_HR)')
    parser.add_argument('--datapath', dest='datapath', help='Path to LR images (./DIV2K_train_LR_bicubic)')
    parser.add_argument('--save', dest='save_path', help='Save path for dataset file', default='train_SR_X2')
    options=parser.parse_args()

    scale = options.scale
    labelpath=os.path.join(options.labelpath, '*.png')
    datapath=options.datapath
    path = options.save_path + '.dataset'

    generate_TFRecord(datapath, labelpath, path,48*scale,48*scale,120)
    print('Done')

