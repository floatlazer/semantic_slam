import sys, os
import torch
import visdom
import argparse
import timeit
import numpy as np
import scipy.misc as misc
import torch.nn as nn
import torchvision.models as models
import torch

from torch.autograd import Variable
from torch.utils import data

from ptsemseg.models import get_model
from ptsemseg.utils import convert_state_dict
import matplotlib.pyplot as plt

def color_map(N=256, normalized=False):
    """
    Return Color Map in PASCAL VOC format
    """

    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255.0 if normalized else cmap
    return cmap

def decode_segmap(temp, n_classes, cmap):
    r = temp.copy()
    g = temp.copy()
    b = temp.copy()
    for l in range(0, n_classes):
        r[temp == l] = cmap[l,0]
        g[temp == l] = cmap[l,1]
        b[temp == l] = cmap[l,2]

    rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    return rgb

def test():
    # model
    model_name ='segnet'
    checkpoint_path = '/home/interns/xuan/pre_catkin_ws/src/pre2018/seg_cnn/training/2018-06-19/segnet_sunrgbd_best_model.pkl'
    # dataset
    dataset = 'sunrgbd'
    n_classes = 38
    mean = np.array([104.00699, 116.66877, 122.67892])

    # Setup Model
    model = get_model(model_name, n_classes)
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    model.cuda()
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    # Setup image
    image_path = '/home/interns/xuan/datasets/SUNRGBD/test/img-000048.jpg'
    color_img = misc.imread(image_path)
    orig_size = color_img.shape[:-1]
    input_size = (240, 320)
    img = misc.imresize(color_img, input_size, interp='bicubic')
    img = img[:, :, ::-1]
    img = img.astype(float)
    img -= mean
    img = img / 255.0
    # NHWC -> NCHW
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, 0)
    img = torch.from_numpy(img).float()
    images = Variable(img.cuda(0), volatile=True)

    # do prediction
    outputs = model(images)
    pred = np.squeeze(outputs.data.max(1)[1].cpu().numpy(), axis=0)
    pred = pred.astype(np.float32)
    pred = misc.imresize(pred, orig_size, 'nearest', mode='F') # float32 with F mode, resize back to orig_size

    cmap = color_map()
    decoded = decode_segmap(pred, n_classes, cmap)

    # show images
    plt.subplot(1,2,1), plt.imshow(color_img), plt.title('input')
    plt.subplot(1,2,2), plt.imshow(pred), plt.title('prediction')
    plt.show()


if __name__ == '__main__':
    test()
