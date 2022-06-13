# 加载生成器模型，生成多张GAN图片

import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(100, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(1, 28, 28))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *(3, 224, 224)) # 3通道彩色图像，尺寸为224x224
        return img



if __name__ == "__main__":
    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    # 加载生成器模型参数
    generator = torch.load("220607_ROP_models\generator_rop.pth")
    if cuda:
        generator.cuda()
    
    # 创建目录
    if not os.path.isdir("220610_generator_rop"):
        os.mkdir("220610_generator_rop")
    
    # 随机生成多张图片
    for i in range(10000):
        z = Variable(Tensor(np.random.normal(0, 1, (64, 100))))
        # Generate a batch of images
        gen_imgs = generator(z)
        save_image(gen_imgs.data[:1], "220610_generator_rop/220610_{}_rop.png".format(i), nrow=1, normalize=True)
