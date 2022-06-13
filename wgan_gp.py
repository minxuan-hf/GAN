# 根据自定义数据集修改的版本
# 训练WGAN_GP网络
import argparse
import os
import numpy as np
import math
import sys
from regex import D

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch
from my_dataset import myDataset
import matplotlib.pyplot as plt
from plot_test import plot_show

os.makedirs("220607_ROP_images", exist_ok=True)
os.makedirs("220607_ROP_models", exist_ok=True)
os.makedirs("220607_ROP_resultspic", exist_ok=True)
os.makedirs("220607_ROP_txt", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=100000, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=224, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
parser.add_argument("--sample_interval", type=int, default=10000, help="interval betwen image samples")
opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False


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
            *block(opt.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        return validity


# Loss weight for gradient penalty
lambda_gp = 10

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()

# 载入自定义数据集
dataset = "."
faces_directory = os.path.join(dataset, 'ROP_train445_resize224')

image_transforms = {
    'ROP_train445_resize224': transforms.Compose([
        transforms.Resize([opt.img_size, opt.img_size]),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ]),
}

data = {
    'ROP_train445_resize224':
        myDataset(data_dir=faces_directory, transform=image_transforms['ROP_train445_resize224']),
}

dataloader = DataLoader(data['ROP_train445_resize224'], batch_size=opt.batch_size, shuffle=True)
faces_data_size = len(data['ROP_train445_resize224'])

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


# ----------
#  Training
# ----------

batches_done = 0
d_loss_list = []
g_loss_list = []
for epoch in range(opt.n_epochs):
    for i, imgs in enumerate(dataloader):

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

        # Generate a batch of images_wgan_gp
        fake_imgs = generator(z)

        # Real images_wgan_gp
        real_validity = discriminator(real_imgs)
        # Fake images_wgan_gp
        fake_validity = discriminator(fake_imgs)
        # Gradient penalty
        gradient_penalty = compute_gradient_penalty(discriminator, real_imgs.data, fake_imgs.data)
        # Adversarial loss
        d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty

        d_loss.backward()
        optimizer_D.step()

        optimizer_G.zero_grad()

        # Train the generator every n_critic steps
        if i % opt.n_critic == 0:

            # -----------------
            #  Train Generator
            # -----------------

            # Generate a batch of images_wgan_gp
            fake_imgs = generator(z)
            # Loss measures generator's ability to fool the discriminator
            # Train on fake images_wgan_gp
            fake_validity = discriminator(fake_imgs)
            g_loss = -torch.mean(fake_validity)

            g_loss.backward()
            optimizer_G.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
            )
            d_loss_list.append(d_loss.item())
            g_loss_list.append(g_loss.item())

            if batches_done % opt.sample_interval == 0:
                save_image(fake_imgs.data[:25], "220607_ROP_images/%d.png" % batches_done, nrow=5, normalize=True)

            batches_done += opt.n_critic

# 将d_loss和g_loss写入txt文档保存，可用于生成损失函数图像
d = open("220607_ROP_txt/d.txt", "w")
for i in d_loss_list:
    d.writelines(str(i) + '\n')
d.close()

g = open("220607_ROP_txt/g.txt", "w")
for i in g_loss_list:
    g.writelines(str(i) + '\n')
g.close()

# 保存最终的训练模型
torch.save(generator, "220607_ROP_models/generator_rop.pth")
# 生成器和鉴别器损失图像
plot_show(d_loss_list, "The Loss of Discriminator", "220607_ROP_resultspic/Discriminator_Loss.png")
plot_show(g_loss_list, "The Loss of Generator", "220607_ROP_resultspic/Generator_Loss.png")
