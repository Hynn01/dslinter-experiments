#!/usr/bin/env python
# coding: utf-8

# # WGAN Pests  - 128x128

# Import libraries

# In[ ]:


from __future__ import print_function
import os
import time
import datetime
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as utils
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image

get_ipython().run_line_magic('matplotlib', 'inline')


# Set up paths

# In[ ]:


PEST = "TU"
dataset_dir = f'../input/tomato-pests-gan/{PEST}'
load_checkpoints = f'../input/checkpoints'
figures_dir = f'./figures/{PEST}/'
checkpoints_dir = f'./checkpoints/{PEST}/'
graphs_dir = f'./graphs/{PEST}'

if not(os.path.exists(figures_dir)): os.makedirs(figures_dir)
if not(os.path.exists(checkpoints_dir)): os.makedirs(checkpoints_dir)
if not(os.path.exists(graphs_dir)): os.makedirs(graphs_dir)
    
fg = open("g_losses.txt", "a")
fd = open("d_losses.txt", "a")
fe = open("epoch.txt", "a")


# Set up hyperparameters

# In[ ]:


workers = 2

batch_size = 128
image_size = 128

nc = 3
noise_dim = 100

nfg = 64
nfd = 64

epochs = 14401

g_learning_rate = 5e-5
d_learning_rate = 5e-5
beta1 = 0.5
beta2 = 0.999
critic_iterations = 5
lambda_gp=10


# True if you want to load the model from disk, False if you want the model to be initialized from scratch

# In[ ]:


load_model = True


# Set up GPU device for training

# In[ ]:


device = torch.device('cuda:0')


# Load the dataset

# In[ ]:


transform = transforms.Compose([
                                transforms.ToTensor()
                                ])
train_data = datasets.ImageFolder(dataset_dir, transform=transform)
data_loader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=batch_size, num_workers=workers)


# View samples from the dataset

# In[ ]:


ds_sample = next(iter(data_loader))

plt.figure(figsize=(8, 8))
plt.axis('off')
plt.title('Train data')
grid = np.transpose(utils.make_grid(ds_sample[0].to(device)[:64], padding=4, normalize=True).cpu(), (1, 2, 0))
plt.imshow(grid)


# Define a method for weights initialization

# In[ ]:


def init_weights(model):
    if model.__class__.__name__.find('Conv') != -1:
        nn.init.normal_(model.weight, 0.0, 0.02)
    elif model.__class__.__name__.find('BatchNorm') != -1:
        nn.init.normal_(model.weight, 1.0, 0.02)
        nn.init.zeros_(model.bias)


# Define the generator network

# In[ ]:


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
        self.model = nn.Sequential(
            nn.ConvTranspose2d(noise_dim, nfg*16, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(nfg*16),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(nfg*16, nfg*8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(nfg*8),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(nfg*8, nfg*4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(nfg*4),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(nfg*4, nfg*2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(nfg*2),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(nfg*2, nfg, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(nfg),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(nfg, nc, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )
        
    def forward(self, input):
        return self.model(input)


# Define the discriminator network

# In[ ]:


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.model = nn.Sequential(
            nn.Conv2d(nc, nfd, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Dropout2d(0.5, inplace=False),
            
            nn.Conv2d(nfd, nfd*2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(nfg*2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(nfd*2, nfd*4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(nfg*4),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(nfd*4, nfd*8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(nfg*8),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(nfd*8, nfd*16, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(nfg*16),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Dropout2d(0.5, inplace=False),
            
            nn.Conv2d(nfd*16, 1, kernel_size=4, stride=2, padding=0, bias=False),
        )
        
    def forward(self, input):
        return self.model(input)


# Instantiate the generator and initialize its weights (option 1)

# In[ ]:


if load_model == False:
    generator = Generator().to(device)
    generator.apply(init_weights)


# Instantiate the discriminator and initialize its weights (option 1)

# In[ ]:


if load_model == False:
    discriminator = Discriminator().to(device)
    discriminator.apply(init_weights)


# Load the model from the disk (option 2)

# In[ ]:


if load_model == True:
    for filename in os.listdir(load_checkpoints):
        root, ext = os.path.splitext(filename)
        if root.startswith('disc'):
            discriminator = torch.load(os.path.join(load_checkpoints, filename))
        elif root.startswith('gen'):
            generator = torch.load(os.path.join(load_checkpoints, filename))


# Define the loss function (BinaryCrossEntropy)

# In[ ]:


# cross_entropy = nn.BCELoss()


# Define a noise vector to use to track progress

# In[ ]:


sample_noise = torch.randn(64, noise_dim, 1, 1, device=device)


# Define the optimizers (Adam)

# In[ ]:


disc_optimizer = optim.RMSprop(discriminator.parameters(), lr=d_learning_rate)
gen_optimizer = optim.RMSprop(generator.parameters(), lr=g_learning_rate)


# Define a function to plot loss

# In[ ]:


def plot_loss(gen_losses, disc_losses, epoch=None, save=False, show=True):
    plt.figure(figsize=(10, 5))
    plt.title('Generator and Discriminator losses')
    plt.plot(gen_losses, label='G')
    plt.plot(disc_losses, label='D')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    
    if save == True:
        plt.savefig(os.path.join(graphs_dir, f'loss_{epoch}.jpg'))
    if show == True:
        plt.show()


# In[ ]:


def gradient_penalty(critic, real, fake, device):
    batch_size, C, H, W = real.shape
    epsilon = torch.rand((batch_size, 1, 1, 1)).repeat(1, C, H, W).to(device)
    try:
        interpolated_images = real * epsilon + fake * (1 - epsilon)
    except:
        print(real.shape)
        print(fake.shape)
        print(epsilon.shape)
    
    mixed_scores = critic(interpolated_images)
    
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty


# Train both networks simultaneously

# In[ ]:


# gen_losses = []
# disc_losses = []

# for epoch in range(epochs):
#     start = time.time()
#     for i, data in enumerate(data_loader, 0):
#         # Train the discriminator
#         # Fetch the real data
#         ###TRAIN DISCRIMINATOR
#         #Put the real images on the GPU
#         real = data[0].to(device)
#         #Iterate
#         for _ in range(critic_iterations):
#         #Generate fake images for later use
#             size = real.size(0)
#             noise = torch.randn(size, noise_dim, 1, 1, device=device)
#             fake = generator(noise)
#             label = torch.full((size,), 1, device=device, dtype=torch.float)
#             output_real = discriminator(real).view(-1)  
#             real_mean = output_real.mean().item()
#             output_fake = discriminator(fake.detach()).view(-1)
#             fake_mean = output_fake.mean().item()
#             gp = gradient_penalty(discriminator, real, fake, device)
#             disc_err = (-(torch.mean(output_real) - torch.mean(output_fake))+lambda_gp*gp)
#             # Zero out gradients prior to backward passes
#             discriminator.zero_grad()    
#             disc_err.backward(retain_graph=True)
#             disc_optimizer.step()
            
#         # TRAIN THE GENERATOR
#         # Discriminate on fake with updated discriminator
#         output = discriminator(fake).view(-1)
#         gen_mean = output.mean().item()
#         # Calculate loss on fake
#         gen_err = -torch.mean(output)
#         generator.zero_grad()
#         gen_err.backward()
#         gen_optimizer.step()
        
#         if i % 100 == 0:
#             print('[%d/%d][%d/%d] \tD-Loss:%.4f\t G-Loss:%.4f\t D(x):%.4f\t D(G(z)):%.4f\t G(z):%.4f' 
#                   % (epoch + 1, epochs, i + 1, len(data_loader), disc_err.item(), gen_err.item(), real_mean, fake_mean, gen_mean))
            
#         gen_losses.append(gen_err.item())
#         disc_losses.append(disc_err.item())
        
#         with open("g_losses.txt", "a") as file_object:
#             # Append 'hello' at the end of file
#             file_object.write(str(gen_err.item())+"\n")
#         with open("d_losses.txt", "a") as file_object:
#             # Append 'hello' at the end of file
#             file_object.write(str(disc_err.item())+"\n")
        
#     end = time.time()
#     timedelta = datetime.timedelta(seconds=int(end - start))
#     print(f'Time elapsed for epoch {epoch + 1}: {timedelta}\n')
#     with torch.no_grad():
#         sample = generator(sample_noise).detach().cpu()
#     grid = np.transpose(utils.make_grid(sample, padding=4, normalize=True).cpu(), (1, 2, 0))
    

    
#     if epoch % 1000 == 0 or epoch == 14400:
#         # Generate loss graph
#         plot_loss(gen_losses, disc_losses, epoch + 1, save=True, show=False)
        
#         # Save progress
#         torch.save(generator, os.path.join(checkpoints_dir, f'generator{epoch}.pt'))
#         torch.save(discriminator, os.path.join(checkpoints_dir, f'discriminator{epoch}.pt'))
        
#             # Generate samples after every epoch
#         plt.figure(figsize=(8, 8))
#         plt.axis('off')
#         plt.imshow(grid)
#         plt.savefig(os.path.join(figures_dir, f'epoch_{epoch + 1}.png'))
#         plt.close()


# Clear CUDA cache if needed

# In[ ]:


# torch.cuda.empty_cache()


# Plot the loss graph

# In[ ]:


# plot_loss(gen_losses, disc_losses, epoch='final', save=True, show=True)


# Generate samples on random noise

# In[ ]:


noise = torch.randn(1000, noise_dim, 1, 1, device=device).detach()
with torch.no_grad():
    sample = generator(noise).detach().cpu()
# grid = np.transpose(utils.make_grid(sample, padding=4, nrow=5).cpu(), (1, 2, 0))
# plt.figure(figsize=(5, 5))
# plt.axis('off')
# plt.imshow(grid)
# plt.savefig(os.path.join(figures_dir, f'random_{i}.jpg'))
# plt.close()


# In[ ]:


os.mkdir('sample')


# In[ ]:


from torchvision.utils import save_image
for i in range (1000):
    img = sample[i]
    save_image(img, f'gan{i}.png')


# In[ ]:


# !ls

