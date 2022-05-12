#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import cv2
import math
import imageio

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import pytorch_lightning as pl

from pytorch_lightning.callbacks import TQDMProgressBar
# from pytorch_lightning.profiler import SimpleProfiler

# os.environ['CUDA_VISIBLE_DEVICES']="0"
torch.set_default_dtype(torch.float32)
fast_dev_run = False


# # Define Dataset

# In[ ]:


class CustomImageDataset(Dataset):
    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.im = cv2.imread(self.img_dir)
        self.im = np.array(self.im, dtype=np.float32)
        self.x_len, self.y_len = self.im.shape[0], self.im.shape[1]

    def __len__(self):
        return self.x_len * self.y_len

    def __getitem__(self, idx):
        y = idx//self.x_len
        x = idx - y*self.x_len
        pixel = (self.im[x, y, :])/255.
        y = y/self.y_len*2*math.pi
        x = x/self.x_len*2*math.pi
        
        return (x,y), pixel


# # Model Setup

# In[ ]:


class MLP_PE(pl.LightningModule):
    def __init__(self, middle_dim=128, pe_size=10, im=None, use_pe=True):
        super().__init__()
        self.use_pe=use_pe
        if self.use_pe==True:
            self.layer1 = nn.Sequential(
                nn.Linear(2*pe_size*2,middle_dim),
                nn.ReLU())            
        else:
            self.layer1 = nn.Sequential(
                nn.Linear(2,middle_dim),
                nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Linear(middle_dim,middle_dim),
            nn.ReLU(),
            nn.Linear(middle_dim,3))
        self.im = np.array(im, dtype=np.float32)
        self.x_len, self.y_len = self.im.shape[0], self.im.shape[1]
        self.im_recon = np.zeros((self.x_len,self.y_len,3),dtype=np.uint8)
        self.register_buffer("pe_param",torch.Tensor([2**n for n in range(pe_size)])[None, ...])
        self.img_counter = 0

    def pe(self, pos):
        pos = pos[..., None]        
        pos_ =torch.reshape(pos*self.pe_param,(pos.shape[0],-1))
        pos_pe = torch.stack((torch.sin(pos_),torch.cos(pos_)),axis=1)
        return torch.reshape(pos_pe,(pos.shape[0],-1))

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        return out

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        pos, pixel = batch
        if self.use_pe==True:
            pos = torch.transpose(torch.stack(pos), 0, 1)
            pos_input = self.pe(pos)
        else:
            pos_input = torch.transpose(torch.stack(pos), 0, 1)
        pixel_hat = self(pos_input.float())
        loss = F.mse_loss(pixel_hat, pixel)
        # Logging to TensorBoard by default
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        pos, pixel = batch
        if self.use_pe==True:
            pos = torch.transpose(torch.stack(pos), 0, 1)
            pos_input = self.pe(pos)
        else:
            pos_input = pos = torch.transpose(torch.stack(pos), 0, 1)
        pos = pos.detach().cpu().numpy()
        pixel_hat = torch.clamp(self(pos_input.float()), 0, 1)        
        pixel_hat = np.array((pixel_hat*255).detach().cpu().numpy(), dtype=np.uint8)
        self.im_recon[np.array(np.around(pos[:, 0]/(2*math.pi)*self.x_len), dtype=np.uint32)             , np.array(np.around(pos[:, 1]/(2*math.pi)*self.y_len), dtype=np.uint32), :] = pixel_hat
        

        # Plot
        if batch_idx == len(train_dataloader) - 1:
            font={
                'family': 'Times New Roman',
                'weight': 'normal',
                'size'  : 20,
            }
            plt.figure("Image",dpi=200)
            plt.imshow(self.im_recon)
            plt.axis("on")
            plt.title("Room (Epoch={:02d})".format(self.current_epoch),font=font)
            if self.use_pe==True:
                plt.savefig("./images/photo_w_pe/{:03d}.png".format(self.img_counter), dpi=200)
            else:
                plt.savefig("./images/photo_wo_pe/{:03d}.png".format(self.img_counter), dpi=200)
            self.img_counter += 1
            plt.close('all')

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


# # Dataloader Setup

# In[ ]:


image_dir = "../input/room-small/room_small.jpg"
im = cv2.imread(image_dir)
if not os.path.exists("./images/photo_wo_pe/"):
    os.makedirs("./images/photo_wo_pe/")
if not os.path.exists("./images/photo_w_pe/"):
    os.makedirs("./images/photo_w_pe/")

training_dataset = CustomImageDataset(image_dir)
train_dataloader = torch.utils.data.DataLoader(training_dataset, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)
val_dataloader = torch.utils.data.DataLoader(training_dataset, batch_size=64, num_workers=2, pin_memory=True)


# # Training

# In[ ]:


mlp_pe = MLP_PE(middle_dim=128, pe_size=10, im=im, use_pe=True)

bar = TQDMProgressBar(refresh_rate=50)
# profiler = SimpleProfiler()
trainer = pl.Trainer(max_epochs=20, accelerator="gpu", devices=1, callbacks=[bar],                      check_val_every_n_epoch=2, fast_dev_run=fast_dev_run)

trainer.fit(model=mlp_pe, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)


# In[ ]:


mlp = MLP_PE(middle_dim=128, pe_size=10, im=im, use_pe=False)

bar = TQDMProgressBar(refresh_rate=50)
trainer = pl.Trainer(max_epochs=50, accelerator="gpu", devices=1, callbacks=[bar],                      check_val_every_n_epoch=2, fast_dev_run=fast_dev_run) # (if you have GPUs)

trainer.fit(model=mlp, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)


# # Creat Animation

# In[ ]:


png_dir = './images/photo_w_pe/'
images = []
for file_name in sorted(os.listdir(png_dir)):
    if file_name.endswith('.png'):
        file_path = os.path.join(png_dir, file_name)
        images.append(imageio.imread(file_path))
imageio.mimsave('./images/photo_w_pe.gif', images, duration = 0.5)

png_dir = './images/photo_wo_pe/'
images = []
for file_name in sorted(os.listdir(png_dir)):
    if file_name.endswith('.png'):
        file_path = os.path.join(png_dir, file_name)
        images.append(imageio.imread(file_path))
imageio.mimsave('./images/photo_wo_pe.gif', images, duration = 0.5)


# MLP with Positional Encoding | MLP without Positional Encoding
# ---|---
# ![figure 1](./images/photo_w_pe.gif) | ![figure 2](./images/photo_wo_pe.gif)
# 

# In[ ]:




