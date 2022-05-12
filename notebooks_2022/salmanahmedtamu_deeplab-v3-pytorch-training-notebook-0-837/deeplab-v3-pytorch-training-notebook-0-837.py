#!/usr/bin/env python
# coding: utf-8

# # Note (Its just a script for Training. You might need some processing to run on Kaggle.)

# # Imports|

# In[ ]:


get_ipython().system('pip install -q segmentation-models-pytorch')


# In[ ]:


import pandas as pd
import numpy as np
import os
from glob import glob
import gc
gc.enable()
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import torchvision
import torch.nn.functional as F
from torch.autograd import Variable
from PIL import Image
import cv2
import albumentations as A
import time
import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from torch.cuda import amp
scaler = amp.GradScaler()
from torch.autograd import Variable
import torch.nn.functional as F

import numba
import numpy as np
from math import sqrt
from scipy.spatial.distance import directed_hausdorff
from scipy.ndimage import convolve
from scipy.ndimage.morphology import distance_transform_edt as edt

from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau

import pandas as pd
import numpy as np
import os
from glob import glob
import gc
from sklearn.model_selection import KFold
gc.enable()


# # CONFIG

# In[ ]:


CFG = {
    'fold' : 0, 
    'batch_size' : 8,
    'image_size' : 256,
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    'init_lr' : 1e-3,
    'warmup_factor' : 10,
    'warmup_epo' : 3,
    'n_epochs' : 25,
    'num_workers' : 4,
}


# # Helping Functons

# In[ ]:


def hd_dist(preds, targets):
    preds_coords = np.argwhere(preds) / np.array(preds.shape)
    targets_coords = np.argwhere(targets) / np.array(preds.shape)
    haussdorf_dist = directed_hausdorff(preds_coords, targets_coords)[0]
    return haussdorf_dist

def dice(im1, im2, empty_score=1.0):

    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / im_sum

class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError('multiplier should be greater thant or equal to 1.')
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_last_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
                self._last_lr = self.after_scheduler.get_last_lr()
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)
            
class TractDataset(Dataset):
    def __init__(self, df, n_chans = 3, transform=None):
        self.df = df
        self.mask_path = df['mask_path']
        self.image_path = df['image_path']
        self.transform = transform
        self.n_chans = n_chans
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):

        img = Image.open(self.image_path[idx]).convert('RGB')
        img = np.array(img)
        mask = np.load(self.mask_path[idx])
        if self.transform is not None:
            aug = self.transform(image=img, mask=mask)
            img = aug['image']
            mask = aug['mask']
        img = img / 255.0
        img = np.transpose(img, (2, 0, 1))
        mask = np.transpose(mask, (2, 0, 1))
        
        return torch.tensor(img, dtype=torch.float), torch.tensor(mask, dtype=torch.float)


# # Define Loss Functions

# In[ ]:


bce_loss = nn.BCEWithLogitsLoss()
lovasz_loss = smp.losses.LovaszLoss(mode='binary', per_image=False)
tversky_loss = smp.losses.TverskyLoss(mode='binary', log_loss=False, from_logits=True)

def bce_lovasz(output, target):
    return (0.5 * bce_loss(output, target)) + (0.5 * lovasz_loss(output, target))

def bce_lovasz_tversky_loss(output, target):
    return (0.25 * bce_loss(output, target)) + (0.25 * lovasz_loss(output, target)) + (0.5 * tversky_loss(output, target))

def get_loss(epoch):
    if epoch <= 5:
        return bce_lovasz
    else:
        return bce_lovasz_tversky_loss


# In[ ]:


def train_epoch(epoch, model, loader, optimizer):

    model.train()
    loss_func = get_loss(epoch)

    train_loss = []
    bar = tqdm(loader, total=len(loader))
    
    for (data, target) in bar:
        
        data = data.to(CFG['device'])
        target = target.to(CFG['device'])
        
        optimizer.zero_grad()
        
        with amp.autocast(False):
            output = model(data)
            
            loss = loss_func(output, target)
        
        scaler.scale(loss).backward() 
        scaler.step(optimizer)
        scaler.update()
        
        loss_np = loss.detach().cpu().numpy()
        train_loss.append(loss_np)
        smooth_loss = sum(train_loss[-100:]) / min(len(train_loss), 100)
        bar.set_description('loss: %.5f, smth: %.5f' % (loss_np, smooth_loss))
        
    return model, np.mean(train_loss)


# In[7]:


def valid_epoch(epoch, model, loader):

    loss_func = get_loss(epoch)

    model.eval()
    val_loss = []
    LOGITS = []
    TARGETS = []
    
    with torch.no_grad():
        
        for (data, targets) in tqdm(loader, total=len(loader)):
            
            data, targets = data.to(CFG['device']), targets.to(CFG['device'])
            
            output = model(data)
            
            loss = loss_func(output, targets)

            val_loss.append(loss.item())
            
            LOGITS.append(output.cpu())
            TARGETS.append(targets.cpu())
            
    val_loss = np.mean(val_loss)
    LOGITS = torch.cat(LOGITS)
    TARGETS = torch.cat(TARGETS).numpy()
    
    return val_loss, LOGITS, TARGETS


# # Prepare Dataframe

# In[ ]:


df = pd.read_csv("train.csv")

df["case_id_str"] = df["id"].apply(lambda x: x.split("_", 2)[0])
df["case_id"] = df["id"].apply(lambda x: int(x.split("_", 2)[0].replace("case", "")))

# 2. Get Day as a column
df["day_num_str"] = df["id"].apply(lambda x: x.split("_", 2)[1])
df["day_num"] = df["id"].apply(lambda x: int(x.split("_", 2)[1].replace("day", "")))

# 3. Get Slice Identifier as a column
df["slice_id"] = df["id"].apply(lambda x: x.split("_", 2)[2])

TRAIN_DIR = 'train'
# Get all training images
all_train_images = glob(os.path.join(TRAIN_DIR, "**", "*.png"), recursive=True)

p = []
x = all_train_images[0].rsplit("/", 4)[0]
for i in range(0, df.shape[0]):
    p.append(os.path.join(x, df["case_id_str"].values[i], df["case_id_str"].values[i]+"_"+df["day_num_str"].values[i], "scans", df["slice_id"].values[i]))
df["_partial_ident"] = p

p = []
for i in range(0, len(all_train_images)):
    p.append(str(all_train_images[i].rsplit("_",4)[0]))
    
_tmp_merge_df = pd.DataFrame()
_tmp_merge_df['_partial_ident'] = p
_tmp_merge_df['f_path'] = all_train_images

df = df.merge(_tmp_merge_df, on="_partial_ident").drop(columns=["_partial_ident"])

# 5. Get slice dimensions from filepath (int in pixels)
df["slice_h"] = df["f_path"].apply(lambda x: int(x[:-4].rsplit("_",4)[1]))
df["slice_w"] = df["f_path"].apply(lambda x: int(x[:-4].rsplit("_",4)[2]))

# 6. Pixel spacing from filepath (float in mm)
df["px_spacing_h"] = df["f_path"].apply(lambda x: float(x[:-4].rsplit("_",4)[3]))
df["px_spacing_w"] = df["f_path"].apply(lambda x: float(x[:-4].rsplit("_",4)[4]))

df1 = df[df.index % 3 == 0]
df2 = df[df.index % 3 == 1]
df3 = df[df.index % 3 == 2]
df = df1.copy()
df.pop('class')
gc.collect()

x = df1.pop('segmentation')
x1 = df2.pop('segmentation')
x2 = df3.pop('segmentation')
df['large_bowel_segmentation'] = x.values
df['small_bowel_segmentation'] = x1.values
df['stomach_segmentation'] = x2.values
df.pop('segmentation')
del x, x2, x1, df1, df2, df3
gc.collect()
df = df.reset_index(drop=True)

un, co = np.unique(df['case_id'], return_counts=True)
pdf = pd.DataFrame()
pdf['cases'] = un
pdf['count'] = co

skf = KFold(n_splits=5)
i = 0
for fold, (train_index, test_index) in enumerate(skf.split(un, co)):
    pdf.loc[test_index, "kfold"] = fold
    i += 1
    
df['kfold'] = -1
for i in range(0, 5):
    df.loc[df[df['case_id'].isin(pdf[pdf['kfold'] == i]['cases'].values)].index, 'kfold'] = i
print (df['kfold'].value_counts())

rle_paths = []
image_paths = []
for index, row in tqdm(df.iterrows(), total=df.shape[0]):
    rle_paths.append(os.path.join("train_masks", row['id'] + '.npy'))
    image_paths.append(os.path.join("train_images", row['id'] + '.png'))
df['mask_path'] = rle_paths
df['image_path'] = image_paths


# # Transformations

# In[ ]:


transform_train = A.Compose([A.Resize(CFG['image_size'], CFG['image_size'], interpolation=cv2.INTER_NEAREST), A.HorizontalFlip(), A.VerticalFlip(), 
                     A.GridDistortion(p=0.2), A.GaussNoise()])
transform_val = A.Compose([A.Resize(CFG['image_size'], CFG['image_size'], interpolation=cv2.INTER_NEAREST)])


# # Train One Fold Function

# In[ ]:


def train_one_fold(fold):

    train_set = TractDataset(df[df['kfold']!=fold].reset_index(drop=True), 3, transform_train)
    val_set = TractDataset(df[df['kfold']==fold].reset_index(drop=True), 3, transform_val)

    train_loader = DataLoader(train_set, batch_size=CFG['batch_size'], shuffle=True, num_workers=CFG['num_workers'])
    val_loader = DataLoader(val_set, batch_size=CFG['batch_size'], shuffle=False, num_workers=CFG['num_workers'])      


    model = smp.DeepLabV3Plus('efficientnet-b1', encoder_weights='imagenet', classes=3, activation=None)
    model = model.to(CFG['device'])
    model = nn.DataParallel(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=CFG["init_lr"]/CFG["warmup_factor"])
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, CFG["n_epochs"]-CFG['warmup_epo'])
    scheduler = GradualWarmupScheduler(optimizer, multiplier=CFG["warmup_factor"], total_epoch=CFG["warmup_epo"], after_scheduler=scheduler_cosine)


    max_vmetric = 0

    model_file = f'deep_lab_best_fold_{fold}.pth'
    final_file = f'deep_lab_final_fold_{fold}.pth'

    for epoch in range(1, CFG['n_epochs']+1):

        print(time.ctime(), 'Epoch:', epoch)
        print ("Training")
        model, train_loss = train_epoch(epoch, model, train_loader, optimizer)        
        scheduler.step(epoch-1)
        print ("Validating")
        val_loss, LOGITS, TARGETS = valid_epoch(epoch, model, val_loader)
        LOGITS = torch.sigmoid(LOGITS.cpu()).numpy()
        LOGITS = np.round(LOGITS)

        gc.collect()

        dice1 = []
        dice2 = []
        dice3 = []

        print ("calculating validation scores")
        for i in tqdm(range(0, LOGITS.shape[0]), total=LOGITS.shape[0]):
            dice1.append(dice(LOGITS[i, 0, :, :], TARGETS[i, 0, :, :]))
            dice2.append(dice(LOGITS[i, 1, :, :], TARGETS[i, 1, :, :]))
            dice3.append(dice(LOGITS[i, 2, :, :], TARGETS[i, 2, :, :]))

 
        dice1 = np.mean(dice1)
        dice2 = np.mean(dice2)
        dice3 = np.mean(dice3)


        h_dists1 = 1 - hd_dist(LOGITS[:, 0, :, :], TARGETS[:, 0, :, :])
        h_dists2 = 1 - hd_dist(LOGITS[:, 1, :, :], TARGETS[:, 1, :, :])
        h_dists3 = 1 - hd_dist(LOGITS[:, 2, :, :], TARGETS[:, 2, :, :])

        vdice = (dice1 + dice2 + dice3) / 3.0
        vhd = (h_dists1 + h_dists2 + h_dists3) / 3.0

        vmetric = (0.4 * vdice) + (0.6 * vhd)
        
        content = time.ctime() + ' ' + f'Fold {fold} Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, train loss: {train_loss:.4f}, valid loss: {(val_loss):.4f}, val_dice_large: {dice1:.5f}, val_dice_small: {dice2:.5f}, val_dice_stomach: {dice3:.5f}, val_dice: {vdice:.5f}, val_hd_large: {h_dists1:.5f}, val_hd_small: {h_dists2:.5f}, val_hd_stomach: {h_dists3:.5f}, val_hd: {vhd:.5f}, val_hd_dice: {vmetric:.5f}.'
        print(content)

        with open(f'deep_lab_log_{fold}_fold.txt', 'a') as appender:
            appender.write(content + '\n')

        if max_vmetric < vmetric:
            print('Metric Increased ({:.6f} --> {:.6f}).  Saving model ...'.format(max_vmetric, vmetric))
            torch.save(model.state_dict(), model_file)
            max_vmetric = vmetric

    torch.save(model.state_dict(), final_file)


# In[ ]:


for fold in range(0, 2):
    train_one_fold(fold)


# In[ ]:




