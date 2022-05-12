#!/usr/bin/env python
# coding: utf-8

# # [UW-Madison GI Tract Image Segmentation](https://www.kaggle.com/competitions/uw-madison-gi-tract-image-segmentation/)
# > Track healthy organs in medical scans to improve cancer treatment
# 
# <img src="https://storage.googleapis.com/kaggle-competitions/kaggle/27923/logos/header.png?t=2021-06-02-20-30-25">

# # ⚽ Methodlogy
# <img src="https://i.ibb.co/sgsPf4v/Capture.png" width=800>
# <img src="https://i.ibb.co/KKtZ7Gn/Picture1-3d.png" width=500>
# 
# * In this notebook I'll demonstrate how to train using 2.5D images with **Unet** model using PyTorch.
# * 2.5D images take leverage of the extra depth information like our typical RGB image.
# * In this notebook I'll be using 3 channels with 2 strides for 2.5D images
# * Instead of Resize I'll be using Padding to avoid info loss.
# * For mask I'll be using pre-computed 2.5D images & mask from [here](https://www.kaggle.com/code/awsaf49/uwmgi-2-5d-stride-2-dataset)
# * As there are overlaps between **Stomach**, **Large Bowel** & **Small Bowel** classes, this is a **MultiLabel Segmentation** task, so final activaion should be `sigmoid` instead of `softmax`.
# * For data split I'll be using **StratifiedGroupFold** to avoid data leakage due to `case` and to stratify `empty` and `non-empty` mask cases.
# * You can play with different models and losses.

# # 🚩 Version Info

# # 📒 Notebooks
# 📌 **2.5D**:
# * Train: [UWMGI: 2.5D [Train] [PyTorch]](https://www.kaggle.com/awsaf49/uwmgi-2-5d-train-pytorch/)
# * Infer: [UWMGI: 2.5D [Infer] [PyTorch]](https://www.kaggle.com/awsaf49/uwmgi-2-5d-infer-pytorch/)
# * Data: [UWMGI: 2.5D stride=2 Data](https://www.kaggle.com/code/awsaf49/uwmgi-2-5d-stride-2-data/)
# 
# 📌 **UNet**:
# * Train: [UWMGI: Unet [Train] [PyTorch]](https://www.kaggle.com/code/awsaf49/uwmgi-unet-train-pytorch/)
# * Infer: [UWMGI: Unet [Infer] [PyTorch]](https://www.kaggle.com/code/awsaf49/uwmgi-unet-infer-pytorch/)
# 
# 📌 **MMDetection**:
# * Train: [UWMGI: MMDetection [Train]](https://www.kaggle.com/code/awsaf49/uwmgi-mmdetection-train)
# 
# 📌 **Data/Dataset**:
# * Data: [UWMGI: Mask Data](https://www.kaggle.com/datasets/awsaf49/uwmgi-mask-data)
# * Dataset: [UWMGI: Mask Dataset](https://www.kaggle.com/datasets/awsaf49/uwmgi-mask-dataset)

# ## Please Upvote if you Find this Useful :)

# # 🛠 Install Libraries

# In[ ]:


get_ipython().system('pip install -q ../input/pytorch-segmentation-models-lib/pretrainedmodels-0.7.4/pretrainedmodels-0.7.4')
get_ipython().system('pip install -q ../input/pytorch-segmentation-models-lib/efficientnet_pytorch-0.6.3/efficientnet_pytorch-0.6.3')
get_ipython().system('pip install -q ../input/pytorch-segmentation-models-lib/timm-0.4.12-py3-none-any.whl')
get_ipython().system('pip install -q ../input/pytorch-segmentation-models-lib/segmentation_models_pytorch-0.2.0-py3-none-any.whl')


# # 📚 Import Libraries 

# In[ ]:


import numpy as np
import pandas as pd
pd.options.plotting.backend = "plotly"
import random
from glob import glob
import os, shutil
from tqdm import tqdm
tqdm.pandas()
import time
import copy
import joblib
from collections import defaultdict
import gc
from IPython import display as ipd

# visualization
import cv2
import matplotlib.pyplot as plt

# Sklearn
from sklearn.model_selection import StratifiedKFold, KFold

# PyTorch 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torch.cuda import amp
import torch.nn.functional as F

import timm

# Albumentations for augmentations
import albumentations as A
from albumentations.pytorch import ToTensorV2

# For colored terminal text
from colorama import Fore, Back, Style
c_  = Fore.GREEN
sr_ = Style.RESET_ALL

import warnings
warnings.filterwarnings("ignore")

# For descriptive error messages
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


# # ⚙️ Configuration 

# In[ ]:


class CFG:
    seed          = 101
    debug         = False # set debug=False for Full Training
    exp_name      = 'v4'
    comment       = 'unet-efficientnet_b0-320x384'
    model_name    = 'Unet'
    backbone      = 'efficientnet-b0'
    train_bs      = 64
    valid_bs      = train_bs*2
    img_size      = [320, 384]
    epochs        = 16
    lr            = 2e-3
    scheduler     = 'CosineAnnealingLR'
    min_lr        = 1e-6
    T_max         = int(30000/train_bs*epochs)+50
    T_0           = 25
    warmup_epochs = 0
    wd            = 1e-6
    n_accumulate  = max(1, 32//train_bs)
    n_fold        = 5
    folds         = [0]
    num_classes   = 3
    thr           = 0.40
    device        = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# # ❗ Reproducibility

# In[ ]:


def set_seed(seed = 42):
    '''Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.'''
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    print('> SEEDING DONE')
    
set_seed(CFG.seed)


# # 🔨 Utility

# In[ ]:


def get_metadata(row):
    data = row['id'].split('_')
    case = int(data[0].replace('case',''))
    day = int(data[1].replace('day',''))
    slice_ = int(data[-1])
    row['case'] = case
    row['day'] = day
    row['slice'] = slice_
    return row

def path2info(row):
    path = row['image_path']
    data = path.split('/')
    slice_ = int(data[-1].split('_')[1])
    case = int(data[-3].split('_')[0].replace('case',''))
    day = int(data[-3].split('_')[1].replace('day',''))
    width = int(data[-1].split('_')[2])
    height = int(data[-1].split('_')[3])
    row['height'] = height
    row['width'] = width
    row['case'] = case
    row['day'] = day
    row['slice'] = slice_
#     row['id'] = f'case{case}_day{day}_slice_{slice_}'
    return row


# In[ ]:


def load_img(path, size=CFG.img_size):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    shape0 = np.array(img.shape[:2])
    resize = np.array(size)
    if np.any(shape0!=resize):
        diff = resize - shape0
        pad0 = diff[0]
        pad1 = diff[1]
        pady = [pad0//2, pad0//2 + pad0%2]
        padx = [pad1//2, pad1//2 + pad1%2]
        img = np.pad(img, [pady, padx])
        img = img.reshape((*resize))
    return img, shape0

def load_imgs(img_paths, size=CFG.img_size):
    imgs = np.zeros((*size, len(img_paths)), dtype=np.float32)
    for i, img_path in enumerate(img_paths):
        if i==0:
            img, shape0 = load_img(img_path, size=size)
        else:
            img, _ = load_img(img_path, size=size)
        img = img.astype('float32') # original is uint16
        mx = np.max(img)
        if mx:
            img/=mx # scale image to [0, 1]
        imgs[..., i]+=img
    return imgs, shape0

def load_msk(path, size=CFG.img_size):
    msk = np.load(path)
    shape0 = np.array(msk.shape[:2])
    resize = np.array(size)
    if np.any(shape0!=resize):
        diff = resize - shape0
        pad0 = diff[0]
        pad1 = diff[1]
        pady = [pad0//2, pad0//2 + pad0%2]
        padx = [pad1//2, pad1//2 + pad1%2]
        msk = np.pad(msk, [pady, padx, [0,0]])
        msk = msk.reshape((*resize, 3))
    msk = msk.astype('float32')
    msk/=255.0
    return msk

def show_img(img, mask=None):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(img)
    plt.imshow(img, cmap='bone')
    
    if mask is not None:
        # plt.imshow(np.ma.masked_where(mask!=1, mask), alpha=0.5, cmap='autumn')
        plt.imshow(mask, alpha=0.5)
        handles = [Rectangle((0,0),1,1, color=_c) for _c in [(0.667,0.0,0.0), (0.0,0.667,0.0), (0.0,0.0,0.667)]]
        labels = ["Large Bowel", "Small Bowel", "Stomach"]
        plt.legend(handles,labels)
    plt.axis('off')


# In[ ]:


# ref: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
def rle_decode(mask_rle, shape):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)  # Needed to align to RLE direction


# ref.: https://www.kaggle.com/stainsby/fast-tested-rle
def rle_encode(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


# # 📖 Meta Data

# In[ ]:


BASE_PATH  = '/kaggle/input/uw-madison-gi-tract-image-segmentation'
CKPT_DIR = '/kaggle/input/uwmgi-unet-effnetb0-25d-320x384-ckpt-ds'


# ## Train

# In[ ]:


# df = pd.read_csv('../input/uwmgi-mask-dataset/uw-madison-gi-tract-image-segmentation/train.csv')
# df['empty'] = df.segmentation.map(lambda x: int(pd.isna(x)))

# df2 = df.groupby(['id'])['class'].agg(list).to_frame().reset_index()
# df2 = df2.merge(df.groupby(['id'])['segmentation'].agg(list), on=['id'])
# # df = df[['id','case','day','image_path','mask_path','height','width', 'empty']]

# df = df.drop(columns=['segmentation', 'class'])
# df = df.groupby(['id']).head(1).reset_index(drop=True)
# df = df.merge(df2, on=['id'])
# df.head()


# ## Test

# In[ ]:


sub_df = pd.read_csv('../input/uw-madison-gi-tract-image-segmentation/sample_submission.csv')
if not len(sub_df):
    debug = True
    sub_df = pd.read_csv('../input/uw-madison-gi-tract-image-segmentation/train.csv')
    sub_df = sub_df[~sub_df.segmentation.isna()][:1000*3]
    sub_df = sub_df.drop(columns=['class','segmentation']).drop_duplicates()
else:
    debug = False
    sub_df = sub_df.drop(columns=['class','predicted']).drop_duplicates()
sub_df = sub_df.progress_apply(get_metadata,axis=1)


# In[ ]:


if debug:
    paths = glob(f'/kaggle/input/uw-madison-gi-tract-image-segmentation/train/**/*png',recursive=True)
#     paths = sorted(paths)
else:
    paths = glob(f'/kaggle/input/uw-madison-gi-tract-image-segmentation/test/**/*png',recursive=True)
#     paths = sorted(paths)
path_df = pd.DataFrame(paths, columns=['image_path'])
path_df = path_df.progress_apply(path2info, axis=1)
path_df.head()


# ## Merge Data

# In[ ]:


test_df = sub_df.merge(path_df, on=['case','day','slice'], how='left')
test_df.head()


# ## 2.5D MetaData

# In[ ]:


channels=3
stride=2
for i in range(channels):
    test_df[f'image_path_{i:02}'] = test_df.groupby(['case','day'])['image_path'].shift(-i*stride).fillna(method="ffill")
test_df['image_paths'] = test_df[[f'image_path_{i:02d}' for i in range(channels)]].values.tolist()
if debug:
    test_df = test_df.sample(frac=1.0)
test_df.image_paths[0]


# # 🍚 Dataset

# In[ ]:


class BuildDataset(torch.utils.data.Dataset):
    def __init__(self, df, label=False, transforms=None):
        self.df         = df
        self.label      = label
        self.img_paths  = df['image_paths'].tolist()
        self.ids        = df['id'].tolist()
        if 'msk_path' in df.columns:
            self.msk_paths  = df['mask_path'].tolist()
        else:
            self.msk_paths = None
        self.transforms = transforms
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        img_path  = self.img_paths[index]
        id_       = self.ids[index]
        img = []
        img, shape0 = load_imgs(img_path)
        h, w = shape0
        if self.label:
            msk_path = self.msk_paths[index]
            msk = load_msk(msk_path)
            if self.transforms:
                data = self.transforms(image=img, mask=msk)
                img  = data['image']
                msk  = data['mask']
            img = np.transpose(img, (2, 0, 1))
            msk = np.transpose(msk, (2, 0, 1))
            return torch.tensor(img), torch.tensor(msk)
        else:
            if self.transforms:
                data = self.transforms(image=img)
                img  = data['image']
            img = np.transpose(img, (2, 0, 1))
            return torch.tensor(img), id_, h, w


# # 🌈 Augmentations

# In[ ]:


data_transforms = {
    "train": A.Compose([
#         A.Resize(*CFG.img_size, interpolation=cv2.INTER_NEAREST),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
#         A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.05, rotate_limit=5, p=0.5),
        A.OneOf([
            A.GridDistortion(num_steps=5, distort_limit=0.05, p=1.0),
# #             A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=1.0),
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0)
        ], p=0.25),
#         A.CoarseDropout(max_holes=8, max_height=CFG.img_size[0]//20, max_width=CFG.img_size[1]//20,
#                          min_holes=5, fill_value=0, mask_fill_value=0, p=0.5),
        ], p=1.0),
    
    "valid": A.Compose([
#         A.Resize(*CFG.img_size, interpolation=cv2.INTER_NEAREST),
        ], p=1.0)
}


# # 🍰 DataLoader

# In[ ]:


# test_dataset = BuildDataset(test_df, transforms=data_transforms['valid'])
# test_loader  = DataLoader(test_dataset, batch_size=64, 
#                           num_workers=4, shuffle=False, pin_memory=True)


# In[ ]:


# imgs, ids, (h, w) = next(iter(test_loader))
# imgs = imgs.permute((0, 2, 3, 1))
# imgs.size()


# # 📦 Model
# 

# ## UNet
# 
# <img src="https://developers.arcgis.com/assets/img/python-graphics/unet.png" width="600">
# 
# 📌 **Pros**:
# * Performs well even with smaller data
# * Can be used with `imagenet` pretrain models
# 
# 📌 **Cons**:
# * Struggles with **edge** cases
# * Semantic Difference in **Skip Connection**

# In[ ]:


import segmentation_models_pytorch as smp

def build_model():
    model = smp.Unet(
        encoder_name=CFG.backbone,      # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights=None,     # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=CFG.num_classes,        # model output channels (number of classes in your dataset)
        activation=None,
    )
    model.to(CFG.device)
    return model

def load_model(path):
    model = build_model()
    model.load_state_dict(torch.load(path))
    model.eval()
    return model


# In[ ]:


# # test
# img = torch.randn(1, 1, *CFG.img_size).to(CFG.device)
# img = (img - img.min())/(img.max() - img.min())
# model = build_model()
# _ = model(img)


# # 🔨 Helper

# In[ ]:


import cupy as cp

def mask2rle(msk, thr=0.5):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    msk    = cp.array(msk)
    pixels = msk.flatten()
    pad    = cp.array([0])
    pixels = cp.concatenate([pad, pixels, pad])
    runs   = cp.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def masks2rles(msks, ids, heights, widths):
    pred_strings = []; pred_ids = []; pred_classes = [];
    for idx in range(msks.shape[0]):
        msk = msks[idx]
        height = heights[idx].item()
        width = widths[idx].item()
        shape0 = np.array([height, width])
        resize = np.array([320, 384])
        if np.any(shape0!=resize):
            diff = resize - shape0
            pad0 = diff[0]
            pad1 = diff[1]
            pady = [pad0//2, pad0//2 + pad0%2]
            padx = [pad1//2, pad1//2 + pad1%2]
            msk = msk[pady[0]:-pady[1], padx[0]:-padx[1], :]
            msk = msk.reshape((*shape0, 3))
        rle = [None]*3
        for midx in [0, 1, 2]:
            rle[midx] = mask2rle(msk[...,midx])
        pred_strings.extend(rle)
        pred_ids.extend([ids[idx]]*len(rle))
        pred_classes.extend(['large_bowel', 'small_bowel', 'stomach'])
    return pred_strings, pred_ids, pred_classes


# # 🔭 Inference

# In[ ]:


@torch.no_grad()
def infer(model_paths, test_loader, num_log=1, thr=CFG.thr):
    msks = []; imgs = [];
    pred_strings = []; pred_ids = []; pred_classes = [];
    for idx, (img, ids, heights, widths) in enumerate(tqdm(test_loader, total=len(test_loader), desc='Infer ')):
        img = img.to(CFG.device, dtype=torch.float) # .squeeze(0)
        size = img.size()
        msk = []
        msk = torch.zeros((size[0], 3, size[2], size[3]), device=CFG.device, dtype=torch.float32)
        for path in model_paths:
            model = load_model(path)
            out   = model(img) # .squeeze(0) # removing batch axis
            out   = nn.Sigmoid()(out) # removing channel axis
            msk+=out/len(model_paths)
        msk = (msk.permute((0,2,3,1))>thr).to(torch.uint8).cpu().detach().numpy() # shape: (n, h, w, c)
        result = masks2rles(msk, ids, heights, widths)
        pred_strings.extend(result[0])
        pred_ids.extend(result[1])
        pred_classes.extend(result[2])
        if idx<num_log and debug:
            img = img.permute((0,2,3,1)).cpu().detach().numpy()
            imgs.append(img[::5])
            msks.append(msk[::5])
        del img, msk, out, model, result
        gc.collect()
        torch.cuda.empty_cache()
    return pred_strings, pred_ids, pred_classes, imgs, msks


# In[ ]:


test_dataset = BuildDataset(test_df, transforms=data_transforms['valid'])
test_loader  = DataLoader(test_dataset, batch_size=CFG.valid_bs, 
                          num_workers=4, shuffle=False, pin_memory=False)
model_paths  = glob(f'{CKPT_DIR}/best_epoch*.bin')
pred_strings, pred_ids, pred_classes, imgs, msks = infer(model_paths, test_loader)


# # 📈 Visualization

# In[ ]:


if debug:
    for img, msk in zip(imgs[0][:5], msks[0][:5]):
        plt.figure(figsize=(12, 7))
        plt.subplot(1, 3, 1); plt.imshow(img, cmap='bone');
        plt.axis('OFF'); plt.title('image')
        plt.subplot(1, 3, 2); plt.imshow(msk*255); plt.axis('OFF'); plt.title('mask')
        plt.subplot(1, 3, 3); plt.imshow(img, cmap='bone'); plt.imshow(msk*255, alpha=0.4);
        plt.axis('OFF'); plt.title('overlay')
        plt.tight_layout()
        plt.show()


# In[ ]:


del imgs, msks
gc.collect()


# # 📝 Submission

# In[ ]:


pred_df = pd.DataFrame({
    "id":pred_ids,
    "class":pred_classes,
    "predicted":pred_strings
})
if not debug:
    sub_df = pd.read_csv('../input/uw-madison-gi-tract-image-segmentation/sample_submission.csv')
    del sub_df['predicted']
else:
    sub_df = pd.read_csv('../input/uw-madison-gi-tract-image-segmentation/train.csv')[:1000*3]
    del sub_df['segmentation']
    
sub_df = sub_df.merge(pred_df, on=['id','class'])
sub_df.to_csv('submission.csv',index=False)
display(sub_df.head(5))


# In[ ]:




