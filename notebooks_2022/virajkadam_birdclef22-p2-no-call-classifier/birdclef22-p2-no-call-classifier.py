#!/usr/bin/env python
# coding: utf-8

# # Training a Classifier to detect No-call
# 
# 
# * credits to : https://github.com/namakemono/kaggle-birdclef-2021/blob/master/share_solution/working/build_nocall_detector.ipynb

# # About 
# 
#     After clipping the audio signals into smaller chunks, a lot of the chunks do not have any audio signal in them as a lot of the audios are sparsely populated with primary bird call. Hence this will harm the training as it will make the model learn incorrectly on signal that is not at all present in the spectrogram.
#     
#     
#     Hence to avoid this scenario, we can train a no-call classifier, which classifies if a spectrogram has a bird call signal present or not. We can use this classifier on the spectrograms of the birclef 22 data, and then construct labels depending upon if there is a bird call signal present in the spectrogram, and if the probablity of no call is high , we can train it with a "no-call" label. 

# 
# 
# **Freefield Data**
# 
# 
#     This dataset contains 7690 10-second audio files in a standardised format, extracted from contributions on the Freesound archive which were labelled with the "field-recording" tag. Note that the original tagging (as well as the audio submission) is crowdsourced, so the dataset is not guaranteed to consist purely of "field recordings" as might be defined by practitioners. The intention is to represent the content of an archive collection on such a topic, rather than to represent a controlled definition of such a topic.
# 
#     Each audio file has a corresponding text file, containing metadata such as author and tags. The dataset has been randomly split into 10 equal-size subsets. This is so that you can perform 10-fold crossvalidation in machine-learning experiments, or can use fixed subsets of the data (e.g. use one subset for development, and others for later validation). Each of the 10 subsets has about 128 minutes of audio; the dataset totals over 21 hours of audio.
# 

# # Imports

# In[ ]:



import os
import math
import time
import random
import shutil
from pathlib import Path
from contextlib import contextmanager
from collections import defaultdict, Counter


import scipy as sp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold

from tqdm.auto import tqdm
from functools import partial

import cv2
from PIL import Image

#torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
import torchvision.models as models
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


#augmentations
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations import ImageOnlyTransform


#getting models
get_ipython().system('pip install timm -q')
import timm

import warnings 
warnings.filterwarnings('ignore')


# # Config

# In[ ]:



class CFG:
    print_freq=50
    num_workers=4
    model_name= 'resnext50_32x4d'
    dim=(128, 281)
    scheduler='CosineAnnealingWarmRestarts'
    epochs=15
    lr=1e-4
    T_0= 5 # for CosineAnnealingWarmRestarts
    min_lr=5e-7 # for CosineAnnealingWarmRestarts
    batch_size=32
    weight_decay=1e-5
    max_grad_norm=100
    seed=7
    target_size=2
    target_col='hasbird'
    n_fold = 5
    pretrained = True
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def seed_torch(seed=7):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_torch(seed=CFG.seed)


# In[ ]:


#what device? 
CFG.device


# **Loading csv**
# 
#     Using the dataset and spectrograms by : https://www.kaggle.com/datasets/startjapan/ff1010bird-duration7

# In[ ]:


train = pd.read_csv('../input/ff1010bird-duration7/rich_metadata.csv')
train.loc[train['hasbird']==0, 'filepath'] = '../input/ff1010bird-duration7/nocall/' + train.query('hasbird==0')['filename'] + '.npy'
train.loc[train['hasbird']==1, 'filepath'] = '../input/ff1010bird-duration7/bird/' + train.query('hasbird==1')['filename'] + '.npy'

train = train.dropna().reset_index(drop=True)


# In[ ]:


# #freefield birdcall dir

# train_dir = '../input/birdclef22-p1-extracting-spectograms/Freefield_Spectrograms'
# OUTPUT_DIR = './'
# train = pd.read_csv('../input/birdclef22-p1-extracting-spectograms/freefield_downsampled.csv',
#                     usecols=['has_bird_call','id','filepath'])


# #add a filepath to retrive files
# train.filepath=train.filepath.apply(lambda x: train_dir +'/'+ x.split('/')[-1].replace('.wav','_0.jpg'))

# train.head()


# In[ ]:


train['hasbird'].value_counts()


# **Split data into folds**

# In[ ]:


folds = train.copy()
folds.reset_index(drop=True,inplace=True)

Fold = StratifiedKFold(n_splits=CFG.n_fold, shuffle=True, random_state=CFG.seed)
for n, (train_index, val_index) in enumerate(Fold.split(folds, folds[CFG.target_col])):
    folds.loc[val_index, 'fold'] = int(n)
    
    
folds['fold'] = folds['fold'].astype(int)
print(folds.groupby(['fold', CFG.target_col]).size())


# # Helper Functions

# In[ ]:


OUTPUT_DIR='./'


# In[ ]:


def get_score(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

def get_confusion_matrix(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)

@contextmanager
def timer(name):
    t0 = time.time()
    LOGGER.info(f'[{name}] start')
    yield
    LOGGER.info(f'[{name}] done in {time.time() - t0:.0f} s.')


def init_logger(log_file=OUTPUT_DIR+'train.log'):
    from logging import getLogger, INFO, FileHandler,  Formatter,  StreamHandler
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=log_file)
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger


LOGGER = init_logger()


# In[ ]:


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))


# **Training Dataset and Transformation**

# In[ ]:


class TrainDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.file_paths = df['filepath'].values
        self.labels = df[CFG.target_col].values
        self.transform = transform
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx] 
        
        image = np.load(file_path)
        image = image.transpose(1,2,0)
        image = np.squeeze(image)
        image = np.stack((image,)*3, -1)
        
        
        #read image
#         image = cv2.imread(file_path)
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
#         image = cv2.imread(file_path,cv2.IMREAD_GRAYSCALE)
#         image = np.dstack((image,image,image)) # stack to form rgb image(so we can use pretrained weights)
        
        #apply transformations if any
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
            
        
        # temp check 
#         fig,ax = plt.subplots(figsize=(10,6))
#         plt.imshow(image.reshape((128,512,3)))
#         plt.show()
        
        
        #return as tensor    
        label = torch.tensor(self.labels[idx]).long()
        return image, label
    
def get_transforms(*, data):
    
    if data == 'train':
        return A.Compose([
            A.Resize(CFG.dim[0], CFG.dim[1]),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.augmentations.transforms.JpegCompression(p=0.5),
            A.augmentations.transforms.ImageCompression(p=0.5, compression_type=A.augmentations.transforms.ImageCompression.ImageCompressionType.WEBP),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])

    elif data == 'valid':
        return A.Compose([
            A.Resize(CFG.dim[0], CFG.dim[1]),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])


# **Checking some sample spectrograms**

# In[ ]:


train_dataset = TrainDataset(train, 
                             transform=get_transforms(data='train'))


fig,ax = plt.subplots(5,1,figsize=(15,30))
for i,ax1 in enumerate(ax):
    i += np.random.randint(low=0,high = 1500)
    image, label = train_dataset[i]
    
    
    # reshape for visualization (from depth * height * width -> height*width * depth)
    im = np.zeros(shape = (CFG.dim[0],CFG.dim[1],3))  
    im[:,:,0] = image[0]
    im[:,:,1] = image[1]
    im[:,:,2] = image[2]
    
    ax1.imshow(im,cmap='jet')
    ax1.set_title(f'label: {label}')

plt.tight_layout()    
plt.show() 


# # **Model**

# In[ ]:




class CustomResNext(nn.Module):
    def __init__(self, model_name='resnext50_32x4d', pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        n_features = self.model.fc.in_features
        self.model.fc = nn.Linear(n_features, CFG.target_size)

    def forward(self, x):
        x = self.model(x)
        return x


# In[ ]:


def get_scheduler(optimizer):
    '''cosine annealing scheduler'''
    scheduler = CosineAnnealingWarmRestarts(optimizer, 
                                            T_0=CFG.T_0, 
                                            T_mult=1, 
                                            eta_min=CFG.min_lr, 
                                            last_epoch=-1)
    return scheduler


# **Training and validation Functions**

# In[ ]:



def train_fn(train_loader, model, criterion, optimizer, epoch, scheduler, device):
    '''perform training on one epoch of data.'''
    
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    scores = AverageMeter()
    
        
    # switch to train mode
    model.train()
    start = end = time.time()
    global_step = 0
    for step, (images, labels) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        
        #load data
        images = images.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)
        
        #forward pass
        y_preds = model(images)
        
        #calculate loss
        loss = criterion(y_preds, labels)
        
        
        # record loss
        losses.update(loss.item(), batch_size)
        loss.backward()
        
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                   CFG.max_grad_norm)
        
        optimizer.step()
        optimizer.zero_grad()
        global_step += 1
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if step % CFG.print_freq == 0 or step == (len(train_loader)-1):
            print('Epoch: [{0}][{1}/{2}] '
                  'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  'Grad: {grad_norm:.4f}  '
                  .format(
                   epoch+1, step+1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses,
                   remain=timeSince(start, float(step+1)/len(train_loader)),
                   grad_norm=grad_norm,
                   ))
            
    return losses.avg

def valid_fn(valid_loader, model, criterion, device):
    '''perform validation'''
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    scores = AverageMeter()
    # switch to evaluation mode
    model.eval()
    preds = []
    start = end = time.time()
    for step, (images, labels) in enumerate(valid_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        images = images.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)
        # compute loss
        with torch.no_grad():
            y_preds = model(images)
        loss = criterion(y_preds, labels)
        losses.update(loss.item(), batch_size)
        
        
        # record accuracy
        preds.append(y_preds.softmax(1).to('cpu').numpy())
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        if step % CFG.print_freq == 0 or step == (len(valid_loader)-1):
            print('EVAL: [{0}/{1}] '
                  'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  .format(
                   step+1, len(valid_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses,
                   remain=timeSince(start, float(step+1)/len(valid_loader)),
                   ))
    predictions = np.concatenate(preds)
    return losses.avg, predictions


def inference(model, states, test_loader, device):
    '''inference'''
    model.to(device)
    tk0 = tqdm(enumerate(test_loader), total=len(test_loader))
    probs = []
    for i, (images) in tk0:
        images = images.to(device)
        avg_preds = []
        for state in states:
            model.load_state_dict(state['model'])
            model.eval()
            with torch.no_grad():
                y_preds = model(images)
            avg_preds.append(y_preds.softmax(1).to('cpu').numpy())
        avg_preds = np.mean(avg_preds, axis=0)
        probs.append(avg_preds)
    probs = np.concatenate(probs)
    return probs


# **Training Procedure / loop**

# In[ ]:



def train_loop(train_folds, valid_folds):

    LOGGER.info(f"========== training ==========")

    # ====================================================
    # loader
    # ====================================================
    train_dataset = TrainDataset(train_folds, 
                                 transform=get_transforms(data='train'))
    valid_dataset = TrainDataset(valid_folds, 
                                 transform=get_transforms(data='valid'))

    train_loader = DataLoader(train_dataset, 
                              batch_size=CFG.batch_size, 
                              shuffle=False, 
                              num_workers=CFG.num_workers, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, 
                              batch_size=CFG.batch_size, 
                              shuffle=False, 
                              num_workers=CFG.num_workers, pin_memory=True, drop_last=False)
    
    # ====================================================
    # model & optimizer
    # ====================================================
    model = CustomResNext(CFG.model_name, pretrained=True)
    model.to(CFG.device)
    
    optimizer = Adam(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay, amsgrad=False)
    scheduler = get_scheduler(optimizer)

    # ====================================================
    # loop
    # ====================================================
    criterion = nn.CrossEntropyLoss()

    best_score = 0.
    best_loss = np.inf
    
    scores = []
    
    for epoch in range(CFG.epochs):
        
        start_time = time.time()
        
        # train
        avg_loss = train_fn(train_loader, 
                            model, 
                            criterion, 
                            optimizer, 
                            epoch, 
                            scheduler, 
                            CFG.device)
        
        # eval
        avg_val_loss, preds = valid_fn(valid_loader, model, criterion, CFG.device)
        valid_labels = valid_folds[CFG.target_col].values
        
        scheduler.step()

        # scoring
        score = get_score(valid_labels, preds.argmax(1))

        elapsed = time.time() - start_time

        LOGGER.info(f'Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s')
        LOGGER.info(f'Epoch {epoch+1} - Accuracy(validation): {score}')
        
        scores.append(score)
        
        
        # save the model weights with the best score 
        if score > best_score:
            best_score = score
            LOGGER.info(f'Epoch {epoch+1} - Save Best Score: {best_score:.4f} Model')
            torch.save({'model': model.state_dict(), 
                        'preds': preds},
                        OUTPUT_DIR+f'{CFG.model_name}_best.pth')
    
    check_point = torch.load(OUTPUT_DIR+f'{CFG.model_name}_best.pth')
    valid_folds[[str(c) for c in range(CFG.target_size)]] = check_point['preds']
    valid_folds['preds'] = check_point['preds'].argmax(1)

    return valid_folds, scores


# In[ ]:


def get_result(result_df):
    
    preds = result_df['preds'].values
    labels = result_df[CFG.target_col].values
    score = get_score(labels, preds)
    LOGGER.info(f'Score: {score:<.5f}')

def get_confusion_mat(result_df):
    preds = result_df['preds'].values
    labels = result_df[CFG.target_col].values
    matrix = get_confusion_matrix(labels, preds)
    print('TN', matrix[0,0])
    print('FP', matrix[0,1])
    print('FN', matrix[1,0])
    print('TP', matrix[1,1])


# In[ ]:




def main(fold):
    '''run training on the dataset, with validation on the input fold'''
    
    # train 
    train_folds = folds.query(f'fold!={fold}').reset_index(drop=True)
    valid_folds = folds.query(f'fold=={fold}').reset_index(drop=False)
    oof_df, scores = train_loop(train_folds, valid_folds)
    
    
    # CV result
    LOGGER.info(f"========== CV ==========")
    get_result(oof_df)
    get_confusion_mat(oof_df)
    
    
    
    # save result
    oof_df.to_csv(OUTPUT_DIR+'oof_df.csv', index=False)
    plt.plot([i for i in range(CFG.epochs)], scores)
    plt.title('valid score')
    plt.show()


# # Training

# **Training with fold 5 as the validation set**

# In[ ]:


if __name__ == '__main__':
    main(0)


# **We can use the saved model on the birdclef spectrograms to filter out the no-bird calls for training.**
