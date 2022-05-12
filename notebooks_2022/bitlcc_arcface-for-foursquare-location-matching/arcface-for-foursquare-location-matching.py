#!/usr/bin/env python
# coding: utf-8

# # Arcface for Foursquare - Location Matching

# Though it's a save&run version on Kaggle, I recommend running offline. Not only does it require lots of time to converge, but also it's a raw version with parameters to be finetuned, e. g. knn, threshold, epochs. I think it requires dozens of epochs to acquire good result, but I believe it will be better than the baseline(e. g. lgb, xgb).

# # CFG

# In[ ]:


import wandb


# In[ ]:


class CFG:
    model = 'microsoft/deberta-v3-base'
    wandb=False
    seed=42
    epochs=30
    emb_size=128
    bs = 32
    start_epoch = 5
    knn=3
    threshold=.99
    
    lr = 1e-5
    eps=1e-6
    betas=(0.9, 0.999)
    
    max_grad_norm=1000
    scheduler='CosineAnnealingLR'
    T_max=500
    weight_decay=1e-6
    min_lr=1e-6
    
    folds=2
    
    s=30
    m=.5
    ls_eps=0
    easy_margin=False
    
    n_neighbors = 500


# In[ ]:


tokenizer_path = 'tokenizer/'
root_path = '../input/foursquare-location-matching/'


# # Set Seed

# In[ ]:


import numpy as np
import torch
import os
import gc

def set_seed(seed=42):
    '''Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.'''
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    
set_seed(CFG.seed)


# # Tokenizer

# In[ ]:


from transformers import AutoTokenizer, AutoModel, AutoConfig


# In[ ]:


if os.path.exists(tokenizer_path):
    CFG.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
else:
    CFG.tokenizer = AutoTokenizer.from_pretrained(CFG.model)
    CFG.tokenizer.save_pretrained('tokenizer')


# # Read In Data

# In[ ]:


import pandas as pd


# In[ ]:


df = pd.read_csv(root_path + 'train.csv')
display(df.head())
df.shape


# In[ ]:


df['categories'] = df['categories'].fillna('')
df['name'] = df['name'].fillna('')
df['country'] = df['country'].fillna('')


# In[ ]:


from sklearn.preprocessing import LabelEncoder


# In[ ]:


encoder = LabelEncoder()
df['point_of_interest'] = encoder.fit_transform(df['point_of_interest'])
CFG.num_classes = df['point_of_interest'].nunique()
CFG.num_classes


# In[ ]:


df['text'] = df['name'] + '[SEP]' + df['categories'] + '[SEP]' + df['country']
df = df[['text', 'latitude', 'longitude', 'point_of_interest']]


# # CV Split

# In[ ]:


from sklearn.model_selection import GroupKFold


# In[ ]:


gkf = GroupKFold(n_splits=CFG.folds)

for fold, ( _, val_) in enumerate(gkf.split(X=df, y=df.point_of_interest, groups=df.point_of_interest)):
      df.loc[val_ , "fold"] = fold


# # Dataset

# In[ ]:


from torch.utils.data import DataLoader, Dataset


# In[ ]:


class FourSquareDataset(Dataset):
    def __init__(self, df):
        self.text = df['text'].values
        self.latitudes = df['latitude'].values
        self.longitudes = df['longitude'].values
        self.labels = df['point_of_interest'].values
        
    def __len__(self):
        return len(self.text)
    
    def __getitem__(self, index):
        text = self.text[index]
        latitude = self.latitudes[index]
        longitude = self.longitudes[index]
        label = self.labels[index]
        return text, torch.tensor(latitude).float(), torch.tensor(longitude).float(), torch.tensor(label).long()


# # Arcface

# In[ ]:


import math
import torch.nn as nn


# In[ ]:


class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
        """
    def __init__(self, in_features, out_features, s=30.0, 
                 m=0.50, easy_margin=False, ls_eps=0.0):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.ls_eps = ls_eps  # label smoothing
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size()).cuda()
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features
        # -------------torch.where(out_i = {x_i if condition_i else y_i) ------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output


# # Model

# In[ ]:


import torch.nn.functional as F


# In[ ]:


class FourSquareModel(nn.Module):
    def __init__(self, CFG):
        super(FourSquareModel, self).__init__()
        self.config = AutoConfig.from_pretrained(CFG.model)
        torch.save(self.config, 'config.pth')
        self.model = AutoModel.from_pretrained(CFG.model, 
                                               config=self.config)
        self.embedding = nn.Linear(self.config.hidden_size + 2, CFG.emb_size)
        self.fc = ArcMarginProduct(CFG.emb_size, 
                                   CFG.num_classes,
                                   s=CFG.s, 
                                   m=CFG.m, 
                                   easy_margin=CFG.easy_margin, 
                                   ls_eps=CFG.ls_eps)
        self.attention = nn.Sequential(
            nn.Linear(self.config.hidden_size, 512),
            nn.Tanh(),
            nn.Linear(512, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, text, lat, long, labels):
        out = self.model(**text)
        features = torch.cat([self.feature(out[0]), lat.view(-1, 1), long.view(-1, 1)], axis=1)
        embedding = self.embedding(features)
        output = self.fc(embedding, labels)
        return output
    
    def feature(self, last_hidden_states):
        # feature = torch.mean(last_hidden_states, 1)
        weights = self.attention(last_hidden_states)
        feature = torch.sum(weights * last_hidden_states, dim=1)
        return feature
    
    def extract(self, text, lat, long):
        out = self.model(**text)
        features = torch.cat([self.feature(out[0]), lat.view(-1, 1), long.view(-1, 1)], axis=1)
        embedding = self.embedding(features)
        embedding = F.normalize(embedding)
        return embedding


# # Helper Func

# In[ ]:


from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts


# In[ ]:


def get_optimizer_params(model, lr, weight_decay=0.0):
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {'params': [p for n, p in model.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'lr': lr, 'weight_decay': weight_decay},
            {'params': [p for n, p in model.model.named_parameters() if any(nd in n for nd in no_decay)],
             'lr': lr, 'weight_decay': 0.0},
            {'params': [p for n, p in model.named_parameters() if "model" not in n],
             'lr':lr, 'weight_decay': 0.0}
        ]
        return optimizer_parameters

def get_scheduler(optimizer):
    if CFG.scheduler == 'CosineAnnealingLR':
        scheduler = CosineAnnealingLR(optimizer,T_max=CFG.T_max, 
                                                   eta_min=CFG.min_lr)
    elif CFG.scheduler == 'CosineAnnealingWarmRestarts':
        scheduler = CosineAnnealingWarmRestarts(optimizer,T_0=CFG.T_0, 
                                                             eta_min=CFG.min_lr)
    elif CONFIG['scheduler'] == None:
        return None
        
    return scheduler


# # Train Func

# In[ ]:


from tqdm import tqdm


# In[ ]:


def train_one_epoch(model, optimizer, scheduler, dataloader,epoch):
    model.train()
    
    dataset_size = 0
    running_loss = 0.0
    
    criterion = nn.CrossEntropyLoss()
    
    bar = tqdm(dataloader, total=len(dataloader))
    for text, latitude, longitude, label in bar:
        text = CFG.tokenizer(text, padding=True, return_tensors='pt')
        for k,v in text.items():
            text[k] = v.cuda()
        latitude = latitude.cuda()
        longitude = longitude.cuda()
        labels = label.cuda()
        
        batch_size = labels.size(0)
        
        outputs = model(text, latitude, longitude, labels)
        loss = criterion(outputs, labels)
        loss.backward()
        
        nn.utils.clip_grad_norm_(model.parameters(), CFG.max_grad_norm)
    
        optimizer.step()

        optimizer.zero_grad()

        if scheduler is not None:
            scheduler.step()
                
        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size
        
        epoch_loss = running_loss / dataset_size
        
        bar.set_postfix(Epoch=epoch, Train_Loss=epoch_loss,)
    
    return epoch_loss


# In[ ]:


import torch


# # get_emb

# In[ ]:


@torch.inference_mode()
def get_emb(model, loader):
    model.eval()
    emb = []
    labels = []
    for text, latitude, longitude, label in tqdm(loader, total=len(loader)):
        text = CFG.tokenizer(text, padding=True, return_tensors='pt')
        for k,v in text.items():
            text[k] = v.cuda()
        latitude = latitude.cuda()
        longitude = longitude.cuda()
        emb.append(model.extract(text, latitude, longitude).cpu().numpy())
        labels.append(label.numpy())
    return np.concatenate(emb), np.concatenate(labels)


# # Val func

# In[ ]:


get_ipython().system(' pip install faiss-gpu')


# In[ ]:


import faiss


# In[ ]:


def valid_one_epoch(model, valid_loader, targets):
    val_embs, labels = get_emb(model, valid_loader)
    index = faiss.IndexFlatIP(CFG.emb_size)
    index.add(val_embs)
    similarity, neightbors = index.search(val_embs, CFG.n_neighbors)
    
    return get_score(similarity, neightbors, targets, labels),contain_self_score(targets, labels)
    
def get_score(similarity, neightbors, targets, labels):
    score = 0
    for i in range(similarity.shape[0]):
        top = neightbors[i][:CFG.knn] 
        index = set(top[similarity[i][:CFG.knn] > CFG.threshold])
        target = targets[labels[i]]
        score += len(target.intersection(index))/len(target.union(index))
    return score / similarity.shape[0]

def contain_self_score(targets, labels):
    score = 0
    for i in range(labels.shape[0]):
        index = set([i])
        target = targets[labels[i]]
        score += len(target.intersection(index))/len(target.union(index))
    return score / labels.shape[0]


# # Run Training

# In[ ]:


from torch.optim import AdamW


# In[ ]:


def run_training(df, fold):    
    train = df[df['fold'] != fold].reset_index(drop=True)
    valid = df[df['fold'] == fold].reset_index(drop=True)
    train_dataset, val_dataset = FourSquareDataset(train), FourSquareDataset(valid)
    train_loader, val_loader = DataLoader(train_dataset, CFG.bs, shuffle=True, pin_memory=True), DataLoader(val_dataset, CFG.bs * 4, pin_memory=True)
    print(f'train on {len(train)} samples, val on {len(valid)} samples')
    
    targets = valid.copy()
    targets['index'] = targets.index
    targets = targets.groupby('point_of_interest')['index'].agg(set)
        
    model = FourSquareModel(CFG).cuda()
    params = get_optimizer_params(model ,CFG.lr, weight_decay=CFG.weight_decay)
    optimizer= AdamW(params, lr=CFG.lr, eps=CFG.eps, betas=CFG.betas)
    scheduler = get_scheduler(optimizer)
    
    if os.path.exists(f'checkpoint_epoch{CFG.start_epoch}.pth'):
        checkpoint = torch.load(f'checkpoint_epoch{CFG.start_epoch}.pth')
        model.load_state_dict(checkpoint['checkpoint'])
        start_epoch = checkpoint['epoch']
        optimizer = checkpoint['optimizer']
        scheduler = checkpoint['scheduler']
    else:
        start_epoch = 0
    best_acc = 0    
    for epoch in range(start_epoch + 1, CFG.epochs + 1): 
        train_epoch_loss = train_one_epoch(model, optimizer, scheduler, 
                                           dataloader=train_loader, 
                                            epoch=epoch)
        #first is valid  score, last is the score only self-match.
        acc, least_acc = valid_one_epoch(model, val_loader, targets)
        print(f'acc:{acc}, least acc:{least_acc}')
        # it takes about one hour to train on single 3080Ti per epoch, and 20-30min to valid, 
        # so it's recommonded to valid every once in a while
        torch.save({'checkpoint':model.state_dict(),'epoch':epoch,'optimizer':optimizer,'scheduler':scheduler}, f'checkpoint_epoch{epoch}.pth')
        if acc > best_acc:
            print(f"score imporved ({best_acc} ---> {acc})")
            best_acc = acc
            torch.save(model.state_dict(), f'model_fold{fold}.pth')
        if CFG.wandb:
            wandb.log({f'fold{fold}_acc':acc, f'fold{fold}_best_acc':best_acc, 'epoch':epoch})


# In[ ]:


if CFG.wandb:
    wandb.init(project="foursquare-location-matching", name='v1')


# In[ ]:


for fold in range(CFG.folds):
    print(10*'-'+f'fold{fold}'+10*'-')
    run_training(df, fold)

