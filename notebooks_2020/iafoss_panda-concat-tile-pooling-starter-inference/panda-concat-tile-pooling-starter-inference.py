#!/usr/bin/env python
# coding: utf-8

# # Description
# This kernel performs inference for [PANDA concat tile pooling starter](https://www.kaggle.com/iafoss/panda-concat-fast-ai-starter) kernel with use of multiple models and 8 fold TTA. Check it for more training details. The image preprocessing pipline is provided [here](https://www.kaggle.com/iafoss/panda-16x128x128-tiles).

# In[ ]:


import cv2
from tqdm import tqdm_notebook as tqdm
import fastai
from fastai.vision import *
import os
from mish_activation import *
import warnings
warnings.filterwarnings("ignore")
import skimage.io
import numpy as np
import pandas as pd
sys.path.insert(0, '../input/semisupervised-imagenet-models/semi-supervised-ImageNet1K-models-master/')
from hubconf import *


# In[ ]:


DATA = '../input/prostate-cancer-grade-assessment/test_images'
TEST = '../input/prostate-cancer-grade-assessment/test.csv'
SAMPLE = '../input/prostate-cancer-grade-assessment/sample_submission.csv'
MODELS = [f'../input/panda-starter-models/RNXT50_{i}.pth' for i in range(4)]

sz = 128
bs = 2
N = 12
nworkers = 2


# # Model

# In[ ]:


def _resnext(url, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    #state_dict = load_state_dict_from_url(url, progress=progress)
    #model.load_state_dict(state_dict)
    return model

class Model(nn.Module):
    def __init__(self, arch='resnext50_32x4d', n=6, pre=True):
        super().__init__()
        #m = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', arch)
        m = _resnext(semi_supervised_model_urls[arch], Bottleneck, [3, 4, 6, 3], False, 
                progress=False,groups=32,width_per_group=4)
        self.enc = nn.Sequential(*list(m.children())[:-2])       
        nc = list(m.children())[-1].in_features
        self.head = nn.Sequential(AdaptiveConcatPool2d(),Flatten(),nn.Linear(2*nc,512),
                Mish(),nn.BatchNorm1d(512),nn.Dropout(0.5),nn.Linear(512,n))
        
    def forward(self, x):
        shape = x.shape
        n = shape[1]
        x = x.view(-1,shape[2],shape[3],shape[4])
        x = self.enc(x)
        shape = x.shape
        x = x.view(-1,n,shape[1],shape[2],shape[3]).permute(0,2,1,3,4).contiguous()          .view(-1,shape[1],shape[2]*n,shape[3])
        x = self.head(x)
        return x


# In[ ]:


models = []
for path in MODELS:
    state_dict = torch.load(path,map_location=torch.device('cpu'))
    model = Model()
    model.load_state_dict(state_dict)
    model.float()
    model.eval()
    model.cuda()
    models.append(model)

del state_dict


# # Data

# In[ ]:


def tile(img):
    shape = img.shape
    pad0,pad1 = (sz - shape[0]%sz)%sz, (sz - shape[1]%sz)%sz
    img = np.pad(img,[[pad0//2,pad0-pad0//2],[pad1//2,pad1-pad1//2],[0,0]],
                 constant_values=255)
    img = img.reshape(img.shape[0]//sz,sz,img.shape[1]//sz,sz,3)
    img = img.transpose(0,2,1,3,4).reshape(-1,sz,sz,3)
    if len(img) < N:
        img = np.pad(img,[[0,N-len(img)],[0,0],[0,0],[0,0]],constant_values=255)
    idxs = np.argsort(img.reshape(img.shape[0],-1).sum(-1))[:N]
    img = img[idxs]
    return img

mean = torch.tensor([1.0-0.90949707, 1.0-0.8188697, 1.0-0.87795304])
std = torch.tensor([0.36357649, 0.49984502, 0.40477625])

class PandaDataset(Dataset):
    def __init__(self, path, test):
        self.path = path
        self.names = list(pd.read_csv(test).image_id)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        img = skimage.io.MultiImage(os.path.join(DATA,name+'.tiff'))[-1]
        tiles = torch.Tensor(1.0 - tile(img)/255.0)
        tiles = (tiles - mean)/std
        return tiles.permute(0,3,1,2), name


# # Prediction

# In[ ]:


sub_df = pd.read_csv(SAMPLE)
if os.path.exists(DATA):
    ds = PandaDataset(DATA,TEST)
    dl = DataLoader(ds, batch_size=bs, num_workers=nworkers, shuffle=False)
    names,preds = [],[]

    with torch.no_grad():
        for x,y in tqdm(dl):
            x = x.cuda()
            #dihedral TTA
            x = torch.stack([x,x.flip(-1),x.flip(-2),x.flip(-1,-2),
              x.transpose(-1,-2),x.transpose(-1,-2).flip(-1),
              x.transpose(-1,-2).flip(-2),x.transpose(-1,-2).flip(-1,-2)],1)
            x = x.view(-1,N,3,sz,sz)
            p = [model(x) for model in models]
            p = torch.stack(p,1)
            p = p.view(bs,8*len(models),-1).mean(1).argmax(-1).cpu()
            names.append(y)
            preds.append(p)
    
    names = np.concatenate(names)
    preds = torch.cat(preds).numpy()
    sub_df = pd.DataFrame({'image_id': names, 'isup_grade': preds})
    sub_df.to_csv('submission.csv', index=False)
    sub_df.head()


# In[ ]:


sub_df.to_csv("submission.csv", index=False)
sub_df.head()


# In[ ]:




