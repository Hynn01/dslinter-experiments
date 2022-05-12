#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from tqdm import tqdm
import sys, os
from transformers import DistilBertModel, DistilBertTokenizer
import torch.nn.functional as F
import torch.nn as nn
import torch

MAX_LEN = 32

class Cat2VecModel(nn.Module):
    def __init__(self):
        super(Cat2VecModel, self).__init__()
        self.distill_bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        
    def forward(self, ids, mask):
        x = self.distill_bert(ids, mask)[0]
        x = F.normalize((x[:, 1:, :]*mask[:, 1:, None]).mean(axis=1))
        return x
    
cat2vec_model = Cat2VecModel()
cat2vec_model = cat2vec_model.cuda()


# In[ ]:


from torch.utils.data import DataLoader, Dataset


class InferenceDataset(Dataset):
    
    def __init__(self, df, max_len):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.max_len = max_len
        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased", do_lower_case=True)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        
        inputs = self.tokenizer.encode_plus(
            row.categories,
            None,
            add_special_tokens=True,
            max_length=MAX_LEN,
            padding="max_length",
            return_token_type_ids=True,
            truncation=True
        )
        ids = torch.LongTensor(inputs['input_ids'])
        mask = torch.LongTensor(inputs['attention_mask'])

        return ids, mask

    def __len__(self):
        return self.df.shape[0]
    
    
cat_df = pd.read_csv("../input/foursquare-location-matching/train.csv")[["categories"]].drop_duplicates()
cat_df["categories"] = cat_df["categories"].fillna("null")

cat_ds = InferenceDataset(cat_df, max_len=MAX_LEN)
print(len(cat_ds))
cat_ds[0]


# In[ ]:


BS = 256
NW = 2    

def inference(ds):
    loader = DataLoader(ds, batch_size=BS, shuffle=False, num_workers=NW,
                        pin_memory=False, drop_last=False)
    tbar = tqdm(loader, file=sys.stdout)
    
    vs = []
    with torch.no_grad():
        for idx, (ids, masks) in enumerate(tbar):
            v = cat2vec_model(ids.cuda(), masks.cuda()).detach().cpu().numpy()
            vs.append(v)
    return np.concatenate(vs)


V = inference(cat_ds)
V.shape


# In[ ]:


from cuml.neighbors import NearestNeighbors

N = 3

matcher = NearestNeighbors(n_neighbors=N, metric="cosine")
matcher.fit(V)


distances, indices = matcher.kneighbors(V)


# In[ ]:


for i in range(1, N):
    cat_df[f"match_{i}"] = cat_df["categories"].values[indices[:, i]]
    cat_df[f"sim_{i}"] = np.clip(1 - distances[:, i], 0, None)
    
cat_df


# In[ ]:


cat_df.sort_values("sim_1")


# In[ ]:




