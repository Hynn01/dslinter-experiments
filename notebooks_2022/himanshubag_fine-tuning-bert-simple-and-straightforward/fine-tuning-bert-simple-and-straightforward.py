#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


train_df = pd.read_csv('/kaggle/input/us-patent-phrase-to-phrase-matching/train.csv')
print(train_df.info())
train_df.head()


# In[ ]:


for col in ['anchor','target','context','score']:
    print('\n====',col,'=====')
    print(train_df[col].value_counts())


# In[ ]:


cpc = pd.read_csv('../input/cpc-codes/titles.csv')
cpc.head()


# - adding cpc title to add more context

# In[ ]:


cpc = cpc.rename(columns = {"code" : "context"})
train_df = pd.merge(train_df, cpc[["context","title"]], on ="context", how = "left")
train_df.head()


# In[ ]:


def clean(x):
    t = x.lower()
    t = t.replace("[",'')
    t = t.replace(";",'')
    t = t.replace(",",'')
    t = t.replace("]",'')
    t = t.replace(":",'')
    return t

train_df['title'] = train_df['title'].apply(lambda x: clean(x))
train_df.head()


# In[ ]:


train_df['sen1'] = train_df['anchor'].astype('str')+' '+train_df['title'].astype('str')
train_df = train_df.drop(['anchor','context','title'],axis=1)
# train_df['all_sen'] = train_df['sen1']+' [SEP '+train_df['target']
train_df.head()


# In[ ]:


seq1_len = [len(i.split()) for i in train_df['sen1'].values]
pd.Series(seq1_len).hist(bins = 30)

tar_len = [len(i.split()) for i in train_df['target'].values]
pd.Series(tar_len).hist(bins = 30)


# In[ ]:


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(train_df[['target','sen1']],train_df['score'],random_state=1234,test_size=0.3)
print(x_train.shape,x_test.shape)
print(y_train.shape,y_test.shape)


# In[ ]:


x_test.head()


# In[ ]:


y_test.head()


# ## Model

# In[ ]:


import torch
import torch.nn as nn
import transformers
from torch.nn.utils.clip_grad import clip_grad_norm


torch.cuda.is_available()


# In[ ]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()


# #### class defination for model , dataset and training functions

# In[ ]:


class my_model(nn.Module):
    def __init__(self,bert_path):
        super(my_model,self).__init__()
        self.bert_path = bert_path
        self.bert = transformers.BertModel.from_pretrained(self.bert_path)
        self.fc_layer = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(1024,256),
            nn.ReLU(inplace=True),
            nn.Linear(256,1),
            nn.Sigmoid()
        )
    def forward(self,ids,mask,token_type_ids):
        out = self.bert(input_ids=ids,attention_mask=mask,token_type_ids=token_type_ids)
        pooler_output = out.get('pooler_output')
        bo = self.fc_layer(pooler_output)
        return bo
     


# In[ ]:


class my_dataset_train:
    def __init__(self,text1,text2,label,tokenizer,max_len):
        self.text1=text1
        self.text2=text2
        self.label=label
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.text1)
    
    def __getitem__(self,idx):
        text_1 = str(self.text1[idx])
        text_2 = str(self.text2[idx])
        label = self.label[idx]
        
        inputs = self.tokenizer(
            text_1,
            text_2,
            add_special_tokens=True,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_attention_mask=True
        )
        
        ids = inputs['input_ids']
        token_type_ids = inputs["token_type_ids"]
        mask = inputs['attention_mask']
        
        padding_len = self.max_len - len(ids)
        ids = ids + ([0]*padding_len)
        token_type_ids = token_type_ids + ([0]*padding_len)
        mask = mask + ([0]*padding_len)
        
        return {
            "ids": torch.tensor(ids,dtype=torch.long),
            "mask": torch.tensor(mask,dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids,dtype=torch.long),
            "targets": torch.tensor(label,dtype=torch.float),
        }
        


# #### dataset for train and valid

# In[ ]:


max_len=128
train_batch_size = 16
epochs=6
bert_path = '../input/bert-for-patents/bert-for-patents'#'../input/bert-base-uncased'

tokenizer = transformers.BertTokenizer.from_pretrained(bert_path)

# Training dataset prep

train_text1 = list(x_train['target'].values)
train_text2 = list(x_train['sen1'].values)
train_label = list(y_train.values)

train_dataset = my_dataset_train(
    text1 = train_text1,
    text2 = train_text2,
    label = train_label,
    tokenizer=tokenizer ,
    max_len=max_len
)

train_data_loader = torch.utils.data.DataLoader(train_dataset,batch_size=train_batch_size,shuffle=True)

# validation dataset prep
val_text1 = list(x_test['target'].values)
val_text2 = list(x_test['sen1'].values)
val_label = list(y_test.values)

valid_dataset = my_dataset_train(
    text1 = val_text1,
    text2 = val_text2,
    label = val_label,
    tokenizer=tokenizer,
    max_len=max_len
)

valid_data_loader = torch.utils.data.DataLoader(valid_dataset,batch_size=train_batch_size,shuffle=True)

    


# #### Train

# In[ ]:


def train(model, optimizer, scheduler, loss_function, epochs,train_dataloader, device, clip_value=2):
    model.train()
    for epoch in range(epochs):
#         print(epoch)
#         print("-----")
        best_loss = []
        for step, batch in enumerate(train_dataloader): 
            batch_inputs, batch_masks, batch_labels = batch['ids'].to(device), batch['mask'].to(device), batch['targets'].to(device)
            batch_token_type_ids = batch['token_type_ids'].to(device)
            model.zero_grad()
            outputs = model(batch_inputs, batch_masks, batch_token_type_ids)
            loss = loss_function(outputs.squeeze(),batch_labels.squeeze())
            best_loss.append(loss)
            loss.backward()
            clip_grad_norm(model.parameters(), clip_value)
            optimizer.step()
            scheduler.step()
#             print(f"step > {step},loss > {loss}")
        loss2 = sum(best_loss)/len(best_loss)
        print(f'Epoch : {epoch} ,Train loss : {loss2}')
                
    return model

def r2_score(outputs, labels):
    labels_mean = torch.mean(labels)
    ss_tot = torch.sum((labels - labels_mean) ** 2)
    ss_res = torch.sum((labels - outputs) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2

def evaluate(model,loss_function,test_dataloader,device):
    model.eval()
    test_loss, test_r2 = [], []
    for step,batch in enumerate(test_dataloader):
        batch_inputs, batch_masks, batch_labels = batch['ids'].to(device), batch['mask'].to(device), batch['targets'].to(device)
        batch_token_type_ids = batch['token_type_ids'].to(device)
        with torch.no_grad():
            outputs = model(batch_inputs, batch_masks, batch_token_type_ids)
        loss = loss_function(outputs, batch_labels)
        test_loss.append(loss.item())
        r2 = r2_score(outputs, batch_labels)
        test_r2.append(r2.item())
    return test_loss, test_r2


# In[ ]:


num_train_steps = len(train_data_loader) * epochs

model = my_model(bert_path).to(device)

optimizer = transformers.AdamW(model.parameters(),lr=3e-5,eps=1e-8)

scheduler = transformers.get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=num_train_steps
)

loss_function = nn.MSELoss()


model = train(model, optimizer, scheduler, loss_function, epochs,train_data_loader, device)


loss1,r2_ = evaluate(model,loss_function,valid_data_loader,device)

loss = sum(loss1)/len(loss1)
r2 = sum(r2_)/len(r2_)
print(f"eval mean result : loss {loss}, r2 {r2}")


# In[ ]:


torch.save(model.state_dict(),f'./my_bert')


# ## Inference
# 
# - follow evaluate function
# - create dataset class

# In[ ]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()


# In[ ]:


import gc
gc.collect()


# In[ ]:


test_df = pd.read_csv('../input/us-patent-phrase-to-phrase-matching/test.csv')
test_df.head()


# In[ ]:


cpc = pd.read_csv('../input/cpc-codes/titles.csv')
cpc = cpc.rename(columns = {"code" : "context"})
test_df = pd.merge(test_df, cpc[["context","title"]], on ="context", how = "left")

def clean(x):
    t = x.lower()
    t = t.replace("[",'')
    t = t.replace(";",'')
    t = t.replace(",",'')
    t = t.replace("]",'')
    t = t.replace(":",'')
    return t

test_df['title'] = test_df['title'].apply(lambda x: clean(x))
test_df.head()


# In[ ]:


test_df.info()


# In[ ]:


test_df['sen1'] = test_df['anchor'].astype('str')+' '+test_df['title'].astype('str')
test_df = test_df.drop(['anchor','context','title'],axis=1)
test_df.head()


# In[ ]:


class my_dataset_test:
    def __init__(self,text1,text2,idf,tokenizer,max_len):
        self.text1=text1
        self.text2=text2
        self.idf = idf
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.text1)
    
    def __getitem__(self,idx):
        text_1 = str(self.text1[idx])
        text_2 = str(self.text2[idx])
        idf = self.idf[idx]
        
        inputs = self.tokenizer(
            text_1,
            text_2,
            add_special_tokens=True,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_attention_mask=True
        )
        
        ids = inputs['input_ids']
        token_type_ids = inputs["token_type_ids"]
        mask = inputs['attention_mask']
        
        padding_len = self.max_len - len(ids)
        ids = ids + ([0]*padding_len)
        token_type_ids = token_type_ids + ([0]*padding_len)
        mask = mask + ([0]*padding_len)
        
        return {
            "ids": torch.tensor(ids,dtype=torch.long),
            "mask": torch.tensor(mask,dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids,dtype=torch.long),
            "idf": idf
        }
        


# In[ ]:


bert_path = '../input/bert-for-patents/bert-for-patents'#'../input/bert-base-uncased'
max_len=64
# tokenizer = transformers.BertTokenizer.from_pretrained(bert_path)

test_text1 = list(test_df['target'].values)
test_text2 = list(test_df['sen1'].values)

test_dataset = my_dataset_test(
    text1 = test_text1,
    text2 = test_text2,
    idf=list(test_df['id'].values),
    tokenizer=tokenizer,
    max_len=max_len
)

test_data_loader = torch.utils.data.DataLoader(test_dataset,batch_size=64,shuffle=True)


# In[ ]:


def predict(model,test_dataloader,device):
    model.eval()
    result = []
    for step,batch in enumerate(test_dataloader):
        batch_inputs, batch_masks = batch['ids'].to(device), batch['mask'].to(device)
        batch_token_type_ids = batch['token_type_ids'].to(device)
        with torch.no_grad():
            outputs = model(batch_inputs, batch_masks, batch_token_type_ids)
        out = [i[0] for i in outputs.cpu().detach().numpy()]
        batch_idf = batch['idf']
        temp = [[i,j] for i,j in zip(batch_idf,out)]
        result.extend(temp)
    return result


# In[ ]:


model = my_model(bert_path).to(device)
model.load_state_dict(torch.load('my_bert'))

final_res = predict(model,test_data_loader,device)


# In[ ]:


final_res


# In[ ]:


sample = pd.read_csv('../input/us-patent-phrase-to-phrase-matching/sample_submission.csv')
sample.head()


# In[ ]:


submit_csv = pd.DataFrame(final_res,columns=['id','score'])
submit_csv.head()


# In[ ]:


submit_csv.to_csv('submission.csv',index=False)


# In[ ]:




