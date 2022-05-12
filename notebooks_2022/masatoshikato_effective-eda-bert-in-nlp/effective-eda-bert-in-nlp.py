#!/usr/bin/env python
# coding: utf-8

# ## Effective EDA & BERT in NLP
# Natural Language Processing with Disaster Tweets🚒

# #### In my kernel
# 1. Points to note
# 2. Text Cleaning (regular expression)
# 3. BERT (using PyTorch)
# 4. Reference

# I explained the methods I have learned about Natural Language Processing. 
# I especially focused on EDA (text cleaning) and model design with BERT. After removing noise (such as http://- or HTML or punctuation) in the text and extracting text features, I created models using BERT. This resulted in a high score. I explained in detail what I struggled with and what stopped me, to not leave anything out!

# ## 1. Points to note

# * To use a GPU, go to Settings→Accelerator and set the GPU.
# 
# * If you see the error **CUDA error: out of memory**, check GPU Memory in the upper right corner of the kernel. (It is shown to the right of Draft Session (-m)).　      
# In some cases, stopping the session and resetting it may solve the problem.
# 
# * When assembling the model with the class, I set in_features=768. Any other number will cause an error. error: **mat1 and mat2 shapes cannot be multiplied (16x768 and 500x20).** In this case, the number of batches is 16 and in_features is 768.

# ### Import libraries and basic EDA

# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')


import warnings
warnings.filterwarnings('ignore')


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')
test  = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
submission = pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')


# In[ ]:


train.head(5)


# In[ ]:


print(train.shape)


# In[ ]:


plt.subplots(1,3,figsize=(14,4))
plt.subplot(1,3,1); sns.countplot(train['target'], palette='gray', alpha=1.0); plt.xlabel(''), plt.ylabel(''); plt.title('target')
plt.subplot(1,3,2); plt.hist(train['text'].str.len(), color='black', alpha=0.6); plt.ylabel(''); plt.title('Words in sentences (train)')
plt.subplot(1,3,3); plt.hist(test['text'].str.len(), color='black', alpha=0.6); plt.ylabel(''); plt.title('Words in sentences (test)')
plt.show()


# In[ ]:


for i in range(50):
    print(train['text'][i])


# ## Text Cleaning (regular expression)
# 
# I converted all text to lowercase, converted http://--- to httpsmark. I also pre-processed HTML(< h >), other symbols(@, #) and punctuation. I thought that some number in the text were meaningless, so I converted numbers such as 2000 or 15000 into "number" so that we could see there was some kind of number in the text.

# Reference : https://gist.github.com/slowkow/7a7f61f495e3dbb7e3d767f97bd7304b (for the pre-processing of Emoji)

# In[ ]:


import re
import string


# In[ ]:


def remove_URL(text):
    url = re.compile(r'https?://\S+')
    return url.sub(r' httpsmark ', text)


def remove_html(text):
    html = re.compile(r'<.*?>')
    return html.sub(r'', text)


def remove_atsymbol(text):
    name = re.compile(r'@\S+')
    return name.sub(r' atsymbol ', text)


def remove_hashtag(text):
    hashtag = re.compile(r'#')
    return hashtag.sub(r' hashtag ', text)


def remove_exclamation(text):
    exclamation = re.compile(r'!')
    return exclamation.sub(r' exclamation ', text)


def remove_question(text):
    question = re.compile(r'?')
    return question.sub(r' question ', text)


def remove_punc(text):
    return text.translate(str.maketrans('','',string.punctuation))


def remove_number(text):
    number = re.compile(r'\d+')
    return number.sub(r' number ', text)


def remove_emoji(string):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002500-\U00002BEF"  # chinese char
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  # dingbats
                               u"\u3030"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r' emoji ', string)


# In[ ]:


train['text'] = train['text'].str.lower()
train['text'] = train['text'].apply(lambda text: remove_URL(text))
train['text'] = train['text'].apply(lambda text: remove_html(text))
train['text'] = train['text'].apply(lambda text: remove_atsymbol(text))
train['text'] = train['text'].apply(lambda text: remove_hashtag(text))
train['text'] = train['text'].apply(lambda text: remove_exclamation(text))
train['text'] = train['text'].apply(lambda text: remove_punc(text))
train['text'] = train['text'].apply(lambda text: remove_number(text))
train['text'] = train['text'].apply(lambda text: remove_emoji(text))




test['text']  = test['text'].str.lower()
test['text']  = test['text'].apply(lambda text: remove_URL(text))
test['text']  = test['text'].apply(lambda text: remove_html(text))
test['text']  = test['text'].apply(lambda text: remove_atsymbol(text))
test['text']  = test['text'].apply(lambda text: remove_hashtag(text))
test['text']  = test['text'].apply(lambda text: remove_exclamation(text))
test['text']  = test['text'].apply(lambda text: remove_punc(text))
test['text']  = test['text'].apply(lambda text: remove_number(text))
test['text']  = test['text'].apply(lambda text: remove_emoji(text))


# In[ ]:


for i in range(50):
    print(test['text'][i])


# In[ ]:


plt.subplots(1,2,figsize=(14,4))
plt.subplot(1,2,1); plt.hist(train['text'].str.len(), color='black', alpha=0.6); plt.ylabel(''); plt.title('Number of words in text (train)')
plt.subplot(1,2,2); plt.hist(test['text'].str.len(), color='black', alpha=0.6); plt.ylabel(''); plt.title('Number of words in text (test)')
plt.show()


# In[ ]:


TRAINING_SIZE = 6090
train_shuffle = train.sample(frac=1, random_state=0)


train = train_shuffle[0:TRAINING_SIZE]
valid = train_shuffle[TRAINING_SIZE:]


# # BERT (using PyTorch)
# 
# We need to design Dataset, DataLoader and Model using PyTorch.

# In[ ]:


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# In[ ]:


class Data(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, idx):
        text = self.data['text'].values[idx]
        label = self.data['target'].values[idx]
        return text, torch.tensor(label, dtype=torch.float)
    
    
    
class TestData(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, idx):
        text = self.data['text'].values[idx]
        return text


# In[ ]:


train_ds = Data(train)
valid_ds = Data(valid)
test_ds  = TestData(test)



print(train_ds[0])
print('-'*85)
print(train_ds[0][0])
print('-'*85)
print(train_ds[0][1])


# In[ ]:


train_dl = DataLoader(train_ds, batch_size=16, shuffle=True, drop_last=False)
valid_dl = DataLoader(valid_ds, batch_size=16*2, shuffle=True, drop_last=False)
test_dl  = DataLoader(test_ds, batch_size=16, shuffle=False, drop_last=False)



batch = next(iter(train_dl))
print(len(batch[0]))


# In[ ]:


# tokenizer

get_ipython().system('pip install transformers -q')
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')


# I used a tokenizer to break the text into tokens and convert it into a form that could be entered into BERT.
# 
# 
# 
# * **input_ids** : Token converted to ID  (ex. my→101, brothernlaw→2026)
# 
#     [CLS] is added to the beginning of the token sequence and [SEP] to the end by default.
# 
# * **token_type_ids** : Distinguish between tokens used and not used in learning.
# 
# * **attention_mask** : Used to distinguish between two sentences when they are entered as a pair. In this case, it has no meaning.
# 

# In[ ]:


for text in train['text'].values:
    encoded = tokenizer.encode_plus(text.lower())
    
    
print(encoded)


# In[ ]:


decoded = tokenizer.decode(encoded['input_ids'])


print(decoded)


# In[ ]:


MAX_LEN = 50
encoded = tokenizer.encode_plus(text, padding='max_length', max_length=MAX_LEN, truncation=True)


# In[ ]:


from torch import nn
from transformers import AutoModel

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = AutoModel.from_pretrained('bert-base-uncased')
        self.classifier = nn.Linear(in_features=768, out_features=2)
    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooler_output = outputs.pooler_output
        logits = self.classifier(pooler_output).squeeze(-1)
        return logits


# In[ ]:


model = Model().cuda()


optimizer = torch.optim.AdamW(model.parameters())
criterion = nn.CrossEntropyLoss()


# In[ ]:


model.train()
train_loss=0

for batch in train_dl:
    optimizer.zero_grad()
    text = batch[0]
    label = batch[1].long().cuda()
    encoded = tokenizer.batch_encode_plus(
                  list(text),
                  padding='max_length',
                  max_length=MAX_LEN,
                  truncation=True,
                  return_tensors='pt',
                  return_attention_mask=True,  
                  return_token_type_ids=True)
    input_ids=encoded['input_ids'].cuda()
    attention_mask=encoded['attention_mask'].cuda()
    token_type_ids=encoded['token_type_ids'].cuda()
    preds=model(input_ids, attention_mask, token_type_ids)
    loss=criterion(preds, label)
    loss.backward()
    optimizer.step()
    train_loss += loss.item()
train_loss/=len(train_dl)
print(train_loss)


# In[ ]:


model.eval()
valid_loss=0

with torch.no_grad():
    for batch in valid_dl:
        optimizer.zero_grad()
        text = batch[0]
        label = batch[1].long().cuda()
        encoded = tokenizer.batch_encode_plus(
                      list(text),
                      padding='max_length',
                      max_length=MAX_LEN,
                      truncation=True,
                      return_tensors='pt',
                      return_attention_mask=True,  
                      return_token_type_ids=True)
        input_ids=encoded['input_ids'].cuda()
        attention_mask=encoded['attention_mask'].cuda()
        token_type_ids=encoded['token_type_ids'].cuda()
        preds=model(input_ids, attention_mask, token_type_ids)
        loss=criterion(preds, label)
        valid_loss += loss.item()
    valid_loss/=len(valid_dl)
print(valid_loss)


# In[ ]:


model = Model().cuda()

optimizer = torch.optim.Adam(model.parameters(), lr = 1e-5)
criterion = nn.CrossEntropyLoss()



best_loss=np.inf
for epoch in range(5):
    model.train()
    train_loss=0
    for batch in train_dl:
        optimizer.zero_grad()
        text = batch[0]
        label = batch[1].long().cuda()
        encoded = tokenizer.batch_encode_plus(
                      list(text),
                      padding='max_length',
                      max_length=MAX_LEN,
                      truncation=True,
                      return_tensors='pt',
                      return_attention_mask=True,  
                      return_token_type_ids=True)
        input_ids=encoded['input_ids'].cuda()
        attention_mask=encoded['attention_mask'].cuda()
        token_type_ids=encoded['token_type_ids'].cuda()
        preds=model(input_ids, attention_mask, token_type_ids)
        loss=criterion(preds, label)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss/=len(train_dl)

    
    
    model.eval()
    valid_loss=0
    with torch.no_grad():
        for batch in valid_dl:
            optimizer.zero_grad()
            text = batch[0]
            label = batch[1].long().cuda()
            encoded = tokenizer.batch_encode_plus(
                          list(text),
                          padding='max_length',
                          max_length=MAX_LEN,
                          truncation=True,
                          return_tensors='pt',
                          return_attention_mask=True,  
                          return_token_type_ids=True)
            input_ids=encoded['input_ids'].cuda()
            attention_mask=encoded['attention_mask'].cuda()
            token_type_ids=encoded['token_type_ids'].cuda()
            preds=model(input_ids, attention_mask, token_type_ids)
            loss=criterion(preds, label)
            valid_loss+=loss.item()
        valid_loss/=len(valid_dl)
        
    print(f"EPOCH[{epoch}]")
    print(train_loss)
    print(valid_loss)
    if valid_loss<best_loss:
        best_loss=valid_loss
        torch.save(model.state_dict(), "bert.pth")
        print('saved.....')


# In[ ]:


model.load_state_dict(torch.load("bert.pth", map_location="cpu"))
model.eval()
pred_valid = []
with torch.no_grad():
    for batch in valid_dl:
        optimizer.zero_grad()
        text = batch[0]
        encoded = tokenizer.batch_encode_plus(
                      list(text),
                      padding='max_length',
                      max_length=MAX_LEN,
                      truncation=True,
                      return_tensors='pt',
                      return_attention_mask=True,  
                      return_token_type_ids=True)
        input_ids = encoded['input_ids'].cuda()
        attention_mask = encoded['attention_mask'].cuda()
        token_type_ids = encoded['token_type_ids'].cuda()
        preds = model(input_ids, attention_mask, token_type_ids)
        pred_valid.append(preds.cpu().numpy())
pred_valid = np.concatenate(pred_valid, axis=0)


# In[ ]:


pred_valid.argmax(axis=1)


# In[ ]:


from sklearn.metrics import accuracy_score
acc = accuracy_score(valid['target'], pred_valid.argmax(axis=1))
print(acc)


# ### Testset

# In[ ]:


model.load_state_dict(torch.load("bert.pth", map_location="cpu"))
model.eval()
pred = []
with torch.no_grad():
    for batch in test_dl:
        optimizer.zero_grad()
        text = batch
        encoded = tokenizer.batch_encode_plus(
                      list(text),
                      padding='max_length',
                      max_length=MAX_LEN,
                      truncation=True,
                      return_tensors='pt',
                      return_attention_mask=True,  
                      return_token_type_ids=True)
        input_ids = encoded['input_ids'].cuda()
        attention_mask = encoded['attention_mask'].cuda()
        token_type_ids = encoded['token_type_ids'].cuda()
        preds = model(input_ids, attention_mask, token_type_ids)
        pred.append(preds.cpu().numpy())
pred = np.concatenate(pred, axis=0)


# In[ ]:


print(pred.argmax(axis=1))


# In[ ]:


test['target'] = pred.argmax(axis=1)
test['target'] = test['target'].astype(int)
test = test[['id', 'target']]
test.to_csv('submission.csv', index = False)


# In[ ]:


test


# ## Reference
# * https://gist.github.com/slowkow/7a7f61f495e3dbb7e3d767f97bd7304b　
# 
# * 【Python】BERTの実装方法｜文章分類, pytorch　https://htomblog.com/python-bert  HTOMblog 
# 
# * BERTによる自然言語処理入門 Transformersを使った実践プログラミング ストックマーク株式会社　近江崇宏,金田健太郎,森長誠,江間見亜利 共著 

# ## Thank you!!
