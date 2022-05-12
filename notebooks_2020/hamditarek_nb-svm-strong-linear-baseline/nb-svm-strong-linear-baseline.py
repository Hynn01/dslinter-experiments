#!/usr/bin/env python
# coding: utf-8

# #### Please refer to the original kernel: https://www.kaggle.com/jhoward/nb-svm-strong-linear-baseline

# In[ ]:


import pandas as pd, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy.sparse import hstack
import gc
import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))


# train1 = pd.read_csv("/kaggle/input/jigsaw-train-translated/train_mic.csv")
# #train2 = pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-unintended-bias-train.csv")
# #train2.toxic = train2.toxic.round().astype(int)
# 
# valid = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-test-translated/jigsaw_miltilingual_valid_translated.csv')
# #valid1 = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/validation.csv')

# valid = pd.concat([
#     valid[['comment_text', 'toxic']],
#     valid[['translated', 'toxic']].rename(columns={'translated': 'comment_text'})])
# 
# #valid['comment_text'] = valid['translated'] #+' '+valid1['comment_text']

# train = pd.concat([
#     train1[['comment_text', 'toxic']],
#     train1[['tr', 'toxic']].rename(columns={'tr': 'comment_text'}).dropna(),
#     train1[['ru', 'toxic']].rename(columns={'ru': 'comment_text'}).dropna(),
#     train1[['it', 'toxic']].rename(columns={'it': 'comment_text'}).dropna(),
#     train1[['fr', 'toxic']].rename(columns={'fr': 'comment_text'}).dropna(),
#     train1[['pt', 'toxic']].rename(columns={'pt': 'comment_text'}).dropna(),
#     train1[['es', 'toxic']].rename(columns={'es': 'comment_text'}).dropna(),
#     #train2[['comment_text', 'toxic']].query('toxic==1'),
#     #train2[['comment_text', 'toxic']].query('toxic==0'),
#     valid[['comment_text', 'toxic']]
#     #valid1[['comment_text', 'toxic']]
# ]).reset_index(drop=True)

# In[ ]:


# Loading data

train1 = pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv")
train1['lang'] = 'en'

train_es = pd.read_csv('/kaggle/input/jigsaw-train-multilingual-coments-google-api/jigsaw-toxic-comment-train-google-es-cleaned.csv')
train_es['lang'] = 'es'

train_fr = pd.read_csv('/kaggle/input/jigsaw-train-multilingual-coments-google-api/jigsaw-toxic-comment-train-google-fr-cleaned.csv')
train_fr['lang'] = 'fr'

train_pt = pd.read_csv('/kaggle/input/jigsaw-train-multilingual-coments-google-api/jigsaw-toxic-comment-train-google-pt-cleaned.csv')
train_pt['lang'] = 'pt'

train_ru = pd.read_csv('/kaggle/input/jigsaw-train-multilingual-coments-google-api/jigsaw-toxic-comment-train-google-ru-cleaned.csv')
train_ru['lang'] = 'ru'

train_it = pd.read_csv('/kaggle/input/jigsaw-train-multilingual-coments-google-api/jigsaw-toxic-comment-train-google-it-cleaned.csv')
train_it['lang'] = 'it'

train_tr = pd.read_csv('/kaggle/input/jigsaw-train-multilingual-coments-google-api/jigsaw-toxic-comment-train-google-tr-cleaned.csv')
train_tr['lang'] = 'tr'

#train2 = pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-unintended-bias-train.csv")
#train2.toxic = train2.toxic.round().astype(int)
#train2['lang'] = 'en'

train = pd.concat([
    
    train1[['comment_text', 'lang', 'toxic']],
    train_es[['comment_text', 'lang', 'toxic']],
    train_tr[['comment_text', 'lang', 'toxic']],
    train_fr[['comment_text', 'lang', 'toxic']],
    train_pt[['comment_text', 'lang', 'toxic']],
    train_ru[['comment_text', 'lang', 'toxic']],
    train_it[['comment_text', 'lang', 'toxic']]
    
]).sample(n=300000).reset_index(drop=True)

del train1, train_es, train_fr, train_pt, train_ru, train_it, train_tr
gc.collect()


# In[ ]:


#train = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv')
#train1 = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-unintended-bias-train.csv')

#valid = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-test-translated/jigsaw_miltilingual_valid_translated.csv')
#valid1 = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/validation.csv')

#test = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-test-translated/jigsaw_miltilingual_test_translated.csv')
test = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/test.csv')

subm = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/sample_submission.csv')


# In[ ]:


train.head()


# In[ ]:


#train = pd.concat([train,train1])
#train = pd.concat([train,valid])
#valid['comment_text'] = valid['translated'] #+' '+valid1['comment_text']
#test['content'] = test['translated'] #+' '+test1['content']
#train = pd.concat([train,valid])
#train = valid.copy()


# # Looking at the data
# ## The training data contains a row per comment, with an id, the text of the comment, and 6 different labels that we'll try to predict.

# In[ ]:


train.head()


# In[ ]:


train['comment_text'][0]


# In[ ]:


train['comment_text'][2]


# In[ ]:


lens = train.comment_text.str.len()
lens.mean(), lens.std(), lens.max()


# In[ ]:


lens.hist();


# In[ ]:


label_cols = ['toxic']
train['none'] = 1-train[label_cols].max(axis=1)
train.describe()


# In[ ]:


len(train),len(test)


# In[ ]:


train['comment_text'].fillna("unknown", inplace=True)
test['content'].fillna("unknown", inplace=True)


# In[ ]:


import re, string
re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
def tokenize(s): return re_tok.sub(r' \1 ', s).split()


# In[ ]:


n = train.shape[0]

vec = TfidfVectorizer(ngram_range=(1,2), tokenizer=tokenize,
               strip_accents='unicode', use_idf=1,
               smooth_idf=1, sublinear_tf=1 )


trn_term_doc = vec.fit_transform(train['comment_text'])
test_term_doc = vec.transform(test['content'])


# In[ ]:


trn_term_doc, test_term_doc


# In[ ]:


def pr(y_i, y):
    p = x[y==y_i].sum(0)
    return (p+1) / ((y==y_i).sum()+1)


# In[ ]:


x = trn_term_doc
test_x = test_term_doc


# In[ ]:


def get_mdl(y):
    y = y.values
    r = np.log(pr(1,y) / pr(0,y))
    m = LogisticRegression(C=4, dual=False)
    x_nb = x.multiply(r)
    return m.fit(x_nb, y), r


# In[ ]:


preds = np.zeros((len(test), len(label_cols)))

for i, j in enumerate(label_cols):
    print('fit', j)
    m,r = get_mdl(train[j])
    preds[:,i] = m.predict_proba(test_x.multiply(r))[:,1]


# In[ ]:


submid = pd.DataFrame({'id': subm["id"]})
submission = pd.concat([submid, pd.DataFrame(preds, columns = label_cols)], axis=1)

submission.to_csv('submission.csv', index=False)


# In[ ]:


submission.head(n=20)

