#!/usr/bin/env python
# coding: utf-8

# I am really surprised how people are open to share their codes and findings for the new starters. I am one of the starters and thanks to them, I could save my time quite a lot. So I would like to share one of mine.
# 
# This notebook is for pretraining with another twitter dataset. In this notebook, dataset from https://www.kaggle.com/ywang311/twitter-sentiment is used for pretraining.
# > 
# **References:**
# Codes mainly adopted from following notebooks:
# * [RoBERTa Baseline Starter (+ simple postprocessing)](https://www.kaggle.com/cheongwoongkang/roberta-baseline-starter-simple-postprocessing)
# 
# and related discussion threads about pretraining with another dataset:
# * [Language Model Pretraining](https://www.kaggle.com/c/tweet-sentiment-extraction/discussion/146262)
# * [Ideas Which Should Improve Scores Hopefully](https://www.kaggle.com/c/tweet-sentiment-extraction/discussion/142011)
# 
# Appreciated if you report anything need to be fixed. :)
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tqdm

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # Pretrain
# Pretrain with tweeter sentiment data in https://www.kaggle.com/ywang311/twitter-sentiment

# In[ ]:


pre_train_set = pd.read_csv("/kaggle/input/twitter-sentiment/Sentiment Analysis Dataset 2.csv", error_bad_lines=False)


# In[ ]:


pre_train_set.pop("ItemID")
pre_train_set.pop("SentimentSource")


# In[ ]:


pre_train_set.dropna(inplace=True)


# In[ ]:


tweets = []
for idx, row in tqdm.tqdm(pre_train_set.iterrows(), total=len(pre_train_set)):
    tweets.append(row["SentimentText"])


# In[ ]:


import random
random.shuffle(tweets)


# In[ ]:


len(tweets)


# In[ ]:


# The training data is separated with one blank line.
f_train = open("data-pretrain-twitter-train","w")
f_test = open("data-pretrain-twitter-test","w")
for idx, tweet in tqdm.tqdm(enumerate(tweets), total=len(tweets)):
    if idx < 0.9 * len(tweets):
        f_train.write(tweet)
        f_train.write("\n")
        f_train.write("\n")
    else:
        f_test.write(tweet)
        f_test.write("\n")
        f_test.write("\n")


# In[ ]:


get_ipython().system('git clone https://github.com/huggingface/transformers')


# In[ ]:


get_ipython().system('cd transformers && pip install .')


# In[ ]:


get_ipython().system('git clone https://github.com/NVIDIA/apex.git')


# In[ ]:


get_ipython().system('cd apex && pip install .')


# In[ ]:


get_ipython().system('python /kaggle/working/transformers/examples/run_language_modeling.py     --fp16     --model_type roberta     --model_name_or_path roberta-base     --config_name  /kaggle/input/train-config/config.json     --do_lower_case     --do_train     --do_eval     --train_data_file /kaggle/working/data-pretrain-twitter-train     --eval_data_file=/kaggle/working/data-pretrain-twitter-test     --learning_rate 5e-5     --num_train_epochs 3     --max_seq_length 192     --doc_stride 64     --output_dir results_roberta_pretrain     --per_gpu_eval_batch_size=8     --per_gpu_train_batch_size=8     --save_steps=100000     --mlm')


# # Train with pretrained data
# 5-folded dataset used from [roberta inference 5 folds](https://www.kaggle.com/abhishek/roberta-inference-5-folds)

# In[ ]:


train = pd.read_csv("/kaggle/input/folded-dataset/train_folds.csv") # Used abhishek's 5-folded dataset for validation. Only fold=0 validation is done for simplicity.
test = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/test.csv")
ss = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/sample_submission.csv')


# In[ ]:


from sklearn.model_selection import train_test_split

def get_train_val_with_fold(folded_train, fold, sentiment=None, fake_ratio=None):
    if fold > 4 or fold < 0:
        return None, None
    
    train = folded_train[folded_train["kfold"] != fold]
    valid = folded_train[folded_train["kfold"] == fold]
    
    train.pop("kfold")
    valid.pop("kfold")
    
    if fake_ratio:
        if sentiment == None:
            return None, None
        
        positive = train[train["sentiment"] == "positive"]
        fake_negative = positive.sample(n=int(fake_ratio * len(positive)), random_state=None)
        fake_negative["sentiment"] = "negative"
        
        negative = train[train["sentiment"] == "negative"]
        fake_positive = negative.sample(n=int(fake_ratio * len(negative)), random_state=None)
        fake_positive["sentiment"] = "positive"
        
        train = pd.concat([train, fake_positive, fake_negative])
        
    if sentiment:
        train = train[train["sentiment"] == sentiment]
    
    return train, valid


# In[ ]:


train_set, valid_set = get_train_val_with_fold(train, 0)


# In[ ]:


import json

np_train = np.array(train_set)
np_test = np.array(valid_set)


# In[ ]:


def find_all(input_str, search_str):
    l1 = []
    length = len(input_str)
    index = 0
    while index < length:
        i = input_str.find(search_str, index)
        if i == -1:
            return l1
        l1.append(i)
        index = i + 1
    return l1


# In[ ]:


get_ipython().system('rm -rf data-fold-0')


# In[ ]:


get_ipython().system('mkdir data-fold-0')
DATA_DIR = "data-fold-0"


# In[ ]:


output = {}
output["version"] = "v1.0"
output["data"] = []

for line in np_train:
    paragraphs = []
    context = line[1]
    
    qas = []
    question = line[-1]
    qid = line[0]
    answers = []
    answer = line[2]
    if type(answer) != str or type(context) != str or type(question) != str:
        print(context, type(context))
        print(answer, type(answer))
        print(question, type(question))
        continue
    answer_starts = find_all(context, answer)
    for answer_start in answer_starts:
        answers.append({'answer_start': answer_start, 'text': answer})
    qas.append({'question': question, 'id': qid, 'is_impossible': False, 'answers': answers})
    
    paragraphs.append({'context': context, 'qas': qas})
    output['data'].append({'title': 'None', 'paragraphs': paragraphs})

with open(f"{DATA_DIR}/train.json", 'w') as outfile:
    json.dump(output, outfile)


# In[ ]:


output = {}
output['version'] = 'v1.0'
output['data'] = []

for line in np_test:
    paragraphs = []
    
    context = line[1]
    
    qas = []
    question = line[-1]
    qid = line[0]
    if type(context) != str or type(question) != str:
        print(context, type(context))
        print(answer, type(answer))
        print(question, type(question))
        continue
    answers = []
    answers.append({'answer_start': 1000000, 'text': '__None__'})
    qas.append({'question': question, 'id': qid, 'is_impossible': False, 'answers': answers})
    
    paragraphs.append({'context': context, 'qas': qas})
    output['data'].append({'title': 'None', 'paragraphs': paragraphs})

with open(f"{DATA_DIR}/test.json", 'w') as outfile:
    json.dump(output, outfile)


# In[ ]:


get_ipython().system('git clone https://github.com/huggingface/transformers')


# In[ ]:


get_ipython().system('cd transformers && pip install .')


# In[ ]:


get_ipython().system('rm -rf /kaggle/working/cached-roberta-large-pretrained/cache')
get_ipython().system('rm -rf results_roberta_large_all')


# In[ ]:


get_ipython().system('python /kaggle/working/transformers/examples/run_squad.py --model_type roberta --model_name_or_path /kaggle/working/results_roberta_pretrain --config_name  /kaggle/input/train-config/config.json --do_lower_case --do_train --do_eval --data_dir ./data-fold-0 --cache_dir /kaggle/working/cached-roberta-large-pretrained/cache --train_file train.json --predict_file test.json --learning_rate 5e-5 --num_train_epochs 3 --max_seq_length 192 --doc_stride 64 --output_dir results_roberta_large_all --per_gpu_eval_batch_size=16 --per_gpu_train_batch_size=16 --save_steps=100000')


# In[ ]:


def post_process(selected):
    return " ".join(set(selected.lower().split())) 


# In[ ]:


def jaccard(str1, str2): 
    a = set(str1.lower().split()) 
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


# # Simple Validation

# In[ ]:


# Copy predictions to submission file.
predictions = json.load(open('./results_roberta_large_all/predictions_.json', 'r'))

jaccards = []
for idx, row in valid_set.iterrows():
    if row['sentiment'] == 'neutral': # neutral postprocessing
        id_ = row['textID']
        jaccards.append(jaccard(row["text"], row['selected_text']))
    else:
        id_ = row['textID']
        jaccards.append(jaccard(post_process(predictions[id_]), row['selected_text']))

#     if row['sentiment'] == 'neutral':
#         continue
#     else:
#         id_ = row['textID']
#         jaccards.append(jaccard(post_process(predictions[id_]), row['selected_text']))
        
#     if row['sentiment'] == 'positive': 
#         id_ = row['textID']
#         jaccards.append(jaccard(post_process(predictions[id_]), row['selected_text']))
        
#     if row['sentiment'] == 'negative': 
#         id_ = row['textID']
#         jaccards.append(jaccard(post_process(predictions[id_]), row['selected_text']))


# In[ ]:


sum(jaccards)/len(jaccards)


# In[ ]:




