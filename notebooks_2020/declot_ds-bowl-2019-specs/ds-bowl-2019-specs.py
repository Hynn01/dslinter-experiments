#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# Any results you write to the current directory are saved as output.


# In[ ]:


import scipy
import sklearn


# In[ ]:


specs = pd.read_csv('/kaggle/input/data-science-bowl-2019/specs.csv')
specs


# In[ ]:


data = pd.read_csv('/kaggle/input/data-science-bowl-2019/train.csv')
test = pd.read_csv('/kaggle/input/data-science-bowl-2019/test.csv')


# # Event_id

# In[ ]:


ids = specs.event_id.unique()[~np.isin(specs.event_id.unique(), data.event_id.unique())]
print(ids)
specs.loc[specs.event_id.isin(ids), 'info'].values


# In[ ]:


np.isin(ids, test.event_id.unique())


# # Args

# In[ ]:


import json


# In[ ]:


args = np.vectorize(json.loads)(specs.args.values)


# In[ ]:


def get_names(args_one_event_id, name_unique):
    for info in args_one_event_id:
#         print(info, info['name'])
        name_unique[info['name']] = info['info']


# In[ ]:


name_unique = dict()

np.vectorize(lambda x: get_names(x, name_unique=name_unique))(args)
len(name_unique)


# In[ ]:


name_unique


# Here is names of event_id which looks more important ro prediction accuracy:
# * 'round': 'number of the current round when the event takes place or 0 if no round',
# * 'level': 'number of the current level when the event takes place or 0 if no level',
# * 'media_type': "the type of media that has just played:\n'audio' || 'animation' || 'other'",
# * 'duration': 'the duration of the media playback in milliseconds',
# * 'dwell_time': 'how long the mouse cursor dwells over the object when the player dwells for longer than 1sec',
# *  'misses': 'the number of times the done button was pressed with the chests in the wrong location',
# *  'prompt': '“bucket that holds the most”, “holds least”, etc. – the prompt given to the player',
# * 'mode': '“arranging” or “picking” – current mode when the help button is clicked',
# * 'round_number': 'the number of the round that is about to start',
# * 'exit_type': '“closed container”, “game ended”, or browser exit/navigation',
# * 'tutorial_step': 'the current step of the tutorial when the tutorial was skipped',
# * 'time_played': 'the time the media has been playing milliseconds when the button is pressed',
# * 'round_prompt': 'the prompt given to the player e.g. between cliff and tree" etc."',
# *  'target_water_level': 'level the water must reach to pass the round',
#  
#  So looking on text by eyes one can lost some info. Use tokenization and lemming to find all event_id names with : 'target', 'prompt', 'help', 'media', 'tutorial', 'wrong', 'correct', 'pass' in description

# In[ ]:


names = set(['round', 'level','media_type', 'duration', 'dwell_time', 'misses', 'prompt',
         'mode', 'round_number', 'exit_type', 'tutorial_step','time_played',
         'round_prompt', 'target_water_level'])
to_search = pd.Series(['target', 'prompt', 'help', 'media', 'tutorial', 'wrong', 'correct', 'pass'])

print(len(names))


# In[ ]:


import nltk
from nltk.stem import *


# In[ ]:


name_unique = pd.Series(name_unique)
name_unique = name_unique.apply(nltk.word_tokenize)
name_unique


# In[ ]:


lemma =  WordNetLemmatizer()
name_unique = name_unique.apply(lambda x: [lemma.lemmatize(xi, pos='v') for xi in x])


# In[ ]:


name_unique


# In[ ]:


res = name_unique.apply( lambda x: to_search.isin(x).any())
names.update(res[res].index)
len(names)


# In[ ]:


name_unique = dict()

np.vectorize(lambda x: get_names(x, name_unique=name_unique))(args)
for name in names:
    print(name, ':', name_unique[name])


# In[ ]:


to_drop = ['target_containers', 'target_bucket', 'round_target', 'target_distances'] # just info about target
for name in to_drop:
    names.remove(name)
    
len(names)


# In[ ]:


def get_event_data(temp):
    args = json.loads(temp)
    return {k:args[k] for k in names if k in args.keys()}


# In[ ]:


get_ipython().run_cell_magic('time', '', "event_data = pd.DataFrame(data.event_data.apply(get_event_data).tolist())\n# event_data.columns = 'event_data_'+event_data.columns ")


# In[ ]:


event_data.nunique()


# In[ ]:


categorical = event_data.columns[event_data.nunique() < 10]
categorical


# In[ ]:


for f in categorical:
    display(event_data[f].value_counts(), name_unique[f])


# 'exit_type' has smallest number of values. Drop it too.

# In[ ]:


names.remove('exit_type')
len(categorical)


# In[ ]:


real = event_data.columns[event_data.apply(lambda x: x.nunique()) >= 10]
len(real), real


# In[ ]:


event_data['level'].plot.hist()


# In[ ]:


event_data['misses'].plot.hist()


# In[ ]:


names, len(names)


# Data for these new features wiil used in train_test set

# In[ ]:


event_data[names].isna().sum()/ event_data.shape[0]


# # Train data for event_id

# In[ ]:


data_event_id = data.groupby('event_id')[['event_code', 'title', 'type', 'world']].agg([lambda x: x.unique(), lambda x: x.nunique()])
data_event_id


# In[ ]:


for f in ['event_code', 'title', 'type', 'world']:
    if data_event_id[data_event_id[f]['<lambda_1>'] != 1].shape[0] == 0:
        assert np.isin(data_event_id[f]['<lambda_0>'].unique(), data[f].unique()).all()
    else: display(data_event_id[data_event_id[f]['<lambda_1>'] != 1])
    


# Only one event_id corresponds to different titles and worlds. But this event_id always have type=Clip

# In[ ]:


specs[specs.event_id == '27253bdc']['info'].values


# In[ ]:


specs_ini = specs.copy()
# specs = specs[specs.event_id != '27253bdc']


# In[ ]:


specs = specs_ini.copy()


# In[ ]:


for index in data_event_id.columns.levels[0]:
    data_event_id.drop(columns=(index, '<lambda_1>'), inplace=True)
    
data_event_id.columns = data_event_id.columns.droplevel(1)
specs = specs.merge(data_event_id.reset_index(), on='event_id', how='outer') 


# In[ ]:


specs.head()


# # Text study - info
# Just for practice in word embedding

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[ ]:


tfidf = TfidfVectorizer(ngram_range=(1,3), stop_words='english', max_df=0.9, min_df=5)
description = tfidf.fit_transform(specs['info'])
description = pd.DataFrame(description.toarray())
description


# In[ ]:


num_to_word = {k:v for v,k in tfidf.vocabulary_.items()}
files_num = list(range(description.shape[0]))
num_to_event_id= {k:v for k,v in enumerate(specs.event_id)}


# In[ ]:


from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


# idx = np.argsort(new.toarray(), axis=1)[:, -10:]


# In[ ]:


similarity = cosine_similarity(description, description)# dense_output=False)
similarity = np.tril(similarity, k=-1)


# In[ ]:


threshold = 0.999
n_similar_ids = (similarity >= threshold).sum(axis=0) 
# calculate total numbers of id which have duplicates
(n_similar_ids > 0).sum(), (n_similar_ids > 0).sum()/similarity.shape[0]*100


# In[ ]:


to_drop = set()
for i in range(similarity.shape[1]):
    add = np.argwhere(similarity[:, i] >= threshold).ravel().tolist()
    print(i, add)    
    to_drop.update(add)
new = description.drop(index=to_drop)
new


# In[ ]:


new.shape


# In[ ]:


# check
specs['info'].iloc[[219, 223, 224, 239, 240, 319, 320, 323, 344, 360]].values


# In[ ]:


# now we want to see distribution of similarity coeffitient
similarity = cosine_similarity(new, new)
similarity = np.tril(similarity, k=-1)

n_similar_ids = (similarity >= 0.99).sum(axis=0) 
assert (n_similar_ids > 0).sum() == 0

idx = np.argwhere(similarity.ravel() != 0)
sns.distplot(similarity.ravel()[idx])


# Right tail corresponds to the simillary event_ids but differ only in some words. Lets find why these event_is are differ
