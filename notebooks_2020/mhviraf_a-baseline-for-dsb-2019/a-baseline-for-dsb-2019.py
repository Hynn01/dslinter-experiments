#!/usr/bin/env python
# coding: utf-8

# ## Summary
# I was confused as to what is it that we are trying to forecast in this competition. Apparently, if you group your test set by `installation_id` you will get a long list of the activities of a user. It contains games, clips, activities, Assessments, etc. We should use this history to predict the very last row of each `installation_id`. Below is an example for the first user in test set `00abaee7`. This data shows, he/she started the app, watched the welcome to app clip, then magma peak - level 1 clip and then two other clips. Then he/she played the "Chow Time" game. As you scroll down you can see all of his/her acitivies. At the very end of the activities list you can see that he/she started to play "Cauldron Filler" Assessment task which only has 1 row. The test set is truncated there to indicate that we should predict this user's Assessment on this task.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# In[ ]:


train = pd.read_csv('/kaggle/input/data-science-bowl-2019/train.csv')
train_labels = pd.read_csv('/kaggle/input/data-science-bowl-2019/train_labels.csv')
specs = pd.read_csv('/kaggle/input/data-science-bowl-2019/specs.csv')
test = pd.read_csv('/kaggle/input/data-science-bowl-2019/test.csv')
submission = pd.read_csv('/kaggle/input/data-science-bowl-2019/sample_submission.csv')


# In[ ]:


pd.set_option('max_rows', None)
test.query('installation_id=="00abaee7"').head(20)


# In[ ]:


test.query('installation_id=="00abaee7"').tail(5)


# ## Baseline model
# Now to get started with a very basic baseline and make everything more clear, I have used the mode value of each Assessment task in the train_labels set and used them as predictions. As you see it gives us LB 0.395

# In[ ]:


labels_map = dict(train_labels.groupby('title')['accuracy_group'].agg(lambda x:x.value_counts().index[0])) # get the mode
labels_map


# In[ ]:


submission['accuracy_group'] = test.groupby('installation_id').last()['title'].map(labels_map).reset_index(drop=True)
submission.to_csv('submission.csv', index=None)
submission.head()


# In[ ]:


submission['accuracy_group'].plot(kind='hist')


# In[ ]:


train_labels['accuracy_group'].plot(kind='hist')


# Hope this gets you started. Kudos to yasufuminakama for clarifying it for me. 
# 
# Happy kaggling!
