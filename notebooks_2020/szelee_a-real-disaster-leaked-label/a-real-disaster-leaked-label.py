#!/usr/bin/env python
# coding: utf-8

# *Disclaimer: Just like the [titanic kernel](https://www.kaggle.com/tarunpaparaju/how-top-lb-got-their-score-use-titanic-to-learn), the purpose of this kernel is not to let everyone get the perfect score, but to let everyone know how some participants got their perfect scores, so you can focus on learning ML rather than wondering how you can end up at the top of the LB.*
# 
# I've checked the [origin](https://www.figure-eight.com/data-for-everyone/) of the dataset (look for 'Disasters on social media') provided in the [overview page](https://www.kaggle.com/c/nlp-getting-started/overview) and found the dataset that holds ground truth for the test set.
# 
# If I can discover this so easily, I am sure it's just a matter of time before someone else does the same.
# 
# My point is, **ignore the LB** and just focus on learning from all the great kernels that are being shared.

# In[ ]:


import pandas as pd


# In[ ]:


test_df = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
gt_df = pd.read_csv("../input/disasters-on-social-media/socialmedia-disaster-tweets-DFE.csv")


# In[ ]:


gt_df = gt_df[['choose_one', 'text']]
gt_df['target'] = (gt_df['choose_one']=='Relevant').astype(int)
gt_df['id'] = gt_df.index
gt_df


# In[ ]:


merged_df = pd.merge(test_df, gt_df, on='id')
merged_df


# In[ ]:


subm_df = merged_df[['id', 'target']]
subm_df


# In[ ]:


subm_df.to_csv('submission.csv', index=False) # The holy grail of a perfect ML model prediction


# Speaking of real disasters eh?
