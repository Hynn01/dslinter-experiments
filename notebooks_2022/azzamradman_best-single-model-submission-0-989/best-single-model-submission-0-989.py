#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd


# In[ ]:


sample = pd.read_csv('../input/tabular-playground-series-apr-2022/sample_submission.csv')
preds = pd.read_csv('../input/tps04-best-single-model-0-987/test_preds_array_9867.csv').values.flatten()


# In[ ]:


sample.iloc[:, -1] = preds
sample.to_csv('submission.csv', index=False)


# In[ ]:




