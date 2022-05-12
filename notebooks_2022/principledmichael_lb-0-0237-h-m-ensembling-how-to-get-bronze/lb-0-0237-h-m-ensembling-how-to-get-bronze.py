#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import numpy as np
import pandas as pd
import gc


# # To ensemble I used submissions from 9 public notebooks:
# * LB: 0.0225 - https://www.kaggle.com/lunapandachan/h-m-trending-products-weekly-add-test/notebook
# * LB: 0.0217 - https://www.kaggle.com/tarique7/hnm-exponential-decay-with-alternate-items/notebook
# * LB: 0.0221 - https://www.kaggle.com/astrung/lstm-sequential-modelwith-item-features-tutorial
# * LB: 0.0224 - https://www.kaggle.com/code/hirotakanogami/h-m-eda-customer-clustering-by-kmeans
# * LB: 0.0220 - https://www.kaggle.com/code/hengzheng/time-is-our-best-friend-v2/notebook
# * LB: 0.0227 - https://www.kaggle.com/code/hechtjp/h-m-eda-rule-base-by-customer-age
# * LB: 0.0231 - https://www.kaggle.com/code/ebn7amdi/trending/notebook?scriptVersionId=90980162
# * LB: 0.0225 - https://www.kaggle.com/code/mayukh18/svd-model-reranking-implicit-to-explicit-feedback

# In[ ]:



sub0 = pd.read_csv('../input/hm-00231-solution/submission.csv').sort_values('customer_id').reset_index(drop=True)                                             # 0.0231
sub1 = pd.read_csv('../input/handmbestperforming/h-m-trending-products-weekly-add-test.csv').sort_values('customer_id').reset_index(drop=True)                # 0.0225
sub2 = pd.read_csv('../input/handmbestperforming/hnm-exponential-decay-with-alternate-items.csv').sort_values('customer_id').reset_index(drop=True)           # 0.0217
sub3 = pd.read_csv('../input/handmbestperforming/lstm-sequential-modelwith-item-features-tutorial.csv').sort_values('customer_id').reset_index(drop=True)     # 0.0221
sub4 = pd.read_csv('../input/hm-00224-solution/submission.csv').sort_values('customer_id').reset_index(drop=True)                                             # 0.0224
sub5 = pd.read_csv('../input/handmbestperforming/time-is-our-best-friend-v2.csv').sort_values('customer_id').reset_index(drop=True)                           # 0.0220
sub6 = pd.read_csv('../input/handmbestperforming/rule-based-by-customer-age.csv').sort_values('customer_id').reset_index(drop=True)                           # 0.0227
sub7 = pd.read_csv('../input/h-m-faster-trending-products-weekly/submission.csv').sort_values('customer_id').reset_index(drop=True)                           # 0.0231
sub8 = pd.read_csv('../input/h-m-framework-for-partitioned-validation/submission.csv').sort_values('customer_id').reset_index(drop=True)                      # 0.0225


# In[ ]:


sub0.columns = ['customer_id', 'prediction0']
sub0['prediction1'] = sub1['prediction']
sub0['prediction2'] = sub2['prediction']
sub0['prediction3'] = sub3['prediction']
sub0['prediction4'] = sub4['prediction']
sub0['prediction5'] = sub5['prediction']
sub0['prediction6'] = sub6['prediction']
sub0['prediction7'] = sub7['prediction']
sub0['prediction8'] = sub8['prediction'].astype(str)

del sub1, sub2, sub3, sub4, sub5, sub6, sub7, sub8
gc.collect()
sub0.head()


# In[ ]:


def cust_blend(dt, W = [1,1,1,1,1,1,1,1,1]):
    #Global ensemble weights
    #W = [1.15,0.95,0.85]

    #Create a list of all model predictions
    REC = []

    # Second Try
    REC.append(dt['prediction0'].split())
    REC.append(dt['prediction1'].split())
    REC.append(dt['prediction2'].split())
    REC.append(dt['prediction3'].split())
    REC.append(dt['prediction4'].split())
    REC.append(dt['prediction5'].split())
    REC.append(dt['prediction6'].split())
    REC.append(dt['prediction7'].split())
    REC.append(dt['prediction8'].split())
    #Create a dictionary of items recommended.
    #Assign a weight according the order of appearance and multiply by global weights
    res = {}
    for M in range(len(REC)):
        for n, v in enumerate(REC[M]):
            if v in res:
                res[v] += (W[M]/(n+1))
            else:
                res[v] = (W[M]/(n+1))

    # Sort dictionary by item weights
    res = list(dict(sorted(res.items(), key=lambda item: -item[1])).keys())

    # Return the top 12 items only
    return ' '.join(res[:12])

sub0['prediction'] = sub0.apply(cust_blend, W = [0.425,0.88,0.72,0.8,0.88,0.7,0.92,0.92,1.22], axis=1)
sub0.head()


# # Make a submission

# In[ ]:


del sub0['prediction0']
del sub0['prediction1']
del sub0['prediction2']
del sub0['prediction3']
del sub0['prediction4']
del sub0['prediction5']
del sub0['prediction6']
del sub0['prediction7']
del sub0['prediction8']
gc.collect()


sub0.to_csv('submission.csv', index=False)


# In[ ]:




