#!/usr/bin/env python
# coding: utf-8

# # About this notebook
# In this competition, the metric is correlation coefficient.<br>
# When ensembling the results of multiple models, the ensemble may not be properly effective if there are differences in the scales of each model.
# 
# 
# Please upvote if this helps!
# 
# 
# This is a bit of an extreme example, but let's experiment with it below.

# In[ ]:


import numpy as np


# In[ ]:


target = np.array([1.0, 1.0, 0.5, 0, 0.25])

pred1 = np.array([10, 3, 7, 1, 4])
pred2 = np.array([1.0, 0.5, 0.4, 0.4, 0.1])


# In[ ]:


print('pred1 corrcoef: ', np.corrcoef(target, pred1)[0][1])
print('pred2 corrcoef: ', np.corrcoef(target, pred2)[0][1])


# Since metric is a correlation coefficient, the score will not be so low in cases like pred1. (This is an extreme example.)
# 
# 
# Let's ensemble this as it is and calculate the score.

# In[ ]:


ensamble = (pred1 + pred2) / 2
print('ensabled array: ', ensamble)
print('ensabled corrcoef: ', np.corrcoef(target, ensamble)[0][1])


# When ensembling predictions with different scales, the ensemble is not correct, approaching the larger absolute value.
# 
# So I ensemble after scaling.

# In[ ]:


from sklearn.preprocessing import MinMaxScaler

MMscaler = MinMaxScaler()

pred1_mm = MMscaler.fit_transform(pred1.reshape(-1,1)).reshape(-1)
pred2_mm = MMscaler.fit_transform(pred2.reshape(-1,1)).reshape(-1)

print('pred1 array after　scaling: ', pred1_mm)
print('pred1 corrcoef after scaling: ', np.corrcoef(target, pred1_mm)[0][1])
print('')
print('pred2 array after　scaling: ', pred2_mm)
print('pred2 corrcoef after　scaling: ', np.corrcoef(target, pred2_mm)[0][1])


# Scores do not change after scaling.
# 
# 
# I'll try to ensemble these.

# In[ ]:


ensamble_mm = (pred1_mm + pred2_mm) / 2

print('ensabled array　after　scaling: ', ensamble_mm)
print('ensabled corrcoef　after　scaling: ', np.corrcoef(target, ensamble_mm)[0][1])


# This seems to be a more correct ensemble for our purposes.
