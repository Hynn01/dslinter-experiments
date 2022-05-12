#!/usr/bin/env python
# coding: utf-8

# # THIS KERNAL IS BLEND OF So awesome kernels present Right now
# # Vote if you love blend. 
# 
# ## Kernels used comming from these awesome people:
# ### For the TF-IDF submissions they are comming from this kernel:
# [NB-SVM strong linear baseline](https://www.kaggle.com/hamditarek/nb-svm-strong-linear-baseline)
# 
# [[TPU-Inference] Super Fast XLMRoberta](https://www.kaggle.com/shonenkov/tpu-inference-super-fast-xlmroberta)
# 
# [Jigsaw TPU: BERT with Huggingface and Keras](https://www.kaggle.com/miklgr500/jigsaw-tpu-bert-with-huggingface-and-keras)
# 
# [inference of bert tpu model ml w/ validation](https://www.kaggle.com/abhishek/inference-of-bert-tpu-model-ml-w-validation)
# [Train from MLM finetuned XLM-R large](https://www.kaggle.com/riblidezso/train-from-mlm-finetuned-xlm-roberta-large)

# # phase 1 [Ensemble]

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# 0.9422 /kaggle/input/train-from-mlm-finetuned-xlmr-large/submission (71).csv


# In[ ]:


#submission1 = pd.read_csv('/kaggle/input/train-from-mlm-finetuned-xlmr-large/submission (71).csv')
submission2 = pd.read_csv('/kaggle/input/009473-v/submission - 2020-06-23T075709.806.csv')
submission1 = pd.read_csv('/kaggle/input/009488/submission - 2020-06-23T074910.830.csv')
submission3 = pd.read_csv('../input/tpuinference-super-fast-xlmroberta/submission (47).csv')


# # Hist Graph of scores

# In[ ]:


sns.set()
plt.hist(submission1['toxic'],bins=100)
plt.show()


# In[ ]:


sns.set()
plt.hist(submission2['toxic'],bins=100)
plt.show()


# In[ ]:


submission1['toxic'] = submission1['toxic']*0.9 + submission2['toxic']*0.1


# In[ ]:


submission1.to_csv('submission.csv', index=False)

