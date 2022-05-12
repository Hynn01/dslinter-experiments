#!/usr/bin/env python
# coding: utf-8

# # <h><center>⭐️⭐️ Tabular Playground Series May 2022 ⭐️⭐️</center></h>
# 
# ## **The goal of these competitions is to provide a fun and approachable-for-anyone tabular dataset to model.** 
# 
# <img src='https://miro.medium.com/max/1200/1*PQTzNNvBlmjW0Eca-nw14g.png'>
# 
# 
# ## **Try different! I am trying one of my favourite framework Fastai**
# 
# ### ***Fast.ai : https://www.fast.ai/***

# In[ ]:


get_ipython().system('pip3 install --upgrade fastai')


# In[ ]:


get_ipython().system('pip install fast_tabnet')


# # **Import Necessary Library**

# In[ ]:


from fastai.tabular.all import *
from fast_tabnet.core import *
from pathlib import Path


# In[ ]:


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic = True
seed_everything(1234)


# # **Load, Read Data**

# In[ ]:


root = "../input/tabular-playground-series-may-2022"
train = Path(root)/"train.csv"
test = Path(root )/"test.csv"
sub = Path(root )/"sample_submission.csv"

df = pd.read_csv(train)
dft = pd.read_csv(test)
sample = pd.read_csv(sub)
df.head()


# # **Feature Selections**

# In[ ]:


dep_var  = 'target'
cont_names = ['f_00', 'f_01', 'f_02', 'f_03', 'f_04', 'f_05', 'f_06','f_11', 'f_12', 'f_13',
                 'f_17', 'f_18', 'f_19', 'f_20', 'f_21', 'f_22', 'f_23', 'f_24', 'f_25','f_26', 'f_28', ]
cat_names = ['f_27']
procs = [Categorify,FillMissing,Normalize]


# In[ ]:


split_sample = np.random.choice(df.shape[0],400)


# # **Build the Model**

# In[ ]:


dls = TabularDataLoaders.from_df(df,root,procs,cat_names,cont_names,y_names=dep_var,valid_idx=split_sample,bs=64,y_block=CategoryBlock)


# In[ ]:


learn = tabular_learner(dls, model_dir="/tmp/model/", metrics=[accuracy]).to_fp16()


# In[ ]:


dls.valid.show_batch()


# # **Find learning rate and finetuning**

# In[ ]:


learn.lr_find()


# In[ ]:


learn.fine_tune(20,1e-3)


# In[ ]:


learn.lr_find()


# In[ ]:


learn.fit_one_cycle(20,lr_max=1e-3)


# In[ ]:


learn.lr_find()


# In[ ]:


learn.fit_one_cycle(50, lr_max=2e-7)


# # **Confusion Matrix**

# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix(figsize=(5,5), dpi=60)


# # **Predict_Output in Test data**

# In[ ]:


dft = pd.read_csv(test)
dl = learn.dls.test_dl(dft, bs=64)
dlp, _  = learn.get_preds(dl=dl)


# # **Submission file**

# In[ ]:


sample.target = np.argmax(dlp, axis=1)
sample.to_csv('submission.csv', index=False)
print('sucessfully generate submission file')


# In[ ]:


sample.sample(2)


# ## **⭐️⭐️Thankyou for visiting guys⭐️⭐️**
# 
# ## **if you are interested i was created starter notebook in (Audio and Nlp notebooks)**
# 
# 1. https://www.kaggle.com/code/venkatkumar001/nlp-starter-almost-all-basic-concept
# 2. https://www.kaggle.com/code/venkatkumar001/audio-starter-almost-all-basic-concepts
# 
# Reference: 
# 
# 1. https://www.kaggle.com/code/krishnakalyan3/titanic-fast-ai-2-0-tabular-minimal-example
# 2. https://docs.fast.ai/tabular.learner.html
