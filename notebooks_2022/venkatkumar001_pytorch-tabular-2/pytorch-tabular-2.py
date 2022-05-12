#!/usr/bin/env python
# coding: utf-8

# # <h><center> ⭐️⭐️Tabular Playground Series May 2022⭐️⭐️ </center></h>
# 
# ## **The goal of these competitions is to provide a fun and approachable-for-anyone tabular dataset to model.** 
# 
# <img src='https://deepandshallowml.files.wordpress.com/2021/01/pytorch_tabular_header.jpg'>
# 
# 
# ### **Try different! I am trying to Pytorch_Tabular(deeplearning)**

# # **Install Pytorch tabular library**

# In[ ]:


get_ipython().system('pip install pytorch_tabular[all]')


# # **Import Necessary Library**

# In[ ]:


#Import necessary Libraries
import pandas as pd
import numpy as np
import datetime as dt

from pytorch_tabular import TabularModel
from pytorch_tabular.models import CategoryEmbeddingModelConfig
from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig, ExperimentConfig


# In[ ]:


start_time=dt.datetime.now()
print("started at",start_time)


# # **Load, Read, Shape of Data**

# In[ ]:


train = pd.read_csv("../input/tabular-playground-series-may-2022/train.csv")
test = pd.read_csv("../input/tabular-playground-series-may-2022/test.csv")
sample = pd.read_csv("../input/tabular-playground-series-may-2022/sample_submission.csv")
print(f'train_shape: {train.shape},test_shape: {test.shape},sample_shape: {sample.shape}')
train.head()


# 
# # **Pytorch-Tabular details**
# 
# ## **Setting up the Configs (Pytorch-tabular)**
# 
# ### ***There are four configs that you need to provide(most of them have intelligent default values), which will drive the rest of the process.***
# 
# - DataConfig — Define the target column names, categorical and numerical column names, any transformation you need to do, etc.
# 
# - ModelConfig — There is a specific config for each of the models. This determines which model we are going to train and also lets you define the hyperparameters of the model
# 
# - TrainerConfig — This let’s you configure the training process by setting things like batch_size, epochs, early stopping, etc. The vast majority of parameters are directly borrowed from PyTorch Lightning and is passed to the underlying Trainer object during training
# 
# - OptimizerConfig — This let’s you define and use different Optimizers and LearningRate Schedulers. Standard PyTorch Optimizers and Learning RateSchedulers are supported. For custom optimizers, you can use the parameter in the fit method to overwrite this. The custom optimizer should be PyTorch compatible
# 
# - ExperimentConfig — This is an optional parameter. If set, this defines the Experiment Tracking. Right now, only two experiment tracking frameworks are supported: Tensorboard and Weights&Biases. W&B experiment tracker has more features like tracking the gradients and logits across epochs.
# 
# 

# # **Feature Selection**

# In[ ]:


num_col_names = ['f_00', 'f_01', 'f_02', 'f_03', 'f_04', 'f_05']
cat_col_names = ['f_07','f_08', 'f_09', 'f_10']

data_config = DataConfig(
    target=['target'], #target should always be a list. Multi-targets are only supported for regression. Multi-Task Classification is not implemented
    continuous_cols=num_col_names,
    categorical_cols=cat_col_names,
)
trainer_config = TrainerConfig(
    auto_lr_find=True, # Runs the LRFinder to automatically derive a learning rate
    batch_size=32,
    max_epochs=2,
    #index of the GPU to use. 0, means CPU
)
optimizer_config = OptimizerConfig()

model_config = CategoryEmbeddingModelConfig(
    task="classification",
    layers="1024-512-256-64", # Number of nodes in each layer
    activation="LeakyReLU", # Activation between each layers
    learning_rate = 1e-4
)


# # **Initializing the Model & Training**

# In[ ]:


tabular_model = TabularModel(
    data_config=data_config,
    model_config=model_config,
    optimizer_config=optimizer_config,
    trainer_config=trainer_config,
)
tabular_model.fit(train=train)


# # **Predict Output and Generate Submission file**

# In[ ]:


result = tabular_model.evaluate(test)
pred_df = tabular_model.predict(test)


# In[ ]:


pred_df.head()


# In[ ]:


pred = pred_df.prediction


# In[ ]:


output = pd.DataFrame({'id': test.id, 'target': pred})
output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")


# In[ ]:


output.sample(20)


# In[ ]:


print('pytorch tabular finished!!')
finish_time = dt.datetime.now()
print("Finished at ", finish_time)
elapsed = finish_time - start_time
print("Elapsed time: ", elapsed)


# ## **⭐️⭐️Thankyou for visiting guys⭐️⭐️**
# 
# ## **if you will interest in audio and text data i was created starter notebook! go and explore it**
# 
# 1. https://www.kaggle.com/code/venkatkumar001/nlp-starter-almost-all-basic-concept
# 2. https://www.kaggle.com/code/venkatkumar001/audio-starter-almost-all-basic-concepts
# 3. https://www.kaggle.com/venkatkumar001/fast-ai-tps-may22-let-s-try-new
# 
# 
# Reference: 
# 1. https://www.kaggle.com/code/venkatkumar001/apc-4-pytorch-tabular
# 2. https://pytorch-tabular.readthedocs.io/en/latest/
