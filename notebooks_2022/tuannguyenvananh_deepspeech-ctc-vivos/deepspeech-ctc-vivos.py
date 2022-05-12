#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('git init')
get_ipython().system('git remote add origin https://github.com/tuanio/deepspeech-ctc')
get_ipython().system('git pull origin main')


# In[ ]:


get_ipython().system('pip install -U -r requirements.txt')


# In[ ]:


import os
os.environ['HYDRA_FULL_ERROR'] = '1'


# In[ ]:


from pathlib import Path

config_path = 'custom_conf/'
Path(f"{config_path}/").mkdir(parents=True, exist_ok=True)
with open(f'{config_path}/configs.yaml', 'w', encoding='utf-8') as f:
    data = r"""dataset:
  root: ../../../../input/vivos-dataset/vivos
  n_fft: 400

text_process:
  lang: vi

datamodule:
  batch_size: 16

model:
  lr: 0.0001
  n_feature: 201 # n_fft // 2 + 1
  n_hidden: 2048
  dropout: 0.1

optimizer:
  momentum: 0.99
  nesterov: True

logger:
  save_dir: loggers
  name: deepspeech

trainer:
  max_epochs: 200
  accelerator: auto
  detect_anomaly: True
  accumulate_grad_batches: 8

ckpt:
  have_ckpt: True
  ckpt_path: ../../../../input/deep-speech-ctc-vivos-ckpt/epoch59-step8760.ckpt"""
    f.write(data)


# In[ ]:


get_ipython().system('git pull origin main')


# In[ ]:


get_ipython().run_line_magic('run', 'main.py -cp custom_conf -cn configs')


# In[ ]:


get_ipython().system('zip outputs.zip -r outputs/')

