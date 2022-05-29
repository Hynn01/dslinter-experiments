#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('git init')
get_ipython().system('git remote add origin https://github.com/tuanio/conformer-rnnt')
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
    data = r"""datasets:
  librispeech:
    clean_path: ../../../../input/librispeech-clean/LibriSpeech/
    other_path: /home/fablab/Downloads/librispeech/LibriSpeech/
    n_fft: 400
    db_path: /home/fablab/Downloads/librispeech/LibriSpeech/pkl/
    train: dev
    val: dev
    test: test

  vivos:
    root: ../../../../input/vivos-dataset/vivos
    n_fft: 400

  dataset_selected: vivos

text_process:
  lang: vi

datamodule:
  vivos:
    batch_size: ${training.batch_size}

model:
  num_classes: -1
  encoder_dim: 144
  decoder_output_dim: ${model.encoder_dim}
  hidden_state_dim: 320
  decoder_num_layers: 1
  input_dim: 201 # n_fft // 2 + 1
  num_heads: 4
  num_layers: 16
  conv_kernel_size: 31
  feed_forward_expansion_factor: 4
  conv_expansion_factor: 2
  dropout: 0.1
  half_step_residual: True
  subsampling_factor: 4
  freq_masks: 2
  time_masks: 10
  freq_width: 27
  time_width: 0.05
  rnn_type: lstm
  sos_id: 1
  eos_id: 2
  grad_ckpt_batchsize: 4

training:
  lr: 0.0001
  batch_size: 4
  max_epoch: 13
  dataloader_numworkers: 8

optim:
  betas: [0.9, 0.98]
  weight_decay: 1e-3

sched:
  T_0: 1000
  eta_min: 0.00001
  last_epoch: -1
  verbose: True

tb_logger:
  save_dir: tb_logs
  name: conformer_logs

trainer:
  max_epochs: ${training.max_epoch}
  enable_progress_bar: True
  accelerator: auto
  detect_anomaly: True
  accumulate_grad_batches: 8

ckpt:
  have_ckpt: True
  ckpt_path: ../../../../input/conformer-rnnt-vivos-ckpt/epoch5-step2190.ckpt
  train: True"""
    f.write(data)


# In[ ]:


get_ipython().system('git pull origin main ')


# In[ ]:


get_ipython().run_line_magic('run', 'main.py -cp custom_conf -cn configs')
