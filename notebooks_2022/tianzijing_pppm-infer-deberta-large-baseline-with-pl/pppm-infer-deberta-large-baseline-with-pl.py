#!/usr/bin/env python
# coding: utf-8

# #### Modified from https://www.kaggle.com/code/yasufuminakama/pppm-deberta-v3-large-baseline-inference
# #### Train notebook: https://www.kaggle.com/tianzijing/pppm-train-deberta-large-baseline-with-pl/edit

# In[ ]:


get_ipython().system('pip install ../input/pytorchlightning160/pytorch_lightning-1.6.0-py3-none-any.whl')


# ## Library

# In[ ]:


import os
os.system('pip uninstall -y transformers')
os.system('pip uninstall -y tokenizers')
os.system('python -m pip install --no-index --find-links=../input/pppm-pip-wheels-dataset transformers')
os.system('python -m pip install --no-index --find-links=../input/pppm-pip-wheels-dataset tokenizers')


# In[ ]:


# ====================================================
# Library
# ====================================================
import os
import gc
import re
import ast
import sys
import copy
import json
import time
import math
import shutil
import string
import pickle
import random
import joblib
import itertools
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import scipy as sp
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
from tqdm.auto import tqdm
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold

import torch
print(f"torch.__version__: {torch.__version__}")
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.optim import Adam, SGD, AdamW
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
print(f"pytorch_lightning.__version__: {pl.__version__}")

# import tokenizers
import transformers
# print(f"tokenizers.__version__: {tokenizers.__version__}")
print(f"transformers.__version__: {transformers.__version__}")
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
get_ipython().run_line_magic('env', 'TOKENIZERS_PARALLELISM=true')

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ## CFG

# In[ ]:


class CFG:
    ## 常规设置
#     data_dir = '/home/tzj/data/pppm-data/us-patent-phrase-to-phrase-matching'
    data_dir = '../input/us-patent-phrase-to-phrase-matching'
    output_dir = './'

    weight_path = [
        '../input/pppm-single'
    ]
    weight_scores = [
    ]
    t = 0.01
    cpc_dir = ''
    cpc_data_path = '../input/cpc-texts/cpc_texts.pth'

    debug = False

    seed = 6001
    n_fold = 4
    trn_fold = [0, 1, 2, 3]

    ## 数据设置
    num_workers = 4
    batch_size = 16
    max_len = 512
    pin_memory = True
    target_size = 1

    ## 模型设置
    model = "../input/deberta-v3-large/deberta-v3-large"
#     model = '/home/tzj/pretrained_models/en-deberta-v3-large'
    fgm = False
    label_smooth = False
    smoothing = 0.1

    fc_dropout = 0.2


# ## Utils

# In[ ]:


def get_score(y_true, y_pred):
    score = sp.stats.pearsonr(y_true, y_pred)[0]
    return score


# ### get_cpc_data

# In[ ]:


def get_cpc_texts(cpc_dir):
    # 找出每个 context 对应的 CPC 描述，包括大类描述和子类描述
    contexts = []
    pattern = '[A-Z]\d+'   # 找出 context， 如：A47
    for file_name in os.listdir(os.path.join(cpc_dir, 'CPCSchemeXML202105')):
        result = re.findall(pattern, file_name)
        if result:
            contexts.append(result)
    contexts = sorted(set(sum(contexts, [])))  # 将列表拼接，并去重排序
    results = {}
    for cpc in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'Y']:
        with open(os.path.join(cpc_dir, f'CPCTitleList202202/cpc-section-{cpc}_20220201.txt')) as f:
            s = f.read()
        pattern = f'{cpc}\t\t.+'
        result = re.findall(pattern, s)
        cpc_result = result[0].lstrip(pattern)  # 找出每个大类的描述
        for context in [c for c in contexts if c[0] == cpc]:
            pattern = f'{context}\t\t.+'
            result = re.findall(pattern, s)
            cpc_sub_result = result[0].lstrip(pattern) # 找出每个子类的描述
            results[context] = cpc_result + '. ' + cpc_sub_result
    return results


# ## DataModule

# ### CV split

# In[ ]:


def CV_group_split(dataset, n_splits=5, shuffle=True, seed=0, debug=False, debug_size=1000):
    # 本次比赛的score是离散型的分值，因此可以看成分类问题（回归问题也可以），因此将得分映射成 5 个类别，并根据类别标签分层采样
    dataset['score_map'] = dataset['score'].map({0.00: 0, 0.25: 1, 0.50: 2, 0.75: 3, 1.00: 4})
    Fold = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)

    for n, (train_index, val_index) in enumerate(Fold.split(dataset, dataset['score_map'])):
        dataset.loc[val_index, 'fold'] = int(n)
    dataset['fold'] = dataset['fold'].astype(int)
    if debug:
        dataset = dataset.sample(n=debug_size, random_state=seed).reset_index(drop=True)
    return dataset


# ### Get Tokenizer

# In[ ]:


def get_tokenizer(tokenizer_path):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    return tokenizer


# In[ ]:


class PPPMDataModule(pl.LightningDataModule):
    def __init__(self, config, prepare_train=True, prepare_test=True):
        super().__init__()
        self.prepare_data_per_node = False
        self.seed = config.seed
        self.debug = config.debug
        self.debug_size = 0 if self.debug == False else config.debug_size
        self.shuffle = (self.debug == False)
        self.batch_size = config.batch_size
        self.pin_memory = config.pin_memory
        self.num_workers = config.num_workers
        self.max_len = config.max_len
        self.data_dir = config.data_dir
        self.output_dir = config.output_dir
        self.cpc_dir = config.cpc_dir
        self.cpc_data_path = config.cpc_data_path
        self.n_fold = config.n_fold

        self.tokenizer = get_tokenizer(config.model)

        self.prepare_train = prepare_train
        self.prepare_test = prepare_test

    def set_trn_fold(self, trn_fold):
        self.trn_fold = trn_fold

    def load_train(self):
        train = pd.read_csv(Path(self.data_dir) / 'train.csv')
        if self.cpc_data_path != '':
            cpc_texts = torch.load(self.cpc_data_path)
        else:
            cpc_texts = get_cpc_texts(self.cpc_dir)
            torch.save(cpc_texts, Path(self.output_dir) / 'cpc_texts.pth')
        train['context_text'] = train['context'].map(cpc_texts)
        train['text'] = train['anchor'] + '[SEP]' + train['target'] + '[SEP]' + train['context_text']
        return train

    def load_test(self):
        test = pd.read_csv(Path(self.data_dir) / 'test.csv')
        submission = pd.read_csv(Path(self.data_dir) / 'sample_submission.csv')
        if self.cpc_data_path != '':
            cpc_texts = torch.load(self.cpc_data_path)
        else:
            cpc_texts = get_cpc_texts(self.cpc_dir)
            torch.save(cpc_texts, Path(self.output_dir) / 'cpc_texts.pth')
        test['context_text'] = test['context'].map(cpc_texts)
        test['text'] = test['anchor'] + '[SEP]' + test['target'] + '[SEP]' + test['context_text']
        return test, submission

    def calculate_max_len(self, dataset):
        dataset['text'].fillna('')
        tqdm.pandas(desc="text_lens")
        text_lens = dataset['text'].progress_apply(
            lambda x: len(self.tokenizer(x, add_special_tokens=False)['input_ids']))
        max_len_text = text_lens.max()
        return max_len_text + 2  # text 包括了 anchor + '[SEP]' + target + '[SEP]' + context_text，还要加上开头 ’[CLS]'，结尾 '[SEP]'

    def prepare_data(self):
        if self.prepare_train == True:
            train = self.load_train()
            # 将数据切分成 n 折
            train = CV_group_split(train, self.n_fold, self.shuffle, self.seed, self.debug, self.debug_size)
            self.train_max_len = self.calculate_max_len(train)
            self.train = train
            self.prepare_train = False
            print('Train data prepared!')

        if self.prepare_test == True:
            self.test, self.submission = self.load_test()
            self.test_max_len = self.calculate_max_len(self.test)
            self.prepare_test = False
            print('Test data prepared!')

    def setup(self, stage='fit'):
        if stage == 'fit':
            self.build_fit_dataset(trn_fold=self.trn_fold)

        elif stage == 'test':
            self.build_test_dataset()

        elif stage == 'predict':
            self.build_predict_dataset()

    def build_fit_dataset(self, trn_fold=None):
        df = self.train
        if trn_fold != None:
            self.train_df = df[df['fold'] != trn_fold].reset_index(drop=True)
            self.val_df = df[df['fold'] == trn_fold].reset_index(drop=True)
            self.train_dataset = PPPMDataset(self.train_df, self.tokenizer, self.train_max_len)
            self.val_dataset = PPPMDataset(self.val_df, self.tokenizer, self.train_max_len)

    def build_test_dataset(self):
        self.test_dataset = PPPMInferDataset(self.test, self.tokenizer, self.test_max_len)

    def build_predict_dataset(self):
        self.predict_dataset = PPPMInferDataset(self.test, self.tokenizer, self.test_max_len)

    def train_dataloader(self):
        loader = DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                            pin_memory=self.pin_memory, shuffle=self.shuffle)
        return loader

    def val_dataloader(self):
        loader = DataLoader(self.val_dataset, batch_size=self.batch_size * 4, num_workers=self.num_workers,
                            shuffle=False)
        return loader

    def test_dataloader(self):
        loader = DataLoader(self.test_dataset, batch_size=self.batch_size * 4, num_workers=self.num_workers,
                            shuffle=False)
        return loader

    def predict_dataloader(self):
        loader = DataLoader(self.predict_dataset, batch_size=self.batch_size * 4, num_workers=self.num_workers,
                            shuffle=False)
        return loader


# ## Dataset

# ### Prepare inputs

# In[ ]:


def tokenize(tokenizer, text, max_len=512, return_offsets_mapping=False):
    inputs = tokenizer(
        text, 
        add_special_tokens=True,  # cls, sep
        max_length=max_len,
        padding='max_length',
        return_offsets_mapping=return_offsets_mapping)
    for k, v in inputs.items():
        inputs[k] = torch.tensor(v, dtype=torch.long)
    return inputs


# In[ ]:


class PPPMInferDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        super().__init__()
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        single_data = self.df.iloc[index]
        inputs = tokenize(self.tokenizer, single_data['text'], self.max_len)
        return inputs


# ## Model

# In[ ]:


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# In[ ]:


# class LabelSmoothLoss(nn.Module):
#     def __init__(self, smoothing=0.0, loss_func=nn.BCEWithLogitsLoss(reduction='sum')):
#         super(LabelSmoothLoss, self).__init__()
#         self.smoothing = smoothing
#         self.loss_func = loss_func

#     def forward(self, inputs, target):
#         # inputs为未经过激活的logits
#         #target为数值时才使用scatter_， 此处target为one-hot
#         '''
#         log_prob = F.log_softmax(inputs, dim=-1)
#         weight = inputs.new_ones(inputs.size()) * self.smoothing / (inputs.size(-1) - 1.)
#         weight.scatter_(-1, target.unsqueeze(-1), (1. - self.smoothing))
#         loss = (-weight * log_prob).sum(dim=-1).mean()'''
#         '''
#         log_prob = F.log_softmax(inputs, dim=-1)
#         # 由于将多标签看为多个二分类，因此不用除以类别数
#         weight = inputs.new_ones(inputs.size()) * self.smoothing
#         weight[target==1] = 1. - self.smoothing
#         loss = (-weight * log_prob).sum(dim=-1).mean()'''

#         weight = inputs.new_ones(inputs.size()) * self.smoothing
#         weight[target == 1] = 1. - self.smoothing
#         loss = self.loss_func(inputs, weight)
#         return loss


# In[ ]:


class FGM():
    """
    定义对抗训练方法FGM,对模型embedding参数进行扰动
    """
    def __init__(self, model, epsilon=0.25):
        self.model = model
        self.epsilon = epsilon
        self.backup = {}

    def attack(self, embed_name='word_embeddings'):
        """
        得到对抗样本
        :param emb_name:模型中embedding的参数名
        :return:
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad and embed_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)

                if norm != 0 and not torch.isnan(norm):
                    r_at = self.epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, embed_name='word_embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and embed_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


# In[ ]:


class PPPMModel(pl.LightningModule):
    def __init__(self, config, model_config_path=None, pretrained=False, weight_path=None):
        super().__init__()
        self.save_hyperparameters('config')

        if model_config_path:
            self.model_config = torch.load(model_config_path)
        else:
            self.model_config = AutoConfig.from_pretrained(config.model, output_hidden_states=True)
        if pretrained:
            self.model = AutoModel.from_pretrained(config.model, config=self.model_config)
        else:
            self.model = AutoModel.from_config(self.model_config)

        self.attention = nn.Sequential(
            nn.Linear(self.model_config.hidden_size, 512),
            nn.Tanh(),
            nn.Linear(512, 1),
            nn.Softmax(dim=1)
        )

        self.fc = nn.Linear(self.model_config.hidden_size, config.target_size)

        # TODO multi_dropout / layer norm
        self.dropout_0 = nn.Dropout(config.fc_dropout / 2.)
        self.dropout_1 = nn.Dropout(config.fc_dropout / 1.5)
        self.dropout_2 = nn.Dropout(config.fc_dropout)
        self.dropout_3 = nn.Dropout(config.fc_dropout * 1.5)
        self.dropout_4 = nn.Dropout(config.fc_dropout * 2.)

        self.__init_weight(self.fc)
        self.__set_metrics()

        if config.label_smooth:
            self.criterion = LabelSmoothLoss(smoothing=config.smoothing,
                                             loss_func=nn.BCEWithLogitsLoss(reduction="mean"))
        else:
            self.criterion = nn.BCEWithLogitsLoss(reduction="mean")

        if hasattr(self.hparams.config, 'fgm') and self.hparams.config.fgm:
            self.automatic_optimization = False
            self.fgm = FGM(self)

        if weight_path != None:
            weight = torch.load(weight_path, map_location='cpu')
            if 'state_dict' in weight.keys():
                weight = weight['state_dict']
            self.load_state_dict(weight)

    def __set_metrics(self):
        self.train_losses = AverageMeter()
        self.val_losses = AverageMeter()
        self.val_acc = AverageMeter()

        self.train_losses.reset()
        self.val_losses.reset()
        self.val_acc.reset()

    def __init_weight(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.model_config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.model_config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, inputs):
        outputs = self.model(**inputs)
        last_hidden_states, pooler_output = outputs[0], outputs[1]
        weights = self.attention(last_hidden_states)
        feature = torch.sum(weights * last_hidden_states, dim=1)
        #         output_0 = self.fc(self.dropout_0(feature))
        #         output_1 = self.fc(self.dropout_1(feature))
        output_2 = self.fc(self.dropout_2(feature))
        #         output_3 = self.fc(self.dropout_3(feature))
        #         output_4 = self.fc(self.dropout_4(feature))
        #         return (output_0 + output_1 + output_2 + output_3 + output_4) / 5
        return output_2

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        y_preds = self.forward(inputs)
        loss = self.criterion(y_preds.view(-1, 1), labels.view(-1, 1))
        # loss = torch.masked_select(loss, labels.view(-1, 1) != -1).mean()
        self.train_losses.update(loss.item(), len(labels))
        self.log('train/avg_loss', self.train_losses.avg)
        # 因为 optimizer 有 3 组参数，所有 get_last_lr() 会返回含有 3 个元素的列表
        en_lr = self.trainer.lr_scheduler_configs[0].scheduler.get_last_lr()[0]
        de_lr = self.trainer.lr_scheduler_configs[0].scheduler.get_last_lr()[-1]
        self.log('train/en_lr', en_lr, prog_bar=True)
        self.log('train/de_lr', de_lr, prog_bar=True)

        if (self.trainer.global_step) % self.hparams.config.print_freq == 0:
            # if (self.trainer.global_step + 1) % self.hparams.config.print_freq == 0:
            self.print('Global step:{global_step}.'
                       'Train Loss: {loss.val:.4f}(avg: {loss.avg:.4f}) '
                       'Encoder LR: {en_lr:.8f}, Decoder LR: {de_lr:.8f}'
                       .format(global_step=self.trainer.global_step,
                               loss=self.train_losses,
                               en_lr=en_lr,
                               de_lr=de_lr))
        # 如果没有FGM，在这里就可以返回loss
        # 为了使用FGM，这里要手动进行求导和优化器更新
        if self.hparams.config.fgm:
            # loss regularization， 但是不加效果要更好一些
            # if self.hparams.config.gradient_accumulation_steps > 1:
            #     loss = loss / self.hparams.config.gradient_accumulation_steps
            self.manual_backward(loss)
            torch.nn.utils.clip_grad_norm(self.parameters(), self.hparams.config.max_grad_norm)
            # 这里不能用 global_step ，否则因为关闭了自动优化，global_step 只能在 step 之后才会更新，会陷入死循环
            if (batch_idx + 1) % self.hparams.config.gradient_accumulation_steps == 0:
                # if (self.trainer.global_step + 1) % self.hparams.config.gradient_accumulation_steps == 0:
                self.fgm.attack()
                y_preds_adv = self.forward(inputs)
                loss_adv = self.criterion(y_preds_adv.view(-1, 1), labels.view(-1, 1))
                loss_adv = torch.masked_select(loss_adv, labels.view(-1, 1) != -1).mean()
                self.manual_backward(loss_adv)
                self.fgm.restore()

                opt = self.optimizers()
                opt.step()
                opt.zero_grad()
                sch = self.lr_schedulers()
                sch.step()

        return loss

    def training_epoch_end(self, outs):
        torch.cuda.empty_cache()

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        y_preds = self.forward(inputs)
        loss = self.criterion(y_preds.view(-1, 1), labels.view(-1, 1))
        # loss = torch.masked_select(loss, labels.view(-1, 1) != -1).mean()
        self.val_losses.update(loss.item(), len(labels))
        self.log('val/avg_loss', self.val_losses.avg)
        return y_preds.sigmoid().cpu().numpy(), labels.cpu()

    def validation_epoch_end(self, outs):
        # val_df = self.trainer.datamodule.val_df
        # val_labels = val_df['score'].values
        val_labels = np.concatenate([item[1] for item in outs])
        preds = np.concatenate([item[0] for item in outs]).squeeze(axis=-1)
        val_loss_avg = self.val_losses.avg
        #  ======================== scoring ============================
        score = get_score(val_labels, preds)
        self.log(f'val/loss_avg', val_loss_avg)
        self.log(f'val/score', score)
        self.print(f'Global step:{self.trainer.global_step}.\n Val loss avg: {val_loss_avg}, score: {score}')

        self.val_losses.reset()
        self.val_acc.reset()

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        inputs = batch
        y_preds = self.forward(inputs)
        return y_preds.sigmoid().cpu().numpy()

    def configure_optimizers(self):
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        encoder_lr = self.hparams.config.encoder_lr
        decoder_lr = self.hparams.config.decoder_lr
        num_cycles = self.hparams.config.num_cycles
        # end_lr = self.hparams.config.min_lr
        weight_decay = self.hparams.config.weight_decay
        eps = self.hparams.config.eps
        betas = self.hparams.config.betas
        optimizer_parameters = [
            {'params': [p for n, p in self.model.named_parameters()
                        if not any(nd in n for nd in no_decay)],
             'lr': encoder_lr, 'weight_decay': weight_decay,
             },
            {'params': [p for n, p in self.model.named_parameters()
                        if any(nd in n for nd in no_decay)],
             'lr': encoder_lr, 'weight_decay': 0.0,
             },
            {'params': [p for n, p in self.named_parameters() if "model" not in n],
             'lr': decoder_lr, 'weight_decay': 0.0,
             }
        ]
        optimizer = AdamW(optimizer_parameters,
                          lr=encoder_lr, eps=eps, betas=betas)

        if self.trainer.max_steps == None or self.trainer.max_epochs != None:
            # 注意，因为使用FGM需要关闭自动优化，传入 trainer 的 accumulate_grad_batches 是None
            # 因此这里计算不能使用 trainer 的参数，要使用 config 里的参数
            # max_steps = (
            #         len(self.trainer.datamodule.train_dataloader()) * self.trainer.max_epochs
            #         // self.trainer.accumulate_grad_batches
            # )
            max_steps = (
                    len(self.trainer.datamodule.train_dataloader()) * self.trainer.max_epochs
                    // self.hparams.config.gradient_accumulation_steps
            )
        else:
            max_steps = self.trainer.max_steps

        warmup_steps = self.hparams.config.warmup_steps
        if isinstance(warmup_steps, float):
            warmup_steps = int(warmup_steps * max_steps)

        print(f'====== Max steps: {max_steps},\t Warm up steps: {warmup_steps} =========')

        if self.hparams.config.scheduler == 'linear':
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=warmup_steps, num_training_steps=max_steps,
            )
        elif self.hparams.config.scheduler == 'cosine':
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, num_warmup_steps=warmup_steps, num_training_steps=max_steps,
                num_cycles=num_cycles
            )
        else:
            scheduler = None
        sched = {
            'scheduler': scheduler, 'interval': 'step'
        }
        return ([optimizer], [sched])


# ## Infer

# In[ ]:


pl.seed_everything(CFG.seed)
dm = PPPMDataModule(CFG, prepare_train=False)
dm.prepare_data()


# In[ ]:


weight_paths = []
for p in CFG.weight_path:
    weight_paths.extend(list(Path(p).rglob('*.ckpt')))
# 手动输入权重，如果没有，则用验证的得分作为权重
if CFG.weight_scores != []:
    cv_score = CFG.weight_scores
else:
    cv_score = [float(re.search('score([\d.]*)', weight_path.stem).group(1)) for weight_path in weight_paths]
cv_score = torch.tensor(cv_score)

weights = nn.functional.softmax(cv_score / CFG.t, dim=0).float().numpy()
weight_paths


# In[ ]:


cv_score, weights


# In[ ]:


predictions = []
for weight_path in weight_paths:
    weight_name = weight_path.name
    print(f"Using weight from {weight_name}.")

    model = PPPMModel(CFG, model_config_path=None, pretrained=False, weight_path=weight_path)
    trainer = pl.Trainer(
        gpus=[0],
        default_root_dir=CFG.output_dir,
    )
    prediction = trainer.predict(model, datamodule=dm)
    prediction = np.concatenate([batch_pred for batch_pred in prediction])
    predictions.append(prediction)

    del model, trainer, prediction
    gc.collect()
    torch.cuda.empty_cache()


# In[ ]:


predictions = np.asarray(predictions)
predictions = np.sum([w * p for w, p in zip(weights, predictions)], axis=0)

dm.submission['score'] = predictions
dm.submission[['id', 'score']].to_csv('submission.csv', index=False)


# In[ ]:


dm.submission

