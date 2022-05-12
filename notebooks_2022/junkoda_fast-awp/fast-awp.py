#!/usr/bin/env python
# coding: utf-8

# # Fast AWP with small overhead
# 
# The *Adversarial Weight Perturbation (AWP)* was used in the [first-place solution of the Feedback Prize](https://www.kaggle.com/competitions/feedback-prize-2021/discussion/313177) (@wht1996) and it is showen to be  [effective in this competition as well](https://www.kaggle.com/competitions/nbme-score-clinical-patient-notes/discussion/315707) (@hengck23). The perturbation, however, requires two gradient computations per optimization step, and therefore takes twice more computation time (correct me if I am wrong). This implementation removes the overhead by approximating the gradient with the running mean stored in the Adam optimizer. 
# 
# Since the gradient given to the optimizer is perturbed by
# AWP, the AWP in this notebook uses "wrong" gradients, and may not work as intended. Still, the factor of two in time is large and seems to have some good effects; maybe AWP works with inaccurate gradients. Let me know in the comment if you see this AWP underperforms compared to the original. My PyTorch coding is pretty random and comments are welcome on that aspect, too, e.g., detach and copy are not appropriate. Maybe my code doing something completely random.
# 
# The removal of additional gradient calculation also allows gradient accumulation in the usual way. The overhead in computation time is little, but overhead in GPU RAM exists for a copy of model weights (0.5 - 1 GB).
# 
# 
# ## Parameters
# 
# Two parameters control the amount of perturbations in units of fractional changes in the weights; 0.01 means 1% perturbation in weights (model parameters).
# 
# ```
# adv_lr or γ: fractional change in weight along the direction of gradient
# adv_eps or ε: the change in weight is limited to this fraction
# ```
# 
# Two parameters are similar to learning rate and max norm in gradient clipping, but in units of fraction of weights.
# 
# The AWP gradients are evaluated at perturbed location:
# 
# $$ w \mapsto w + \delta w = w + \gamma \frac{\nabla}{\lVert{\nabla} \rVert} \lVert w \rVert, $$
# 
# but change in each component is limitted to,
# 
# $$ |\delta w_i| \le \epsilon |w_i|. $$
# 
# where `∇ = param.grad` is the gradient for the weight `w = param.data`.
# 
# 
# You only need to see `class AWP` and use it in your training loop; you do not need to read other parts of my code.
# 
# 
# 
# 
# ## Reference
# 
# This AWP modifies the 1st-place code in the Feedback Prize solution by @wht1996:
# 
# https://www.kaggle.com/code/wht1996/feedback-nn-train 
# 
# 
# The original paper is,
# 
# Wu, Xia, and  Wang (2020), Adversarial Weight Perturbation Helps Robust Generalization
# https://arxiv.org/abs/2004.05884
# 
# Authors' implementation: https://github.com/csdongxian/AWP
# 
# The model and many codes for training are borrowed from the Nakama baseline >ω</ Thanks!:
# 
# https://www.kaggle.com/code/yasufuminakama/nbme-deberta-base-baseline-train/notebook
# 
# And many thanks to hengck23 for sharing the usefulness of AWP and the working parameters:
# 
# https://www.kaggle.com/competitions/nbme-score-clinical-patient-notes/discussion/315707

# In[ ]:


import numpy as np
import pandas as pd
import pickle
import os
import ast
import time
import yaml
import random
import argparse
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedGroupKFold

import transformers
from transformers import AutoTokenizer, AutoConfig, AutoModel
from transformers import get_cosine_schedule_with_warmup

os.environ["TOKENIZERS_PARALLELISM"] = "false"
transformers.logging.set_verbosity_error()

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
device = torch.device('cuda')


# # Class AWP
# 
# Modified from https://www.kaggle.com/code/wht1996/feedback-nn-train (wht1996)

# In[ ]:


class AWP:
    def __init__(self, model, optimizer, *, adv_param='weight',
                 adv_lr=0.001, adv_eps=0.001):
        self.model = model
        self.optimizer = optimizer
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.backup = {}

    def perturb(self, input_ids, attention_mask, y, criterion):
        """
        Perturb model parameters for AWP gradient
        Call before loss and loss.backward()
        """
        self._save()  # save model parameters
        self._attack_step()  # perturb weights

    def _attack_step(self):
        e = 1e-6
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                grad = self.optimizer.state[param]['exp_avg']
                norm_grad = torch.norm(grad)
                norm_data = torch.norm(param.detach())

                if norm_grad != 0 and not torch.isnan(norm_grad):
                    # Set lower and upper limit in change
                    limit_eps = self.adv_eps * param.detach().abs()
                    param_min = param.data - limit_eps
                    param_max = param.data + limit_eps

                    # Perturb along gradient
                    # w += (adv_lr * |w| / |grad|) * grad
                    param.data.add_(grad, alpha=(self.adv_lr * (norm_data + e) / (norm_grad + e)))

                    # Apply the limit to the change
                    param.data.clamp_(param_min, param_max)

    def _save(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                if name not in self.backup:
                    self.backup[name] = param.clone().detach()
                else:
                    self.backup[name].copy_(param.data)

    def restore(self):
        """
        Restore model parameter to correct position; AWP do not perturbe weights, it perturb gradients
        Call after loss.backward(), before optimizer.step()
        """
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])


# # Changes in AWP

# - grad is using Adam exponentially averaged gradient;
#   * this saves computing gradient twice and **speeds up by factor of 2**;
#   * but this grad is pertubed by AWP and not the true gradient.
# - adv_step is removed; since grads are not computed, there is no reason to do multiple steps.
# - since adv_step=1, weight eps ranges do not have to be saved,
#   * which reduce GPU RAM from 3 extra copies of wegiths to 1 copy.
# 
# 
# ## Original code by wht1996
# 
# ```python
# class AWP:
#     def __init__(
#         self,
#         model,
#         optimizer,
#         adv_param="weight",
#         adv_lr=1,
#         adv_eps=0.2,
#         start_epoch=0,
#         adv_step=1,
#         scaler=None
#     ):
#         self.model = model
#         self.optimizer = optimizer
#         self.adv_param = adv_param
#         self.adv_lr = adv_lr
#         self.adv_eps = adv_eps
#         self.start_epoch = start_epoch
#         self.adv_step = adv_step
#         self.backup = {}
#         self.backup_eps = {}
#         self.scaler = scaler
#   
#     def attack_backward(self, x, y, attention_mask,epoch):
#         if (self.adv_lr == 0) or (epoch < self.start_epoch):
#             return None
# 
#         self._save() 
#         for i in range(self.adv_step):
#             self._attack_step() 
#             with torch.cuda.amp.autocast():
#                 adv_loss, tr_logits = self.model(input_ids=x, attention_mask=attention_mask, labels=y)
#                 adv_loss = adv_loss.mean()
#             self.optimizer.zero_grad()
#             self.scaler.scale(adv_loss).backward()
#             
#         self._restore()
# 
#     def _attack_step(self):
#         e = 1e-6
#         for name, param in self.model.named_parameters():
#             if param.requires_grad and param.grad is not None and self.adv_param in name:
#                 norm1 = torch.norm(param.grad)
#                 norm2 = torch.norm(param.data.detach())
#                 if norm1 != 0 and not torch.isnan(norm1):
#                     r_at = self.adv_lr * param.grad / (norm1 + e) * (norm2 + e)
#                     param.data.add_(r_at)
#                     param.data = torch.min(
#                         torch.max(param.data, self.backup_eps[name][0]), self.backup_eps[name][1]
#                     )
#                 # param.data.clamp_(*self.backup_eps[name])
# 
#     def _save(self):
#         for name, param in self.model.named_parameters():
#             if param.requires_grad and param.grad is not None and self.adv_param in name:
#                 if name not in self.backup:
#                     self.backup[name] = param.data.clone()
#                     grad_eps = self.adv_eps * param.abs().detach()
#                     self.backup_eps[name] = (
#                         self.backup[name] - grad_eps,
#                         self.backup[name] + grad_eps,
#                     )
# 
#     def _restore(self,):
#         for name, param in self.model.named_parameters():
#             if name in self.backup:
#                 param.data = self.backup[name]
#         self.backup = {}
#         self.backup_eps = {}
# ```

# # Data

# In[ ]:


def nakama_fix_annotations(train):
    # Fix incorrect annotations (from Nakama baseline)
    train.loc[338, 'annotation'] = "['father heart attack']"
    train.loc[338, 'location'] = "['764 783']"

    train.loc[621, 'annotation'] = "['for the last 2-3 months']"
    train.loc[621, 'location'] = "['77 100']"

    train.loc[655, 'annotation'] = "['no heat intolerance'], ['no cold intolerance']"
    train.loc[655, 'location'] = "['285 292;301 312', '285 287;296 312']"

    train.loc[1262, 'annotation'] = "['mother thyroid problem']"
    train.loc[1262, 'location'] = "['551 557;565 580']"

    train.loc[1265, 'annotation'] = "[\"felt like he was going to 'pass out'\"]"
    train.loc[1265, 'location'] = "['131 135;181 212']"

    train.loc[1396, 'annotation'] = "['stool , with no blood']"
    train.loc[1396, 'location'] = "['259 280']"

    train.loc[1591, 'annotation'] = "['diarrhoe non blooody']"
    train.loc[1591, 'location'] = "['176 184;201 212']"

    train.loc[1615, 'annotation'] = "['diarrhea for last 2-3 days']"
    train.loc[1615, 'location'] = "['249 257;271 288']"

    train.loc[1664, 'annotation'] = "['no vaginal discharge']"
    train.loc[1664, 'location'] = "['822 824;907 924']"

    train.loc[1714, 'annotation'] = "['started about 8-10 hours ago']"
    train.loc[1714, 'location'] = "['101 129']"

    train.loc[1929, 'annotation'] = "['no blood in the stool']"
    train.loc[1929, 'location'] = "['531 539;549 561']"

    train.loc[2134, 'annotation'] = "['last sexually active 9 months ago']"
    train.loc[2134, 'location'] = "['540 560;581 593']"

    train.loc[2191, 'annotation'] = "['right lower quadrant pain']"
    train.loc[2191, 'location'] = "['32 57']"

    train.loc[2553, 'annotation'] = "['diarrhoea no blood']"
    train.loc[2553, 'location'] = "['308 317;376 384']"

    train.loc[3124, 'annotation'] = "['sweating']"
    train.loc[3124, 'location'] = "['549 557']"

    train.loc[3858, 'annotation'] = "['previously as regular', 'previously eveyr 28-29 days', 'previously lasting 5 days'], 'previously regular flow']"
    train.loc[3858, 'location'] = "['102 123', '102 112;125 141', '102 112;143 157', '102 112;159 171']"

    train.loc[4373, 'annotation'] = "['for 2 months']"
    train.loc[4373, 'location'] = "['33 45']"

    train.loc[4763, 'annotation'] = "['35 year old']"
    train.loc[4763, 'location'] = "['5 16']"

    train.loc[4782, 'annotation'] = "['darker brown stools']"
    train.loc[4782, 'location'] = "['175 194']"

    train.loc[4908, 'annotation'] = "['uncle with peptic ulcer']"
    train.loc[4908, 'location'] = "['700 723']"

    train.loc[6016, 'annotation'] = "['difficulty falling asleep']"
    train.loc[6016, 'location'] = "['225 250']"

    train.loc[6192, 'annotation'] = "['helps to take care of aging mother and in-laws']"
    train.loc[6192, 'location'] = "['197 218;236 260']"

    train.loc[6380, 'annotation'] = "['No hair changes', 'No skin changes', 'No GI changes', 'No palpitations', 'No excessive sweating']"
    train.loc[6380, 'location'] = "['480 482;507 519', '480 482;499 503;512 519', '480 482;521 531', '480 482;533 545', '480 482;564 582']"

    train.loc[6562, 'annotation'] = "['stressed due to taking care of her mother', 'stressed due to taking care of husbands parents']"
    train.loc[6562, 'location'] = "['290 320;327 337', '290 320;342 358']"

    train.loc[6862, 'annotation'] = "['stressor taking care of many sick family members']"
    train.loc[6862, 'location'] = "['288 296;324 363']"

    train.loc[7022, 'annotation'] = "['heart started racing and felt numbness for the 1st time in her finger tips']"
    train.loc[7022, 'location'] = "['108 182']"

    train.loc[7422, 'annotation'] = "['first started 5 yrs']"
    train.loc[7422, 'location'] = "['102 121']"

    train.loc[8876, 'annotation'] = "['No shortness of breath']"
    train.loc[8876, 'location'] = "['481 483;533 552']"

    train.loc[9027, 'annotation'] = "['recent URI', 'nasal stuffines, rhinorrhea, for 3-4 days']"
    train.loc[9027, 'location'] = "['92 102', '123 164']"

    train.loc[9938, 'annotation'] = "['irregularity with her cycles', 'heavier bleeding', 'changes her pad every couple hours']"
    train.loc[9938, 'location'] = "['89 117', '122 138', '368 402']"

    train.loc[9973, 'annotation'] = "['gaining 10-15 lbs']"
    train.loc[9973, 'location'] = "['344 361']"

    train.loc[10513, 'annotation'] = "['weight gain', 'gain of 10-16lbs']"
    train.loc[10513, 'location'] = "['600 611', '607 623']"

    train.loc[11551, 'annotation'] = "['seeing her son knows are not real']"
    train.loc[11551, 'location'] = "['386 400;443 461']"

    train.loc[11677, 'annotation'] = "['saw him once in the kitchen after he died']"
    train.loc[11677, 'location'] = "['160 201']"

    train.loc[12124, 'annotation'] = "['tried Ambien but it didnt work']"
    train.loc[12124, 'location'] = "['325 337;349 366']"

    train.loc[12279, 'annotation'] = "['heard what she described as a party later than evening these things did not actually happen']"
    train.loc[12279, 'location'] = "['405 459;488 524']"

    train.loc[12289, 'annotation'] = "['experienced seeing her son at the kitchen table these things did not actually happen']"
    train.loc[12289, 'location'] = "['353 400;488 524']"

    train.loc[13238, 'annotation'] = "['SCRACHY THROAT', 'RUNNY NOSE']"
    train.loc[13238, 'location'] = "['293 307', '321 331']"

    train.loc[13297, 'annotation'] = "['without improvement when taking tylenol', 'without improvement when taking ibuprofen']"
    train.loc[13297, 'location'] = "['182 221', '182 213;225 234']"

    train.loc[13299, 'annotation'] = "['yesterday', 'yesterday']"
    train.loc[13299, 'location'] = "['79 88', '409 418']"

    train.loc[13845, 'annotation'] = "['headache global', 'headache throughout her head']"
    train.loc[13845, 'location'] = "['86 94;230 236', '86 94;237 256']"

    train.loc[14083, 'annotation'] = "['headache generalized in her head']"
    train.loc[14083, 'location'] = "['56 64;156 179']"


# In[ ]:


def _replace_feature_text(text):
    text = text.replace('I-year', '1-year')
    text = text.replace('-OR-', ' or ')
    text = text.replace('-', ' ')

    return text

def get_segments(locations):
    """
    Parse location list to sorted list of segments

    Args:
      location (str): ['85 99', '126 138', '126 131;143 151']

    Returns: list[tuple]
      List of (begin, end)
    """
    segs = []

    assert isinstance(locations, str)

    locations = ast.literal_eval(locations)  # str -> list[str]
    for loc in locations:
        for span in loc.split(';'):
            segs.append(tuple(map(int, span.split(' '))))

    segs.sort(key=lambda pair: pair[0])

    return segs


def create_label(segments, offset_mapping):
    n = len(offset_mapping)
    y = np.zeros(n, dtype=np.float32)

    if not segments:
        return y

    iseg = 0
    seg = segments[iseg]
    nseg = len(segments)

    k = 0
    while k < n:
        begin, end = offset_mapping[k]
        if end <= seg[0]:  # token is left of seg
            k += 1
            continue
        elif begin < seg[1]:  # token overlaps with seg
            y[k] = 1
            k += 1
        else:  # begin passed seg: seg[1] <= begin
            iseg += 1
            if iseg < nseg:
                seg = segments[iseg]
            else:
                break  # All segments processed

    return y


def exclude_feature_label(input_ids, y, sep):
    """
    Set y = -1 for feature_text; compute loss only with note texts excluding feature texts

    Result:
      y modified to -1 after first [SEP]
    """
    n = len(input_ids)
    assert n >= 2 and input_ids[-1] == sep
    y[-1] = -1

    for k in range(n - 2, -1, -1):
        if input_ids[k] == sep:
            break
        else:
            y[k] = -1


def create_character_indices(segments):
    """
    Character indices

    Args:
      segments (list[tuple]): List of (begin, end)
    """
    s = set()

    for begin, end in segments:
        s.update(range(begin, end))

    return sorted(list(s))


def create_data(train, tokenizer, *, max_length=1024, pbar=False):
    """
    Create input_ids and label array y

    Args:
      train (pd.DataFrame)
      tokenizer (str or tokenizer): path to tokenizer dir if str

    Returns: list[dict]
      input_ids (np.array[int])
      n (int): number of tokens
      y (np.array[float32]): binary annotation
    """
    sep = tokenizer.sep_token_id
    nsep = 2

    annotated = 'location' in train.columns

    data = []
    for i, r in tqdm(train.iterrows(), disable=(not pbar), total=len(train)):
        text = r.pn_history
        feature_text = r.feature_text

        o = tokenizer(text, feature_text,
                      add_special_tokens=True, max_length=max_length,
                      truncation=True,
                      return_offsets_mapping=True)

        # Input ids
        input_ids = o['input_ids']
        n = len(input_ids)

        input_ids = np.array(o['input_ids'], dtype=np.int32)
        assert np.sum(input_ids == sep) == nsep  # Two tokens seperated by [SEP], <s/><s/> for roberta

        # Attention mask
        attention_mask = np.array(o['attention_mask'])
        assert np.all(attention_mask == 1)

        # Label
        if annotated:
            segs = get_segments(r['location'])
            y = create_label(segs, o['offset_mapping'])

            exclude_feature_label(input_ids, y, sep)

            # Character label
            label = create_character_indices(segs)
        else:
            y = None
            label = None

        d = {'id': r['id'],
             'input_ids': input_ids,
             'text': text,
             'label': label,
             'y': y,
             'n': n,
             'offset_mapping': o['offset_mapping']}
        data.append(d)

    return np.array(data)


class Dataset(torch.utils.data.Dataset):
    """
    Dataset(data)
      data (np.array or list-like): input_ids and y
    """
    def __init__(self, data, *, max_length=512):
        self.data = data
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        d = self.data[i]
        n = min(d['n'], self.max_length)

        input_ids = np.zeros(self.max_length, dtype=int)
        input_ids[:n] = d['input_ids']

        attention_mask = np.zeros(self.max_length, dtype=int)
        attention_mask[:n] = 1

        y = np.full(self.max_length, -1, dtype=np.float32)
        y[:n] = d['y']

        return {'input_ids': input_ids,
                'attention_mask': attention_mask,
                'y': y, 'n': n}


# # Model

# In[ ]:


class Model(nn.Module):
    def __init__(self, model_dir, *, dropout=0.2, pretrained=True):
        super().__init__()

        # Transformer
        config = AutoConfig.from_pretrained(model_dir, add_pooling_layer=False)
        if pretrained:
            self.transformer = AutoModel.from_pretrained(model_dir, config=config)
        else:
            self.transformer = AutoModel.from_config(config)

        self.fc_dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(config.hidden_size, 1)

        self._init_weights(self.fc, config)

    def _init_weights(self, module, config):
        module.weight.data.normal_(mean=0.0, std=config.initializer_range)
        if module.bias is not None:
            module.bias.data.zero_()

    def forward(self, input_ids, attention_mask):
        out = self.transformer(input_ids, attention_mask)
        x = out['last_hidden_state']  # batch_size x max_length (512) x 768

        x = self.fc_dropout(x)
        x = self.fc(x)

        return x


# # Train and evaluate

# In[ ]:


def create_prediction(y_pred, d, *, th=0.5):
    """
    Create character-level prediction

    Args:
      pred: pred['y_pred'] token-level prediction
      d: d['offset_mapping']
    """
    text = d['text']
    offset_mapping = d['offset_mapping']

    # Map token-level prob to character-level prob
    y_prob = np.zeros(len(text))  # character-wise probabilities
    i = 0
    for p, (begin, end) in zip(y_pred, offset_mapping):
        if i > 0 and begin == 0 and end == 0:
            break  # This is end of patient note [sep]

        y_prob[i:end] = p    # Set space before begin with p too (deberta style)
        i = end

    i_begin = i_last = None
    li = []
    for i, (x, p) in enumerate(zip(text, y_prob)):
        if p >= th:
            if i_begin is None and x != ' ':  # Do not include first space in span
                i_begin = i_last = i
                li.append(i)
            elif i_begin is not None:         # Positive token is continuing
                assert i_last + 1 == i
                i_last = i
                li.append(i)
        else:
            i_begin = i_last = None           # Negative

    d = {'text': text,
         'y_prob': y_prob,
         'indices': li}

    return d


def compute_score(preds, data):
    """
    TP, FN, FP are collected globally and
    f1 score is computed at the end
    """
    assert len(preds) == len(data)

    tp = 0
    denom = 0
    for pred, d in zip(preds, data):
        c = create_prediction(pred['y_pred'], d)
        x = set(d['label'])
        y = set(c['indices'])
        tp += len(x.intersection(y))
        denom += len(x) + len(y)

    return 2 * tp / denom


def evaluate(model, loader, criterion):
    tb = time.time()
    was_training = model.training
    model.eval()

    n_sum = 0
    loss_sum = 0.0
    preds = []
    for d in loader:
        input_ids = d['input_ids'].to(device)
        attention_mask = d['attention_mask'].to(device)
        y = d['y'].to(device)
        n = y.size(0)

        with torch.no_grad():
            y_pred = model(input_ids, attention_mask)

        loss = criterion(y_pred.view(-1), y.view(-1))
        loss = torch.masked_select(loss, y.view(-1) != -1).mean()

        n_sum += n
        loss_sum += n * loss.item()

        y_pred = y_pred.sigmoid().cpu().numpy()
        y = d['y'].numpy()

        for k, m in enumerate(d['n']):
            preds.append({'y_pred': y_pred[k, :m].copy(),
                          'y': y[k, :m].copy()})

        del loss, y_pred, input_ids, attention_mask, y

    model.train(was_training)

    val = {'time': time.time() - tb,
           'loss': loss_sum / n_sum,
           'preds': preds}

    return val


def get_optimizer_params(model, encoder_lr, decoder_lr, weight_decay=0.0):
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_parameters = [
        {'params': [p for n, p in model.transformer.named_parameters() if not any(nd in n for nd in no_decay)],
         'lr': encoder_lr, 'weight_decay': weight_decay},
        {'params': [p for n, p in model.transformer.named_parameters() if any(nd in n for nd in no_decay)],
         'lr': encoder_lr, 'weight_decay': 0.0},
        {'params': [p for n, p in model.named_parameters() if 'transformer' not in n],
         'lr': decoder_lr, 'weight_decay': 0.0}
    ]

    return optimizer_parameters


# In[ ]:


debug = False
tb_train = time.time()

# Output directory
name = 'deberta_base_awp'
odir = name

if not os.path.exists(odir):
    os.mkdir(odir)

# Data
di = '/kaggle/input/nbme-score-clinical-patient-notes/'
train = pd.read_csv(di + 'train.csv')    
features = pd.read_csv(di + 'features.csv')
patient_notes = pd.read_csv(di + 'patient_notes.csv')

features['feature_text'] = features['feature_text'].apply(_replace_feature_text)

# Attach text `pn_history` to train annotations
train = train.merge(features, on=['feature_num', 'case_num'], how='left')
train = train.merge(patient_notes, on=['pn_num', 'case_num'], how='left')

nakama_fix_annotations(train)

# Tokenizer
transformer_path = 'microsoft/deberta-base'
transformer_name = transformer_path.split('/')[-1]   # deberta-base

tokenizer = AutoTokenizer.from_pretrained(transformer_path)
tokenizer.save_pretrained(odir + '/tokenizer')

config = AutoConfig.from_pretrained(transformer_path, add_pooling_layer=False)
config.save_pretrained(odir)

# Tokenize data
data = create_data(train, tokenizer)

seed_everything(42)

# Kfold
nfold = 5
folds = [0, 1, 2, 3, 4]

kfold = StratifiedGroupKFold(nfold, shuffle=True, random_state=42)
groups = train['pn_num'].values
cases = train['case_num'].values

# Parameters
epochs = 5
batch_size = 6
batch_size_val = 16

apex = True
max_grad_norm = 1000
optimizer_batch_size = 6  # batch size for optimization step
gradient_accumulation_steps = max(optimizer_batch_size // batch_size, 1)
print('gradient accumulation', gradient_accumulation_steps)

eval_per_epoch = 4

# Training and evaluation
for ifold, (idx_train, idx_val) in enumerate(kfold.split(data, cases, groups=groups)):
    if ifold not in folds:
        continue

    tb = time.time()
    print('Fold %d' % ifold)
    print('Epoch      Loss       lr  time')

    # Train - validation split
    data_train = data[idx_train]
    data_val = data[idx_val]

    if debug:
        data_train = data_train[:64]
        data_val = data_val[:16]
        epochs = 2
        folds = [0]

    loader_train = DataLoader(Dataset(data_train),
                              batch_size=batch_size, shuffle=True, drop_last=True)
    loader_val = DataLoader(Dataset(data_val), batch_size=batch_size_val)

    # Model
    model = Model(transformer_path)
    model.train()
    model.to(device)

    # Optimizer
    lr = 2e-5
    eps = 1e-6
    betas = (0.9, 0.999)
    weight_decay = 0.01
    optimizer_parameters = get_optimizer_params(model,
                                                encoder_lr=lr,
                                                decoder_lr=lr,
                                                weight_decay=weight_decay)

    optimizer = AdamW(optimizer_parameters, lr=lr, eps=eps, betas=betas)

    # One `step` is one optimizer step, `gradient_accumulation` mini batches
    steps_per_epoch = len(loader_train) // gradient_accumulation_steps
    eval_steps = [int(i * steps_per_epoch / eval_per_epoch) for i in range(1, eval_per_epoch)] +                  [steps_per_epoch]

    # Scheduler
    num_train_steps = steps_per_epoch * epochs
    num_warmup_steps = 0
    num_cycles = 0.5
    scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                num_warmup_steps=num_warmup_steps,
                                                num_training_steps=num_train_steps,
                                                num_cycles=num_cycles)

    criterion = nn.BCEWithLogitsLoss(reduction='none')
    best_score = 0
    best_epoch = None

    print('Enable AWP')
    awp = AWP(model, optimizer, adv_lr=0.001, adv_eps=0.001)
    awp_start = 1.0

    for epoch in range(epochs):
        optimizer.zero_grad()

        scaler = torch.cuda.amp.GradScaler(enabled=apex)

        step = 0
        n_sum = 0
        loss_sum = 0
        for ibatch, d in enumerate(loader_train):
            input_ids = d['input_ids'].to(device)
            attention_mask = d['attention_mask'].to(device)
            y = d['y'].to(device)
            n = y.size(0)

            if epoch >= awp_start:
                awp.perturb(input_ids, attention_mask, y, criterion)

            with torch.cuda.amp.autocast(enabled=apex):
                y_pred = model(input_ids, attention_mask)

            loss = criterion(y_pred.view(-1), y.view(-1))
            loss = torch.masked_select(loss, y.view(-1) != -1).mean()
            loss_sum += n * loss.item()
            n_sum += n

            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps

            scaler.scale(loss).backward()
            awp.restore()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            if (ibatch + 1) % gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if scheduler is not None:
                    scheduler.step()
                step += 1

            del loss, y_pred, input_ids, attention_mask, y

            ep = epoch + step / steps_per_epoch

            # Validation
            if step in eval_steps and (ibatch + 1) % gradient_accumulation_steps == 0:
                val = evaluate(model, loader_val, criterion)
                score = compute_score(val['preds'], data_val)

                ep = epoch + step / steps_per_epoch
                loss_train = loss_sum / n_sum
                lr1 = optimizer.param_groups[0]['lr']
                dt = (time.time() - tb) / 60

                print('Epoch %.2f %.6f %.6f | %.4f %.2e %6.2f %.2f min' %
                      (ep, loss_train, val['loss'], score,
                       lr1, dt, val['time'] / 60))

                n_sum = loss_sum = 0

                if step == steps_per_epoch:
                    break  # drop incomplete gradient accumulation

        # Save model
        if score >= best_score:
            best_score = score
            best_epoch = epoch + 1

            model_filename = 'model%d_best.pytorch' % ifold
            torch.save(model.state_dict(), model_filename)
            #print(model_filename, 'written')

    # Save final model
    if best_epoch == epochs:
        print('Last model is same as best')
    #else:
        #model_filename = '%s/model%d.pytorch' % (odir, ifold)
        #model.to('cpu')
        #model.eval()
        #torch.save(model.state_dict(), model_filename)
        #print(model_filename, 'written')

    del model
    del awp

dt = (time.time() - tb_train) / 3600
print('Train done %.2f hr' % dt)


# In[ ]:


get_ipython().system(' ls')
get_ipython().system(' ls deberta_base_awp')

