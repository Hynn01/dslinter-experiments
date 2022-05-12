#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[ ]:





# # All the important imports

# In[ ]:


import numpy as np
import pandas as pd

import transformers
import tokenizers
import torch
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm.autonotebook import tqdm

from ast import literal_eval
import time


# In[ ]:


torch.cuda.empty_cache()

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 


# In[ ]:


BASE_PATH = "../input/nbme-score-clinical-patient-notes/"
features_df = pd.read_csv(BASE_PATH + "features.csv")
patient_notes_df = pd.read_csv(BASE_PATH + "patient_notes.csv")
train_df = pd.read_csv(BASE_PATH + "train.csv")
test_df = pd.read_csv(BASE_PATH + "test.csv")
submission_df = pd.read_csv(BASE_PATH + "sample_submission.csv")


# In[ ]:


test_df.shape


# In[ ]:


features_df.tail()


# In[ ]:


features_df.nunique()


# In[ ]:


patient_notes_df.head()


# In[ ]:


patient_notes_df.nunique()


# In[ ]:


train_df


# In[ ]:


train_df.nunique()


# In[ ]:


df = pd.merge(train_df, features_df, on=['feature_num','case_num'], how='inner')
df =pd.merge(df, patient_notes_df, on=['pn_num','case_num'], how='inner')
df.head()


# In[ ]:


df.nunique()


# In[ ]:


df['feature_text'].value_counts()


# ##### The 'annotation' is picked form the 'pn_history' and the text location is mentiond in 'location' column. Location:Character spans indicating the location(s) of the feature within the note.
# 
# 

# In[ ]:


df['pn_history'][1]


# In[ ]:


print(df['pn_history'][1][668 : 693])
print(df['pn_history'][1][203 : 217])
print(df['pn_history'][1][696 : 724])


# In[ ]:


df["annotation"] = [literal_eval(x) for x in df["annotation"]]
df["location"] = [literal_eval(x) for x in df["location"]]
df


# In[ ]:


frames = []
df_split = np.array_split(df, 5)
for split in range(0, 5):
    df_split[split]['kfold'] = split
    frames.append(df_split[split])
dfx = pd.concat(frames)
dfx


# In[ ]:


max_len = df['pn_history'].map(lambda x: len(x)).max()
max_len


# # Configrations | Hyperparameters

# In[ ]:


class config:
    MAX_LEN = 416
    TRAIN_BATCH_SIZE = 8
    VALID_BATCH_SIZE = 8
    EPOCHS = 5
    BERT_PATH = "../input/bert-base-uncased/" 
    MODEL_PATH = "model.bin"
    TOKENIZER = tokenizers.BertWordPieceTokenizer(f"{BERT_PATH}/vocab.txt" ,lowercase = True)
    DROPOUT = 0.2
    MAX_GRAD_NORM = 1.0
    LEARNING_RATE = 1e-5


# # Data Processing

# In[ ]:


first = df.loc[3]
example = {
    "feature_text": first.feature_text,
    "pn_history": first.pn_history,
    "location": first.location,
    "annotation": first.annotation
}
for key in example.keys():
    print(key)
    print(example[key])
    print("=" * 100)


# In[ ]:


def loc_list_to_ints(loc_list):
    to_return = []
    for loc_str in loc_list:
        loc_strs = loc_str.split(";")
        for loc in loc_strs:
            start, end = loc.split()
            to_return.append((int(start), int(end)))
    return to_return

example_loc_ints = loc_list_to_ints(example["location"])


# In[ ]:


def process_data_tokenize(pn_history, feature_text, annotation, location, tokenizer, max_len):    ##X , Y, selected_text  
    
    location_list = loc_list_to_ints(location)        
    char_targets = [0] * len(pn_history) #creating empty list(all zeros) of character;it will be made 1 if annotation in text   
    
    for loc,anno in zip(location_list,annotation): 
        len_st = loc[1] - loc[0]

        idx0 = None
        idx1 = None
        for ind in (i for i, e in enumerate(pn_history) if (e == anno[0] and i == loc[0])):
            if pn_history[ind: ind+len_st] == anno:

                idx0 = ind
                idx1 = ind + len_st - 1
                if idx0 != None and idx1 != None:
                    for ct in range(idx0, idx1 + 1):
                        char_targets[ct] = 1  #replacing zeros with 1 if that part of the text is selected text
        
                break
      
    tokenized_input = tokenizer.encode(feature_text,pn_history)
    
        
    input_ids = tokenized_input.ids
    mask = tokenized_input.attention_mask
    token_type_ids = tokenized_input.type_ids
    offsets = tokenized_input.offsets
    
    target_idx = []
    for j, (offset1, offset2) in enumerate(offsets):
        if sum(char_targets[offset1: offset2]) > 0:
            target_idx.append(j)
            
    #padding
    padding_length = max_len - len(input_ids)
    if padding_length > 0:
        input_ids = input_ids + ([0] * padding_length)
        mask = mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length)
        offsets = offsets + ([(0, 0)] * padding_length)
       
    #creating label
    ignore_idxes = np.where(np.array(token_type_ids) != 1)[0]

    label = np.zeros(len(offsets))
    label[ignore_idxes] = -1
    label[target_idx] = 1

    
    return {
    'ids': input_ids,
    'mask': mask,
    'token_type_ids': token_type_ids,
    'labels': label,
    'offsets': offsets
}


# In[ ]:


output = process_data_tokenize(example["pn_history"],example["feature_text"],example["annotation"],example["location"],config.TOKENIZER,config.MAX_LEN)

for key in output.keys():
    print(key)
    print(output[key])
    print("=" * 100)
    


# # Data Loader

# In[ ]:


class NBMEDataset:
    def __init__(self, pn_history, feature_text, annotation, location):   #text(X) #label(Y), #selected_text #start:end
        self.pn_history = pn_history
        self.feature_text = feature_text
        self.annotation = annotation
        self.location = location
        self.tokenizer = config.TOKENIZER
        self.max_len = config.MAX_LEN
        
    def __len__(self):
        return len(self.pn_history)
        
    def __getitem__(self, item):
        data = process_data_tokenize(
            self.pn_history[item],
            self.feature_text[item],
            self.annotation[item],
            self.location[item],
            self.tokenizer,
            self.max_len
        )
        
        return {
            'ids': torch.tensor(data["ids"]), #input_ids
            'mask': torch.tensor(data["mask"], dtype=torch.long), #attention_mask
            'token_type_ids': torch.tensor(data["token_type_ids"], dtype=torch.long), #segment_ids
            'labels': torch.tensor(data["labels"], dtype=torch.long), 
            'offsets': torch.tensor(data["offsets"], dtype=torch.long)
        }


# # The Model

# In[ ]:


class NBMEModel(transformers.BertPreTrainedModel):    #torch.nn.Module
    def __init__(self,conf):
        super(NBMEModel,self).__init__(conf)
        self.bert = transformers.BertModel.from_pretrained(config.BERT_PATH, config=conf)
        self.dropout = torch.nn.Dropout(config.DROPOUT)
        self.classifier = torch.nn.Linear(768, 1)
        torch.nn.init.normal_(self.classifier.weight, std=0.02) 
        
    def forward(self, ids, mask, token_type_ids):
        sequence_out = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)[0] #last_hidden_state
        batch_size,max_len,feat_dim = sequence_out.shape
        sequence_output = self.dropout(sequence_out)
        logits = self.classifier(sequence_output)
        logits = logits.squeeze(-1) 
        return logits


# # Utility Function

# In[ ]:


class AverageMeter:
    """
    Computes and stores the average and current value
    """
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
        


# # Loss Function

# In[ ]:


def loss_fn(logits, labels):
    loss_fct = torch.nn.BCEWithLogitsLoss(reduction = "none")
    loss = loss_fct(logits,labels)
    return loss


# # Training Function

# In[ ]:


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[ ]:


def train_fn(dataloader, model, optimizer, scheduler=None):
    model.train()
    losses = AverageMeter() # Computes and stores the average and current value
    tk = tqdm(dataloader, total=len(dataloader)) #tqdm is a Python library for adding progress bar. 
    
    for batch, data in enumerate(tk):
        ids = data['ids']
        token_type_ids = data["token_type_ids"]
        mask = data["mask"]
        labels = data["labels"]
        offsets = data["offsets"]
        
        #adding the below data to device ;device enables you to specify the device type responsible to load a tensor into memory.
        ids = ids.to(DEVICE, dtype=torch.long)
        token_type_ids = token_type_ids.to(DEVICE, dtype=torch.long)
        mask = mask.to(DEVICE, dtype=torch.long)
        labels = labels.to(DEVICE, dtype=torch.float64)

        model.zero_grad()
        logits = model(ids=ids, mask=mask, token_type_ids=token_type_ids) #last_hidden_state

        loss = loss_fn(logits, labels)
        loss = torch.masked_select(loss, labels > -1.0).mean()
        losses.update(loss.item(),ids.size(0))
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)
        optimizer.step()
        scheduler.step() ## Update learning rate schedule
        
        #output = torch.argmax(torch.softmax(logits, dim=2),dim=2).cpu().detach().numpy()
        tk.set_postfix(loss=losses.avg)
        
    return losses.avg


# # Evaluation Functions

# In[ ]:


def eval_fn(dataloader, model):
    model.eval()
    losses = AverageMeter() # Computes and stores the average and current value

    with torch.no_grad():
        tk = tqdm(dataloader, total=len(dataloader)) 
        for batch, data in enumerate(tk):

            ids = data['ids']
            token_type_ids = data["token_type_ids"]
            mask = data["mask"]
            labels = data["labels"]
            offsets = data["offsets"]

            ids = ids.to(DEVICE, dtype=torch.long)
            token_type_ids = token_type_ids.to(DEVICE, dtype=torch.long)
            mask = mask.to(DEVICE, dtype=torch.long)
            labels = labels.to(DEVICE, dtype=torch.float64)

            logits = model(ids=ids, mask=mask, token_type_ids=token_type_ids) #last_hidden_state
            
            loss = loss_fn(logits, labels)
            loss = torch.masked_select(loss, labels > -1.0).mean()
            losses.update(loss.item(),ids.size(0))
            tk.set_postfix(loss=losses.avg)
        
        return losses.avg


# # Training

# In[ ]:


def run(fold):
    
    train_loss_data, valid_loss_data = [], []
    best_loss = np.inf
    since = time.time()
   
    df_train = dfx[dfx.kfold != fold].reset_index(drop=True) 
    df_valid = dfx[dfx.kfold == fold].reset_index(drop=True)
    
    train_dataset = NBMEDataset(
        pn_history=df_train.pn_history.values,
        feature_text=df_train.feature_text.values,
        annotation=df_train.annotation.values,
        location=df_train.location.values
        
    )
    
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN_BATCH_SIZE,
        num_workers=4
    )

    valid_dataset = NBMEDataset(
        pn_history=df_valid.pn_history.values,
        feature_text=df_valid.feature_text.values,
        annotation=df_valid.annotation.values,
        location=df_valid.location.values
    )

    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.VALID_BATCH_SIZE,
        num_workers=2
    )

    model_config = transformers.BertConfig.from_pretrained(config.BERT_PATH)
    model_config.output_hidden_states = True
    model = NBMEModel(conf=model_config)
    model.to(DEVICE)
    
    num_train_steps = int(len(df_train) / config.TRAIN_BATCH_SIZE * config.EPOCHS)
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    ]
    optimizer = AdamW(optimizer_parameters, lr=config.LEARNING_RATE)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=0, 
        num_training_steps=num_train_steps
    )

    best_loss = np.inf
    
    for i in range(config.EPOCHS):
        print("Epoch: {}/{}".format(i + 1, config.EPOCHS))
    
        # train model
        train_loss = train_fn(train_data_loader, model, optimizer, scheduler=scheduler)
        train_loss_data.append(train_loss)
        print(f"Train loss: {train_loss}")

        # evaluate model
        valid_loss = eval_fn(valid_data_loader, model)
        valid_loss_data.append(valid_loss)
        print(f"Valid loss: {valid_loss}")


        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), "model_fold1.bin")


        time_elapsed = time.time() - since
        print('Training completed in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
    
    


# In[ ]:


run(fold=0)


# In[ ]:


#run(fold=1)


# In[ ]:


#run(fold=2)


# In[ ]:


#run(fold=3)


# In[ ]:


#run(fold=4)


# # Do the evauation on test data
# ##### [inference in progress...]

# In[ ]:


# df_tst = pd.merge(test_df, features_df, on=['feature_num','case_num'], how='inner')
# df_tst = pd.merge(df_tst, patient_notes_df, on=['pn_num','case_num'], how='inner')
# df_tst.shape


# In[ ]:


# device = torch.device("cuda")
# model_config = transformers.BertConfig.from_pretrained(config.BERT_PATH)
# model_config.output_hidden_states = True

# #similarly this can be done for all 5 models
# model1 = NBMEModel(conf=model_config)
# model1.to(config.DEVICE)
# model1.load_state_dict(torch.load("model_fold1.bin"))
# model1.eval()

# test_dataset = NBMEDataset(
#     pn_history=df_test.pn_history.values,
#     feature_text=df_test.feature_text.values,
#     annotation=df_test.annotation.values,
#     location=df_test.location.values

# )

# data_loader = torch.utils.data.DataLoader(
#     test_dataset,
#     shuffle=False,
#     batch_size=config.TRAIN_BATCH_SIZE,
#     num_workers=1
# )

# with torch.no_grad():
#     tk = tqdm(data_loader, total=len(data_loader)) #tqdm is a Python library for adding progress bar. 

#     for batch, data in enumerate(tk):
#         ids = data['ids']
#         token_type_ids = data["token_type_ids"]
#         mask = data["mask"]
#         labels = data["labels"]
#         offsets = data["offsets"]

#         ids = ids.to(DEVICE, dtype=torch.long)
#         token_type_ids = token_type_ids.to(DEVICE, dtype=torch.long)
#         mask = mask.to(DEVICE, dtype=torch.long)
#         labels = labels.to(DEVICE, dtype=torch.float64)

#         logits = model(ids=ids, mask=mask, token_type_ids=token_type_ids) #last_hidden_state

