#!/usr/bin/env python
# coding: utf-8

# # Imports and Defines

# In[ ]:


from IPython.display import clear_output
get_ipython().system('pip install transformers')
clear_output()

import numpy as np
import pandas as pd
import random,os
import warnings
warnings.filterwarnings('ignore')

import transformers
from transformers import BertTokenizer
from transformers import TFBertModel

import tensorflow as tf 
from tensorflow.keras.optimizers import Adam

TRAIN_PATH = "../input/nlp-getting-started/train.csv"
TEST_PATH = "../input/nlp-getting-started/test.csv"
SAMPLE_SUBMISSION_PATH = "../input/nlp-getting-started/sample_submission.csv"
SUBMISSION_PATH = "submission.csv"

ID = "id"
TARGET = "target"

SEED = 2022
def seed_everything(seed=SEED):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    
seed_everything()


MODEL_NAME = "bert-large-uncased"
MODEL_MAX_LENGTH = 60
MODEL_INPUT_IDS_COL = "input_ids"
MODEL_ATTENTION_MASK_COL = "attention_mask"

MODEL_DATATYPE = "int32"
MODEL_DENSE = 32
MODEL_DROPOUT = 0.2
MODEL_ACTIVATION = "relu"
MODEL_LAST_ACTIVATION = "sigmoid"
MODEL_LR = 6e-6
MODEL_LOSS = "binary_crossentropy"
MODEL_METRICS = ['accuracy']
MODEL_EPOCH = 2
MODEL_BATCH_SIZE = 10
MODEL_VAL_SIZE = 0.2


# # Preprocess Data

# In[ ]:


train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

tokenizer = BertTokenizer.from_pretrained(MODEL_NAME, do_lower_case=True)

def bert_encode(data,maximum_length) :
    input_ids = []
    attention_masks = []
  

    for i in range(len(data.text)):
        encoded = tokenizer.encode_plus(

        data.text[i],
        add_special_tokens=True,
        max_length=maximum_length,
        pad_to_max_length=True,

        return_attention_mask=True,

        )

        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])
        
    return np.array(input_ids),np.array(attention_masks)

train_input_ids,train_attention_masks = bert_encode(train,MODEL_MAX_LENGTH)
test_input_ids,test_attention_masks = bert_encode(test,MODEL_MAX_LENGTH)


# # Define Model

# In[ ]:


def create_model(bert_model):
    input_ids = tf.keras.Input(shape=(MODEL_MAX_LENGTH,),dtype=MODEL_DATATYPE)
    attention_masks = tf.keras.Input(shape=(MODEL_MAX_LENGTH,),dtype=MODEL_DATATYPE)

    output = bert_model([input_ids,attention_masks])
    output = output[1]
    output = tf.keras.layers.Dense(MODEL_DENSE,activation=MODEL_ACTIVATION)(output)
    output = tf.keras.layers.Dropout(MODEL_DROPOUT)(output)

    output = tf.keras.layers.Dense(1,activation=MODEL_LAST_ACTIVATION)(output)
    model = tf.keras.models.Model(inputs = [input_ids,attention_masks],outputs = output)
    model.compile(Adam(lr=MODEL_LR), loss=MODEL_LOSS, metrics=MODEL_METRICS)
    return model


bert_model = TFBertModel.from_pretrained(MODEL_NAME)

model = create_model(bert_model)
model.summary()


# # Build Model

# In[ ]:


history = model.fit(
    [train_input_ids,train_attention_masks],
    train[TARGET],
    validation_split=MODEL_VAL_SIZE,
    epochs=MODEL_EPOCH,
    batch_size=MODEL_BATCH_SIZE)


# # Predict Data

# In[ ]:


pred_test = model.predict([test_input_ids,test_attention_masks])

sub = pd.read_csv(SAMPLE_SUBMISSION_PATH)
sub[TARGET] = np.round(pred_test).astype(int)
sub.to_csv(SUBMISSION_PATH,index=False)
sub.head()

