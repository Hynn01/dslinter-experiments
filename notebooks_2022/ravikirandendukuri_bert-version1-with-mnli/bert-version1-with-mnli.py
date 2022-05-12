#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from transformers import BertTokenizer, TFBertModel, AutoTokenizer,TFAutoModel
import tensorflow as tf
from datasets import load_dataset
import kerastuner as kt
#import plotlib as plt
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import warnings

warnings.filterwarnings('ignore') # ignore Jupiter warnings


# In[ ]:


try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
except ValueError:
    strategy = tf.distribute.get_strategy() # for CPU and single GPU
    print('Number of replicas:', strategy.num_replicas_in_sync)


# In[ ]:


model_Bert = 'bert-base-multilingual-cased'
tokenizer = AutoTokenizer.from_pretrained(model_Bert)
model = TFAutoModel.from_pretrained(model_Bert)


# In[ ]:


tokenizer


# In[ ]:


os.environ["WANDB_API_KEY"] = "0" 


# In[ ]:


train = pd.read_csv("../input/contradictory-my-dear-watson/train.csv")
test = pd.read_csv("../input/contradictory-my-dear-watson/test.csv")


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


import matplotlib.pyplot as plt
labels, frequencies =np.unique(train['language'], return_counts=True)
plt.figure(figsize=[10, 10])
plt.pie(frequencies, labels=labels, autopct='%1.1f%%')
plt.show()


# In[ ]:


def load_mnli(use_validation=True):
    result=[]
    dataset=load_dataset('multi_nli')
    print(dataset)
    for record in dataset['train']:
        c1, c2, c3 = record['premise'],record['hypothesis'], record['label']
        if c1 and c2 and c3 in {0, 1, 2}:
            result.append((c1, c2, 'en',c3))
    result=pd.DataFrame(result, columns=['premise', 'hypothesis', 'lang_abv' ,'label'])
    return result


# In[ ]:


mnli=load_mnli()
mnli


# In[ ]:


train = train[['premise', 'hypothesis', 'lang_abv', 'label']]
train


# In[ ]:


train=pd.concat([train, mnli.loc[:20000]], axis=0)


# In[ ]:


train.shape


# In[ ]:


SEQ_LEN = 236  

def bert_encode(df, tokenizer):    
    batch_premises = df['premise'].tolist()
    batch_hypothesis = df['hypothesis'].tolist()

    tokens = tokenizer(batch_premises, batch_hypothesis, max_length = SEQ_LEN,
                   truncation=True, padding='max_length',
                   add_special_tokens=True, return_attention_mask=True,
                   return_token_type_ids=True,
                   return_tensors='tf')
    inputs = {
          'input_ids': tokens['input_ids'], 
          'attention_mask': tokens['attention_mask'],
          'token_type_ids': tokens['token_type_ids']  }  
    return inputs


# In[ ]:


train_input = bert_encode(train, tokenizer)


# In[ ]:


train_input


# In[ ]:


from tensorflow.keras import regularizers

def build_model(): 
   
    encoder = TFAutoModel.from_pretrained(model_Bert)
    input_ids = tf.keras.Input(shape=(SEQ_LEN,), dtype=tf.int32, name="input_ids")
    attention_mask = tf.keras.Input(shape=(SEQ_LEN,), dtype=tf.int32, name="attention_mask")
    token_type_ids = tf.keras.Input(shape=(SEQ_LEN,), 
                                    dtype=tf.int32,  name="token_type_ids")
        
    embedding = encoder([input_ids, attention_mask , token_type_ids])[0] 
    inputs=[input_ids, attention_mask  , token_type_ids ] 
    hp_units1 = 64 
    hp_units2 = 32 
    x = tf.keras.layers.Dense(units = hp_units1, activation=tf.nn.relu)(embedding[:,0,:])
    x = tf.keras.layers.Dense(units = hp_units2, activation=tf.nn.relu, kernel_regularizer=regularizers.l2(l2=1e-4))(x)
    output = tf.keras.layers.Dense(3, activation='softmax')(x)
      
    model = tf.keras.Model(inputs=inputs, outputs=output)
    hp_learning_rate = 1e-6
    model.compile(tf.keras.optimizers.Adam(learning_rate = hp_learning_rate), loss='sparse_categorical_crossentropy', metrics=['accuracy'])   
    return model 


# In[ ]:


with strategy.scope(): 
    model = build_model()
    model.summary()      
    


# In[ ]:


for key in train_input.keys():
    train_input[key] = train_input[key][:,:SEQ_LEN]


# In[ ]:



history = model.fit(train_input, train['label'], epochs = 10, batch_size=64, 
                    validation_split = 0.2) #,callbacks=[hist]) verbose = 1,  


# In[ ]:


import matplotlib.pyplot as plt
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[ ]:


plt.clf() 
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# In[ ]:


test_input = bert_encode(test, tokenizer) 


# In[ ]:



for key in test_input.keys():
    test_input[key] = test_input[key][:,:SEQ_LEN]


# In[ ]:


predictions = [np.argmax(i) for i in model.predict(test_input)]
model.evaluate(test_input)


# In[ ]:


submission = test.id.copy().to_frame()
submission['prediction'] = predictions


# In[ ]:


submission.to_csv("./submission.csv", index = False)


# In[ ]:


submission.head()

