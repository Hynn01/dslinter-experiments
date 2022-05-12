#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


train=pd.read_csv('/kaggle/input/sentiment-analysis-on-movie-reviews/train.tsv.zip',sep='\t')
test=pd.read_csv('/kaggle/input/sentiment-analysis-on-movie-reviews/test.tsv.zip',sep='\t')


# In[ ]:


#!pip install transformers


# In[ ]:


get_ipython().system('pip install keras_sequential_ascii')


# In[ ]:


from tensorflow.keras.layers import Input, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.utils import to_categorical

import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# In[ ]:


data = train[['Phrase', 'Sentiment']]


data['Sentiment_value'] = pd.Categorical(data['Sentiment'])


data['Sentiment'] = data['Sentiment_value'].cat.codes


# In[ ]:


train_data, test_data = train_test_split(data, test_size = 0.25)


# **1) BERT**

# In[ ]:


from transformers import TFBertModel,  BertConfig, BertTokenizerFast


# In[ ]:


# Name of the BERT model to use
model_name = 'bert-base-uncased'

# Max length of tokens
max_length = 45

# Load transformers config and set output_hidden_states to False
config = BertConfig.from_pretrained(model_name)
config.output_hidden_states = False

# Load BERT tokenizer
tokenizer = BertTokenizerFast.from_pretrained(pretrained_model_name_or_path = model_name, config = config)

# Load the Transformers BERT model
transformer_bert_model = TFBertModel.from_pretrained(model_name, config = config)


# In[ ]:


# Load the MainLayer
bert = transformer_bert_model.layers[0]

# Build your model input
input_ids = Input(shape=(max_length,), name='input_ids', dtype='int32')
inputs = {'input_ids': input_ids}

# Load the Transformers BERT model as a layer in a Keras model
bert_model = bert(inputs)[1]
dropout = Dropout(config.hidden_dropout_prob, name='pooled_output')
pooled_output = dropout(bert_model, training=False)

# Then build your model output
Sentiments = Dense(units=len(train_data.Sentiment_value.value_counts()), kernel_initializer=TruncatedNormal(stddev=config.initializer_range), name='Sentiment')(pooled_output)
outputs = {'Sentiment': Sentiments}

# And combine it all in a model object
model_bert = Model(inputs=inputs, outputs=outputs, name='BERT_MultiClass')


# In[ ]:


model_bert.summary()


# In[ ]:


# Set an optimizer
optimizer = Adam(learning_rate=5e-05,epsilon=1e-08,decay=0.01,clipnorm=1.0)

# Set loss and metrics
loss = {'Sentiment': CategoricalCrossentropy(from_logits = True)}

# Compile the model
model_bert.compile(optimizer = optimizer, loss = loss, metrics = ['accuracy'])

# Ready output data for the model
y_train = to_categorical(train_data['Sentiment'])

# Tokenize the input 
x_train = tokenizer(
          text=train_data['Phrase'].to_list(),
          add_special_tokens=True,
          max_length=max_length,
          truncation=True,
          padding=True, 
          return_tensors='tf',
          return_token_type_ids = False,
          return_attention_mask = True,
          verbose = True)

y_val = to_categorical(test_data['Sentiment'])

x_val = tokenizer(
          text=test_data['Phrase'].to_list(),
          add_special_tokens=True,
          max_length=max_length,truncation=True,
          padding=True, 
          return_tensors='tf',
          return_token_type_ids = False,
          return_attention_mask = True,
          verbose = True)


# In[ ]:


# Fit the model
history_bert = model_bert.fit(
    x={'input_ids': x_train['input_ids']},
    y={'Sentiment': y_train},
    validation_data=({'input_ids': x_val['input_ids']},{'Sentiment': y_val}),
    batch_size=64,
    epochs=3,
    verbose=1)


# In[ ]:


loss_train = history_bert.history['loss']
epochs = range(1,4)
plt.plot(epochs, loss_train, 'g', label='Training loss')
plt.title('Training loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[ ]:


accuracy_train = history_bert.history['accuracy']
epochs = range(1,4)
plt.plot(epochs, accuracy_train, 'g', label='Training Accuracy')
plt.title('Training Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# In[ ]:


model_bert.save('MODEL-BERT.h5')


# In[ ]:





# In[ ]:


from keras.utils.vis_utils import plot_model
plot_model(model_bert, to_file='model.png')


# In[ ]:





# In[ ]:


model_eval = model_bert.evaluate(
    x={'input_ids': x_val['input_ids']},
    y={'Sentiment': y_val}
)


# In[ ]:


y_val_predicted = model_bert.predict(
    x={'input_ids': x_val['input_ids']},
)


# In[ ]:


x_test = tokenizer(
          text=test['Phrase'].to_list(),
          add_special_tokens=True,
          max_length=max_length,
          truncation=True,
          padding=True, 
          return_tensors='tf',
          return_token_type_ids = False,
          return_attention_mask = False,
          verbose = True)


# In[ ]:


label_predicted_bert = model_bert.predict(
    x={'input_ids': x_test['input_ids']},
)


# In[ ]:


label_pred_bert=[np.argmax(i) for i in label_predicted_bert['Sentiment']]


# In[ ]:


label_pred_bert[:5]


# In[ ]:


sample_submission=pd.read_csv('../input/sentiment-analysis-on-movie-reviews/sampleSubmission.csv')
sample_submission['Sentiment'] = label_pred_bert
sample_submission.to_csv("submission_BERT.csv", index=False, header=True)


# **2) ROBERTA**

# In[ ]:


from transformers import RobertaTokenizer, TFRobertaModel, RobertaConfig  


# In[ ]:


model_name = 'roberta-base'

# Max length of tokens
max_length = 45

# Load transformers config and set output_hidden_states to False
config = RobertaConfig.from_pretrained(model_name)
config.output_hidden_states = False

# Load Roberta tokenizer
tokenizer = RobertaTokenizer.from_pretrained(pretrained_model_name_or_path = model_name, config = config)

# Load the Roberta model
transformer_roberta_model = TFRobertaModel.from_pretrained(model_name, config = config)


# In[ ]:


# Load the MainLayer
roberta = transformer_roberta_model.layers[0]

# Build your model input
input_ids = Input(shape=(max_length,), name='input_ids', dtype='int32')
inputs = {'input_ids': input_ids}

# Load the Transformers RoBERTa model as a layer in a Keras model
roberta_model = roberta(inputs)[1]
dropout = Dropout(config.hidden_dropout_prob, name='pooled_output')
pooled_output = dropout(roberta_model, training=False)

# Then build your model output
Sentiments = Dense(units=len(train_data.Sentiment_value.value_counts()), kernel_initializer=TruncatedNormal(stddev=config.initializer_range), name='Sentiment')(pooled_output)
outputs = {'Sentiment': Sentiments}

# And combine it all in a model object
model_roberta = Model(inputs=inputs, outputs=outputs, name='RoBERTa_MultiClass')


# In[ ]:


model_roberta.summary()


# In[ ]:


from keras.utils.vis_utils import plot_model
plot_model(model_roberta, to_file='model.png')


# In[ ]:


# Set an optimizer
optimizer = Adam(learning_rate=1e-05,epsilon=1e-06,decay=0.01)

# Set loss and metrics
loss = {'Sentiment': CategoricalCrossentropy(from_logits = True)}

# Compile the model
model_roberta.compile(optimizer = optimizer, loss = loss, metrics = ['accuracy'])

# Ready output data for the model
y_train = to_categorical(train_data['Sentiment'])

# Tokenize the input 
x_train = tokenizer(
          text=train_data['Phrase'].to_list(),
          add_special_tokens=True,
          max_length=max_length,
          truncation=True,
          padding=True, 
          return_tensors='tf',
          return_token_type_ids = False,
          return_attention_mask = True,
          verbose = True)

y_val = to_categorical(test_data['Sentiment'])

x_val = tokenizer(
          text=test_data['Phrase'].to_list(),
          add_special_tokens=True,max_length=max_length,
          truncation=True,
          padding=True, 
          return_tensors='tf',
          return_token_type_ids = False,
          return_attention_mask = True,
          verbose = True)


    


# In[ ]:


# Fit the model
history_roberta = model_roberta.fit(
    x={'input_ids': x_train['input_ids']},
    y={'Sentiment': y_train},
    validation_data=({'input_ids': x_val['input_ids']},{'Sentiment': y_val}),
    batch_size=64,
    epochs=3,
    verbose=1)


# In[ ]:


loss_train = history_roberta.history['loss']
epochs = range(1,4)
plt.plot(epochs, loss_train, 'g', label='Training loss')
plt.title('Training loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[ ]:


accuracy_train = history_roberta.history['accuracy']
epochs = range(1,4)
plt.plot(epochs, accuracy_train, 'g', label='Training Accuracy')
plt.title('Training Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# In[ ]:


model_roberta.save('MODEL-ROBERTA.h5')


# In[ ]:


model_eval_roberta = model_roberta.evaluate(
    x={'input_ids': x_val['input_ids']},
    y={'Sentiment': y_val}
)


# In[ ]:


y_val_predicted = model_roberta.predict(
    x={'input_ids': x_val['input_ids']},
)


# In[ ]:


x_test = tokenizer(
          text=test['Phrase'].to_list(),
          add_special_tokens=True,
          max_length=max_length,
          truncation=True,
          padding=True, 
          return_tensors='tf',
          return_token_type_ids = False,
          return_attention_mask = False,
          verbose = True)


# In[ ]:


label_predicted = model_roberta.predict(
    x={'input_ids': x_test['input_ids']},
)


# In[ ]:


label_pred_roberta=[np.argmax(i) for i in label_predicted['Sentiment']]


# In[ ]:


sample_submission=pd.read_csv('../input/sentiment-analysis-on-movie-reviews/sampleSubmission.csv')
sample_submission['Sentiment'] = label_pred_roberta
sample_submission.to_csv("submission_ROBERTA.csv", index=False, header=True)


# In[ ]:




