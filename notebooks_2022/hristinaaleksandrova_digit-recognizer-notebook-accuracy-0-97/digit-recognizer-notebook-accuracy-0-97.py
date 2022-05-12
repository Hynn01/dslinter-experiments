#!/usr/bin/env python
# coding: utf-8

# <h3>Make the necessary imports</h3>

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout


# <h3>Reading the data:</h3>

# In[ ]:


train_data = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')


# In[ ]:


test_data = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')


# In[ ]:


train_data.head(10)


# <h3>Splitting the train data into train, test, validation datasets</h3>

# In[ ]:


train_data.shape


# In[ ]:


"""
 More than 50% for the train data
"""



train= train_data[:37000]
val = train_data[37000:41000]
test = train_data[41000:42000]


# In[ ]:



print(f'Train dataset shape {train.shape}\nValidation dataset shape {val.shape}')


# <h3>Spliting the datasets into X and y </h3>

# <h5>Train supset split</h5>

# In[ ]:


x_train = train.drop('label',axis=1)      # Reading the images by not including the labels
y_train = train['label']              


# In[ ]:


x_val = val.drop(columns=['label'])      # Reading the images by not including the labels
y_val = val['label']    


# In[ ]:


y_test = test.label
x_test=test.drop('label', axis = 1)


# <h3> Preparing the model </h3>

# In[ ]:


tf.config.list_logical_devices()


# In[ ]:


tf.keras.backend.clear_session()


# In[ ]:


model = Sequential([
    Input(shape = (784)),
   
    Dense(64 , activation = "relu"),
    BatchNormalization(),
    Dense(64 , activation = "relu"),
    BatchNormalization(),
    Dense(64 , activation = "relu"),
    BatchNormalization(),
    Dropout(0.1),
  
    Dense(32,  activation = "relu"),
    BatchNormalization(),
    Dense(32,  activation = "relu"),
    BatchNormalization(),
    Dense(32,  activation = "relu"),
    BatchNormalization(),
    Dropout(0.5),
  
  
    Dense(16,  activation = "relu"),
    BatchNormalization(),
    Dense(16,  activation = "relu"),
    BatchNormalization(),
    Dense(16,  activation = "relu"),
    BatchNormalization(),
    Dropout(0.1),
    
    Dense(10,  activation = "softmax")
])


# In[ ]:


model.summary()


# In[ ]:


model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics = ["accuracy"]
    )


# <h3> Fit the data: </h3>

# In[ ]:


model.fit(
    x_train,
    y_train,
    validation_data =(x_val, y_val),
    epochs=100, 
    batch_size=128
#     callbacks=[TensorBoard(log_dir="./logs_val")]
)


# <h3> Make prediction on x_test </h3>

# In[ ]:


test_predic= model.predict(x_test)


# In[ ]:


test_predic


# <h3> Using argmax to take the most propable prediction</h3>

# In[ ]:


answer_label=test_predic.argmax(axis=1)


# In[ ]:


answer_label[0]


# <h3> Results </h3>

# In[ ]:


for i in range(10):
    print(f'Real value : {y_test.iloc[i]} vs Predicted value :{answer_label[i]}')


# <h3> Making prediction and exporting for Kaggle submition </h3>

# In[ ]:


predictions=model.predict(test_data)


# In[ ]:


submission=predictions.argmax(axis=1)


# In[ ]:


submission =pd.Series(submission)


# In[ ]:


"""Creating DataFrame"""
submission_data =submission.to_frame(name='submission_data')


# In[ ]:


submission_data


# In[ ]:


submission_data=submission_data.rename(columns={ submission_data.columns[0]: "Label" })


# In[ ]:


submission_data['ImageId'] = range(1, len(submission_data)+1)


# In[ ]:


submission_data


# In[ ]:





# In[ ]:


submission_data= submission_data[['ImageId', 'Label']]


# In[ ]:


submission_data


# In[ ]:


"""Export csv"""
sub = submission_data.to_csv('sub.csv', index=False)


# In[ ]:




