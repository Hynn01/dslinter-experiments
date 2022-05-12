#!/usr/bin/env python
# coding: utf-8

# # import library

# In[ ]:


import numpy as np  # vector input
import pandas as pd # readcsv

# clean data
#######################################
import re
import nltk
from nltk.corpus import stopwords
#######################################
from collections import Counter

# model
import tensorflow as tf
#######################################

#plot
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:





# # read file

# In[ ]:


test_df = pd.read_csv("../input/nlp-getting-started/test.csv")
train_df = pd.read_csv("../input/nlp-getting-started/train.csv")


# In[ ]:


train_df.info()


# In[ ]:


test_df.info()


# In[ ]:


display(test_df.head())
display(train_df.head())


# In[ ]:


plt.figure(figsize=(10,15))
for i in range(2):
    new_df = train_df[train_df["target"]==i]

    plt.subplot(2,1,i+1)
    b = new_df["keyword"].value_counts()
    sns.countplot(y=new_df.keyword, order = new_df.keyword.value_counts().iloc[:15].index)
    
plt.show()


# In[ ]:


pie = train_df["target"].value_counts()
y = [pie[0],pie[1]]
lb = ["False","True"]
fig, ax = plt.subplots(figsize=(6, 6))

patches, texts, pcts = ax.pie(
    y, autopct='%.2f%%',
    wedgeprops={ 'edgecolor': 'white'},
    colors = ["#e8a9e0","#fcfb8a"]
)

plt.setp(pcts, color='black', fontweight='bold')
ax.set_title('count_True-False', fontsize=18,color="white")
plt.legend(labels=lb)
plt.tight_layout()


# # count all words

# In[ ]:


all_text = " ".join(train_df["text"].values)
all_words = all_text.split()
all_words = set(all_words)
numwords = len(all_words)


# # convert text to vector

# In[ ]:


x_train = train_df["text"]
y_train = train_df["target"]
x_test = test_df["text"]


# In[ ]:


tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=numwords)
tokenizer.fit_on_texts(x_train)
tokenizer.fit_on_texts(x_test)
x_train2 = tokenizer.texts_to_sequences(x_train)
x_train3 = tf.keras.preprocessing.sequence.pad_sequences(x_train2, maxlen=33)

x_test2 = tokenizer.texts_to_sequences(x_test)
x_test3 = tf.keras.preprocessing.sequence.pad_sequences(x_test2, maxlen=33)


# # Create model

# In[ ]:


model = tf.keras.Sequential([
    tf.keras.layers.Embedding(numwords*2, 20, input_length=33),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])


# In[ ]:


model.summary()


# In[ ]:


epochs = 5
history = model.fit(x_train3, y_train, epochs=epochs,  validation_split=0.3, verbose=2)


# In[ ]:


df = pd.DataFrame(history.history)
df


# In[ ]:


x = np.arange(1,epochs+1)
x = pd.Series(x)
x = x.astype(str)
y1 = history.history["loss"]
y2 = history.history["accuracy"]
y3 = history.history["val_loss"]
y4 = history.history["val_accuracy"]
plt.figure(figsize=(10,10))
plt.subplot(2,2,1)
plt.plot(x,y1,label="loss")
plt.plot(x,y2,label="accuracy")
plt.grid()
plt.legend()

plt.subplot(2,2,2)
plt.plot(x,y3,label="val_loss")
plt.plot(x,y4,label="val_accuracy")
plt.grid()
plt.legend()
plt.show()


# # predict

# In[ ]:


def to_01(predict):
    if predict>0.5:
        x = 1
    else:
        x = 0
    return x

sub_pre = model.predict(x_test3)
sub_pre = sub_pre.flatten()
ans = pd.Series(sub_pre)
ans = ans.apply(lambda x:to_01(x))
ans_count = ans.value_counts()


# In[ ]:


plt.grid()
plt.bar(["false","true"],ans_count,color=["#f91919","#05fb0e"])
plt.show()


# # submission

# In[ ]:


submission = pd.read_csv("../input/nlp-getting-started/sample_submission.csv")
submission.info()


# In[ ]:


submission["target"] = ans
submission.head()


# In[ ]:


submission.to_csv('submission.csv',index=False)


# In[ ]:


submission["target"].value_counts()


# In[ ]:




