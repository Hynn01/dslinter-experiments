#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# In[ ]:


from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
import tensorflow.keras.utils 
import tensorflow as tf
import numpy as np 
import pandas as pd
import numpy as np


# # Load Dataset

# In[ ]:


df = pd.read_csv('/kaggle/input/shakespeare-plays/Shakespeare_data.csv')
df.head(1)


# In[ ]:


import csv

corpus = []

with open('/kaggle/input/shakespeare-plays/Shakespeare_data.csv') as f:
    reader = csv.reader(f, delimiter=',')
    next(reader)        # to pass first row,header
    for row in reader:
        corpus.append(row[5])
        
print(len(corpus))
print(corpus[:3])


# # Data Cleaning

# In[ ]:


import string

def text_cleaner(text):
    text = "".join(car for car in text if car not in string.punctuation).lower()
    text = text.encode("utf8").decode("ascii",'ignore')
    return text

corpus = [text_cleaner(line) for line in corpus]


# In[ ]:


# Tokenization is the process of splitting up a text into a list of individual words, or tokens.
# corpus is too big if you try with all data, you can see this message
corpus = corpus[:5000]
tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)
word_index = tokenizer.word_index
total_words = len(word_index) + 1
total_words


# In[ ]:


# create input sequences using list of tokens
input_sequences =[]

for sentence in corpus:
    token_list = tokenizer.texts_to_sequences([sentence])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)
        


# In[ ]:


# pad sequences 
max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, 
                                         maxlen=max_sequence_len, 
                                         padding='pre'))


# In[ ]:


# create predictors and label
predictors, label = input_sequences[:,:-1],input_sequences[:,-1]
# create one-hot encoding of the labels
label = tensorflow.keras.utils.to_categorical(label, num_classes=total_words)


# In[ ]:


print(label[0])
print(label[0].shape)


# In[ ]:


model = Sequential()
model.add(Embedding(total_words, 10, input_length=max_sequence_len-1))
model.add(Bidirectional(LSTM(512)))
model.add(Dropout(0.3))
model.add(Dense(total_words, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())


# In[ ]:


history = model.fit(predictors, label, epochs=50,  verbose=1)


# In[ ]:


import matplotlib.pyplot as plt

acc = history.history['accuracy']
loss = history.history['loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'b', label='Training accuracy')
plt.title('Training accuracy')

plt.figure()

plt.plot(epochs, loss, 'b', label='Training Loss')
plt.title('Training loss')
plt.legend()

plt.show()


# In[ ]:


seed_text = "help me in this"
next_words = 2

  
for _ in range(next_words):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict_classes(token_list, verbose=0)
    output_word = ""
    for word, index in tokenizer.word_index.items():
        if index == predicted:
            output_word = word
            break
    seed_text += " " + output_word
    if len(seed_text) % 10 == 0 :
        seed_text+= '\n'
print(seed_text)


# In[ ]:


seed_text = "Love all, trust a few"
next_words = 2

  
for _ in range(next_words):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict_classes(token_list, verbose=0)
    output_word = ""
    for word, index in tokenizer.word_index.items():
        if index == predicted:
            output_word = word
            break
    seed_text += " " + output_word
    if len(seed_text) % 10 == 0 :
        seed_text+= '\n'
print(seed_text)

