#!/usr/bin/env python
# coding: utf-8

# <hr style="width:100%;height:3px;border-width:0;background-color:silver">
# <h1 style="text-align:center">   
#       <font color = MidnightBlue >
#                 Fake News Detection with NLP and LSTM Project Review 1
#         </font>    
# </h1>   
# <hr style="width:100%;height:3px;border-width:0;background-color:silver">
# <center><img style = "height:550px;" src="https://images.livemint.com/rf/Image-621x414/LiveMint/Period2/2018/05/05/Photos/Processed/fakereal-k2QC--621x414@LiveMint.jpg"></center>
# <br>
# <h2><font color = MidnightBlue>What is "Fake News"?</font></h2>
# <p>“Fake news” is a term that has come to mean different things to different people. At its core, we are defining “fake news” as those news stories that are false: the story itself is fabricated, with no verifiable facts, sources or quotes. Sometimes these stories may be propaganda that is intentionally designed to mislead the reader, or may be designed as “clickbait” written for economic incentives (the writer profits on the number of people who click on the story). In recent years, fake news stories have proliferated via social media, in part because they are so easily and quickly shared online.</p>
# <h2><font color = MidnightBlue>About Dataset</font></h2>
# <p>This data set consists of 40000 fake and real news. Our goal is to train our model to accurately predict whether a particular piece of news is real or fake. Fake and real news data are given in two separate data sets, with each data set consisting of approximately 20000 articles.</p>
# <h2><font color = MidnightBlue>Content:</font></h2>
# <br>
#  
# 1. [Import Libraries](#1)
# 1. [Load and Check Data](#2)
# 1. [Visualization](#3)
# 1. [Data Cleaning](#4)
#     * [Removal of HTML Contents](#5)
#     * [Removal of Punctuation Marks and Special Characters](#6)
#     * [Removal of Stopwords](#7)
#     * [Lemmatization](#8)
#     * [Perform it for all the examples](#9)  
# 1. [N-Gram Analysis](#10)
#     * [Unigram Analysis](#11)
#     * [Bigram Analysis](#12)
#     * [Trigram Analysis](#13)
# 1. [Modeling](#14)
#     * [Train - Test Split](#15)
#     * [Tokenizing](#16)
#     * [Training LSTM Model](#17)
#     * [Analysis After Training](#18) 

# <a id = 1></a>
# <h1><font color = MidnightBlue>Import Libraries</font></h1>
# <hr style="width:100%;height:1.2px;border-width:0;background-color:silver">

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
import re
import string 

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from tensorflow import keras
from keras.preprocessing import text,sequence
from keras.models import Sequential
from keras.layers import Dense,Embedding,LSTM,Dropout

import warnings
warnings.filterwarnings('ignore')

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# <a id = 2></a>
# <h1><font color = MidnightBlue>Load and Check Data</font></h1>
# <hr style="width:100%;height:1.2px;border-width:0;background-color:silver">

# In[ ]:


real_data = pd.read_csv('/kaggle/input/fake-and-real-news-dataset/True.csv')#[:100]
fake_data = pd.read_csv('/kaggle/input/fake-and-real-news-dataset/Fake.csv')#[:100]


# In[ ]:


real_data.head()


# In[ ]:


fake_data.head()


# In[ ]:


#add column 
real_data['target'] = 1
fake_data['target'] = 0 


# In[ ]:


real_data.tail()


# In[ ]:


#Merging the 2 datasets
data = pd.concat([real_data, fake_data], ignore_index=True, sort=False)
data.head()


# In[ ]:


data.isnull().sum()


# <a id = 3></a>
# <h1><font color = MidnightBlue>Visualization</font></h1>
# <hr style="width:100%;height:1.2px;border-width:0;background-color:silver">

# **1.Count of Fake and Real Data**

# In[ ]:


print(data["target"].value_counts())
fig, ax = plt.subplots(1,2, figsize=(19, 5))
g1 = sns.countplot(data.target,ax=ax[0],palette="pastel");
g1.set_title("Count of real and fake data")
g1.set_ylabel("Count")
g1.set_xlabel("Target")
g2 = plt.pie(data["target"].value_counts().values,explode=[0,0],labels=data.target.value_counts().index, autopct='%1.1f%%',colors=['SkyBlue','PeachPuff'])
fig.show()


# **2.Distribution of The Subject According to Real and Fake Data**

# In[ ]:


print(data.subject.value_counts())
plt.figure(figsize=(10, 5))

ax = sns.countplot(x="subject",  hue='target', data=data, palette="pastel")
plt.title("Distribution of The Subject According to Real and Fake Data")


# <a id = 4></a>
# <h1><font color = MidnightBlue>Data Cleaning</font></h1>
# <hr style="width:100%;height:1.2px;border-width:0;background-color:silver">

# In[ ]:


data['text']= data['subject'] + " " + data['title'] + " " + data['text']
del data['title']
del data['subject']
del data['date']
data.head()


# In[ ]:


first_text = data.text[10]
first_text


# <a id = 5></a>
# <h2><font color = MidnightBlue>Removal of HTML Contents</font></h2>

# **First, let's remove HTML content.**

# In[ ]:


# pip install bs4


# In[ ]:


from bs4 import BeautifulSoup

soup = BeautifulSoup(first_text, "html.parser")
first_text = soup.get_text()
first_text


# <a id = 6></a>
# <h2><font color = MidnightBlue>Removal of Punctuation Marks and Special Characters</font></h2>

# **Let's now remove everything except uppercase / lowercase letters using Regular Expressions.**

# In[ ]:


first_text = re.sub('\[[^]]*\]', ' ', first_text)
first_text = re.sub('[^a-zA-Z]',' ',first_text)  # replaces non-alphabets with spaces
first_text = first_text.lower() # Converting from uppercase to lowercase
first_text


# <a id = 7></a>
# <h2><font color = MidnightBlue>Removal of Stopwords</font></h2>

# **Let's remove stopwords like is,a,the... Which do not offer much insight.**

# In[ ]:


nltk.download("stopwords")   
from nltk.corpus import stopwords  

# we can use tokenizer instead of split
first_text = nltk.word_tokenize(first_text)


# In[ ]:


first_text = [ word for word in first_text if not word in set(stopwords.words("english"))]


# <a id = 8></a>
# <h2><font color = MidnightBlue>Lemmatization</font></h2>

# **Lemmatization to bring back multiple forms of same word to their common root like 'coming', 'comes' into 'come'.**

# In[ ]:


lemma = nltk.WordNetLemmatizer()
first_text = [ lemma.lemmatize(word) for word in first_text] 

first_text = " ".join(first_text)
first_text


# <a id = 9></a>
# <h2><font color = MidnightBlue>Perform it for all the examples</font></h2>
# <b>We performed the steps for a single example. Now let's perform it for all the examples in the data.</b>

# In[ ]:


#Removal of HTML Contents
def remove_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

#Removal of Punctuation Marks
def remove_punctuations(text):
    return re.sub('\[[^]]*\]', '', text)

# Removal of Special Characters
def remove_characters(text):
    return re.sub("[^a-zA-Z]"," ",text)

#Removal of stopwords 
def remove_stopwords_and_lemmatization(text):
    final_text = []
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    for word in text:
        if word not in set(stopwords.words('english')):
            lemma = nltk.WordNetLemmatizer()
            word = lemma.lemmatize(word) 
            final_text.append(word)
    return " ".join(final_text)

#Total function
def cleaning(text):
    text = remove_html(text)
    text = remove_punctuations(text)
    text = remove_characters(text)
    text = remove_stopwords_and_lemmatization(text)
    return text

#Apply function on text column
data['text']=data['text'].apply(cleaning)


# In[ ]:


data.head()


# <h2><b>Let's make some visualization with new data.</b></h2>

# <h3><b>1.WordCloud for Real News </b></h3>

# In[ ]:


from wordcloud import WordCloud,STOPWORDS
plt.figure(figsize = (15,15))
wc = WordCloud(max_words = 500 , width = 1000 , height = 500 , stopwords = STOPWORDS).generate(" ".join(data[data.target == 1].text))
plt.imshow(wc , interpolation = 'bilinear')


# <h3><b>2.WordCloud for Fake News </b></h3>

# In[ ]:


plt.figure(figsize = (15,15))
wc = WordCloud(max_words = 500 , width = 1000 , height = 500 , stopwords = STOPWORDS).generate(" ".join(data[data.target == 0].text))
plt.imshow(wc , interpolation = 'bilinear')


# <h3><b>Number of words in each text</b></h3>

# In[ ]:


fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12,8))
text_len=data[data['target']==0]['text'].str.split().map(lambda x: len(x))
ax1.hist(text_len,color='SkyBlue')
ax1.set_title('Fake news text')
text_len=data[data['target']==1]['text'].str.split().map(lambda x: len(x))
ax2.hist(text_len,color='PeachPuff')
ax2.set_title('Real news text')
fig.suptitle('Words in texts')
plt.show()


# **The number of words seems to be a bit different. 500 words  are most common in real news category while around 250 words are most common in fake news category.**

# <a id = 10></a>
# <h2><font color = MidnightBlue>N-Gram Analysis</font></h2>
# <hr style="width:100%;height:1.2px;border-width:0;background-color:silver">
# <center><img style = "height:450px;" src="https://devopedia.org/images/article/219/7356.1569499094.png"></center>
# <hr style="width:100%;height:1.2px;border-width:0;background-color:silver">

# In[ ]:


texts = ' '.join(data['text'])


# In[ ]:


string = texts.split(" ")


# In[ ]:


def draw_n_gram(string,i):
    n_gram = (pd.Series(nltk.ngrams(string, i)).value_counts())[:15]
    n_gram_df=pd.DataFrame(n_gram)
    n_gram_df = n_gram_df.reset_index()
    n_gram_df = n_gram_df.rename(columns={"index": "word", 0: "count"})
    print(n_gram_df.head())
    plt.figure(figsize = (16,9))
    return sns.barplot(x='count',y='word', data=n_gram_df)


# <a id = 11></a>
# <h2><font color = MidnightBlue>Unigram Analysis</font></h2>

# In[ ]:


draw_n_gram(string,1)


# <a id = 12></a>
# <h2><font color = MidnightBlue>Bigram Analysis</font></h2>

# In[ ]:


draw_n_gram(string,2)


# <a id = 13></a>
# <h2><font color = MidnightBlue>Trigram Analysis</font></h2>

# In[ ]:


draw_n_gram(string,3)


# <a id = 14></a>
# <h2><font color = MidnightBlue>Modeling</font></h2>
# <hr style="width:100%;height:1.2px;border-width:0;background-color:silver">

# <a id = 15></a>
# <h2><font color = MidnightBlue>Train Test Split</font></h2>

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(data['text'], data['target'], random_state=0)


# <a id = 16></a>
# <h2><font color = MidnightBlue>Tokenizing</font></h2>

# * **Tokenizing Text -> Repsesenting each word by a number**
# 
# * **Mapping of orginal word to number is preserved in word_index property of tokenizer**
# 
# <h3><b>Lets keep all news to 300, add padding to news with less than 300 words and truncating long ones </b></h3>

# In[ ]:


max_features = 10000
maxlen = 300


# In[ ]:


tokenizer = text.Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(X_train)
tokenized_train = tokenizer.texts_to_sequences(X_train)
X_train = sequence.pad_sequences(tokenized_train, maxlen=maxlen)


# In[ ]:


tokenized_test = tokenizer.texts_to_sequences(X_test)
X_test = sequence.pad_sequences(tokenized_test, maxlen=maxlen)


# <a id = 17></a>
# <h2><font color = MidnightBlue>Training LSTM Model</font></h2>

# In[ ]:


batch_size = 256
epochs = 10
embed_size = 100


# In[ ]:


model = Sequential()
#Non-trainable embeddidng layer
model.add(Embedding(max_features, output_dim=embed_size, input_length=maxlen, trainable=False))
#LSTM 
model.add(LSTM(units=128 , return_sequences = True , recurrent_dropout = 0.25 , dropout = 0.25))
model.add(LSTM(units=64 , recurrent_dropout = 0.1 , dropout = 0.1))
model.add(Dense(units = 32 , activation = 'relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer=keras.optimizers.Adam(lr = 0.01), loss='binary_crossentropy', metrics=['accuracy'])


# In[ ]:


model.summary()


# In[ ]:


history = model.fit(X_train, y_train, validation_split=0.3, epochs=10, batch_size=batch_size, shuffle=True, verbose = 1)


# <a id = 18></a>
# <h2><font color = MidnightBlue>Analysis After Training </font></h2>

# In[ ]:


print("Accuracy of the model on Training Data is - " , model.evaluate(X_train,y_train)[1]*100 , "%")
print("Accuracy of the model on Testing Data is - " , model.evaluate(X_test,y_test)[1]*100 , "%")


# In[ ]:


plt.figure()
plt.plot(history.history["accuracy"], label = "Train")
plt.plot(history.history["val_accuracy"], label = "Test")
plt.title("Accuracy")
plt.ylabel("Acc")
plt.xlabel("epochs")
plt.legend()
plt.show()


# In[ ]:


plt.figure()
plt.plot(history.history["loss"], label = "Train")
plt.plot(history.history["val_loss"], label = "Test")
plt.title("Loss")
plt.ylabel("Acc")
plt.xlabel("epochs")
plt.legend()
plt.show()


# In[ ]:


predict_x = model.predict(X_test)
predictions = np.argmax(predict_x,axis=1)
pred = predictions#model.predict_classes(X_test)
print(classification_report(y_test, pred, target_names = ['Fake','Real']))


# In[ ]:




