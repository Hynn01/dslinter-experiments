#!/usr/bin/env python
# coding: utf-8

# ### 1) Importing Libraries and our functions

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string
from nltk.tokenize import word_tokenize
import nltk
# nltk.download('stopwords')
from nltk.corpus import stopwords
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from wordcloud import WordCloud,STOPWORDS, ImageColorGenerator


# ### 2) Reading Data

# In[ ]:


data = pd.read_csv('../input/youtube-videos-dataset/youtube.csv')
data.head()


# In[ ]:


print(data.description[0])


# In[ ]:


data.shape


# In[ ]:


# getting count of null values
data.isnull().sum()


# In[ ]:


# getting unqiue values of "category" Column
data['category'].unique()


# In[ ]:


# getting the frequency count for each category in "category" columns
category_valueCounts = data['category'].value_counts()
category_valueCounts


# In[ ]:


plt.figure(figsize=(12,8))
plt.pie(category_valueCounts,labels=category_valueCounts.index, autopct='%1.1f%%',
        shadow=True, startangle=90)
plt.show()


# In[ ]:


def showWordCloud(categoryName,notInclude=['subscribers','SUBSCRIBE','subscribers'],w=15 , h= 15):
    global data
    print(f"Word Cloud for {categoryName}")
    plt.figure(figsize=(w,h))
    text = " ".join(word for word in data[data.category==categoryName].description.astype(str))
    for word in notInclude:
        text = text.replace(word , "")
    wordcloud = WordCloud(background_color='white',stopwords=STOPWORDS,max_words=90).generate(text)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()
    


# In[ ]:


showWordCloud('food',['subscribers','SUBSCRIBE','subscribers','SHOW','video'])


# In[ ]:


showWordCloud('travel',['subscribers','SUBSCRIBE','subscribers','SHOW','video'])


# In[ ]:


showWordCloud('history',['subscribers','SUBSCRIBE','subscribers','SHOW','video'])


# In[ ]:


showWordCloud('art_music',['subscribers','SUBSCRIBE','subscribers','SHOW','video'])


# ### 3) Data Processing

# ##### 1) Remove Punctuation

# In[ ]:


punctuation = string.punctuation
punctuation


# In[ ]:


def removePunctuationFromText(text):
    text = ''.join([char for char in text if char not in punctuation])
    return text


# In[ ]:


data['descriptionNonePunct'] = data['description'].apply(removePunctuationFromText)


# In[ ]:


data.head()


# ##### 2) Tokenize words

# In[ ]:


## tokenize words 
data['descriptionTokenized'] = data['descriptionNonePunct'].apply(word_tokenize)


# In[ ]:


data.head()


# ##### 3) Remove stopwords

# In[ ]:


stopWords = stopwords.words('english')
stopWords[:10]


# In[ ]:


def removeStopWords(text):
    return [word for word in text if word not in stopWords]


# In[ ]:


data['descriptionNoneSW'] =data['descriptionTokenized'].apply(removeStopWords)


# In[ ]:


data.head()


# ##### 4) Text to Squence

# In[ ]:


tokenizer = keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(data['descriptionNoneSW'])
data['textSequence'] = tokenizer.texts_to_sequences(data['descriptionNoneSW'])


# In[ ]:


data.head()


# ### 4) Model Building

# In[ ]:


input_shape = int(sum(data['textSequence'].apply(lambda x:len(x)/len(data['textSequence']))))
input_shape


# In[ ]:


from keras.preprocessing.sequence import pad_sequences
X = pad_sequences(data['textSequence'],maxlen=45)
X


# In[ ]:


y = data['category']


# In[ ]:


y_encoder = LabelEncoder()
y = y_encoder.fit_transform(y)
y


# ##### Spliting the data

# In[ ]:


X_train , X_test , y_train , y_test = train_test_split(X,y ,test_size = 0.2 , random_state=42)


# In[ ]:


X_train.shape ,X_test.shape, y_train.shape , y_test.shape


# In[ ]:


# getting vocabulary len
maxWords=(max(map(max, X)))+1


# In[ ]:


model = keras.models.Sequential([
    keras.layers.Embedding(maxWords , 64 , input_shape=[input_shape]),
    keras.layers.GRU(32),
    keras.layers.Dense(4 , activation='softmax')
])


# ##### Plotting Model

# In[ ]:


# from tensorflow.keras.utils import plot_model
# plot_model(model , show_shapes=True)


# ##### Compiling the model

# In[ ]:


model.compile(loss='sparse_categorical_crossentropy' , optimizer='adam' , metrics='accuracy')


# In[ ]:


history = model.fit(X_train , y_train , epochs=15 , batch_size=32 , validation_split= 0.2)


# In[ ]:


pd.DataFrame(history.history).plot(figsize=(12,8))
plt.grid(True)
# set vertical range between 0 , 1
plt.gca().set_ylim(0,1)


# In[ ]:


model_evaluated = model.evaluate(X_test , y_test)


# In[ ]:


print(f'Model evaluated Loss is {model_evaluated[0]}')
print(f'Model evaluated accuracy is {model_evaluated[1]}')


# ##### Confusion Matrix and Classification Report 

# In[ ]:


y_pred = (model.predict(X_test).argmax(axis=-1)).tolist()
class_names = y_encoder.classes_
print("Classification report : \n",classification_report(y_test, y_pred, target_names = class_names))


# In[ ]:


def drawConfusionMatrix(true, preds, normalize=None):
  confusionMatrix = confusion_matrix(true, preds, normalize = normalize)
  confusionMatrix = np.round(confusionMatrix, 2)
  sns.heatmap(confusionMatrix, annot=True, annot_kws={"size": 12},fmt="g", cbar=False, cmap="viridis")
  plt.show()


# In[ ]:


print("Confusion matrix : \n")
drawConfusionMatrix(y_test, y_pred)
print("Normalized confusion matrix : \n")
drawConfusionMatrix(y_test, y_pred,"true")

