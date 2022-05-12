#!/usr/bin/env python
# coding: utf-8

# > # Emotion Prediction with Semi Supervised Learning of Machine Learning Software with RC Algorithm - By Emirhan BULUT

# I have created an artificial intelligence software that can make an emotion prediction based on the text you have written using the Semi Supervised Learning method and the RC algorithm. I used very simple codes and it was a software that focused on solving the problem. I aim to create the 2nd version of the software using RNN (Recurrent Neural Network). I hope I was able to create an example for you to use in your thesis and projects.
# 
# Happy learning!
# 
# Emirhan BULUT
# 
# Head of AI and AI Inventor
# 
# ###**The coding language used:**
# 
# `Python 3.9.8`
# 
# ###**Libraries Used:**
# 
# `NumPy`
# 
# `Pandas`
# 
# `Scikit-learn (SKLEARN)`
# 
# <img class="fit-picture"
#      src="https://raw.githubusercontent.com/emirhanai/Emotion-Prediction-with-Semi-Supervised-Learning-of-Machine-Learning-Software-with-RC-Algorithm---By/main/Emotion%20Prediction%20with%20Semi%20Supervised%20Learning%20of%20Machine%20Learning%20Software%20with%20RC%20Algorithm%20-%20By%20Emirhan%20BULUT.png"
#      alt="Emotion Prediction with Semi Supervised Learning of Machine Learning Software with RC Algorithm - Emirhan BULUT">
#      
# ### **Developer Information:**
# 
# Name-Surname: **Emirhan BULUT**
# 
# Contact (Email) : **emirhan@isap.solutions**
# 
# LinkedIn : **[https://www.linkedin.com/in/artificialintelligencebulut/][LinkedinAccount]**
# 
# [LinkedinAccount]: https://www.linkedin.com/in/artificialintelligencebulut/
# 
# Kaggle: **[https://www.kaggle.com/emirhanai][Kaggle]**
# 
# Official Website: **[https://www.emirhanbulut.com.tr][OfficialWebSite]**
# 
# [Kaggle]: https://www.kaggle.com/emirhanai
# 
# [OfficialWebSite]: https://www.emirhanbulut.com.tr

# > # Introduction to Emotion AI Software
# > * **Import to ML (scikit-learn), Data (Pandas) and Math (NumPy) Libraries**
# > * **Data Loading from 'Kaggle'**

# In[ ]:


#Import to ML (scikit-learn), Data (Pandas) and Math (NumPy) Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

#Data Loading from 'Kaggle'
df = pd.read_csv('../input/emotion-prediction-with-semi-supervised-learning/tweet_emotions.csv')

df.head(5)


# > # *Process*
# > * **Import to ML (scikit-learn) Libraries**
# > * **Data Preprocessing**
# > * **NLP system entegration to Data**
# > * **Model Creating**

# In[ ]:


#Import to ML (scikit-learn) Libraries
from sklearn.linear_model import RidgeClassifier #RidgeClassifier
from sklearn.semi_supervised import SelfTrainingClassifier #SelfTrainingClassifier

#Data Preprocessing
X_train, X_test, y_train, y_test = train_test_split(df.content, 
                                                    df.sentiment,
                                                    test_size=0.0007000000000000001,
                                                    random_state=25,
                                                    shuffle=True)
#NLP system entegration to Data    
X_CountVectorizer = CountVectorizer(stop_words='english')

X_train_counts = X_CountVectorizer.fit_transform(X_train)

X_TfidfTransformer = TfidfTransformer()

X_train_tfidf = X_TfidfTransformer.fit_transform(X_train_counts)

#Model Creating
model_semi = SelfTrainingClassifier(RidgeClassifier())

model_semi.fit(X_train_tfidf, y_train)


# > # *Prediction*

# In[ ]:


#Data of Prediction
text = """I quite like him. 
I'm so in love with him and my heart flutters when I see him. 
I love her so much!"""

text = [text]

text_counts = X_CountVectorizer.transform(text)

#Prediction Processing
prediction = model_semi.predict(text_counts)

f"Prediction is {prediction[0]}"

