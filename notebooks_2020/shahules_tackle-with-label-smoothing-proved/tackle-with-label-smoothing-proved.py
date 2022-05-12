#!/usr/bin/env python
# coding: utf-8

# ![](https://media.giphy.com/media/3o6MbqwVaVfbxMJTTq/giphy.gif)

# 
# Contents
# * [Threat in using augmented Data]()
# * [Tackle with label smoothing]()
# * [Expirement with label smoothing]()
# 
# 

# ## <font color='blue' size='4'>Please leave an upvote if you like this Notebook</font>
# 

# In this competition we are trying to use many data augmentation methods, especially to generate more data on languages other than English and train on those examples.
# 
# The methods used until now are :
# - [Translating data using Google API](https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification/discussion/141377)
# - [NLP Albumentation](https://www.kaggle.com/shonenkov/nlp-albumentations)
# - [Parellel corpus](https://www.kaggle.com/shonenkov/hack-with-parallel-corpus)
# 
# I went through all of these methods and tried some of the them.No doubt, they are pretty useful. But after reading the recent notebook from [Alex Shonenkov](https://www.kaggle.com/shonenkov) I got a link to the idea of label smoothing which I am going to discuss here.All these data augmentation methods have a possible hidden threat in them which is why many of us are still in a dilemma whether to use them or not.
# 
# The main possible threat is that there is a chance that these methods produce data samples with incorrect labels.I tried using Google translate API to some of the toxic comments and I saw that sometimes it is censoring the toxic content in some comments or reducing the level of toxicity in it( I saw it happen using google API). This can happen to other augmentation methods too...
# 
# So, what's the solution to this problem?
# - Don't bother to use the augmentation methods and lose the edge that it gives.
# - Use Label Smoothing.
# 
# **So let's do label smoothing.**

# ## <font size='3' color='red'>Say hello to Label Smoothing!</font>
# 
# When we apply the cross-entropy loss to a classification task, we’re expecting true labels to have 1, while the others 0. In other words, we have no doubts that the true labels are true, and the others are not. Is that always true in our case? As said above, the translation or other augmentation methods have done some mistakes. They might have different criteria. They might make some mistakes. As a result, the ground truth labels we have had perfect beliefs on are possibly wrong. In our case, a sample with no toxicity will have had perfect belief as toxic or vice versa...
# 
# One possible solution to this is to relax our confidence on the labels. For instance, we can slightly lower the loss target values from 1 to, say, 0.9. And naturally, we increase the target value of 0 for the others slightly as such. This idea is called label smoothing.
# 
# [![label-smoothing.png](https://i.postimg.cc/cHvh4hpW/label-smoothing.png)](https://postimg.cc/VrcnKqsZ)
# In tensorflow,

# In[ ]:


import tensorflow as tf

tf.keras.losses.binary_crossentropy(
    y_true, y_pred,
    from_logits=False,
    label_smoothing=0
)


# If label_smoothing is nonzero, smooth the labels towards 1/num_classes:
# 
# 
# `new_onehot_labels = onehot_labels * (1 – label_smoothing) + label_smoothing / num_classes`
# 
# What does this mean?
# 
# Well, say in our case were training a model for binary classification,Our labels are 0 — Non-toxic, 1 — toxic.
# 
# Now, say you  label_smoothing = 0.2
# 
# Using the equation above, we get:
# 
# `new_labels = [0 1] * (1 — 0.2) + 0.2 / 2 =[0 1]*(0.8) + 0.1`
# 
#  `new_labels = [0.1 ,0.9]`
# 

# OR you can simply convert your one hot encoded array of floating point numbers to the “nuanced” version
# 

# In[ ]:



train[np.where(y == 0)] = 0.1
train[np.where(y == 1)] = 0.9


# ## <font size='3' color='red'>What does this do ?</font>

# One should see the values 0 and 1 as simply true and false. Taking values that vary slightly from the classic values, they are a more nuanced way to describing the data 0.1 could be viewed as: “there is a very low chance this data is <one of two classes>” whereas 0.9
# 
# is, of course, a high chance.
# 
# Using these “nuanced” labels, the cost of an incorrect prediction is slightly lower than using “hard” labels resulting in a smaller gradient. While this intuition helped me understand why it could be a good idea, I was not entirely convinced it would work in an application because the loss is lowered for all wrong classifications. So I decided to read some about some research done on the subject.
# 
# A table copied from [When Does Label Smoothing Help?](https://arxiv.org/pdf/1906.02629.pdf)
# 
# ![](https://rickwierenga.com/assets/images/smoothing.png)

# ## <font size='4' color='red'>Expirement with Label Smoothing</font>

# Now,it's time to expirement with the same.For the sake our expirement I will take the data from `sklearn.datasets`. We need a binary classificatin example,so I selected `breast cancer` dataset which contains approx 500 samples with 30 features.

# In[ ]:





# In[ ]:


from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
plt.style.use('ggplot')
epochs=10


# Let's load the dataset and see

# In[ ]:


data= load_breast_cancer()


# In[ ]:


print("The dataset contains {} samples with {} features".format(data['data'].shape[0],data['data'].shape[1]))


# - Let's do the train and test split now,(80:20)

# In[ ]:


train_X,test_X,train_y,test_y=train_test_split(data['data'],data['target'],random_state=77)


# Now for our expirement I will relabel some of the negative cases (target 0 ) as positive cases (target 1).

# In[ ]:


train_y[:40]=1


# Next,let's build our simple NN model

# In[ ]:


def model():
    inp = tf.keras.Input(shape=(30))
    
    x= tf.keras.layers.Dense(64,activation='relu')(inp)
    x=tf.keras.layers.Dense(32,activation='relu')(x)
    x=tf.keras.layers.Dense(1,'sigmoid')(x)
    
    model=tf.keras.Model(inp,x)
    model.compile(optimizer='Adam',loss='binary_crossentropy',metrics=['accuracy'])
    return model
    
    


# In[ ]:


model=model()
model.summary()


# Now, Fit ,evaluate and predict **without label smoothing**

# In[ ]:



history=model.fit(train_X,train_y,epochs=epochs)


# 

# In[ ]:


plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.plot(np.arange(1,epochs+1),history.history['loss'],color='red',alpha=1)
plt.gca().set_xlabel("Epochs")
plt.gca().set_ylabel("loss")
plt.gca().set_title("Loss without Label smoothing")

plt.subplot(1,2,2)
plt.plot(np.arange(1,epochs+1),history.history['accuracy'],color='red',alpha=1)
plt.gca().set_xlabel("Epochs")
plt.gca().set_ylabel("accuracy")
plt.gca().set_title("accuracy without Label smoothing")


plt.show()


# In[ ]:


y_pre = model.predict(test_X)
print(accuracy_score(test_y,np.round(y_pre)))


# You can see that the model has an accuracy of only 67% in the test set.
# - Now let's add the trick and train our model.Train our model **with label smoothing**

# In[ ]:


def label_smoothing(y_true,y_pred):
    
     return tf.keras.losses.binary_crossentropy(y_true,y_pred,label_smoothing=0.1)


# In[ ]:


def model():
    inp = tf.keras.Input(shape=(30))
    
    x= tf.keras.layers.Dense(64,activation='relu')(inp)
    x=tf.keras.layers.Dense(32,activation='relu')(x)
    x=tf.keras.layers.Dense(1,'sigmoid')(x)
    
    model=tf.keras.Model(inp,x)
    
    model.compile(optimizer='Adam',loss=label_smoothing,metrics=['accuracy'])
    return model
    


# In[ ]:


model=model()
history=model.fit(train_X,train_y,epochs=epochs)


# In[ ]:


plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.plot(np.arange(1,epochs+1),history.history['loss'],color='red',alpha=1)
plt.gca().set_xlabel("Epochs")
plt.gca().set_ylabel("loss")
plt.gca().set_title("Loss with Label smoothing")

plt.subplot(1,2,2)
plt.plot(np.arange(1,epochs+1),history.history['accuracy'],color='red',alpha=1)
plt.gca().set_xlabel("Epochs")
plt.gca().set_ylabel("accuracy")
plt.gca().set_title("accuracy with Label smoothing")


plt.show()


# Now,the moment of truth !
# - let's predict on the same test set and see how much accuracy does it give...

# In[ ]:


y_pre = model.predict(test_X)
print(accuracy_score(test_y,np.round(y_pre)))


# viola..! The accuracy just increased  just by adding **label smoothing**.

# I hope this helps,I encourage you to try this and comment your results below.
# ## <font color='blue' size='4'>Please leave an upvote if you like this Notebook</font>
# 
# References :
# - https://arxiv.org/pdf/1906.02629.pdf
# - https://www.flixstock.com/label-smoothing-an-ingredient-of-higher-model-accuracy/
