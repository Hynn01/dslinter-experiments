#!/usr/bin/env python
# coding: utf-8

# # IRS Practical 8
# > 19BCE245 - Aayush Shah
# 
# - Text Classification using naive Bayesian approach

# In[ ]:


import numpy as np, pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB,BernoulliNB,GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, accuracy_score
sns.set()


# In[ ]:


# Load the dataset
data = fetch_20newsgroups()
# Get the text categories
text_categories = data.target_names
# define the training set
train_data = fetch_20newsgroups(subset="train", categories=text_categories)
# define the test set
test_data = fetch_20newsgroups(subset="test", categories=text_categories)


# In[ ]:


#Let’s find out how many classes and samples we have:

print("We have {} unique classes".format(len(text_categories)))
print("We have {} training samples".format(len(train_data.data)))
print("We have {} test samples".format(len(test_data.data)))


# In[ ]:


# So, this is a 20-class text classification problem with n_train = 11314 training samples (text sentences) and n_test = 7532 test samples (text sentences).

# Let’s visualize the 5th training sample:

# let’s have a look as some training data
print(test_data.data[5])


# - The next step consists of building the Naive Bayes classifier and finally training the model.In our example, we will convert the collection of text documents (train and test sets) into a matrix of token counts
# 
# ## Using MulitnomialNB Model

# In[ ]:


# Build the model
model = make_pipeline(TfidfVectorizer(), MultinomialNB())
# Train the model using the training data
model.fit(train_data.data, train_data.target)
# Predict the categories of the test data
predicted_categories = model.predict(test_data.data)


# In[ ]:


print(np.array(test_data.target_names)[predicted_categories])


# In[ ]:


# plot the confusion matrix
mat = confusion_matrix(test_data.target, predicted_categories)
fig, ax = plt.subplots(figsize=(12,12)) 
sns.heatmap(mat.T, square = True, annot=True, fmt = "d", xticklabels=train_data.target_names,yticklabels=train_data.target_names)
plt.xlabel("true labels")
plt.ylabel("predicted label")
plt.show()
print("The accuracy is {}".format(accuracy_score(test_data.target, predicted_categories)))


# ## Using BernoulliNB Model

# In[ ]:


# Build the model
model = make_pipeline(TfidfVectorizer(), BernoulliNB())
# Train the model using the training data
model.fit(train_data.data, train_data.target)
# Predict the categories of the test data
predicted_categories = model.predict(test_data.data)


# In[ ]:


# plot the confusion matrix
mat = confusion_matrix(test_data.target, predicted_categories)
fig, ax = plt.subplots(figsize=(12,12)) 
sns.heatmap(mat.T, square = True, annot=True, fmt = "d", xticklabels=train_data.target_names,yticklabels=train_data.target_names)
plt.xlabel("true labels")
plt.ylabel("predicted label")
plt.show()
print("The accuracy is {}".format(accuracy_score(test_data.target, predicted_categories)))

