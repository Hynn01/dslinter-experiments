#!/usr/bin/env python
# coding: utf-8

# ## Introduction
# 
# Look at the top-score solutions to a random machine learning comptition in kaggle. Most of them are based on or mixture of NN, DL, boosting, or ensemble. The outside of kaggle kingdom is somewhat similar. 
# 
# | ![](https://i.imgur.com/lEgFmmK.png) |
# | --- |
# | <b>Fig.1: Performance of different models on IMDB dataset.</b> [source](https://arxiv.org/abs/2005.00357)| 

# As an example, you can see in the preceding image the non-NN methods' accuracy on sentiment analysis of IMDB reviews dataset froze in 2012. Even the best non-NN solution is a mixture of NB(Naive Bayes) and SVM(Support Vector Machine). This solution was first introduced in a 2012 [paper](https://aclanthology.org/P12-2018.pdf) ***Baselines and Bigrams: Simple, Good Sentiment and Topic Classification*** by Sida I Wang and [Chris manning](https://profiles.stanford.edu/chris-manning) (the most cited NLP scientist according to [Andrew Ng](https://www.youtube.com/watch?v=H343JRrncfc&ab_channel=DeepLearningAI)). To gain more insight, [@jhoward](https://www.kaggle.com/jhoward) made a fantastic [notebook](https://www.kaggle.com/code/jhoward/nb-svm-strong-linear-baseline) based on their proposed method. In this notebook we will check some aspects of that paper in more detail.

# ## The paper in detail
# 
# Many datasets were evaluated for sentiment analysis in the paper but in this notebook we will focus on IMDB movie reviews. This dataset consists of 50k labeled reviews. 
# The authors used l2-regulerized l2-loss SVM with the LIBLINEAR library. Moreover, they used both unigrams and bigrams to form bag of words (BoW) and found adding bigrams always improved the result.
# 
# |![](https://i.imgur.com/Bbf6jIG.png)|
# |---|
# |<b>Fig.2: Accuracy of several models on sentiment analysis of IMDB and two other datasets reported by Wang and Manning</b>. [source](https://aclanthology.org/P12-2018.pdf)|   

# As you can see in the preceding image the authors achieved their best result for IMDB dataset by using NBSVM with the addition of bigrams.
# 
# Moreover, at the end of the paper authors claimed implementing Logistic Regression will yield a very similar result to SVM.

# 

# ## Objectives of the current notebook
# 
# After understanding the main points of the paper it is time to ask our own questions:
# 
# *   Does the **addition of bigrams** actually improve the accuracy of SVM by nearly 3%?
# *   Do SVM and LR yield **similar** results for the IMDB dataset?
# *   Can we improve the result significantly by **just tweaking the hyperparameters**?
# 

# ## Loading and preprocessing the dataset
# 
# We use `pandas` to load the CSV file of our dataset. By a short inspection of the first three reviews, it is evident that we should clean the text.

# In[ ]:


import numpy as np
import pandas as pd

df = pd.read_csv('../input/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv', 
                 encoding='utf-8')

df.head(3)


# Although there are more robust approaches to clean the text from HTML marks and other peculiarities, we employ regex for the sake of simplification. To expand contractions such as would've or shouldn't the `contraction` library is loaded.

# In[ ]:


get_ipython().system('pip install contractions')
import contractions
import re

def preprocessor(text):
    text = contractions.fix(text)
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',
                           text)
    text = (re.sub('[\W]+', ' ', text.lower()) +
            ' '.join(emoticons).replace('-', ''))
    return text


# In[ ]:


df['review'] = df['review'].apply(preprocessor)


# In[ ]:


df.head(3)


# Now the text looks much better.

# In[ ]:


from nltk.stem.porter import PorterStemmer

porter = PorterStemmer()

# Tokenization without stemming
def tokenizer(text):
    return text.split()

# Tokenization and stemming
def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]

# Tokenization and lemmatization
#def tokenizer_spacy(text):
    #return [word.lemma_ for word in nlp(text)]


# Now our text is clean, so we can split our data to train and test sets. On the hosting [website](http://ai.stanford.edu/~amaas/data/sentiment/) of the dataset, it is mentioned that half of the dataset is dedicated to the test set and the other half to the train set. We also obey this rule.

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df['review'].values, df['sentiment'].values, test_size=0.5)


# In[ ]:


len(X_train) # A sanity check for our train-test split


# A convention in many NLP and especially sentiment analysis notebooks is to remove stopwords or reduce the tokens (words) to their stems and lemmas. After reading this nice [paper](https://aclanthology.org/W18-2502/https://aclanthology.org/W18-2502/) I decided to retain stopwords. Another point against removing stopwords is using **tf-idf**. When tf-idf is implemented the frequent words (most of the stopwords) achieve lower weights in the corresponding BoW (bag of words).
# 
# For the sake of the current notebook, several attempts were made in reducing tokens to their stems and lemmas but what I found was an increase in the runtime and no significant improvement in performance.
# 
# **Grid search** is a good way to check and find the optimal hyperparameters. Hear I did the grid search with 5-fold (the default number in scikit-learn) cross-validation.
# 
# To define SVM and LR, the `SGDClassifier` model was selected. This model is very similar to the `LinearSVC` and `LogisticRegression` with **stochastic gradient descent** as the optimization method. 

# ## Making pipelines and implementing classifiers
# 
# By choosing `hinge` or `squared_hinge` as the loss function your `SGDClassifier` turns into a linear SVM. On the other hand, assigning `log` as the loss function of `SGDClassifier` results in a Logistic Regression classifier. I tweaked many parameters using grid search. In the following code block some of the parameters of both LR and SVM are provided:

# In[ ]:


from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier


lr_params = [{'vect__ngram_range': [(1,1), (1, 2)],
              'vect__tokenizer': [tokenizer, tokenizer_porter],
              'clf__penalty': ['l2'],
              'clf__alpha': [1e-7, 1e-8, 1e-9]}
            ]


svm_param= [
            {'clf__alpha': [ 1e-4, 1e-5],
             #'clf__max_iter': [10, 100],
             'vect__ngram_range': [(1,1), (1,2)],
             'vect__tokenizer': [tokenizer, tokenizer_porter]}
           ]

svm_tfidf = Pipeline([
    ('vect', TfidfVectorizer(norm='l2')),
    ('clf', SGDClassifier(loss='squared_hinge', penalty='l2',
                          max_iter=10, tol=None))
])


lr_tfidf = Pipeline([
    ('vect', TfidfVectorizer(norm='l2', stop_words=None)),
    ('clf', SGDClassifier(loss='log', penalty = 'l2',
                          random_state=None,
                          tol=None))
])

gs_svm_tfidf = GridSearchCV(svm_tfidf, svm_param,
                           scoring='accuracy',
                           cv=5,
                           verbose=3,
                           n_jobs=1,
                           return_train_score=True)

gs_lr_tfidf = GridSearchCV(lr_tfidf, lr_params,
                           scoring='accuracy',
                           cv=5,
                           verbose=3,
                           n_jobs=1,
                           return_train_score=True)


# We can now run the grid search to see our models' performance regarding different hyperparameters.

# In[ ]:


gs_svm_tfidf.fit(X_train, y_train)


# Let's see the best parameter set for our SVM model based on grid-search:

# In[ ]:


print(f'Best parameter set for SVM: {gs_svm_tfidf.best_params_}')
print(f'Cross-validation Accuracy of SVM: {gs_svm_tfidf.best_score_:.4f}')


# At last, it is time to see the performance of our hyperparameter-tuned SVM on test set.

# In[ ]:


clf_svm = gs_svm_tfidf.best_estimator_
print(f'Test Accuracy of tuned SVM: {clf_svm.score(X_test, y_test):.4f}')


# The same procedure goes for LR:

# In[ ]:


gs_lr_tfidf.fit(X_train, y_train)


# In[ ]:


print(f'Best parameter set for LR: {gs_lr_tfidf.best_params_}')
print(f'Cross-validation Accuracy of LR: {gs_lr_tfidf.best_score_:.4f}')


# In[ ]:


clf_lr = gs_lr_tfidf.best_estimator_
print(f'Test Accuracy of tuned LR: {clf_lr.score(X_test, y_test):.4f}')


# ## Conclusion
# 
# We could achieve nearly similar accuracy by tweaking hyperparameters and text cleaning for either SVM or LR in comparison to NBSVM. NBSVM (Naive Bayse + Support Vector Machine) was suggested by Wang and Manning in their [paper](https://aclanthology.org/P12-2018.pdf) to improve the SVM model for sentiment analysis.
# 
# We can list other significant results of the current notebook as follow:
# 
# 
# * Unlike the original paper, using bigram in addition to unigram improved the accuracy of SVM and LR, 6% and 4% respectively. (2.2% for SVM in the paper)
# * Reducing tokens to their stems using the Porter method improves the accuracy slightly but increases the computation time 2 to 6-fold.
# * LR and SVM perform similarly in the sentiment analysis of IMDB movie reviews. 
# 

# ## Acknowledgment
# 
# This notebook is highly inspired by a fabulous book: [Machine Learning with PyTorch and Scikit-Learn](https://www.amazon.com/Machine-Learning-PyTorch-Scikit-Learn-learning/dp/1801819319/ref=sr_1_1?crid=24Q14C5F9HP7Z&keywords=machine+learning+with+pytorch+and+scikit-learn&qid=1650722069&sprefix=machine+learning+with+pytorch+and+scikit-learn%2Caps%2C301&sr=8-1)

# 

# ## Further reading
# 
# In my journey through NLP I found some insightful resources which I am eager to share:
# 
# * [Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp3/) is an amazing textbook written by Daniel Jurafsky and James H. Martin. It explains some of the hardest concepts of NLP in plain language. You can download a free preprint version of the book from the provided link.
# 
# * Two papers by [Poria et al](https://arxiv.org/abs/2005.00357) and [Keramatfar and Amirkhani](https://journals.sagepub.com/doi/pdf/10.1177/0165551518761013) are highly helpful to understand historical trends and the current state of sentiment analysis.
# 
# * For R users who want a beginner guide in NLP [Text Mining with R: A Tidy Approach](https://www.tidytextmining.com/) by David Robinson and Julia Silge is a gem of a book. With a concise and simple tone, it is ideal as a stepping-stone to following the field.

# 
