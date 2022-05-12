#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import collections
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import spacy
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, f1_score, classification_report, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier


# ### Get large spaCy model optimised for CPU

# In[ ]:


get_ipython().system('python -m spacy download en_core_web_lg')


# ### Load spaCy model

# In[ ]:


nlp = spacy.load('en_core_web_lg')


# ### Unpack the datasets

# In[ ]:


get_ipython().system('unzip -n ../input/jigsaw-toxic-comment-classification-challenge/train.csv.zip')
get_ipython().system('unzip -n ../input/jigsaw-toxic-comment-classification-challenge/test.csv.zip')
get_ipython().system('unzip -n ../input/jigsaw-toxic-comment-classification-challenge/test_labels.csv.zip')
get_ipython().system('unzip -n ../input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv.zip')


# ### Load the originally datasets into pandas dataframes

# In[ ]:


train = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')
sample = pd.read_csv('./sample_submission.csv')


# ### Look at the sample submission: what shall be done?
# It seems we are supposed to submit probabilities for each label:

# In[ ]:


sample


# ### Get column names

# In[ ]:


target_columns = list(sample.columns.drop('id'))
target_columns


# # Analyse training data

# In[ ]:


train.head()


# ### Look at the label distribution within the training dataset

# In[ ]:


# The low mean indicates that only few comments are actually labelled:
train.describe()


# In[ ]:


# The counts show an imbalanced dataset, both between labels but also with no label at all:
train[target_columns].sum()


# In[ ]:


# Let's add some more labels for the purpose of the analysis
train['non-toxic'] = 1-train[target_columns].max(axis=1)

train['toxicity_type_defined'] = train[['insult','obscene','identity_hate','threat']].max(axis=1)

train['toxic_undefined'] = 0
train.loc[(train['toxicity_type_defined']==0)&(train['toxic']==1),'toxic_undefined'] = 1

train['soft_toxic'] = 0
train.loc[(train['toxicity_type_defined']==1)&(train['toxic']==0),'soft_toxic'] = 1

analysis_columns = target_columns + ['non-toxic', 'toxic_undefined', 'soft_toxic']


# In[ ]:


label_counts = train[analysis_columns].sum()
plt.figure(figsize=(20, 10))
ax = sns.barplot(x=label_counts.index, y=label_counts.values)
ax.set_yscale("log")
ax.tick_params(labelsize=15)


# In[ ]:


# Visualise any correlations between labels:
# It seems there are some strong correlations between labels
heatmap_data = train[target_columns]
plt.figure(figsize=(10, 10))
ax = sns.heatmap(heatmap_data.corr(), cmap='coolwarm', annot=True)
ax.tick_params(labelsize=10)


# ### Confirming hypotheses on the data

# In[ ]:


# Confirm that all severly toxic comments (n=1595) are toxic:
train.loc[train['severe_toxic']==1,'toxic'].sum()


# ---
# # Prepare a more balanced dataset

# In[ ]:


# Pragmatic approach: Let's take 11% of the unlabelled comments and add everything else:
# later we can try with the original set to see what delivers better results
train_balanced = train[train['non-toxic']==1].sample(frac=0.11)
train_balanced = train_balanced.append(train.loc[train['non-toxic']==0])
train_balanced = train_balanced.sample(frac=1)


# In[ ]:


# The new dataset is somewhat more balanced, with the exception of 'threat'-labelled comments:
# we do not oversample or take other measures for now
label_counts = train_balanced[target_columns+['non-toxic']].sum()
plt.figure(figsize=(20, 10))
ax = sns.barplot(x=label_counts.index, y=label_counts.values)
ax.set_yscale("log")
ax.tick_params(labelsize=15)


# ---
# # Run pre-processing on comments

# In[ ]:


all_tokens = []
lemmas = []
nonstop_tokens = []
total_number_of_tokens = []
number_of_sentences = []
number_of_puncts = []
glove_vector = []

for doc in nlp.pipe(train_balanced['comment_text'].astype('unicode').values, batch_size=50):
    if doc.has_annotation("DEP"):
        all_tokens.append([t.lower_ for t in doc])
        nonstop_tokens.append([t.lower_ for t in doc if t.is_alpha and not t.is_stop])
        lemmas.append([t.lemma_ for t in doc if t.is_alpha and not t.is_stop])
        total_number_of_tokens.append(doc.__len__())
        number_of_sentences.append(sum(1 for _ in doc.sents))
        number_of_puncts.append(sum(1 for t in doc if t.is_punct))
        glove_vector.append(doc.vector.tolist())
    else:
        all_tokens.append(None)
        nonstop_tokens.append(None)
        lemmas.append(None)
        total_number_of_tokens.append(None)
        number_of_sentences.append(None)
        number_of_puncts.append(None)
        glove_vector.append(None)

train_balanced['all_tokens'] = all_tokens
train_balanced['nonstop_tokens'] = nonstop_tokens
train_balanced['lemmas'] = lemmas
train_balanced['total_number_of_tokens'] = total_number_of_tokens
train_balanced['number_of_sentences'] = number_of_sentences
train_balanced['number_of_puncts'] = number_of_puncts
train_balanced['glove_vector'] = glove_vector


# ---
# # Analyse comments

# ### Analyse length of comments

# In[ ]:


length = train_balanced['total_number_of_tokens']
length.hist(bins=100);


# In[ ]:


# Look at number of sentences in a comment:
# it seems that toxic comments are shorter on average.
# We do not do anything about this for the time being. But it indicates value in considering number of sentences when classifying a comment.
mc_length = []

for label in analysis_columns:
    mc_length.append(train_balanced.loc[train_balanced[label]==1,'number_of_sentences'].median())

plt.figure(figsize=(20, 10))
ax = sns.barplot(x=analysis_columns, y=mc_length)
ax.tick_params(labelsize=15)


# In[ ]:


# Look at median number of tokens in a comment:
# it seems that toxic comments are shorter on average.
# We do not do anything about this for the time being. But it indicates value in considering number of tokens when classifying a comment.
mc_length = []

for label in analysis_columns:
    mc_length.append(train_balanced.loc[train_balanced[label]==1,'total_number_of_tokens'].median())

plt.figure(figsize=(20, 10))
ax = sns.barplot(x=analysis_columns, y=mc_length)
ax.tick_params(labelsize=15)


# In[ ]:


log_max = np.log(train_balanced['total_number_of_tokens'].max())
train_balanced['log_tokens'] = train_balanced['total_number_of_tokens']**(1/log_max)
upper = train_balanced['log_tokens'].max()
train_balanced['log_tokens_normalised'] = (train_balanced['log_tokens'])/(upper)
#helpful_log_max_root = df.Helpful_Votes**(1/log_maxlog_max


# In[ ]:


for label in target_columns:
    word_list = list(train_balanced.loc[train_balanced[label]==1, 'nonstop_tokens'].explode())
    most_common = collections.Counter(word_list).most_common(20)
    words = [w for w,_ in most_common]
    counts = [c for _,c in most_common]
    plt.figure(figsize=(20, 10))
    ax = sns.barplot(x=words, y=counts)
    ax.set_title(f'Label = {label}', fontsize=15)
    ax.tick_params(labelsize=15)
    ax.set_xticklabels(ax.get_xticklabels(),rotation = 30)


# 

# ---
# # Split into train/validation sets for analysis purposes

# In[ ]:


# Note 1: train_balanced is already shuffled, so we we split off just the last 40% (40% as there are very few instances of some labels)
val_border = int(len(train_balanced)*0.6)
validation_set = train_balanced[val_border:].copy()


# In[ ]:


# Note 2: due to the severely imbalanced label distribution, we shall not compare effectiveness using labels like 'threat'
compare_set = ['toxic', 'insult', 'obscene']


# ---
# # Prepare TF-IDF vectors

# In[ ]:


vectorizer = TfidfVectorizer(ngram_range=(1, 5), max_features=5000)
vectorizer.fit(train_balanced['comment_text'])
vectorizer


# In[ ]:


vectorizer.get_params()


# In[ ]:


x_train = vectorizer.transform(train_balanced['comment_text'])


# ---
# # Analyse three classifiers with two different encodings (TF-IDF and Glove)
# - Logistic Regression
# - SVM
# - Multi-Layer Perceptron
# 
# Note that we do not use any chaining (i.e. not leveraging multi-label correlation).

# In[ ]:


# Logistic regression based on TF-IDF

lr_classifier = LogisticRegression(solver='liblinear')
average_roc = 0
    
for label in target_columns:
    lr_classifier.fit(x_train[:val_border], train_balanced[label][:val_border])
    predictions = lr_classifier.predict(x_train[val_border:])
    print(f'Label = {label}')
    print(classification_report(validation_set[label], predictions))
    print(f'AUC: {roc_auc_score(validation_set[label], predictions)}')
    average_roc += roc_auc_score(validation_set[label], predictions)
                                 
print(f'Average AUC: {average_roc/len(target_columns)}\n\n')


# In[ ]:


# Logistic regression based on Glove

lr_classifier = LogisticRegression(solver='liblinear')
average_roc = 0
    
for label in target_columns:
    lr_classifier.fit(train_balanced['glove_vector'].to_list()[:val_border], train_balanced[label][:val_border])
    predictions = lr_classifier.predict(train_balanced['glove_vector'].to_list()[val_border:])
    print(f'Label = {label}')
    print(classification_report(validation_set[label], predictions))
    print(f'AUC: {roc_auc_score(validation_set[label], predictions)}')
    average_roc += roc_auc_score(validation_set[label], predictions)
                                 
print(f'Average AUC: {average_roc/len(target_columns)}\n\n')


# In[ ]:


# SVM-classifier based on TF-IDF

sv_classifier = SVC()
average_roc = 0
    
for label in target_columns:
    sv_classifier.fit(x_train[:val_border], train_balanced[label][:val_border])
    predictions = sv_classifier.predict(x_train[val_border:])
    print(f'Label = {label}')
    print(classification_report(validation_set[label], predictions))
    print(f'AUC: {roc_auc_score(validation_set[label], predictions)}')
    average_roc += roc_auc_score(validation_set[label], predictions)
                                 
print(f'Average AUC: {average_roc/len(target_columns)}\n\n')


# In[ ]:


# SVM-classifier based on Glove

sv_classifier = SVC()
average_roc = 0
    
for label in target_columns:
    sv_classifier.fit(train_balanced['glove_vector'].to_list()[:val_border], train_balanced[label][:val_border])
    predictions = sv_classifier.predict(train_balanced['glove_vector'].to_list()[val_border:])
    #probabilities = sv_classifier.predict_proba(train_balanced['glove_vector'].to_list()[val_border:])[:,1]
    print(f'Label = {label}')
    print(classification_report(validation_set[label], predictions))
    print(f'AUC: {roc_auc_score(validation_set[label], predictions)}')
    average_roc += roc_auc_score(validation_set[label], predictions)
                                 
print(f'Average AUC: {average_roc/len(target_columns)}\n\n')


# In[ ]:


# Multi-Layer Perceptron Classifier based on TF-IDF

mlp_classifier = MLPClassifier(max_iter=500)
average_roc = 0
    
for label in target_columns:
    mlp_classifier.fit(x_train[:val_border], train_balanced[label][:val_border])
    predictions = mlp_classifier.predict(x_train[val_border:])
    print(f'Label = {label}')
    print(classification_report(validation_set[label], predictions))
    print(f'AUC: {roc_auc_score(validation_set[label], predictions)}')
    average_roc += roc_auc_score(validation_set[label], predictions)
                                 
print(f'Average AUC: {average_roc/len(target_columns)}\n\n')


# In[ ]:


# Multi-Layer Perceptron Classifier based on Glove

mlp_classifier = MLPClassifier(max_iter=500)
average_roc = 0
    
for label in target_columns:
    mlp_classifier.fit(train_balanced['glove_vector'].to_list()[:val_border], train_balanced[label][:val_border])
    predictions = mlp_classifier.predict(train_balanced['glove_vector'].to_list()[val_border:])
    print(f'Label = {label}')
    print(classification_report(validation_set[label], predictions))
    print(f'AUC: {roc_auc_score(validation_set[label], predictions)}')
    average_roc += roc_auc_score(validation_set[label], predictions)
                                 
print(f'Average AUC: {average_roc/len(target_columns)}\n\n')


# In[ ]:


# Multi-Layer Perceptron Classifier based on Glove

mlp_classifier = MLPClassifier(max_iter=1000)
    

mlp_classifier.fit(train_balanced['glove_vector'].to_list()[:val_border], train_balanced[target_columns][:val_border])
predictions = mlp_classifier.predict(train_balanced['glove_vector'].to_list()[val_border:])
probabilities = mlp_classifier.predict_proba(train_balanced['glove_vector'].to_list()[val_border:])

average_roc = 0
i = 0
for label in target_columns:
    print(f'Label = {label}')
    prediction = [p[i] for p in predictions]
    probability = [p[i] for p in probabilities]
    print(classification_report(validation_set[label], prediction))
    print(f'AUC: {roc_auc_score(validation_set[label], prediction)}\n\n')
    average_roc += roc_auc_score(validation_set[label], prediction)
    i += 1
    
print(f'Average AUC: {average_roc/len(target_columns)}\n\n')


# --- 
# # Prepare data for submission
# Make use of the LR based on TF-IDF, though the MLP/GloVe-based classifier proved best:
# Time required for GloVe vectorisation of the full dataset and MLP training is prohibitive.

# ## Prepare TF-IDF vectors for full train and test datasets
# Note that in this notebook version, the full training dataset is used instead of the train_balanced.
# Reason for this choice are repeated tests, which have shown better results with the original, unbalanced training dataset.

# In[ ]:


vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=10000, min_df=2, max_df=0.7)
vectorizer.fit(train['comment_text'])
vectorizer.get_params()


# In[ ]:


x_train = vectorizer.transform(train['comment_text'])
x_test = vectorizer.transform(test['comment_text'])


# ## Train classifier per label, and predict probabilities

# In[ ]:


# Logistic Regression with linear solver

lr_classifier = LogisticRegression(solver='liblinear')
    
for label in target_columns:
    lr_classifier.fit(x_train, train[label])
    test[label] = lr_classifier.predict_proba(x_test)[:,1]


# In[ ]:


# Look at the data

test.head()


# ## Save results in file for submission

# In[ ]:


# Save columns based on sample submission file
test[sample.columns.to_list()].to_csv('mlp_submission.csv', index=False)


# In[ ]:




