#!/usr/bin/env python
# coding: utf-8

# # [1] Overview
# This notebook follows the [Bag of Words Meets Bags of Popcorn](https://www.kaggle.com/c/word2vec-nlp-tutorial/overview). The tutorial covers sentiment analysis; specifically, text-processing and utilizing Bag-Of-Words and Word2Vec. Bag-Of-Words is the traditional method of understanding semantic relationships among words, while Word2Vec is Google's new deep-learning approach.  
# 
# Sentiment analysis is determining the meaning underlying a language. It is challenging as people express themselves in obscure ways using sarcasm or ambiguity.
# 
# The goal of the kernel is to create a model that can accurately determine if a movie review is positive or negative. The dataset consists of 100,000 IMDB positive and negative multi-paragraph movie reviews. Submissions are juded on area under the ROC curve.

# # [2] Bag of Words

# In[ ]:


import os
os.chdir('/kaggle/input/utilities')
from data import DataLoader
os.chdir("/kaggle/working/")

from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
import re

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from gensim.models import word2vec

import warnings
warnings.filterwarnings("ignore")

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',    level=logging.INFO)


# # [2.1] The Data

# In[ ]:


dl = DataLoader('/kaggle/input/word2vec-nlp-tutorial')
dl.files


# In[ ]:


dl.load_data()


# In[ ]:


train = pd.read_csv(
    'labeledTrainData.tsv',
    header=0,
    delimiter="\t",
    quoting=3)


# In[ ]:


train.head()


# In[ ]:


print(train['review'][0][500:1000])


# # [2.2] Preprocessing
# 
# We need to remove HTML tags, abbreviations, punctiations, etc.  
# 
# We can use BeautifulSoup to remove all HTML content.

# In[ ]:


example = BeautifulSoup(train['review'][0])
print(example.get_text()[500:1000])


# To deal with punctuation, numbers, and stop words, we can use NLTK and regular expressions. Note that some punctuation may be give us a better sentiment insight, such as "!!!" or ":-(", but for simplicity we will remove all punctuation.

# In[ ]:


letters_only = re.sub("[^a-zA-Z]",
                     " ",
                     example.get_text())


# In[ ]:


print(letters_only[500:1000])


# Finally, we need to deal with "stop words". These words include "a", "and", "is", etc. NLTK can help us with this.

# In[ ]:


lower_case = letters_only.lower()        # Convert to lower case
words = lower_case.split()               # Split into words
print(stopwords.words('english'))


# In[ ]:


words = [w for w in words if not w in stopwords.words("english")]
print(words)


# Putting all the steps above together, we have the following function.

# In[ ]:


def review_to_words(raw_review):
    # Function to convert a raw review to a string of words
    # The input is a single string (a raw movie review), and 
    # the output is a single string (a preprocessed movie review)
    #
    # 1. Remove HTML
    review_text = BeautifulSoup(raw_review).get_text() 
    #
    # 2. Remove non-letters        
    letters_only = re.sub("[^a-zA-Z]", " ", review_text) 
    #
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()                             
    #
    # 4. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))                  
    # 
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]   
    #
    # 6. Join the words back into one string separated by space, 
    # and return the result.
    return( " ".join( meaningful_words ))  


# In[ ]:


clean_review_example = review_to_words( train["review"][0] )
print(clean_review_example)


# In[ ]:


# Get the number of reviews based on the dataframe column size
num_reviews = train["review"].size

# Initialize an empty list to hold the clean reviews
clean_train_reviews = ['']*num_reviews

# Loop over each review; create an index i that goes from 0 to the length
# of the movie review list 
for i, review in enumerate(DataLoader.progressbar(train["review"])):
    # Call our function for each one, and add the result to the list of
    # clean reviews
    clean_train_reviews[i] = review_to_words(review)


# In[ ]:


clean_train_reviews[0]


# To convert our reviews into a numeric representation we will use the Bag of Words approach. The Bag of Word approach simply maps each word in the review to how many times it was repeated in that review.

# In[ ]:


vectorizer = CountVectorizer(
    analyzer = "word",
    tokenizer = None,
    preprocessor = None,
    stop_words = None,
    max_features = 5000)

train_data_features = vectorizer.fit_transform(clean_train_reviews).toarray()


# In[ ]:


print(train_data_features.shape)


# The vectorizer creates feature vector for us, where the colums are unique words and the rows are the reviews. For each review, the feature vector tells us how many times each of the 5000 words occurred.

# In[ ]:


vocab = vectorizer.get_feature_names()
print(vocab[0:50])


# In[ ]:


dist = np.sum(train_data_features, axis=0)
for tag, count in zip(vocab[0:20], dist[0:20]):
    print(count, tag)


# # [2.3] Training
# With these features we will be training a Random Forest

# In[ ]:


forest = RandomForestClassifier(n_estimators = 100)
forest = forest.fit( train_data_features, train["sentiment"] )


# Clean the test set

# In[ ]:


# Read the test data
test = pd.read_csv("testData.tsv", header=0, delimiter="\t",                    quoting=3 )


# Create an empty list and append the clean reviews one by one
num_reviews = test["review"].size
clean_test_reviews = [' ']*num_reviews

for i, review in enumerate(DataLoader.progressbar(test["review"])):
    clean_test_reviews[i] = review_to_words(review)


# Make the predictions

# In[ ]:


# Get a bag of words for the test set, and convert to a numpy array
test_data_features = vectorizer.transform(clean_test_reviews)
test_data_features = test_data_features.toarray()

# Use the random forest to make sentiment label predictions
result = forest.predict(test_data_features)

# Copy the results to a pandas dataframe with an "id" column and
# a "sentiment" column
output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )

# Use pandas to write the comma-separated output file
output.to_csv( "Bag_of_Words_model.csv", index=False, quoting=3 )


# # [3] Word2Vec
# Word2vec was published by Google in 2013. It is a neural network implementation that learns distributed represenetations for words. Word2Vec is more impressive because it learns quickly relative to other models.
# 
# It does not need labels to create meaninful representations, and if the network is given enough training data it produces word vectors with intruiging characteristics. Words with similar meanings appear in clusters, and clusters are spaced such that some word relationships, such as analogies, can be reproduced using vector math. The famous example is that, with highly trained word vectors, "king - man + woman = queen." 

# # [3.1] The Data

# In[ ]:


# Read data from files 
train = pd.read_csv(
    "labeledTrainData.tsv",
    header=0, 
    delimiter="\t",
    quoting=3 )
test = pd.read_csv(
    "testData.tsv",
    header=0, delimiter="\t",
    quoting=3 )
unlabeled_train = pd.read_csv(
    "unlabeledTrainData.tsv",
    header=0,
    delimiter="\t",
    quoting=3 )

# Verify the number of reviews that were read (100,000 in total)
print(f'Read {train["review"].size} labeled train reviews,' +       '{test["review"].size} labeled test reviews, and' +       '{unlabeled_train["review"].size} unlabeled reviews')


# # [3.2] Preprocessing
# Similar to the Bag of Words processing, but we want to keep the stop words (because the algorithm relies on them for better understandings)
# 
# Also, we want a specific input format. Word2Vec expects single sentences, each one as a list of words. In other words, the input format is a list of lists.

# In[ ]:


# Load the punkt tokenizer
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

def review_to_wordlist( review, remove_stopwords=False ):
    # Function to convert a document to a sequence of words,
    # optionally removing stop words.  Returns a list of words.
    #
    # 1. Remove HTML
    review_text = BeautifulSoup(review).get_text()
    #  
    # 2. Remove non-letters
    review_text = re.sub("[^a-zA-Z]"," ", review_text)
    #
    # 3. Convert words to lower case and split them
    words = review_text.lower().split()
    #
    # 4. Optionally remove stop words (false by default)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    #
    # 5. Return a list of words
    return(words)

# Define a function to split a review into parsed sentences
def review_to_sentences( review, tokenizer, remove_stopwords=False ):
    # Function to split a review into parsed sentences. Returns a 
    # list of sentences, where each sentence is a list of words
    #
    # 1. Use the NLTK tokenizer to split the paragraph into sentences
    raw_sentences = tokenizer.tokenize(review.strip())
    #
    # 2. Loop over each sentence
    sentences = []
    for raw_sentence in raw_sentences:
        # If a sentence is empty, skip it
        if len(raw_sentence) > 0:
            # Otherwise, call review_to_wordlist to get a list of words
            sentences.append( review_to_wordlist( raw_sentence,               remove_stopwords ))
    #
    # Return the list of sentences (each sentence is a list of words,
    # so this returns a list of lists
    return sentences


# Next, we want a specific input format. Word2Vec expects single sentences, each one as a list of words. In other words, the input format is a list of lists.

# In[ ]:


sentences = []

for review in DataLoader.progressbar(train["review"], 'Parsing training set: ', 'Complete!'):
    sentences += review_to_sentences(review, tokenizer)

for review in DataLoader.progressbar(unlabeled_train["review"], 'Parsing unlabeled set: ', 'Complete!'):
    sentences += review_to_sentences(review, tokenizer)


# In[ ]:


print(len(sentences),
     '\n\n',
     sentences[0])


# # [3.3] Training

# In[ ]:


# Set values for various parameters
num_features = 300    # Word vector dimensionality                      
min_word_count = 40   # Minimum word count                        
num_workers = 4       # Number of threads to run in parallel
context = 10          # Context window size                                                                                    
downsampling = 1e-3   # Downsample setting for frequent words

# Initialize and train the model (this will take some time)
print('Training model...')
model = word2vec.Word2Vec(sentences, workers=num_workers,             size=num_features, min_count = min_word_count,             window = context, sample = downsampling)

# If you don't plan to train the model any further, calling 
# init_sims will make the model much more memory-efficient.
model.init_sims(replace=True)

# It can be helpful to create a meaningful model name and 
# save the model for later use. You can load it later using Word2Vec.load()
model_name = "300features_40minwords_10context"
model.save(model_name)


# Distinguish which word is most dissimilar from the others.

# In[ ]:


model.doesnt_match("man woman child kitchen".split())


# In[ ]:


model.doesnt_match("dog paris london france".split())


# Get insight into the model's word clusters

# In[ ]:


model.most_similar("man")


# In[ ]:


model.most_similar("awful")

