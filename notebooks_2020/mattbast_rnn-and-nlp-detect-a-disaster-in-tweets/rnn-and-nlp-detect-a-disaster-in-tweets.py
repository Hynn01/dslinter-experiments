#!/usr/bin/env python
# coding: utf-8

# Recently I have been learning about RNNs (Recurrent Neural Networks) and NLP (Natural Language Processing) through Andrew Ngs excellent "Sequence Models" course on Coursera ([link](https://www.coursera.org/learn/nlp-sequence-models)). I wanted to have a go implementing a language model using this knowledge and Tensorflow v2.
# 
# I picked the "Real or Not? NLP with Disaster Tweets" ([link](https://www.kaggle.com/c/nlp-getting-started/overview)) getting started competition for its straight forward task (label tweets as either reporting a disaster or not reporting disaster) and the size of the dataset (large enough to contain enough information for the model but not so much that there will be a lot of processing).
# 
# First things first then, let's load the libraries.

# In[ ]:


import numpy as np
import pandas as pd
import tensorflow as tf

import re
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

import matplotlib.pyplot as plt


# In[ ]:


pd.set_option('display.max_colwidth', -1)


# ## Load Data
# 
# Now I'll load the training dataset. 

# In[ ]:


train_data = pd.read_csv(
    '/kaggle/input/nlp-getting-started/train.csv', 
    usecols=['text', 'target'], 
    dtype={'text': str, 'target': np.int64}
)

len(train_data)


# In[ ]:


train_data['text'].head().values


# And load the test dataset for later.

# In[ ]:


test_data = pd.read_csv(
    '/kaggle/input/nlp-getting-started/test.csv', 
    usecols=['text', 'id'], 
    dtype={'text': str, 'id': str}
)


# ## Mislabelled examples
# 
# There are a number of examples in the training dataset that are mislabelled. The keyword can be used to find these.
# 
# Thanks to Dmitri Kalyaevs whose notebook is where I found to do this: https://www.kaggle.com/dmitri9149/transformer-svm-semantically-identical-tweets

# In[ ]:


indices = [4415, 4400, 4399,4403,4397,4396, 4394,4414, 4393,4392,4404,4407,4420,4412,4408,4391,4405]
train_data.loc[indices]


# In[ ]:


train_data.loc[indices, 'target'] = 0


# In[ ]:


indices = [6840,6834,6837,6841,6816,6828,6831]
train_data.loc[indices]


# In[ ]:


train_data.loc[indices, 'target'] = 0


# In[ ]:


indices = [601,576,584,608,606,603,592,604,591, 587]
train_data.loc[indices]


# In[ ]:


train_data.loc[indices, 'target'] = 1


# In[ ]:


indices = [3913,3914,3936,3921,3941,3937,3938,3136,3133,3930,3933,3924,3917]
train_data.loc[indices]


# In[ ]:


train_data.loc[indices, 'target'] = 0


# In[ ]:


indices = [246,270,266,259,253,251,250,271]
train_data.loc[indices]


# In[ ]:


train_data.loc[indices, 'target'] = 0


# In[ ]:


indices = [6119,6122,6123,6131,6160,6166,6167,6172,6212,6221,6230,6091,6108]
train_data.loc[indices]


# In[ ]:


train_data.loc[indices, 'target'] = 0


# In[ ]:


indices = [7435,7460,7464,7466,7469,7475,7489,7495,7500,7525,7552,7572,7591,7599]
train_data.loc[indices]


# In[ ]:


train_data.loc[indices, 'target'] = 0


# ## Split training dataset
# 
# To see if the model overfits the data during training I will take a slice of the training data as a validation dataset.

# In[ ]:


val_data = train_data.tail(1500)
train_data = train_data.head(6113)


# ## Clean text
# 
# As with all datasets, text based data needs a bit of cleaning to. Some common cleaning steps are:
# 
# - **Removing Noise:** So remove things that hold little meaning like URLs and html tags in the text. Punctuation is also usually removed at this stage though the Tensorflow tokenizer I use later does this by default so I leave out the logic at this step.
# 
# - **Remove Stopwords:** Certain words like "a", "the" and "are" are very common and hold little meaning for a sentence. Removing them speeds up training and helps with accuracy.
# 
# - **Stemming or Lemmatization:** Many words are derived from a root or stem word. For example words like "working" and "worked" stem from the word "work". Reverting all words to their stem can help in some tasks though for this task I found that it made very little difference.
# 
# I've defined a few functions to perform this cleaning.

# In[ ]:


def remove_url(sentence):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'', sentence)


# In[ ]:


def remove_at(sentence):
    url = re.compile(r'@\S+')
    return url.sub(r'', sentence)


# In[ ]:


def remove_html(sentence):
    html = re.compile(r'<.*?>')
    return html.sub(r'', sentence)


# In[ ]:


def remove_emoji(sentence):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    
    return emoji_pattern.sub(r'', sentence)


# In[ ]:


def remove_stopwords(sentence):
    words = sentence.split()
    words = [word for word in words if word not in stopwords.words('english')]
    
    return ' '.join(words)


# In[ ]:


stemmer = SnowballStemmer('english')

def stem_words(sentence):
    words = sentence.split()
    words = [stemmer.stem(word) for word in words ]
    
    return ' '.join(words)


# For speed I have wrapped all of these cleaning functions into one. This is applied to all three datasets.

# In[ ]:


def clean_text(data):
    data['text'] = data['text'].apply(lambda x : remove_url(x))
    data['text'] = data['text'].apply(lambda x : remove_at(x))
    data['text'] = data['text'].apply(lambda x : remove_html(x))
    data['text'] = data['text'].apply(lambda x : remove_emoji(x))
    data['text'] = data['text'].apply(lambda x : remove_stopwords(x))
    data['text'] = data['text'].apply(lambda x : stem_words(x))
    
    return data


# In[ ]:


train_data = clean_text(train_data)
val_data = clean_text(val_data)
test_data = clean_text(test_data)

train_data['text'].head().values


# ## Encode sentences
# 
# A model will not understand what to do with a string representing a sentence. Instead it needs to be converted into an array of numbers representing the sentence. This is where a tokenizer is needed. A simple tokenizer will assign a number index to every unique word so that the model can treat it like a categorical value. Fundamentally this is what Tensorflows tokenizer does at this stage of preprocessing. 
# 
# To achieve this I'll define a couple of functions. The first defines the tokenizer by taking all the sentences from all three datasets and assigning an index number to every word present in the sentences. All three datasets are used here to ensure the tokenizer vocabulary includes all the words present in the tweets. I have done a small piece of analysis in the appendix section at the bottom of this notebook to show how many words are in one of the datasets but not the others.
# 
# The second function uses the tokenizer to encode all the sentences into an array of index numbers representing the sentence. The second function also pads the sentences with zeros so that they are all the same size as the longest sentence in the training dataset. Note that the tokenizer reserves the index zero for this purpose.

# In[ ]:


def define_tokenizer(train_sentences, val_sentences, test_sentences):
    sentences = pd.concat([train_sentences, val_sentences, test_sentences])
    
    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(sentences)
    
    return tokenizer
    
def encode(sentences, tokenizer):
    encoded_sentences = tokenizer.texts_to_sequences(sentences)
    encoded_sentences = tf.keras.preprocessing.sequence.pad_sequences(encoded_sentences, padding='post')
    
    return encoded_sentences


# In[ ]:


tokenizer = define_tokenizer(train_data['text'], val_data['text'], test_data['text'])

encoded_sentences = encode(train_data['text'], tokenizer)
val_encoded_sentences = encode(val_data['text'], tokenizer)
encoded_test_sentences = encode(test_data['text'], tokenizer)


# The tokeniser provides some interesting information about the sentences it encodes. To get the index number assigned to a word I can look up the word in the tokenizers word index (which is just a python dict with the words as keys and the index numbers as values).

# In[ ]:


tokenizer.word_index['disaster']


# The word index can also be used to find out how many words are in the vocabulary.

# In[ ]:


len(tokenizer.word_index)


# As well as some configurations that are used to tokenize sentences such as whether the tokenizer changes all characters to lowercase, what split it performs to get the words from the sentences and what characters it filters out.

# In[ ]:


print('Lower: ', tokenizer.get_config()['lower'])
print('Split: ', tokenizer.get_config()['split'])
print('Filters: ', tokenizer.get_config()['filters'])


# ## Import GloVe Embedding
# 
# While I could train my own word embedding for the model it might help to use a pre-trained word embedding. This enables me to take advantage of an embedding that has ungone more rogourous training. Additionally it will also include words that I may not have in my training dataset (but may appear in the test dataset) which helps with overfitting.
# 
# The first thing to do then is to load the embedding. I'll use GloVe for this task.

# In[ ]:


embedding_dict = {}

with open('../input/glove-global-vectors-for-word-representation/glove.6B.100d.txt','r') as f:
    for line in f:
        values = line.split()
        word = values[0]
        vectors = np.asarray(values[1:],'float32')
        embedding_dict[word] = vectors
        
f.close()


# To ensure the encoding of the tokenizer and the embeddings are synchronized I use the below function to update the encoded words in the embedding with the encoding from the tokenizer.

# In[ ]:


num_words = len(tokenizer.word_index) + 1
embedding_matrix = np.zeros((num_words, 100))

for word, i in tokenizer.word_index.items():
    if i > num_words:
        continue
    
    emb_vec = embedding_dict.get(word)
    
    if emb_vec is not None:
        embedding_matrix[i] = emb_vec


# ## Define pipeline
# 
# With the sentences encoded they can now be prepared to be fed into the model. Tensorflow provides an api to format data in its own format. While data can be inserted in a more common format (such as numpy arrays), tensorflow seems to prefer its own format and provides a few handy bits of functionality as incentives. 
# 
# Firstly then I will convert the encoded sentences and labels into tensors.

# In[ ]:


tf_data = tf.data.Dataset.from_tensor_slices((encoded_sentences, train_data['target'].values))


# Now the data is in the tensorflow format a few handy methods can be added to improve the training. This includes shuffling the data per training step, processing the next batch of data for training while the current batch of data is training and defining each batch as a padded batch.

# In[ ]:


def pipeline(tf_data, buffer_size=100, batch_size=32):
    tf_data = tf_data.shuffle(buffer_size)    
    tf_data = tf_data.prefetch(tf.data.experimental.AUTOTUNE)
    tf_data = tf_data.padded_batch(batch_size, padded_shapes=([None],[]))
    
    return tf_data

tf_data = pipeline(tf_data, buffer_size=1000, batch_size=32)


# In[ ]:


print(tf_data)


# A similar pipeline is defined for the validation dataset. The difference is the lack of shuffling to speed up the validation.

# In[ ]:


tf_val_data = tf.data.Dataset.from_tensor_slices((val_encoded_sentences, val_data['target'].values))


# In[ ]:


def val_pipeline(tf_data, batch_size=1):        
    tf_data = tf_data.prefetch(tf.data.experimental.AUTOTUNE)
    tf_data = tf_data.padded_batch(batch_size, padded_shapes=([None],[]))
    
    return tf_data

tf_val_data = val_pipeline(tf_val_data, batch_size=len(val_data))


# In[ ]:


print(tf_val_data)


# ## Train Model
# 
# Now I am ready to define and train the model. Firstly I will define the model:
# 
# - **Embedding layer:** To enable the model to gain an understanding of a words meaning it will need a word embedding. This embedding creates a number of generic features against each word that could represent anything from a words gender to whether it is has a positive or negative sentiment. This also enables the model to identify relationships between words. This page from Tensorflows documentation contains a good diagram of what one might look like: [link](https://www.tensorflow.org/tutorials/text/word_embeddings#word_embeddings_2).
# 
# - **RNN layer:** An RNN layer is a bit complex to explain in it's entirety in this notebook. Check out the course link at the top of this notebook for a full run through of the theory. Tensorflow offers all three of the major RNN layers; simple RNN, GRU and LSTM. I tried all of these layers and found that the LSTM layer worked best here.
# 
# - **Dense Layer:** This final layer takes the LSTM output and applies a class to the sentence i.e. a 1 if the sentence is reporting a real disaster or 0 if not.

# In[ ]:


embedding = tf.keras.layers.Embedding(
    len(tokenizer.word_index) + 1,
    100,
    embeddings_initializer = tf.keras.initializers.Constant(embedding_matrix),
    trainable = True
)


# In[ ]:


model = tf.keras.Sequential([
    embedding,
    tf.keras.layers.SpatialDropout1D(0.2),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, dropout=0.2, recurrent_dropout=0.2)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])


# Then compile the model defining the training function (adam) and the loss function (log loss). I have also added a metrics parameter so that the models accuracy is printed per epoch.

# In[ ]:


model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(0.0001),
    metrics=['accuracy', 'Precision', 'Recall']
)


# To avoid the model stepping over the optimum I'll add learning rate decay logic to reduce the learning rate if the loss plateaus for two or more epochs. I'll also add early stopping if loss hasn't fallen for five epochs. This saves some processing.

# In[ ]:


callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', patience=2, verbose=1),
    tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, verbose=1),
]


# Finally begin training the model.

# In[ ]:


history = model.fit(
    tf_data, 
    validation_data = tf_val_data,
    epochs = 50,
    callbacks = callbacks
)


# ## Evaluate
# 
# Let's take a look at the models output to get an idea how it did. The quickest and easiest evaluation method is to take a look at the metrics produced by the model. The final metrics can be extracted using the evaluate method. Since this competition uses an F1 score to rank submissions it may be worth having a look at it on the training dataset.

# In[ ]:


metrics = model.evaluate(tf_val_data)

precision = metrics[2]
recall = metrics[3]
f1 = 2 * (precision * recall) / (precision + recall)

print('F1 score: ' + str(f1)) 


# Additionally the metrics produced per epoch when the model was training can be visualised to get a better idea for how the training went.

# In[ ]:


fig, axs = plt.subplots(1, 4, figsize=(20, 5))

axs[0].set_title('Loss')
axs[0].plot(history.history['loss'], label='train')
axs[0].plot(history.history['val_loss'], label='val')
axs[0].legend()

axs[1].set_title('Accuracy')
axs[1].plot(history.history['accuracy'], label='train')
axs[1].plot(history.history['val_accuracy'], label='val')
axs[1].legend()

axs[2].set_title('Precision')
axs[2].plot(history.history['Precision'], label='train')
axs[2].plot(history.history['val_Precision'], label='val')
axs[2].legend()

axs[3].set_title('Recall')
axs[3].plot(history.history['Recall'], label='train')
axs[3].plot(history.history['val_Recall'], label='val')
axs[3].legend()


# It is also worth having a look at which sentences the model got wrong. To do this the model needs to produce predictions for the training dataset. This involves a slightly different pipeline.

# In[ ]:


predictions = model.predict(tf_val_data)
predictions = np.concatenate(predictions).round().astype(int)

val_data['predictions'] = predictions


# First take a look at the false postives (when the model thought there was a disaster in the tweet but there was not).

# In[ ]:


false_positives = val_data[(val_data['predictions'] == 1) & (val_data['target'] == 0)]

print('Count of false positives: ' + str(len(false_positives)))


# In[ ]:


false_positives.head(10)


# And then do the same with the false negatives (when the model didn't think there was a disaster in a tweet when in fact there was).

# In[ ]:


false_negatives = val_data[(val_data['predictions'] == 0) & (val_data['target'] == 1)]

print('Count of false negatives: ' + str(len(false_negatives)))


# In[ ]:


false_positives.tail(10)


# ## Submission
# 
# With the model trained it now takes a few more steps to load the test data and use the model to label the test sentences as either disaster or no disaster. First convert the data to a tensorflow dataset and apply the pipeline methods. The pipeline has been adjusted slightly to account for not wanting any shuffling and the different shape of the input (no label).

# In[ ]:


tf_test_data = tf.data.Dataset.from_tensor_slices((encoded_test_sentences))


# In[ ]:


def test_pipeline(tf_data, batch_size=1):        
    tf_data = tf_data.prefetch(tf.data.experimental.AUTOTUNE)
    tf_data = tf_data.padded_batch(batch_size, padded_shapes=([None]))
    
    return tf_data

tf_test_data = test_pipeline(tf_test_data)


# Then use the model to apply labels to the test data.

# In[ ]:


predictions = model.predict(tf_test_data)


# The model outputs a probability per sentence. The easy way to set a threshold of 0.5 (i.e. if the probability is less than 0.5 set the label to 0 and vice versa) is to use the round method.

# In[ ]:


predictions = np.concatenate(predictions).round().astype(int)


# Write the submission to a csv file.

# In[ ]:


submission = pd.DataFrame(data={'target': predictions}, index=test_data['id'])
submission.index = submission.index.rename('id')
submission.to_csv('submission.csv')


# In[ ]:


submission.head()


# ## Appendix
# 
# ### Word mismatch
# 
# Earlier in the notebook I mentioned that the training, validation and test datasets are likely to contain words that the other datasets do not. If the model is only trained on the words in the training dataset there may be an overfitting problem when the model tries to read words it doesn't recognise in the validation and the test datasets.
# 
# The below function takes two datasets and counts how words are matching and not matching to see how severe the issue is.

# In[ ]:


def compare_words(train_words, test_words):
    unique_words = len(np.union1d(train_words, test_words))
    matching = len(np.intersect1d(train_words, test_words))
    not_in_train = len(np.setdiff1d(test_words, train_words))
    not_in_test = len(np.setdiff1d(train_words, test_words))
    
    print('Count of unique words in both arrays: ' + str(unique_words))
    print('Count of matching words: ' + str(matching))
    print('Count of words in first array but not in second: ' + str(not_in_test))
    print('Count of words in second array but not first: ' + str(not_in_train))


# In[ ]:


compare_words(encoded_sentences, val_encoded_sentences)


# In[ ]:


compare_words(encoded_sentences, encoded_test_sentences)


# ### Randomly initialised word embedding
# 
# Before I started using the imported GloVe word embeddings I trained the model using a randomly initialised embedding. It performed quite well (though not as well as the GloVe embedding) so I thought I would keep the code down here.

# In[ ]:


# model = tf.keras.Sequential([
#     tf.keras.layers.Embedding(len(tokenizer.word_index) + 1, 200),
#     tf.keras.layers.Bidirectional(tf.keras.layers.SimpleRNN(64)),
#     tf.keras.layers.Dense(1, activation='sigmoid')
# ])

