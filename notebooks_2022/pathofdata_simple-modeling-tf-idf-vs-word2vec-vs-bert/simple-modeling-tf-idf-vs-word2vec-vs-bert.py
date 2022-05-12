#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from transformers import AutoTokenizer,TFBertModel
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
import re
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
import gensim
from nltk.data import find
from tensorflow.keras.layers import LSTM, Dense, Masking, Dropout
from tensorflow.keras import Sequential, Input, optimizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import LearningRateSchedule
import numpy as np
import pandas as pd

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


train_data_path = '/kaggle/input/nlp-getting-started/train.csv'
test_data_path = '/kaggle/input/nlp-getting-started/test.csv'
submission_path = '/kaggle/input/nlp-getting-started/sample_submission.csv'

original_data = pd.read_csv(train_data_path)
test_data = pd.read_csv(test_data_path)
submission_data = pd.read_csv(submission_path)


# In[ ]:


display(original_data.head())
display(original_data.info(show_counts=True))


# ## Dataset Pre-processing
# Simple data cleaning functions that may be useful

# In[ ]:



# %20 is the URL encoding of space, let's replace them with '_'
def re_encode_space(input_string):
    return None if pd.isna(input_string) else input_string.replace('%20', '_')


# Let's try to find hastags
import re

def find_hash_tags(input_string):
    hash_tags = re.findall(r"#(\w+)", str(input_string))
    return ','.join(hash_tags)


# Let's turn hashtags to normal words
def re_encode_hashtags(input_string):
    return None if pd.isna(input_string) else input_string.replace('#', '')


# Let's remove URLs from the tweets
def remove_links(input_string):
    res = input_string
    urls = re.findall(r'(https?://[^\s]+)', res)
    for link in urls:
        res = res.strip(link)
    return res


# Let's remove the state abbreviations
def state_renaming(input_string):

    states = {
        'AK': 'Alaska',
        'AL': 'Alabama',
        'AR': 'Arkansas',
        'AZ': 'Arizona',
        'CA': 'California',
        'CO': 'Colorado',
        'CT': 'Connecticut',
        'DC': 'District_of_Columbia',
        'DE': 'Delaware',
        'FL': 'Florida',
        'GA': 'Georgia',
        'HI': 'Hawaii',
        'IA': 'Iowa',
        'ID': 'Idaho',
        'IL': 'Illinois',
        'IN': 'Indiana',
        'KS': 'Kansas',
        'KY': 'Kentucky',
        'LA': 'Louisiana',
        'MA': 'Massachusetts',
        'MD': 'Maryland',
        'ME': 'Maine',
        'MI': 'Michigan',
        'MN': 'Minnesota',
        'MO': 'Missouri',
        'MS': 'Mississippi',
        'MT': 'Montana',
        'NC': 'North_Carolina',
        'ND': 'North_Dakota',
        'NE': 'Nebraska',
        'NH': 'New_Hampshire',
        'NJ': 'New_Jersey',
        'NM': 'New_Mexico',
        'NV': 'Nevada',
        'NY': 'New_York',
        'OH': 'Ohio',
        'OK': 'Oklahoma',
        'OR': 'Oregon',
        'PA': 'Pennsylvania',
        'RI': 'Rhode_Island',
        'SC': 'South_Carolina',
        'SD': 'South_Dakota',
        'TN': 'Tennessee',
        'TX': 'Texas',
        'UT': 'Utah',
        'VA': 'Virginia',
        'VT': 'Vermont',
        'WA': 'Washington',
        'WI': 'Wisconsin',
        'WV': 'West_Virginia',
        'WY': 'Wyoming'
    }

    result = input_string
    
    if isinstance(input_string, str):
        input_candidates = input_string.split(', ')
        
        if len(input_candidates) > 1:
            for candidate in input_candidates:
                if candidate in states.keys():
                    result = states[candidate]
                
    if input_string in states.keys():
        result = states[input_string]

    return result


# In[ ]:


# Let's wrap the preprocessing functions so it's easier to
# process both train and test dataset
def preprocess_data(input_data):
    input_df = input_data.copy()
    input_df['keyword'] = input_df['keyword'].map(re_encode_space)
    input_df['keyword'].fillna('Missing', inplace=True)
    input_df['hashtags'] = input_df['text'].map(find_hash_tags)
    input_df['text'] = input_df['text'].map(re_encode_hashtags)
    input_df['text'] = input_df['text'].map(remove_links)
    input_df['location'] = input_df['location'].map(state_renaming)
    return input_df


# In[ ]:


original_data = preprocess_data(original_data)
test_data = preprocess_data(test_data)


# In[ ]:


# We notice that keyword field has missing values
# We notice location has missing values

# Let's Visualize the keyword field in a word cloud to get an idea of what it is
import matplotlib.pyplot as plt
from wordcloud import WordCloud

keyword_words = str(original_data['keyword']
    .dropna()
    .unique()
    .tolist()
)

location_words = str(original_data['location']
    .dropna()
    .unique()
    .tolist()
)

hashtag_words = str(original_data['hashtags']
    .dropna()
    .unique()
    .tolist()
)

keyword_wordcloud = WordCloud(
    background_color='white',
    stopwords=None,
    max_words=200,
    max_font_size=40, 
    random_state=42
).generate(keyword_words)

location_wordcloud = WordCloud(
    background_color='white',
    stopwords=None,
    max_words=200,
    max_font_size=40, 
    random_state=42
).generate(location_words)

hashtag_wordcloud = WordCloud(
    background_color='white',
    stopwords=None,
    max_words=200,
    max_font_size=40, 
    random_state=42
).generate(hashtag_words)

fig, ax = plt.subplots(1,3, figsize=(16,9), constrained_layout=True)
ax[0].set_title("Keywords")
ax[0].imshow(keyword_wordcloud)
ax[0].axis(False)
ax[1].set_title('Location')
ax[1].imshow(location_wordcloud)
ax[1].axis(False)
ax[2].set_title('Hashtags')
ax[2].imshow(hashtag_wordcloud)
ax[2].axis(False)
plt.show()


# In[ ]:


def na_proportions(data, column_name, as_pct):
    if column_name in data.columns:
        na_counts = len(data[pd.isna(data[column_name])])
        non_na_counts = len(data[~pd.isna(data[column_name])])
    else:
        na_counts = None
        non_na_counts = None
        
    if as_pct:
        na_counts /= data.shape[0]
        non_na_counts /= data.shape[0]
    return (column_name, na_counts, non_na_counts)


# In[ ]:


keyword_props = na_proportions(data=original_data, column_name='keyword', as_pct=True)
print(f'The {keyword_props[0]} variable has: NA={keyword_props[1]:.3f} NON-NA={keyword_props[2]:.3f}')
location_props = na_proportions(data=original_data, column_name='location', as_pct=True)
print(f'The {location_props[0]} variable has: NA={location_props[1]:.3f} NON-NA={location_props[2]:.3f}')


# ## First approach: TF-IDF + Logistic Regression
# A simple model to use as baseline

# In[ ]:


X = original_data['text']
y = original_data['target']


# In[ ]:



skf = StratifiedKFold(n_splits=5)
train_average_score = 0
validation_average_score = 0
validation_oof_predictions = np.zeros((len(X)))

for fold_n, (train_idx, test_idx) in enumerate(skf.split(X, y)):
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    
    model = Pipeline([
        ('Encoder', TfidfVectorizer(max_features=None)),
        ('Clf', LogisticRegression(penalty='l2', C=1, solver='liblinear'))
    ])
    
    model.fit(X_train, y_train)
    
    train_predictions = model.predict_proba(X_train)[:, 1]
    validation_predictions = model.predict_proba(X_test)[:, 1]
    
    train_score = roc_auc_score(y_train, train_predictions)
    validation_score = roc_auc_score(y_test, validation_predictions)
    
    train_average_score += train_score / 5
    validation_average_score += validation_score / 5
    validation_oof_predictions[test_idx,] = (validation_predictions > 0.5).astype(int)
    
    print(f'Fold: {fold_n}, train auc: {train_score:.3f}, validation auc: {validation_score:.3f}')
print(f'Train average: {train_average_score:.3f}, validation average: {validation_average_score:.3f}')
print(f'OOF Accuracy Score: {accuracy_score(y, validation_oof_predictions)}')


# A simple linear model appears to have an idea of what is going on.
# We'll use these scores as baseline when trying more advanced models

# ## Second approach: Word2Vec embeddings and a simple LSTM-based network
# Let's slightly increase the complexity of the model:
# * Word2Vec embeddings using a pre-trained model for vector representation of words
# * LSTM based model as a classifier

# In[ ]:


# The tokenizer is responsible to turn a string of words
# into a list of tokens (words) for which we'll get their
# vector representation (embeddings)
tknzr = TweetTokenizer(
    preserve_case=False,
    reduce_len=True,
    strip_handles=True,
)


def tokenize_tweets(tokenizer, input_text):
    tokens = list(tokenizer.tokenize(input_text))
    tokens = [re.sub('[^A-Za-z0-9]+', '', i) for i in tokens]
    return tokens

original_data['tokens'] = original_data['text']
original_data['tokens'] = original_data['tokens'].apply(lambda x: tokenize_tweets(tknzr, x))

test_data['tokens'] = test_data['text'].apply(lambda x: tokenize_tweets(tknzr, x))


# In[ ]:


# Our dataset is quite small so if we train the word2vec model the
# resulting embeddings will be poor in quality. Therefore we use a
# pre-trained model
word2vec_sample = str(find('models/word2vec_sample/pruned.word2vec.txt'))
word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_sample, binary=False)
features = 300


# In[ ]:


# We'll pad all embeddings to match the length of the biggest tweet
# in order to account for the variability in tweet length
# Later on the model is going to mask the padded values, so that
# they won't influence the result
max_tweet_length = max(original_data['tokens'].apply(lambda x: len(x)).max(), 
                       test_data['tokens'].apply(lambda x: len(x)).max())


# In[ ]:


# Let's compute the embeddings for every word that the pre-trained model
# has in its vocabulary.
def vectorize_tokens(data_, vec_model, max_seq, num_features):
    data_in = data_.copy()
    # List to save all text embeddings
    all_vectors = []
    # Iterate over each text
    for _, row in data_in.iterrows():
        # Initialize a 2D matrix with zeros. Equivalent to 0 padding
        # in the 1st dimension, to accomondate for variable text length
        text_vectors = np.zeros((max_seq, num_features))
        # If the word exists in the model vocabulary add its embeddings
        # else keep the zeros as unknown words
        for i, item in enumerate(row['tokens']):
            try:
                text_vectors[i, :] = vec_model[item]
            except:
                continue
        all_vectors.append(text_vectors)
    
    return all_vectors


# In[ ]:


original_data['vectors'] = vectorize_tokens(data_=original_data, 
                                            vec_model=word2vec_model, 
                                            max_seq=max_tweet_length, 
                                            num_features=features)

test_data['vectors'] = vectorize_tokens(data_=test_data, 
                                        vec_model=word2vec_model, 
                                        max_seq=max_tweet_length, 
                                        num_features=features)


# In[ ]:


# Logic to have the training dataset as a 3D array of (text_count, max_sequence_length, embedding_size)
X = np.asarray(original_data['vectors'].tolist()).astype(np.float32)
y = np.asarray(original_data['target'].tolist()).astype(np.float32)

test_array = np.asarray(test_data['vectors'].tolist()).astype(np.float32)


# In[ ]:



model = Sequential([
    Input(shape=(max_tweet_length, features)),
    Masking(),
    LSTM(32),
    Dropout(0.1),
    Dense(1, activation='sigmoid')
])


skf = StratifiedKFold(n_splits=5)
train_average_score = 0
validation_average_score = 0
validation_oof_predictions = np.zeros((len(X)))

# It's a good practice to predict the test set on every fold
# And average the predictions over the folds
averaged_test_predictions = np.zeros((test_array.shape[0]))

# It's standard practice to use Stratified k-fold cross validation
# so we're also using it here
for fold_n, (train_idx, test_idx) in enumerate(skf.split(X, y)):
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    
    # Re-compile the model at every fold to "reset" it
    model.compile(optimizer='adam', loss='binary_crossentropy')
    
    # Simple training strategy, hyper-parameters haven't been tuned
    model.fit(x=X_train, y=y_train, batch_size=32, epochs=3)
    
    train_predictions = model.predict(X_train)
    validation_predictions = model.predict(X_test)
    
    train_score = roc_auc_score(y_train, train_predictions)
    validation_score = roc_auc_score(y_test, validation_predictions)
    
    train_average_score += train_score / 5
    validation_average_score += validation_score / 5
    validation_oof_predictions[test_idx,] = (validation_predictions > 0.5).astype(int).flatten()
    
    print(f'Fold: {fold_n}, train auc: {train_score:.3f}, validation auc: {validation_score:.3f}')
    
    test_predictions = model.predict(test_array).flatten()
    averaged_test_predictions += test_predictions / 5
    
print(f'Train average: {train_average_score:.3f}, validation average: {validation_average_score:.3f}')
print(f'OOF Accuracy Score: {accuracy_score(y, validation_oof_predictions)}')


# The new model has a much better performance than the baseline
# In addition, the train-validation overfitting is greatly reduced
# increasing the probability of getting the model to generalize well

# ## Third approach: Bert Embeddings
# Let's use the Powerful Bert Model and see if it performs better than our word2vec model.
# We'll follow the code from the TF tutorial

# In[ ]:


X = original_data['text'].tolist()
y = np.asarray(original_data['target'].tolist()).astype(np.float32)

test_array = test_data['text'].tolist()


# In[ ]:


bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
bert = TFBertModel.from_pretrained('bert-base-uncased')

X = bert_tokenizer(
    text=X,
    add_special_tokens=True,
    max_length=max_tweet_length,
    truncation=True,
    padding=True, 
    return_tensors='tf',
    return_token_type_ids = False,
    return_attention_mask = True,
    verbose = True)

test_array = bert_tokenizer(
    text=test_array,
    add_special_tokens=True,
    max_length=max_tweet_length,
    truncation=True,
    padding=True, 
    return_tensors='tf',
    return_token_type_ids = False,
    return_attention_mask = True,
    verbose = True)


# In[ ]:



epochs = 5
steps_per_epoch = X['input_ids'].numpy().shape[0]
num_train_steps = steps_per_epoch * epochs
num_warmup_steps = int(0.1*num_train_steps)

class BertLrSchedule(LearningRateSchedule):

    @tf.function
    def __init__(self, initial_learning_rate, num_warmups, num_train_steps):
        self.overshoot = 1000
        self.initial_learning_rate = initial_learning_rate
        self.num_warmups = num_warmups
        self.num_train_steps = num_train_steps
        self.angle_warm = self.initial_learning_rate / self.num_warmups
        self.angle_decay = - self.initial_learning_rate /             (self.num_train_steps - self.num_warmups - self.overshoot)
    
    @tf.function
    def __call__(self, step):
        if step <= self.num_warmups:
            return (tf.cast(step, tf.float32) + 1) * self.angle_warm
        else:
            return self.initial_learning_rate + (tf.cast(step, tf.float32) - self.num_warmups + 1 + self.overshoot) * self.angle_decay
        
        
schedule = BertLrSchedule(initial_learning_rate=2e-5, 
                          num_warmups=num_warmup_steps, 
                          num_train_steps=num_train_steps)

steps = np.arange(num_train_steps)
lrs = [schedule.__call__(i) for i in steps]

import matplotlib.pyplot as plt
fig, ax = plt.subplots(1,1)
ax.plot(steps, lrs)
ax.set_xlabel('Train steps')
ax.set_ylabel('Learning Rate')
ax.set_title('Bert scheduler')
plt.show()


# In[ ]:



def build_bert_classifier():
    input_ids = Input(shape=(max_tweet_length,), dtype=tf.int32, name="input_ids")
    input_mask = Input(shape=(max_tweet_length,), dtype=tf.int32, name="attention_mask")
    embeddings = bert(input_ids,attention_mask = input_mask)['pooler_output']
    net = tf.keras.layers.Dropout(0.1)(embeddings)
    net = tf.keras.layers.Dense(128, activation='relu', name='pre-clf')(net)
    net = tf.keras.layers.Dropout(0.1)(net)
    net = tf.keras.layers.Dense(1, activation=None, name='classifier')(net)
    return tf.keras.Model(inputs=[input_ids, input_mask], outputs=net)

skf = StratifiedKFold(n_splits=5)
train_average_score = 0
validation_average_score = 0
validation_oof_predictions = np.zeros((len(X['input_ids'].numpy())))

loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
optimizer = Adam(learning_rate=schedule)
epochs = 5

# It's a good practice to predict the test set on every fold
# And average the predictions over the folds
averaged_test_predictions = np.zeros((test_array['input_ids'].shape[0]))

# It's standard practice to use Stratified k-fold cross validation
# so we're also using it here
for fold_n, (train_idx, test_idx) in enumerate(skf.split(X['input_ids'].numpy(), y)):
    X_train_ids = X['input_ids'].numpy()[train_idx]
    X_train_att = X['attention_mask'].numpy()[train_idx]
    y_train = y[train_idx]
    
    X_test_ids = X['input_ids'].numpy()[test_idx]
    X_test_att = X['attention_mask'].numpy()[test_idx]
    y_test = y[test_idx]
    
    # Re-build the model at every fold to "reset" it
    model = build_bert_classifier()
    model.layers[2].trainable = True
    
    model.compile(optimizer=optimizer,
                  loss=loss)
    
    model.fit(x={'input_ids':X_train_ids,'attention_mask':X_train_att}, 
              y=y_train, batch_size=32, epochs=epochs)
    
    train_predictions = model.predict({'input_ids':X_train_ids,'attention_mask':X_train_att})
    validation_predictions = model.predict({'input_ids':X_test_ids,'attention_mask':X_test_att})
    
    train_score = roc_auc_score(y_train, train_predictions)
    validation_score = roc_auc_score(y_test, validation_predictions)
    
    train_average_score += train_score / 5
    validation_average_score += validation_score / 5
    validation_oof_predictions[test_idx,] = (validation_predictions > 0.5).astype(int).flatten()
    
    print(f'Fold: {fold_n}, train auc: {train_score:.3f}, validation auc: {validation_score:.3f}')
    
    test_predictions = model.predict({'input_ids':test_array['input_ids'],
                                      'attention_mask':test_array['attention_mask']}).flatten()
    averaged_test_predictions += test_predictions / 5
    
print(f'Train average: {train_average_score:.3f}, validation average: {validation_average_score:.3f}')
print(f'OOF Accuracy Score: {accuracy_score(y, validation_oof_predictions)}')


# In[ ]:


# Let's save the test predictions and submit them
submission_data['target'] = (averaged_test_predictions > 0.5).astype(int)
submission_data.to_csv('model_predictions.csv', index=False)
submission_data.head()

