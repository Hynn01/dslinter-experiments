#!/usr/bin/env python
# coding: utf-8

# # TensorFlow Natural Language Processing Guide
# 
# **In this notebook I will demonstrate how to process natural language and predict sentiment in TensorFlow.**
# 
# -------------------------------------------------------------------------
# **Notebook Prerequisites:**
# - Python                               > https://www.kaggle.com/learn/python
# - Data Visualization with Matplotlib   > https://www.kaggle.com/learn/data-visualization
# - Basic Linear Algebra                 > https://www.youtube.com/watch?v=fNk_zzaMoSs&list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab
# - Basic Knowledge of Machine Learning  > https://www.kaggle.com/learn/intro-to-machine-learning
# 
# 
# ---------------------------------------------------------------------------------------------------
# 
# **This notebook will cover the following topics:**
# - Loading datasets from the Tensorflow Datasets API
# - Tokenizing text data into a numerical representation
# - Creating sequences (with padding) out of tokenized data
# - Creating a sequence-aware LSTM model with text embedding
# - Using custom and premade callbacks
# - Evaluating a models performance
# 
# 
# *Note: You should research anything in this notebook that you do not understand. Some links will be provided*

# # Load IMDB Dataset
# 
# We will load the `imdb_reviews` dataset from TensorFlow datasets API. This dataset contains 50,000 movie reviews that are categorized as either positive (1) or negative (0). We will extract just the first 20 words from each review to speed up training.
# 
# Link: https://www.tensorflow.org/datasets/catalog/imdb_reviews

# In[ ]:


#Import libraries
import tensorflow as tf #TensorFlow
import tensorflow_datasets as tfds #Datasets
import numpy as np #linear algebra
import matplotlib.pyplot as plt #Data visualization

#Make sure TensorFlow is version 2.0 or higher
print('TensorFlow Version:', tf.__version__)


# In[ ]:


#Download dataset
#dataset documentation -> https://www.tensorflow.org/datasets/catalog/imdb_reviews
text_data = tfds.load('imdb_reviews', split=['train', 'test']) 


# In[ ]:


#Create empty lists to hold our training and test text and labels
train_text = []
train_labels = []
test_text = []
test_labels = []

#Iterate over the imdb data set and add text + labels to their respective list variables
for row in text_data[0]: #training set
    #takes just the first 20 words from the review
    train_text.append(row['text'].numpy().decode('utf-8').split()[:20]) 
    train_labels.append(row['label'].numpy())
    
for row in text_data[1]: #testing set
    #takes just the first 20 words from the review
    test_text.append(row['text'].numpy().decode('utf-8').split()[:20]) 
    test_labels.append(row['label'].numpy())
    

#prints a samples from the training set
print("FIRST 5 SAMPLES OF TRAINING DATA")
print("============================================================")
for i in range(5):
    review = "POSITIVE" if train_labels[i] == 1 else "NEGATIVE"
    print("REVIEW SENTIMENT:", review)
    print("REVIEW TEXT:", " ".join(train_text[i]))
    print("============================================================")


# # Transform Text into Numerical Representation
# 
# Words have meaning to us but computers can only understand numbers. Because of this, we must somehow transform the words into a numerical representation. The TensorFlow `Tokenizer` class can do just that. The `Tokenizer` maps each new word it encounters (limited by the `num_words` attribute) to an number. From here, the numeric representations need to be padded with the `pad_sequences` function so that all inputs to the future model will be of equal length.
# 
# Links: https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/text/Tokenizer
#        https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/sequence/pad_sequences

# In[ ]:


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

#Creates and fits a TensorFlow Tokenizer
tokenizer = Tokenizer(num_words=10_000, oov_token='<OOV>')
tokenizer.fit_on_texts(train_text)

#Creates sequences of numeric representations of words
training_sequences = tokenizer.texts_to_sequences(train_text)
#pads sequences so they all have the same length
training_sequences = pad_sequences(training_sequences, maxlen=20)

#Process test data in the same way for later evaluation
testing_sequences = tokenizer.texts_to_sequences(test_text)
testing_sequences = pad_sequences(testing_sequences, maxlen=20)

        
#prints a sample of the new sequences
print('PROCESSED TEXT DATA')
print('=========================')
for i in range(5):
    print(training_sequences[i], '\n')


# # Custom Callback
# 
# Callbacks are used in Tensorflow to allow user intervention during model training. A callback can be executed at a number of specific intances during model training. 
# For example: 
# - `on_batch_begin`/`end`
# - `on_epoch_begin`/`end`
# - `on_predict_batch_begin`/`end`
# - `on_predict_begin`/`end`
# - `on_test_batch_begin`/`end`
# - `on_test_begin`/`end`
# - `on_train_batch_begin`/`end`
# - `on_train_begin`/`end`
# 
# We will create `CustomCallback` which will stop the model from training once the model reaches 95% acccuracy on the training set.
# 
# Link: https://keras.io/api/callbacks/

# In[ ]:


from tensorflow.keras.callbacks import Callback

class CustomCallback(Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') > 0.95:
            print("Accuracy over 95%... Stopping training")
            self.model.stop_training = True

my_callback = CustomCallback()


# # Predefined Callback - `LearningRateScheduler`
# 
# There are also a number of predefined callbacks. We will use the `LearningRateScheduler` to dynamically update the learning rate of our optimizer. This predefined callback takes a funtion that updates the learning rate as an argument.
# 
# Link: https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/LearningRateScheduler

# In[ ]:


from tensorflow.keras.callbacks import LearningRateScheduler

#creates a function that updates the learning rate based on the epoch number
def scheduler(epoch, lr):
    if epoch < 2:
        return 0.01
    else:
        return lr * 0.99

lr_scheduler = LearningRateScheduler(scheduler)


# # Neural Network for Sentiment Analysis
# 
# We will now create a neural network that takes the processed movie reviews and outputs its sentiment (0-1). 
# Basic NLP models have a structure similar to the following:
# 
# **Embedding Layer ->Bidirectional LSTM Layer -> Dense Layer -> Output**
# 
# Lets break down the role of each layer:
# 
# - **Embedding**: This layer will transform the previous scalar representation of our words into an n-dimensional vector. This will put words more associated with negative/positive reviews closer with similar words in the n-dimensional space.
# - **Bidirectional LSTM** (*Long Short-term Memory*): This layer is sequence aware in both the forward and backward direction. This means that this layer can interpret meaning carried across a phrase which is very important in understanding language.
# - **Dense**: This is the most simple layer of a neural network. It applies multiplication and addition operators and a non-linear activation function to find non-linear patterns in the data.

# In[ ]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Dropout
from tensorflow.keras.optimizers import Adam

#input dimensions is equal to number of words tokenized (defined above)
input_dim = 10_000
# input length will be the length of our padded sequences
input_length = 20


#defines a text classifier model
model = Sequential([
    Embedding(input_dim=input_dim, output_dim=64, input_length=input_length),
    Bidirectional(LSTM(150)),
    Dropout(0.4),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(
    loss='binary_crossentropy',
    optimizer=Adam(),
    metrics=['accuracy']
)

model.summary()


# In[ ]:


#trains the model
history = model.fit(
    np.array(training_sequences), #must convert to numpy array before sending to model
    np.array(train_labels),       #must convert to numpy array before sending to model
    epochs=100, 
    batch_size=128,                
    callbacks=[my_callback, lr_scheduler], 
    verbose=0)


# In[ ]:


#Plots history of model training
plt.rcParams["figure.figsize"] = (20,5)
fig, axs = plt.subplots(1, 2)

axs[0].plot(history.history['loss'], color='red')
axs[0].set_xlabel('Epoch')
axs[0].set_ylabel('Loss')
axs[0].set_title('Training Loss')

axs[1].plot(history.history['accuracy'])
axs[1].set_xlabel('Epoch')
axs[1].set_ylabel('Accuracy')
axs[1].set_title('Training Accuracy')

plt.show()


# # Evaluate Model Performance
# 
# The model performed very well on training data, but it is likely overfitting. Here we will evaluate the model on test data and see what the model thinks of a review that we write.

# In[ ]:


#Measures models preformance on the testing data
evaluation = model.evaluate(
    np.array(testing_sequences), #must convert to numpy array before sending to model
    np.array(test_labels),       #must convert to numpy array before sending to model
    batch_size=128,
    verbose=0
)

#Prints accuracy of model on testing data
print("MODEL ACCURACY ON TEST DATA: {}%".format(round(evaluation[1], 3) * 100))


# In[ ]:


#Lets see what the model thinks of a new review (keep in mind the review will be cut off after 20 words)
#Enter your review here
NEW_REVIEW ="""
This movie was garbage. I wish I never came to the theater to watch it.
"""


# In[ ]:


#Process the new review the same way the test text was processed
new_review_sequence = tokenizer.texts_to_sequences([NEW_REVIEW])
new_review_sequence = pad_sequences(new_review_sequence, maxlen=20)

#sends new review to be predicted by the model
new_review_prediction = round(model.predict(np.array(new_review_sequence))[0][0])
sentiment = "NEGATIVE" if new_review_prediction == 0 else "POSITIVE"

#displays what the model thinks the sentiment of the review was
print("MOVIE REVIEW:", NEW_REVIEW)
print("MODEL PREDICTED SENTIMENT:", sentiment)


# ### Now try and see if you can achieve a better accuracy!
# 
# 
# ### Similar Notebooks
# **TensorFlow Image Classification Guide**: https://www.kaggle.com/code/calebreigada/tensorflow-image-classification-guide
# 
# **TensorFlow Time Series Forecasting Guide**:
# https://www.kaggle.com/code/calebreigada/tensorflow-time-series-forecasting-guide
