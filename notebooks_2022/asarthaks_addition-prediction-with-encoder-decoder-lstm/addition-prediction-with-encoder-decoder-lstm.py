#!/usr/bin/env python
# coding: utf-8

# ### This code is implemented from the excercises given in the book Long Short-Term Memory Networks With Python By Jason Brownlee, It's a really good read if you want to learn about how LSTM works and about different types of LSTMs.

# ### Using Tensorflow-GPU

# In[ ]:


import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


# One approach to seq2seq prediction problems that has proven very e↵ective is called the Encoder-
# Decoder LSTM. This architecture is comprised of two models: one for reading the input sequence
# and encoding it into a fixed-length vector, and a second for decoding the fixed-length vector
# and outputting the predicted sequence. The use of the models in concert gives the architecture
# its name of Encoder-Decoder LSTM designed specifically for seq2seq problems.

# # Importing Libraries

# In[ ]:


from math import ceil, log10
from random import seed, randint
from numpy import array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, RepeatVector, TimeDistributed
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten, Dense
from numpy import argmax
from matplotlib import pyplot


# # About the problem

# ### Addition Prediction Problem

# The problem is defined as calculating the sum output of two input numbers. This is
# challenging as each digit and mathematical symbol is provided as a character and the expected
# output is also expected as characters. For example, the input 10+6 with the output 16 would
# be represented by the sequences:<br>
# Input: ['1', '0', '+', '6']<br>
# Output: ['1', '6']

# We will
# divide this into the following steps:<br>
# <ol>1. Generate Sum Pairs.</ol>
# <ol>2. Integers to Padded Strings.</ol>
# <ol>3. Integer Encoded Sequences.</ol>
# <ol>4. One Hot Encoded Sequences.</ol>
# <ol>5. Sequence Generation Pipeline.</ol>
# <ol>6. Decode Sequences.</ol>

# ### Generate Sum Pairs

# In[ ]:


#generate lists of random integers and their sum
def random_sum_pairs(n_examples, n_numbers, largest):
    X, y = list(), list()
    for i in range(n_examples):
        in_pattern = [randint(1,largest) for _ in range(n_numbers)]
        out_pattern = sum(in_pattern)
        X.append(in_pattern)
        y.append(out_pattern)
    return X, y


# In[ ]:


seed(1)
n_samples = 1
n_numbers = 3
largest = 10
#generate pairs
X, y = random_sum_pairs(n_samples, n_numbers, largest)
print(X, y)


# ### Integers to Padded Strings

# Padding requires we know how long the longest sequence may be.
# We can calculate this easily by taking the log10() of the largest integer we can generate and the
# ceiling of that number to get an idea of how many chars are needed for each number. We add 1
# to the largest number to ensure we expect 3 chars instead of 2 chars for the case of a round
# largest number, like 200 and take the ceiling of the result (e.g. ceil(log10(largest+1))). We
# then need to add the right number of plus symbols (e.g. n numbers - 1).

# In[ ]:


#max length of input sequence
max_length_i = n_numbers*ceil(log10(largest+1)) + n_numbers - 1
print(max_length_i)


# In[ ]:


#max length of output sequence
max_length_o = ceil(log10(n_numbers*(largest+1)))
print(max_length_o)


# In[ ]:


#convert data to strings
def to_string(X, y, n_numbers, largest):
    max_length = int(n_numbers*ceil(log10(largest+1)) + n_numbers - 1)
    Xstr = list()
    for pattern in X:
        strp = '+'.join([str(n) for n in pattern])
        strp = ''.join([' ' for _ in range(max_length-len(strp))])  + strp
        Xstr.append(strp)
    max_length = int(ceil(log10(n_numbers*(largest+1))))
    ystr = list()
    for pattern in y:
        strp = str(pattern)
        strp = ''.join([' ' for _ in range(max_length-len(strp))]) + strp
        ystr.append(strp)
    return Xstr, ystr


# In[ ]:


seed(1)
n_samples = 1
n_numbers = 2
largest = 10


# In[ ]:


#generate pairs
X, y = random_sum_pairs(n_samples, n_numbers, largest)
print(X, y)


# In[ ]:


X, y = to_string(X, y, n_numbers, largest)
print(X, y)


# ### Integer Encoded Sequences

# Next, we need to encode each character in the string as an integer value. We have to work with
# numbers in neural networks after all, not characters. Integer encoding transforms the problem
# into a classification problem where the output sequence may be considered class outputs with
# 11 possible values each. This just so happens to be integers with some ordinal relationship (the
# first 10 class values). To perform this encoding, we must define the full alphabet of symbols
# that may appear in the string encoding, as follows:

# In[ ]:


alphabet = [str(i) for i in range(0, 10)]
alphabet.append('+')
alphabet.append(' ')
print(alphabet)


# In[ ]:


#integer encode strings
def integer_encode(X, y, alphabet):
    char_to_int = dict((c, i) for i, c in enumerate(alphabet))
    Xenc = list()
    for pattern in X:
        integer_encoded = [char_to_int[char] for char in pattern]
        Xenc.append(integer_encoded)
    yenc = list()
    for pattern in y:
        integer_encoded = [char_to_int[char] for char in pattern]
        yenc.append(integer_encoded)
    return Xenc, yenc


# In[ ]:


seed(1)
n_samples = 1
n_numbers = 2
largest = 10


# In[ ]:


#generate pairs
X, y = random_sum_pairs(n_samples, n_numbers, largest)
print(X, y)


# In[ ]:


#convert to strings
X, y = to_string(X, y, n_numbers, largest)
print(X, y)


# In[ ]:


#creating alphabet list
alphabet = [str(i) for i in range(0, 10)]
alphabet.append('+')
alphabet.append(' ')
print(alphabet)


# In[ ]:


#integer encode
X, y = integer_encode(X, y, alphabet)
print(X, y)


# ### One Hot Encoded Sequences

# The next step is to binary encode the integer encoding sequences. This involves converting each
# integer to a binary vector with the same length as the alphabet and marking the specific integer
# with a 1. For example, a 0 integer represents the ‘0’ character and would be encoded as a
# binary vector with a 1 in the 0th position of an 11 element vector: [1, 0, 0, 0, 0, 0, 0, 0,
# 0, 0, 0, 0].

# In[ ]:


# one hot encode
def one_hot_encode(X, y, max_int):
    Xenc = list()
    for seq in X:
        pattern = list()
        for index in seq:
            vector = [0 for _ in range(max_int)]
            vector[index] = 1
            pattern.append(vector)
        Xenc.append(pattern)
    yenc = list()
    for seq in y:
        pattern = list()
        for index in seq:
            vector = [0 for _ in range(max_int)]
            vector[index] = 1
            pattern.append(vector)
        yenc.append(pattern)
    return Xenc, yenc


# In[ ]:


#one hot encode
X, y = one_hot_encode(X, y, len(alphabet))
print(X, y)
X = array(X)
y = array(y)
print(X.shape, y.shape)


# ### Sequence Generation Pipeline

# We can tie all of these steps together into a function called generate data(), listed below.
# Given a designed number of samples, number of terms, the largest value of each term, and the
# alphabet of possible characters, the function will generate a set of input and output sequences.

# In[ ]:


# generate an encoded dataset
def generate_data(n_samples, n_numbers, largest, alphabet):
    # generate pairs
    X, y = random_sum_pairs(n_samples, n_numbers, largest)
    # convert to strings
    X, y = to_string(X, y, n_numbers, largest)
    # integer encode
    X, y = integer_encode(X, y, alphabet)
    # one hot encode
    X, y = one_hot_encode(X, y, len(alphabet))
    # return as numpy arrays
    X, y = array(X), array(y)
    return X, y


# In[ ]:


#invert encoding
def invert(seq, alphabet):
    int_to_char = dict((i, c) for i, c in enumerate(alphabet))
    strings = list()
    for pattern in seq:
        string = int_to_char[argmax(pattern)]
        strings.append(string)
    return ''.join(strings)


# # Define and Compile the Model

# In[ ]:


# number of math terms
n_terms = 3
# largest value for any single input digit
largest = 10
# scope of possible symbols for each input or output time step
alphabet = [str(x) for x in range(10)] + ['+', ' ']


# In[ ]:


# size of alphabet: (12 for 0-9, + and ' ')
n_chars = len(alphabet)
# length of encoded input sequence (8 for '10+10+10')
n_in_seq_length = int(n_terms * ceil(log10(largest+1)) + n_terms - 1)
# length of encoded output sequence (2 for '30')
n_out_seq_length = int(ceil(log10(n_terms * (largest+1))))


# In[ ]:


#define LSTM
model = Sequential()
model.add(LSTM(75, input_shape=(n_in_seq_length, n_chars)))
model.add(RepeatVector(n_out_seq_length))
model.add(LSTM(50, return_sequences=True))
model.add(TimeDistributed(Dense(n_chars, activation='softmax')))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())


# # Fit the Model

# In[ ]:


# fit LSTM
X, y = generate_data(10000, n_terms, largest, alphabet)


# In[ ]:


history = model.fit(X, y, epochs=20, batch_size=32, validation_split=0.33)


# In[ ]:


pyplot.plot(history.history['loss'])
pyplot.plot(history.history['val_loss'])
pyplot.title('model train vs validation loss')
pyplot.ylabel('loss')
pyplot.xlabel('epoch')
pyplot.legend(['train', 'validation'], loc='upper right')
pyplot.show()


# # Evaluate the Model

# In[ ]:


# evaluate LSTM
X, y = generate_data(100, n_terms, largest, alphabet)
loss, acc = model.evaluate(X, y, verbose=0)
print('Loss: %f, Accuracy: %f' %(loss, acc*100))


# # Make Predictions with the Model

# In[ ]:


# predict
for _ in range(10):
    # generate an input-output pair
    X, y = generate_data(1, n_terms, largest, alphabet)
    #make prediction
    yhat = model.predict(X, verbose=0)
    #decode input, expected and predicted
    in_seq = invert(X[0], alphabet)
    out_seq = invert(y[0], alphabet)
    predicted = invert(yhat[0], alphabet)
    print('%s = %s (expected %s)' %(in_seq, predicted, out_seq))


# In[ ]:




