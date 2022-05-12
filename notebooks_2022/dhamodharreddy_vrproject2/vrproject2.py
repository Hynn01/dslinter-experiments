#!/usr/bin/env python
# coding: utf-8

# ### Import libraries

# In[ ]:


import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import string


# ### Import keras libraries

# In[ ]:


import tensorflow
from tensorflow.keras.models import Model
from tensorflow.keras import Input, layers
from tensorflow.keras import optimizers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import LSTM, Embedding, Dense, Activation, Flatten, Reshape, Dropout


# ### Import train,test,val image names from given datafiles  

# In[ ]:


train_image_names = open('../input/image-captioning/flickr-8k/flickr-8k/Flickr8k_text-20220427T180132Z-001/Flickr8k_text/Flickr_8k.trainImages.txt','r').read().splitlines()
val_image_names = open('../input/image-captioning/flickr-8k/flickr-8k/Flickr8k_text-20220427T180132Z-001/Flickr8k_text/Flickr_8k.valImages.txt','r').read().splitlines()
test_image_names = open('../input/image-captioning/flickr-8k/flickr-8k/Flickr8k_text-20220427T180132Z-001/Flickr8k_text/Flickr_8k.testImages.txt','r').read().splitlines()
images_path = '../input/image-captioning/flickr-8k/flickr-8k/Flicker8k_Images-20220427T175615Z-001/Flicker8k_Images/'


# ### Import lemmatized text decriptions from datafiles

# In[ ]:


file = open('../input/image-captioning/flickr-8k/flickr-8k/Flickr8k_text-20220427T180132Z-001/Flickr8k_text/Flickr8k.lemma.token.txt','r')
doc = file.read()


# ### Cleaning the text descriptions

# In[ ]:


descriptors = dict()
    # process lines
for line in doc.split('\n'):
    # split line by white space
    tokens = line.split()
    if len(line) < 2:
        continue
    # take the first token as the image id, the rest as the description
    image_id, image_desc = tokens[0], tokens[1:]
    # extract filename from image id
    image_id = image_id.split('#')[0]
    # convert description tokens back to string
    image_desc = ' '.join(image_desc)
    # create the list if needed
    if image_id not in descriptors:
        descriptors[image_id] = list()
    descriptors[image_id].append(image_desc)
len(descriptors)


# In[ ]:


# prepare translation table for removing punctuation
table = str.maketrans('', '', string.punctuation)
for key, desc_list in descriptors.items():
    for i in range(len(desc_list)):
        desc = desc_list[i]
        # tokenize
        desc = desc.split()
        # convert to lower case
        desc = [word.lower() for word in desc]
        # remove punctuation from each token
        desc = [w.translate(table) for w in desc]
        # remove hanging 's' and 'a'
        desc = [word for word in desc if len(word)>1]
        # remove tokens with numbers in them
        desc = [word for word in desc if word.isalpha()]
        # store as string
        desc_list[i] =  ' '.join(desc)


# In[ ]:


vocab = set()
for key in descriptors.keys():
    [vocab.update(d.split()) for d in descriptors[key]]
len(vocab)


# In[ ]:


image_name_list = list(descriptors.keys())
# image_name = image_name_list[4]
# x = plt.imread(images_path+image_name)
# plt.imshow(x)
# plt.show()
# for i in descriptors[image_name]:
#     print(i)


# #### Append startseq and endseq to each sentence to distinguish the end of the sentence

# In[ ]:


for image_name in image_name_list:
    for i in range(len(descriptors[image_name])):
        descriptors[image_name][i] = 'startseq ' + descriptors[image_name][i] + ' endseq'


# #### Split descriptors into train,val and test descriptors  

# In[ ]:


train_text = {}
val_text = {}
test_text = {}
for i in train_image_names:
    if i not in descriptors:
        descriptors[i] = list()
    train_text[i] = descriptors[i]
for i in val_image_names:
    if i not in descriptors:
        descriptors[i] = list()
    val_text[i] = descriptors[i]
for i in test_image_names:
    if i not in descriptors:
        descriptors[i] = list()
    test_text[i] = descriptors[i]


# #### Check the maximum length before training

# In[ ]:


max_length = 0
for filename,texts in descriptors.items():
    for i in texts:
        if(max_length < len(i.split())):
            max_length = len(i.split())
            max_string = i
            max_list = i.split()


# In[ ]:


max_length


# #### Delete the words which are not repeated for more than a certain number of times. (No usefule information from these words to train) 

# In[ ]:


word_counts = {}
nsents = 0
for key,values in train_text.items():
    for i in values:
        nsents += 1
        for w in i.split(' '):
            word_counts[w] = word_counts.get(w, 0) + 1
vocabulary = [w for w in word_counts if word_counts[w] >= 10]
print(len(vocabulary))


# #### Import all the word embeddings from GloVe 6b

# In[ ]:


embeddings_index = {} 
glove_file = open(os.path.join('../input/image-captioning/glove.6B.200d.txt/glove.6B.200d.txt'), encoding="utf-8")
for line in glove_file:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs


# #### Assign indices to the words and vice-versa 

# In[ ]:


ixtoword = {}
wordtoix = {}
ix = 1
for w in vocabulary:
    wordtoix[w] = ix
    ixtoword[ix] = w
    ix += 1

vocab_size = len(ixtoword) + 1


# #### Convert the words into corresponding word embedding vectors

# In[ ]:


embedding_dim = 200
embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, i in wordtoix.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
    break


# ### CNN model (ResNet50) for feature extraction from images

# In[ ]:


resnet_model = ResNet50(include_top=False,weights='imagenet',input_shape=(224,224,3),pooling='avg')
resnet_model.summary()


# In[ ]:


def img_preprocess(img_path):
    im = cv2.imread(images_path  + img_path)   
    im_res = cv2.resize(im,(224,224))
    im_res = np.expand_dims(im_res, axis=0)
    return im_res


# #### Predict the feature vectors for each training image 

# In[ ]:


# train_data = {}
# ctr=0
# for ix in train_image_names:
#     if ix == "":
#         continue
#     ctr+=1
#     if ctr%500==0:
#         print(ctr)
#     path = ix
#     img = img_preprocess(path)
#     pred = resnet_model.predict(img).reshape(2048)
#     train_data[ix] = pred


# #### Store and load the image feature vectors

# In[ ]:


filename = '../input/imagefeatures/cnn_train_features.pickle'
file = open(filename, 'rb')
trainImg_features = pkl.load(file)


# In[ ]:


# filename = 'cnn_train_features.pickle'
# file = open(filename, 'wb')
# pkl.dump(train_data,file)


# ### CNN model InceptionV3 for feature extraction from images

# In[ ]:


# from time import time
# from tensorflow.keras.applications.inception_v3 import InceptionV3
# from tensorflow.keras.applications.inception_v3 import preprocess_input


# In[ ]:


# inc_model = InceptionV3(weights='imagenet')
# model_new = Model(inc_model.input, inc_model.layers[-2].output)


# In[ ]:


# # Function to encode a given image into a vector of size (2048, )
# def encode(image):
#     image = img_preprocess(image) # preprocess the image
#     fea_vec = model_new.predict(image) # Get the encoding vector for the image
#     fea_vec = np.reshape(fea_vec, fea_vec.shape[1]) # reshape from (1, 2048) to (2048, )
#     return fea_vec


# In[ ]:


# start = time()
# encoding_train = {}
# x = 0
# for img in train_image_names:
#     encoding_train[img] = encode(img)
#     if x %100 == 0:
#         print(x)
#     x+=1
# print("Time taken in seconds =", time()-start)


# ### Sequential LSTM model to predict captions

# In[ ]:


inputs1 = Input(shape=(2048,))
fe1 = Dropout(0.5)(inputs1)
fe2 = Dense(256, activation='relu')(fe1)

inputs2 = Input(shape=(max_length,))
se1 = Embedding(vocab_size, embedding_dim, mask_zero=True)(inputs2)
se2 = Dropout(0.5)(se1)
se3 = LSTM(256)(se2)

decoder1 = add([fe2, se3])
decoder2 = Dense(256, activation='relu')(decoder1)
outputs = Dense(vocab_size, activation='softmax')(decoder2)

model = Model(inputs=[inputs1, inputs2], outputs=outputs)
model.summary()


# #### Setting the embedding layer weights to the weights we predicted from the word embeddings

# In[ ]:


model.layers[2].set_weights([embedding_matrix])
model.layers[2].trainable = False


# In[ ]:


model.compile(loss='categorical_crossentropy', optimizer='adam')


# In[ ]:


def data_generator(descriptions, photos, wordtoix, max_length, num_photos_per_batch):
    X1, X2, y = list(), list(), list()
    n=0
    # loop for ever over images
    while 1:
        for key, desc_list in descriptions.items():
            n+=1
            # retrieve the photo feature
            photo = photos[key]
            for desc in desc_list:
                # encode the sequence
                seq = [wordtoix[word] for word in desc.split(' ') if word in wordtoix]
                # split one sequence into multiple X, y pairs
                for i in range(1, len(seq)):
                    # split into input and output pair
                    in_seq, out_seq = seq[:i], seq[i]
                    # pad input sequence
                    in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                    # encode output sequence
                    out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                    # store
                    X1.append(photo)
                    X2.append(in_seq)
                    y.append(out_seq)

            if n==num_photos_per_batch:
                yield ([np.array(X1), np.array(X2)], np.array(y))
                X1, X2, y = list(), list(), list()
                n=0


# In[ ]:


epochs = 15
batch_size = 3
steps = len(train_text)//batch_size

generator = data_generator(train_text, trainImg_features, wordtoix, max_length, batch_size)
model.fit(generator, epochs=epochs, steps_per_epoch=steps, verbose=1)


# In[ ]:


# generator = data_generator(train_text, trainImg_features, wordtoix, max_length, batch_size)
# model.fit(generator, epochs=epochs, steps_per_epoch=steps, verbose=1)


# In[ ]:


from nltk.translate.bleu_score import sentence_bleu
def greedySearch(photo):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = [wordtoix[w] for w in in_text.split() if w in wordtoix]
        sequence = pad_sequences([sequence], maxlen=max_length)
        
        yhat = model.predict([photo,sequence], verbose=0)
        
        yhat = np.argmax(yhat)
        word = ixtoword[yhat]
        in_text += ' ' + word
        if word == 'endseq':
            break

    final = in_text.split()
    final = final[1:-1]
    final = ' '.join(final)
    return final


# In[ ]:


z = 0


# In[ ]:


z+=1
img_name = test_image_names[z]
img = img_preprocess(img_name)
pred = resnet_model.predict(img).reshape(1,2048)
x=plt.imread(images_path+img_name)
plt.imshow(x)
plt.show()
candidate = greedySearch(pred)
print("Greedy Search:",candidate)

reference = test_text[img_name]
print('BLEU score -> {}'.format(sentence_bleu(reference, candidate)))


# In[ ]:


model = model
model.save('./model_2')


# In[ ]:


model=tensorflow.keras.models.load_model('.../input/model/model_2')


# In[ ]:


from keras.preprocessing import sequence
def beam_search_predictions(image, beam_index = 3):
    start = [wordtoix["startseq"]]
    start_word = [[start, 0.0]]
    while len(start_word[0][0]) < max_length:
        temp = []
        for s in start_word:
            par_caps = sequence.pad_sequences([s[0]], maxlen=max_length, padding='post')
            preds = model.predict([image,par_caps], verbose=0)
            word_preds = np.argsort(preds[0])[-beam_index:]
            # Getting the top <beam_index>(n) predictions and creating a 
            # new list so as to put them via the model again
            for w in word_preds:
                next_cap, prob = s[0][:], s[1]
                next_cap.append(w)
                prob += preds[0][w]
                temp.append([next_cap, prob])
                    
        start_word = temp
        # Sorting according to the probabilities
        start_word = sorted(start_word, reverse=False, key=lambda l: l[1])
        # Getting the top words
        start_word = start_word[-beam_index:]
    
    start_word = start_word[-1][0]
    intermediate_caption = [ixtoword[i] for i in start_word]
    final_caption = []
    
    for i in intermediate_caption:
        if i != 'endseq':
            final_caption.append(i)
        else:
            break

    final_caption = ' '.join(final_caption[1:])
    return final_caption


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




