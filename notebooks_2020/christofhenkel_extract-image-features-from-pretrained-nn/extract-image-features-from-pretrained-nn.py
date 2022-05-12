#!/usr/bin/env python
# coding: utf-8

# In this kernel I want to demonstrate how to extract features from the pet images using a pretrained network. Since there are often none or multiple images of different resoltuions and aspect ratio I make the following preprocessing steps:
# 
# - Take only profile picture (if existing else black)
# - pad to square aspect ratio
# - resize to 256
# 

# In[ ]:


import cv2
import pandas as pd
import numpy as np
import os
from tqdm import tqdm, tqdm_notebook

train_df = pd.read_csv('../input/train/train.csv')
img_size = 256
batch_size = 16


# In[ ]:


pet_ids = train_df['PetID'].values
n_batches = len(pet_ids) // batch_size + 1


# In[ ]:


from keras.applications.densenet import preprocess_input, DenseNet121


# In[ ]:


def resize_to_square(im):
    old_size = im.shape[:2] # old_size is in (height, width) format
    ratio = float(img_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    # new_size should be in (width, height) format
    im = cv2.resize(im, (new_size[1], new_size[0]))
    delta_w = img_size - new_size[1]
    delta_h = img_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,value=color)
    return new_im

def load_image(path, pet_id):
    image = cv2.imread(f'{path}{pet_id}-1.jpg')
    new_image = resize_to_square(image)
    new_image = preprocess_input(new_image)
    return new_image


# Lets define our model for feature extraction. Normally DenseNet121 would output 1024 features after GlobalAveragePooling. To further narrow it down, I again pool 4 features each.

# In[ ]:


from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Input, Lambda, AveragePooling1D
import keras.backend as K
inp = Input((256,256,3))
backbone = DenseNet121(input_tensor = inp, include_top = False)
x = backbone.output
x = GlobalAveragePooling2D()(x)
x = Lambda(lambda x: K.expand_dims(x,axis = -1))(x)
x = AveragePooling1D(4)(x)
out = Lambda(lambda x: x[:,:,0])(x)

m = Model(inp,out)


# In[ ]:


features = {}
for b in tqdm_notebook(range(n_batches)):
    start = b*batch_size
    end = (b+1)*batch_size
    batch_pets = pet_ids[start:end]
    batch_images = np.zeros((len(batch_pets),img_size,img_size,3))
    for i,pet_id in enumerate(batch_pets):
        try:
            batch_images[i] = load_image("../input/train_images/", pet_id)
        except:
            pass
    batch_preds = m.predict(batch_images)
    for i,pet_id in enumerate(batch_pets):
        features[pet_id] = batch_preds[i]


# In[ ]:


train_feats = pd.DataFrame.from_dict(features, orient='index')


# We save the features as a csv to disk, so others can link and join the data frame with their train.csv

# In[ ]:


train_feats.to_csv('train_img_features.csv')
train_feats.head()


# and repeat the procedure again for test images

# In[ ]:


test_df = pd.read_csv('../input/test/test.csv')


# In[ ]:


pet_ids = test_df['PetID'].values
n_batches = len(pet_ids) // batch_size + 1


# In[ ]:


features = {}
for b in tqdm_notebook(range(n_batches)):
    start = b*batch_size
    end = (b+1)*batch_size
    batch_pets = pet_ids[start:end]
    batch_images = np.zeros((len(batch_pets),img_size,img_size,3))
    for i,pet_id in enumerate(batch_pets):
        try:
            batch_images[i] = load_image("../input/test_images/", pet_id)
        except:
            pass
    batch_preds = m.predict(batch_images)
    for i,pet_id in enumerate(batch_pets):
        features[pet_id] = batch_preds[i]


# In[ ]:


test_feats = pd.DataFrame.from_dict(features, orient='index')


# In[ ]:


test_feats.to_csv('test_img_features.csv')
test_feats.head()


# In[ ]:




