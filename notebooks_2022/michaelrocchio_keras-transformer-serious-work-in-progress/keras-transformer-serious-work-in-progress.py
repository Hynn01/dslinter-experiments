#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
pd.read_csv('../input/uw-madison-gi-tract-image-segmentation/train.csv')


# In[ ]:


pd.read_csv('../input/uw-madison-gi-tract-image-segmentation/sample_submission.csv')


# # Credit where it is very due:
# #### A huge shoutout goes to DARIEN SCHETTLER; he saved me an enormous amount of time and did it way better that I could have! I used alot amount of his code for the pre-processing of the data. Go upvote his notebook: kaggle.com/code/dschettler8845/uwmgit-deeplabv3-end-to-end-pipeline-tf

# # Preprocessing

# In[ ]:


import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from tqdm.notebook import tqdm
from datetime import datetime
import json,itertools
from typing import Optional
from glob import glob

from sklearn.model_selection import StratifiedKFold

import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib as mpl


# In[ ]:


BATCH_SIZE = 32

n_splits=5

fold_selected=1

df = pd.read_csv('../input/uw-madison-gi-tract-image-segmentation/train.csv')
print(df.shape)
df=df.tail(10000)


# In[ ]:


df.rename(columns = {'class':'class_name'}, inplace = True)
#--------------------------------------------------------------------------
df["case"] = df["id"].apply(lambda x: int(x.split("_")[0].replace("case", "")))
df["day"] = df["id"].apply(lambda x: int(x.split("_")[1].replace("day", "")))
df["slice"] = df["id"].apply(lambda x: x.split("_")[3])
#--------------------------------------------------------------------------
TRAIN_DIR="../input/uw-madison-gi-tract-image-segmentation/train"
all_train_images = glob(os.path.join(TRAIN_DIR, "**", "*.png"), recursive=True)
x = all_train_images[0].rsplit("/", 4)[0] ## ../input/uw-madison-gi-tract-image-segmentation/train

path_partial_list = []
for i in range(0, df.shape[0]):
    path_partial_list.append(os.path.join(x,
                          "case"+str(df["case"].values[i]),
                          "case"+str(df["case"].values[i])+"_"+ "day"+str(df["day"].values[i]),
                          "scans",
                          "slice_"+str(df["slice"].values[i])))
df["path_partial"] = path_partial_list
#--------------------------------------------------------------------------
path_partial_list = []
for i in range(0, len(all_train_images)):
    path_partial_list.append(str(all_train_images[i].rsplit("_",4)[0]))
    
tmp_df = pd.DataFrame()
tmp_df['path_partial'] = path_partial_list
tmp_df['path'] = all_train_images

#--------------------------------------------------------------------------
df = df.merge(tmp_df, on="path_partial").drop(columns=["path_partial"])
#--------------------------------------------------------------------------
df["width"] = df["path"].apply(lambda x: int(x[:-4].rsplit("_",4)[1]))
df["height"] = df["path"].apply(lambda x: int(x[:-4].rsplit("_",4)[2]))
#--------------------------------------------------------------------------
del path_partial_list,tmp_df
#--------------------------------------------------------------------------
df_meta = df.groupby('id').first().reset_index()
df_meta = df_meta.reset_index(drop=True)
df_meta.head(5)

def rle_encode(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def rle_decode(mask_rle, shape, color=1):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros((shape[0] * shape[1], shape[2]), dtype=np.float32)
    for lo, hi in zip(starts, ends):
        img[lo : hi] = color
    return img.reshape(shape)


def build_masks(labels,input_shape, colors=True):
    height, width = input_shape
    if colors:
        mask = np.zeros((height, width, 3))
        for label in labels:
            mask += rle_decode(label, shape=(height,width , 3), color=np.random.rand(3))
    else:
        mask = np.zeros((height, width, 1))
        for label in labels:
            mask += rle_decode(label, shape=(height, width, 1))
    mask = mask.clip(0, 1)
    return mask

def rle2maskResize(rle):
    # CONVERT RLE TO MASK 
    if (len(rle)==0): 
        return np.zeros((256,256) ,dtype=np.uint8)
    
    height= 520
    width = 704
    mask= np.zeros( width*height ,dtype=np.uint8)

    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]-1
    lengths = array[1::2]    
    for index, start in enumerate(starts):
        mask[int(start):int(start+lengths[index])] = 1
    
    return mask.reshape( (height,width), order='F' )[::2,::2]

sample_filename = 'case85_day23_slice_0098'
sample_image_df = df[df['id'] == sample_filename]
sample_path = sample_image_df['path'].values[0]
sample_img = cv2.imread(sample_path)
sample_rles = sample_image_df[~sample_image_df.segmentation.isna()]['segmentation'].values

sample_filename = 'case85_day23_slice_0098'
sample_image_df = df[df['id'] == sample_filename]
sample_path = sample_image_df['path'].values[0]
sample_img = cv2.imread(sample_path)
sample_rles = sample_image_df[~sample_image_df.segmentation.isna()]['segmentation'].values

w=sample_image_df['width'].values[0]
h=sample_image_df['height'].values[0]

mask1=rle_decode(sample_rles[0], shape=(w, h, 1))
mask2=rle_decode(sample_rles[1], shape=(w, h, 1))
mask3=rle_decode(sample_rles[2], shape=(w, h, 1))


# In[ ]:


fig = plt.figure(figsize=(10, 5))
gs = gridspec.GridSpec(nrows=1, ncols=2)

ax0 = fig.add_subplot(gs[0, 0])
im = ax0.imshow(sample_img, cmap='bone')
ax0.set_title("Image", fontsize=15, weight='bold', y=1.02)

ax1 = fig.add_subplot(gs[0, 1])
ax1.set_title("Mask", fontsize=15, weight='bold', y=1.02)




colors1 = ['yellow']
colors2 = ['green']
colors3 = ['red']

cmap1 = mpl.colors.ListedColormap(colors1)
cmap2 = mpl.colors.ListedColormap(colors2)
cmap3= mpl.colors.ListedColormap(colors3)

l0 = ax1.imshow(sample_img, cmap='bone')
l1 = ax1.imshow(np.ma.masked_where(mask1== False,  mask1),cmap=cmap1, alpha=1)
l2 = ax1.imshow(np.ma.masked_where(mask2== False,  mask2),cmap=cmap2, alpha=1)
l3 = ax1.imshow(np.ma.masked_where(mask3== False,  mask3),cmap=cmap3, alpha=1)



_ = [ax.set_axis_off() for ax in [ax0,ax1]]

colors = [im.cmap(im.norm(1)) for im in [l1,l2, l3]]
labels = ["Large Bowel", "Small Bowel", "Stomach"]
patches = [ mpatches.Patch(color=colors[i], label=f"{labels[i]}") for i in range(len(labels))]

plt.legend(handles=patches, bbox_to_anchor=(1.1, 0.65), loc=2, borderaxespad=0.4,fontsize = 14,title='Mask Labels', title_fontsize=14, edgecolor="black",  facecolor='#c5c6c7')
plt.suptitle("", fontsize=20, weight='bold')


# In[ ]:


import tensorflow as tf
class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, df, mode='fit',
                 batch_size=32, dim=(256, 256), n_channels=3,
                 n_classes=3, random_state=2019, shuffle=True):
        self.dim = dim
        self.batch_size = batch_size
        self.df = df
        self.mode = mode
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.random_state = random_state  
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_batch = [self.list_IDs[k] for k in indexes]
        
        X = self.__generate_X(list_IDs_batch)
        
        if self.mode == 'fit':
            y = self.__generate_y(list_IDs_batch)
            return X, y
        
        elif self.mode == 'predict':
            return X

        else:
            raise AttributeError('The mode parameter should be set to "fit" or "predict".')
        
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.seed(self.random_state)
            np.random.shuffle(self.indexes)
    
    def __generate_X(self, list_IDs_batch):
        'Generates data containing batch_size samples'
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        
        # Generate data
        for i, ID in enumerate(list_IDs_batch):
            img_path = self.df['path'].iloc[ID]
            img = self.__load_grayscale(img_path)
            # Store samples
            X[i,] = img 

        return X
    
    def __generate_y(self, list_IDs_batch):
        y = np.empty((self.batch_size, *self.dim, self.n_classes), dtype=int)
        
        for i, ID in enumerate(list_IDs_batch):
            im_name = self.df['id'].iloc[ID]
            
            image_df = self.df[self.df['id'] == im_name]
            
            rles = image_df[~image_df.segmentation.isna()]['segmentation'].values
            w    = image_df['width'].values[0]
            h    = image_df['height'].values[0]
            
            masks = build_masks(rles,(h,w), colors=False)
            masks = cv2.resize(masks, (256, 256))
            #masks=masks.transpose(1,0)
            masks=np.expand_dims(masks, axis=-1)
            y[i, ] = masks

        return y
    
    def __load_grayscale(self, img_path):
        img = cv2.imread(img_path, cv2.IMREAD_ANYDEPTH)
        # resize image
        dsize = (256, 256)
        img = cv2.resize(img, dsize)
        
        img = img.astype(np.float32) / 255.
        img = np.expand_dims(img, axis=-1)
        return img


# In[ ]:


skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
for fold, (_, val_idx) in enumerate(skf.split(X=df, y=df['case']), 1):
    df.loc[val_idx, 'fold'] = fold
    
df['fold'] = df['fold'].astype(np.uint8)

train_ids = df[df["fold"]!=fold_selected].index
valid_ids = df[df["fold"]==fold_selected].index

df.groupby('fold').size()


# In[ ]:


train_generator = DataGenerator(
    train_ids, 
    df=df,
    batch_size=BATCH_SIZE, 
    n_classes=3
)
val_generator = DataGenerator(
    valid_ids, 
    df=df,
    batch_size=BATCH_SIZE, 
    n_classes=3
)

plt.figure(figsize=(7,7))
for i in range(1):
    images, mask = val_generator[i]
    print("Dimension of image:", images.shape)
    print("Dimension of mask:", mask.shape)
    plt.imshow(images[0,:,:,0], cmap="gray")
    plt.imshow(mask[0,:,:,0],  alpha=0.3, cmap="Reds")
    plt.show()


# In[ ]:


plt.figure(figsize=(7,7))
for i in range(1):
    images, mask = val_generator[i]
    print("Dimension of image:", images.shape)
    print("Dimension of mask:", mask.shape)
    plt.imshow(images[15,:,:,2], cmap="gray")
    plt.imshow(mask[15,:,:,2],  alpha=0.3, cmap="Reds")
    plt.show()


# In[ ]:


images, mask = val_generator[1]


# In[ ]:


# !pip install tensorflow==2.8


# ### Alot below is prior code I have written that I think will be usable for this problem but I need alot of work with the mask input

# # Gathering Possibly reusable code 

# In[ ]:


import tensorflow_addons as tfa
import matplotlib.pyplot as plt
from tensorflow import keras

wd = 0.0001
lr = 0.001
batch = 128
epochs = 120
image_size = 256
patch_size = 16
patch_dist = 16**2
projection = 64
# ENCODER and DECODER
LAYER_NORM_EPS = 1e-6
ENC_PROJECTION_DIM = 128
DEC_PROJECTION_DIM = 64
ENC_NUM_HEADS = 4
ENC_LAYERS = 6
DEC_NUM_HEADS = 4
DEC_LAYERS = (
    2  # The decoder is lightweight but should be reasonably deep for reconstruction.
)
ENC_TRANSFORMER_UNITS = [
    ENC_PROJECTION_DIM * 2,
    ENC_PROJECTION_DIM,
]  # Size of the transformer layers.
DEC_TRANSFORMER_UNITS = [
    DEC_PROJECTION_DIM * 2,
    DEC_PROJECTION_DIM,
]

num_heads = 4
transformer_val = [128, 64]
layers = 8
mlp_head_units = [2048, 1024]
data_augmentation = keras.Sequential(
    [
        keras.layers.experimental.preprocessing.Normalization(),
        keras.layers.experimental.preprocessing.Resizing(image_size, image_size),
        keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
        keras.layers.experimental.preprocessing.RandomRotation(factor=0.02),
        keras.layers.experimental.preprocessing.RandomZoom(
            height_factor=0.2, width_factor=0.2
        ),
    ],
    name="data_augmentation",
)
data_augmentation.layers[0].adapt(images[0])
def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = keras.layers.Dense(units, activation=tf.nn.gelu)(x)
        x = keras.layers.Dropout(dropout_rate)(x)
    return x

class Patches(keras.layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch, -1, patch_dims])
        return patches
image=images[15]
plt.figure(figsize=(4, 4))
plt.imshow(image[:,:,0], cmap="gray")
plt.axis("off")
resized_image = tf.image.resize(
    tf.convert_to_tensor([image]), size=(image_size, image_size)
)
patches = Patches(patch_size)(resized_image)
print(f"Image size: {image_size} X {image_size}")
print(f"Patch size: {patch_size} X {patch_size}")
print(f"Patches per image: {patches.shape[1]}")
print(f"Elements per patch: {patches.shape[-1]}")
n = int(np.sqrt(patches.shape[1]))
plt.figure(figsize=(4, 4))
for i, patch in enumerate(patches[0]):
    ax = plt.subplot(n, n, i + 1)
    patch_img = tf.reshape(patch, (patch_size, patch_size, 3))
    plt.imshow(patch_img[:,:,0], cmap='gray')
    plt.axis("off")


# In[ ]:


## plt.figure(figsize=(4, 4))
image = mask[15]
plt.imshow(image[:,:,0], cmap="Reds")
plt.axis("off")
resized_image = tf.image.resize(
    tf.convert_to_tensor([image]), size=(image_size, image_size)
)
patches = Patches(patch_size)(resized_image)
print(f"Image size: {image_size} X {image_size}")
print(f"Patch size: {patch_size} X {patch_size}")
print(f"Patches per image: {patches.shape[1]}")
print(f"Elements per patch: {patches.shape[-1]}")
n = int(np.sqrt(patches.shape[1]))
plt.figure(figsize=(4, 4))
for i, patch in enumerate(patches[0]):
    ax = plt.subplot(n, n, i + 1)
    patch_img = tf.reshape(patch, (patch_size, patch_size, 3))
    plt.imshow(patch_img[:,:,:], cmap='Reds')
    plt.axis("off")


# In[ ]:


images, mask = val_generator[5]
print(images.shape)
print(mask.shape)


# In[ ]:


full_images, full_mask = val_generator[i]
for j in range(0,32):
    if j==0:
        image = full_mask[j]
        resized_image = tf.image.resize(
            tf.convert_to_tensor([image]), size=(256, 256)
        )
        patches = Patches(patch_size)(resized_image)
        party=[]
        for i, patch in enumerate(patches[0]):
            patch_img = tf.reshape(patch, (patch_size, patch_size, 3))
            mean_val=np.mean(np.array(patch_img))
            mean_val=mean_val.astype(int)
            party.append(mean_val)
        party=np.array(party)
        fully=party.reshape((1,party.shape[0]))
    else:
        image = full_mask[j]
        resized_image = tf.image.resize(
            tf.convert_to_tensor([image]), size=(256, 256)
        )
        patches = Patches(patch_size)(resized_image)
        party=[]
        for i, patch in enumerate(patches[0]):
            patch_img = tf.reshape(patch, (patch_size, patch_size, 3))
            mean_val=np.mean(np.array(patch_img))
            mean_val=mean_val.astype(int)
            party.append(mean_val)
        party=np.array(party)
        fully=np.vstack((fully,party.reshape((1,party.shape[0]))))
for i in range(1,70):
    print(i)
    images, mask = val_generator[i]

    full_images=np.vstack((full_images,images))
    
    for j in range(0,32):
        image = mask[j]
        resized_image = tf.image.resize(
            tf.convert_to_tensor([image]), size=(image_size, image_size))
        patches = Patches(patch_size)(resized_image)
        party=[]
        for i, patch in enumerate(patches[0]):
            patch_img = tf.reshape(patch, (patch_size, patch_size, 3))
            mean_val=np.mean(np.array(patch_img))
            mean_val=mean_val.astype(int)
            party.append(mean_val)
        party=np.array(party)
        party=party.reshape((1,256))
        fully=np.vstack((fully,party))
    print(fully.shape)
    print(full_images.shape)


# In[ ]:


fully_val=fully[0:320]
fully=fully[320:]
full_images_val=full_images[0:320]
full_images=full_images[320:]


# In[ ]:


import keras
from keras.models import Sequential
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.datasets import cifar10
from keras import regularizers
from keras.callbacks import LearningRateScheduler
import numpy as np
 
weight_decay = 1e-4
model = Sequential()
model.add(Conv2D(32, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), input_shape=(256,256,3)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(32, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))
 
model.add(Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.3))
 
model.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))

model.add(Conv2D(256, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(256, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(256, activation='sigmoid'))
 
model.summary()


# In[ ]:


from keras.optimizers import *
# opt_rms = tf.keras.optimizers.RMSprop(lr=0.001,decay=1e-6)
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])


# In[ ]:


fully.shape


# In[ ]:


model.fit(x=full_images, y=fully, epochs=250, validation_data=(full_images_val, fully_val))


# In[ ]:


full_images, full_mask = val_generator[i]


for j in range(0,32):
    if j==0:
        image = full_mask[j]
        resized_image = tf.image.resize(
            tf.convert_to_tensor([image]), size=(256, 256)
        )
        patches = Patches(patch_size)(resized_image)
        party=[]
        for i, patch in enumerate(patches[0]):
            patch_img = tf.reshape(patch, (patch_size, patch_size, 3))
            mean_val=np.mean(np.array(patch_img))
            mean_val=mean_val.astype(int)
            party.append(mean_val)
        party=np.array(party)
        fully=party.reshape((1,party.shape[0]))
    else:
        image = full_mask[j]
        resized_image = tf.image.resize(
            tf.convert_to_tensor([image]), size=(256, 256)
        )
        patches = Patches(patch_size)(resized_image)
        party=[]
        for i, patch in enumerate(patches[0]):
            patch_img = tf.reshape(patch, (patch_size, patch_size, 3))
            mean_val=np.mean(np.array(patch_img))
            mean_val=mean_val.astype(int)
            party.append(mean_val)
        party=np.array(party)
        fully=np.vstack((fully,party.reshape((1,party.shape[0]))))
for i in range(100,102):
    print(i)
    images, mask = val_generator[i]

    full_images=np.vstack((full_images,images))
    
    for j in range(0,32):
        image = mask[j]
        resized_image = tf.image.resize(
            tf.convert_to_tensor([image]), size=(image_size, image_size))
        patches = Patches(patch_size)(resized_image)
        party=[]
        for i, patch in enumerate(patches[0]):
            patch_img = tf.reshape(patch, (patch_size, patch_size, 3))
            mean_val=np.mean(np.array(patch_img))
            mean_val=mean_val.astype(int)
            party.append(mean_val)
        party=np.array(party)
        party=party.reshape((1,256))
        fully=np.vstack((fully,party))
    print(fully.shape)
    print(full_images.shape)


# In[ ]:


mask[0]


# In[ ]:


val_y=model.predict(full_images[0].reshape((1,256,256,3)))


# In[ ]:


dftest=pd.DataFrame({
    'realy': fully[0],
    'val_y': val_y[0]
})
dftest.to_csv('100_1_test')


# In[ ]:


print(fully[5])


# In[ ]:


dftest


# In[ ]:




