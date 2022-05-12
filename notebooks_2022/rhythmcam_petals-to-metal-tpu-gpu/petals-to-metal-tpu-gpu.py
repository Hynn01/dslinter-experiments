#!/usr/bin/env python
# coding: utf-8

# #### reference : https://www.kaggle.com/code/dmitrynokhrin/start-with-ensemble-v2

# # Imports and Defines

# In[ ]:


import numpy as np
import pandas as pd
import random,os,re

import matplotlib.pyplot as plt

from kaggle_datasets import KaggleDatasets

import tensorflow as tf
from tensorflow.keras.applications.densenet import DenseNet201
from tensorflow.keras.applications import Xception 
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense

SAMPLE_SUBMISSION_PATH = "../input/tpu-getting-started/sample_submission.csv"
SUBMISSION_PATH = "submission.csv"

ID = "id"
TARGET = "label"

AUTO = tf.data.experimental.AUTOTUNE

try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy() 
print("REPLICAS: ", strategy.num_replicas_in_sync)

GCS_DS_PATH = KaggleDatasets().get_gcs_path("tpu-getting-started")

IMAGE_SIZE = [512, 512] # при таком размере графическому процессору не хватит памяти. Используйте TPU
BATCH_SIZE = 16 * strategy.num_replicas_in_sync

GCS_PATH_SELECT = { # available image sizes
    192: GCS_DS_PATH + '/tfrecords-jpeg-192x192',
    224: GCS_DS_PATH + '/tfrecords-jpeg-224x224',
    331: GCS_DS_PATH + '/tfrecords-jpeg-331x331',
    512: GCS_DS_PATH + '/tfrecords-jpeg-512x512'
}
GCS_PATH = GCS_PATH_SELECT[IMAGE_SIZE[0]]

VALIDATION_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/val/*.tfrec')
TEST_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/test/*.tfrec')

SEED = 2022
def seed_everything(seed=SEED):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
seed_everything()


# # Build Model

# In[ ]:


def decode_image(image_data):
    image = tf.image.decode_jpeg(image_data, channels=3) 
    image = tf.cast(image, tf.float32) / 255.0  
    image = tf.reshape(image, [*IMAGE_SIZE, 3]) 
    
    return image

def read_labeled_tfrecord(example):
    LABELED_TFREC_FORMAT = {
        "image": tf.io.FixedLenFeature([], tf.string), 
        "class": tf.io.FixedLenFeature([], tf.int64),  
    }
    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT) 
    image = decode_image(example['image']) 
    label = tf.cast(example['class'], tf.int32)
    return image, label 

def read_unlabeled_tfrecord(example):
    UNLABELED_TFREC_FORMAT = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "id": tf.io.FixedLenFeature([], tf.string), 
    }
    example = tf.io.parse_single_example(example, UNLABELED_TFREC_FORMAT)
    image = decode_image(example['image']) 
    idnum = example['id']
    return image, idnum 

def load_dataset(filenames, labeled=True, ordered=False):
    ignore_order = tf.data.Options() 
    if not ordered:
        ignore_order.experimental_deterministic = False 

    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO) 
    dataset = dataset.with_options(ignore_order) 
    dataset = dataset.map(read_labeled_tfrecord if labeled else read_unlabeled_tfrecord, num_parallel_calls=AUTO)
    
    return dataset

def get_validation_dataset():
    dataset = load_dataset(VALIDATION_FILENAMES, labeled=True, ordered=True)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.cache() 
    dataset = dataset.prefetch(AUTO) 
    return dataset

def get_test_dataset(ordered=False):
    dataset = load_dataset(TEST_FILENAMES, labeled=False, ordered=ordered)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTO) 
    return dataset
                               
def count_data_items(filenames):
    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]
    return np.sum(n)

NUM_VALIDATION_IMAGES = count_data_items(VALIDATION_FILENAMES)
NUM_TEST_IMAGES = count_data_items(TEST_FILENAMES)
print('Dataset: {} validation images, {} unlabeled test images'.format(NUM_VALIDATION_IMAGES, NUM_TEST_IMAGES))

def get_model(use_model):
    base_model = use_model(weights='imagenet', 
                      include_top=False, pooling='avg',
                      input_shape=(*IMAGE_SIZE, 3))
#     base_model.trainable = False
    x = base_model.output
    predictions = Dense(104, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(
                    optimizer='nadam',
                    loss = 'sparse_categorical_crossentropy',
                    metrics=['sparse_categorical_accuracy']
                 )
    return model
with strategy.scope():    
    model1 = get_model(DenseNet201)
model1.load_weights("/kaggle/input/densenet201-aug-additional-data/my_denceNet_201.h5")

with strategy.scope():    
    model2 = get_model(Xception) 
model2.load_weights("/kaggle/input/xception-aug-additional-data/my_Xception.h5") 


# # Predict Data

# In[ ]:


test_ds = get_test_dataset(ordered=True) 
best_alpha = 0.45
test_images_ds = test_ds.map(lambda image, idnum: image)

probabilities1 = model1.predict(test_images_ds)
probabilities2 = model2.predict(test_images_ds)
probabilities = best_alpha * probabilities1 + (1 - best_alpha) * probabilities2
predictions = np.argmax(probabilities, axis=-1)

test_ids_ds = test_ds.map(lambda image, idnum: idnum).unbatch()
test_ids = next(iter(test_ids_ds.batch(NUM_TEST_IMAGES))).numpy().astype('U') 
np.savetxt(SUBMISSION_PATH, np.rec.fromarrays([test_ids, predictions]), fmt=['%s', '%d'], delimiter=',', header='id,label', comments='')

sub = pd.read_csv(SUBMISSION_PATH)
sub.head()

