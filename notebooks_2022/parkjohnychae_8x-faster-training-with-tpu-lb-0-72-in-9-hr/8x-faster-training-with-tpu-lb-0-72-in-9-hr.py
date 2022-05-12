#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# 
# ## Acknowledgments
# Guide for using TPU from tensorflow:
# 
# https://www.tensorflow.org/guide/tpu
# 
# https://www.tensorflow.org/tutorials/load_data/tfrecord
# 
# Also adapted part from Dimitre Oliveria's code below:
# 
# https://keras.io/examples/keras_recipes/creating_tfrecords/
# 
# And cdeotte's notebook below:
# 
# https://www.kaggle.com/code/cdeotte/how-to-create-tfrecords/notebook
# 

# ## Introduction
# 
# Hi everyone,
# 
# I hope you guys are having fun here so far!
# 
# We have about 3 weeks until the competition deadline. 3 weeks is a short timeframe for someone, but it could be enough for others, especially if they want to learn and have fun. 
# 
# I think this competition could be frustratinig beacuse of relatively bigger size of data compared to other competitions. It's actually an extremely good thing we have lots of labeled data. But we got limited time and resources for the competition, and not everyone has access to expensive GPUs with high VRAMs.
# 
# But don't give up! TPUs are faster than GPUs, but many of us may are not fully utilizing them as we don't see any discussions or notebooks about TPU training. To fill this gap, I decided to share Tfrecord data I created, to facilitate TPU training with our data. It takes about 8 - 9 hours in TPU time to train standard ResNet50 for good performance (LB =0.72). So no worries if you think you are late for the party. You are not too late as long as you have TPU time left in your kaggle account!
# 
# 

# ## TFRecords
# 
# In order to use TPU, you need to prepare the data in a specific format, TFrecord. In TFrecord, images are stored in bytes which enables faster data reading. I created TFrecords in 480x480 resolution for training and testing. Here are the links:
# - [TFREC 480 TRAINING](https://www.kaggle.com/datasets/parkjohnychae/herbarium-2022-train-tfrec-480) 
# - [TFREC 480 TESTING](https://www.kaggle.com/datasets/parkjohnychae/herbarium-2022-test-tfrec-480)

# ## Install packages
# 

# In[ ]:


get_ipython().system('pip install -q mytflib')
get_ipython().system('pip install -q one-cycle-tf')


# In[ ]:


import tensorflow as tf
import mytflib as tfl
import tensorflow_addons as tfa
import os
from one_cycle_tf import OneCycle


# ## Set up TPU 

# In[ ]:


try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect() # TPU detection
    strategy = tf.distribute.TPUStrategy(tpu)
except ValueError:
    strategy = tf.distribute.MirroredStrategy() # for GPU or multi-GPU machines
print("Number of accelerators: ", strategy.num_replicas_in_sync)


# ## GET GS BUCKET ADDRESS FOR TFREC
# 
# We need to get gsbucket address from each data to use TFRECs. Gsbucket is a type cloud stroage serviced in google cloud service. Kaggle datasets are stored in gsbuckets. 

# In[ ]:


from kaggle_datasets import KaggleDatasets
trPATH = KaggleDatasets().get_gcs_path('herbarium-2022-train-tfrec-480')
ttPATH = KaggleDatasets().get_gcs_path('herbarium-2022-test-tfrec-480')


# In[ ]:





# ## GET TFREC LIST TRAIN & TEST DATA

# In[ ]:


tr_fns = tf.io.gfile.glob(trPATH +"/*train*.tfrec")
tt_fns = tf.io.gfile.glob(ttPATH +"/*test*.tfrec")
metadata_fns = tf.io.gfile.glob(trPATH +"/*metadata*")
N_tt_imgs = tfl.count_tfrec_items(tt_fns)
N_tr_imgs = tfl.count_tfrec_items(tr_fns)
print("{} training images, and {} testing images.".format(N_tr_imgs, N_tt_imgs))
metadata_fns


# ## DISPLAYING TRAIN & TEST TFREC

# In[ ]:


### TRAIN TFREC
tfl.display_sample_from_TFrec(tfrec_PATH = tr_fns[0], TFREC_FORMAT = {"image":"str",
                                  "image_id":"str",
                                  "scientificName":"int",
                                 "family":"int" ,
                                  "genus":"int"
                                  } , display_size = (15,10)
                         )
### TEST TFREC
tfl.display_sample_from_TFrec(tfrec_PATH = tt_fns[0], TFREC_FORMAT = {"image":"str",
                                  "image_id":"int"}
                                  , display_size = (15,10)
                         )


# ## LABEL MAPPING 
# 
# The TFREC data labels are bit different from that of the herbarium 2022 metadata. It is because in the metadata it has some missing values due to quality assurance. 
# 
# ### Feature description of the training tfrecord:
# ```
#      {
#        'image': 'bytes',
#        'image_id': 'bytes',
#         'family': 'int',
#         'genus': 'int',
#         'scientificName': 'int'
#        }
# ```
# ### Feature description of the testing tfrecord:
# ```
#      {
#        'image': 'bytes',
#        'image_id': 'int'
#        }
# ```
# 
# 
# ### Some complications in the training data labels:
# 
# Family, genus, and scientificName are converted to integers in alphabetical order. Please note that the labels are different from the competition metadata.
# 
# Therefore, it needs to be converted back to its original mapping (i.e. ```catgeroical_id``` and ```genus_id``` in the competition metadata) when doing the inference. It is because some scientificNames got removed from the label during quilty check (to be precise, four of them). Thus, ```len(categorical_id) = 15501```, but ```max, min = (15504, 0)``` for ```categorical_id``` in the metadata, whereas ```len(scientificName) = 15501```  and ```max, min = (15500, 0)``` for ```scientificName``` in TFREC-480.
# 

# 
# ####  How to revert ```(int): scientificName``` back to ```(int) categorical_id```: 
# 
# get two mappings below and use later in the inference 

# In[ ]:


import pandas as pd
df_train = pd.read_table('../input/herbarium-2022-metadata/herbarium2022_train.tsv')
map_name_to_cat_id = dict(zip(df_train.scientificName, df_train.category_id))
map_label_to_name = dict(zip(range(15501), sorted(set(df_train.scientificName))))


# ## CONFIGURATION

# In[ ]:


config_dict = dict()
config_dict["ls_train_files"] = tr_fns
config_dict["tfrec_structure"] = {"image":"str",
                                  "image_id":"str",
                                  "scientificName":"int",
                                 "family":"int" ,
                                  "genus":"int"
                                  }
config_dict["tfrec_shape"] =[480, 480]
config_dict["resize_resol"] =[380, 380]
config_dict["N_cls"] = len(set(df_train.scientificName))
config_dict["batch_size"] = 32*strategy.num_replicas_in_sync

Ntot_train = tfl.count_tfrec_items(config_dict["ls_train_files"])
tr_ds = tfl.get_train_ds_tfrec_from_dict(config_dict, 
                                     label_name = "scientificName", 
                                     DataRepeat =True) 

STEPS_PER_EPOCH = int(N_tr_imgs/config_dict["batch_size"])

config_dict["steps_per_epoch"] = STEPS_PER_EPOCH


# ## MODEL
# 
# We are going to use ResNet50 with bottleneck head in this notebook.

# In[ ]:


def get_model(num_classes, resize_resol):

    base_model = tf.keras.applications.resnet50.ResNet50(
              
        input_shape=(*resize_resol, 3),
        include_top=False,
        pooling="avg",
        weights="imagenet"
    )   
    model=tf.keras.Sequential([
        base_model,
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1024, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation="softmax")
 
    ])

    return model


# ## COMPILE

# In[ ]:


from one_cycle_tf import OneCycle

N_EPOCH = 10
cycle_size = N_EPOCH*STEPS_PER_EPOCH
mLR = 6e-1
iLR = mLR/30
ocLR = OneCycle(initial_learning_rate=iLR,
             maximal_learning_rate=mLR,
             cycle_size = cycle_size,
             final_lr_scale = 1e-3)

with strategy.scope():
    model = get_model(config_dict['N_cls'], config_dict['resize_resol'])
    model.compile(
        optimizer= tfa.optimizers.SGDW(learning_rate = ocLR, 
                                       weight_decay = 3e-5, 
                                       momentum = 0.9,
                                       nesterov = True),
        loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing = 0.2),
        metrics = tfa.metrics.F1Score(num_classes =config_dict['N_cls'])
    )


# In[ ]:


import os


# ## TRAINING

# In[ ]:


OutFileName = "ResNet50_bn1024_OneCycle_CE-ls2e-1__"
oPATH = "./"
history = model.fit(
    tr_ds, 
    epochs= N_EPOCH, steps_per_epoch = config_dict["steps_per_epoch"],
    verbose=1,
  callbacks=[tfl.SaveModelHistory(config_dict,
                               OutFileName,oPATH)])
oPATHw = os.path.join(oPATH, OutFileName+"_weights.h5")
model.save_weights(oPATHw) 


# In[ ]:


import pandas as pd
train_history = pd.read_csv(os.path.join(oPATH,OutFileName+".csv"))
import matplotlib.pyplot as plt

plt.plot(range(10),train_history.loss)
plt.plot(range(10),train_history.f1_score)


# ## INFERENCE

# In[ ]:


### Inference functions
tt_fns = tf.io.gfile.glob(ttPATH + '/*test*.tfrec')
N_tt_imgs = tfl.count_tfrec_items(tt_fns)
print('Dataset: {} test images'.format(N_tt_imgs))

test_dict = {"image":"str", "image_id":"int"}

AUTO = tf.data.experimental.AUTOTUNE

def _load_tfrec_dataset(filenames, tfrec_format, tfrec_sizes, label_name, labeled=True, ordered=False):
    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO) 
    dataset = dataset.map(lambda Example: tfl.read_tfrecord(Example, 
                                                        TFREC_FORMAT = tfrec_format, 
                                                        TFREC_SIZES = tfrec_sizes,
                                                        LABEL_NAME = label_name))
    return dataset

def prepare_test_images(image, label, resize_factor):
    img = tf.image.central_crop(image, central_fraction = 0.9)
    img = tf.image.resize( img, size = resize_factor)
    return img, label

def get_test_ds_tfrec(LS_FILENAMES, TFREC_DICT, TFREC_SIZES, RESIZE_FACTOR, NUM_CLASSES, BATCH_SIZE, LABEL_NAME, MoreAugment = False):

    tfrec_format = tfl.tfrec_format_generator(TFREC_DICT)
    dataset = _load_tfrec_dataset(LS_FILENAMES, tfrec_format = tfrec_format, tfrec_sizes = TFREC_SIZES, 
                                  label_name = LABEL_NAME)
    dataset = dataset.map(tfl.normalize_RGB, num_parallel_calls=AUTO).prefetch(AUTO)
    dataset = dataset.map(lambda image, label: prepare_test_images(image, label, resize_factor = RESIZE_FACTOR), num_parallel_calls=AUTO)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTO)
    return dataset


# In[ ]:


tt_ds = get_test_ds_tfrec( LS_FILENAMES = tt_fns,
                              TFREC_DICT = test_dict,
                              TFREC_SIZES =  config_dict["tfrec_shape"],
                              RESIZE_FACTOR = config_dict["resize_resol"],
                              NUM_CLASSES = config_dict["N_cls"],
                              BATCH_SIZE = config_dict["batch_size"],
                            LABEL_NAME = "image_id"
                          )

test_images_ds = tt_ds.map(lambda image, idnum: image)
test_Ids_ds = tt_ds.map(lambda image, idnum: idnum)


# In[ ]:


import pandas as pd
df_train = pd.read_table('../input/herbarium-2022-metadata/herbarium2022_train.tsv')
map_name_to_cat_id = dict(zip(df_train.scientificName, df_train.category_id))
map_label_to_name = dict(zip(range(15501), sorted(set(df_train.scientificName))))


# In[ ]:


import numpy as np
from tqdm import tqdm

predictions = np.zeros(N_tt_imgs, dtype=np.int32)

for i, image in tqdm(enumerate(test_images_ds), total= (N_tt_imgs//config_dict["batch_size"] + 1)):
    idx1 = i*config_dict["batch_size"]
    if (idx1 + config_dict["batch_size"]) > N_tt_imgs:
        idx2 = N_tt_imgs
    else:
        idx2 = idx1 + config_dict["batch_size"]
    predictions[idx1:idx2] = np.argmax(model.predict_on_batch(image), axis=-1)


# In[ ]:


predict_image_nums = np.zeros(N_tt_imgs, dtype=np.int32)

for i, image_nums in tqdm(enumerate(test_Ids_ds), total= (N_tt_imgs//config_dict["batch_size"] + 1)):
    idx1 = i*config_dict["batch_size"]
    if (idx1 + config_dict["batch_size"]) > N_tt_imgs:
        idx2 = N_tt_imgs
    else:
        idx2 = idx1 + config_dict["batch_size"]
    predict_image_nums[idx1:idx2] = image_nums

prediction_cat_id = [map_name_to_cat_id[map_label_to_name[ele]] for ele in predictions]

pd.DataFrame({"Id":predict_image_nums,"Predicted":prediction_cat_id}).to_csv("sample_submissions.csv",index=False )

