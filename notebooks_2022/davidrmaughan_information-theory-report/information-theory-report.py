#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
import numpy as np
from sklearn.metrics import roc_auc_score


# In[ ]:


# adapted from this
# https://github.com/keras-team/keras-applications/blob/master/keras_applications/resnet50.py

#from __future__ import absolute_import
#from __future__ import division
#from __future__ import print_function

import os
from tensorflow.keras import layers, models
import warnings

#from . import get_submodules_from_kwargs
#from . import imagenet_utils
#from .imagenet_utils import decode_predictions
#from .imagenet_utils import _obtain_input_shape

#preprocess_input = imagenet_utils.preprocess_input

WEIGHTS_PATH = ('https://github.com/fchollet/deep-learning-models/'
                'releases/download/v0.2/'
                'resnet50_weights_tf_dim_ordering_tf_kernels.h5')
WEIGHTS_PATH_NO_TOP = ('https://github.com/fchollet/deep-learning-models/'
                       'releases/download/v0.2/'
                       'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')
"""
backend = None
layers = None
models = None
keras_utils = None
"""

def identity_block(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    """
    if backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    """
    bn_axis = 3
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.Conv2D(filters1, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2a')(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters2, kernel_size,
                      padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2b')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters3, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2c')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = layers.Activation('relu')(x)
    return x


def conv_block(input_tensor,
               kernel_size,
               filters,
               stage,
               block,
               strides=(2, 2)):
    """A block that has a conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        strides: Strides for the first conv layer in the block.
    # Returns
        Output tensor for the block.
    Note that from stage 3,
    the first conv layer at main path is with strides=(2, 2)
    And the shortcut should have strides=(2, 2) as well
    """
    filters1, filters2, filters3 = filters
    """
    if backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    """
    bn_axis = 3
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.Conv2D(filters1, (1, 1), strides=strides,
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2a')(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters2, kernel_size, padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2b')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters3, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2c')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = layers.Conv2D(filters3, (1, 1), strides=strides,
                             kernel_initializer='he_normal',
                             name=conv_name_base + '1')(input_tensor)
    shortcut = layers.BatchNormalization(
        axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)
    return x


def ResNet50(include_top=True,
             weights='imagenet',
             input_tensor=None,
             input_shape=None,
             pooling=None,
             classes=1000,
             spatial_dropout_rate=0,
             **kwargs):
    """Instantiates the ResNet50 architecture.
    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.
    # Arguments
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or `(3, 224, 224)` (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 32.
            E.g. `(200, 200, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional block.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional block, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
    # Returns
        A Keras model instance.
    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    #global backend, layers, models, keras_utils
    #backend, layers, models, keras_utils = get_submodules_from_kwargs(kwargs)

    if not (weights in {'imagenet', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as `"imagenet"` with `include_top`'
                         ' as true, `classes` should be 1000')

    # Determine proper input shape
    """
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=32,
                                      data_format=backend.image_data_format(),
                                      require_flatten=include_top,
                                      weights=weights)
    """

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        """
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
        """
        img_input = input_tensor
    """
    if backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    """
    bn_axis = 3
    x = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(img_input)
    x = layers.Conv2D(64, (7, 7),
                      strides=(2, 2),
                      padding='valid',
                      kernel_initializer='he_normal',
                      name='conv1')(x)
    x = layers.BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = layers.Activation('relu')(x)
    x = layers.ZeroPadding2D(padding=(1, 1), name='pool1_pad')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')
    x = tf.keras.layers.SpatialDropout2D(spatial_dropout_rate)(x)#, training=True)
    #x = spatial_dropout(x, 1-spatial_dropout_rate)
    #x = tf.keras.layers.Dropout(spatial_dropout_rate,noise_shape=(x.get_shape()[0], 1, 1, x.get_shape()[3]))(x, training=True)

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')
    x = tf.keras.layers.SpatialDropout2D(spatial_dropout_rate)(x)#, training=True)

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')
    x = tf.keras.layers.SpatialDropout2D(spatial_dropout_rate)(x)#, training=True)

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    if include_top:
        x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        x = layers.Dense(classes, activation='softmax', name='fc1000')(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D()(x)
        else:
            warnings.warn('The output shape of `ResNet50(include_top=False)` '
                          'has been changed since Keras 2.2.0.')

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = tf.keras.utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = models.Model(inputs, x, name='resnet50')

    # Load weights.
    if weights == 'imagenet':
        if include_top:
            weights_path = keras_utils.get_file(
                'resnet50_weights_tf_dim_ordering_tf_kernels.h5',
                WEIGHTS_PATH,
                cache_subdir='models',
                md5_hash='a7b3fe01876f51b976af0dea6bc144eb')
        else:
            weights_path = tf.keras.utils.get_file(
                'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                WEIGHTS_PATH_NO_TOP,
                cache_subdir='models',
                md5_hash='a268eb855778b3df3c7506639542a6af')
        model.load_weights(weights_path)
        #if backend.backend() == 'theano':
        #    keras_utils.convert_all_kernels_in_model(model)
    elif weights is not None:
        model.load_weights(weights)

    return model


# In[ ]:


image_size = 224
data_augmentation = tf.keras.Sequential(
    [
        #layers.Normalization(),
        tf.keras.layers.Resizing(image_size, image_size),
        
    ],
    name="data_augmentation",)

def get_model(num_classes,dropout_rate,spatial_dropout_rate):
    inputs = tf.keras.layers.Input(shape=(32,32,3))
    aug_input = data_augmentation(inputs)
    model = ResNet50(input_shape=(image_size,image_size,3),include_top=False,input_tensor=aug_input,spatial_dropout_rate=spatial_dropout_rate)
    x = tf.keras.layers.GlobalAveragePooling2D()(model.layers[-1].output)
    if dropout_rate > 0:
        x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.Dense(num_classes,activation='softmax')(x)
    return tf.keras.Model(inputs,x)

model = get_model(10,0,.2)
#model.summary()
    


# In[ ]:


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
# shuffle so that you can keep x_test independent 
(x_train_100, y_train_100), (x_test_100, y_test_100) = tf.keras.datasets.cifar100.load_data()
def scale_x(x):
    #x = 2*x/255
    #x = x - 1
    return tf.keras.applications.resnet50.preprocess_input(x)

x_train = scale_x(x_train)
x_test = scale_x(x_test)
x_train_100 = scale_x(x_train_100)
x_test_100 = scale_x(x_test_100)
len(x_train)


# In[ ]:


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

train_size = 45000
val_start = 45000
x_t, y_t = unison_shuffled_copies(x_train,y_train)
x_train, x_val = x_t[:train_size], x_t[val_start:]
y_train, y_val = y_t[:train_size], y_t[val_start:]


# In[ ]:


len(x_train),len(x_val),len(y_train),len(y_val)


# In[ ]:


model.compile('sgd','sparse_categorical_crossentropy',metrics=['accuracy'])


# In[ ]:


es = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=2, restore_best_weights=True)
model.fit(x_train,y_train,validation_data=(x_val,y_val),epochs=150,callbacks=[es])


# In[ ]:


model.evaluate(x_test,y_test)


# In[ ]:


def predict_as_training(model,x,bs=100):
    scores = np.zeros((len(x),10))
    for ii in range(0,len(x),bs):
        scores[ii:ii+bs] = model(x[ii:ii+bs],training=True)
    return scores


# In[ ]:





# In[ ]:





# In[ ]:


def get_max_softmax(y_pred):
    return np.amax(y_pred,axis=1)

def get_entropy(y_pred):
    logs = -np.log2(y_pred+1e-6)
    temp = logs*y_pred
    return np.sum(temp,axis=1)


# In[ ]:


def get_max_softmax_roc(model,x):
    y_pred = model.predict(x)
    y_true = [1]*10000 + [0]*10000
    y_pred = get_max_softmax(y_pred)
    return roc_auc_score(y_true,y_pred)

def get_softmax_entropy_roc(model,x):
    y_pred = model.predict(x)
    y_true = [0]*10000 + [1]*10000
    y_pred = get_entropy(y_pred)
    return roc_auc_score(y_true,y_pred)

def get_mutual_information_roc(model,x,num_runs=3):
    y_true = [0]*10000 + [1]*10000
    distributions = []
    for i in range(num_runs):
        distributions.append(predict_as_training(model,x))
        
    average_distributions = distributions[0]
    for i in range(1,num_runs):
        average_distributions = average_distributions + distributions[i]
    average_distributions = average_distributions / num_runs
    
    
    average_distribution_entropy = get_entropy(average_distributions)
    
    softmax_entropies = []
    for i in range(num_runs):
        softmax_entropies.append(get_entropy(distributions[i]))
    average_of_entropies = get_entropy(distributions[0])
    for i in range(1,num_runs):
        average_of_entropies = average_of_entropies + get_entropy(distributions[i])
    y_pred = average_distribution_entropy - average_of_entropies/num_runs
    mutual_information_roc = roc_auc_score(y_true,y_pred)
    y_pred = average_distribution_entropy
    predictive_entropy_roc = roc_auc_score(y_true,y_pred)
    return mutual_information_roc, predictive_entropy_roc
    
        


# In[ ]:


import gc
gc.collect()


# In[ ]:


softmax_entropy_roc = get_softmax_entropy_roc(model,np.concatenate((x_test,x_test_100)))
max_softmax_roc = get_max_softmax_roc(model,np.concatenate((x_test,x_test_100)))


# In[ ]:


mutual_information_roc, predictive_entropy_roc = get_mutual_information_roc(model,np.concatenate((x_test,x_test_100)),10)


# In[ ]:


softmax_entropy_roc, max_softmax_roc, mutual_information_roc, predictive_entropy_roc 


# In[ ]:


# .5 dropout 
# 1) (0.916898015, 0.91155739, 0.9080468200000001, 0.90977567) 95.6
# 2) (0.9219112449999999, 0.915462275, 0.90541482, 0.90693588) 95.6
# 3) (0.921514355, 0.916517885, 0.9101641600000001, 0.91168075) 96.0

# .2 spatial dropout
# 1) (0.9260553800000001, 0.920252355, 0.9044787899999999, 0.92521013) 96.3
# 2) (0.924613495, 0.9185143849999999, 0.9036850599999999, 0.9231581699999999) 95.9
# 3) (0.926366895, 0.9205227500000002, 0.90567517, 0.92422546) 96.1

# Note that in this paper from Google and Stanford https://arxiv.org/pdf/2106.03004.pdf
# they used ResNet50 to achieve an AUC ROC of 85.8 with Maximum Softmax Probability
# whereas we achieved an AUC ROC of 92.1  in this kaggle kernel. 

