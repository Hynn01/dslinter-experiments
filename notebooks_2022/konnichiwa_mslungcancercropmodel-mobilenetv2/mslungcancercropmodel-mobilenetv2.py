#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, cohen_kappa_score, precision_score, recall_score, accuracy_score, confusion_matrix
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.models import Sequential
from keras import optimizers
from keras.applications import *
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
#from keras.layers.normalization import BatchNormalization
from keras.layers import Input,Dense,Dropout,BatchNormalization,Conv2D,MaxPooling2D,AveragePooling2D,concatenate,Activation,ZeroPadding2D
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, multiply, Permute, Add,Lambda, Concatenate
from keras.models import Model, Sequential
from keras.applications.xception import Xception
from keras.applications import *
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, TensorBoard
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from PIL import Image
import glob
import random
import os
import cv2
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier 
from keras.callbacks import EarlyStopping
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from sklearn.decomposition import PCA
from tensorflow.keras.utils import to_categorical

#Max batch size= available GPU memory bytes / 4 / (size of tensors + trainable parameters)
#model.summary()


# In[ ]:


lung_aca = "../input/lung-and-colon-cancer-histopathological-images/lung_colon_image_set/lung_image_sets/lung_aca/"

plt.figure(figsize = (10, 10))
plt.subplot(131)
img = cv2.imread(lung_aca + os.listdir(lung_aca)[0])
plt.title('Lung ACA') # lung adenocarcinoma (ACA)
plt.imshow(img)

plt.subplot(132)
lung_n = "../input/lung-and-colon-cancer-histopathological-images/lung_colon_image_set/lung_image_sets/lung_n/"
img = cv2.imread(lung_n + os.listdir(lung_n)[0])
plt.title('Lung N')
plt.imshow(img)

plt.subplot(133)
lung_scc = "../input/lung-and-colon-cancer-histopathological-images/lung_colon_image_set/lung_image_sets/lung_scc/"
img = cv2.imread(lung_scc + os.listdir(lung_scc)[0])
plt.title('Lung SCC') #small cell carcinomas (SCCs)
plt.imshow(img)

colon_aca= "../input/lung-and-colon-cancer-histopathological-images/lung_colon_image_set/colon_image_sets/colon_aca/"
plt.figure(figsize = (10, 10))
plt.subplot(131)
img = cv2.imread(colon_aca + os.listdir(colon_aca)[0])
plt.title('Colon ACA')
plt.imshow(img)

plt.subplot(132)
colon_n = "../input/lung-and-colon-cancer-histopathological-images/lung_colon_image_set/colon_image_sets/colon_n/"
img = cv2.imread(colon_n + os.listdir(colon_n)[0])
plt.title('Colon N')
plt.imshow(img)
plt.show()


# In[ ]:


data_dir = "../input/lung-and-colon-cancer-histopathological-images/lung_colon_image_set/lung_image_sets/"

SIZE_X = SIZE_Y = 128

datagen = tf.keras.preprocessing.image.ImageDataGenerator(validation_split = 0.3)

train_it = datagen.flow_from_directory(data_dir,
                                       class_mode = "categorical",
                                       target_size = (SIZE_X,SIZE_Y),
                                       color_mode="rgb",
                                       batch_size = 32, 
                                       shuffle = False,
                                       subset='training',
                                       seed = 42)

validate_it = datagen.flow_from_directory(data_dir,
                                       class_mode = "categorical",
                                       target_size = (SIZE_X, SIZE_Y),
                                       color_mode="rgb",
                                       batch_size = 32, 
                                       shuffle = False,
                                       subset='validation',
                                       seed = 42)


# In[ ]:


def fit_model(model, train_it, validate_it, epochs = 10):
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
    
    for layer in model.layers:
        layer.trainable = False
    
    flat1 = Flatten()(model.layers[-1].output)
    output = Dense(len(train_it.class_indices), activation='softmax')(flat1)
    
    model = Model(inputs=model.inputs, outputs=output)
    print(model.summary())
    
    model.compile(loss='categorical_crossentropy', optimizer ='adam', metrics=['accuracy'])
    model.summary()
    history = model.fit(train_it, validation_data=validate_it, epochs=epochs, verbose=1, callbacks=[es])
    model.evaluate(validate_it)
    return model


# In[ ]:


def get_accuracy_metrics(model, train_it, validate_it):
    y_val = validate_it.classes
    
    val_pred_proba = model.predict(validate_it)
    
    val_pred_proba, predicted_proba, y_val, y_test = train_test_split(val_pred_proba, y_val, test_size = 0.5, shuffle = True)
    
    val_pred = np.argmax(val_pred_proba, axis = 1)
    predicted = np.argmax(predicted_proba, axis = 1)
    
    print("Train accuracy Score------------>")
    print ("{0:.3f}".format(accuracy_score(train_it.classes, np.argmax(model.predict(train_it), axis = 1))*100), "%")
    
    print("Val accuracy Score--------->")
    print("{0:.3f}".format(accuracy_score(y_val, val_pred)*100), "%")
    
    print("Test accuracy Score--------->")
    print("{0:.3f}".format(accuracy_score(y_test, predicted)*100), "%")
    
    print("F1 Score--------------->")
    print("{0:.3f}".format(f1_score(y_test, predicted, average = 'weighted')*100), "%")
    
    print("Cohen Kappa Score------------->")
    print("{0:.3f}".format(cohen_kappa_score(y_test, predicted)*100), "%")
    
    
    print("ROC AUC Score------------->")
    print("{0:.3f}".format(roc_auc_score(to_categorical(y_test, num_classes = 3), predicted_proba, multi_class='ovr')*100), "%")
    
    print("Recall-------------->")
    print("{0:.3f}".format(recall_score(y_test, predicted, average = 'weighted')*100), "%")
    
    print("Precision-------------->")
    print("{0:.3f}".format(precision_score(y_test, predicted, average = 'weighted')*100), "%")
    
    cf_matrix_test = confusion_matrix(y_test, predicted)
    cf_matrix_val = confusion_matrix(y_val, val_pred)
    
    plt.figure(figsize = (12, 6))
    plt.subplot(121)
    sns.heatmap(cf_matrix_val, annot=True, cmap='Blues')
    plt.title("Val Confusion matrix")
    
    plt.subplot(122)
    sns.heatmap(cf_matrix_test, annot=True, cmap='Blues')
    plt.title("Test Confusion matrix")
    
    plt.show()


# In[ ]:


def focal_loss(gamma=2.):            
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        return -K.sum( K.pow(1. - pt_1, gamma) * K.log(pt_1)) 
    return focal_loss_fixed


# In[ ]:


def Conv2d_BN(x, nb_filter,kernel_size, strides=(1,1), padding='same',name=None):  
    if name is not None:  
        bn_name = name + '_bn'  
        conv_name = name + '_conv'  
    else:  
        bn_name = None  
        conv_name = None  
  
    x = Conv2D(nb_filter,kernel_size,padding=padding,strides=strides,activation='relu',name=conv_name)(x)  
    x = BatchNormalization(axis=3,name=bn_name)(x)  
    return x  

def Conv_Block(inpt,nb_filter,kernel_size,strides=(1,1), with_conv_shortcut=False):  
    x = Conv2d_BN(inpt,nb_filter=nb_filter[0],kernel_size=(1,1),strides=strides,padding='same')  
    x = Conv2d_BN(x, nb_filter=nb_filter[1], kernel_size=(3,3), padding='same')  
    x = Conv2d_BN(x, nb_filter=nb_filter[2], kernel_size=(1,1), padding='same')  
    if with_conv_shortcut:  
        shortcut = Conv2d_BN(inpt,nb_filter=nb_filter[2],strides=strides,kernel_size=kernel_size)  
        x = add([x,shortcut])  
        return x  
    else:  
        x = add([x,inpt])  
        return x


# In[ ]:


def channel_attention(input_feature, ratio=8): #final features are attained
    print("Channel Attention:")
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    #channel = input_feature._keras_shape[channel_axis]
    channel = input_feature.shape[channel_axis]
    shared_layer_one = Dense(channel//ratio,
                            kernel_initializer='he_normal',
                            activation = 'relu',
                            use_bias=True,
                            bias_initializer='zeros')

    shared_layer_two = Dense(channel,
                            kernel_initializer='he_normal',
                            use_bias=True,
                            bias_initializer='zeros')

    avg_pool = GlobalAveragePooling2D()(input_feature)    
    avg_pool = Reshape((1,1,channel))(avg_pool)
    #assert avg_pool._keras_shape[1:] == (1,1,channel)
    assert avg_pool.shape[1:] == (1,1,channel)
    avg_pool = shared_layer_one(avg_pool)
    #assert avg_pool._keras_shape[1:] == (1,1,channel//ratio)
    assert avg_pool.shape[1:] == (1,1,channel//ratio)    
    avg_pool = shared_layer_two(avg_pool)
    #assert avg_pool._keras_shape[1:] == (1,1,channel)
    assert avg_pool.shape[1:] == (1,1,channel)    

    max_pool = GlobalMaxPooling2D()(input_feature)
    max_pool = Reshape((1,1,channel))(max_pool)
    #assert max_pool._keras_shape[1:] == (1,1,channel)
    assert max_pool.shape[1:] == (1,1,channel)    
    max_pool = shared_layer_one(max_pool)
    #assert max_pool._keras_shape[1:] == (1,1,channel//ratio)
    assert max_pool.shape[1:] == (1,1,channel//ratio)    
    max_pool = shared_layer_two(max_pool)
    #assert max_pool._keras_shape[1:] == (1,1,channel)
    assert max_pool.shape[1:] == (1,1,channel)    

    cbam_feature = Add()([avg_pool,max_pool])
    cbam_feature = Activation('hard_sigmoid')(cbam_feature)
    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)
    return multiply([input_feature, cbam_feature])


# In[ ]:


#the spatial attention concatenates the final features attained by channel
#attention and convolved by a regular convolution layer, thereby generating the spatial attention map.
def spatial_attention(input_feature): 
    print("Spatial Attnetion:")
    kernel_size = 7
    if K.image_data_format() == "channels_first":
        #channel = input_feature._keras_shape[1]
        channel = input_feature.shape[1]        
        cbam_feature = Permute((2,3,1))(input_feature)
    else:
        #channel = input_feature._keras_shape[-1]
        channel = input_feature.shape[-1]        
        cbam_feature = input_feature

    avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)
    #assert avg_pool._keras_shape[-1] == 1
    assert avg_pool.shape[-1] == 1    
    max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)
    #assert max_pool._keras_shape[-1] == 1
    assert max_pool.shape[-1] == 1    
    concat = Concatenate(axis=3)([avg_pool, max_pool])
    #assert concat._keras_shape[-1] == 2
    assert concat.shape[-1] == 2    
    cbam_feature = Conv2D(filters = 1,
                        kernel_size=kernel_size,
                        activation = 'hard_sigmoid',
                        strides=1,
                        padding='same',
                        kernel_initializer='he_normal',
                        use_bias=False)(concat)
    #assert cbam_feature._keras_shape[-1] == 1
    assert cbam_feature.shape[-1] == 1    

    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)
    return multiply([input_feature, cbam_feature])


# In[ ]:


def cbam_block(cbam_feature,ratio=1):
    cbam_feature = channel_attention(cbam_feature, ratio)
    cbam_feature = spatial_attention(cbam_feature, )
    return cbam_feature


# In[ ]:


#data_dir = "../input/lung-and-colon-cancer-histopathological-images/lung_colon_image_set/lung_image_sets/"
#batch_size increased from 32 to 64 by MS 5/5/2022
batch_size=64
#Reduce from 30 to 17 5/5/2022 by MS 
epochs = 10

#Log directory for TensorBoard
board_name1 = "./obj_reco/stage1/' + now + '/"
board_name2 = "./obj_reco/stage2/' + now + '/"
nb_train_samples = len(glob.glob(data_dir + '/*/*.*'))  
nb_validation_samples = len(glob.glob(data_dir + '/*/*.*'))    
#---------Attention embedded MobileNetV2--------------------------------------------------------------
import tensorflow as tf
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras import regularizers

print("Attention embedded mobileNetV2:")
IMG_SHAPE=(128, 128, 3)
img_size = (128, 128) 
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,include_top=False,weights="imagenet")

base_model.trainable = True
print("Number of layers in the base model: ", len(base_model.layers))
fine_tune_at = 100

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False
    
base_out = base_model.output

print("Soft attention module started Conv2D:")
#--------------------Soft attention module-------------------------------------------------------------- 
ipts = base_out
model = BatchNormalization()(ipts)
model = Conv2D(filters = 1280, kernel_size = (1, 1), strides=(1,1), padding = 'same', activation='relu')(model)
model = BatchNormalization(-1)(model)

cbam = cbam_block(model)
base_out = tf.keras.layers.add([base_out, model, cbam])

#------------------------------------------------------------------------------------------------------------ 

x = GlobalAveragePooling2D()(base_out)

# softmax
print("Sofmax as activation in prediction:")
#predictions = Dense(len(ont_hot_labels[0]), activation='softmax', kernel_regularizer =regularizers.l2(0.01) )(x)  #l1_reg
predictions = Dense(len(train_it.class_indices), activation='softmax', kernel_regularizer =regularizers.l2(0.01) )(x)  #l1_reg

model = Model(inputs=base_model.input, outputs=predictions)

#base_learning_rate = 0.0001 #added 3/5/2022 by MS (https://blog.roboflow.com/how-to-train-mobilenetv2-on-a-custom-dataset/)
#model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adadelta(), metrics=['accuracy'])
model.compile(optimizer='adam', loss='categorical_crossentropy',  metrics = ['accuracy'])  #rmsprop

#model_checkpoint1 = ModelCheckpoint(filepath=MODEL_INIT, save_best_only=True, monitor='val_accuracy', mode='max')
model_checkpoint1 = ModelCheckpoint(filepath="./model_checkpoint1/", save_best_only=True, monitor='val_accuracy',mode='max',verbose=1)
board1 = TensorBoard(log_dir=board_name1,
                     histogram_freq=10,
                     write_graph=True,
                     write_images=True)
callback_list1 = [model_checkpoint1, board1]

print("Softmax model fitting")
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)

model.fit(train_it, steps_per_epoch=nb_train_samples / float(batch_size),
                           epochs = epochs,
                           validation_steps=nb_validation_samples / float(batch_size),
                           validation_data=validate_it,
                           #callbacks=callback_list1, verbose=2)
                           callbacks=[es], verbose=1) #Early stopping is implemented instead of callback by MS. 5/5/2022

#---------------2-nd stage---------------------------------------------
model_checkpoint2 = ModelCheckpoint(filepath="./model_checkpoint2/",  monitor='val_accuracy')
board2 = TensorBoard(log_dir=board_name2,
                     histogram_freq=0,
                     write_graph=True,
                     write_images=True)
callback_list2 = [model_checkpoint2, board2]

model.save("./newModel.h5")
model.load_weights("./newModel.h5")
for model1 in model.layers:
    model1.trainable = True

#model.compile(optimizer=optimizers.Adam(), loss =[focal_loss(gamma=2)], metrics=['accuracy']) #loss='categorical_crossentropy',
#model.compile(optimizer=optimizers.Adadelta(), loss = [focal_loss(gamma=2)], metrics=['accuracy']) #loss='categorical_crossentropy',
model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.0001), loss = [focal_loss(gamma=2)], metrics=['accuracy']) #loss='categorical_crossentropy',
model.summary()

#Reduce 20 to 20 5/5/2022 by MS
validation_steps=20

loss0,accuracy0 = model.evaluate(validate_it, steps = validation_steps)

print(loss0, accuracy0)
print("Last history:")
#es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
history=model.fit(train_it, steps_per_epoch=nb_train_samples / float(batch_size), epochs=epochs,
                    validation_data=validate_it, validation_steps=nb_validation_samples / float(batch_size),
                    #callbacks=callback_list2, verbose=2) #Early stopping is introduced to minimize loss. 5/5/2022 By MS
                    callbacks=[es], verbose=1)

print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

print("Evaluation:")
model.evaluate(validate_it)
print("Accuracy:")
get_accuracy_metrics(model, train_it, validate_it)

print("Completed")


# MobileNet-V2

# In[ ]:


#model = tf.keras.applications.MobileNetV2(include_top=False, input_shape=(SIZE_X, SIZE_Y, 3), weights='imagenet')
#model = fit_model(model, train_it, validate_it)

