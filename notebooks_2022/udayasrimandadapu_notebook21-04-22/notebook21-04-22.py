#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from numpy.random import seed
seed(1)
import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import os
from glob import glob
import seaborn as sns
from PIL import Image
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import itertools
import tensorflow
import tensorflow.keras
from tensorflow.keras.applications import ResNet152, Xception,VGG16,EfficientNetB4
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Activation,Dense, Dropout, Flatten, Conv2D, MaxPool2D,AveragePooling2D,GlobalMaxPooling2D
from tensorflow.keras import backend as K
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
np.random.seed(123)
print("done")


# In[ ]:


lesion_type_dict = {'akiec': 'Actinic keratoses',
                    'bcc': 'Basal cell carcinoma',
                    'bkl': 'Benign keratosis-like lesions ',
                    'df': 'Dermatofibroma',
                    'nv': 'Melanocytic nevi',
                    'mel': 'Melanoma',
                    'vasc': 'Vascular lesions'}

base_skin_dir = os.path.join('..','input')
#print(base_skin_dir)
#for x in glob(os.path.join(base_skin_dir,'*.jpg')):
  #imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x}

imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x for x in glob(os.path.join(base_skin_dir,'*','*','*.jpg'))}
#print(imageid_path_dict)
#print(glob(os.path.join(base_skin_dir,'*.jpg')))
skin_df = pd.read_csv('../input/skin-cancer-mnist-ham10000/HAM10000_metadata.csv')
print(skin_df)
skin1=[]
skin2=[]
for x in skin_df['image_id']:
  if x in imageid_path_dict:
    skin1.append(imageid_path_dict.get(x))
  
skin_1 = pd.DataFrame(skin1)
skin_df['path']=skin_1

#skin_df['path'] = skin_df['image_id'].map(imageid_path_dict.get)
#print(skin_df['image_id'])
#skin_df['cell_type'] = skin_df['dx'].map(lesion_type_dict.get)
for x in skin_df['dx']:
  if x in lesion_type_dict:
    skin2.append(lesion_type_dict.get(x))

skin_2 = pd.DataFrame(skin2)
skin_df['cell_type']=skin_2

skin_df['cell_type_idx'] = pd.Categorical(skin_df['cell_type']).codes
skin_df.groupby(['dx']).count()



print("*************************************************************************")
print(skin_df['path'])
print("*************************************************************************")
print(skin_df['cell_type'])
print("*************************************************************************")
print(skin_df['cell_type_idx'])


# In[ ]:


import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
ax=sns.countplot(x="dx", data=skin_df,palette = "cool")
for bar in ax.patches:
    ax.annotate(format(bar.get_height(), '.0f'),
                   (bar.get_x() + bar.get_width() / 2,
                    bar.get_height()), ha='center', va='center',
                   size=15, xytext=(0, 8),
                   textcoords='offset points')
plt.show()
ax=sns.countplot(x="sex", data=skin_df,palette = "hot")
for bar in ax.patches:
    ax.annotate(format(bar.get_height(), '.0f'),
                   (bar.get_x() + bar.get_width() / 2,
                    bar.get_height()), ha='center', va='center',
                   size=15, xytext=(0, 8),
                   textcoords='offset points')


# In[ ]:


skin_df['image'] = skin_df['path'].map(lambda x: 
                                       np.asarray(Image.open(x).resize((120,120))))
features=skin_df.drop(columns=['cell_type_idx'],axis=1)
target=skin_df['cell_type_idx']


# In[ ]:


from tensorflow.keras.utils import to_categorical
x_train_o, x_test_o, y_train_o, y_test_o = train_test_split(features, target, test_size=0.2,random_state=666)
x_train = np.asarray(x_train_o['image'].tolist())
x_test = np.asarray(x_test_o['image'].tolist())
x_train_mean = np.mean(x_train)
x_train_std = np.std(x_train)
x_test_mean = np.mean(x_test)
x_test_std = np.std(x_test)
x_train = (x_train - x_train_mean)/x_train_std
x_test = (x_test - x_test_mean)/x_test_std
y_train = to_categorical(y_train_o, num_classes = 7)
y_test = to_categorical(y_test_o, num_classes = 7)
x_train, x_validate, y_train, y_validate = train_test_split(x_train, y_train, test_size = 0.1, random_state = 999)


# In[ ]:


#Reshaping the Images into 3 channels (RGB)
x_train = x_train.reshape(x_train.shape[0], *(120, 120, 3))
x_test = x_test.reshape(x_test.shape[0], *(120, 120, 3))
x_validate = x_validate.reshape(x_validate.shape[0], *(120, 120, 3))


# In[ ]:


input_shape = (120, 120, 3)
num_classes = 7
optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=4, verbose=1, factor=0.5, min_learning_rate=0.00001)

#Data Generation
datagen = ImageDataGenerator(
        featurewise_center=False, 
        samplewise_center=False, 
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=90,
        zoom_range = 0.1, 
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        shear_range = 10) 
datagen.fit(x_train)


# In[ ]:


#Basic CNN Model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',padding = 'Same',input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu',padding = 'Same'))
model.add(AveragePooling2D(pool_size = (2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), activation='relu',padding = 'Same'))
model.add(Conv2D(64, (3, 3), activation='relu',padding = 'Same'))
model.add(AveragePooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), activation='relu',padding = 'Same'))
model.add(AveragePooling2D(pool_size = (2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), activation='relu',padding = 'Same'))
model.add(Conv2D(64, (3, 3), activation='relu',padding = 'Same'))
model.add(AveragePooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(num_classes, activation='softmax'))
model.summary()


# In[ ]:


model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
history = model.fit(datagen.flow(x_train,y_train, batch_size=64),
                              epochs = 50, validation_data = (x_validate,y_validate),
                              verbose = 1, steps_per_epoch=x_train.shape[0] // 100
                              , callbacks=[learning_rate_reduction])


# In[ ]:


loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
predictions = model.predict(x_test)
loss_v, accuracy_v = model.evaluate(x_validate, y_validate, verbose=0)
loss_t, accuracy_t = model.evaluate(x_train, y_train, verbose=0)
print("CNN Training: accuracy = %f" % (accuracy_t))
print("CNN Validation: accuracy = %f" % (accuracy_v))
print("CNN Test: accuracy = %f" % (accuracy))


# In[ ]:


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Custom Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['Test', 'Validation'])
plt.show()


# In[ ]:


predictions = np.array(list(map(lambda x: np.argmax(x), predictions)))
categories = ['Actinic keratoses',
              'Basal cell carcinoma',
              'Benign keratosis-like lesions ', 
              'Dermatofibroma', 
              'Melanocytic nevi', 
              'Melanoma',
              'Vascular lesions']

CMatrix = pd.DataFrame(confusion_matrix(y_test_o, predictions), columns=categories, index =categories)
plt.figure(figsize=(12, 6)) 
ax = sns.heatmap(CMatrix, annot = True, fmt = 'g' ,vmin = 0, vmax = 10,cmap = 'copper_r') 
ax.set_xlabel('Predicted',fontsize = 14,weight = 'bold') 
ax.set_xticklabels(ax.get_xticklabels(),rotation =90); 
ax.set_ylabel('Actual',fontsize = 14,weight = 'bold')
ax.set_title('Confusion Matrix - Test Set',fontsize = 16,weight = 'bold',pad=20);
plt.show()

