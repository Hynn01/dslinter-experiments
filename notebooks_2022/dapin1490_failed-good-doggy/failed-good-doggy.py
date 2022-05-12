#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, _ in os.walk('/kaggle/input'):
    print(dirname)

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# ## 120 Dog Breeds - classification
# ---
# \* All English explanations have been translated.   
#    
# \* 모든 영어 설명은 번역되었다.   

# ## set seed
# ---

# In[ ]:


import tensorflow as tf
import numpy as np

seed = 3
np.random.seed(seed)
tf.random.set_seed(seed)

print("seed =", seed)


# ## set variables and load data
# ---
# reference : [Load and preprocess images](https://www.tensorflow.org/tutorials/load_data/images)   
#    
# 텐서플로우 공식 문서 참고 : [이미지 로드 및 전처리하기](https://www.tensorflow.org/tutorials/load_data/images?hl=ko)   

# In[ ]:


from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras import layers

epoch = 50
batch = 5000
img_height = 32
img_width = 32

train_ds = image_dataset_from_directory("/kaggle/input/120-dog-breeds-breed-classification/Images/", validation_split=0.2, subset="training", seed=seed, image_size=(img_height, img_width), batch_size=batch)
test_ds = image_dataset_from_directory("/kaggle/input/120-dog-breeds-breed-classification/Images/", validation_split=0.2, subset="validation", seed=seed, image_size=(img_height, img_width), batch_size=batch)

resize_and_rescale = tf.keras.Sequential([
  # layers.experimental.preprocessing.Resizing(img_height, img_width),
  layers.experimental.preprocessing.Rescaling(1./255),  # layers.experimental.preprocessing.Rescaling(1./127.5, offset=-1)
  layers.experimental.preprocessing.RandomFlip("horizontal"),
  layers.experimental.preprocessing.RandomRotation(0.2),
])

normalization_layer = layers.experimental.preprocessing.Rescaling(1./255)
# normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./127.5, offset=-1)

train_augment = train_ds.map(lambda x, y: (resize_and_rescale(x, training=True), y))
test_normal = test_ds.map(lambda x, y: (normalization_layer(x), y))

# show sample
# image_batch, labels_batch = next(iter(train_augment))
# first_image = image_batch[0]
# Notice the pixels values are now in `[0,1]`.
# print(first_image)


# ## build model
# ---
# * Additional Features Used   
#     * `ModelCheckpoint` : You can save the best performing model and recall it after the learning is completed.   
#     * `EarlyStopping` : prevention of overfitting   
#    
# * Model Configuration   
#     * `dense` : basic layer   
#     * `dropout` : prevention of overfitting   
#     * `Conv2D` : Features of the image(2D) are extracted using the kernel.   
#     * `MaxPooling2D` : Reduce the size of the image(2D) with Max pooling.   
#     * `Flatten` : The image in the form of a two-dimensional array is converted into a one-dimensional array.   
# 
# ---   
#    
# * 사용한 부가기능   
#     * `ModelCheckpoint` : 가장 성능이 좋은 모델을 저장해 두고 학습이 완료된 후 다시 불러올 수 있다.   
#     * `EarlyStopping` : 과적합 방지   
#    
# * 모델 구성   
#     * `dense` : 기본적인 층   
#     * `dropout` : 과적합 방지   
#     * `Conv2D` : 커널을 이용해 이미지(2D)의 특징을 추출한다.   
#     * `MaxPooling2D` : 맥스 풀링으로 이미지(2D)의 크기를 줄인다.   
#     * `Flatten` : 2차원 배열 형태의 이미지를 1차원 배열로 바꾼다.   

# In[ ]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

MODEL_DIR = '/kaggle/working/model/'
if not os.path.exists(MODEL_DIR):
   os.mkdir(MODEL_DIR)

modelpath="/kaggle/working/model/my_best_model.hdf5"

model = Sequential()
model.add(Conv2D(128, (4, 4), input_shape=(img_height, img_width, 3), activation="relu"))
model.add(Dropout(0.5))
model.add(MaxPooling2D(pool_size=(3, 3)))

model.add(Conv2D(128, (4, 4), activation="relu", padding="same"))
model.add(Dropout(0.25))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(120, activation="softmax"))

model.summary()


# In[ ]:


from keras.callbacks import EarlyStopping, ModelCheckpoint

# additional features
checkpointer = ModelCheckpoint(filepath=modelpath, monitor='val_accuracy', verbose=1, save_best_only=True)
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10)

# model compile and fit
model.compile(loss='sparse_categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
history = model.fit(train_augment, validation_data=test_normal, epochs=epoch, batch_size=batch, verbose=0, callbacks=[checkpointer, early_stopping_callback])


# ## evaluate model
# ---

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import load_model

model = load_model('/kaggle/working/model/my_best_model.hdf5')

loss, acc = model.evaluate(test_normal)
print("acc : {:.3f}\nloss : {:.3f}\n".format(acc, loss))

# sns.set(rc={'figure.figsize':(16, 9)})
sns.set_style("ticks")

print("\n정확도 그래프")
# 테스트 셋의 정확도
y_vacc = history.history['val_accuracy']

# 학습셋의 정확도
y_acc = history.history['accuracy']

# 그래프로 표현
x_len = np.arange(len(y_acc))
plt.plot(x_len, y_acc, c="blue", label='Trainset_accuracy')
plt.plot(x_len, y_vacc, c="green", label='Testset_accuracy')

# 그래프에 그리드를 주고 레이블을 표시
plt.legend(loc='lower right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()

print("\n오차 그래프")
# 테스트 셋의 오차
y_vloss = history.history['val_loss']

# 학습셋의 오차
y_loss = history.history['loss']

# 그래프로 표현
x_len = np.arange(len(y_loss))
plt.plot(x_len, y_loss, c="blue", label='Trainset_loss')
plt.plot(x_len, y_vloss, c="green", label='Testset_loss')

# 그래프에 그리드를 주고 레이블을 표시
plt.legend(loc='upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()


# ## Record Results | 결과 기록
# ---
# ### model 1   
# ```python
# epoch = 30
# batch = 1000
# img_height = 32
# img_width = 32
# ```   
# ```python
# model = Sequential()
# model.add(Conv2D(64, (4, 4), input_shape=(img_height, img_width, 3), activation="relu"))
# model.add(Dropout(0.5))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# 
# model.add(Conv2D(128, (8, 8), activation="relu"))
# model.add(Dropout(0.5))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# 
# model.add(Flatten())
# model.add(Dense(64, activation="relu"))
# model.add(Dropout(0.5))
# model.add(Dense(120, activation="softmax"))
# 
# # additional features
# checkpointer = ModelCheckpoint(filepath=modelpath, monitor='val_accuracy', verbose=1, save_best_only=True)
# early_stopping_callback = EarlyStopping(monitor='val_loss', patience=epoch//10 + 2)
# 
# # model compile and fit
# model.compile(loss='sparse_categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
# history = model.fit(train_normal, validation_data=test_normal, epochs=epoch, batch_size=batch, verbose=0, callbacks=[checkpointer, early_stopping_callback])
# ```
# \- used : 0~1 normalized images   
#    
# **accuracy : 0.0628**   
# **loss : 4.397**   
#    
# opinion : I have many things to do   
#    
# ---   
#    
# ### model 2
# ```python
# epoch = 500
# batch = 1000
# img_height = 32
# img_width = 32
# ```   
#    
# **acc : 0.115**   
# **loss : 3.957**   
# Number of epochs performed : 422 / 500   
#    
# opinion : oh I forgot to use data augmentation!    
#     
# my plan : batch size up, EarlyStopping patience down    
#    
# ---   
#    
# ### model 3   
#    
# ```python
# batch = 5000
# patience = 10
# ```
#    
# result : I interrupted it. too slow, too low accuracy.   
# opinion : should I have to increase batch size more? or change model layer factors?   
#    
# ---   
#    
# ### model 4   
#    
# ```python
# epoch = 50
# ```   
#    
# ```python
# model = Sequential()
# model.add(Conv2D(128, (4, 4), input_shape=(img_height, img_width, 3), activation="relu"))
# model.add(Dropout(0.5))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# 
# model.add(Conv2D(256, (8, 8), activation="relu"))
# model.add(Dropout(0.5))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# 
# model.add(Flatten())
# model.add(Dense(512, activation="relu"))
# model.add(Dropout(0.5))
# model.add(Dense(120, activation="softmax"))
# ```   
#    
# \- train data augmented   
#    
# **acc : 0.090**   
# **loss : 4.293**   
#    
# opinion : more nodes -> more accuracy! but still incomplete   
# my plan : a little bit more nodes and more layers? or bigger image?   
#    
# ---   
#    
# ### model 5   
#    
# ```python
# img_height = 64
# img_width = 64
# ```   
# ```python
# model = Sequential()
# model.add(Conv2D(256, (4, 4), input_shape=(img_height, img_width, 3), activation="relu"))
# model.add(Dropout(0.5))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# 
# model.add(Conv2D(364, (8, 8), activation="relu"))
# model.add(Dropout(0.5))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# 
# model.add(Conv2D(512, (2, 2), activation="relu"))
# model.add(Dropout(0.25))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# 
# model.add(Flatten())
# model.add(Dense(768, activation="relu"))
# model.add(Dropout(0.5))
# model.add(Dense(256, activation="relu"))
# model.add(Dropout(0.5))
# model.add(Dense(120, activation="softmax"))
# ```   
#    
# `ResourceExhaustedError`   
#    
# opinion : oh..   
# my plan : less nodes   
# result : one more `ResourceExhaustedError`   
# So I decided to reduce the size of the image.   
#    
# ---   
#    
# ### model 6   
#    
# ```python
# img_height = 32
# img_width = 32
# ```   
# ```python
# model = Sequential()
# model.add(Conv2D(256, (4, 4), input_shape=(img_height, img_width, 3), activation="relu"))
# model.add(Dropout(0.5))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# 
# model.add(Conv2D(128, (2, 2), activation="relu"))
# model.add(Dropout(0.25))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# 
# model.add(Flatten())
# model.add(Dense(128, activation="relu"))
# model.add(Dropout(0.5))
# model.add(Dense(120, activation="softmax"))
# ```
#    
# `ResourceExhaustedError`   
#    
# opinion : I need to reduce the number of nodes. or image size.   
#    
# ---   
#    
# ### model 7   
#    
# ```python
# model = Sequential()
# model.add(Conv2D(256, (4, 4), input_shape=(img_height, img_width, 3), activation="relu"))
# model.add(Dropout(0.5))
# model.add(MaxPooling2D(pool_size=(3, 3)))
# 
# model.add(Conv2D(128, (4, 4), activation="relu", padding="same"))
# model.add(Dropout(0.25))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# 
# model.add(Flatten())
# model.add(Dense(128, activation="relu"))
# model.add(Dropout(0.5))
# model.add(Dense(120, activation="softmax"))
# ```
#    
# **acc : 0.052**   
# **loss : 4.462**   
#    
# opinion : I think changing the number of nodes or batch size will not solve the performance problem. I don't know how to preprocess image data yet. There's nothing more I can do.   
# 의견 : 내 생각에 노드 수나 배치 사이즈를 바꾸는 것만으로는 성능 문제가 해결되지 않을 것 같다. 나는 아직 이미지 데이터를 전처리할 줄 모른다. 내가 더 이상 할 수 있는 게 없다.   
#    

# ## memo
# ---
# \- `flow_from_directory`와 달리 `image_dataset_from_directory`는 모델을 실행하면서 하나의 `epoch`가 끝날 때마다 메모리를 비우는 것 같다. `flow_from_directory`를 쓸 때는 GPU 용량 부족 오류가 자주 났는데 이번에 데이터 로드 함수를 바꾸면서 그 오류가 거의 발생하지 않았다. 그리고 매번 `epoch`가 끝날 때마다 `Cleanup called...`라는 문구가 출력되면서 다음 에포크가 바로 시작되지 않고 잠시 지연된다. 이때 메모리를 비우는 것이라고 추측했다. 검색이라도 해서 제대로 찾아보고 싶었는데 검색어를 어떻게 써야 나올지 모르겠다. 못찾았다.   
#    
# \- 드롭아웃은 `0.5` 정도는 써줘야 티가 난다!   
#    
# \- `EarlyStopping`의 `patience` 설정을 바꿔봤다. 숫자로 직접 넣지 않고 `epoch`의 일정 비율만큼 자동으로 계산되어 들어가도록 했다. `+ 2`가 붙은 이유는 `epoch`가 적을 때 `patience`가 `0`이나 `1`이 되지 않도록 하기 위해서이다. -> 4시간 걸려서 학습시켜봤는데 일단 과적합 시작되면 기다려도 안 돌아오니 그냥 작게 잡기로 했다.   
