#!/usr/bin/env python
# coding: utf-8

# <h1 align=center><font size = 5>CNN transfer learning detect cracks in concrete</font></h1>

# The data and the estrucutre of the notebook is obtained from the IBM AI engineering course.This notebook is part of the final project of the course. 
# If you are another student of the course, please be careful to copy
# 
# <a href="https://www.coursera.org/learn/ai-deep-learning-capstone?specialization=ai-engineer">Link to the course</a>
# 
# 

# Detect concrete cracks using convolutional neural networks pre trained. 
# In this case use ResNet50 and VGG16
# 
# 0. <a href="https://keras.io/api/applications/">Keras documentation for pre-trained models</a>
# 1. <a href="https://keras.io/api/applications/vgg/">Keras documentation for VGG16</a>
# 2. <a href="https://towardsdatascience.com/step-by-step-vgg16-implementation-in-keras-for-beginners-a833c686ae6c">Example for VGG16</a>
# 3. <a href="https://keras.io/api/applications/resnet/#resnet50-function">Keras documentation for ResNet50</a>

# <h2 align=center><font size = 5>VGG16 Architecture</font></h2>
# <img align= center src = "https://www.researchgate.net/profile/Max-Ferguson/publication/322512435/figure/fig3/AS:697390994567179@1543282378794/Fig-A1-The-standard-VGG-16-network-architecture-as-proposed-in-32-Note-that-only.png" width = 400> </a>
# 

# ## 1.-Import Libraries and packages

# In[ ]:


from keras.preprocessing.image import ImageDataGenerator
import matplotlib as plt 
import numpy as np 
import tensorflow.keras.callbacks as keras_callback
import keras
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input as preprocess_input_vgg16
import tensorflow


# In[ ]:


tensorflow.__version__


# In[ ]:


keras.__version__


# ## 2.-Get the data 

# In[ ]:


get_ipython().system('wget https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0321EN/data/concrete_data_week3.zip')


# In[ ]:


get_ipython().system('unzip -qq concrete_data_week3.zip')


# ## 3.-Create the data generator 

# In[ ]:


#Data augmentation isn't necesarie for this type and quantity of images.
#train_datagen_vgg16 = ImageDataGenerator(rescale = 1./255.,
#                                   rotation_range = 40,
#                                   width_shift_range = 0.2,
#                                   height_shift_range = 0.2,
#                                   #shear_range = 0.2,
#                                   #zoom_range = 0.2,
#                                   horizontal_flip = True)
train_datagen_vgg16 = ImageDataGenerator(rescale = 1./255.,
                                   preprocessing_function = preprocess_input_vgg16)

test_datagen_vgg16 = ImageDataGenerator( rescale = 1.0/255.,
                                   preprocessing_function = preprocess_input_vgg16)


# In[ ]:


batch_size_training = 100
batch_size_validation = 100
image_resize = 224
num_classes = 2


# In[ ]:


train_generator_vgg16 = train_datagen_vgg16.flow_from_directory(
    './concrete_data_week3/train',
    target_size=(image_resize, image_resize),
    batch_size=batch_size_training,
    class_mode='categorical')


# In[ ]:


valid_generator_vgg16 = test_datagen_vgg16.flow_from_directory(
    './concrete_data_week3/valid',
    target_size=(image_resize, image_resize),
    batch_size=batch_size_training,
    class_mode='categorical')


# In[ ]:


train_datagen_resnet = ImageDataGenerator(rescale = 1./255.,
                                   preprocessing_function = preprocess_input)

test_datagen_resnet = ImageDataGenerator( rescale = 1.0/255.,
                                   preprocessing_function = preprocess_input)


# In[ ]:


train_generator_resnet = train_datagen_resnet.flow_from_directory(
    './concrete_data_week3/train',
    target_size=(image_resize, image_resize),
    batch_size=batch_size_training,
    class_mode='categorical')


# In[ ]:


valid_generator_resnet = test_datagen_resnet.flow_from_directory(
    './concrete_data_week3/valid',
    target_size=(image_resize, image_resize),
    batch_size=batch_size_training,
    class_mode='categorical')


# ## 4.-Create the neural net

# In[ ]:


model_vgg16 = Sequential([
    VGG16(include_top=False,pooling='avg',weights='imagenet'),
    Dense(num_classes, activation='softmax')])
model_vgg16.layers[0].trainable = False
model_vgg16.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model_vgg16.summary()


# In[ ]:


model_resnet = Sequential([
    ResNet50(include_top=False,pooling='avg',weights='imagenet'),
    Dense(num_classes, activation='softmax')])
model_resnet.layers[0].trainable = False
model_resnet.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model_resnet.summary()


# ## 5.-Train the model 
# 

# In[ ]:


steps_per_epoch_training = len(train_generator_vgg16) #train_generator
steps_per_epoch_validation = len(valid_generator_vgg16)
num_epochs = 5

class myCallback(keras_callback.Callback):
      def on_epoch_end(self, epoch, logs={}):
        if(logs.get('val_accuracy')>0.99):
          print("\nReached 99% accuracy in validation data so cancelling training!")
          self.model.stop_training = True
class BCP(keras_callback.Callback):  
    def __init__(self):
        self.batch_accuracy = [] # accuracy at given batch
        self.batch_loss = [] # loss at given batch 
    def on_batch_end(self, batch, logs={}):                
        self.batch_accuracy.append(logs.get('accuracy'))
        self.batch_loss.append(logs.get('loss'))


# In[ ]:


callbacks = myCallback()
BCP_vgg16 = BCP()
fit_history_vgg16 = model_vgg16.fit_generator(
    train_generator_vgg16,
    steps_per_epoch=steps_per_epoch_training,
    epochs=num_epochs,
    validation_data=valid_generator_vgg16,
    validation_steps=steps_per_epoch_validation,
    verbose=1,
    callbacks=[callbacks,BCP_vgg16])


# In[ ]:


callbacks = myCallback()
BCP_resnet = BCP()
fit_history_resnet = model_resnet.fit_generator(
    train_generator_resnet,
    steps_per_epoch=steps_per_epoch_training,
    epochs=num_epochs,
    validation_data=valid_generator_resnet,
    validation_steps=steps_per_epoch_validation,
    verbose=1,
    callbacks=[callbacks,BCP_resnet])


# ### 5.1 save the model

# In[ ]:


model_vgg16.save('model_vgg16.h5')
#model_resnet.save('model_resnet.h')

import pandas as pd

#hist_df_resnet = pd.DataFrame(fit_history_resnet.history) 
#hist_df_resnet.to_csv('history_resnet')
hist_df_vgg16 = pd.DataFrame(fit_history_vgg16.history)
hist_df_vgg16.to_csv('history_vgg16')


# In[ ]:


df_history = pd.DataFrame (BCP_vgg16.batch_accuracy, columns = ['batch accuracy'])
df_history.head()
df_history.to_csv('df_history.csv')


# In[ ]:


# Guardar configuración JSON en el disco
json_config = model_vgg16.to_json()
with open('model_vgg16_config.json', 'w') as json_file:
    json_file.write(json_config)
# Guardar pesos en el disco
model_vgg16.save_weights('model_vgg16_weights.h5')


# In[ ]:


with open('model_config.json') as json_file:
    json_config = json_file.read()
new_model = keras.models.model_from_json(json_config)
new_model.load_weights('path_to_my_weights.h5')

config = model.get_config()
weights = model.get_weights()

new_model = keras.Model.from_config(config)
new_model.set_weights(weights)


# ## 6.-Visualize the learning curves and compare models

# In[ ]:


valid_generator = test_datagen_vgg16.flow_from_directory(
    './concrete_data_week3/valid',
    target_size=(image_resize, image_resize),
    batch_size=batch_size_training,
    shuffle = False,
    class_mode='categorical')


# In[ ]:


scores_vgg16 = model_vgg16.evaluate_generator(valid_generator_vgg16)
print("%s%s: %.2f%%" % ("VGG16 evaluate_generator ",model_vgg16.metrics_names[1], scores_vgg16[1]*100))
    
scores_resnet = model_resnet.evaluate_generator(valid_generator_vgg16)    
print("%s%s: %.2f%%" % ("ResNet50 evaluate_generator ",model_resnet.metrics_names[1], scores_resnet[1]*100))


# In[ ]:


filenames = valid_generator_vgg16.filenames
nb_samples = len(filenames)
predict_vgg16 = model_vgg16.predict_generator(valid_generator_vgg16,steps = nb_samples)
predict_vgg16[0:6]


# In[ ]:


for i in predict_vgg16[0:6,0]:
    if i >0.5:
        print("Positive")
    else: 
        print("Negative")


# In[ ]:





# In[ ]:


import matplotlib.pyplot as plt
acc_v = BCP_vgg16.batch_accuracy
acc_res =  BCP_resnet.batch_accuracy

epochs = range(len(acc_v))
plt.plot(epochs, acc_v, 'r', label='Training accuracy VGG16')
plt.plot(range(len(acc_res)), acc_res, 'b', label='Training accuracy ResNet50')
plt.title('Training accuracy VGG16 and ResNet50')
plt.legend(loc=0)
plt.figure()
plt.show()


# In[ ]:


acc_v = fit_history_vgg16.history['accuracy']
val_acc_v = fit_history_vgg16.history['val_accuracy']
acc_res =  fit_history_resnet.history['accuracy']
val_acc_res =  fit_history_resnet.history['val_accuracy']
epochs = range(len(acc_v))
plt.plot(epochs, acc_v, 'r', label='Training accuracy VGG16')
plt.plot(epochs, val_acc_v, 'b', label='Validation accuracy VGG16')
plt.plot(range(len(acc_res)), acc_res, 'g', marker='o', label='Trainign accuracy ResNet50')
plt.plot(range(len(acc_res)), val_acc_res, 'y', marker='o', label='Validation accuracy ResNet50')
plt.title('Training and validation accuracy ')
plt.legend(loc=0)
plt.figure()
plt.show()


# ## 7.-Make prediction and show mismatching

# In[ ]:


import os
from keras.preprocessing import image


# In[ ]:


images_neg = os.listdir('./concrete_data_week3/valid/negative/')
images_pos = os.listdir('./concrete_data_week3/valid/positive/')


# In[ ]:


prediction_neg = []
missclassified_neg = []
contador = 0
n_images = 1000
for i in images_neg:
    path = './concrete_data_week3/valid/negative/'+i
    img = image.load_img(path, target_size=(224, 224))
    x = image.img_to_array(img)/255.
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    classes = model_vgg16.predict(images, batch_size=10)
    prediction_neg.append(classes)
    contador = contador +1
    if classes[0][0] < 0.5:
        missclassified_neg.append(path)
        print("mistake  in classification")
        plt.imshow(image.img_to_array(img).astype(np.uint8))
        plt.title('negative probability:'+str(classes[0][0])+str(path))
        plt.show()
    if contador == n_images:
        break


# In[ ]:


prediction_pos = []
missclassified_pos = []
contador = 0
n_images = 1000
for i in images_pos:
    path = './concrete_data_week3/valid/positive/'+i
    img = image.load_img(path, target_size=(224, 224))
    x = image.img_to_array(img)/255.
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    classes = model_vgg16.predict(images, batch_size=10)
    prediction_pos.append(classes)
    contador = contador +1
    if classes[0][1] < 0.5:
        missclassified_pos.append(path)
        print("mistake  in classification")
        plt.imshow(image.img_to_array(img).astype(np.uint8))
        plt.title('positive probability:'+str(classes[0][1])+str(path))
        plt.show()
    if contador == n_images:
        break


# ## 9.-Retrain the model

# In[ ]:


steps_per_epoch_training = len(train_generator_vgg16) #train_generator
steps_per_epoch_validation = len(valid_generator_vgg16)
num_epochs = 10

BCP_vgg16 = BCP()
fit_history_vgg16 = model_vgg16.fit_generator(
    train_generator_vgg16,
    steps_per_epoch=steps_per_epoch_training,
    epochs=num_epochs,
    validation_data=valid_generator_vgg16,
    validation_steps=steps_per_epoch_validation,
    verbose=1,
    callbacks=[BCP_vgg16])


# In this case may be acceptable misclassified a negative case as positive. Because we need detect all cracks. 
# 
# We can modify the probability in predictions for classified as positive the case with a probability major to 0.4

# In[ ]:


for path in missclassified_neg:
    img = image.load_img(path, target_size=(224, 224))
    x = image.img_to_array(img)/255.
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    classes = model_vgg16.predict(images, batch_size=10)
    if classes[0][0] < 0.4:
        print("mistake  in classification")
        plt.imshow(image.img_to_array(img).astype(np.uint8))
        plt.title('negative probability:'+str(classes[0][0])+str(path))
        plt.show()


# In[ ]:


for path in missclassified_pos:
    img = image.load_img(path, target_size=(224, 224))
    x = image.img_to_array(img)/255.
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    classes = model_vgg16.predict(images, batch_size=10)
    if classes[0][1] < 0.4:
        print("mistake  in classification")
        plt.imshow(image.img_to_array(img).astype(np.uint8))
        plt.title('negative probability:'+str(classes[0][1])+str(path))
        plt.show()


# In[ ]:




