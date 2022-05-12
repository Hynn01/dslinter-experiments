#!/usr/bin/env python
# coding: utf-8

# # Sea Turtle Face Detection
# 
# This notebook shows how to do simple face detection using the Kaggle Sea Turtle dataset.  I shold be run connected to the Sea Turtle dataset--check the Data->Input tab on the right side to make sure you are connected to sea-turtle-face-detection
# 
# ## Face Detection vs Recognition
# 
# * Face detections is the process of finding faces in an image and drawing bounding boxes around them. 
#   * This requires a dataset with bounding boxes marked on each image like the sea turtle dataset 
# * Face recognition is the process of identifing the person represented by a face
#   * This requires a dataset with bounding boxes and IDs for each face. We will not be doing this with the dataset since we don't have unique ids for each turtle.
# 
# ## Running with a GPU
# 
# This notebooks runs much faster with the GPU enabled by clicking on the ... menu on the right and selecting Accelerator

# In[ ]:


# This simply makes sure Kaggle session is started and that we can run code
print ("Getting started")


# In[ ]:


# First, we'll import pandas and numpy, two data processing libraries
import pandas as pd
import numpy as np

# We'll also import seaborn and matplot, twp Python graphing libraries
import seaborn as sns
import matplotlib.pyplot as plt
# Import the needed sklearn libraries
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

# The Keras library provides support for neural networks and deep learning
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, SpatialDropout2D, Activation, Lambda, Flatten, LSTM
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam, RMSprop
#from tensorflow.keras.utils import np_utils
from tensorflow.keras import utils

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

print ("Libraries Imported")


# 

# In[ ]:


# define path to the read-only input folder
dataPath = "/kaggle/input/sea-turtle-face-detection/data/"
# define path to the writable working folder
working_dir = "/kaggle/working/"

# read text file into pandas DataFrame
df_labels = pd.read_csv("/kaggle/input/sea-turtle-face-detection/data/labels.csv", header=None)
df_labels.columns = ['species', 'upper_left_x', 'upper_left_y', 'bbwidth', 'bblength', 'filename', 'image_width', 'image_length' ]
# display DataFrame
df_labels.head


# # Notes on file names and bounding boxes
# 
# ## file names
# 
# Most of the files follow the format Image_<num>.jpg but some files have extendsion of ".jpeg" or ".png" instead of ".jpg"
# 
# ## Multiple turtle images
# 
# There are about 100 images with 2+ turtle faces in them. There are 2000 images and 2103 bounding boxes. We will remove the images with multiple turtles and end up with 1934 images
#     
# * The labels.csv file only has multiple rows for multi-turtle images
# * The image.txt file has all the bounding boxes for the multi-turtles listed in the text file
# 

# ## Remove images with multiple faces

# In[ ]:


print("Shape before dropping duplicates " + str(df_labels.shape))
df_labels = df_labels.drop_duplicates(subset='filename', keep=False)
print("Shape after dropping duplicates " + str(df_labels.shape))


# # Set number of images
# 
# There are 2,000 images. We elinimate 66 of them due to mulitple turtle faces
# 
# Of the remaining 1934 images, the following code selects how many to use for training...

# In[ ]:


# This code uses all the images available
df_labels = df_labels[0:]
# This code limits the images to the first 500 and will run faster for training if you need to
#df_labels = df_labels[0:500]

df_labels


# In[ ]:


# Set the image size used by the network input later
IMAGE_SIZE = (224, 224)


# # Bounding Box Formats and Units
# 
# # Formats
# 
# There are two basic formats for bounding boxes:
# * Corners (x1, y1, x2, y2): specifies the upper left corner (x1, y1) and the lower right corner (x2, y2) of the bounding box
# * Center-Size (x, y, width, lenght): this specifies the center of the bounding box and the width and lenght of the box
# 
# # Units
# 
# The units can be specified in absolute pixes or as a percent of the entire image size.  YOLO uses a popular format that specifies the Center-Size of the bounding box is percent of the image sizes.  
# 
# So, an 200 x 300 pixel image with a bounding box from (150, 150) to (200,200) would have the following values:
# *  Corners in pixels: (x1, y1, x2, y2) = (150, 150, 200, 200)
# *  YOLO Center-Size as percent: (x, y, width, lenght) = (0.75, 0.5, 1.0, 0.67)
# 
# The label.txt files contain YOLO format bounding boxes.  We will read these in below.

# ## Append yolo bounding boxes and create a new dataframe named df_all

# In[ ]:


# Get YOLO bounding boxes from text files
#df_yolo = pd.DataFrame.empty
labelName = dataPath + "labels/" + df_labels["filename"][0][:-3] + 'txt'
df_yolo = pd.read_csv(labelName, delim_whitespace=True, header=None)
df_yolo.columns = ['idd', 'yolo_x', 'yolo_y','yolo_width', 'yolo_length'] 

# Loop over all the file names, skipping the first one that is done above
for imageName in df_labels["filename"][1:]:
    if (imageName[-4] == '.'):
        # some images have ".jpeg" instead of ".jpg"
        labelName = dataPath + "labels/" + imageName[:-3] + 'txt'
    else:
        labelName = dataPath + "labels/" + imageName[:-4] + 'txt'
    #print (labelName)
    bb_txt = pd.read_csv(labelName, delim_whitespace=True, header=None)
    bb_txt.columns = ['idd', 'yolo_x', 'yolo_y','yolo_width', 'yolo_length'] 
    df_yolo = pd.concat([df_yolo, bb_txt.iloc[[0]] ], ignore_index=True)

    #print (df_yolo)
    
df_yolo.head(10)


# In[ ]:


df_labels.reset_index(drop=True, inplace=True)
df_yolo.reset_index(drop=True, inplace=True)

df_all = pd.concat([df_labels, df_yolo], axis=1)
print ("df_labels shape = "+ str(df_labels.shape))
print ("df_yolo shape = "+ str(df_yolo.shape))
print ("df_all shape = "+ str(df_all.shape))


# # Resize all the images
# 
# Most models will work with images of size 224x224.  It is faster to resize the images once, rather than each time they are used for training.  The code below resizes all the images and stores them in folder working/images224.
# 
# We also need to recalculate bounding boxes that use the pixel format for images resized to 224x224

# In[ ]:


# Resizing the images into a new folder images224
import os
import skimage.io
import skimage
import cv2
from tqdm import tqdm    # needed for progress bar

load_dir = "/kaggle/input/sea-turtle-face-detection/data/images/"
save_dir = "/kaggle/working/images224/"
print ('Creating a new folder at '+save_dir)
os.makedirs(save_dir, exist_ok=True)

print ('Looping through all the images in the data frame')
#for imageName in df_labels["filename"][1:]:
for imageName in tqdm(df_labels["filename"]):

    load_path = load_dir + imageName
    save_path = save_dir + imageName
    img = mpimg.imread(load_path)
    img_resize = cv2.resize(img, (224, 224))
    mpimg.imsave(save_path, img_resize)

print ("All images resized and saved to /kaggle/working/images224/ folder")


# # add bounding box fields for the resized images assuming size is 224x224

# In[ ]:


width = 224
length = 224
df_all['resize_upper_x'] = df_all.apply(lambda row: round(row['yolo_x'] * width - 0.5 * row['yolo_width'] * width), axis = 1)
df_all['resize_left_y'] = df_all.apply(lambda row:  round(row['yolo_y'] * length - 0.5 * row['yolo_length'] * length), axis = 1)
df_all['resize_lower_x'] = df_all.apply(lambda row: round(row['yolo_x'] * width + 0.5 * row['yolo_width'] * width), axis = 1)
df_all['resize_right_y'] = df_all.apply(lambda row: round(row['yolo_y'] * length + 0.5 * row['yolo_length'] * length), axis = 1)


# In[ ]:


df_all


# In[ ]:


# https://colab.research.google.com/github/d2l-ai/d2l-en-colab/blob/master/chapter_computer-vision/bounding-box.ipynb#scrollTo=wASmLzN_HAlN

def box_corner_to_center_arr(boxes):
    """Convert from (upper-left, lower-right) to (center, width, height)."""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    boxes = np.stack((cx, cy, w, h), axis=-1)
    return boxes

def box_center_to_corner_arr(boxes):
    """Convert from (center, width, height) to (upper-left, lower-right)."""
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    boxes = np.stack((x1, y1, x2, y2), axis=-1)
    return boxes

def bbox_to_rect(bbox, color):
    """Convert bounding box to matplotlib format."""
    # Convert the bounding box (upper-left x, upper-left y, lower-right x,
    # lower-right y) format to the matplotlib format: ((upper-left x,
    # upper-left y), width, height)
    return plt.Rectangle(
        xy=(bbox[0], bbox[1]), width=bbox[2]-bbox[0], height=bbox[3]-bbox[1],
        fill=False, edgecolor=color, linewidth=2)


def box_center_to_corner(box, width, length):
    """Convert from (center, width, height) to (upper-left, lower-right)."""
    cx, cy, w, h = box[0], box[1], box[2], box[3]
    x1 = round(cx * width - 0.5 * w * width)
    y1 = round(cy * length - 0.5 * h * length)
    x2 = round(cx * width + 0.5 * w * width)
    y2 = round(cy * length + 0.5 * h * length)
    bbox = [x1, y1, x2, y2]
    return bbox


# In[ ]:


# Extract bounding boxes from the labels.
#trainBboxes = df_labels.iloc[:,1:6]
#trainBboxes.columns = ['upper_left_x', 'upper_left_y', 'width', 'length', 'filename' ]
#trainBboxes.columns = ['yolo_x', 'yolo_y', 'yolo_width', 'yolo_length', 'filename' ]
#trainBboxes.columns = ['resize_upper_x', 'resize_left_y', 'resize_lower_x', 'resize_right_y', 'filename' ]


#print (trainBboxes)


# # Display some sample image and check bounding boxes

# In[ ]:


from glob import glob
from PIL import Image

def plot_images(imgs, rows=5):
    # Set figure to 15 inches x 8 inches
    figure = plt.figure(figsize=(15, 8))
    cols = len(imgs) // rows + 1
    for i in range(len(imgs)):
        subplot = figure.add_subplot(rows, cols, i + 1)
        subplot.axis('Off')
        plt.imshow(imgs[i])

def plot_images_for_filenames(filenames, rows=5):
    imgs = [plt.imread(f'{filename}') for filename in filenames]
    return plot_images(imgs, rows)

def displayTurtle(turtleNum):
    dataPath = "/kaggle/input/sea-turtle-face-detection/data/"
    image_path = dataPath + "images/Image_" + str(turtleNum) + ".jpg"
    lable_path = dataPath + "labels/Image_" + str(turtleNum) + ".txt"
    # display the image
    img = mpimg.imread(image_path)
    height, width, depth = img.shape
    plt.figure(figsize = (15,15))
    fig = plt.imshow(img)
    # read the yolo cordinates.  read_csv stores the numbers in the column names
    label = pd.read_csv(lable_path, sep=" ")
    yolobbox = list(label.columns[1:5])
    yolobbox = [float(i) for i in yolobbox]    # convert numbers from strings to floats
    print (yolobbox)
    # convert to corners
    bbox = box_center_to_corner(yolobbox, width, height)
    print (bbox)
    fig.axes.add_patch(bbox_to_rect(bbox, 'blue'))

def displayTurtle_resized(turtleNum):
    dataPath = "/kaggle/input/sea-turtle-face-detection/data/"
    image_path = '/kaggle/working/images224/' + "Image_" + str(turtleNum) + ".jpg"
    lable_path = dataPath + "labels/Image_" + str(turtleNum) + ".txt"
    # display the image
    img = mpimg.imread(image_path)
    height, width, depth = img.shape
    plt.figure(figsize = (5,5))
    fig = plt.imshow(img)
    # read the yolo cordinates.  read_csv stores the numbers in the column names
    label = pd.read_csv(lable_path, sep=" ")
    yolobbox = list(label.columns[1:5])
    yolobbox = [float(i) for i in yolobbox]    # convert numbers from strings to floats
    print (yolobbox)
    # convert to corners
    bbox = box_center_to_corner(yolobbox, width, height)
    print (bbox)
    fig.axes.add_patch(bbox_to_rect(bbox, 'blue'))


# In[ ]:


sample_images = glob("../input/sea-turtle-face-detection/data/images/Image_101*.jpg")

print ("Original Sized Images")
print (sample_images)
plot_images_for_filenames(sample_images)  


# In[ ]:


sample_images = glob("/kaggle/working/images224/Image_101*.jpg")

print ("Resized Images")
print (sample_images)
plot_images_for_filenames(sample_images)  


# In[ ]:


#This is an example of a image with two turtle faces
# from https://www.kaggle.com/code/foucardm/draw-bounding-boxes-for-object-detection
one_image_path = "/kaggle/input/sea-turtle-face-detection/data/images/Image_18.png"
image = mpimg.imread(one_image_path)
plt.figure(figsize = (10,10))
imgplot = plt.imshow(image)
plt.show()


# In[ ]:


displayTurtle(34)


# In[ ]:


displayTurtle_resized(34)


# In[ ]:


displayTurtle(1015)


# In[ ]:


displayTurtle_resized(1015)


# # Set up CNN 

# In[ ]:





# In[ ]:


# from https://stackoverflow.com/questions/41749398/using-keras-imagedatagenerator-in-a-regression-model

IMAGE_SIZE = (224, 224)

train_datagen = ImageDataGenerator(
    rescale = 1./255, 
    horizontal_flip = True
) 

# This version uses the resized images
train_generator = train_datagen.flow_from_dataframe(
    dataframe=df_all, 
    directory=save_dir,                                               
    x_col="filename", 
    #y_col=['resize_upper_x', 'resize_left_y', 'resize_lower_x', 'resize_right_y'], 
    y_col=['yolo_x', 'yolo_y', 'yolo_width', 'yolo_length'], 
    #has_ext=True, 
    class_mode="raw", 
    target_size=IMAGE_SIZE,
    batch_size=32
)


# In[ ]:


from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

learning_rate_reduction = ReduceLROnPlateau(monitor='loss', 
                                            patience=5, 
                                            verbose=2, 
                                            factor=0.5,                                            
                                            min_lr=0.000001)

early_stops = EarlyStopping(monitor='loss', 
                            min_delta=0, 
                            patience=10, 
                            verbose=2, 
                            mode='auto')

checkpointer = ModelCheckpoint(filepath = 'cis3115.{epoch:02d}-{accuracy:.6f}.hdf5',
                               verbose=2,
                               save_best_only=True, 
                               save_weights_only = True)


# In[ ]:


# Set up the Neural Network
IMAGE_SIZE = (224, 224)
# ==== Select one of the pre-trained models from Keras.  Samples are shown below
#pretrained_model = tf.keras.applications.DenseNet201(weights='imagenet', include_top=False ,input_shape=[*IMAGE_SIZE, 3])
#pretrained_model = tf.keras.applications.EfficientNetB4(weights='imagenet', include_top=False ,input_shape=[*IMAGE_SIZE, 3])
#pretrained_model = tf.keras.applications.InceptionV3(weights='imagenet', include_top=False ,input_shape=[*IMAGE_SIZE, 3])
#pretrained_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False ,input_shape=[*IMAGE_SIZE, 3])
#pretrained_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False ,input_shape=[*IMAGE_SIZE, 3])
#pretrained_model = tf.keras.applications.ResNet152V2(weights='imagenet', include_top=False ,input_shape=[*IMAGE_SIZE, 3])
pretrained_model = tf.keras.applications.VGG19(weights='imagenet', include_top=False ,input_shape=[*IMAGE_SIZE, 3])
#pretrained_model = tf.keras.applications.Xception(weights='imagenet', include_top=False ,input_shape=[*IMAGE_SIZE, 3])

# Set the following to False so that the pre-trained weights are not changed 
pretrained_model.trainable = False 

model = Sequential()
#  Start with the pretrained model defined above
model.add(pretrained_model)

# Flatten 2D images into 1D data for final layers like traditional neural network
model.add(Flatten())
# GlobalAveragePooling2D is an alternative to Flatter and reduces the size of the layer while flattening
#model.add(GlobalAveragePooling2D())
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))

# The final output layer
# Use Sigmoid when predicting YOLO bounding box since that output is between 0 and 1
#model.add(Dense(4, activation='sigmoid'))
# Use relu when predicting corner pixels since the outputs are intergers larger than 1
#model.add(Dense(4, activation='relu'))
model.add(Dense(4, activation='linear'))


print ("Pretrained model used:")
pretrained_model.summary()

print ("Final model created:")
model.summary()

# Compile neural network model
#model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
model.compile(optimizer='adam', loss='mean_absolute_error', metrics=['accuracy'])


# In[ ]:


# Train the model with the images in the folders
history = model.fit(
        train_generator,
        # validation_data=(testImages, testTargets),
        batch_size=16,                  # Number of image batches to process per epoch 
        epochs=100,                      # Number of epochs
        callbacks=[learning_rate_reduction, early_stops],
        )


# In[ ]:


# We will display the loss and the accuracy of the model for each epoch
# NOTE: this is a little fancier display than is shown in the textbook
def display_training_curves(training, validation, title, subplot):
    if subplot%10==1: # set up the subplots on the first call
        plt.subplots(figsize=(10,10), facecolor='#F0F0F0')
        plt.tight_layout()
    ax = plt.subplot(subplot)
    ax.set_facecolor('#F8F8F8')
    ax.plot(training)
    ax.plot(validation)
    ax.set_title('model '+ title)
    ax.set_ylabel(title)
    #ax.set_ylim(0.28,1.05)
    ax.set_xlabel('epoch')
    ax.legend(['train', 'valid.'])


# In[ ]:


#display_training_curves(history.history['loss'], history.history['val_loss'], 'loss', 211)
display_training_curves(history.history['loss'], history.history['loss'], 'loss', 211)
display_training_curves(history.history['accuracy'], history.history['accuracy'], 'accuracy', 212)


# In[ ]:





# In[ ]:


# Predict the bounding box
# Original box is in Blue
# Predicted box is in Yellow

def predict_resized_pixels(rowNum):
    row = df_all.iloc[[rowNum]]
    filename = row['filename']
    filename = filename.iloc[0]
    image_path = '/kaggle/working/images224/' + str(filename)
    # display the image
    img = mpimg.imread(image_path)
    img = img/255     # rescale the image values
    #height, width, depth = img.shape
    plt.figure(figsize = (5,5))
    fig = plt.imshow(img)
    
    # Display to original bounding box
    orig_bbox = row[['resize_upper_x', 'resize_left_y', 'resize_lower_x', 'resize_right_y']]
    orig_bbox = orig_bbox.values.tolist()
    orig_bbox = orig_bbox[0]
    print ('Actual Bounding Box')
    print (orig_bbox)
    fig.axes.add_patch(bbox_to_rect(orig_bbox, 'blue'))
    
    # Predict and display the predicted bounding box
    img2 = cv2.imread(image_path)
    #img = cv2.resize(img,(320,240))
    img = np.reshape(img,[1,224,224,3])

    pred_bbox = model.predict(img)
    pred_bbox = pred_bbox[0]
    print ('Predicted Bounding Box')
    print (pred_bbox)
    fig.axes.add_patch(bbox_to_rect(pred_bbox, 'yellow'))


# In[ ]:


# Predict the bounding box
# Original box is in Blue
# Predicted box is in Yellow

def predict_yolo(rowNum):
    row = df_all.iloc[[rowNum]]
    filename = row['filename']
    filename = filename.iloc[0]
    image_path = '/kaggle/working/images224/' + str(filename)
    # display the image
    img = mpimg.imread(image_path)
    height, width, depth = img.shape
    print ('Read in image of size '+str(height) + ' by ' + str(width) + ' by '+ str(depth))
    img = img/255     # rescale the image values
    plt.figure(figsize = (5,5))
    fig = plt.imshow(img)
    
    # Display to original bounding box
    yolo_orig_bbox = row[['yolo_x', 'yolo_y', 'yolo_width', 'yolo_length']]
    yolo_orig_bbox = yolo_orig_bbox.values.tolist()
    yolo_orig_bbox = yolo_orig_bbox[0]
    print ('Actual YOLO Bounding Box')
    print (yolo_orig_bbox)
    orig_bbox = box_center_to_corner(yolo_orig_bbox, 224, 224)
    print ('Actual Pixel Corners Bounding Box')
    print (orig_bbox)
    fig.axes.add_patch(bbox_to_rect(orig_bbox, 'blue'))
    
    # Predict and display the predicted bounding box
    img = np.reshape(img,[1,224,224,3])
    yolo_pred_bbox = model.predict(img)
    yolo_pred_bbox = yolo_pred_bbox[0]
    print ('Predicted YOLO Bounding Box')
    print (yolo_pred_bbox)
    pred_bbox = box_center_to_corner(yolo_pred_bbox, 224, 224)
    print ('Predicted Pixel Corners Bounding Box')
    print (pred_bbox)
    fig.axes.add_patch(bbox_to_rect(pred_bbox, 'yellow'))


# In[ ]:


#predict_resized(5)
for i in range(9):
    predict_yolo(i)


# In[ ]:


#predict_resized_pixels(10)
predict_yolo(10)


# In[ ]:




