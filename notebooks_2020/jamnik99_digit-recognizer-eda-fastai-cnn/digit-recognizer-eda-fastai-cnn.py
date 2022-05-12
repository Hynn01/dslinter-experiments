#!/usr/bin/env python
# coding: utf-8

# # Introduction

# This notebook has 2 major goals:
# 1. Make viewer more familiar with MNIST dataset by Exploratory Data Analysis and dimensionality reduction with visualization
# 2. Make intro to Convolutional Neural Networks in Fastai library

# <font size = 3, font color = 'red'>Voting is appreciated.</font>

# ### MNIST Introduction

# MNIST ("Modified National Institute of Standards and Technology") is the de facto “hello world” dataset of computer vision. Since its release in 1999, this classic dataset of handwritten images has served as the basis for benchmarking classification algorithms. As new machine learning techniques emerge, MNIST remains a reliable resource for researchers and learners alike.

# In[ ]:


from IPython.display import YouTubeVideo
from IPython import display


# In[ ]:


video_id = 'oKzNUGz21JM'
YouTubeVideo(video_id, width = 800, height = 500)


# # Acknowledgement

# * [Digit Recognizer Competition](https://www.kaggle.com/c/digit-recognizer)
# * [Video about Multi-dimensional Scaling (MDS)](https://youtu.be/GEn-_dAyYME)
# * [Video about t-distributed Stochastic Neighbor Embedding (t-SNE)](https://youtu.be/NEaUSP4YerM)
# * [Fastai documentation](https://docs.fast.ai/)
# * [Article about residual neural nets](https://arxiv.org/abs/1512.03385)

# # Content

# * [Imports](#Imports)
# * [Data Structure](#Data-Structure)
# * [Exploratory Data Analysis](#Exploratory-Data-Analysis)
#     * [Train](#Train)
#     * [Test](#Test)
#     * [Visualization](#Visualization)
#         * [Multi-dimensional Scaling](#Multi-dimensional-Scaling)
#         * [t-distributed Stochastic Neighbor Embedding](#t-distributed-Stochastic-Neighbor-Embedding)
# * [Fastai Convolutional Neural Network](#Fastai-Convolutional-Neural-Network)
#     * [Create Folder](#Create-Folder)
#     * [Create ImageDataBunch](#Create-ImageDataBunch)
#     * [Create Model](#Create-Model)
#     * [Train Model](#Train-Model)
#     * [Parameters Tuning](#Parameters-Tuning)
# * [Conclusion](#Conclusion)

# # Imports

# In[ ]:


# Imports for data loading and array math
import numpy as np
import pandas as pd

# Imports for
from ipywidgets import interact

# Imports for visualization
import matplotlib.pyplot as plt
import matplotlib as mpl
get_ipython().run_line_magic('matplotlib', 'inline')

# Imports for dimensionality reduction
from sklearn.manifold import MDS, TSNE

# Import for working with directories and files
import os

# Import for working with vision applications of fastai
from fastai.vision import *

# Import for data split
from sklearn.model_selection import train_test_split


# # Data Structure

# There are 3 files in the 'digit-recognizer' directory:
# * sample_submission.csv - contains sample submission for the competition 
# * train.csv - contains train data
# * test.csv - contains test data
# 
# Train data table consist of 785 columns: 784 columns for each pixel of the image and 1 column for the target. Test data table has 784 columns and don't have target. Each pixel is the number in range [0; 255] inlcusive. 

# # Exploratory Data Analysis 

# In[ ]:


sample_submission_path = '../input/digit-recognizer/sample_submission.csv'
train_path = '../input/digit-recognizer/train.csv'
test_path = '../input/digit-recognizer/test.csv'


# Let's load train and test data, reshape images to shape (28, 28) and normalize them by dividing data by maximum, which equals to 255. Then we will look at some samples of the train and test data.

# In[ ]:


train = pd.read_csv(train_path)

X = train.iloc[:, 1:].values
X = X.reshape((X.shape[0], 28, 28))/255.

Y = train.iloc[:, 0].values

X_test = pd.read_csv(test_path)
X_test = X_test.values.reshape((X_test.shape[0], 28, 28))/255.


# ## Train

# In[ ]:


@interact
def plot_train_set(target = range(10), batch = (0, 263, 1)):
    mpl.rcParams['figure.figsize'] = 6, 6
    class_pictures = X[Y == target]
    side = 3
    for i in range(side):
        for j in range(1, side + 1):
            plt.subplot(side, side, i * side + j)
            temp_index = batch * (side ** 2) + i * side + j - 1
            if temp_index < class_pictures.shape[0]:
                plt.imshow(class_pictures[temp_index], cmap = 'gray')


# In[ ]:


plot_train_set(target = 4, batch = 13)


# In[ ]:


plot_train_set(target = 5, batch = 15)


# In[ ]:


plot_train_set(target = 6, batch = 13)


# In[ ]:


plot_train_set(target = 9, batch = 391)


# As it can be seen from the plots above digits are distinguashable. However there can be problems in some classes:
# * 4: It can be connected at it's upper part
# * 5: It mat be similar to 6, 0 or even 1 with angle
# * 6: it can be written without a space in center or even be like a comma
# * 9: If it doesn't have circle it can be like 1 and without connection at the upper part can be similar to 4

# ## Test

# In[ ]:


@interact
def plot_test_set(batch = (0, 3100, 1)):
    mpl.rcParams['figure.figsize'] = 6, 6
    side = 3
    for i in range(side):
        for j in range(1, side + 1):
            plt.subplot(side, side, i * side + j)
            temp_index = batch * (side ** 2) + i * side + j - 1
            if temp_index < X_test.shape[0]:
                plt.imshow(X_test[temp_index], cmap = 'gray')


# In[ ]:


plot_test_set(batch = 13)


# At the first glance test data seems to be like train and is easily distinguishable. 

# For now we can say that our images do not realy need all 784 pixels to categorize them. Many pixels near the borders of the images are almost always black. That's why now we will try to reduce dimensionality of our image dataset from 784 to 2. After it we can plot them on the graph.

# ## Visualization

# In this section we will apply 2 methods of dimensionality reduction. They are called Multi-dimensional scaling (MDS) and t-distributed Stochastic Neighbor Embedding (t-SNE).

# ### Multi-dimensional Scaling

# In MDS images which were far from each other in the 784-d space will be far away from each other in the 2-d space. And images which were close to each other in the first space will be close to each other in the second space. 

# In[ ]:


count = 1000
embedding = MDS(n_components = 2, metric = True, random_state = 42)
X_train_transformed = embedding.fit_transform(X[:count].reshape((count, 28 * 28)))


# In[ ]:


plt.style.use('seaborn-darkgrid')
plt.rcParams['figure.figsize'] = 12, 10


# In[ ]:


plt.scatter(X_train_transformed[:, 0], X_train_transformed[:, 1], c = Y[:count], cmap = plt.cm.get_cmap('tab10', 10))
plt.title('linear MDS transformed')
plt.xlabel('Axis 1')
plt.ylabel('Axis 2')
plt.colorbar(ticks = range(10));


# It can be seen from the graph above that 0s, 1s, 2s, 3s, 6s and 9s do not overlap. However, other digits seem to be not so well separated. For example, 7s overlap with 9s and 4s overlap with 9s. It's not so bad for linear MDS. Even by applying it we can have some results. But now let's try non-linear MDS. 

# In[ ]:


non_linear_embedding = MDS(n_components = 2, metric = False, random_state = 42)
X_train_non_linear = embedding.fit_transform(X[:count].reshape((count, 28 * 28)))


# In[ ]:


plt.scatter(X_train_non_linear[:, 0], X_train_non_linear[:, 1], c = Y[:count], cmap = plt.cm.get_cmap('tab10', 10))
plt.title('non-linear MDS transformed')
plt.xlabel('Axis 1')
plt.ylabel('Axis 2')
plt.colorbar(ticks = range(10));


# From the graph below it can be seen that results don't become better than in the previous case. Now let's apply t-SNE algorithm.

# ### t-distributed Stochastic Neighbor Embedding

# t-SNE method projects all samples from the first space to the second randomly. Then it starts to move projected examples accordingly to the similarity of the examples in the first space. Similarity is measured by t-distribution. 

# In[ ]:


tsne_embedding = TSNE(n_components = 2, random_state = 42)
X_train_tsne_transformed = tsne_embedding.fit_transform(X[:count].reshape((count, 28 * 28)))


# In[ ]:


plt.scatter(X_train_tsne_transformed[:, 0], X_train_tsne_transformed[:, 1], c = Y[:count], cmap = plt.cm.get_cmap('tab10', 10), s = 50)
plt.title('t-SNE transformed')
plt.xlabel('Axis 1')
plt.ylabel('Axis 2')
plt.colorbar(ticks = range(10));


# t-SNE has much better results. It can be seen that some classes have totally separated from others and the area of overlap has decreased dramatically.

# To summarize, with dimensionality reduction techniques we can transform our images with 784 pixels to 2-d space and still categorize them pretty well. Now let's try more sophisticated models called CNN. They will pay attention to the structure of the image.

# # Fastai Convolutional Neural Network

# Fastai library make applying deep learning easy. It has different modules. And one of them (vision module) we will be using in this notebook.

# Before creating deep neural nets we must create data for the model. Fastai has many opportunities of creating data. We will be using method from_folder of class ImageDataBunch. At the beginning we must create folder, which will the structure below:
# * working/
#     * train/
#         * 0/
#         * 1/
#         * 2/
#         * 3/
#         * 4/
#         * 5/
#         * 6/
#         * 7/
#         * 8/
#         * 9/
#     * valid/
#         * 0/
#         * 1/
#         * 2/
#         * 3/
#         * 4/
#         * 5/
#         * 6/
#         * 7/
#         * 8/
#         * 9/
#     * test/

# ## Create Folder

# In[ ]:


new_path = '../working/'
new_train_path = new_path + 'train/'
new_valid_path = new_path + 'valid/'
new_test_path = new_path + 'test/'


# In[ ]:


os.mkdir(new_train_path)
os.mkdir(new_valid_path)
os.mkdir(new_test_path)


# In[ ]:


for target_label in [(str(i) + '/') for i in range(10)]:
    os.mkdir(new_train_path + target_label)
    os.mkdir(new_valid_path + target_label)


# In order to create train and validation sets we will shuffle our data and after it 90% of all training samples will go to the training set and 10% will go to the validation set. Also we will stratify our data by target variables, so our classes in the validation set were as balanced as they are in the training set.

# In[ ]:


X_train, X_valid, y_train, y_valid = train_test_split(X, Y, test_size = 0.1, shuffle = True, random_state = 42, stratify = Y)


# In[ ]:


def save_image_data(X_data, path):
    for i in range(X_data.shape[0]):
        save_path = path + str(i) + '.png'
        mpl.image.imsave(save_path, X_data[i])


# In[ ]:


for label in range(10):
    X_data = X_train[y_train == label]
    save_path = new_train_path + str(label) + '/'
    save_image_data(X_data, save_path)
    
    X_data = X_valid[y_valid == label]
    save_path = new_valid_path + str(label) + '/'
    save_image_data(X_data, save_path)
    
save_image_data(X_test, new_test_path)


# ## Create ImageDataBunch

# Now we will create ImageDataBunch from our folder. After it we will check if the data was loaded correctly. 

# In[ ]:


data = ImageDataBunch.from_folder(new_path, test = 'test')


# In[ ]:


data.show_batch(rows = 4, figsize = (8, 8))


# Data was loaded correctly.

# ## Create Model

# In this notebook we will apply the concept of Transfer Learning and will use resnet18 model, which was pretrained om ImageNet data. We will replace its last layer with two layers. Last layer will have 10 output neurons, one for each category. At the beginning first layers will be frozen, which means that they won't train. Only the last two layers will train.

# In[ ]:


learner = cnn_learner(data, models.resnet18, metrics = accuracy)


# ## Train Model

# Now let's train our model with maximum learning rate equals to 1e-3 for 5 epochs. Method fit_one_cycle will explore learning rates less or equal than max_lr = 1e-3.

# In[ ]:


learner.fit_one_cycle(5, 1e-3)


# In[ ]:


learner.save('five_epochs')


# After the 4-th epoch accuracy has not growth a lot. Moreover, loss on train is greater than loss on validation, which means that we have to tune learning rate. Let's look at the errors of our model.

# In[ ]:


interp = ClassificationInterpretation.from_learner(learner)


# In[ ]:


interp.plot_top_losses(12, figsize = (12, 12))


# It can be seen from the graph above, that model makes mistakes not because of strange samples, but because of model's quality. Let's look at the confusion matrix of our model. 

# In[ ]:


interp.plot_confusion_matrix(figsize = (12, 12), dpi = 70)


# It can be seen from the confusion matrix, that most samples are categorized correctly. So, now we will try to imrove accuracy by unfreezing first layers of our model and tuning maximum learning rate.

# ## Parameters Tuning

# In[ ]:


learner.unfreeze()


# I chose 10 times smaller maximum learning rate for the last layer and 100 times smaller maximum learning rate for the first layer. Other layers will have maximum learning rate between 1е-5 and 1e-4.

# In[ ]:


learner.fit_one_cycle(5, max_lr = slice(1e-5, 1e-4))


# It can be seen from the table above, that accuracy has improved. It means, that unfreezing and training previous layers was quite useful.

# In[ ]:


learner.save('unfreeze_5')


# # Conclusion

# 1. MNIST is a simple data. MNIST digits can be classified even in 2-d space.
# 2. Fastai CNN can achieve high accuracy (about 98%) on MNIST data by applying simple transfer learning.
