#!/usr/bin/env python
# coding: utf-8

# This notebook kernel attemps to do basic analysis of images in the dataset and uncover evidence of tampering using steganography algorithms. It is targeted at anyone just starting out with this competition. You need no prior knowledge of steganography to understand this kernel
# 
# 
# # Basics
# First up, let us understand the basics of [steganogarphy](https://www.wired.com/story/steganography-hacker-lexicon/). 
# * ***Definition:*** Steganography is essentially the practice of hiding secret messages in an otherwise non-secret medium. It has existed for centuries in different forms.
# * ***Current:*** Modern day steganography (including this competition) deals with encoding secret information in digital images. How is this done:
#     * The pixel values are altered to store hidden messages (henceforth called the payload).
#     * However, the altered & original image does **not** show any visual difference to the naked eye. 
# * ***Techniques:*** Now, different algorithms can be used to encode the payload (i.e., change the pixel values). This competition uses 3 algorithms:
#     1. JMiPOD
#     2. JUNIWARD
#     3. UERD
# * ***Data:*** We are given a dataset of 75k images with each image in its original unaltered state(cover) and modified by each of the 3 algorithms respectively.
#     
# * ***Objective:*** The goal of this competition  is to come up with a machine learning model to predict whether a given test image has a payload encoded or not. We are not concerend with recovering the contents of the payload.
#   * Essentially, we are dealing with one aspect of steganalysis which is the study of detecting and possibly recovering messages hidden using steganography.

# # Load Libraries

# In[ ]:


import numpy as np
import pandas as pd
import cv2
import os
import matplotlib
# matplotlib.use('nbagg')
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
# %matplotlib inline
# import mpld3
# mpld3.enable_notebook()


# # Display samples
# Let us load & display some random sample images from the dataset

# In[ ]:


base_path = '/kaggle/input/alaska2-image-steganalysis/'
algorithm = ('Cover(Unaltered)', 'JMiPOD', 'UERD', 'JUNIWARD')
fig, axes = plt.subplots(nrows=4, ncols=4, figsize = (11,11) )
np.random.seed(55)
for i,id in enumerate(np.random.randint(0,75001,4)):
    id = '{:05d}'.format(id)    
    cover_path = os.path.join(base_path, 'Cover', id + '.jpg')
    jmipod_path = os.path.join(base_path, 'JMiPOD', id + '.jpg')
    uerd_path = os.path.join(base_path, 'UERD', id + '.jpg')
    juniward_path = os.path.join(base_path, 'JUNIWARD', id + '.jpg')
    cover_img = plt.imread(cover_path)
    jmipod_img = plt.imread(jmipod_path)
    uerd_img = plt.imread(uerd_path)
    juniward_img = plt.imread(juniward_path)
    axes[i,0].imshow(cover_img)
    axes[i,1].imshow(jmipod_img)
    axes[i,2].imshow(uerd_img)
    axes[i,3].imshow(juniward_img)
    axes[i,0].set(ylabel=id+'.jpg')

for i,algo in enumerate(algorithm):
    axes[0,i].set(title=algo) 
for ax in axes.flat:
    ax.set(xticks=[], yticks=[])
plt.show()


# * As you can see above, we displayed 5 random images.
# * By mere visual observation, there is no sign of any modification to the image.
# * They look completely visually similar. 
# * Lets look deeper

# # Histogram analysis
# Let us plot the histogram of the images and look to comapare the images. We will pick one of the samples and plot its histogram for all 3 channels and all 3 algorithms.

# In[ ]:


cover_hist = {}
jmipod_hist = {}
uerd_hist = {}
juniward_hist = {}
color = ('b','g','r')
for i,col in enumerate(color):
    cover_hist[col] = cv2.calcHist([cover_img],[i],None,[256],[0,256])
    jmipod_hist[col] = cv2.calcHist([jmipod_img],[i],None,[256],[0,256])
    uerd_hist[col] = cv2.calcHist([uerd_img],[i],None,[256],[0,256])
    juniward_hist[col] = cv2.calcHist([juniward_img],[i],None,[256],[0,256])
    
fig_hist, axes_hist = plt.subplots(nrows=2, ncols=2, figsize=(12,12))
for ax, hist, algo in zip(axes_hist.flat, [cover_hist, jmipod_hist, uerd_hist, juniward_hist], algorithm):
    ax.plot(hist['r'], color = 'r', label='r')
    ax.plot(hist['g'], color = 'g', label='g')
    ax.plot(hist['b'], color = 'b', label='b')
    ax.set(ylabel='# of pixels', xlabel='Pixel value(0-255)', title=algo)
    ax.legend()
fig_hist.subplots_adjust(wspace=0.4, hspace=0.3)
fig_hist.suptitle('Histogram of a sample (' + id + '.jpg)', fontsize=20)
    #     ax.xlim([0,256])
plt.show()


# * Looking at the histogram gives us more clues.
# * Indeed, all 4 histograms are quite similar. 
#   * Yet, if we look closely, we observe some minor differences.
#   * If you open the notebook in editing mode and run it, you can zoom and check for yourself.
# 
# Let us zoom in and plot the red channel of all 4 images in a single plot for pixel values between 50-80.

# In[ ]:


fig, ax = plt.subplots(figsize=(10,10))
ax.plot(cover_hist['r'][50:80], color = 'c', label=algorithm[0])
ax.plot(jmipod_hist['r'][50:80], color = 'm', label=algorithm[1])
ax.plot(uerd_hist['r'][50:80], color = 'y', label=algorithm[2])
ax.plot(juniward_hist['r'][50:80], color = 'g', label=algorithm[3])
ax.legend()
ax.set_ylabel('# of pixels', fontsize=15) 
ax.set_xlabel('Pixel value(50-80)', fontsize=15)
ax.xaxis.set(ticklabels=np.linspace(50,80,8, dtype=np.int))
ax.set_title('R-channel Histogram Compared (zoomed in)', fontsize=20)
plt.show()


# * The difference was subtle before.
# * But this histogram shows the images are clearly modified with minor changes using the steganography algorithms.
# * But they are modified in a manner not visible to the naked eye.

# # Difference image
# Let us print & look at some pixel values directly now.

# In[ ]:


print('Cover image:\n', cover_img[10:20,10:20,0])
print('\nJMiPOD image:\n', jmipod_img[10:20,10:20,0])
print('\nUERD image:\n', uerd_img[10:20,10:20,0])
print('\nJUNIWARD image:\n', juniward_img[10:20,10:20,0])


# Above, you can clearly see similar but slightly modified pixel values. 
# 
# An easy way to observe the dissimilarities is to display the absolute value difference of altered and cover image.

# In[ ]:


fig, axes = plt.subplots(nrows=4, ncols=4, figsize = (11,11) )
np.random.seed(55)
def disp_diff_img(alt, ref, ax, chnl=0):
    diff = np.abs(alt.astype(np.int)-ref.astype(np.int)).astype(np.uint8)
    ax.imshow(diff[:,:,chnl], vmin=0, vmax=np.amax(diff[:,:,chnl]), cmap='hot')
for i,id in enumerate(np.random.randint(0,75001,4)):
    id = '{:05d}'.format(id)    
    cover_path = os.path.join(base_path, 'Cover', id + '.jpg')
    jmipod_path = os.path.join(base_path, 'JMiPOD', id + '.jpg')
    uerd_path = os.path.join(base_path, 'UERD', id + '.jpg')
    juniward_path = os.path.join(base_path, 'JUNIWARD', id + '.jpg')
    cover_img = plt.imread(cover_path)
    jmipod_img = plt.imread(jmipod_path)
    uerd_img = plt.imread(uerd_path)
    juniward_img = plt.imread(juniward_path)
    axes[i,0].imshow(cover_img)
    disp_diff_img(jmipod_img, cover_img, axes[i,1], 0)
    disp_diff_img(uerd_img, cover_img, axes[i,2], 0)
    disp_diff_img(juniward_img, cover_img, axes[i,3], 0)
    axes[i,0].set(ylabel=id+'.jpg')

for i,algo in enumerate(algorithm):
    axes[0,i].set(title=algo + 'diff') 
for ax in axes.flat:
    ax.set(xticks=[], yticks=[])
plt.show()


# * We took absolute of (original - altered image) and plot it using the 'hot' colorspace.
# * This basic analysis clearly shows evidence of tampering with the pixel values.
# * Also, its possible to see how all 3 algorithms has modified the original image in very different ways.
# * The exact technique of each algorithm needs to be known to understand why each image has been modified so differently.
# 
# 
# I plan to update it further with more analysis on the algorithms in the future.
# 
# Please upvote if you have enjoyed reading the kernel :)
