#!/usr/bin/env python
# coding: utf-8

# ## Introduction & Loading The Data
# 
# Climate change is one of the most important and pressing issues of our time. In this competition, we have the opportunity to use data to understand important climatic variables. The idea is if we can understand the cloud, we can better understand our environment and how it is changing. 
# 
# The idea of the competition is fairly simple. There are different types or patterns of clouds, and our task is to identify them and classify these types.
# 
# **If you like my work please upvote this Kernel. This encourages or motivates people like me, who contributes to Kaggle on their own time with the intention to share knowledge, to continue the effort. Furthermore, if I made a mistake or can do something more, please leave a comment in the comments section to help me out. Many thanks in advance!**

# In[ ]:


import numpy as np 
import pandas as pd
import os
import cv2
# visualization
import matplotlib.pyplot as plt
from matplotlib import patches as patches
# plotly offline imports
from plotly.offline import download_plotlyjs, init_notebook_mode, iplot
from plotly import subplots
import plotly.express as px
import plotly.figure_factory as ff
from plotly.graph_objs import *
from plotly.graph_objs.layout import Margin, YAxis, XAxis
init_notebook_mode()
# frequent pattern mining
from mlxtend.frequent_patterns import fpgrowth


# In[ ]:


data_path = '/kaggle/input/understanding_cloud_organization'
train_csv_path = os.path.join('/kaggle/input/understanding_cloud_organization','train.csv')
train_image_path = os.path.join('/kaggle/input/understanding_cloud_organization','train_images')


# In[ ]:


pd.read_csv(train_csv_path).head()


# In[ ]:


# load full data and label no mask as -1
train_df = pd.read_csv(train_csv_path).fillna(-1)


# In[ ]:


# image id and class id are two seperate entities and it makes it easier to split them up in two columns
train_df['ImageId'] = train_df['Image_Label'].apply(lambda x: x.split('_')[0])
train_df['Label'] = train_df['Image_Label'].apply(lambda x: x.split('_')[1])
# lets create a dict with class id and encoded pixels and group all the defaults per image
train_df['Label_EncodedPixels'] = train_df.apply(lambda row: (row['Label'], row['EncodedPixels']), axis = 1)


# Now we have our modified train.csv, we answer a few basic questions below:

# In[ ]:


print('Total number of images: %s' % len(train_df['ImageId'].unique()))
print('Images with at least one label: %s' % len(train_df[train_df['EncodedPixels'] != -1]['ImageId'].unique()))
print('Total instance or examples of defects: %s' % len(train_df[train_df['EncodedPixels'] != -1]))


# ## Different Types of Clouds
# 
# The first important thing I want to understand are the followings:
# 1. What are the different types of clouds we have in our dataset.
# 2. How does the mask for these formation looks like.
# 2. Value cout of how many types of clouds we have per image.
# 3. Distribution / frequency of each types of clouds in our dataset.

# In[ ]:


# different types of clouds we have in our dataset
train_df['Label'].unique()


# In[ ]:


# visualize steel image with four classes of faults in seperate columns
def viz_two_instance_of_a_mask(encoded_masks_1, encoded_masks_2):
    '''
    visualize an image with two types of defects by plotting them on two columns
    with the defect overlayed on top of the original image.

    Parameters: 
    img_path (str): path of images
    img_id (str): image id or filename of the path
    encoded_masks (list): a list of strings of encoded masks 
    
    Returns: 
    matplotlib image plot in columns for two classes iwth defect
    '''
    
    fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(20,10))
    axid = 0
    
    for idx, encoded_mask in enumerate(encoded_masks):
        class_id = idx + 1
        if encoded_mask == -1:
            pass
        else:
            mask_decoded = rle_to_mask(encoded_mask, 256, 1600)
            ax[axid].get_xaxis().set_ticks([])
            ax[axid].get_yaxis().set_ticks([])
            ax[axid].text(0.25, 0.25, 'Image Id: %s - Class Id: %s' % (img_id, class_id), fontsize=12)
            ax[axid].imshow(img)
            ax[axid].imshow(mask_decoded, alpha=0.15, cmap="Blues")
            axid += 1


# In[ ]:


# lets group each of the types and their mask in a list so we can do more aggregated counts
grouped_EncodedPixels = train_df.groupby('ImageId')['Label_EncodedPixels'].apply(list)


# In[ ]:


# count the number of labels per image has
labels_per_image_count = grouped_EncodedPixels.apply(lambda x: len([x[0] for x in x if x[1]!=-1])).value_counts()
# count frequency of each type of cloud
label_type_per_image = train_df[train_df['EncodedPixels']!=-1]['Label'].value_counts()


# In[ ]:


# now we have the data ready lets plot them to answer our questions
trace0 = Bar(x=labels_per_image_count.index, y=labels_per_image_count.values, name = 'Number of Cloud Types Per Image')
trace1 = Bar(x=label_type_per_image.index, y=label_type_per_image.values, name = 'Frequency of Different Clouds')
fig = subplots.make_subplots(rows=1, cols=2)
fig.append_trace(trace0, 1, 1)
fig.append_trace(trace1, 1, 2)
fig['layout'].update(height=400, width=900, title='Label Count and Frequency Per Image')
iplot(fig)


# Looks like most of the time we have 2-3 types of cloud formation in one image. 4 types of cloud formation in one image is very rare. Only one type of cloud formation in the image is also somewhat common. Furthermore, the data looks very evenly distributed for all four types of cloud formation. This is going to be an awesome compeition!

# ## Drawing Clouds
# 
# Now we know a little bit about the distribution of our data, we need to take a look at it and get an understanding of what it's all about. First, the masks are encoded so we will need the following function to decode the mask.

# In[ ]:


def rle_to_mask(rle_string, height, width):
    '''
    convert RLE(run length encoding) string to numpy array

    Parameters: 
    rle_string (str): string of rle encoded mask
    height (int): height of the mask
    width (int): width of the mask 

    Returns: 
    numpy.array: numpy array of the mask
    '''
    
    rows, cols = height, width
    
    if rle_string == -1:
        return np.zeros((height, width))
    else:
        rle_numbers = [int(num_string) for num_string in rle_string.split(' ')]
        rle_pairs = np.array(rle_numbers).reshape(-1,2)
        img = np.zeros(rows*cols, dtype=np.uint8)
        for index, length in rle_pairs:
            index -= 1
            img[index:index+length] = 255
        img = img.reshape(cols,rows)
        img = img.T
        return img


# Lets just plot a single image and a mask to get an idea of what it looks like.

# In[ ]:


img = cv2.imread(os.path.join(train_image_path, train_df['ImageId'][0]))
mask_decoded = rle_to_mask(train_df['Label_EncodedPixels'][0][1], img.shape[0], img.shape[1])
fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(20,10))
ax[0].imshow(img)
ax[1].imshow(mask_decoded)


# As it seems, there are images of clouds, and a mask not outlining the exact clouds but roughly the area with the same kind of patterns. And from our last section, we know an image can have more than one type of cloud patterns. So, I propose we visualize an image in two columns. First, shows the different types of cloud formation with a bounding box. On the second column, we visualize the cloud picture with the mask segments as an overlay. Lets go!

# In[ ]:


def bounding_box(img):
    # return max and min of a mask to draw bounding box
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax


# In[ ]:


def plot_cloud(img_path, img_id, label_mask):
    img = cv2.imread(os.path.join(img_path, img_id))
    
    fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(20,10))
    ax[0].imshow(img)
    ax[1].imshow(img)
    cmaps = {'Fish': 'Blues', 'Flower': 'Reds', 'Gravel': 'Greys', 'Sugar':'Purples'}
    colors = {'Fish': 'Blue', 'Flower': 'Red', 'Gravel': 'Gray', 'Sugar':'Purple'}
    for label, mask in label_mask:
        mask_decoded = rle_to_mask(mask, img.shape[0], img.shape[1])
        if mask != -1:
            rmin, rmax, cmin, cmax = bounding_box(mask_decoded)
            bbox = patches.Rectangle((cmin,rmin),cmax-cmin,rmax-rmin,linewidth=1,edgecolor=colors[label],facecolor='none')
            ax[0].add_patch(bbox)
            ax[0].text(cmin, rmin, label, bbox=dict(fill=True, color=colors[label]))
            ax[1].imshow(mask_decoded, alpha=0.3, cmap=cmaps[label])
            ax[0].text(cmin, rmin, label, bbox=dict(fill=True, color=colors[label]))


# Lets print out 10 random samples!

# In[ ]:


for image_id, label_mask in grouped_EncodedPixels.sample(10).iteritems():
    plot_cloud(train_image_path, image_id, label_mask)


# ## Zooming In To The Cloud Formations
# 
# In the last section, we visualized many types of cloud formation at once. This visualization is good as it gives us a good high level picture of our training images. Also, excuse me for the terrible pun. However, in this section we will zoom into the different type of cloud formations by using the mask to mute out the section that doesn't belong to the type of cloud formation we are interested in. Our main question for this section is the following:
# 
# * How do different types of cloud formations look like to the naked eye.

# In[ ]:


def get_mask_cloud(img_path, img_id, label, mask):
    img = cv2.imread(os.path.join(img_path, img_id), 0)
    mask_decoded = rle_to_mask(mask, img.shape[0], img.shape[1])
    mask_decoded = (mask_decoded > 0.0).astype(int)
    img = np.multiply(img, mask_decoded)
    return img


# In[ ]:


def draw_label_only(label):
    samples_df = train_df[(train_df['EncodedPixels']!=-1) & (train_df['Label']==label)].sample(2)
    count = 0
    fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(20,10))
    for idx, sample in samples_df.iterrows():
        img = get_mask_cloud(train_image_path, sample['ImageId'], sample['Label'],sample['EncodedPixels'])
        ax[count].imshow(img, cmap="gray")
        count += 1


# #### Fish Cloud Formation

# In[ ]:


draw_label_only('Fish')


# #### Flower Cloud Formation

# In[ ]:


draw_label_only('Flower')


# #### Gravel Cloud Formation

# In[ ]:


draw_label_only('Gravel')


# #### Sugar Cloud Formation

# In[ ]:


draw_label_only('Sugar')


# Well, I sort of get the sense why the cloud formations are called what they are called. What I am afraid of is often times, it seems like there is a little bit of cloud formation from one type overlaps with cloud formation of another type. I am interested to see how our algorithm performs in these scenarios.

# ## Cloud That Forms Together Stays Together
# As we saw in the previous visualizations, it's very common for multiple cloud formations to be present in a single image. In this section we want to answer:
# 1. Which cloud formations occur frequently.
# 2. Which cloud formations hardly ever appears together.
# 
# To do this, we will use a really simple data mining algorithm called Frequent Pattern Mining. FP Growth is a particular algorithmic implementation of Frequent Pattern, which aims to identify items that appear frequently together in a list. 

# In[ ]:


# create a series with fault classes
label_per_image = grouped_EncodedPixels.apply(lambda encoded_list: [x[0] for x in encoded_list if x[1] != -1])


# In[ ]:


# create a list of dict with count of each fault class
label_per_image_list = []
for r in label_per_image.iteritems():
    label_count = {'Fish':0,'Flower':0,'Gravel':0,'Sugar':0}
    # go over each class and 
    for image_label in r[1]:
        label_count[image_label] = 1
    label_per_image_list.append(label_count)


# In[ ]:


# do FP calculation with all image
label_per_image_df = pd.DataFrame(label_per_image_list)
label_fp_df = fpgrowth(label_per_image_df, use_colnames=True, min_support=0.001)
label_fp_df = label_fp_df.sort_values(by=['support'])
label_combi_fp_df = label_fp_df[label_fp_df['itemsets'].apply(lambda x: len(x) > 1)]
label_combi_fp_df['itemsets'] = label_combi_fp_df['itemsets'].apply(lambda x: ', '.join(x))


# In[ ]:


fig = px.bar(label_combi_fp_df, x="support", y="itemsets", orientation='h',             title='Frequent Patterns of The Cloud Formation')
fig.show()


# Sugar cloud formation frequently appears together in the images with Gravel or Fish cloud formation. Sugar also appears with Flower cloud formation but is less frequent. Gravels and Fish cloud formation also appears with other cloud formation. Sugar, Gravel, and Fish also appears all together in some instances.
# 
# Flower tends to occur less frequently with other clouds, and the combination of Gravel and Flower occurs but at much less frequency compared to others. In fact, Sugar, Gravel, and Fish appear all together more frequently than Grave and Flower. However, it's not like Flower cloud formation never occurs with other cloud formation, just occurs less frequently compared to others.
# 
# In summary, they are all combination of cloud formations appearing together is a possibility, and the combinations between Sugar, Fish, and Gravel are more likely than with Flower cloud formation.

# ## Surface Area Ratio Per Cloud Formation
# 
# In this section we are interested in the following questions:
# 1. How does the distribution of the surface area for different cloud formation masks look like?
# 2. Are there any types of cloud formations that appear very frequently but have a very small surface area in our training dataset?

# In[ ]:


# we will use the following function to decode our mask to binary and count the sum of the pixels for our mask.
def get_binary_mask_sum(encoded_mask):
    mask_decoded = rle_to_mask(encoded_mask, width=2100, height=1400)
    binary_mask = (mask_decoded > 0.0).astype(int)
    return binary_mask.sum()


# In[ ]:


# calculate sum of the pixels for the mask per cloud formation
train_df['mask_pixel_sum'] = train_df.apply(lambda x: get_binary_mask_sum(x['EncodedPixels']), axis=1)


# In[ ]:


# plot a histogram and boxplot combined of the mask pixel sum per cloud formation
fig = px.histogram(train_df[train_df['mask_pixel_sum']!=0][['Label','mask_pixel_sum']], 
                   x="mask_pixel_sum", y="Label", color="Label", marginal="box")

fig['layout'].update(title='Histogram and Boxplot of Sum of Mask Pixels Per Cloud Formation')

fig.show()


# Analysis to come soon!

# *Well, thats awesome! Now you can see what cloud formations that occur frequently together. I will be back soon with more updates, and good luck for the competition!!*
