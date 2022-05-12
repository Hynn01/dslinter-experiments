#!/usr/bin/env python
# coding: utf-8

# ## Introduction
# 
# The purpose of this competition is to identify "Pneumothorax" or a collapsed lung from chest x-rays. Pneumothorax is a condition that is responsible for making people suddenly gasp for air, and feel helplessly breathless for no apparent reason. Pneumothorax is visually diagnosed by radiologist, and even for a professional with years of experience; it is difficult to confirm. Neural networks and advanced data science techniques can hopefully help capture all the latent features and detect pneumothorax consistently.
# 
# So ultimately, we want to develop a model to identify and segment pneumothorax from a set of chest radiographic images.
# 
# **I feel that this is an important competition, with potential of making a huge amount of impact, and we (the Kaggle community) should give it our best shot. The lines of python codes we write staying up a few nights and sacrificing a few leisure activity can help save someone's life one day.**
# 
# Therefore, I wanted to create a Kernel that lowers the barrier to enter this competition. For those who are new, I will try to add as much explanation in the comment section as possible. For those who has very little time outside their professional and personal responsibilities, I hope this serves you a quick gateway to get started.
# 
# #### Credits
# 
# I very often see plagiarism in Kaggle community. I really find it difficult to understand why as "a candle loses nothing by lighting another candle". We should always credit original authors and upvote their work before using it. Below is the list of great Kernels I had the good fortune of taking help from while building this Kernel:
# 
# * [Chest xray, DICOM, viz, U-nets](https://www.kaggle.com/jesperdramsch/intro-chest-xray-dicom-viz-u-nets-full-data) by [Jesper](https://www.kaggle.com/jesperdramsch)
# * [Full Dataset](https://www.kaggle.com/seesee/full-dataset) by [See--](https://www.kaggle.com/seesee)
# * [Keras Data Generator 1024x1024 (unet)](https://www.kaggle.com/ppepijn/keras-data-generator-1024x1024-unet) by [pepijn](https://www.kaggle.com/ppepijn)
# * [EDA Can you see the Pneumothorax?](https://www.kaggle.com/freeman89/eda-can-you-see-the-pneumothorax) by [Kostadinov](https://www.kaggle.com/freeman89)
# * [Image Pre-processing for Chest X-ray](https://www.kaggle.com/seriousran/image-pre-processing-for-chest-x-ray) by [Chanran Kim
# ](https://www.kaggle.com/seriousran)
# 
# 
# #### PS: I am still working on this competition, but in a different Kernel: https://www.kaggle.com/ekhtiar/finding-pneumo-part-2-resunet. If you like the extension of this work and find it useful, please upvote it or leave a comment to let me know that what I am doing is indeed useful :)

# In[ ]:


# Basic imports for the entire Kernel
import numpy as np
import pandas as pd


# This competition provides a library for masking. I will go more in depth into what masking is in later part of this competition. For now, we import this library in a hacky way below, by adding the path into our system.

# In[ ]:


# import mask function
import sys
sys.path.insert(0, '../input/siim-acr-pneumothorax-segmentation')
from mask_functions import rle2mask, mask2rle


# ## Section 1: Down/loading The Data

# **Before we can dive deep into our data, we need to structure the data provided. In this section, we will create a pandas dataframe with metadata and filepath of our x-rays.**
# 
# The competition organizers are hosting this data in Google Cloud Platform. Downloading this data is very simple, and full instruction is provided [here](https://storage.googleapis.com/kaggle-media/competitions/siim/SIIM%20Cloud%20Healthcare%20API%20Documentation.pdf). However, we are allowed to download the data and upload it for the sake of this competition. To help us all out, this [kernel](https://www.kaggle.com/seesee/full-dataset) uploads the full dataset. I forked this kernel to have a quick start.
# 
# After you download the dataset, it is important to understand the hierarchy or folder structure of the images. We have images seperated in a test and train folder. Python has a great utility library called *glob*, which makes pealing these oninion like file structures effortless. 
# 
# Within these folders, for every dataset, we have two more folders encapsulating it. The data itself is in a format called DICOM (Digital Imaging and Communications in Medicine), which is the international standard to transmit, store, retrieve, print, process, and display medical imaging information. This format can store these images with metadata (patient id, age, sex, etc.) in one file with .dcm extension. Parsing this is also super easy with *pydicom*, which we will also use.

# In[ ]:


# imports for loading data
import pydicom
from glob import glob
from tqdm import tqdm


# Run-length encoding (RLE) is a very simple form of lossless data compression. This [video](https://www.youtube.com/watch?v=Yl50cJScObI) on YouTube explains how it works. This competition provides a seperate csv file with encodings for each image, which annotates or labels the segment of the image consisting pneumothorax. However, images without pneumothorax have a mask value of -1.

# In[ ]:


# load rles
rles_df = pd.read_csv('../input/siim-train-test/siim/train-rle.csv')
# the second column has a space at the start, so manually giving column name
rles_df.columns = ['ImageId', 'EncodedPixels']


# In[ ]:


def dicom_to_dict(dicom_data, file_path, rles_df, encoded_pixels=True):
    """Parse DICOM dataset and returns a dictonary with relevant fields.

    Args:
        dicom_data (dicom): chest x-ray data in dicom format.
        file_path (str): file path of the dicom data.
        rles_df (pandas.core.frame.DataFrame): Pandas dataframe of the RLE.
        encoded_pixels (bool): if True we will search for annotation.
        
    Returns:
        dict: contains metadata of relevant fields.
    """
    
    data = {}
    
    # Parse fields with meaningful information
    data['patient_name'] = dicom_data.PatientName
    data['patient_id'] = dicom_data.PatientID
    data['patient_age'] = int(dicom_data.PatientAge)
    data['patient_sex'] = dicom_data.PatientSex
    data['pixel_spacing'] = dicom_data.PixelSpacing
    data['file_path'] = file_path
    data['id'] = dicom_data.SOPInstanceUID
    
    # look for annotation if enabled (train set)
    if encoded_pixels:
        encoded_pixels_list = rles_df[rles_df['ImageId']==dicom_data.SOPInstanceUID]['EncodedPixels'].values
       
        pneumothorax = False
        for encoded_pixels in encoded_pixels_list:
            if encoded_pixels != ' -1':
                pneumothorax = True
        
        # get meaningful information (for train set)
        data['encoded_pixels_list'] = encoded_pixels_list
        data['has_pneumothorax'] = pneumothorax
        data['encoded_pixels_count'] = len(encoded_pixels_list)
        
    return data


# In[ ]:


# create a list of all the files
train_fns = sorted(glob('../input/siim-train-test/siim/dicom-images-train/*/*/*.dcm'))
# parse train DICOM dataset
train_metadata_df = pd.DataFrame()
train_metadata_list = []
for file_path in tqdm(train_fns):
    dicom_data = pydicom.dcmread(file_path)
    train_metadata = dicom_to_dict(dicom_data, file_path, rles_df)
    train_metadata_list.append(train_metadata)
train_metadata_df = pd.DataFrame(train_metadata_list)


# In[ ]:


# create a list of all the files
test_fns = sorted(glob('../input/siim-train-test/siim/dicom-images-test/*/*/*.dcm'))
# parse test DICOM dataset
test_metadata_df = pd.DataFrame()
test_metadata_list = []
for file_path in tqdm(test_fns):
    dicom_data = pydicom.dcmread(file_path)
    test_metadata = dicom_to_dict(dicom_data, file_path, rles_df, encoded_pixels=False)
    test_metadata_list.append(test_metadata)
test_metadata_df = pd.DataFrame(test_metadata_list)


# ## Section 2: Visualizing The Chest X-Ray 
# 
# In this section, we will print out a few x-rays to the output, to get a feel for what we are dealing with. To get a better feel, we will also use the mask and print out annotated area with pneumothorax. 

# In[ ]:


import matplotlib.pyplot as plt
from matplotlib import patches as patches


# In the pandas dataframe we created earlier, we created a column to contain the path to the DICOM data. We did this because fitting and keeping the entire image data also in pandas dataframe would be a waste of resource. Below, we use that path to load the dataset and use retrieve the image data.

# In[ ]:


num_img = 4
subplot_count = 0
fig, ax = plt.subplots(nrows=1, ncols=num_img, sharey=True, figsize=(num_img*10,10))
for index, row in train_metadata_df.sample(n=num_img).iterrows():
    dataset = pydicom.dcmread(row['file_path'])
    ax[subplot_count].imshow(dataset.pixel_array, cmap=plt.cm.bone)
    # label the x-ray with information about the patient
    ax[subplot_count].text(0,0,'Age:{}, Sex: {}, Pneumothorax: {}'.format(row['patient_age'],row['patient_sex'],row['has_pneumothorax']),
                           size=26,color='white', backgroundcolor='black')
    subplot_count += 1


# For most of us, we are not a radiologist, who are trained to catch Pneumothorax in the xray. So we will use the masking data and visualize the spot where Pneumothorax is on the xray.
# 
# I feel in this competition, our ability to use different filters will play a big role. In [Image Pre-processing for Chest X-ray notebook](https://www.kaggle.com/seriousran/image-pre-processing-for-chest-x-ray)  there is an example of using CLAHE [(Contrast Limited Adaptive Histogram Equalization)](https://docs.opencv.org/3.1.0/d5/daf/tutorial_py_histogram_equalization.html). I am also visualizing a version of the x-ray with CLAHE applied.

# In[ ]:


def bounding_box(img):
    # return max and min of a mask to draw bounding box
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax

def plot_with_mask_and_bbox(file_path, mask_encoded_list, figsize=(20,10)):
    
    import cv2
    
    """Plot Chest Xray image with mask(annotation or label) and without mask.

    Args:
        file_path (str): file path of the dicom data.
        mask_encoded (numpy.ndarray): Pandas dataframe of the RLE.
        
    Returns:
        plots the image with and without mask.
    """
    
    pixel_array = pydicom.dcmread(file_path).pixel_array
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
    clahe_pixel_array = clahe.apply(pixel_array)
    
    # use the masking function to decode RLE
    mask_decoded_list = [rle2mask(mask_encoded, 1024, 1024).T for mask_encoded in mask_encoded_list]
    
    fig, ax = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(20,10))
    
    # print out the xray
    ax[0].imshow(pixel_array, cmap=plt.cm.bone)
    # print the bounding box
    for mask_decoded in mask_decoded_list:
        # print out the annotated area
        ax[0].imshow(mask_decoded, alpha=0.3, cmap="Reds")
        rmin, rmax, cmin, cmax = bounding_box(mask_decoded)
        bbox = patches.Rectangle((cmin,rmin),cmax-cmin,rmax-rmin,linewidth=1,edgecolor='r',facecolor='none')
        ax[0].add_patch(bbox)
    ax[0].set_title('With Mask')
    
    # plot image with clahe processing with just bounding box and no mask
    ax[1].imshow(clahe_pixel_array, cmap=plt.cm.bone)
    for mask_decoded in mask_decoded_list:
        rmin, rmax, cmin, cmax = bounding_box(mask_decoded)
        bbox = patches.Rectangle((cmin,rmin),cmax-cmin,rmax-rmin,linewidth=1,edgecolor='r',facecolor='none')
        ax[1].add_patch(bbox)
    ax[1].set_title('Without Mask - Clahe')
    
    # plot plain xray with just bounding box and no mask
    ax[2].imshow(pixel_array, cmap=plt.cm.bone)
    for mask_decoded in mask_decoded_list:
        rmin, rmax, cmin, cmax = bounding_box(mask_decoded)
        bbox = patches.Rectangle((cmin,rmin),cmax-cmin,rmax-rmin,linewidth=1,edgecolor='r',facecolor='none')
        ax[2].add_patch(bbox)
    ax[2].set_title('Without Mask')
    plt.show()


# In[ ]:


# lets take 10 random samples of x-rays with 
train_metadata_sample = train_metadata_df[train_metadata_df['has_pneumothorax']==1].sample(n=10)
# plot ten xrays with and without mask
for index, row in train_metadata_sample.iterrows():
    file_path = row['file_path']
    mask_encoded_list = row['encoded_pixels_list']
    print('image id: ' + row['id'])
    plot_with_mask_and_bbox(file_path, mask_encoded_list)


# Wow, this is very difficult to spot pneumothorax in chest x-ray. For some cases, a poorly visible contour is in the x-ray, but more often than not, I just can't see how they spot it. It feels like Clahe helps it make it more visible, but I don't know for sure. Our network result will give us the final verdict later.

# ## Section 3: Explore The Metadata

# In this section we will explore the metadata that is associated with the x-rays. Although there isn't a lot of metadata, but it is always good to understand how the data is distributed before we dive deep into the images. For this section we will use plotly in offline mode. I still can't believe sometimes that plotly is open sourced to work offline as well!

# In[ ]:


# plotly offline imports
from plotly.offline import download_plotlyjs, init_notebook_mode, iplot
from plotly import tools
from plotly.graph_objs import *
from plotly.graph_objs.layout import Margin, YAxis, XAxis
init_notebook_mode()


# ### Annotation / Mask / Label

# First thing first, let's take a look at the annotation and see how our data is distributed over positive and negative instances.

# In[ ]:


# print missing annotation
missing_vals = train_metadata_df[train_metadata_df['encoded_pixels_count']==0]['encoded_pixels_count'].count()
print("Number of x-rays with missing labels: {}".format(missing_vals))


# In[ ]:


nok_count = train_metadata_df['has_pneumothorax'].sum()
ok_count = len(train_metadata_df) - nok_count
x = ['No Pneumothorax','Pneumothorax']
y = [ok_count, nok_count]
trace0 = Bar(x=x, y=y, name = 'Ok vs Not OK')
nok_encoded_pixels_count = train_metadata_df[train_metadata_df['has_pneumothorax']==1]['encoded_pixels_count'].values
trace1 = Histogram(x=nok_encoded_pixels_count, name='# of annotations')
fig = tools.make_subplots(rows=1, cols=2)
fig.append_trace(trace0, 1, 1)
fig.append_trace(trace1, 1, 2)
fig['layout'].update(height=400, width=900, title='Pneumothorax Instances')
iplot(fig)


# * We do have small amount of x-rays where the patient has pneumothorax
# * There are 37 instance of x-rays without any annotation
# * For most of the x-rays, there is only one mask / area that is labeled
# * There are a considerable amount of images with more than one area where you can see pneumothorax
# * It is good to keep in mind that an image can have more than one annotation

# ### Age, Sex, and Pneumothorax

# My first curiosity is to see if there is a relationship with the patient's age and this condition. To investigate, let's plot a histogram of the age for people who has pneumothorax and doesn't have pneumothorax.

# In[ ]:


pneumo_pat_age = train_metadata_df[train_metadata_df['has_pneumothorax']==1]['patient_age'].values
no_pneumo_pat_age = train_metadata_df[train_metadata_df['has_pneumothorax']==0]['patient_age'].values


# In[ ]:


pneumothorax = Histogram(x=pneumo_pat_age, name='has pneumothorax')
no_pneumothorax = Histogram(x=no_pneumo_pat_age, name='no pneumothorax')
fig = tools.make_subplots(rows=1, cols=2)
fig.append_trace(pneumothorax, 1, 1)
fig.append_trace(no_pneumothorax, 1, 2)
fig['layout'].update(height=400, width=900, title='Patient Age Histogram')
iplot(fig)


# What the! There is a patient who is like 400 plus, and there is another patient who suspiciously looks too old to be true. I will remove these two anomalies and do a box plot below.

# In[ ]:


trace1 = Box(x=pneumo_pat_age, name='has pneumothorax')
trace2 = Box(x=no_pneumo_pat_age[no_pneumo_pat_age <= 120], name='no pneumothorax')
data = [trace1, trace2]
iplot(data)


# Few observations:
# * There are two anomalies for the age value in the dataset.
# * The median age for people who has pneumothorax is not considerably but is slightly lower.

# In[ ]:


train_male_df = train_metadata_df[train_metadata_df['patient_sex']=='M']
train_female_df = train_metadata_df[train_metadata_df['patient_sex']=='F']


# In[ ]:


male_ok_count = len(train_male_df[train_male_df['has_pneumothorax']==0])
female_ok_count = len(train_female_df[train_female_df['has_pneumothorax']==0])
male_nok_count = len(train_male_df[train_male_df['has_pneumothorax']==1])
female_nok_count = len(train_female_df[train_female_df['has_pneumothorax']==1])


# In[ ]:


ok = Bar(x=['male', 'female'], y=[male_ok_count, female_ok_count], name='no pneumothorax')
nok = Bar(x=['male', 'female'], y=[male_nok_count, female_nok_count], name='has pneumothorax')

data = [ok, nok]
layout = Layout(barmode='stack', height=400)

fig = Figure(data=data, layout=layout)
iplot(fig, filename='stacked-bar')


# Now let's plot two donut chart next to each other, to show the portion of people with pneumothorax and without pneomothorax between male and female.

# In[ ]:


m_pneumo_labels = ['no pneumothorax','has pneumothorax']
f_pneumo_labels = ['no pneumothorax','has pneumothorax']
m_pneumo_values = [male_ok_count, male_nok_count]
f_pneumo_values = [female_ok_count, female_nok_count]
colors = ['#FEBFB3', '#E1396C']


# In[ ]:


# original source code: https://plot.ly/python/pie-charts/#donut-chart

fig = {
  "data": [
    {
      "values": m_pneumo_values,
      "labels": m_pneumo_labels,
      "domain": {"column": 0},
      "name": "Male",
      "hoverinfo":"label+percent",
      "hole": .4,
      "type": "pie"
    },
    {
      "values": f_pneumo_values,
      "labels": f_pneumo_labels,
      "textposition":"inside",
      "domain": {"column": 1},
      "name": "Female",
      "hoverinfo":"label+percent",
      "hole": .4,
      "type": "pie"
    }],
  "layout": {
        "title":"Pneumothorax - Male vs Female",
        "grid": {"rows": 1, "columns": 2},
        "annotations": [
            {
                "font": {
                    "size": 20
                },
                "showarrow": False,
                "text": "Male",
                "x": 0.20,
                "y": 0.5
            },
            {
                "font": {
                    "size": 20
                },
                "showarrow": False,
                "text": "Female",
                "x": 0.8,
                "y": 0.5
            }
        ]
    }
}
iplot(fig)


# Observations:
# * We have slightly more instanes of male x-rays.
# * Male and female are equally likely to have Pneumothorax (at least according to our dataset)

# ## Section 4: Analysing The Affected Area
# 
# In the metadata, we don't have an attribute explicitly describing how big the area in patient's chest where gas has built up. However, we do have the masking information. In this section, I will propose a rather naive and simple solution to convert this to an approximate area. All the masks are in a 1024x1024 matrix. First we will count the non zero elements in this array to count the number of pixels we have. Pixel spacing gives us the length and width of patient's chest per pixel. We will multiply the area per pixel driven of pixel spacing by total number of pixels to get an estimation of the affected area in sq mm.

# In[ ]:


def get_affected_area(encoded_pixels_list, pixel_spacing):
    
    # take the encoded mask, decode, and get the sum of nonzero elements
    pixel_sum = 0
    
    for encoded_mask in encoded_pixels_list:
        mask_decoded = rle2mask(encoded_mask, 1024, 1024).T
        pixel_sum += np.count_nonzero(mask_decoded)
        
    area_per_pixel = pixel_spacing[0] * pixel_spacing[1]
    
    return pixel_sum * area_per_pixel


# In[ ]:


# create a subset of dataframe for pneumothorax patients
pneumothorax_df = train_metadata_df[train_metadata_df['has_pneumothorax']==1].copy()
# get sum of non zero elements in mask
pneumothorax_df['pneumothorax_area'] = pneumothorax_df.apply(lambda row: get_affected_area(row['encoded_pixels_list'], row['pixel_spacing']),axis=1)


# Now we have the data in a format we can plot, let's do a boxplot to see if there is a difference in the size of affected area between male and female.

# In[ ]:


pneumothorax_df_m = pneumothorax_df[pneumothorax_df['patient_sex']=='M']
pneumothorax_df_f = pneumothorax_df[pneumothorax_df['patient_sex']=='F']
pneumo_size_m = pneumothorax_df_m['pneumothorax_area'].values
pneumo_size_f = pneumothorax_df_f['pneumothorax_area'].values


# In[ ]:


pneumo_size_m_trace = Box(x = pneumo_size_m, name='M')
pneumo_size_f_trace = Box(x = pneumo_size_f, name='F')
layout = Layout(title='Pneumothorax Affected Area for Male and Female Population', 
               xaxis = XAxis(title='Area (in sq mm)'))

data = [pneumo_size_m_trace, pneumo_size_f_trace]
fig = Figure(data=data, layout=layout)
iplot(fig)


# That's interesting! The size of affected area is clearly larger in male population vs female population. There is a considerable amount of difference for the median, q3, and upper fence for both of the groups. Also, there are a lot of outliers in the data. Let's do a scatter plot between age and size of the affected area to zoom into our data more.

# In[ ]:


pneumo_size_m_trace = Scatter(x=pneumothorax_df_m['patient_age'].values, 
                              y=pneumothorax_df_m['pneumothorax_area'].values, 
                              mode='markers', name='Male')

pneumo_size_f_trace = Scatter(x=pneumothorax_df_f['patient_age'].values, 
                              y=pneumothorax_df_f['pneumothorax_area'].values, 
                              mode='markers', name='Female')

layout = Layout(title='Pneumothorax Affected Area vs Age for Male and Female Population', 
                yaxis=YAxis(title='Area (in sq mm)'), xaxis=XAxis(title='Age'))

data = [pneumo_size_m_trace, pneumo_size_f_trace]
fig = Figure(data=data, layout=layout)
iplot(fig)


# From the scatter plot we see the point between size of the affected area being large in male than female. However for the different age groups, I don't observe something super interesting. To zoom in even more, we will do a bubble chart, where x axis is the age, y axis is the number of bounding boxes or annotated  labeled area, and the size of the bubble or marker is the total area of the affected parts of the chest.

# In[ ]:


size_m = pneumothorax_df_m['pneumothorax_area'].values
size_ref_m = 2.*max(size_m)/(40.**2)
size_f = pneumothorax_df_f['pneumothorax_area'].values
size_ref_f = 2.*max(size_f)/(40.**2)

pneumo_size_m_trace = Scatter(x=pneumothorax_df_m['patient_age'].values, 
                              y=pneumothorax_df_m['encoded_pixels_count'].values,
                              marker=dict(size= size_m, sizemode='area', sizeref=size_ref_m, sizemin=4), 
                              mode='markers', name='Male')

pneumo_size_f_trace = Scatter(x=pneumothorax_df_f['patient_age'].values, 
                              y=pneumothorax_df_f['encoded_pixels_count'].values,
                              marker=dict(size=size_f, sizemode='area', sizeref=size_ref_f, sizemin=4), 
                              mode='markers', name='Female')

layout = Layout(title='Pneumothorax Affected Area vs Age for Male and Female Population', yaxis=YAxis(title='Area (in sq mm)'), xaxis=XAxis(title='Age'))

data = [pneumo_size_m_trace, pneumo_size_f_trace]
fig = Figure(data=data, layout=layout)
iplot(fig)


# From the scatter plot we observe that size of the area and number of zones are not exactly correlated. However the graph above is too zoomed in. Perhaps, if we categorize age into four age groups (child, youth, adult, and senior) and look into the difference in area per age group and gender, we may observe something interesting. Let's do that!

# In[ ]:


def age_categories(age):
    # take age as input and return age category
    if age <= 14:
        return 'Child'
    if age >=15 and age <= 24:
        return 'Youth'
    if age >=25 and age <=64:
        return 'Adult'
    if age >= 65:
        return 'Senior'

# get age categories
pneumothorax_df['age_category'] = pneumothorax_df['patient_age'].apply(age_categories)


# In[ ]:


# here we loop over the different age categories and M and F genders to create a subplot
data = []
fig = tools.make_subplots(rows=2, cols=2, subplot_titles=('Child','Youth','Adult','Senior'))
subplot_positions = [(1,1),(1,2),(2,1),(2,2)]

# loop over each age category
for i, cat in enumerate(['Child','Youth','Adult','Senior']):
    # and gender
    for gender in ['M','F']:
        # get affected area for given age group and gender
        values = pneumothorax_df[(pneumothorax_df['patient_sex']==gender) 
                        & (pneumothorax_df['age_category']==cat)]['pneumothorax_area'].values
        # add to the respective trace
        trace = Box(x=values, name=gender)
        # add to figure
        fig.append_trace(trace, subplot_positions[i][0], subplot_positions[i][1])
    


# In[ ]:


fig['layout'].update(height=600, width=900, title='Pneumothorax Size in Different Age Categories', showlegend=False)
iplot(fig)


# This is more like it. The following things we observe is interesting:
# 
# 
# * For children, the affected area is typically very small.
# * Affected area for Youth seems to be the largest! Median for youth female is 110 sqmm, and for male is 242 sqmm.
# * In terms of size, for adults and youth we have a lot of outliers, and specially in adults.
# * Gap between median and q3 is bigger in male over females, for all age groups, except for seniors.
# * However, for male seniors and female seniors almost have the same median. 
# 
# So in summary, males typically developes gas built up over a larger area than female. But that is understandable, as men typically tend to have larger lungs than women.

# ## Section 5: UNet
# 
# In this section we will create a Convolutional Neural Network to try to predict Pneumothorax. We will use a well known architecture for bio-medical image segmentation called UNet. The original authors of UNet published the webpage [U-Net: Convolutional Networks for Biomedical Image Segmentation
# ](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/), which is a good read in my opinion. You will notice that we resize the image from 1024x1024 to 256x256. It's because the image size otherwise is too big for us to process in a decent batch size.

# In[ ]:


# defining configuration parameters
img_size = 512 # image resize size
batch_size = 16 # batch size for training unet
k_size = 3 # kernel size 3x3
val_size = .25 # split of training set between train and validation set
no_pneumo_drop = 0 # dropping some data to balance the class a little bit better


# In[ ]:


import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, UpSampling2D, Concatenate
from sklearn.model_selection import train_test_split
import cv2


# #### Data Generator
# 
# To push the data to our model, we will create a custom data generator. A generator lets us load data progressively, instead of loading it all into memory at once. A custom generator allows us to also fit in more customization during the time of loading the data. As the model is being procssed in the GPU, we can use a custom generator to pre-process images via a generator. At this time, we can also take advantage multiple processors to parallelize our pre-processing.

# In[ ]:


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, file_path_list, labels, batch_size=32, 
                 img_size=256, channels=1, shuffle=True):
        self.file_path_list = file_path_list
        self.labels = labels
        self.batch_size = batch_size
        self.img_size = img_size
        self.channels = channels
        self.shuffle = shuffle
        self.on_epoch_end()
    
    def __len__(self):
        'denotes the number of batches per epoch'
        return int(np.floor(len(self.file_path_list)) / self.batch_size)
    
    def __getitem__(self, index):
        'generate one batch of data'
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # get list of IDs
        file_path_list_temp = [self.file_path_list[k] for k in indexes]
        # generate data
        X, y = self.__data_generation(file_path_list_temp)
        # return data 
        return X, y
    
    def on_epoch_end(self):
        'update ended after each epoch'
        self.indexes = np.arange(len(self.file_path_list))
        if self.shuffle:
            np.random.shuffle(self.indexes)
            
    def __data_generation(self, file_path_list_temp):
        'generate data containing batch_size samples'
        X = np.empty((self.batch_size, self.img_size, self.img_size, self.channels))
        y = np.empty((self.batch_size, self.img_size, self.img_size, self.channels))
        
        for idx, file_path in enumerate(file_path_list_temp):
            
            id = file_path.split('/')[-1][:-4]
            rle = self.labels.get(id)
            image = pydicom.read_file(file_path).pixel_array
            image_resized = cv2.resize(image, (self.img_size, self.img_size))
            image_resized = np.array(image_resized, dtype=np.float64)
            
            X[idx,] = np.expand_dims(image_resized, axis=2)
            
            # if there is no mask create empty mask
            # notice we are starting of with 1024 because we need to use the rle2mask function
            if rle is None:
                mask = np.zeros((1024, 1024))
            else:
                if len(rle) == 1:
                    mask = rle2mask(rle[0], 1024, 1024).T
                else: 
                    mask = np.zeros((1024, 1024))
                    for r in rle:
                        mask =  mask + rle2mask(r, 1024, 1024).T
                        
            mask_resized = cv2.resize(mask, (self.img_size, self.img_size))
            y[idx,] = np.expand_dims(mask_resized, axis=2)
            
        # normalize 
        X = X / 255
        y = y / 255
            
        return X, y


# In[ ]:


masks = {}
for index, row in train_metadata_df[train_metadata_df['has_pneumothorax']==1].iterrows():
    masks[row['id']] = list(row['encoded_pixels_list'])


# In[ ]:


bad_data = train_metadata_df[train_metadata_df['encoded_pixels_count']==0].index
new_train_metadata_df = train_metadata_df.drop(bad_data)


# In[ ]:


drop_data = new_train_metadata_df[new_train_metadata_df['has_pneumothorax'] == False].sample(no_pneumo_drop).index
new_train_metadata_df = new_train_metadata_df.drop(drop_data)


# In[ ]:


# split the training data into train and validation set (stratified)
X_train, X_val, y_train, y_val = train_test_split(new_train_metadata_df.index, new_train_metadata_df['has_pneumothorax'].values, test_size=val_size, random_state=42)
X_train, X_val = new_train_metadata_df.loc[X_train]['file_path'].values, new_train_metadata_df.loc[X_val]['file_path'].values


# In[ ]:


params = {'img_size': img_size,
          'batch_size': batch_size,
          'channels': 1,
          'shuffle': True}

# Generators
training_generator = DataGenerator(X_train, masks, **params)
validation_generator = DataGenerator(X_val, masks, **params)


# We can verify that our generator class is working and is passing the right data visually in the following way.

# In[ ]:


x, y = training_generator.__getitem__(2)
print(x.shape, y.shape)


# In[ ]:


fig = plt.figure()
fig.subplots_adjust(hspace=0.4, wspace=0.4)
ax = fig.add_subplot(1, 2, 1)
ax.imshow(x[0].reshape(img_size, img_size), cmap=plt.cm.bone)
ax = fig.add_subplot(1, 2, 2)
ax.imshow(np.reshape(y[0], (img_size, img_size)), cmap="gray")


# #### Net TensorFlow Keras Implementation
# 
# Lets build the UNet model in this section. Actually, I found a nice implementation of UNet on [Github](https://github.com/nikhilroxtomar/UNet-Segmentation-in-Keras-TensorFlow), which I am using for this section of the Kernel.

# In[ ]:


def down_block(x, filters, kernel_size=3, padding='same', strides=1, activation='relu'):
    'down sampling block of our UNet'
    conv = Conv2D(filters, kernel_size, padding=padding, strides=strides, activation=activation)(x)
    conv = Conv2D(filters, kernel_size, padding=padding, strides=strides, activation=activation)(conv)
    pool = MaxPool2D((2,2), (2,2))(conv)
    return conv, pool


# In[ ]:


def up_block(x, skip, filters, kernel_size=3, padding='same', strides=1, activation='relu'):
    'up sampling block of our UNet'
    up_sample = UpSampling2D((2,2))(x)
    concat = Concatenate()([up_sample, skip])
    conv = Conv2D(filters, kernel_size, padding=padding, strides=strides, activation=activation)(concat)
    conv = Conv2D(filters, kernel_size, padding=padding, strides=strides, activation=activation)(conv)
    return conv


# In[ ]:


def bottleneck(x, filters, kernel_size=3, padding='same', strides=1, activation='relu'):
    'bottle neck that sits inbetween the down sampling side and the up sampling side'
    conv = Conv2D(filters, kernel_size, padding=padding, strides=strides, activation=activation)(x)
    conv = Conv2D(filters, kernel_size, padding=padding, strides=strides, activation=activation)(conv)
    return conv


# In[ ]:


def UNet(img_size):
    'constructing UNet using the blocks defined above'
    
    # number of filters per block
    f = [32,64,128,256,512]
    inputs = Input((img_size, img_size, 1))
    
    p0 = inputs
    c1, p1 = down_block(p0, f[0])
    c2, p2 = down_block(p1, f[1])
    c3, p3 = down_block(p2, f[2])
    c4, p4 = down_block(p3, f[3])
    
    bn = bottleneck(p4, f[4])
    
    u1 = up_block(bn, c4, f[3])
    u2 = up_block(u1, c3, f[2])
    u3 = up_block(u2, c2, f[1])
    u4 = up_block(u3, c1, f[0])
    
    outputs = Conv2D(1, (1,1), padding='same', activation='sigmoid')(u4)
    model = Model(inputs, outputs)
    return model


# In[ ]:


# defining the loss function and metrics

smooth = 1.

def dice_coef(y_true, y_pred):
    y_true_f = tf.keras.layers.Flatten()(y_true)
    y_pred_f = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)


# In[ ]:


model = UNet(img_size)
model.compile(optimizer='adam', loss=dice_coef_loss, metrics=[dice_coef])
# model.summary() # enable to see the summary of the model we built


# In[ ]:


model.fit_generator(generator=training_generator, validation_data=validation_generator, epochs=50, verbose=2)


# #### Checking out our model
# 
# We can visually inspect how our model is doing for our model in the following way.

# In[ ]:


def plot_train(img, mask, pred):
    
    fig, ax = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(15,5))
    
    ax[0].imshow(img, cmap=plt.cm.bone)
    ax[0].set_title('Chest X-Ray')
    
    ax[1].imshow(mask, cmap=plt.cm.bone)
    ax[1].set_title('Mask')
    
    ax[2].imshow(pred, cmap=plt.cm.bone)
    ax[2].set_title('Pred Mask')
    
    plt.show()


# In[ ]:


# lets loop over the predictions and print some good-ish results
count = 0
for i in range(0,30):
    if count <= 15:
        x, y = validation_generator.__getitem__(i)
        predictions = model.predict(x)
        for idx, val in enumerate(x):
            if y[idx].sum() > 0 and count <= 15: 
                img = np.reshape(x[idx]* 255, (img_size, img_size))
                mask = np.reshape(y[idx]* 255, (img_size, img_size))
                pred = np.reshape(predictions[idx], (img_size, img_size))
                pred = pred > 0.5
                pred = pred * 255
                plot_train(img, mask, pred)
                count += 1


# #### Making Predictions
# 
# In this section, we predict using our model and create a submission without taking advantage of the leak.

# In[ ]:


def get_test_tensor(file_path, batch_size, img_size, channels):
    
        X = np.empty((batch_size, img_size, img_size, channels))

        # Store sample
        pixel_array = pydicom.read_file(file_path).pixel_array
        image_resized = cv2.resize(pixel_array, (img_size, img_size))
        image_resized = np.array(image_resized, dtype=np.float64)
        image_resized -= image_resized.mean()
        image_resized /= image_resized.std()
        X[0,] = np.expand_dims(image_resized, axis=2)

        return X


# In[ ]:


submission = []

for i, row in test_metadata_df.iterrows():

    test_img = get_test_tensor(test_metadata_df['file_path'][i],1,img_size,1)
    
    pred_mask = model.predict(test_img).reshape((img_size,img_size))
    prediction = {}
    prediction['ImageId'] = str(test_metadata_df['id'][i])
    pred_mask = (pred_mask > .5).astype(int)
    
    
    if pred_mask.sum() < 1:
        prediction['EncodedPixels'] =  -1
    else:
        prediction['EncodedPixels'] = mask2rle(pred_mask * 255, img_size, img_size)
    submission.append(prediction)


# In[ ]:


submission_df = pd.DataFrame(submission)
submission_df = submission_df[['ImageId','EncodedPixels']]
submission_df.head()


# In[ ]:


submission_df.to_csv('./submission.csv', index=False)


# Our network is not working at all! Either the network architecture is too vanilla, or the data is too imbalanced, or we need to define a better loss function. However, the code is working and serves as a good boiler plate for someone starting out. So I will leave this as it is. In the other Kernel, [Finding Pneumo Part 2](https://www.kaggle.com/ekhtiar/finding-pneumo-part-2-resunet), we see that adding residual elements to UNet does provide results.
