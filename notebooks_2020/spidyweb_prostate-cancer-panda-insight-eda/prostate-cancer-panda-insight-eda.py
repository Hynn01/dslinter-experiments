#!/usr/bin/env python
# coding: utf-8

# # About this Notebook
# 
# Getting the domain knowledge is the first thing we should do for any Kaggle Competition. Being fully aware of the problem statement will help you to build a better model.
# 
# Here, in this notebook, I will be sharing complete insight of Prostate Cancer, it's detection and using this knowledge we will be doing EDA and will build a model.  
# 
# **This kernel will be a work in Progress,and I will keep on updating it as the competition progresses and I learn more and more things about the data**
# 
# **<span style="color:Red">If you find this kernel useful, Please consider Upvoting it , it motivates me to write more Quality content**

# # Domain Knowledge
# let's start with some basic questions that should come to everyones mind before getting your hands on dataset given for this Competition. 
# 
# ### Q(1) What is Prostate?
# The prostate is a part of the male reproductive system. The prostate is about the size of a walnut and weighs about one ounce.
# ### Q(2) Where is it located?
# The prostate is below the bladder and in front of the rectum. The prostate surrounds the urethra. The urethra is a tube that carries urine from the bladder out through the penis.
# ### Q(3) What is the job of Prostate?
# The main job of the prostate and seminal vesicles is to make fluid to bathe semen. During ejaculation, sperm is made in the testicles, and then moves to the urethra. At the same time, fluid from the prostate and the seminal vesicles also moves into the urethra. This mixture of semen and fluid from the prostate and seminal vesicles forms the ejaculate that passes through the urethra and out of the penis.
# ### Q(4) What is Prostate Cancer?
# Prostate cancer is a form of cancer that develops in the prostate gland. It is the second-leading cause of cancer deaths for men in the U.S. About 1 in 9 men will be diagnosed with prostate cancer in their lifetime. When prostate cancer occurs, it starts in the prostate gland and occasionally spreads to the seminal vesicles.
# 
# Prostate cancer cells can spread by breaking away from a prostate tumor. They can travel through blood vessels or lymph nodes to reach other parts of the body. After spreading, cancer cells may attach to other tissues and grow to form new tumors, causing damage where they land.
# 
# When prostate cancer spreads from its original place to another part of the body, the new tumor has the same kind of abnormal cells and the same name as the primary (original) tumor. For example, if prostate cancer spreads to the bones, the cancer cells in the bones are actually prostate cancer cells. The disease is metastatic prostate cancer, not bone cancer. For that reason, it's treated as prostate cancer in bone.
# 
# <img src="https://www.cancer.org/cancer/prostate-cancer/about/what-is-prostate-cancer/_jcr_content/par/textimage/image.img.jpg/1568058472882.jpg" height="100px">
# 
# ### Q(5) Causes?
# No one knows why or how prostate cancer starts. Autopsy studies show 1 in 3 men over the age of 50 have some cancer cells in the prostate. Eight out of ten "autopsy cancers" found are small, with tumors that are not harmful.
# 
# Even though there is no known reason for prostate cancer, there are many risks associated with the disease like Age,Ethnicity, Family History, Smoking, Diet etc.
# 
# 

# So, till now we are we gained the knowledge about this disease. Now let's explore the Diagnosis of this disease and come to our problem statetemt of how we can use Gleason Score.
# 
# 
# ### Q(6) How is it Diagnosed ?
# Screening : "Screening" means testing for a disease even if you have no symptoms. The two main types of screenings are:
# 
# * PSA Blood Test : The prostate-specific antigen (PSA) blood test is one way to screen for prostate cancer. This blood test measures the level of PSA in the blood. PSA is a protein made only by the prostate and prostate cancers. Very little PSA is found in the blood of a man with a healthy prostate. A low PSA is a sign of prostate health. A rapid rise in PSA may be a sign that something is wrong. Prostate cancer is the most serious cause of a high PSA result. Another reason for a high PSA can be benign (non-cancer) enlargement of the prostate. Prostatitis, inflammation of the prostate, can also cause high PSA results. A rise in PSA level does not tell us the type of cancer cells present. The rise tells us that cancer may be present.
# 
# * DRE :The digital rectal examination (DRE)   helps your doctor find prostate problems. For this exam, the healthcare provider puts a lubricated gloved finger into the rectum. The man either bends over or lies curled on his side on a table. During this test, the doctor feels for an abnormal shape or thickness to the prostate. DRE is safe and easy to do. But the DRE by itself cannot detect early cancer. It should be done with a PSA test.
# 
# ### Q(7) What are the benefits and risks of screening?
# 
# The PSA test and DRE are very important tools. They help to find prostate cancer early, before it spreads. When found early, it can be treated early which helps stop or slow the spread of cancer. This is likely to help some men live longer.
# 
# A risk of a PSA test is that it may miss detecting cancer (a "false negative"). Or, the test may be a "false positive," suggesting something is wrong when you are actually healthy. A false positive result may lead to a biopsy that isn't needed. The test might also detect very slow growing cancer that will never cause problems if left untreated.
# 
# 
# 

# Upon seeing any abnormality in the PSA and DRE test results, doctor recommends Prostate biopsy.
# 
# ### Q(8) So, what is Prostat biopsy and how it is done?
# 
# The decision to have a biopsy is based on PSA and DRE results. Your doctor will also consider your family history of prostate cancer, ethnicity, biopsy history and other health factors.
# 
# A biopsy is a type of minor surgery. For a prostate biopsy , tiny pieces of tissue are removed from the prostate and looked at under a microscope. The pathologist is the doctor who will look carefully at the tissue samples to look for cancer cells. **This is the only way to know for sure if you have prostate cancer.**
# 
# It looks scary!!! don't be scared watch this video:
# 
# 
# 
# 

# In[ ]:


from IPython.display import IFrame, YouTubeVideo
YouTubeVideo('NSawVyi8aro',width=600, height=400)


# Now, once the tissue is collected and if cancer cells are found, the pathologist will assign a "Gleason Score" which helps to determine the severity/risk of the disease.
# 
# ## Q(9)What is GLEASON score ?
# 
# The Gleason system, which has been in use for many years, assigns grades based on how much the cancer looks like normal prostate tissue.
# 
# * If the cancer looks a lot like normal prostate tissue, a grade of 1 is assigned.
# * If the cancer looks very abnormal, it is given a grade of 5.
# * Grades 2 through 4 have features in between these extremes.
# * Almost all cancers are grade 3 or higher; grades 1 and 2 are not often used.
# 
# Since prostate cancers often have areas with different grades, a grade is assigned to the 2 areas that make up most of the cancer. These 2 grades are added to yield the Gleason score (also called the Gleason sum).
# 
# The first number assigned is the grade that is most common in the tumor. For example, if the Gleason score is written as 3+4=7, it means most of the tumor is grade 3 and less is grade 4, and they are added for a Gleason score of 7.
# 
# Although most often the Gleason score is based on the 2 areas that make up most of the cancer, there are some exceptions when a biopsy sample has either a lot of high-grade cancer or there are 3 grades including high-grade cancer. In these cases, the way the Gleason score is determined is modified to reflect the aggressive (fast-growing) nature of the cancer.
# In theory, the Gleason score can be between 2 and 10, but scores below 6 are rarely used.
# 
# Based on the Gleason score, prostate cancers are often divided into 3 groups:
# 
# * Cancers with a Gleason score of 6 or less may be called well-differentiated or low-grade.
# * Cancers with a Gleason score of 7 may be called moderately-differentiated or intermediate-grade.
# * Cancers with Gleason scores of 8 to 10 may be called poorly-differentiated or high-grade.
# 
# Grade Groups
# 
# In recent years, doctors have come to realize that the Gleason score might not always be the best way to describe the grade of the cancer, for a couple of reasons:
# 
# Prostate cancer outcomes can be divided into more than just the 3 groups mentioned above. For example, men with a Gleason score 3+4=7 cancer tend to do better than those with a 4+3=7 cancer. And men with a Gleason score 8 cancer tend to do better than those with a Gleason score of 9 or 10.
# The scale of the Gleason score can be misleading for patients. For example, a man with a Gleason score 6 cancer might assume that his cancer is in the middle of the range of grades (which in theory go from 2 to 10), even though grade 6 cancers are actually the lowest grade seen in practice. This assumption might lead a man to think his cancer is more likely to grow and spread quickly than it really is, which might affect his decisions about treatment.
# Because of this, doctors have developed Grade Groups, ranging from 1 (most likely to grow and spread slowly) to 5 (most likely to grow and spread quickly):
# 
# * Grade Group 1 = Gleason 6 (or less)
# * Grade Group 2 = Gleason 3+4=7
# * Grade Group 3 = Gleason 4+3=7
# * Grade Group 4 = Gleason 8
# * Grade Group 5 = Gleason 9-10
# 

# Below is the video I should recommend you to watch for better understanding of GLEASON score:

# In[ ]:


from IPython.display import IFrame, YouTubeVideo
YouTubeVideo('1Q7ERNtLcvk',width=600, height=400)


# ### Q(11) So, What is ISUP grade now?
# According to current guidelines by the International Society of Urological Pathology (ISUP), the Gleason scores are summarized into an ISUP grade on a scale from 1 to 5 according to the following rule:
# 
# * Gleason score 6 = ISUP grade 1 
# * Gleason score 7 (3 + 4) = ISUP grade 2 
# * Gleason score 7 (4 + 3) = ISUP grade 3 
# * Gleason score 8 = ISUP grade 4 
# * Gleason score 9-10 = ISUP grade 5 
# 
# If there is no cancer in the sample, we use the label ISUP grade 0 in this competition. 
# 
# <img src="https://storage.googleapis.com/kaggle-media/competitions/PANDA/Screen%20Shot%202020-04-08%20at%202.03.53%20PM.png" height="100px">
# 
# ### Q(11) How has the Gleason scores been generated in the dataset?
# Each WSI in this challenge contains one, or in some cases two, thin tissue sections cut from a single biopsy sample. Prior to scanning, the tissue is stained with haematoxylin & eosin (H&E). This is a standard way of staining the originally transparent tissue to produce some contrast. The samples are made up of glandular tissue and connective tissue. The glands are hollow structures, which can be seen as white “holes” or branched cavities in the WSI. The appearance of the glands forms the basis of the Gleason grading system. The glandular structure characteristic of healthy prostate tissue is progressively lost with increasing grade. The grading system recognizes three categories: 3, 4, and 5. 
# 
# * [A]Benign prostate glands with folded epithelium :The cytoplasm is pale and the nuclei small and regular. The glands are grouped together.
# * [B]Prostatic adenocarcinoma : Gleason Pattern 3 has no loss of glandular differentiation. Small glands infiltrate between benign glands. The cytoplasm is often dark and the nuclei enlarged with dark chromatin and some prominent nucleoli. Each epithelial unit is separate and has a lumen.
# * [C]Prostatic adenocarcinoma : Gleason Pattern 4 has partial loss of glandular differentiation. There is an attempt to form lumina but the tumor fails to form complete, well-developed glands. This microphotograph shows irregular cribriform cancer, i.e. epithelial sheets with multiple lumina. There are also some poorly formed small glands and some fused glands. All of these are included in Gleason Pattern 4.
# * [D]Prostatic adenocarcinoma : Gleason Pattern 5 has an almost complete loss of glandular differentiation. Dispersed single cancer cells are seen in the stroma. Gleason Pattern 5 may also contain solid sheets or strands of cancer cells. All microphotographs show hematoxylin and eosin stains at 20x lens magnification.
# 
# <img src="https://storage.googleapis.com/kaggle-media/competitions/PANDA/GleasonPattern_4squares%20copy500.png" height="100px">

# # Now, let's deep dive into the Data

# In[ ]:


import numpy as np 
import pandas as pd 
import os
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import PIL
from IPython.display import Image, display
from plotly import graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff
import openslide


# In[ ]:


# Location of the training images
BASE_FOLDER = "/kaggle/input/prostate-cancer-grade-assessment/"
get_ipython().system('ls {BASE_FOLDER}')
# image and mask directories
data_dir = f'{BASE_FOLDER}/train_images'
mask_dir = f'{BASE_FOLDER}/train_label_masks'


# In[ ]:


# Location of training labels
train = pd.read_csv(BASE_FOLDER+"train.csv")
test = pd.read_csv(BASE_FOLDER+"test.csv")
sub = pd.read_csv(BASE_FOLDER+"sample_submission.csv")


# #### Let's see the training data

# In[ ]:


train.head()


# #### Now, let's check the number of images, types of data provider, ISUP Grades and Gleason Scores.

# In[ ]:


print("number of unique images : ", len(train.image_id.unique()))
print("number of unique data provider : ", len(train.data_provider.unique()))
print("number of unique isup_grade : ", len(train.isup_grade.unique()))
print("number of unique gleason_score : ", len(train.gleason_score.unique()))


# In[ ]:


print("Data Providers : ", train['data_provider'].unique())
print("ISUP Grdes : ", train['isup_grade'].unique())
print("Gleason Scores : ", train['gleason_score'].unique())


# #### Let's check the ISUP Grdes of different Gleason Scores  

# In[ ]:


print((train[train['gleason_score']=='negative']['isup_grade'].unique()))
print((train[train['gleason_score']=='0+0']['isup_grade'].unique()))
print((train[train['gleason_score']=='3+3']['isup_grade'].unique()))
print((train[train['gleason_score']=='3+4']['isup_grade'].unique()))
print((train[train['gleason_score']=='4+3']['isup_grade'].unique()))
print((train[train['gleason_score']=='4+4']['isup_grade'].unique()))
print((train[train['gleason_score']=='3+5']['isup_grade'].unique()))
print((train[train['gleason_score']=='5+3']['isup_grade'].unique()))
print((train[train['gleason_score']=='4+5']['isup_grade'].unique()))
print((train[train['gleason_score']=='5+4']['isup_grade'].unique()))
print((train[train['gleason_score']=='5+5']['isup_grade'].unique()))


# Two things we see:
# 
# * Here, we see that "negative" and "0+0" Gleason Scores have some ISUP Grade which is '0'. So, We need to merge these two.
# 
# * Also, we see that Gleason Score "4+3" have two ISUP Grades which is '3' and '2'. Since we know that the correct ISUP Grade for Gleason Score is '3', we need to find the one which is labeled as Gleason Score as '2' and remove that.

# #### Converting negative Gleason Score to "0+0"

# In[ ]:


train['gleason_score'] = train['gleason_score'].apply(lambda x: "0+0" if x=="negative" else x)
print("Gleason Scores : ", train['gleason_score'].unique())


# #### Dropping the image having incorrect ISUP Score from the train set

# In[ ]:


train[(train['isup_grade'] == 2) & (train['gleason_score'] == '4+3')]


# In[ ]:


train.drop([7273],inplace=True)
print("number of unique images : ", len(train.image_id.unique()))


# ### Let's see the test Data

# In[ ]:


test.head()


# In[ ]:


print("number of unique images : ", len(test.image_id.unique()))


# They are only 3 images in the test and they too are hidden and we have access to them only when we are submitting. To know more about test submission folow this discussion:
# 
# https://www.kaggle.com/c/prostate-cancer-grade-assessment/discussion/145219

# ### Now, it's time for presenting our Data in a beautiful way which is often called EDA

# In[ ]:


def plot_count(df, feature, title='', size=2):
    f, ax = plt.subplots(1,1, figsize=(3*size,2*size))
    total = float(len(df))
    sns.countplot(df[feature],order = df[feature].value_counts().index, palette='Set2')
    plt.title(title)
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x()+p.get_width()/2.,
                height + 3,
                '{:1.2f}%'.format(100*height/total),
                ha="center") 
    plt.show()


# In[ ]:


plot_count(train, 'data_provider', 'Data provider - data count and percent')


# Karolinska has provided more images than Radbond.

# In[ ]:


plot_count(train, 'isup_grade','ISUP grade - data count and percent', size=3)


# In Training set, Biopsys are of more ISUP Grade 0 and ISUP Grade 1.

# In[ ]:


fig = plt.figure(figsize=(10,6))
ax = sns.countplot(x="isup_grade", hue="data_provider", data=train)
for p in ax.patches:
   
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2,
                height +3,
                '{:1.2f}%'.format(100*height/10616),
                ha="center")


# * ISUP Grade 0 and 1 are more provided by Karolinska
# * ISUP Grade 3,4,5 are more provided by Radbound
# * Isup 2 in training set has come almost equal from the both data provider

# In[ ]:


plot_count(train, 'gleason_score', 'Gleason score - data count and percent', size=3)


# * Gleason Score 0+0 and 3+3 contiribute more to the Training dataset

# In[ ]:


fig = plt.figure(figsize=(10,6))
ax = sns.countplot(x="gleason_score", hue="data_provider", data=train)
for p in ax.patches:

    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2,
                height +3,
                '{:1.2f}%'.format(100*height/10616),
                ha="center")


# In[ ]:


fig, ax = plt.subplots(nrows=1,figsize=(12,6))
tmp = train.groupby('isup_grade')['gleason_score'].value_counts()
df = pd.DataFrame(data={'Exams': tmp.values}, index=tmp.index).reset_index()
sns.barplot(ax=ax,x = 'isup_grade', y='Exams',hue='gleason_score',data=df, palette='Set1')
plt.title("Number of examinations grouped on ISUP grade and Gleason score")
plt.show()


# In[ ]:


fig, ax = plt.subplots(nrows=1,figsize=(8,8))
heatmap_data = pd.pivot_table(df, values='Exams', index=['isup_grade'], columns='gleason_score')
sns.heatmap(heatmap_data, cmap="nipy_spectral",linewidth=0.5, linecolor='White')
plt.title('Number of examinations grouped on ISUP grade and Gleason score')
plt.show()


# ## Now, let's check the Image 

# #### Displaying few images
# 
# In the following sections we will load data from the slides with OpenSlide. The benefit of OpenSlide is that we can load arbitrary regions of the slide, without loading the whole image in memory. 
# 
# 

# In[ ]:


'''
Example for using Openslide to display an image
'''
# Open the image (does not yet read the image into memory)
example = openslide.OpenSlide(os.path.join(BASE_FOLDER+"train_images", '005e66f06bce9c2e49142536caf2f6ee.tiff'))

# Read a specific region of the image starting at upper left coordinate (x=17800, y=19500) on level 0 and extracting a 256*256 pixel patch.
# At this point image data is read from the file and loaded into memory.
patch = example.read_region((17800,19500), 0, (256, 256))

# Display the image
display(patch)

# Close the opened slide after use
example.close()


# In[ ]:


train = train.set_index('image_id')
train.head()


# #### Displaying an image with it's Zoomed version

# In[ ]:


def get_values(image,max_size=(600,400)):
    slide = openslide.OpenSlide(os.path.join(BASE_FOLDER+"train_images", f'{image}.tiff'))
    
    # Here we compute the "pixel spacing": the physical size of a pixel in the image.
    # OpenSlide gives the resolution in centimeters so we convert this to microns.
    f,ax =  plt.subplots(2 ,figsize=(6,16))
    spacing = 1 / (float(slide.properties['tiff.XResolution']) / 10000)
    patch = slide.read_region((1780,1950), 0, (256, 256)) #ZOOMED FUGURE
    ax[0].imshow(patch) 
    ax[0].set_title('Zoomed Image')
    ax[1].imshow(slide.get_thumbnail(size=max_size)) #UNZOOMED FIGURE
    ax[1].set_title('Full Image')
    
    
    print(f"File id: {slide}")
    print(f"Dimensions: {slide.dimensions}")
    print(f"Microns per pixel / pixel spacing: {spacing:.3f}")
    print(f"Number of levels in the image: {slide.level_count}")
    print(f"Downsample factor per level: {slide.level_downsamples}")
    print(f"Dimensions of levels: {slide.level_dimensions}\n\n")
    print(f"ISUP grade: {train.loc[image, 'isup_grade']}")
    print(f"Gleason score: {train.loc[image, 'gleason_score']}")


# In[ ]:


get_values('07a7ef0ba3bb0d6564a73f4f3e1c2293')


# #### Displaying multiple images with it's attributes like Data Provider, Gleason Score and ISUP Grade

# In[ ]:


def display_images(slides): 
    f, ax = plt.subplots(3,3, figsize=(18,22))
    for i, slide in enumerate(slides):
        image = openslide.OpenSlide(os.path.join(BASE_FOLDER+"train_images", f'{slide}.tiff'))
        spacing = 1 / (float(image.properties['tiff.XResolution']) / 10000)
        patch = image.read_region((1780,1950), 0, (256, 256))
        ax[i//3, i%3].imshow(patch) 
        image.close()       
        ax[i//3, i%3].axis('off')
        
        image_id = slide
        data_provider = train.loc[slide, 'data_provider']
        isup_grade = train.loc[slide, 'isup_grade']
        gleason_score = train.loc[slide, 'gleason_score']
        ax[i//3, i%3].set_title(f"ID: {image_id}\nSource: {data_provider} ISUP: {isup_grade} Gleason: {gleason_score}")

    plt.show() 


# In[ ]:


images = [
    '037504061b9fba71ef6e24c48c6df44d',
    '035b1edd3d1aeeffc77ce5d248a01a53',
    '059cbf902c5e42972587c8d17d49efed',
'06a0cbd8fd6320ef1aa6f19342af2e68',
    '06eda4a6faca84e84a781fee2d5f47e1',
    '0a4b7a7499ed55c71033cefb0765e93d',
'0838c82917cd9af681df249264d2769c',
    '046b35ae95374bfb48cdca8d7c83233f',
    '074c3e01525681a275a42282cd21cbde',]

display_images(images)


# ##### Few Insights:
# 
# * The image dimensions are quite large (typically between 5.000 and 40.000 pixels in both x and y).
# * Each slide has 3 levels you can load, corresponding to a downsampling of 1, 4 and 16. Intermediate levels can be created by downsampling a higher resolution level.
# * The dimensions of each level differ based on the dimensions of the original image.
# * Biopsies can be in different rotations. This rotation has no clinical value, and is only dependent on how the biopsy was collected in the lab.
# * There are noticable color differences between the biopsies, this is very common within pathology and is caused by different laboratory procedures.

# ### Visualizing masks (using matplotlib)
# 
# Apart from the slide-level label (present in the csv file), almost all slides in the training set have an associated mask with additional label information. These masks directly indicate which parts of the tissue are healthy and which are cancerous. The information in the masks differ from the two centers:
# 
# * Radboudumc: Prostate glands are individually labelled. Valid values are: 0: background (non tissue) or unknown
# 1: stroma (connective tissue, non-epithelium tissue)
# 2: healthy (benign) epithelium
# 3: cancerous epithelium (Gleason 3)
# 4: cancerous epithelium (Gleason 4)
# 5: cancerous epithelium (Gleason 5)
# * Karolinska: Regions are labelled. Valid values:
# 0: background (non tissue) or unknown
# 1: benign tissue (stroma and epithelium combined)
# 2: cancerous tissue (stroma and epithelium combined)
# The label masks of Radboudumc were semi-automatically generated by several deep learning algorithms, contain noise, and can be considered as weakly-supervised labels. The label masks of Karolinska were semi-autotomatically generated based on annotations by a pathologist.
# 
# The label masks are stored in an RGB format so that they can be easily opened by image readers. The label information is stored in the red (R) channel, the other channels are set to zero and can be ignored. As with the slides itself, the label masks can be opened using OpenSlide.

# In[ ]:


def display_masks(slides): 
    f, ax = plt.subplots(3,3, figsize=(18,22))
    for i, slide in enumerate(slides):
        
        mask = openslide.OpenSlide(os.path.join(mask_dir, f'{slide}_mask.tiff'))
        mask_data = mask.read_region((0,0), mask.level_count - 1, mask.level_dimensions[-1])
        cmap = matplotlib.colors.ListedColormap(['black', 'gray', 'green', 'yellow', 'orange', 'red'])

        ax[i//3, i%3].imshow(np.asarray(mask_data)[:,:,0], cmap=cmap, interpolation='nearest', vmin=0, vmax=5) 
        mask.close()       
        ax[i//3, i%3].axis('off')
        
        image_id = slide
        data_provider = train.loc[slide, 'data_provider']
        isup_grade = train.loc[slide, 'isup_grade']
        gleason_score = train.loc[slide, 'gleason_score']
        ax[i//3, i%3].set_title(f"ID: {image_id}\nSource: {data_provider} ISUP: {isup_grade} Gleason: {gleason_score}")
        f.tight_layout()
        
        
    plt.show()


# In[ ]:


display_masks(images)


# ### Displaying Images and it's Mask Side by Side 

# In[ ]:


def mask_img(image,max_size=(600,400)):
    slide = openslide.OpenSlide(os.path.join(BASE_FOLDER+"train_images", f'{image}.tiff'))
    mask =  openslide.OpenSlide(os.path.join(mask_dir, f'{image}_mask.tiff'))
    # Here we compute the "pixel spacing": the physical size of a pixel in the image.
    # OpenSlide gives the resolution in centimeters so we convert this to microns.
    f,ax =  plt.subplots(1,2 ,figsize=(18,22))
    spacing = 1 / (float(slide.properties['tiff.XResolution']) / 10000)
    img = slide.get_thumbnail(size=(600,400)) #IMAGE 
    
    mask_data = mask.read_region((0,0), mask.level_count - 1, mask.level_dimensions[-1])
    cmap = matplotlib.colors.ListedColormap(['black', 'gray', 'green', 'yellow', 'orange', 'red'])
    
    ax[0].imshow(img) 
    #ax[0].set_title('Image')
    
    
    ax[1].imshow(np.asarray(mask_data)[:,:,0], cmap=cmap, interpolation='nearest', vmin=0, vmax=5) #IMAGE MASKS
    #ax[1].set_title('Image_MASK')
    
    
    image_id = image
    data_provider = train.loc[image, 'data_provider']
    isup_grade = train.loc[image, 'isup_grade']
    gleason_score = train.loc[image, 'gleason_score']
    ax[0].set_title(f"ID: {image_id}\nSource: {data_provider} ISUP: {isup_grade} Gleason: {gleason_score} IMAGE")
    ax[1].set_title(f"ID: {image_id}\nSource: {data_provider} ISUP: {isup_grade} Gleason: {gleason_score} IMAGE_MASK")


# In[ ]:


images1= [
    '08ab45297bfe652cc0397f4b37719ba1',
    '090a77c517a7a2caa23e443a77a78bc7'
]

for image in images1:
    mask_img(image)


# ### Overlaying masks on the slides
# 
# As the masks have the same dimension as the slides, we can overlay the masks on the tissue to directly see which areas are cancerous. This overlay can help you identifying the different growth patterns. To do this, we load both the mask and the biopsy and merge them using PIL.
# 
# Tip: Want to view the slides in a more interactive way? Using a WSI viewer you can interactively view the slides. Examples of open source viewers that can open the PANDA dataset are ASAP and QuPath. ASAP can also overlay the masks on top of the images using the "Overlay" functionality. If you use Qupath, and the images do not load, try changing the file extension to .vtif.

# In[ ]:


def overlay_mask_on_slide(images, center='radboud', alpha=0.8, max_size=(800, 800)):
    """Show a mask overlayed on a slide."""
    f, ax = plt.subplots(3,3, figsize=(18,22))
    
    
    for i, image_id in enumerate(images):
        slide = openslide.OpenSlide(os.path.join(data_dir, f'{image_id}.tiff'))
        mask = openslide.OpenSlide(os.path.join(mask_dir, f'{image_id}_mask.tiff'))
        slide_data = slide.read_region((0,0), slide.level_count - 1, slide.level_dimensions[-1])
        mask_data = mask.read_region((0,0), mask.level_count - 1, mask.level_dimensions[-1])
        mask_data = mask_data.split()[0]
        
        
        # Create alpha mask
        alpha_int = int(round(255*alpha))
        if center == 'radboud':
            alpha_content = np.less(mask_data.split()[0], 2).astype('uint8') * alpha_int + (255 - alpha_int)
        elif center == 'karolinska':
            alpha_content = np.less(mask_data.split()[0], 1).astype('uint8') * alpha_int + (255 - alpha_int)

        alpha_content = PIL.Image.fromarray(alpha_content)
        preview_palette = np.zeros(shape=768, dtype=int)

        if center == 'radboud':
            # Mapping: {0: background, 1: stroma, 2: benign epithelium, 3: Gleason 3, 4: Gleason 4, 5: Gleason 5}
            preview_palette[0:18] = (np.array([0, 0, 0, 0.5, 0.5, 0.5, 0, 1, 0, 1, 1, 0.7, 1, 0.5, 0, 1, 0, 0]) * 255).astype(int)
        elif center == 'karolinska':
            # Mapping: {0: background, 1: benign, 2: cancer}
            preview_palette[0:9] = (np.array([0, 0, 0, 0, 1, 0, 1, 0, 0]) * 255).astype(int)

        mask_data.putpalette(data=preview_palette.tolist())
        mask_rgb = mask_data.convert(mode='RGB')
        overlayed_image = PIL.Image.composite(image1=slide_data, image2=mask_rgb, mask=alpha_content)
        overlayed_image.thumbnail(size=max_size, resample=0)

        
        ax[i//3, i%3].imshow(overlayed_image) 
        slide.close()
        mask.close()       
        ax[i//3, i%3].axis('off')
        
        data_provider = train.loc[image_id, 'data_provider']
        isup_grade = train.loc[image_id, 'isup_grade']
        gleason_score = train.loc[image_id, 'gleason_score']
        ax[i//3, i%3].set_title(f"ID: {image_id}\nSource: {data_provider} ISUP: {isup_grade} Gleason: {gleason_score}")


# In[ ]:


overlay_mask_on_slide(images)


# Note: In the example above you can also observe a few pen markings on the slide (dark green smudges). These markings are not part of the tissue but were made by the pathologists who originally checked this case. These pen markings are available on some slides in the training set.

# ### Exploring images with pen markers
# 
# It is mentioned that in training dataset, there are few images with pen markers on them. The organizers left us with a Note as described below.
# 
# Note that slightly different procedures were in place for the images used in the test set than the training set. Some of the training set images have stray pen marks on them, but the test set slides are free of pen marks.
# 
# Let's take a look on few of these images.

# In[ ]:


pen_marked_images = [
    'fd6fe1a3985b17d067f2cb4d5bc1e6e1',
    'ebb6a080d72e09f6481721ef9f88c472',
    'ebb6d5ca45942536f78beb451ee43cc4',
    'ea9d52d65500acc9b9d89eb6b82cdcdf',
    'e726a8eac36c3d91c3c4f9edba8ba713',
    'e90abe191f61b6fed6d6781c8305fe4b',
    'fd0bb45eba479a7f7d953f41d574bf9f',
    'ff10f937c3d52eff6ad4dd733f2bc3ac',
    'feee2e895355a921f2b75b54debad328',
]

overlay_mask_on_slide(pen_marked_images)


# Finally, I would like to thank the authors of following kernels. Learnt a lot from: 
# * https://www.kaggle.com/wouterbulten/getting-started-with-the-panda-dataset
# * https://www.kaggle.com/rohitsingh9990/panda-eda-better-visualization-simple-baseline
# * https://www.kaggle.com/tanulsingh077/prostate-cancer-in-depth-understanding-eda-model
# * https://www.kaggle.com/gpreda/panda-challenge-starting-eda
# 
# **This kernel will be a work in Progress,and I will keep on updating it as the competition progresses and I learn more and more things about the data**
# 
# 

# #### Future Work:
# 
# Building a Baseline model
# 
# **<span style="color:Red">If you find this kernel useful, Please consider Upvoting it , it motivates me to write more Quality content**
#     
#   
#  
#  **<span style="color:Red">Thank you, Happy Kaggling!!**
#     
# 
# 
