#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# This Kernel objective is to explore the dataset for Prostate cANcer graDe Assessment (PANDA) Challenge.  
# 
# Prostate cancer is the second most common cancer among males worldwide that results in more than 350k deaths annually. The key to decreasing mortality is developing more precise screeening procedures and diagnostics. Diagnosis of PCa is based on the grading of prostate tissue biopsies. These tissue samples are examined by a pathologist and scored according to the **Gleason grading system**. In the next Figure we show the principle of Gleason grading system.
# 
# ![](https://murtagh.mhmedical.com/data/books/2471/murtagh7e_c114_f04.png)
# 
# The grading process consists of finding and classifying cancer tissue into so-called Gleason patterns (3, 4, or 5) based on the architectural growth patterns of the tumor (see Figure below). Based on presence of various formations, the Gleason score is given for majority (first digit in the score) and minority Gleason score (the second digit). After the biopsy is assigned a Gleason score (a combination of the two digits), it is converted into an ISUP grade on a 1-5 scale, using the correspondence matrix shown in the next Figure. 
# 
# ![](https://storage.googleapis.com/kaggle-media/competitions/PANDA/Screen%20Shot%202020-04-08%20at%202.03.53%20PM.png)
# 

# # Prepare the Data Analysis
# 
# ## Load Packages

# In[ ]:


import pandas as pd 
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook
from matplotlib.patches import Rectangle
import seaborn as sns
import openslide
get_ipython().run_line_magic('matplotlib', 'inline')
PATH = "/kaggle/input/prostate-cancer-grade-assessment/"


# ## Load the Data
# 
# Let's load the tabular data. There are three files:
# 
# * Sample submission;
# * Train;
# * Test.
# 

# In[ ]:


sample_submission_df = pd.read_csv(os.path.join(PATH,'sample_submission.csv'))
train_df = pd.read_csv(os.path.join(PATH,'train.csv'))
test_df = pd.read_csv(os.path.join(PATH,'test.csv'))


# In[ ]:


print(f"sample submission shape: {sample_submission_df.shape}")
print(f"train shape: {train_df.shape}")
print(f"test shape: {test_df.shape}")


# In[ ]:


sample_submission_df.head()


# In[ ]:


train_df.head()


# In[ ]:


test_df.head()


# Test and sample submission csv files have only 3 rows (samples).
# 
# In train data there is a column (gleason_score) that is not present either in test (with image_id & data_provider) or in sample submission (image_id & isup_grade) csv.
# 
# The objective is to predict isup_grade for each image in test set.
# 
# 
# Let's check now the train data in train_images and train_label masks folders.

# In[ ]:


train_image_list = os.listdir(os.path.join(PATH, 'train_images'))
train_label_masks_list = os.listdir(os.path.join(PATH, 'train_label_masks'))


# In[ ]:


print(f"train image_id list: {train_df.image_id.nunique()}")
print(f"train image list: {len(train_image_list)}")
print(f"train label masks list: {len(train_label_masks_list)}")


# We can observe that there are totally 10616 train images, with same number 10616 corresponding images and only 10516 mask images.  
# 
# 
# Consequently there are 100 images with missing masks.

# In[ ]:


print(f"sample of image_id list: {train_df.image_id.values[0:3]}")
print(f"sample of image list: {train_image_list[0:3]}")
print(f"sample of label masks list: {train_label_masks_list[0:3]}")


# ## Missing Data
# 
# Let's check for missing values in these lists. 
# 
# For this, we will remove the suffix and extension from label masks and will remove the extension from image list.

# In[ ]:


trimmed_image_list = []
for img in train_image_list:
    trimmed_image_list.append(img.split('.tiff')[0])


# In[ ]:


trimmed_label_masks_list = []
for img in train_label_masks_list:
    trimmed_label_masks_list.append(img.split('_mask.tiff')[0])


# We compare now the intersections of sets of resulted lists.

# In[ ]:


intersect_i_m = (set(trimmed_image_list) & set(trimmed_label_masks_list))
intersect_id_m = (set(train_df.image_id.unique()) & set(trimmed_label_masks_list))
intersect_id_i = (set(train_df.image_id.unique()) & set(trimmed_image_list))

print(f"image (tiff) & label masks: {len(intersect_i_m)}")
print(f"image_id (train) & label masks: {len(intersect_id_m)}")
print(f"image_id (train) & image (tiff): {len(intersect_id_i)}")


# Both intersections with masks set cardinality is equal with the mask set data dimmensions. And intersection between train image ids and image equals train cardinality.   
# 
# That means that, besides the missing 100 mask images, is no missing data: all images indexed in train data has corresponding tiff images and only 100 tiff label masks are missing. 
# 
# 
# Let's see what images have missing masks.

# In[ ]:


missing_masks  = np.setdiff1d(trimmed_image_list,trimmed_label_masks_list)
print(f'missing masks: {len(missing_masks)} images (press output button to see the list)')


# In[ ]:


print(list(missing_masks))


# # Data Exploration
# 
# 
# Let's check distribution of data_provider, isup_grade and gleason_score in train data.

# In[ ]:


def plot_count(df, feature, title='', size=2):
    f, ax = plt.subplots(1,1, figsize=(3*size,2*size))
    total = float(len(df))
    sns.countplot(df[feature],order = df[feature].value_counts().index, palette='Set3')
    plt.title(title)
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x()+p.get_width()/2.,
                height + 3,
                '{:1.2f}%'.format(100*height/total),
                ha="center") 
    plt.show()


# In[ ]:


plot_count(train_df, 'data_provider', 'Data provider - data count and percent')


# Two research sources are present: Karolinska Institute and Radboud University.

# In[ ]:


plot_count(train_df, 'isup_grade','ISUP grade - data count and percent', size=3)


# In 2012, the International Society of Urologic Pathologists (ISUP) proposed a novel, validated grading system for clear cell renal cell carcinoma (ccRCC) and papillary renal cell carcinoma (pRCC) that has been implemented by the World Health Organization (WHO).This system is based primarily on the nucleoli assessment of the tumors, as follows [1]:
# 
# * Grade 1: Inconspicuous nucleoli at ×400 magnification and basophilic
# * Grade 2: Clearly visible nucleoli at ×400 magnification and eosinophilic
# * Grade 3: Clearly visible nucleoli at ×100 magnification
# * Grade 4: Extreme pleomorphism or rhabdoid and/or sarcomatoid morphology  
# 
# 
# Majority of data samples in train set have ISUP grade values 0 or 1 (total > 50%) and the rest of the data samples have associated ISUP grades from 2 to 5 with all ranging in the 11-12% each. 

# In[ ]:


plot_count(train_df, 'gleason_score', 'Gleason score - data count and percent', size=3)


# The data is unbalanced with respect of Gleason score values distribution. 
# 
# Let's check now relative distribution of ISUP grade anf Gleason score values.

# In[ ]:


fig, ax = plt.subplots(nrows=1,figsize=(12,6))
tmp = train_df.groupby('isup_grade')['gleason_score'].value_counts()
df = pd.DataFrame(data={'Exams': tmp.values}, index=tmp.index).reset_index()
sns.barplot(ax=ax,x = 'isup_grade', y='Exams',hue='gleason_score',data=df, palette='Set1')
plt.title("Number of examinations grouped on ISUP grade and Gleason score")
plt.show()


# 
# Discussion:
# * All exams with ISUP grade = 0 have Gleason score 0+0 or negative.
# * All exams with ISUP grade = 1 have Gleason score 3+3.
# * All exams with ISUP grade = 2 have Gleason score 3+4.
# * All exams with ISUP grade = 3 have Gleason score 4+3.
# * All exams with ISUP grade = 4 have Gleason score 4+4 (majority), 3+5 or 5+3.
# * All exams with ISUP grade = 5 have Gleason score 4+5 (majority), 5+4 or 5+5.
# 
# This data distribution can be better visualized using a heatmap.
# 

# In[ ]:


fig, ax = plt.subplots(nrows=1,figsize=(8,8))
heatmap_data = pd.pivot_table(df, values='Exams', index=['isup_grade'], columns='gleason_score')
sns.heatmap(heatmap_data, cmap="YlGnBu",linewidth=0.5, linecolor='blue')
plt.title('Number of examinations grouped on ISUP grade and Gleason score')
plt.show()


# This seems to be aligned with the correspondence matrix as shown in the following table:

# In[ ]:


from IPython.display import HTML, display

data = [["Gleason Score", "ISUP Grade"],
        ["0+0", "0"], ["negative", "0"],
        ["3+3", "1"], ["3+4", "2"], ["4+3", "3"], 
        ["4+4", "4"], ["3+5", "4"], ["5+3", "4"],
        ["4+5", "5"], ["5+4", "5"], ["5+5", "5"],
        ]

display(HTML(
   '<table><tr>{}</tr></table>'.format(
       '</tr><tr>'.join(
           '<td>{}</td>'.format('</td><td>'.join(str(_) for _ in row)) for row in data)
       )
))


# The only misalignment is in the fact that for ISUP Grade 2, in the data we also have Gleason score 4+3 (which also appears for ISUP Grade 3).  
# 
# Let's see how Gleason score is grouped by Data source.

# In[ ]:


fig, ax = plt.subplots(nrows=1,figsize=(12,6)) 
tmp = train_df.groupby('data_provider')['gleason_score'].value_counts() 
df = pd.DataFrame(data={'Exams': tmp.values}, index=tmp.index).reset_index() 
sns.barplot(ax=ax,x = 'data_provider', y='Exams',hue='gleason_score',data=df, palette='Set1') 
plt.title("Number of examinations grouped on Data provider and Gleason score") 
plt.show()


# We can observe that all of the 0+0 Gleason score data samples are from Karolinska while from Radboud we have most of negative data.
# 
# For Karolinska, next (in terms of frequency) are samples with Gleason score 3+3, 3+4, 4+4.
# 
# For Radboud, next (in terms of frequency) most frequent are samples with Gleason score 4+3, 3+3, 3+4, 4+4, 4+5.
# 
# 
# Let's see how ISUP grade is distributed with respect of Data provider.

# In[ ]:


fig, ax = plt.subplots(nrows=1,figsize=(12,6)) 
tmp = train_df.groupby('data_provider')['isup_grade'].value_counts() 
df = pd.DataFrame(data={'Exams': tmp.values}, index=tmp.index).reset_index() 
sns.barplot(ax=ax,x = 'data_provider', y='Exams',hue='isup_grade',data=df, palette='Set1') 
plt.title("Number of examinations grouped on Data provider and ISUP Grade") 
plt.show()


# Most of the 0 & 1 ISUP Grade samples are originated from Karolinska.  
# 
# Most of the 2-5 ISUP Grade samples are originated from Radboud.

# ## Image samples
# 
# 
# Let's plot some image samples from train_images.  
# 
# 
# For this visualization, I reused the code from this Kernel: https://www.kaggle.com/rohitsingh9990/panda-eda-better-visualization

# In[ ]:


def show_images(df, read_region=(1780,1950)):
    data = df
    f, ax = plt.subplots(3,3, figsize=(16,18))
    for i,data_row in enumerate(data.iterrows()):
        image = str(data_row[1][0])+'.tiff'
        image_path = os.path.join(PATH,"train_images",image)
        image = openslide.OpenSlide(image_path)
        spacing = 1 / (float(image.properties['tiff.XResolution']) / 10000)
        patch = image.read_region(read_region, 0, (256, 256))
        ax[i//3, i%3].imshow(patch) 
        image.close()       
        ax[i//3, i%3].axis('off')
        ax[i//3, i%3].set_title(f'ID: {data_row[1][0]}\nSource: {data_row[1][1]} ISUP: {data_row[1][2]} Gleason: {data_row[1][3]}')

    plt.show()


# In[ ]:


images = [
    '059cbf902c5e42972587c8d17d49efed', '06a0cbd8fd6320ef1aa6f19342af2e68', '06eda4a6faca84e84a781fee2d5f47e1',
    '037504061b9fba71ef6e24c48c6df44d', '035b1edd3d1aeeffc77ce5d248a01a53', '046b35ae95374bfb48cdca8d7c83233f',
    '074c3e01525681a275a42282cd21cbde', '05abe25c883d508ecc15b6e857e59f32', '05f4e9415af9fdabc19109c980daf5ad']   
data_sample = train_df.loc[train_df.image_id.isin(images)]
show_images(data_sample)


# ## Masks samples
# 
# Let's show now the masks for the same images.

# In[ ]:


def display_masks(df, read_region=(0,0)):
    data = df
    f, ax = plt.subplots(3,3, figsize=(16,18))
    for i,data_row in enumerate(data.iterrows()):
        image = str(data_row[1][0])+'_mask.tiff'
        image_path = os.path.join(PATH,"train_label_masks",image)
        mask = openslide.OpenSlide(image_path)
        
        mask_data = mask.read_region(read_region, mask.level_count - 1, mask.level_dimensions[-1])
        cmap = matplotlib.colors.ListedColormap(['black', 'lightgray', 'darkgreen', 'yellow', 'orange', 'red'])
        ax[i//3, i%3].imshow(np.asarray(mask_data)[:,:,0], cmap=cmap, interpolation='nearest', vmin=0, vmax=5) 
        mask.close()       
        ax[i//3, i%3].axis('off')
        ax[i//3, i%3].axis('off')
        ax[i//3, i%3].set_title(f'ID: {data_row[1][0]}\nSource: {data_row[1][1]} ISUP: {data_row[1][2]} Gleason: {data_row[1][3]}')
        
    plt.show()


# In[ ]:


display_masks(data_sample)


# 
# ### Gleason score = 5+5
# 
# Let's look now to few images and associated masks for samples with Gleason score (5+5).

# In[ ]:


sample_images = list(train_df.loc[train_df.gleason_score=="5+5", "image_id"])
print(f"total samples (Gleason score=5+5): {len(sample_images)}")
sample_images = [ '08459aaedfda0679aab403ababbd6ece','0a848ccbbb065ef5ee59dd01710f8531', '0bbbb6734f721f4df4d2ba60ade0ed15', 
                 '0bd231c85b2695e2cf021299e67a6afc',  '0efdb66c93d6b474d93dfe41e40be6ca', '1364c10e1e7f1ad0457f649a44d74888', 
                 '1e644a98460e4f7ea50717720a001efd',  '1fb65315d7ded63d688194863a1b123e', '244d9617bd58fa1db73ab4c1f40d298e']
data_sample = train_df.loc[train_df.image_id.isin(sample_images)]
show_images(data_sample)


# In[ ]:


display_masks(data_sample)


# ### Gleason score = 4+5
# 
# Let's look now to few images and associated masks for samples with Gleason score (4+5).

# In[ ]:


sample_images = list(train_df.loc[train_df.gleason_score=="4+5", "image_id"])
print(f"total samples (Gleason score=4+5): {len(sample_images)}")
sample_images = ['010670e9572e67e5a7e00fb791a343ef', '02f35b793b4fe3032ad6d91f181e391c', '0373ed8a095e0a283da690de360ccc21',
                 '03b3788b5dca6fdae323d0f1a03c03f6', '046b35ae95374bfb48cdca8d7c83233f', '0550b23f29085f41b10d165a46ad4371', 
                 '05abe25c883d508ecc15b6e857e59f32', '05f4e9415af9fdabc19109c980daf5ad',   '07aa24f15ce062d65979b6a8bc7eb3f0']
data_sample = train_df.loc[train_df.image_id.isin(sample_images)]
show_images(data_sample)


# In[ ]:


display_masks(data_sample)


# ### Gleason score = 3+4
# 
# Let's look now to images with Gleason score 3+4.

# In[ ]:


sample_images = list(train_df.loc[train_df.gleason_score=="3+4", "image_id"])
print(f"total samples (Gleason score=3+4): {len(sample_images)}")
sample_images =[
   '022544d1446c2c44f8ca8ff53262dc5b', '061054331a952dd1c9df45c283d756b0', '06bf945aaacb9a67d9f2439d9a7d73ea', 
    '06fb67ffa126811dad6ffb6ecdb8558b', '07697ea97bfbbac071d41a375c3ad036', '083ab9e2c95fb0ea2b999c592fb41653', 
    '08f055372c7b8a7e1df97c6586542ac8', '08f12f69f71b3b4c3c45eafbd710b156', '0a107c91216d62b2543122a46eb26541']
data_sample = train_df.loc[train_df.image_id.isin(sample_images[0:9])]
show_images(data_sample)


# In[ ]:


display_masks(data_sample)


# ### Gleason score = 3+3
# 
# Let's look now to images with Gleason score 3+3.
# 

# In[ ]:


sample_images = list(train_df.loc[train_df.gleason_score=="3+3", "image_id"])
print(f"total samples (Gleason score=3+3): {len(sample_images)}")
sample_images =[
 '004dd32d9cd167d9cc31c13b704498af', '00c46b336b5b06423fcdec1b4d5bee06', '00e6511435645e50673991768a713c66',
 '00ee879798782aca1248baa9132d7307', '0280f8b612771801229e2dde52371141', '03849b0243900d79446bb27849dc0bb2', 
 '0dccbb58add854759951038d5ee736ab', '0dfaba8f37fac34150fb70ca8e425141',  '0e347ad243e4019fad579d3282a730f9']
data_sample = train_df.loc[train_df.image_id.isin(sample_images[0:9])]
show_images(data_sample)


# In[ ]:


display_masks(data_sample)


# 
# ## Extract image characteristics
# 
# Let's parse all images for train data to extract image characteristics.

# In[ ]:


import time
start_time = time.time()
slide_dimensions, spacings, level_counts = [], [], []

for image_id in train_df.image_id:
    image = str(image_id)+'.tiff'
    image_path = os.path.join(PATH,"train_images",image)
    slide = openslide.OpenSlide(image_path)
    spacing = 1 / (float(slide.properties['tiff.XResolution']) / 10000)
    slide_dimensions.append(slide.dimensions)
    spacings.append(spacing)
    level_counts.append(slide.level_count)
    slide.close()
    del slide

train_df['width']  = [i[0] for i in slide_dimensions]
train_df['height'] = [i[1] for i in slide_dimensions]
train_df['spacing'] = spacings
train_df['level_count'] = level_counts

end_time = time.time()
print(f"Total processing time: {round(end_time - start_time,2)} sec.")


# In[ ]:


train_df.head()


# Let's now represent the distribution of image dimmensions (width, height), the spacing and level_count, also related to the other features, namely data_provide, isup_grade and gleason_score.

# In[ ]:


print(f" level count: {train_df.level_count.nunique()}")
print(f" spacing: {train_df.spacing.nunique()}")


# In[ ]:


fig, ax = plt.subplots(nrows=1,figsize=(12,6)) 
tmp = train_df.groupby('data_provider')['spacing'].value_counts() 
df = pd.DataFrame(data={'Exams': tmp.values}, index=tmp.index).reset_index() 
sns.barplot(ax=ax,x = 'data_provider', y='Exams',hue='spacing',data=df, palette='Set1') 
plt.title("Number of examinations grouped on Data provider and Spacing") 
plt.show()


# In[ ]:


fig, ax = plt.subplots(nrows=1,figsize=(12,6)) 
tmp = train_df.groupby('isup_grade')['spacing'].value_counts() 
df = pd.DataFrame(data={'Exams': tmp.values}, index=tmp.index).reset_index() 
sns.barplot(ax=ax,x = 'isup_grade', y='Exams',hue='spacing',data=df, palette='Set1') 
plt.title("Number of examinations grouped on ISUP Grade and Spacing") 
plt.show()


# In[ ]:


fig, ax = plt.subplots(nrows=1,figsize=(12,6)) 
tmp = train_df.groupby('gleason_score')['spacing'].value_counts() 
df = pd.DataFrame(data={'Exams': tmp.values}, index=tmp.index).reset_index() 
sns.barplot(ax=ax,x = 'gleason_score', y='Exams',hue='spacing',data=df, palette='Set1') 
plt.title("Number of examinations grouped on Gleason Score and Spacing") 
plt.show()


# In[ ]:


fig, ax = plt.subplots(nrows=1,figsize=(12,6)) 
sns.distplot(train_df['width'], kde=True, label='width')
sns.distplot(train_df['height'], kde=True, label='height')
plt.xlabel('dimension')
plt.title('Images Width and Height distribution')
plt.legend()
plt.show()


# In[ ]:


def plot_distribution_grouped(feature, feature_group, hist_flag=True):
    fig, ax = plt.subplots(nrows=1,figsize=(12,6)) 
    for f in train_df[feature_group].unique():
        df = train_df.loc[train_df[feature_group] == f]
        sns.distplot(df[feature], hist=hist_flag, label=f)
    plt.title(f'Images {feature} distribution, grouped by {feature_group}')
    plt.legend()
    plt.show()


# In[ ]:


plot_distribution_grouped('width', 'data_provider')


# In[ ]:


plot_distribution_grouped('height', 'data_provider')


# In[ ]:


plot_distribution_grouped('width', 'isup_grade', False)


# In[ ]:


plot_distribution_grouped('height', 'isup_grade', False)


# In[ ]:


plot_distribution_grouped('width', 'gleason_score', False)


# In[ ]:


plot_distribution_grouped('height', 'gleason_score', False)


# # References
# 
# [1] YiFen Zhang, MD, PhD, Pathology Grading of Renal Cell Carcinoma, https://emedicine.medscape.com/article/1612022-overview  
# [2] Rohit Singh, PANDA - EDA + Better Visualization, https://www.kaggle.com/rohitsingh9990/panda-eda-better-visualization  
# [3] Dhananjay Raut, PANDA: EDA All you need to know, https://www.kaggle.com/dhananjay3/panda-eda-all-you-need-to-know
# 
