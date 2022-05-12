#!/usr/bin/env python
# coding: utf-8

# ## Pneumonia classification with CNN on X-Ray images using PyTorch

# # 1. Introduction and setups
# Hello everyone! In this notebook, I'm gonna use convolutional neural network models to predict pneumonia on x-ray images using PyTorch. Let's start!

# # 1.1 What is pneumonia
# **Pneumonia** is an infection in one or both lungs caused by bacteria, viruses, or fungi. The infection leads to inflammation in the air sacs of the lungs, which are called alveoli. The alveoli fill with fluid or pus, making it difficult to breathe. Symptoms can range from mild to serious and may include a cough with or without mucus (a slimy substance), fever, chills, and trouble breathing. How serious your pneumonia is depends on your age, your overall health, and what caused your infection. Both viral and bacterial pneumonia are contagious. 
# 
# To diagnose pneumonia, your healthcare provider will review your medical history, perform a physical exam, and order diagnostic tests such as a chest X-ray. 
# 
# ![Pneumonia illustration](https://img.freepik.com/vector-gratis/saluda-e-insalubre-pulmones-humanos_1308-29197.jpg?t=st=1651922559~exp=1651923159~hmac=cc743ddf052394279d4437886c5661103454ddeca79a28b5261b3f1fe7c17603&w=996)

# # 1.2 Import libraries

# In[ ]:


import pandas as pd
import numpy as np
import os, random, itertools

from PIL import Image
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_confusion_matrix

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import torchvision.transforms as T
import torchvision


# # 1.3 Reproducibility
# **Reproducibility** in machine learning means that you can repeatedly run your algorithm on certain datasets and obtain the same (or similar) results on a particular project. It's important to have reproducibility when you testing different models and hyperparameters or taking part in Kaggle competitions.
# 
# To obtain reproducibility, I used the next function (which was taken from [this repository](https://gist.github.com/ihoromi4/b681a9088f348942b01711f251e5f964) with some changes).

# In[ ]:


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# In[ ]:


seed_everything(42)


# # 1.4 Choosing device
# The next line enables us to automatically choose GPU as accelerator if it is able. Otherwise, it will choose CPU.

# In[ ]:


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# # 2. Loading data
# **From the considered dataset description:**
# 
# The dataset is organized into 3 folders (train, test, val) and contains subfolders for each image category (Pneumonia/Normal). There are 5,863 X-Ray images (JPEG) and 2 categories (Pneumonia/Normal).
# 
# Chest X-ray images (anterior-posterior) were selected from retrospective cohorts of pediatric patients of one to five years old from Guangzhou Women and Children’s Medical Center, Guangzhou. All chest X-ray imaging was performed as part of patients’ routine clinical care.
# 
# For the analysis of chest x-ray images, all chest radiographs were initially screened for quality control by removing all low quality or unreadable scans. The diagnoses for the images were then graded by two expert physicians before being cleared for training the AI system. In order to account for any grading errors, the evaluation set was also checked by a third expert.

# In[ ]:


path = '../input/chest-xray-pneumonia/chest_xray'


# The next function creates a dataframe that contains links to images from both classes and target values.

# In[ ]:


def df_from_dir(path, directory):
    path_to_nrm = os.path.join(path, directory, 'NORMAL')
    path_to_pn = os.path.join(path, directory, 'PNEUMONIA')
    
    list_class_nrm = [0] * len(os.listdir(path_to_nrm))
    list_class_pn = [1] * len(os.listdir(path_to_pn))
    
    list_path_nrm = [os.path.join(path_to_nrm, filename) for filename in os.listdir(path_to_nrm)]
    list_path_pn = [os.path.join(path_to_pn, filename) for filename in os.listdir(path_to_pn)]
    
    return pd.DataFrame({'path': list_path_nrm + list_path_pn, 'class': list_class_nrm + list_class_pn})


# In[ ]:


train_df = df_from_dir(path, 'train')
valid_add_df = df_from_dir(path, 'val')
test_df = df_from_dir(path, 'test')


# In[ ]:


train_df.head()


# # 3. Data analysis
# Let's start from visualizing number of objects of each class in the training dataset.

# In[ ]:


def balance_class_plot(df):
    sns.set_theme(style='darkgrid')
    plt.figure(figsize=(6, 6))
    ax = sns.countplot(data=df, x='class')
    plt.xlabel('Class', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.xticks(range(2), ['Normal', 'Pneumonia'], fontsize=14)
    plt.show()


# In[ ]:


balance_class_plot(train_df)


# As we can see, our **training dataset is imbalanced** with major class "Pneumonia" and minor class "Normal". It's a quite typical situation for medical datasets.
# 
# Now let's see what we have in the validation dataset.

# In[ ]:


balance_class_plot(valid_add_df)


# So, our validation dataset is balanced, but **only has 16 objects**. Therefore, we can't rely on it.
# 
# And finally, let's visualize what we have in the test dataset.

# In[ ]:


balance_class_plot(test_df)


# As we can see, the test dataset is absolutely balanced, and we can safely use accuracy as a metric.
# 
# Also, we should look closely at our images, specifically on dimensions.

# In[ ]:


first_random_image = Image.open(train_df.loc[0, 'path'])
first_random_image = np.array(first_random_image)

second_random_image = Image.open(train_df.loc[1345, 'path'])
second_random_image = np.array(second_random_image)

print(f'Shape of first random image: {first_random_image.shape}\nShape of second random image: {second_random_image.shape}')


# There are two conclusions from this output:
# 
# **First**, images in this dataset have different resolutions;
# 
# **Second**, some of images even have different amount of channels.

# # 4. Data preprocessing
# Before training the model and making predictions, we need to prepare our data based on conclusions from the analysis.

# # 4.1 Improving validation dataset
# To correct the validation dataset we sacrifice and give 10% of the training dataset to validation.

# In[ ]:


train_ds, valid_ds = train_test_split(train_df, test_size=0.1)
valid_ds = valid_ds.append(valid_add_df)


# In[ ]:


train_ds = train_ds.reset_index(drop=True)
valid_ds = valid_ds.reset_index(drop=True)


# # 4.2 Image augmentation
# 
# **Image augmentation** is a process of creating new training examples from the existing ones. To make a new sample, you slightly change the original image. For instance, you could make a new image a little brighter; you could cut a piece from the original image; you could make a new image by mirroring the original one, etc. By applying those transformations to the original training dataset, you could create an almost infinite amount of new training samples.
# 
# There is two main reasons for applying image augmentation:
# 
# **First**, increase the size of the dataset;
# 
# **Second**, prevent overfitting.  
#  
# The next transformation will be applied to each image in the validation and test dataset.

# In[ ]:


resize_transformation = T.Compose([T.Resize((224, 224)),
                                   T.Grayscale(num_output_channels=3),
                                   T.ToTensor()
])


# This transformation only resize image to 224x224, which is widespread resolution for input in pretrained models, ensure to have 3 chanels in each image and convert it to pytorch tensor.
# 
# The next transformation will be applied to the test dataset.

# In[ ]:


PT_transformation = T.Compose([T.Resize((224, 224)),
                               T.RandomAffine(degrees=(-10, 10), translate=(0.1, 0.1)),
                               T.RandomHorizontalFlip(p=0.5),
                               T.Grayscale(num_output_channels=3),
                               T.ToTensor()
])


# This transformation will do next:
# - Resize an image to 224x224;
# - Randomly rotate an image in the range (-10; 10) degrees;
# - Randomly shift, both vertical and horizontal, by up to 10% of its resolution;
# - Flip the image horizontal with a 50% chance;
# - Ensure to have 3 channels in an image;
# - Convert image to pytorch tensor.
# 
# You may ask, why did I choose specifically these transformations with these values? Because there were tested different combinations of transformations and values behind the scenes, and namely this configuration gives the best result with the choosed model (about which we will talk later :) ). 
# 
# **Note:** applying transformations before converting images to a three-channel format will noticeably decrease training time. For instance, performing the same notebook (one of the versions) but with mentioned changes speed it up from 1 hour 10 minutes to 57 minutes.
# 
# **Note 2:** ToTensor() convert a PIL Image in the range [0, 255] to a torch.FloatTensor in the range [0.0, 1.0], i.e. it also perform image normalization.

# # 4.3 Applying oversampling technique
# As we know, our training dataset is imbalanced. If we ignore this, we will lose "real accuracy". Why "real accuracy" instead of common accuracy? If the dataset is highly imbalanced, the model will get a pretty high accuracy just by predicting the majority class, but fail to capture the minority class. For example, the major class contains 99% of the objects and the minor only 1%. If the model always predict major class, the accuracy will be 99%. 
# 
# There are some techniques for handling class imbalance.
# 
# First is **undersampling**. In this technique, we randomly remove samples from the majority class until all the classes have the same number of samples. This technique has a significant disadvantage in that it discards data which might lead to a reduction in the number of representative samples in the dataset.
# 
# Next is **oversampling**. In this technique, we try to make the distribution of all the classes equal in a mini-batch by sampling an equal number of samples from all the classes thereby sampling more examples from minority classes as compared to majority classes. Practically it is done by increasing the sampling probability of examples belonging to minority class thereby down-weighing the sampling probability of examples belonging to the majority class. 
# 
# In **cost-sensitive learning**, the basic idea is to assign different costs to classes according to their distribution. There are various ways of implementing cost-sensitive learning like using higher learning rate for examples belonging to majority class as compared to examples belonging to minority class or using class weighted loss functions which calculate loss by taking the class distribution into account and hence penalize the classifier more for misclassifying examples from minority class as compared to majority class.
# 
# (most information about this techniques was taken from [this article](https://towardsdatascience.com/class-imbalance-d90f985c681e))
# 
# Oversampling and cost-sensitive learning showed the almost identical result, although the second was a little harder to implement and work with. So, I used oversampling technique in this notebook.

# In[ ]:


def weight_sampler_over(targets, length):
    classes, counts = np.unique(targets, return_counts=True)
    weight = (1/counts)[targets]
    sampler = WeightedRandomSampler(weight, length)
    return sampler


# In[ ]:


train_sampler = weight_sampler_over(np.array(train_ds.loc[:,'class']), np.array(train_ds.loc[:,'class']).shape[0])
valid_sampler = weight_sampler_over(np.array(valid_ds.loc[:,'class']), np.array(valid_ds.loc[:,'class']).shape[0])


# # 4.4 Creating custom dataset
# Now we are going to create a class for a custom dataset. It will store the dataframe and transformation for it. Also, it will give a transformed image with a label on request.

# In[ ]:


class MyDataset(Dataset):
    def __init__(self, dataframe, transform):
        self.dataframe = dataframe
        self.transform = transform
        
    def __len__(self):
        return self.dataframe.shape[0]
    
    def __getitem__(self, index):
        label = self.dataframe.loc[index, 'class']
        image = Image.open(self.dataframe.loc[index, 'path'])
        image = self.transform(image)
        return image, label


# In[ ]:


train_dataset = MyDataset(train_ds, PT_transformation)
valid_dataset = MyDataset(valid_ds, resize_transformation)
test_dataset = MyDataset(test_df, resize_transformation)


# Then, create dataloaders for each dataset to give batches of images instead of single images to the model.

# In[ ]:


bs = 32
train_dl = DataLoader(train_dataset, batch_size=bs, pin_memory=True, sampler=train_sampler)
valid_dl = DataLoader(valid_dataset, batch_size=bs*2, pin_memory=True, sampler=valid_sampler)
test_dl = DataLoader(test_dataset, batch_size=bs*2, pin_memory=True)


# # 5. Model selection
# After dozens of hours of testing different versions of ResNet, VGG, DenseNet, EfficientNet, and custom nn models with different configurations, the best results showed VGG-16 with batch normalization with centered RMSprop optimizer and BCELoss. Also, a little bit worse, but close result showed VGG-13.

# In[ ]:


pt_model = torchvision.models.vgg16_bn(pretrained=True)


# Freeze all pretrained model parameters, so this part will not be trained, and add a linear classifier with 1 output (1 and 2 outputs showed almost equivalent results).

# In[ ]:


for param in pt_model.parameters():
    param.requires_grad = False

pt_model.classifier[-1] = nn.Linear(in_features=pt_model.classifier[-1].in_features, out_features=1, bias=True)

pretrained_model = nn.Sequential(pt_model, nn.Sigmoid())
pretrained_model.to(device)


# In[ ]:


opt = optim.RMSprop(pretrained_model.parameters(), centered=True)
loss = nn.BCELoss()


# The next scheduler will reduce the learning rate by half if there is no improvement in the loss metric after 3 epochs.

# In[ ]:


scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)


# # 6. Model training
# The next function will train our model. It saves the best model state based on validation loss value. Also, it stores and shows metric and loss values after each epoch.

# In[ ]:


def fit(epochs, train_dl, valid_dl, model, opt, loss_func, metric, scheduler=None, tracker=None):
    
    val_loss_min = np.Inf
    
    for epoch in range(epochs):
        
        train_loss = 0.0
        train_metric = 0.0
        val_loss = 0.0
        val_metric = 0.0
        
        model.train()
        for xb, yb in train_dl:
            xb = xb.to(device)
            yb = yb.to(device)
            
            pred = model(xb)
            loss = loss_func(pred, yb.unsqueeze(1).float())
            loss.backward()
            
            opt.step()
            opt.zero_grad()
            
            train_loss += loss.item()
            train_metric += metric(np.round(pred.detach().cpu().numpy()), yb.cpu().numpy())
        
        avg_train_loss = train_loss / len(train_dl)
        avg_train_metric = train_metric / len(train_dl)
        
        model.eval()
        with torch.no_grad():
            for xb, yb in valid_dl:
                xb = xb.to(device)
                yb = yb.to(device)
                
                pred = model(xb)
                loss = loss_func(pred, yb.unsqueeze(1).float())
                
                val_loss += loss.item()
                val_metric += metric(np.round(pred.detach().cpu().numpy()), yb.cpu().numpy())
                
            avg_val_loss = val_loss / len(valid_dl) 
            avg_val_metric = val_metric / len(valid_dl) 
            
        if tracker:
            tracker['train_loss'].append(avg_train_loss)
            tracker['train_metric'].append(avg_train_metric)
            tracker['validation_loss'].append(avg_val_loss)
            tracker['validation_metric'].append(avg_val_metric)
        
        if scheduler:
            scheduler.step(avg_val_loss)
            
        if avg_val_loss < val_loss_min:
            print(f'Epoch: {epoch + 1} - Train loss: {avg_train_loss:.3} - Train metric: {avg_train_metric:.3}                - Val loss: {avg_val_loss:.3} - Val metric: {avg_val_metric:.3}. Best result!')
            val_loss_min = avg_val_loss
            torch.save(model.state_dict(), 'Pneumonia_model_weights.pt')
        else:
            print(f'Epoch: {epoch + 1} - Train loss: {avg_train_loss:.3} - Train metric: {avg_train_metric:.3}                - Val loss: {avg_val_loss:.3} - Val metric: {avg_val_metric:.3}.')


# **Note:** F1-score as a metric will not reflect reality if it is calculated separately on batches.
# 
# Then, declare and initialize necessary variables.

# In[ ]:


epochs = 50
tracker = {'train_loss': [], 'train_metric': [], 'validation_loss': [], 'validation_metric': []}


# Finally, train our model during 50 epochs with mentioned above parameters.

# In[ ]:


fit(epochs, train_dl, valid_dl, pretrained_model, opt, loss, metrics.accuracy_score, scheduler, tracker)


# As we can see, the best model was on 47 epoch. Next, let's visualize losses and metrics over epochs during the training process.

# In[ ]:


x = np.arange(1, epochs + 1)

plt.figure(figsize=(16, 8))

plt.subplot(1, 2, 1)
plt.plot(x, tracker['train_loss'], label='Train loss')
plt.plot(x, tracker['validation_loss'], label='Validation loss')
plt.title('Loss over epochs', fontsize=16)
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Loss value', fontsize=14)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(x, tracker['train_metric'], label='Train metric')
plt.plot(x, tracker['validation_metric'], label='Validation metric')
plt.title('Metric over epochs', fontsize=16)
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Metric value', fontsize=14)
plt.legend()

plt.show()


# Ok, let's look closer starting from third epoch.

# In[ ]:


x = np.arange(3, epochs + 1)

plt.figure(figsize=(16, 8))

plt.subplot(1, 2, 1)
plt.plot(x, tracker['train_loss'][2:], label='Train loss')
plt.plot(x, tracker['validation_loss'][2:], label='Validation loss')
plt.title('Loss over epochs', fontsize=16)
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Loss value', fontsize=14)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(x, tracker['train_metric'][2:], label='Train metric')
plt.plot(x, tracker['validation_metric'][2:], label='Validation metric')
plt.title('Metric over epochs', fontsize=16)
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Metric value', fontsize=14)
plt.legend()

plt.show()


# As we can see, the best result is on around the 47th epoch. And at the end, load our final model.

# In[ ]:


pretrained_model.load_state_dict(torch.load('./Pneumonia_model_weights.pt'))


# # 7. Predicting on test data
# Now, let's test our model by predicting on the test dataset.

# In[ ]:


predicted = []
true = []

for xb, yb in test_dl:
    xb = xb.to(device)
    predicted.append(pretrained_model(xb).detach().cpu().numpy())
    true.append(yb.numpy())
    
predicted = list(itertools.chain(*list(itertools.chain(*predicted))))
true = list(itertools.chain(*true))

f1_score = metrics.f1_score(true, np.round(predicted))
acc_score = metrics.accuracy_score(true, np.round(predicted))

print(f'F1-score: {f1_score:.3}\nAccuracy: {acc_score:.3}')


# As we can see, the accuracy of the model is 90.9%, and F1-score is 92.9, which is a pretty good result. Also, let's build a confusion matrix.

# In[ ]:


conf_matrix = metrics.confusion_matrix(true, np.round(predicted))
fig, ax = plot_confusion_matrix(conf_mat=conf_matrix, figsize=(6, 6), cmap=plt.cm.Blues)
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.xticks(range(2), ['Normal', 'Pneumonia'], fontsize=14)
plt.yticks(range(2), ['Normal', 'Pneumonia'], fontsize=14)
plt.show()


# We can see that around 3/4 of all misclassified objects are false positives, which is quite good in our case. This means that if the model predicts wrong - most of the misclassified patients will have additional medical tests rather than be released and have more serious health issues later.

# ## Thank you for your attention!
# Hope, it was interesting or useful for you. Please, upvote if you like it.
