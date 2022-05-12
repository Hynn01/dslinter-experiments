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
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # Handling Data

# In[ ]:


from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
import zipfile
import os
import shutil
import copy
import time


# In[ ]:


z= zipfile.ZipFile('../input/dogs-vs-cats/train.zip')
z.extractall()

z1= zipfile.ZipFile('../input/dogs-vs-cats/test1.zip')
z1.extractall()


# In[ ]:


train_path = "./train"
test_path = "./test1"


# In[ ]:


filepath_train = os.listdir(train_path)
filepath_test1 = os.listdir(test_path)
#shutil.rmtree("./data")


# In[ ]:


cat=0
dog=0
others=0
total_len=0
for i in tqdm(filepath_train):
    if i.split(".")[0]== "cat":
        cat+=1
    elif i.split(".")[0] == "dog":
        dog+=1
    else:
        others+=1

print(f"Cats: {cat}, Dogs: {dog}, Others: {others}, Length: {cat+dog+others}")


# In[ ]:


os.mkdir("./data")


# In[ ]:


#try:
os.mkdir("./data/train_data")
os.mkdir("./data/train_data/cats")
os.mkdir("./data/train_data/dogs")
    
#except Exception as e:
#print(e)


# In[ ]:


os.mkdir("./data/val_data")
os.mkdir("./data/val_data/cats")
os.mkdir("./data/val_data/dogs")


# In[ ]:


#os.remove("./train_data/dogs")
os.listdir("./data/train_data")


# In[ ]:


train_data_root = "./data/train_data"
cats_train_folder = "./data/train_data/cats"
dogs_train_folder = "./data/train_data/dogs"

val_data_root = "./data/val_data"
cats_val_folder = "./data/val_data/cats"         
dogs_val_folder = "./data/val_data/dogs"


# In[ ]:


os.listdir(train_data_root)


# In[ ]:


cat=0
dog=0
others=0
total_len=0
for i in tqdm(filepath_train):
    src = os.path.join(train_path,i)
    if i.split(".")[0]== "cat":
        cat+=1
        shutil.copy(src,cats_train_folder)
    elif i.split(".")[0] == "dog":
        dog+=1
        shutil.copy(src,dogs_train_folder)
    else:
        others+=1

print(f"Cats: {cat}, Dogs: {dog}, Others: {others}, Length: {cat+dog+others}")
print("---------------------<SUCESSFULLY MOVED>---------------------")


# In[ ]:


len(os.listdir(cats_train_folder))


# In[ ]:


os.listdir(dogs_train_folder)[:5]


# In[ ]:


cat_train_images = os.listdir(cats_train_folder)
dog_train_images = os.listdir(dogs_train_folder)
cat_train_images[:5]


# In[ ]:


img = cv2.imread(os.path.join(cats_train_folder,cat_train_images[0]))
#print(img)
plt.imshow(img)


# In[ ]:


val_cat_images = cat_train_images[-int((len(cat_train_images)*0.2)):]
src = cats_train_folder
destination = cats_val_folder
for cat in val_cat_images:
    shutil.move(os.path.join(src,cat),destination)
    
print("-------COMPLETE-------")
print(len(os.listdir(cats_val_folder)))


# In[ ]:


val_dog_images = dog_train_images[-int((len(dog_train_images)*0.2)):]
src = dogs_train_folder
destination = dogs_val_folder
for dog in val_dog_images:
    shutil.move(os.path.join(src,dog),destination)
    
print("-------COMPLETE-------")
print(len(os.listdir(dogs_val_folder)))


# # Working with Data

# In[ ]:


import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder


# In[ ]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device


# In[ ]:


data_dir = "./data"
os.listdir(data_dir)


# In[ ]:


sets = ['train_data', 'val_data']


# In[ ]:


train_transform = transforms.Compose([transforms.Resize((224,224)),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
                                     transforms.RandomHorizontalFlip(0.5)])

val_transform = transforms.Compose([transforms.Resize((224,224)),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

data_transforms = {
    "train_data": train_transform,
    "val_data": val_transform
}

data_transforms["val_data"]


# In[ ]:


#train_dataset = ImageFolder(os.path.join(data_dir,"train_data"), transform=train_transform)
#val_dataset = ImageFolder(os.path.join(data_dir,"val_data"), transform=val_transform)

image_datasets = {x: ImageFolder(os.path.join(data_dir,x), transform=data_transforms[x]) for x in ["train_data","val_data"]}


# In[ ]:


idx = 19213
img_permuted = image_datasets["train_data"][idx][0].permute(1,2,0)
plt.imshow(img_permuted)
print(image_datasets["train_data"].classes[ image_datasets["train_data"][idx][1]])


# In[ ]:


#train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
#val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=True)

dataloaders = {x: DataLoader(image_datasets[x],batch_size=16,shuffle=True) for x in ["train_data","val_data"]}


# In[ ]:


dataloaders["train_data"]
#train_dataloader


# In[ ]:


a = iter(dataloaders["train_data"])
a.next()[0].shape


# # Train Function

# In[ ]:


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        
        for phase in ['train_data', 'val_data']:
            if phase == 'train_data':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

           
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                
                with torch.set_grad_enabled(phase == 'train_data'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    
                    if phase == 'train_data':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

               
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train_data':
                scheduler.step()

            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = running_corrects.double() / len(image_datasets[phase])

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

          
            if phase == 'val_data' and epoch_acc > best_acc:
                best_acc = epoch_acc
                

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    
    return model
            


# # Transfer Learning with Resnet18

# In[ ]:


model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features


# In[ ]:


model.fc = nn.Linear(num_ftrs, 2)


# In[ ]:


model = model.to(device)


# In[ ]:


criterion = nn.CrossEntropyLoss()


# In[ ]:


optimizer = optim.SGD(model.parameters(), lr=0.001)


# In[ ]:


step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)


# In[ ]:


model = train_model(model, criterion, optimizer, step_lr_scheduler, num_epochs=3)


# # Saving Model

# In[ ]:


FILE = "dogs_vs_cats_resnet18.pth"


# In[ ]:


torch.save(model.state_dict(),FILE)


# In[ ]:


from IPython.display import FileLink
FileLink(r'./dogs_vs_cats_resnet18.pth')


# # Testing

# In[ ]:


test_images = os.listdir(test_path)


# In[ ]:


#test_dataset = ImageFolder(os.path.join(data_dir,"val_data"), transform=val_transform)


# In[ ]:


test_transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize((224,224)),
                                     transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])


# In[ ]:


classes = ["cat","dog"]


# In[ ]:


model.eval()


# In[ ]:


for i in range(230,240):
    img_path = os.path.join(test_path,test_images[i])
    img = cv2.imread(img_path)
    tensor_img = test_transform(img)
    tensor_img = tensor_img.reshape(-1,3,224,224)
    tensor_img=tensor_img.to(device)
    prediction = model(tensor_img)
    _,pred = torch.max(prediction,axis=1)
    result = classes[pred]
    print(result)
    #print(tensor_img)
    plt.imshow(img)
    plt.show()


# # Summary
# 
# ## Resnet18 validation accuracy is very good (greater than 98%)
# ## Using ImageFolder is an easy method of loading images
# ## ImageFolder required a lot of folder organizing for this project
# ## With transfer learning, 3 epochs were enough to reach this accuracy

# In[ ]:


import shutil
from IPython.display import FileLink

OUTPUT_NAME = "test1"
DIRECTORY_TO_ZIP = "./test1"
shutil.make_archive(OUTPUT_NAME, 'zip', DIRECTORY_TO_ZIP)

FileLink(r'./test1.zip')

