#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('wget -c https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz')
get_ipython().system('tar -vxf cifar-10-python.tar.gz')
get_ipython().system('rm cifar-10-python.tar.gz')


# In[ ]:


get_ipython().system('pip install torchviz')


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import pickle
import time
import skimage.io
import numpy as np
import pandas as pd
import cv2
import PIL.Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler, RandomSampler, SequentialSampler
import torchvision.models as models
import albumentations
from sklearn.model_selection import StratifiedKFold, KFold
from tqdm import tqdm_notebook as tqdm
import os
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import gc

from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from torchviz import make_dot


device = torch.device('cuda')

### Utility functions:

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo,encoding='latin1')
    return dict


def scorer(models,x_test,y_true):
    for model in models:
        y_pred = model.predict(x_test)
        print(accuracy_score(y_true, y_pred))
        

from prettytable import PrettyTable

def count_parameters(model):
    #reference and credits : https://stackoverflow.com/a/62508086/9017542
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params+=params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params
        


# In[ ]:


ROOT_PATH='./' 
batch1 = unpickle(ROOT_PATH+"cifar-10-batches-py/data_batch_1")
batch2 = unpickle(ROOT_PATH+"cifar-10-batches-py/data_batch_2")
batch3 = unpickle(ROOT_PATH+"cifar-10-batches-py/data_batch_3")
batch4 = unpickle(ROOT_PATH+"cifar-10-batches-py/data_batch_4")
batch5 = unpickle(ROOT_PATH+"cifar-10-batches-py/data_batch_5")
test_batch = unpickle(ROOT_PATH+"cifar-10-batches-py/test_batch")


# ### Visualising Images

# In[ ]:


class_mapping = {0: 'airplane',
1: 'automobile',
2: 'bird',
3: 'cat',
4: 'deer',
5: 'dog',
6: 'frog' ,
7: 'horse',
8: 'ship',
9: 'truck'}

def visualize(batch):
    from pylab import rcParams
    rcParams['figure.figsize'] = 20,10
    for i in range(2):
        f, axarr = plt.subplots(1,5)
        for p in range(5):
            idx = np.random.randint(0, len(batch['data']))
            img = batch['data'][idx]
            label = batch['labels'][idx]
            name = batch['filenames'][idx]
            name = name.split('_')[0]
            axarr[p].imshow(np.fliplr(np.rot90(np.transpose(img.flatten().reshape(3,32,32)), k=-1)))
            axarr[p].set_title(class_mapping[label]+' ('+str(label)+')')
        


# In[ ]:


visualize(batch1)


# In[ ]:


visualize(batch2)


# In[ ]:


visualize(batch3)


# In[ ]:


visualize(batch4)


# In[ ]:


visualize(batch5)


# # Numpy DataSet

# In[ ]:


def load_data0(btch):
    labels = btch['labels']
    imgs = btch['data'].reshape((-1, 32, 32, 3))
    
    res = []
    for ii in range(imgs.shape[0]):
        img = imgs[ii].copy()
        img = np.fliplr(np.rot90(np.transpose(img.flatten().reshape(3,32,32)), k=-1))
        res.append(img.flatten())
    imgs = np.stack(res)
    return labels, imgs


def load_data():
    x_train_l = []
    y_train_l = []
    for ibatch in [batch1, batch2, batch3, batch4, batch5]:
        labels, imgs = load_data0(ibatch)
        x_train_l.append(imgs)
        y_train_l.extend(labels)
    x_train = np.vstack(x_train_l)
    y_train = np.vstack(y_train_l)
    
    x_test_l = []
    y_test_l = []
    labels, imgs = load_data0(test_batch)
    x_test_l.append(imgs)
    y_test_l.extend(labels)
    x_test = np.vstack(x_test_l)
    y_test = np.vstack(y_test_l)
    return (x_train, y_train), (x_test, y_test)

(x_train, y_train), (x_test, y_test) = load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')


# In[ ]:


y_train = y_train.ravel()
y_test = y_test.ravel()


# # Building Logistic Regression, Decision Trees, Random Forests. 

# In[ ]:


print('Building Logistic Regression')

lg = make_pipeline(StandardScaler(), LogisticRegression(max_iter = 250, n_jobs = -1, random_state=0))
lg = lg.fit(x_train, y_train)
gc.collect()


print('Building Decision Trees')
dt = make_pipeline(StandardScaler(),DecisionTreeClassifier(random_state=42))
dt = dt.fit(x_train,y_train)


print('Building Random Forests')
rf = make_pipeline(StandardScaler(),RandomForestClassifier(max_depth=2, random_state=0))
rf = rf.fit(x_train,y_train)


scorer([lg,dt,rf],x_test,y_test)


# In[ ]:


gc.collect()


# ## Neural Networks
# 

# In[ ]:


criterion = nn.CrossEntropyLoss()

transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) 


# In[ ]:


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu = nn.ReLU(inplace=False)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x1 = self.relu(x)
        x = x+x1 #resnet like arch
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# In[ ]:


class Vanilla_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(32*32*3, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# In[ ]:


class resnet_cifar10(nn.Module):
    def __init__(self, out_dim):
        super(resnet_cifar10, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.myfc = nn.Linear(self.model.fc.in_features, out_dim)
        self.model.fc = nn.Identity()

    def extract(self, x):
        return self.model(x)

    def forward(self, x):
        x = self.extract(x)
        x = self.myfc(x)
        return x


# In[ ]:


def train_epoch(loader, optimizer):
    total = 0
    correct = 0
    model.train()
    train_loss = []
    bar = tqdm(loader)
    for (data, target) in bar:
        
        data, target = data.to(device), target.to(device)
        loss_func = criterion
        optimizer.zero_grad()
        logits = model(data)
        _,pred = torch.max(logits,1)
        
        # Accuaracy code
        total += target.size(0)
        correct += (pred == target).sum().item()
        acc = 100 * (correct / total)
        
        loss = loss_func(logits, target)
        loss.backward()
        optimizer.step()

        loss_np = loss.detach().cpu().numpy()
        train_loss.append(loss_np)
        #smooth_loss = sum(train_loss[-100:]) / min(len(train_loss), 100)
        bar.set_description('loss: %.5f, Accuracy: %.5f' % (loss_np, acc))
    return train_loss,acc


def val_epoch(loader, get_output=False):
    total = 0
    correct = 0
    model.eval()
    val_loss = []
    LOGITS = []
    PREDS = []
    TARGETS = []

    with torch.no_grad():
        for (data, target) in tqdm(loader):
            data, target = data.to(device), target.to(device)
            logits = model(data)
            
            loss = criterion(logits, target)
            #print(logits)
            _, pred = torch.max(logits, 1)
            LOGITS.append(logits)
            PREDS.append(pred)
            TARGETS.append(target)
            total += target.size(0)
            correct += (pred == target).sum().item()
            
            val_loss.append(loss.detach().cpu().numpy())
        val_loss = np.mean(val_loss)

    LOGITS = torch.cat(LOGITS).cpu().numpy()
    PREDS = torch.cat(PREDS).cpu().numpy()
    TARGETS = torch.cat(TARGETS).cpu().numpy()
    
    #print(correct,total)
    acc = 100 * (correct / total)
        

    if get_output:
        return LOGITS
    else:
        return val_loss, acc


# In[ ]:


trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)


testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)


# In[ ]:


batch_size = 128
num_workers = 2
n_epochs = 10


train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)
    
valid_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)


# In[ ]:


def run(model,mtype,tl,vl):


    #dataset_train = CIFARDataset(data , transform=transform)
    #dataset_valid = CIFARDataset(test , transform=transform)
  

    #model = Net()
    model = model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=0.01,momentum=0.9, weight_decay=5e-4)
    #optim.Adam(model.parameters(), lr=0.01)
    

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    #print(len(dataset_train), len(dataset_valid))
    
    train_list = []
    train_acc_list = []
    val_list = []
    val_acc_list = []
    
    acc_max = 0
    for epoch in range(1, n_epochs+1):
        print(time.ctime(), 'Epoch:', epoch)

        train_loss,t_acc = train_epoch(tl, optimizer)
        train_list.append(np.mean(train_loss))
        train_acc_list.append(t_acc)
        val_loss, acc = val_epoch(vl)
        val_list.append(np.mean(val_loss))
        val_acc_list.append(acc)
        scheduler.step()
        
        #print)

        content = time.ctime() + ' ' + f'Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, train loss: {np.mean(train_loss):.5f}, val loss: {np.mean(val_loss):.5f}, acc: {(acc):.5f}'
        print(content)
        with open(f'log_basic_cnn.txt', 'a') as appender:
            appender.write(content + '\n')

        if acc > acc_max:
            print('score2 ({:.6f} --> {:.6f}).  Saving model ...'.format(acc_max, acc))
            torch.save(model.state_dict(), f'model.pth')
            acc_max = acc
    
    
    x_ticks = [1,2,3,4,5,6,7,8,9,10]
    x_labels = [1,2,3,4,5,6,7,8,9,10] 
    
    plt.figure(figsize=(10, 8))
    plt.plot(train_list)
    plt.plot(val_list)
    plt.title(f'Training Loss vs Validation Loss for {mtype}')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.xticks(ticks=x_ticks, labels=x_labels)
    plt.legend(['train', 'val'], loc='upper right')
    plt.show()
    
    plt.figure(figsize=(10, 8))
    plt.plot(train_acc_list)
    plt.plot(val_acc_list)
    plt.title(f'Training Accuracy vs Validation Accuracy for {mtype}')
    plt.ylabel('Accuracy')
    plt.xlabel('epoch')
    plt.xticks(ticks=x_ticks, labels=x_labels)
    plt.legend(['train', 'val'], loc='upper right')
    plt.show()
    
    torch.save(model.state_dict(), os.path.join(f'basic_cnn_final.pth'))


# In[ ]:


model = Net()
run(model,'Convolutional Neural Network',train_loader,valid_loader)
batch = next(iter(train_loader))
yhat = model(batch[0].to(device))
make_dot(yhat, params=dict(list(model.named_parameters()))).render("cnn")
count_parameters(model)


# In[ ]:


model = Vanilla_Net()
run(model,'Neural Network',train_loader,valid_loader)
batch = next(iter(train_loader))
yhat = model(batch[0].to(device))
make_dot(yhat, params=dict(list(model.named_parameters()))).render("nn")
count_parameters(model)


# In[ ]:


model = resnet_cifar10(out_dim = 10)
run(model,'Resnet 18',train_loader,valid_loader)
batch = next(iter(train_loader))
yhat = model(batch[0].to(device))
make_dot(yhat, params=dict(list(model.named_parameters()))).render("resnet")
count_parameters(model) 

