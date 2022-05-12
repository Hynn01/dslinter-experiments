#!/usr/bin/env python
# coding: utf-8

# 快来查看我用飞书分享的【Medical imaging final project】👉https://bytedance.feishu.cn/docx/doxcnUZ0zoBiu33CscEGVrDxF8b

# # 0. Dependency And Configuration

# In[ ]:


get_ipython().system('pip install torchinfo')
get_ipython().system('pip install openpyxl')
get_ipython().system('pip install hvplot')


# In[ ]:


get_ipython().system('pip install d2l')


# In[ ]:


import os  
import glob
import sklearn
from sklearn.model_selection import train_test_split

import PIL 
import numpy as np
import matplotlib.pyplot as plt 

import torch
import torch.nn as nn
from torchinfo import summary 

import torch.optim as optim
from IPython.display import Image
from torch.utils.data import DataLoader, Dataset

from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms

import cv2

import pandas as pd

from dask import bag, diagnostics 
import hvplot.pandas  

from d2l import torch as d2l
import torchvision
import copy


# Configuration

# In[ ]:


class CFG:
    seed          = 124
    debug         = False                             # set debug=False for Full Training
    exp_name      = 'AI-Medical-Final'
    model_name    = 'Unet'
    backbone      = 'efficientnet-b2'
    train_bs      = 24
    valid_bs      = 48
    img_size      = [250,250]
    epochs        = 50
    lr            = 5e-3
    scheduler     = 'CosineAnnealingLR'
    min_lr        = 1e-6
    T_max         = int(100*6*1.8)
    T_0           = 25
    warmup_epochs = 0
    wd            = 1e-6
    n_accumulate  = 32//train_bs
    n_fold        = 5
    num_classes   = 1
    device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _wandb_kernel = 'awsaf49'
    # 文件路径
    root          = '/kaggle/input/covidct/'
    csv_file      = root + 'COVID-CT-MetaInfo.xlsx'
    sheet_name    = 'positive_captions'


# In[ ]:


np.random.seed(CFG.seed)
torch.manual_seed(CFG.seed)
torch.backends.cudnn.deterministic = True


# # 1. Data Preparation

# In[ ]:


# List files available
list(os.listdir(CFG.root))


# In[ ]:


pos_files = glob.glob(os.path.join(CFG.root, "CT_COVID",'*.*'))
neg_files = glob.glob(os.path.join(CFG.root, 'CT_NonCOVID','*.*'))

images = pos_files + neg_files

labels = np.array([1]*len(pos_files)+[0]*len(neg_files))


images_tv, images_test, y_tv, y_test  = train_test_split(images, labels, shuffle=True, test_size=0.2, random_state=CFG.seed)
images_train, images_val, y_train, y_val  = train_test_split(images_tv, y_tv, shuffle=True, test_size=0.25, random_state=CFG.seed)


# In[ ]:


meta_df = pd.read_excel(CFG.csv_file, sheet_name=CFG.sheet_name, header=None)
meta_df


# # 2. EDA

# In[ ]:


num_pos, num_neg = len(pos_files), len(neg_files)

plt.title('Distribution of labels')
plt.bar(['Positive', 'Negative'], [num_pos, num_neg])
plt.show()


# In[ ]:


im = [cv2.imread(images_train[i]) for i in range(6)]

fig,ax = plt.subplots(ncols=6, figsize=(18,6))
for i in range(len(im)):
    ax[i].imshow(im[i],cmap='gray')

plt.show()


# In[ ]:


print(f'Number of samples in each set (train, val, test): {len(y_train), len(y_val), len(y_test)}')

print(f'Number of positive samples in each set: {y_train.sum(), y_val.sum(), y_test.sum()}')


# The images are clearly of different dimensions. We can observe the distribution of images' dimension.

# In[ ]:


# get image dimensions
def get_dims(file):
    img = cv2.imread(file)
    h,w = img.shape[:2]
    return h,w

# parallelize
dimsbag = bag.from_sequence(images).map(get_dims)
with diagnostics.ProgressBar():
    dims = dimsbag.compute()
    
dim_df = pd.DataFrame(dims, columns=['height', 'width'])
sizes = dim_df.groupby(['height', 'width']).size().reset_index().rename(columns={0:'count'})
sizes.hvplot.scatter(x='height', y='width', size='count', xlim=(0,1200), ylim=(0,1200), grid=True, xticks=2, 
        yticks=2, height=500, width=600).options(scaling_factor=0.1, line_alpha=1, fill_alpha=0)


# # 3. Dataset

# In[ ]:


class CT_Dataset(Dataset):
    def __init__(self, img_path, img_labels, img_transforms=None, grayscale=True):
        self.img_path = img_path
        self.img_labels = torch.Tensor(img_labels)
        self.img_transfroms = img_transforms
        if (img_transforms is None) & (grayscale == True):
            self.transforms = transforms.Compose([transforms.Grayscale(),
                                                  transforms.Resize((250, 250)),
                                                  transforms.ToTensor()])
        elif grayscale == False:
            self.transforms = transforms.Compose([transforms.Resize((250, 250)),
                                                  transforms.ToTensor()])
        else:
            self.transforms = img_transforms
    
    def __getitem__(self, index):
        # load image
        cur_path = self.img_path[index]
        cur_img = PIL.Image.open(cur_path).convert('RGB')
        cur_img = self.transforms(cur_img)

        return cur_img, self.img_labels[index]
    
    def __len__(self):
        return len(self.img_path)


# # 4. Data Augmentation

# In[ ]:


class D_aug:
    def __init__(self,status):
        self.status = status
        self.transform = {
            'train':
                    transforms.Compose([transforms.Grayscale(),
                                transforms.RandomRotation(5),
                                transforms.Resize(CFG.img_size),
                                transforms.RandomVerticalFlip(),
                                transforms.RandomHorizontalFlip(),
                                transforms.ColorJitter(brightness=0.3, contrast=0.5, saturation=0.5, hue=0.5),
                                transforms.RandomAffine(degrees=0, scale=(1.1, 1.1), shear=0.9),
                                transforms.ToTensor()
                                ]),
            'valid':
                    transforms.Compose([transforms.Grayscale(),
                                transforms.Resize(CFG.img_size),
                                transforms.ToTensor()
                                ]),
            'test' :
                    transforms.Compose([transforms.Grayscale(),
                                transforms.Resize(CFG.img_size),
                                transforms.ToTensor()
                                ])                    
                         }

    def __call__(self, image):
        trans = self.transform[self.status](image)
        return trans


# In[ ]:


img = PIL.Image.open(images_train[10])
D_instance = D_aug('train')
display(transforms.ToPILImage()(D_instance(img)))


# In[ ]:


temp_train_aug = CT_Dataset(img_path=images_train, img_labels=y_train, img_transforms=D_instance)
print(len(temp_train_aug))


# In[ ]:


# Single instance of data augmentation method
def apply_trans_method(img, aug, num_rows=2, num_cols=4, scale=4):
    Y = [aug(img) for _ in range(num_rows * num_cols)]
    d2l.show_images(Y, num_rows, num_cols, scale=scale)

apply_trans_method(d2l.Image.open(images_train[10]),transforms.ColorJitter(
    brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5))


# In[ ]:


# Compose
def visualization_aug_method(dataset,index,row,col):
    fig, axes = plt.subplots(row, col, figsize=(15, 10))
    for j in range(row):
        for i in range(col):
        # Each time the data is accessed, the result is different due to random augmentation!
            img, label = temp_train_aug[index]
            ax = axes[j][i]
            ax.imshow(img.cpu().numpy().transpose((1, 2, 0)) / 255.)
            ax.set_title(f"{index}-th image: label {label}")
    plt.show()
visualization_aug_method(temp_train_aug,10,2,4)


# # 5. DataLoader

# ## 5.1 Original Dataset

# In[ ]:


train_dataset = CT_Dataset(img_path=images_train, img_labels=y_train)
val_dataset = CT_Dataset(img_path=images_val, img_labels=y_val)
test_dataset = CT_Dataset(img_path=images_test, img_labels=y_test)


# ## 5.2 Augmenting the Entire Training Set

# In[ ]:


D_instance = D_aug('train')
train_dataset_full_aug = CT_Dataset(img_path=images_train, img_labels=y_train, img_transforms=D_instance)


# ## 5.3 Concatenate Augmented Data to Original Dataset

# In[ ]:


train_dataset_fin = torch.utils.data.ConcatDataset([train_dataset,train_dataset_full_aug])

print(len(train_dataset_fin))


# # 6. Baseline Model

# ## 6.1 Encoder & Decoder

# In[ ]:


# Encoder
class Encoder(nn.Module):
    def __init__(self, dropout=0.5):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
          # input (num_batch, 1, 250, 250)
          nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3),  # (num_batch, 64, 248, 248)
          nn.BatchNorm2d(64),
          nn.ReLU(),
          nn.MaxPool2d(kernel_size=2),  # (num_batch, 64, 124, 124)

          nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3), # (num_batch, 128, 122, 122)
          nn.BatchNorm2d(128),
          nn.ReLU(),
          nn.MaxPool2d(kernel_size=2),  # (num_batch, 128, 61, 61)

          nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3), # (num_batch, 256, 59, 59)
          nn.BatchNorm2d(256),
          nn.ReLU(),
          nn.MaxPool2d(kernel_size=2),  # (num_batch, 256, 29, 29)

          nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3), # (num_batch, 128, 27, 27)
          nn.BatchNorm2d(512),
          nn.ReLU(),
          nn.MaxPool2d(kernel_size=2),  # (num_batch, 128, 13, 13)

          nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3), # (num_batch, 64, 11, 11)
          nn.BatchNorm2d(512),
          nn.ReLU(),
          nn.MaxPool2d(kernel_size=2),  # (num_batch, 64, 5, 5)
          nn.Flatten() # (num_batch, 1600)
        )
    def forward(self, x):
        x = self.encoder(x)
        return x


# In[ ]:


# Decoder
class Decoder(nn.Module):
    def __init__(self, dropout=0.5):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Dropout(dropout),  # Dropout before first linear layer since it has a large number of trainable parameters
            nn.Linear(in_features= 12800, out_features=512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=1)
        )
    
    def forward(self, x):
        x = self.decoder(x)
        return x


# ## 6.2 Convnet

# In[ ]:


class Convnet(nn.Module):
    
    def __init__(self, encoder, decoder, **kwargs):
        super(Convnet, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, *args):
        enc_outputs = self.encoder(enc_X, *args)
        return (enc_outputs,self.decoder(enc_outputs,*args))


# ## 6.3 Unit Test

# In[ ]:


encoder_temp = Encoder()
encoder_temp.eval()
X_temp = torch.zeros((1,1, 250, 250), dtype=torch.float)
encoder_temp_output = encoder_temp(X_temp)
encoder_temp_output.shape


# In[ ]:


summary(encoder_temp,(1,1, 250, 250))


# In[ ]:


decoder_temp = Decoder()
summary(decoder_temp,encoder_temp_output.shape)


# In[ ]:


convnet_temp = Convnet(encoder_temp,decoder_temp)
summary(convnet_temp,(1,1,250, 250))


# In[ ]:


convnet_temp.eval()
X_temp = torch.zeros((1,1, 250, 250), dtype=torch.float).to(CFG.device)
convnet_temp(X_temp)


# ## 6.4 Training Part

# In[ ]:


get_ipython().system('pwd')


# In[ ]:


get_ipython().system('ls ')


# In[ ]:





# In[ ]:


# define training function
def train_model(model, train_dataset, val_dataset, test_dataset, device, model_name,
                lr=0.02, epochs=35, batch_size=32, weight_decay=0.9, gamma=0.9,
                patience=5, early_stop=3, verbose=False, save=True):
    model = model.to(device)

    # construct dataloader
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # history
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "test_acc":0}
    best_valid_acc = float("-inf")
    early_stop_step = 0
    best_epoch = 0

    # set up loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()  
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=patience, gamma=gamma)

    # Training Loop
    if verbose:
        print("Training Start:")
    for epoch in range(epochs):
        # shuffle training data in each epoch
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        # train
        train_loss = 0
        train_acc = 0
        model.train() 
        for i, (images, labels) in enumerate(train_loader):
            # move batch to device
            images = images.to(device)
            labels = labels.to(device)
            # forward
            outputs_list = model(images)
            enc_outputs = outputs_list[0].view(-1)
            outputs = outputs_list[1].view(-1)
            # pred
            pred = torch.sigmoid(outputs)
            pred = torch.round(pred)
            # loss
            cur_train_loss = criterion(outputs, labels)
            # acc
            cur_train_acc = (pred == labels).sum().item() / batch_size
            # backward
            cur_train_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # update total loss
            train_loss += cur_train_loss 
            train_acc += cur_train_acc
        
        # valid
        val_loss = 0
        val_acc = 0
        model.eval()
        with torch.no_grad():
            for images, labels in val_loader:
                # calculate validation loss
                images = images.to(device)
                labels = labels.to(device)
                outputs_list = model(images)
                enc_outputs = outputs_list[0].view(-1)
                outputs = outputs_list[1].view(-1)
                # loss
                cur_valid_loss = criterion(outputs, labels)
                val_loss += cur_valid_loss
                # acc
                pred = torch.sigmoid(outputs)
                pred = torch.round(pred)
                val_acc += (pred == labels).sum().item() / batch_size

        # update learning rate after every epoch
        scheduler.step()
        # print training feedback
        train_loss = train_loss / len(train_loader)
        train_acc = train_acc / len(train_loader)
        val_loss = val_loss / len(val_loader)
        val_acc = val_acc / len(val_loader)
        if verbose:
            print(f"Epoch:{epoch + 1} / {epochs}, lr: {optimizer.param_groups[0]['lr']:.5f} train loss:{train_loss:.5f}, train acc: {train_acc:.5f}, valid loss:{val_loss:.5f}, valid acc:{val_acc:.5f}")
        # update history
        history['train_loss'].append(train_loss.cpu().detach().numpy())
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss.cpu().detach().numpy())
        history['val_acc'].append(val_acc)
        # early stop
        if val_acc >= best_valid_acc:
            best_valid_acc = val_acc
            best_model = copy.deepcopy(model)
            early_stop_step = 0
            best_epoch = epoch
        else:
            early_stop_step += 1
            if early_stop_step >= early_stop:
                model = best_model
                break 
                
    # test
    test_acc = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs_list = model(images)
            enc_outputs = outputs_list[0]
            outputs = outputs_list[1]
            pred = torch.sigmoid(outputs)
            pred = torch.round(pred)
            test_acc += (pred == labels).sum().item()

    history["test_acc"] = test_acc / len(test_loader)
    print(f'Best Epoch: {best_epoch}, Test Accuracy: {history["test_acc"]}')

    model = model.to("cpu")
    if save:
        torch.save(model, model_name)
    
    return history


# In[ ]:


# plot loss and accuracy
def plot_training_curves(hist):
    # plot training curves
    epochs = range(1, len(hist['train_loss']) + 1)

    fig, ax = plt.subplots(1,2, figsize=(18,6))
    ax[0].plot(epochs, hist['train_loss'], 'r-', label='Train')
    ax[0].plot(epochs, hist['val_loss'], 'b-', label='Evaluation')
    ax[0].set_title('Loss')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss')
    ax[0].legend()

    ax[1].plot(epochs, hist['train_acc'], 'r-', label='Train')
    ax[1].plot(epochs, hist['val_acc'], 'b-', label='Evaluation')
    ax[1].set_title('Accuracy')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Acc')
    ax[1].legend()

    plt.show()


# ## 6.5 Original DS Training

# In[ ]:


result_set = {}
encoder = Encoder()
decoder = Decoder()
#baseline_model1 = Convnet(encoder,decoder)
baseline_model1 = torch.load("../input/baseline-model-data/model_baseline_model1.pth")

#=========注释训练=============
hist = train_model(baseline_model1, train_dataset, val_dataset, test_dataset, CFG.device, 
                   model_name='./model_baseline_model1.pth',
                   lr=0.002, epochs=50, batch_size=32, weight_decay=0.05, gamma=0.5,
                   patience=5, early_stop=10, verbose=True)

#=========导入文件=============
#hist = np.load('../input/baseline-model-data/hist.npy',allow_pickle=True)
#print(hist)
plot_training_curves(hist)

#result_set["selfModel_origData"] = hist


# ## 6.6 Aug DS Training

# In[ ]:


encoder = Encoder()
decoder = Decoder()
#baseline_model2_aug = Convnet(encoder,decoder)

baseline_model2_aug = torch.load('../input/baseline-model-data/model_baseline_model2.pth')

hist2 = train_model(baseline_model2_aug, train_dataset_fin, val_dataset, test_dataset, CFG.device, 
                   model_name='./model_baseline_model2.pth',
                   lr=0.002, epochs=50, batch_size=32, weight_decay=0.05, gamma=0.5,
                   patience=5, early_stop=10, verbose=True)


plot_training_curves(hist2)

#result_set["selfModel_ConcatData"] = hist2


# # 7. Contrastive Loss

# The next part:
# - https://www.kaggle.com/code/zhengxiaofancn/ai-medical-final-7d5a2d/notebook
