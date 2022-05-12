#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import cv2
import copy
from tqdm import tqdm
import PIL
import time

import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models


ROOT_FOLDER = '..'
DATA_FOLDER = os.path.join(ROOT_FOLDER,'input/mds-misis-dl-car-classificationn/')
OUTPUT_FOLDER = '/kaggle/working/'
IMG_SHAPE = (224, 224)
CHANNELS = 3
IMG_SIZE = (*IMG_SHAPE, CHANNELS)
BATCH_SIZE = 64
EPOCHS = 120
NUM_WORKERS = 2
LR = 0.001
TEST_SPLIT = 0.1
VAL_SPLIT = 0.3
NUM_CLASSES = 10

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
os.listdir(DATA_FOLDER)


# # 1. Load and Prepare Data

# In[ ]:


df = pd.read_csv(os.path.join(DATA_FOLDER, 'train.csv'))
df.head()


# In[ ]:


df['Category'].value_counts().sort_index().plot.bar()
plt.show()


# ## 1.1. Create image file path (img_fpath)

# In[ ]:


df['img_fpath'] = df.apply(lambda x: os.path.join(DATA_FOLDER, 'train', 'train', str(x['Category']), x['Id']), axis=1)
plt.imshow(PIL.Image.open(df.iloc[0]['img_fpath']))
plt.show()


# ## 1.2. Split Data

# In[ ]:


train_df, test_df = train_test_split(df,
                                     test_size=TEST_SPLIT,
                                     random_state=RANDOM_SEED,
                                     stratify=df[['Category']])
train_df, val_df = train_test_split(train_df,
                                    test_size=1/(1+(1-VAL_SPLIT-TEST_SPLIT)/VAL_SPLIT),
                                    random_state=RANDOM_SEED,
                                    stratify=train_df[['Category']])

print(
    "Train split:", train_df.size/df.size,
    "\nVal split:", val_df.size/df.size,
    "\nTest split:", test_df.size/df.size
)


# In[ ]:


plt.figure(figsize=(12, 8))
plt.hist(train_df['Category'].to_numpy(), bins=NUM_CLASSES, alpha=0.2, label='Train')
plt.hist(val_df['Category'].to_numpy(), bins=NUM_CLASSES, alpha=0.2, label='Validate')
plt.hist(test_df['Category'].to_numpy(), bins=NUM_CLASSES, alpha=0.2, label='Test')
plt.legend()
plt.show()


# ## 1.3. Create Datasets

# In[ ]:


class CarDataset(Dataset):
    def __init__(self, df, is_sub=False, transform=None):
        self.img_fpaths = df['img_fpath'].to_numpy()
        self.is_sub = is_sub
        if not self.is_sub:
            self.labels = df['Category'].to_numpy()
        self.transform = transform

    def __len__(self):
        return len(self.img_fpaths)

    def __getitem__(self, idx):
        image_filepath = self.img_fpaths[idx]
        image = cv2.imread(image_filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            image = self.transform(image=image)["image"]
            
        if not self.is_sub:
            label = int(self.labels[idx])
            return image, label
        
        return image


# ### 1.3.1. Train Dataset

# In[ ]:


train_transform = A.Compose(
    [
        A.LongestMaxSize(max_size=IMG_SHAPE[0]),
        
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=20, p=0.5),
        A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.CoarseDropout(5, 25, 25, p=0.2),
        
        A.PadIfNeeded(*IMG_SHAPE, border_mode=0),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)
train_dataset = CarDataset(df=train_df, transform=train_transform)


# ### 1.3.2. Val Dataset

# In[ ]:


val_transform = A.Compose(
    [
        A.LongestMaxSize(max_size=IMG_SHAPE[0]),
        
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        
        A.PadIfNeeded(*IMG_SHAPE, border_mode=0),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)
val_dataset = CarDataset(df=val_df, transform=val_transform)


# ### 1.3.3. Test Dataset

# In[ ]:


test_transform = A.Compose(
    [
        A.LongestMaxSize(max_size=IMG_SHAPE[0]),
        
        A.PadIfNeeded(*IMG_SHAPE, border_mode=0),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)
test_dataset = CarDataset(df=test_df, transform=test_transform)


# ### 1.3.4. Viz Train Dataset

# In[ ]:


def visualize_augmentations(dataset, idx=0, samples=10, cols=5):
    dataset = copy.deepcopy(dataset)
    dataset.transform = A.Compose([t for t in dataset.transform if not isinstance(t, (A.Normalize, ToTensorV2))])
    rows = samples // cols
    figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 6))
    for i in range(samples):
        image, _ = dataset[idx]
        ax.ravel()[i].imshow(image)
        ax.ravel()[i].set_axis_off()
    plt.tight_layout()
    plt.show()
    
visualize_augmentations(train_dataset)


# # 2. Model

# In[ ]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: '{device}'")


# ## 2.1. Prepare Datasets

# In[ ]:


dataloaders = {_phase: DataLoader(_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
              for _phase, _dataset in zip(['train', 'val', 'test'], [train_dataset, val_dataset, test_dataset])}
dataset_sizes = {_phase: len(_dataloder.dataset.labels)
              for _phase, _dataloder in dataloaders.items()}
dataset_sizes


# ## 2.2. Load Model

# In[ ]:


model = models.resnet50(pretrained=True)

# ResNet18/34/50
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, NUM_CLASSES)

# # Vgg16
# num_ftrs = model.classifier[6].in_features
# model.classifier[6] = nn.Linear(num_ftrs, NUM_CLASSES)

model = model.to(device)


# In[ ]:


criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.SGD(model.parameters(), lr=0.002, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=15, gamma=0.5, verbose=True)


# ## 2.3. Train

# In[ ]:


def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, num_epochs=EPOCHS):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    history = {_phase: {'loss': [], 'acc': []} for _phase in ['train', 'val']}
    best_acc = 0.0
   
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            # History
            history[phase]['loss'].append(round(float(epoch_loss), 4))
            history[phase]['acc'].append(round(float(epoch_acc), 4))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    
    return model, history


# In[ ]:


torch.cuda.empty_cache()
model, history = train_model(model, dataloaders, dataset_sizes, criterion, optimizer_ft, exp_lr_scheduler)


# In[ ]:


def history_plot(history):
    epochs = ['train', 'val']
    
    for metric in ['acc', 'loss']:
        
        for epoch in ['train', 'val']:
            plt.plot(history[epoch][metric])

        plt.title(f'model {metric}')
        plt.ylabel(metric)
        plt.xlabel('epoch')
        plt.legend(epochs, loc='upper left')
        plt.show()
        
history_plot(history)


# ## 2.4. Validate on Test

# In[ ]:


def calculate_preds(model, dataloader, is_sub=False):
    model.eval()
    
    all_preds, all_labels = [torch.Tensor(0), torch.Tensor(0)]

    with torch.no_grad():
        for data in tqdm(dataloader):
            if not is_sub:
                inputs, labels = data
                labels = labels.to(device)
            else:
                inputs = data
            
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds = torch.cat((all_preds, preds.cpu()))
            if not is_sub:
                all_labels = torch.cat((all_labels, labels.cpu()))
    
    return all_preds, all_labels


# In[ ]:


test_preds, test_labels = calculate_preds(model, dataloaders['test'])
test_acc = float(torch.sum(test_preds == test_labels)/test_labels.shape[0])
print(f'Test Accuracy: {test_acc:.4f}')


# # 3. Run on Test Submission

# In[ ]:


sub_img_folder = os.path.join(DATA_FOLDER, 'test', 'test_upload')
img_fnames = os.listdir(sub_img_folder)
img_fpaths = [os.path.join(sub_img_folder, img_fname) for img_fname in img_fnames]


# In[ ]:


sub_df = pd.DataFrame({'Id': img_fnames, 'img_fpath': img_fpaths})
sub_df.head()


# In[ ]:


sub_dataset = CarDataset(df=sub_df, is_sub=True, transform=test_transform)
sub_dataloader = DataLoader(sub_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)


# In[ ]:


sub_preds, _ = calculate_preds(model, sub_dataloader, is_sub=True)
sub_preds


# # 4. Save Results

# In[ ]:


sub_df['Category'] = sub_preds.detach().numpy().astype(int)
sub_df = sub_df.set_index('Id')
try:
    del sub_df['img_fpath']
except KeyError:
    pass
sub_df.head()


# In[ ]:


sub_df.to_csv(os.path.join(OUTPUT_FOLDER, 'submission.csv'))


# In[ ]:




