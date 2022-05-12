#!/usr/bin/env python
# coding: utf-8

# # Group Kfold is added to this notebook
# The original nootbook is [here](https://www.kaggle.com/code/ahmetcelik158/tps-apr-22-lstm-with-pytorch).
# 
# 1. Use nn.Sigmoid() as the last layer (instead of nn.Linear()) to map the prediction scores to the range from 0 to 1.
# 2. Use nn.BCELoss() as loss (instead of nn.MSELoss())
# 
# # TPS April 2022 - Time Series Classification
# 
# In Kaggle's TPS April 2022 competition, we are challenged a time series classification problem. The dataset contains biological sensor data recorded from different participants. Each observation is a sixty second recordings from 13 sensors which has a state as either 0 or 1. While the train set has nearly 26.000 sequences, we will be classifying nearly 12.000 sequence of test set.
# 
# I am going to implement RNN with LSTM layers for this problem. This will be my first experience with RNN. There were great kernels that helped me to understand LSTM, i mentioned them in references \[1-4\]. The structure of the notebook will be as follows:
# 
# **Index**
# 
# 1. [Data Preparation](#1.-Data-Preparation)
# 
# 2. [Model Definition](#2.-Model-Definition)
# 
# 3. [Function Definitions - Validation, Training and Prediction](#3.-Function-Definitions---Validation,-Training-and-Prediction)
# 
# 4. [Training the Model](#4.-Training-the-Model)
# 
# 5. [Prediction](#5.-Prediction)
# 
# 6. [References](#6.-References)

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
get_ipython().run_line_magic('matplotlib', 'inline')

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

train = pd.read_csv("/kaggle/input/tabular-playground-series-apr-2022/train.csv")
test = pd.read_csv("/kaggle/input/tabular-playground-series-apr-2022/test.csv")
train_labels = pd.read_csv("/kaggle/input/tabular-playground-series-apr-2022/train_labels.csv")


# In[ ]:


def reduce_mem_usage(df, verbose=True):
    numerics = ['int8','int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2

    for col in df.columns:
        col_type = df[col].dtypes

        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()

            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose:
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
 
    return df


# In[ ]:


train = reduce_mem_usage(train)
test = reduce_mem_usage(test)


# In[ ]:


groups = train["sequence"]
y = train_labels.state
train = train.set_index(["sequence", "subject", "step"])
test = test.set_index(["sequence", "subject", "step"])


# # 1. Data Preparation
# 
# In this section, I will
# * check missing values
# * add new features
# * scale data
# * reshape data
# * create tensors and dataloaders
# 
# 
# I created first order lag, difference, rolling mean, rolling std, rolling min and rolling max for new features. While plotting some sensors in time domain, I saw instant value drops at the last second (for example sequence=0, sensor=12). I thought that these sudden drops may be noises while closing the sensor and I should not trust beginning and ending of the sensor data. By dropping NaN values after using a centered window with size 5 in add_features function, I have automatically dropped the first and last 2 seconds. As a result, I will be using 56 seconds of data to train my model.

# In[ ]:


print("Checking if there are any missing values:")
print("Train: {}".format(train.isnull().sum().sum()))
print("Test: {}".format(test.isnull().sum().sum()))


# In[ ]:


def add_features(df, features):
    for feature in features:
        df_grouped = df.groupby("sequence")[feature]
        df_rolling = df_grouped.rolling(5, center=True)
        
        df[feature + "_lag1"] = df_grouped.shift(1)
        df[feature + "_diff1"] = df[feature] - df[feature + "_lag1"]
        df[feature + "_roll_mean"] = df_rolling.mean().reset_index(0, drop=True)
        df[feature + "_roll_std"] = df_rolling.std().reset_index(0, drop=True)
        df[feature + "_roll_min"] = df_rolling.min().reset_index(0, drop=True)
        df[feature + "_roll_max"] = df_rolling.max().reset_index(0, drop=True)
    df.dropna(axis=0, inplace=True)
    return

features = ["sensor_{:02d}".format(i) for i in range(13)]
add_features(train, features)
add_features(test, features)
train.head()


# In[ ]:


input_size = train.shape[1]
sequence_length = len(train.index.get_level_values(2).unique())

# Scaling test and train
scaler = StandardScaler()
train = scaler.fit_transform(train)
test = scaler.transform(test)

# Reshaping:
train = train.reshape(-1, sequence_length, input_size)
test = test.reshape(-1, sequence_length, input_size)
print("After Reshape")
print("Shape of training set: {}".format(train.shape))
print("Shape of test set: {}".format(test.shape))


# # 2. Model Definition
# 
# In this section, I will define my RNN model including LSTM layers. For LSTM layers, input and output shapes will be as follows:
# 
# * Input: (batch_size, sequence_length, input_size)
# * Output: (batch_size, sequence_length, D * hidden_size)
# 
# where D is 2 for bidirectional LSTM, otherwise 1. For details, check references for PyTorch LSTM documentation \[5\].

# In[ ]:


# Definition of a RNN Model class
class RNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, seq_len, dropout=0.4, output_size=1):
        super(RNN, self).__init__()
        
        # LSTM Layers
        self.lstm_1 = nn.LSTM(input_size, hidden_sizes[0], num_layers=2,
                            batch_first=True, bidirectional=True, dropout=dropout)
        self.lstm_21 = nn.LSTM(2*hidden_sizes[0], hidden_sizes[1], num_layers=2,
                             batch_first=True, bidirectional=True, dropout=dropout)
        self.lstm_22 = nn.LSTM(input_size, hidden_sizes[1], num_layers=2,
                             batch_first=True, bidirectional=True, dropout=dropout)
        self.lstm_31 = nn.LSTM(2*hidden_sizes[1], hidden_sizes[2], num_layers=2,
                             batch_first=True, bidirectional=True, dropout=dropout)
        self.lstm_32 = nn.LSTM(4*hidden_sizes[1], hidden_sizes[2], num_layers=2,
                             batch_first=True, bidirectional=True, dropout=dropout)
        self.lstm_41 = nn.LSTM(2*hidden_sizes[2], hidden_sizes[3], num_layers=2,
                             batch_first=True, bidirectional=True, dropout=dropout)
        self.lstm_42 = nn.LSTM(4*hidden_sizes[2], hidden_sizes[3], num_layers=2,
                             batch_first=True, bidirectional=True, dropout=dropout)
        hidd = 2*hidden_sizes[0] + 4*(hidden_sizes[1]+hidden_sizes[2]+hidden_sizes[3])
        self.lstm_5 = nn.LSTM(hidd, hidden_sizes[4], num_layers=2,
                             batch_first=True, bidirectional=True, dropout=dropout)
        
        # Fully Connected Layer
        self.fc = nn.Sequential(nn.Linear(2*hidden_sizes[4]*seq_len, 4096),
                                nn.ReLU(inplace=True),
                                nn.Dropout(p=dropout),
                                nn.Linear(4096, 1024),
                                nn.ReLU(inplace=True),
                                nn.Dropout(p=dropout),
                                nn.Linear(1024, output_size),
                                nn.Sigmoid()
                               )
        
    def forward(self, x):
        # lstm layers:
        x1, _ = self.lstm_1(x)
        
        x_x1, _ = self.lstm_21(x1)
        x_x2, _ = self.lstm_22(x)
        x2 = torch.cat([x_x1, x_x2], dim=2)
        
        x_x1, _ = self.lstm_31(x_x1)
        x_x2, _ = self.lstm_32(x2)
        x3 = torch.cat([x_x1, x_x2], dim=2)
        
        x_x1, _ = self.lstm_41(x_x1)
        x_x2, _ = self.lstm_42(x3)
        x4 = torch.cat([x_x1, x_x2], dim=2)
        x = torch.cat([x1, x2, x3, x4], dim=2)
        x, _ = self.lstm_5(x)
        
        # fully connected layers:
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x


# # 3. Function Definitions - Validation, Training and Prediction

# In[ ]:


### VALIDATION FUNCTION
def validation(model, loader, criterion, device="cpu"):
    model.eval()
    loss = 0
    preds_all = torch.LongTensor()
    labels_all = torch.LongTensor()
    
    with torch.no_grad():
        for batch_x, labels in loader:
            labels_all = torch.cat((labels_all, labels), dim=0)
            batch_x, labels = batch_x.to(device), labels.to(device)
            labels = labels.unsqueeze(1).float()
            
            output = model.forward(batch_x)
            loss += criterion(output,labels).item()
            preds_all = torch.cat((preds_all, output.to("cpu")), dim=0)
    total_loss = loss/len(loader)
    auc_score = roc_auc_score(labels_all, preds_all)
    return total_loss, auc_score


# In[ ]:


### TRAINING FUNCTION
def train_model(model, trainloader, validloader, criterion, optimizer, 
                scheduler, epochs=20, device="cpu", print_every=1):
    model.to(device)
    best_auc = 0
    best_epoch = 0
    for e in range(epochs):
        model.train()
        
        for batch_x, labels in trainloader:
            batch_x, labels = batch_x.to(device), labels.to(device)
            labels = labels.unsqueeze(1).float()
            
            # Training 
            optimizer.zero_grad()
            output = model.forward(batch_x)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
        # at the end of each epoch calculate loss and auc score:
        model.eval()
        train_loss, train_auc = validation(model, trainloader, criterion, device)
        valid_loss, valid_auc = validation(model, validloader, criterion, device)
        if valid_auc > best_auc:
            best_auc = valid_auc
            best_epoch = e
            torch.save(model.state_dict(), "best-state.pt")
        if e % print_every == 0:
            to_print = "Epoch: "+str(e+1)+" of "+str(epochs)
            to_print += ".. Train Loss: {:.4f}".format(train_loss)
            to_print += ".. Valid Loss: {:.4f}".format(valid_loss)
            to_print += ".. Valid AUC: {:.3f}".format(valid_auc)
            print(to_print)
    # After Training:
    model.load_state_dict(torch.load("best-state.pt"))
    to_print = "\nTraining completed. Best state dict is loaded.\n"
    to_print += "Best Valid AUC is: {:.4f} after {} epochs".format(best_auc,best_epoch+1)
    print(to_print)
    return


# In[ ]:


### PREDICTION FUNCTION
def prediction(model, loader, device="cpu"):
    model.to(device)
    model.eval()
    preds_all = torch.LongTensor()
    
    with torch.no_grad():
        for batch_x in loader:
            batch_x = batch_x.to(device)
            
            output = model.forward(batch_x).to("cpu")
            preds_all = torch.cat((preds_all, output), dim=0)
    return preds_all


# # 4. Training the Model
# 
# In this section, I will
# * initiate RNN model
# * define criterion, optimizer and schedular
# * train the model
# 
# I selected OneCycleLR as a schedular to adjust learning rate. With OneCycleLR, learning rate will increase at the beginning until max learning rate, and then start decreasing. I used pct_start = 0.2, so learning rate will reach to max value at the 0.2 of the learning phase. You can check references \[6,7\] for more details.

# In[ ]:


test_tensor = torch.tensor(test).float()
dataloaders_test = DataLoader(test_tensor, batch_size=32)


# In[ ]:


hidden_sizes = [288, 192, 144, 96, 32]
max_learning_rate = 0.001
epochs = 5

# Model
model_lstm = RNN(input_size, hidden_sizes, sequence_length)
print("Model: ")
print(model_lstm)

# criterion, optimizer, scheduler
criterion = nn.BCELoss()
optimizer = optim.Adam(model_lstm.parameters(), lr=max_learning_rate)


# In[ ]:


# Checking if GPU is available
if torch.cuda.is_available():
    my_device = "cuda"
    print("GPU is enabled")
else:
    my_device = "cpu"
    print("No GPU :(")


# In[ ]:


gkf = GroupKFold(n_splits=10)
for fold, (train_idx, valid_idx) in enumerate(gkf.split(train, y, groups.unique())):
    print(f"** fold: {fold+1} ** ........training ...... \n")
    
    X_train, X_valid = train[train_idx], train[valid_idx]
    y_train, y_valid = y[train_idx], y[valid_idx]
    
    train_X_tensor = torch.tensor(X_train).float()
    val_X_tensor = torch.tensor(X_valid).float()

    # Converting train and validation labels into tensors
    train_y_tensor = torch.tensor(y_train.values)
    val_y_tensor = torch.tensor(y_valid.values)

    # Creating train and validation tensors
    train_tensor = TensorDataset(train_X_tensor, train_y_tensor)
    val_tensor = TensorDataset(val_X_tensor, val_y_tensor)

    # Defining the dataloaders
    dataloaders_train = DataLoader(train_tensor, batch_size=64, shuffle=True)
    dataloaders_val = DataLoader(val_tensor, batch_size=32)
    
    # Scheduler
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer,
                                          max_lr = max_learning_rate,
                                          epochs = epochs,
                                          steps_per_epoch = len(dataloaders_train),
                                          pct_start = 0.2,
                                          anneal_strategy = "cos")
    
    # Training
    train_model(model = model_lstm,
                trainloader = dataloaders_train,
                validloader = dataloaders_val,
                criterion = criterion,
                optimizer = optimizer,
                scheduler = scheduler,
                epochs = epochs,
                device = my_device,
                print_every = 2)


# # 5. Prediction
# 
# In this section, I will
# * predict the test dataset
# * submit my results

# In[ ]:


y_pred = prediction(model_lstm, dataloaders_test, device=my_device)
print("Prediction completed, first 5 states:")
y_pred[:5]


# In[ ]:


submission = pd.read_csv("/kaggle/input/tabular-playground-series-apr-2022/sample_submission.csv")
submission.state = y_pred
submission.head()


# In[ ]:


submission.to_csv('submission.csv', index=False)
print("Resuls are saved to submission.csv")


# # 6. References
# 
# 1. [Tps April Tensorflow Bi-LSTM by @hamzaghanmi](https://www.kaggle.com/code/hamzaghanmi/tps-april-tensorflow-bi-lstm)
# 2. [TPSApr22 - FE + Pseudo Labels + Bi-LSTM by @hasanbasriakcay](https://www.kaggle.com/code/hasanbasriakcay/tpsapr22-fe-pseudo-labels-bi-lstm)
# 3. [[TPS-Apr][PyTorch] Bidirectional LSTM by @lordozvlad](https://www.kaggle.com/code/lordozvlad/tps-apr-pytorch-bidirectional-lstm)
# 4. [TPS Apr22 - EDA / FE + LSTM Tutorial by @javigallego](https://www.kaggle.com/code/javigallego/tps-apr22-eda-fe-lstm-tutorial)
# 5. [PyTorch LSTM documentation](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html)
# 6. [PyTorch documentation for adjusting learning rate](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate)
# 7. [Guide to Pytorch Learning Rate Scheduling by @isbhargav](https://www.kaggle.com/code/isbhargav/guide-to-pytorch-learning-rate-scheduling/notebook)
