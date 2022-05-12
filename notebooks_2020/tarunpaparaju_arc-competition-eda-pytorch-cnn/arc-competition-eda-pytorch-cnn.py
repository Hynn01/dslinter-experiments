#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# <img src="https://i.imgur.com/x2XjjFV.jpg" width="250px">

# Welcome to the Abstraction and Reasoning Challenge (ARC), a potential major step towards achieving artificial general intelligence (AGI). In this competition, we are challenged to build an algorithm that can perform reasoning tasks it has never seen before. Classic machine learning problems generally involve one specific task which can be solved by training on millions of data samples. But in this challenge, we need to build an algorithm that can learn patterns from a minimal number of examples.
# 
# In this notebook, I will be demonstrating how one can use **data augmentation** and **supervised machine learning** to build a baseline model to solve this problem.
# 
# <font color="red" size=3>Please upvote this kernel if you like it. It motivates me to produce more quality content :)</font>

# # Contents
# 
# * [<font size=4>Preparing the ground</font>](#preparing-the-ground)
#     * [Import libraries and define hyperparameters](#import-libraries-and-define-hyperparameters)
#     * [Load the ARC data](#load-the-arc-data)
#     
# 
# * [<font size=4>Basic exploration</font>](#basic-exploration)
#     * [Look at few train/test input/output pairs](#look-at-few)
#     * [Number frequency](#number-frequency)
#     * [Matrix mean values](#matrix-mean-values)
#     * [Matrix heights](#matrix-heights)
#     * [Matrix widths](#matrix-widths)
#     * [Height vs. Width](#height-vs-width)
#     
# 
# * [<font size=4>My approach</font>](#my-approach)
#     * [Data processing](#data-processing)
#     * [Modeling](#modeling)
# 
# 
# * [<font size=4>Training and postprocessing</font>](#training-and-postprocessing)
#     * [Loss (MSE)](#loss)
#     * [Backpropagation and optimization (Adam)](#backprop)
# 
# 
# * [<font size=4>Submission</font>](#submission)
# 
# 
# * [<font size=4>Ending note</font>](#ending-note)

# # Preparing the ground <a id="preparing-the-ground"></a>

# ## Import libraries and define hyperparameters <a id="import-libraries-and-define-hyperparameters"></a> 

# In[ ]:


import os
import gc
import cv2
import json
import time

import numpy as np
import pandas as pd
from pathlib import Path
from keras.utils import to_categorical

import seaborn as sns
import plotly.express as px
from matplotlib import colors
import matplotlib.pyplot as plt
import plotly.figure_factory as ff

import torch
T = torch.Tensor
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader


# In[ ]:


SIZE = 1000
EPOCHS = 50
CONV_OUT_1 = 50
CONV_OUT_2 = 100
BATCH_SIZE = 128

TEST_PATH = Path('../input/abstraction-and-reasoning-challenge/')
SUBMISSION_PATH = Path('../input/abstraction-and-reasoning-challenge/')

TEST_PATH = TEST_PATH / 'test'
SUBMISSION_PATH = SUBMISSION_PATH / 'sample_submission.csv'


# ## Load the ARC data <a id="load-the-arc-data"></a>

# ### Get testing tasks

# In[ ]:


test_task_files = sorted(os.listdir(TEST_PATH))

test_tasks = []
for task_file in test_task_files:
    with open(str(TEST_PATH / task_file), 'r') as f:
        task = json.load(f)
        test_tasks.append(task)


# ### Extract training and testing data

# In[ ]:


Xs_test, Xs_train, ys_train = [], [], []

for task in test_tasks:
    X_test, X_train, y_train = [], [], []

    for pair in task["test"]:
        X_test.append(pair["input"])

    for pair in task["train"]:
        X_train.append(pair["input"])
        y_train.append(pair["output"])
    
    Xs_test.append(X_test)
    Xs_train.append(X_train)
    ys_train.append(y_train)


# In[ ]:


matrices = []
for X_test in Xs_test:
    for X in X_test:
        matrices.append(X)
        
values = []
for matrix in matrices:
    for row in matrix:
        for value in row:
            values.append(value)
            
df = pd.DataFrame(values)
df.columns = ["values"]


# # Basic exploration <a id="basic-exploration"></a>

# ## Look at a few train/test input/output pairs <a id="look-at-few"></a>
# 
# These are some of the pairs present in the training data. I use functions from Walter's excellent starter kernel to plot these pairs.

# In[ ]:


data_path = Path('/kaggle/input/abstraction-and-reasoning-challenge/')
training_path = data_path / 'training'
training_tasks = sorted(os.listdir(training_path))

for i in [1, 19, 8, 15, 9]:

    task_file = str(training_path / training_tasks[i])

    with open(task_file, 'r') as f:
        task = json.load(f)

    def plot_task(task):
        """
        Plots the first train and test pairs of a specified task,
        using same color scheme as the ARC app
        """
        cmap = colors.ListedColormap(
            ['#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00',
             '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
        norm = colors.Normalize(vmin=0, vmax=9)
        fig, ax = plt.subplots(1, 4, figsize=(15,15))
        ax[0].imshow(task['train'][0]['input'], cmap=cmap, norm=norm)
        width = np.shape(task['train'][0]['input'])[1]
        height = np.shape(task['train'][0]['input'])[0]
        ax[0].set_xticks(np.arange(0,width))
        ax[0].set_yticks(np.arange(0,height))
        ax[0].set_xticklabels([])
        ax[0].set_yticklabels([])
        ax[0].tick_params(length=0)
        ax[0].grid(True)
        ax[0].set_title('Train Input')
        ax[1].imshow(task['train'][0]['output'], cmap=cmap, norm=norm)
        width = np.shape(task['train'][0]['output'])[1]
        height = np.shape(task['train'][0]['output'])[0]
        ax[1].set_xticks(np.arange(0,width))
        ax[1].set_yticks(np.arange(0,height))
        ax[1].set_xticklabels([])
        ax[1].set_yticklabels([])
        ax[1].tick_params(length=0)
        ax[1].grid(True)
        ax[1].set_title('Train Output')
        ax[2].imshow(task['test'][0]['input'], cmap=cmap, norm=norm)
        width = np.shape(task['test'][0]['input'])[1]
        height = np.shape(task['test'][0]['input'])[0]
        ax[2].set_xticks(np.arange(0,width))
        ax[2].set_yticks(np.arange(0,height))
        ax[2].set_xticklabels([])
        ax[2].set_yticklabels([])
        ax[2].tick_params(length=0)
        ax[2].grid(True)
        ax[2].set_title('Test Input')
        ax[3].imshow(task['test'][0]['output'], cmap=cmap, norm=norm)
        width = np.shape(task['test'][0]['output'])[1]
        height = np.shape(task['test'][0]['output'])[0]
        ax[3].set_xticks(np.arange(0,width))
        ax[3].set_yticks(np.arange(0,height))
        ax[3].set_xticklabels([])
        ax[3].set_yticklabels([])
        ax[3].tick_params(length=0)
        ax[3].grid(True)
        ax[3].set_title('Test Output')
        plt.tight_layout()
        plt.show()

    plot_task(task)


# ## Number frequency <a id="number-frequency"></a>

# In[ ]:


px.histogram(df, x="values", title="Numbers present in matrices")


# From the above graph, we can clearly see that the number distribution has a string positive skew. Most numbers in the matrices are clearly 0. This is reflected by the dominance of black color in most matrices.

# ## Matrix mean values <a id="matrix-mean-values"></a>

# In[ ]:


means = [np.mean(X) for X in matrices]
fig = ff.create_distplot([means], group_labels=["Means"], colors=["green"])
fig.update_layout(title_text="Distribution of matrix mean values")


# From the above graph, we can see that lower means are more common than higher means. The graph, once again, has a strong positive skew. This is further proof that black is the most dominant color in the matrices.

# ## Matrix heights <a id="matrix-heights"></a>

# In[ ]:


heights = [np.shape(matrix)[0] for matrix in matrices]
widths = [np.shape(matrix)[1] for matrix in matrices]


# In[ ]:


fig = ff.create_distplot([heights], group_labels=["Height"], colors=["magenta"])
fig.update_layout(title_text="Distribution of matrix heights")


# From the above graph, we can see that matrix heights have a much more uniform distribution (with significantly less skew). The distribution is somewhat normal with a mean of approximately 15.

# ## Matrix widths <a id="matrix-widths"></a>

# In[ ]:


fig = ff.create_distplot([widths], group_labels=["Width"], colors=["red"])
fig.update_layout(title_text="Distribution of matrix widths")


# From the above graph, we can see that matrix widths also have a uniform distribution (with significantly less skew). The distribution is also somewhat uniform with a mean of approximately 16.

# ## Height vs. Width <a id="height-vs-width"></a>

# In[ ]:


plot = sns.jointplot(widths, heights, kind="kde", color="blueviolet")
plot.set_axis_labels("Width", "Height", fontsize=14)
plt.show(plot)


# In[ ]:


plot = sns.jointplot(widths, heights, kind="reg", color="blueviolet")
plot.set_axis_labels("Width", "Height", fontsize=14)
plt.show(plot)


# From the above graphs, we can see that heights and widths have a strong positive correlation, *i.e.* greater widths generally result in greater heights. This is consistent with the fact that most matrices are square-shaped.

# # My approach <a id="my-approach"></a>

# My approach to this problem imvolves simple data augmentation techniques and a supervised 2D CNN model to make predictions. The model takes a 2D matrix as input and outputs the softmax probabilities of different values occuring in the output matrix. But since we have only few training examples for each task, I create new input-output pairs by randomly switching colors. The extra augmented data helps the model capture patterns more easily.

# <img src="https://i.imgur.com/ott07Lh.png" width="750px">

# <img src="https://i.imgur.com/H96WieH.png" width="300px">

# It can be seen from the above diagram that the same training pairs are augmented (100s of times) to produce a large dataset. This dataset is used to train the CNN for each task. The CNN predicts a probability distribution over the "pixels" or values in the matrix. This probability distribution is used to generate the final output matrix.
# 
# The trained CNN model can be used to make predictions on the test samples as follows:

# <img src="https://i.imgur.com/I3n0Q2k.png" width="750px">

# ## Data processing <a id="data-processing"></a>

# <img src="https://i.imgur.com/pa9C1rz.png" width="400px">

# The basic steps in my data processing pipeline are given above. These steps can be summarized as:
# 
# 1. **Handle matrix (input and output) dimensions:** Ensure consistent dimensions among inputs and outputs
# 2. **Randomly augment input and output matrices:** Mutate the matrix values in order to generate new data for each task
# 3. **Return input-output pairs along with dimension information:** Return the X-y data with dimensions

# ### Helper functions

# In[ ]:


def replace_values(a, d):
    return np.array([d.get(i, -1) for i in range(a.min(), a.max() + 1)])[a - a.min()]

def repeat_matrix(a):
    return np.concatenate([a]*((SIZE // len(a)) + 1))[:SIZE]

def get_new_matrix(X):
    if len(set([np.array(x).shape for x in X])) > 1:
        X = np.array([X[0]])
    return X

def get_outp(outp, dictionary=None, replace=True):
    if replace:
        outp = replace_values(outp, dictionary)

    outp_matrix_dims = outp.shape
    outp_probs_len = outp.shape[0]*outp.shape[1]*10
    outp = to_categorical(outp.flatten(),
                          num_classes=10).flatten()

    return outp, outp_probs_len, outp_matrix_dims


# ### PyTorch DataLoader

# In[ ]:


class ARCDataset(Dataset):
    def __init__(self, X, y, stage="train"):
        self.X = get_new_matrix(X)
        self.X = repeat_matrix(self.X)
        
        self.stage = stage
        if self.stage == "train":
            self.y = get_new_matrix(y)
            self.y = repeat_matrix(self.y)
        
    def __len__(self):
        return SIZE
    
    def __getitem__(self, idx):
        inp = self.X[idx]
        if self.stage == "train":
            outp = self.y[idx]

        if idx != 0:
            rep = np.arange(10)
            orig = np.arange(10)
            np.random.shuffle(rep)
            dictionary = dict(zip(orig, rep))
            inp = replace_values(inp, dictionary)
            if self.stage == "train":
                outp, outp_probs_len, outp_matrix_dims = get_outp(outp, dictionary)
                
        if idx == 0:
            if self.stage == "train":
                outp, outp_probs_len, outp_matrix_dims = get_outp(outp, None, False)
        
        return inp, outp, outp_probs_len, outp_matrix_dims, self.y


# # Modeling <a id="modeling"></a>

# <img src="https://i.imgur.com/cpUtXRR.png" width="600px">

# 
# 
# I use a basic CNN model that takes 2D input and returns 2D output. The sequential architecture is follows:
# 
# 1. (Conv2D + ReLU) **x** 2
# 2. MaxPool **x** 2
# 3. Dense
# 4. Softmax
# 
# The softmax probabilities are converted to the final 2D matrix through argmax and resize functions.

# ### PyTorch CNN model

# In[ ]:


class BasicCNNModel(nn.Module):
    def __init__(self, inp_dim=(10, 10), outp_dim=(10, 10)):
        super(BasicCNNModel, self).__init__()
        
        CONV_IN = 3
        KERNEL_SIZE = 3
        DENSE_IN = CONV_OUT_2
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.dense_1 = nn.Linear(DENSE_IN, outp_dim[0]*outp_dim[1]*10)
        
        if inp_dim[0] < 5 or inp_dim[1] < 5:
            KERNEL_SIZE = 1

        self.conv2d_1 = nn.Conv2d(CONV_IN, CONV_OUT_1, kernel_size=KERNEL_SIZE)
        self.conv2d_2 = nn.Conv2d(CONV_OUT_1, CONV_OUT_2, kernel_size=KERNEL_SIZE)

    def forward(self, x, outp_dim):
        x = torch.cat([x.unsqueeze(0)]*3)
        x = x.permute((1, 0, 2, 3)).float()
        self.conv2d_1.in_features = x.shape[1]
        conv_1_out = self.relu(self.conv2d_1(x))
        self.conv2d_2.in_features = conv_1_out.shape[1]
        conv_2_out = self.relu(self.conv2d_2(conv_1_out))
        
        self.dense_1.out_features = outp_dim
        feature_vector, _ = torch.max(conv_2_out, 2)
        feature_vector, _ = torch.max(feature_vector, 2)
        logit_outputs = self.dense_1(feature_vector)
        
        out = []
        for idx in range(logit_outputs.shape[1]//10):
            out.append(self.softmax(logit_outputs[:, idx*10: (idx+1)*10]))
        return torch.cat(out, axis=1)


# # Training and postprocessing <a id="training-and-postprocessing"></a>

# I train the model using PyTorch's autograd functionality. Specifically, I use the **Adam** optimizer and the **MSE** loss function.

# ## Loss (MSE) <a id="loss"></a>

# ### The idea behind a loss function

# <img src="https://i.imgur.com/WeQbG9M.png" width="275px">

# As shown above, the target vector *t* and the output vector *o* are diverging from each other. The loss function measures the degree to which these two diverge, *i.e.* the size of *o - t*. Here, *t* is the actual pixel probability and *o* is the predicted pixel probability. The mean squared error calculates the average squared error between *o* and *t*. 
# 
# In the code, the line <code>train_loss = nn.MSELoss()(train_preds, train_y)</code> calculates the MSE loss.

# ## Backpropagation and optimization (Adam) <a id="backprop"></a>

# <img src="https://i.imgur.com/IEeg94y.png" width="550px">

# In the above diagram, we can see that Newton's Chain rule is used to calculate the gradient of the loss function *w.r.t.* the parameters in the model.

# <img src="https://i.imgur.com/yEVIBzj.png" width="350px">

# The general update equation above is used to optimize the parameters using the gradients calculated above. Note that more complex algorithms like Adam use more complex update equations than the ones specified above.
# 
# In the code, the line <code>train_loss.backward()</code> and <code>optimizer.step()</code> perform backpropagation and optimization respectively.

# ### Helper functions

# In[ ]:


def transform_dim(inp_dim, outp_dim, test_dim):
    return (test_dim[0]*outp_dim[0]/inp_dim[0],
            test_dim[1]*outp_dim[1]/inp_dim[1])

def resize(x, test_dim, inp_dim):
    if inp_dim == test_dim:
        return x
    else:
        return cv2.resize(flt(x), inp_dim,
                          interpolation=cv2.INTER_AREA)

def flt(x): return np.float32(x)
def npy(x): return x.cpu().detach().numpy()
def itg(x): return np.int32(np.round(x))


# ### Train the model and postprocess probabilties

# <img src="https://i.imgur.com/rD53yoI.png" width="350px">

# In my postprocessing, I follow the steps given below:
# 
# 1. **Get output probabilites from the CNN model:** 
# 
# <code>npy(network.forward(T(X).unsqueeze(0), out_d))</code>
# 2. **Perform argmax on probabilities to get indices of maximum prbabilities:**
# 
# <code>np.argmax(test_preds.reshape((10, *outp_dim)), axis=0)</code>
# 3. **Resize the output matrix to match the dimension ratios and round off:**
# 
# <code>itg(resize(test_preds, np.shape(test_preds), tuple(itg(transform_dim(inp_dim, outp_dim, test_dim))))))</code>

# ### Train the CNN model on loop

# In[ ]:


idx = 0
start = time.time()
test_predictions = []

for X_train, y_train in zip(Xs_train, ys_train):
    print("TASK " + str(idx + 1))

    train_set = ARCDataset(X_train, y_train, stage="train")
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)

    inp_dim = np.array(X_train[0]).shape
    outp_dim = np.array(y_train[0]).shape
    network = BasicCNNModel(inp_dim, outp_dim).cuda()
    optimizer = Adam(network.parameters(), lr=0.01)
    
    for epoch in range(EPOCHS):
        for train_batch in train_loader:
            train_X, train_y, out_d, d, out = train_batch
            train_preds = network.forward(train_X.cuda(), out_d.cuda())
            train_loss = nn.MSELoss()(train_preds, train_y.cuda())
            
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

    end = time.time()        
    print("Train loss: " + str(np.round(train_loss.item(), 3)) + "   " +          "Total time: " + str(np.round(end - start, 1)) + " s" + "\n")
    
    X_test = np.array([resize(flt(X), np.shape(X), inp_dim) for X in Xs_test[idx-1]])
    for X in X_test:
        test_dim = np.array(T(X)).shape
        test_preds = npy(network.forward(T(X).unsqueeze(0).cuda(), out_d.cuda()))
        test_preds = np.argmax(test_preds.reshape((10, *outp_dim)), axis=0)
        test_predictions.append(itg(resize(test_preds, np.shape(test_preds),
                                           tuple(itg(transform_dim(inp_dim,
                                                                   outp_dim,
                                                                   test_dim))))))
    idx += 1


# # Submission <a id="submission"></a>

# ### Define function to flatten submission matrices

# In[ ]:


def flattener(pred):
    str_pred = str([row for row in pred])
    str_pred = str_pred.replace(', ', '')
    str_pred = str_pred.replace('[[', '|')
    str_pred = str_pred.replace('][', '|')
    str_pred = str_pred.replace(']]', '|')
    return str_pred


# ### Prepare submission dataframe

# In[ ]:


test_predictions = [[list(pred) for pred in test_pred] for test_pred in test_predictions]

for idx, pred in enumerate(test_predictions):
    test_predictions[idx] = flattener(pred)
    
submission = pd.read_csv(SUBMISSION_PATH)
submission["output"] = test_predictions


# ### Convert submission to .csv format

# In[ ]:


submission.head()


# In[ ]:


submission.to_csv("submission.csv", index=False)


# # Ending note <a id="ending-note"></a>
# 
# <font color="red" size=4>Please upvote this kernel if you like it. It motivates me to produce more quality content :)</font>
