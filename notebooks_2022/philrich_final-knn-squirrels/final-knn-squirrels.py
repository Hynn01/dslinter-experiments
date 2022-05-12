#!/usr/bin/env python
# coding: utf-8

# # Final solution to Squirrel color prediction

# After numerous failed experiments struggling to find any signal in the data, the variables are reduced to the coordinates, these variables appear to have some signal given that there are clusters of squirrels that are located in similar positions along the park. This can be better spotted in the plot shared by @gonzalorecioc.
# 
# <img src= "https://i.gyazo.com/da06daef28aa9b4544e85543ba44de0f.png" style='width: 500px;'>
# 
# However, given how skewed the data is towards Grey squirrels, about 83% of the dataset, the challenge is whether we can still find a method using this insight that beats the random baseline of predicting every single squirrel as Grey.

# ## Set up

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from tqdm import tqdm

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


path_train = "/kaggle/input/central-park-squirrel-color/train.csv"
path_test = "/kaggle/input/central-park-squirrel-color/test.csv"

train_df = pd.read_csv(path_train)
test_df = pd.read_csv(path_test)

depend_var = "Fur Color"
model_cols = ["X", "Y"]


# ## Model Training
# 
# - We use a K Nearest Neighbors algorithm to find the most obvious clusters of squirrels based on the coordinates
# - Given the few black squirrels, which also seem the most condensed on the plot, the data is only split in train and test (60/40)
# - To get more stable results the predictions are averaged across different datasets randomly split in train/test

# In[ ]:


main_class = "Gray"
baselines = 0
k_nn = np.arange(3,50,2)
total_seeds = 5
accuracies = np.zeros((total_seeds,len(k_nn)))
b_squirr = np.zeros((total_seeds,len(k_nn))) 
c_squirr = np.zeros((total_seeds,len(k_nn)))

for n_iter in tqdm(range(total_seeds)):
    X_train, X_test, y_train, y_test = train_test_split(train_df[model_cols], train_df[depend_var], test_size=0.4,
                                                        random_state=n_iter)
    baselines += (y_test == main_class).mean()
    
    for k in k_nn:
        knn = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        
        accuracies[n_iter, np.where(k_nn == k)[0][0]] = accuracy_score(y_test, y_pred)
        b_squirr[n_iter, np.where(k_nn == k)[0][0]] = (y_pred == "Black").sum()
        c_squirr[n_iter, np.where(k_nn == k)[0][0]] = (y_pred == "Cinnamon").sum()
        
avg_baseline = baselines / total_seeds
avg_accuracies = np.sum(accuracies, axis=0) / total_seeds
b_squirr =  np.sum(b_squirr, axis=0) / total_seeds
c_squirr =  np.sum(c_squirr, axis=0) / total_seeds


# We can see that this method beats the baseline for a 'K' in the region of around [11-21] and peaking at 17-19

# In[ ]:


plt.plot(k_nn, avg_accuracies)
plt.axhline(y=avg_baseline, color='r', label="Baseline")
plt.title(f"K-Nearest Neighbors\n Variables: {model_cols} - Averaged {total_seeds} seeds")
plt.ylabel("Accuracy")
plt.xlabel("K")
plt.legend()
plt.show()


# Checking the distribution of the predicted class for different values of K.
# 
# As K increases the model gains accuracy, peaking at a point when only the most obvious squirrels are predicted and until it reaches a point around K > 27 when the model starts predicting almost everything as Gray.

# In[ ]:


plt.plot(k_nn, b_squirr, c="black", label="Black Squirrels")
plt.plot(k_nn, c_squirr, c="orange", label="Cinnamon Squirrels")
plt.title(f"Minority class Squirrels predicted by K Nearest Neighbors")
plt.ylabel("Total squirrels")
plt.xlabel("K")
plt.legend()
plt.show()


# ## Test dataset
# 1. Training with all data
# 2. Checking the distribution of the model predictions on test dataset for consistency with training predictions
# 3. Probing Public LB with predictions based on the best region of K values

# In[ ]:


X_train = train_df[model_cols]
y_train = train_df[depend_var]
X_test = test_df[model_cols]


# In[ ]:


k_nn = np.arange(9,27,2)
g_squirr, b_squirr, c_squirr = [], [], []

for k in k_nn:
    knn = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    g_squirr.append((y_pred == "Gray").sum())
    b_squirr.append((y_pred == "Black").sum())
    c_squirr.append((y_pred == "Cinnamon").sum())


# The distribution of the target class is not consisten with training, however given how few black examples there are it could be normal to see large differences between train and test data.
# 
# Selected K = 17, and also submitted other K values around this region which ended up performing worse.

# In[ ]:


plt.plot(k_nn, b_squirr, c="black", label="Black Squirrels")
plt.plot(k_nn, c_squirr, c="orange", label="Cinnamon Squirrels")
plt.title(f"Minority class Squirrels predicted by K Nearest Neighbors")
plt.ylabel("Total squirrels")
plt.xlabel("K")
plt.legend()
plt.show()


# In[ ]:


knn = KNeighborsClassifier(n_neighbors=17, metric="euclidean")
knn.fit(X_train, y_train)
test_df = pd.read_csv(path_test)
y_pred = knn.predict(X_test)

print(pd.Series(y_pred).value_counts())


# ## Submission

# In[ ]:


test_df["Fur Color"] = y_pred
submit_df = test_df[["ID", "Fur Color"]]
submit_df.to_csv('submission_knn1.csv', index=False)


# ## Results
# - LB Baseline: 0.83333, 495 correct
# - Accuracy on LB: 0.84006, +4 correct squirrels over baseline
