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


# In[ ]:


train_data = pd.read_csv('/kaggle/input/ml2021-2022-2-kmeans/train.csv').values
test_data = pd.read_csv('/kaggle/input/ml2021-2022-2-kmeans/test.csv').values


# In[ ]:


class KMeans():
    def __init__(self, num_k, label=False, max_epochs=100):
        self.num_k = num_k
        self.label = label
        self.center = None
        self.max_epochs = max_epochs
        self.model = None
    
    def init_center(self, dataset):
        index_list = []
        label_list = []
        for data in dataset:
            label_list.append(data[-1])
        label_set = set(label_list)
        while(len(index_list) < self.num_k):
            index = np.random.randint(0, len(dataset)-1)
            if index not in index_list:
                if self.label is True:
                    if dataset[index][-1] in label_set:
                        index_list.append(index)
                        label_set.remove(dataset[index][-1])
                else:
                    index_list.append(index)
        return np.array([dataset[i] for i in index_list])
    
    def cal_dist(self, x, y):
        if self.model is 'train':
            return sum(np.square(np.array(x[:-1]) - np.array(y[:-1])))
        if self.model is 'predict':
            return sum(np.square(np.array(x) - np.array(y[:-1])))
        
    def cal_mean(self, dataset):
        data = np.array(dataset)[:, :-1]
        result = np.mean(data, axis=0)
        result = list(result)
        result.append(dataset[0][-1])
        return result
    
    def cal_stop_flag(self, new_center_list, center_list):
        if (np.array(new_center_list) == np.array(center_list)).all():
            return True
        threshold = 0.
        for i in range(len(center_list)):
            threshold += self.cal_dist(center_list[i], new_center_list[i])
        if threshold < 0.001:
            return True
        return False
    
    def fit(self, dataset):
        self.model = 'train'
        cnt = 0
        result_list = []
        center_list = self.init_center(dataset)
        for i in range(self.num_k):
            temp = []
            result_list.append(temp)
        while True:
            for data in dataset:
                dist_list = []
                for center in center_list:
                    dist_list.append(self.cal_dist(data, center))
                data[-1] = center_list[dist_list.index(min(dist_list))][-1]
                result_list[dist_list.index(min(dist_list))].append(data)
            new_center_list = [self.cal_mean(i) for i in result_list]
            stop_flag = self.cal_stop_flag(new_center_list, center_list)
            if stop_flag is True:
                self.center_list = new_center_list
                print("迭代次数:{}".format(cnt))
                break
            center_list = new_center_list
            cnt = cnt + 1
    
    def predict(self, test_data):
        self.model = 'predict'
        predictions = []
        for data in test_data:
            min_dist = self.cal_dist(data, self.center_list[0])
            label = self.center_list[0][-1]
            for center in self.center_list[1:]:
                dist = self.cal_dist(data, center)
                if dist < min_dist:
                    min_dist = dist
                    label = center[-1]
            predictions.append(label)
        return predictions 


# In[ ]:


model = KMeans(num_k=2, label=True)

model.fit(train_data)


# In[ ]:


result = model.predict(test_data)


# In[ ]:


out_dict = {
    'id':list(np.arange(len(test_data))),
    'style':list(result)
}
out = pd.DataFrame(out_dict)
out.to_csv('submission.csv',index=False)

