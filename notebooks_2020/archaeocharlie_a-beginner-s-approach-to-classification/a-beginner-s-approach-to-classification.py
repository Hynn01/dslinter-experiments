#!/usr/bin/env python
# coding: utf-8

# ## Introduction
# ###Let me start by saying, this is not the best way to classify digits! This notebook is rather meant to be for someone who might not know where to start. As an ml beginner myself, I find it helpful to play with these sorts of commented kernels. Any suggestions for improvement or comments on poor coding practices are appreciated!

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt, matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from sklearn import svm
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Loading the data
# - We use panda's [read_csv][1]  to read train.csv into a [dataframe][2].
# - Then we separate our images and labels for supervised learning. 
# - We also do a [train_test_split][3] to break our data into two sets, one for training and one for testing. This let's us measure how well our model was trained by later inputting some known test data.
# 
# ### For the sake of time, we're only using 5000 images. You should increase or decrease this number to see how it affects model training.
# 
# 
#   [1]: http://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html
#   [2]: http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html#pandas.DataFrame
#   [3]: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html

# In[ ]:


labeled_images = pd.read_csv('../input/train.csv')
images = labeled_images.iloc[0:5000,1:]
labels = labeled_images.iloc[0:5000,:1]
train_images, test_images,train_labels, test_labels = train_test_split(images, labels, train_size=0.8, random_state=0)


# ## Viewing an Image
# - Since the image is currently one-dimension, we load it into a [numpy array][1] and [reshape][2] it so that it is two-dimensional (28x28 pixels)
# - Then, we plot the image and label with matplotlib
# 
# ### You can change the value of variable <i>i</i> to check out other images and labels.
# 
# 
#   [1]: https://docs.scipy.org/doc/numpy/reference/generated/numpy.array.html
#   [2]: https://docs.scipy.org/doc/numpy/reference/generated/numpy.reshape.html

# In[ ]:


i=1
img=train_images.iloc[i].as_matrix()
img=img.reshape((28,28))
plt.imshow(img,cmap='gray')
plt.title(train_labels.iloc[i,0])


# ## Examining the Pixel Values
# ### Note that these images aren't actually black and white (0,1). They are gray-scale (0-255). 
# - A [histogram][1] of this image's pixel values shows the range.
# 
# 
#   [1]: http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.hist

# In[ ]:


plt.hist(train_images.iloc[i])


# ## Training our model
# - First, we use the [sklearn.svm][1] module to create a [vector classifier][2]. 
# - Next, we pass our training images and labels to the classifier's [fit][3] method, which trains our model. 
# - Finally, the test images and labels are passed to the [score][4] method to see how well we trained our model. Fit will return a float between 0-1 indicating our accuracy on the test data set
# 
# ### Try playing with the parameters of svm.SVC to see how the results change. 
# 
# 
#   [1]: http://scikit-learn.org/stable/modules/svm.html
#   [2]: http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
#   [3]: http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC.fit
#   [4]: http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC.score
#   [5]: http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC.score

# In[ ]:


clf = svm.SVC()
clf.fit(train_images, train_labels.values.ravel())
clf.score(test_images,test_labels)


# ## How did our model do?
# ### You should have gotten around 0.10, or 10% accuracy. This is terrible. 10% accuracy is what get if you randomly guess a number. There are many ways to improve this, including not using a vector classifier, but here's a simple one to start. Let's just simplify our images by making them true black and white.
# 
# - To make this easy, any pixel with a value simply becomes 1 and everything else remains 0.
# - We'll plot the same image again to see how it looks now that it's black and white. Look at the histogram now.

# In[ ]:


test_images[test_images>0]=1
train_images[train_images>0]=1

img=train_images.iloc[i].as_matrix().reshape((28,28))
plt.imshow(img,cmap='binary')
plt.title(train_labels.iloc[i])


# In[ ]:


plt.hist(train_images.iloc[i])


# ## Retraining our model
# ### We follow the same procedure as before, but now our training and test sets are black and white instead of gray-scale. Our score still isn't great, but it's a huge improvement.

# In[ ]:


clf = svm.SVC()
clf.fit(train_images, train_labels.values.ravel())
clf.score(test_images,test_labels)


# ## Labelling the test data
# ### Now for those making competition submissions, we can load and predict the unlabeled data from test.csv. Again, for time we're just using the first 5000 images. We then output this data to a results.csv for competition submission.

# In[ ]:


test_data=pd.read_csv('../input/test.csv')
test_data[test_data>0]=1
results=clf.predict(test_data[0:5000])


# In[ ]:


results


# In[ ]:


df = pd.DataFrame(results)
df.index.name='ImageId'
df.index+=1
df.columns=['Label']
df.to_csv('results.csv', header=True)


# In[ ]:




