#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Directive pour afficher les graphiques dans Jupyter
get_ipython().run_line_magic('matplotlib', 'inline')

# Pandas : librairie de manipulation de données
# NumPy : librairie de calcul scientifique
# MatPlotLib : librairie de visualisation et graphiques
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn import model_selection

from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score,auc, accuracy_score

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn import datasets


# In[ ]:


from tensorflow.keras.models import Sequential, load_model

from tensorflow.keras.layers import Dense, Dropout, Flatten

from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization

from tensorflow.keras.utils import to_categorical


# In[ ]:


df=pd.read_csv('../input/skin-cancer-mnist-ham10000/hmnist_28_28_RGB.csv')


# In[ ]:


df.shape


# In[ ]:


df.head()


# In[ ]:


df.label.value_counts()


# In[ ]:


labels = [i for i in range(7)]
print(labels)


# In[ ]:


n_samples = len(df.index)
images = np.array(df.drop(['label'],axis=1))
images = images.reshape(n_samples,28,28,3)


# In[ ]:


plt.figure(figsize=(20,10))
for i in range(50) :
    plt.subplot(5,10,i+1)
    plt.axis('off')
    i *= 200
    plt.imshow(images[i], cmap="gray_r")
    plt.title(labels[df.label[i]])


# In[ ]:


y = df['label'].values
X = df.drop(['label'] , axis=1).values


# In[ ]:


X = X/255


# In[ ]:


num_classes = 7


# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


# In[ ]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# In[ ]:


model = Sequential()
model.add(Dense(200, activation='relu'))
model.add(Dense(60, activation='relu'))
model.add(Dense(7, activation='softmax'))


# In[ ]:


model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[ ]:


train = model.fit(X_train , y_train , validation_data=(X_test,y_test), epochs=30, verbose=1)


# In[ ]:


model.evaluate(X_test,y_test)


# In[ ]:


print(train.history['accuracy'])


# In[ ]:


print(train.history['val_accuracy'])


# In[ ]:


def plot_scores(train) :
    accuracy = train.history['accuracy']
    val_accuracy = train.history['val_accuracy']
    epochs = range(len(accuracy))
    plt.plot(epochs, accuracy, 'b', label='Score apprentissage')
    plt.plot(epochs, val_accuracy, 'r', label='Score validation')
    plt.title('Scores')
    plt.legend()
    plt.show()


# In[ ]:


plot_scores(train)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(images, y, test_size=0.2, random_state=1)


# In[ ]:


cnn = Sequential()
cnn.add(Conv2D(32, (3,3), activation = 'relu', input_shape = (28,28,3)))
cnn.add(Dropout(0.2))
cnn.add(BatchNormalization())
cnn.add(MaxPooling2D((2,2) , strides = 2 , padding = 'same'))
cnn.add(Flatten())
cnn.add(Dense(7, activation='softmax'))
cnn.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
cnn.summary()


# In[ ]:


train = cnn.fit(X_train , y_train , validation_data=(X_test,y_test), epochs=100, verbose=1)


# In[ ]:


cnn.evaluate(X_test,y_test)


# In[ ]:


plot_scores(train)

