#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from IPython.display import display
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


# In[ ]:


train = pd.read_csv("../input/tabular-playground-series-apr-2022/train.csv")
test = pd.read_csv("../input/tabular-playground-series-apr-2022/test.csv")
sample_submission = pd.read_csv("../input/tabular-playground-series-apr-2022/sample_submission.csv")
labels = pd.read_csv("../input/tabular-playground-series-apr-2022/train_labels.csv")


# In[ ]:


train = train.merge(labels, how='left')


# In[ ]:


for col in train.columns[0:3]:
    if col == "sequence":
        train[col] = train[col].astype("int32")
    else:
        train[col] = train[col].astype("int16")
for col in train.columns[3:16]:
    train[col] = train[col].astype("float32")


# In[ ]:


sensors = ['sensor_00', 'sensor_01', 'sensor_02',
       'sensor_03', 'sensor_04', 'sensor_05', 'sensor_06', 'sensor_07',
       'sensor_08', 'sensor_09', 'sensor_10', 'sensor_11', 'sensor_12']


# plt.figure(figsize=(16,10))
# 
# sns.heatmap(train[sensors+["state"]].corr(),annot=True,cbar=False)

# sns.pairplot(train[sensors+["state"]][30:90],hue="state")

# In[ ]:


labels=labels["state"]


# In[ ]:





# In[ ]:


seqs = train["sequence"]
import warnings
warnings.filterwarnings("ignore")
def features(df,sensors):
    for w in [3,5]:
        
        for sensor in sensors:
            num = int(sensor.split("_")[1])
            if num < 10  :
                df[f"roll{w} sensor_0{num}"] = (df[f"sensor_0{num}"].ewm(com=float(w),min_periods=w).mean()).fillna(0)
                df[f"lag sensor_0{num}"] = df[f"sensor_0{num}"].shift(3).fillna(0)
                df[f"diff sensor_0{num}"] = df[f"lag sensor_0{num}"]-df[f"sensor_0{num}"]
                df[f"lag2 sensor_0{num}"] = df[f"roll{w} sensor_0{num}"].shift(1).fillna(0)
                df[f"diff2 sensor_0{num}"] = df[f"lag2 sensor_0{num}"]-df[f"roll{w} sensor_0{num}"]

            elif num >9  :
                df[f"roll{w} sensor_{num}"] = (df[f"sensor_{num}"].ewm(com=float(w),min_periods=w).mean()).fillna(0)
                df[f"lag sensor_{num}"] = df[f"sensor_{num}"].shift(3).fillna(0)
                df[f"diff sensor_{num}"] = df[f"lag sensor_{num}"]-df[f"sensor_{num}"]
                df[f"lag2 sensor_{num}"] = df[f"roll{w} sensor_{num}"].shift(1).fillna(0)
                df[f"diff2 sensor_{num}"] = df[f"lag2 sensor_{num}"]-df[f"roll{w} sensor_{num}"]


    return df
train = features(train,sensors)
test = features(test,sensors)


# In[ ]:


train.shape


# In[ ]:


train = train.drop(["sequence", "subject", "step","state"], inplace = False, axis = 1).values
test = test.drop(["sequence", "subject", "step"], inplace = False, axis = 1).values


# In[ ]:


features = train.shape[1]
from sklearn.preprocessing import StandardScaler
norm = StandardScaler()
train = norm.fit_transform(train)
test = norm.transform(test)
train = train.reshape(int(len(train)/60), 60, features)
test = test.reshape(int(len(test)/60), 60, features)


# In[ ]:


import random


# In[ ]:


total_sample = 15
randInt = np.array([random.choice(labels[labels==1].index) for x in range(int(total_sample/2))]+
                        [random.choice(labels[labels==0].index) for x in range(int(total_sample/2)+1)])
fig,axs=plt.subplots(nrows=5,ncols=3,figsize=(20,20))
plt.subplots_adjust(wspace=-0.2, hspace=0.6)
for i, ax in enumerate(axs.flat):
    pcm = ax.imshow(train[randInt[i]].squeeze() ,cmap=plt.cm.magma_r)
    if labels[i]==0:
        ax.set_title(f"loc:{randInt[i]} state :{labels[i]}",fontdict={"color":"blue"})
    else :
        ax.set_title(f"loc:{randInt[i]} state :{labels[i]}",fontdict={"color":"green"})
    ax.set_xlabel("features",fontdict={"color":"green"})
    ax.set_ylabel("sequence steps",fontdict={"color":"green"})
    fig.colorbar(pcm, ax=ax)
    ax.set_xticks([])
    ax.set_yticks([])
plt.show()


# In[ ]:


from sklearn.metrics import roc_curve
def plot_loss_auc(history,y_true,prediction):
    """
    history: history = model.fit() 
    y_true: true validation or test set labels
    prediction: prediction on val set or test set
    """
    fp, tp, _ = roc_curve(y_true, prediction)
     
    _,ax = plt.subplots( ncols=4,nrows=1,figsize=(20,3))
    ax[1].set_xlabel("epochs")
    ax[1].set_ylabel("loss")
    ax[1].set_title("final val_loss %1.4f"%(history.history["val_loss"][-1:][0]))
    ax[2].set_xlabel("epochs")
    ax[2].set_ylabel("auc")
    ax[2].set_title("final val_auc %1.4f"%(history.history["val_auc"][-1:][0]))
    ax[0].set_xlabel("learning rate")
    ax[0].set_ylabel("loss")
    ax[0].set_title("log lr vs loss")
    ax[3].plot(fp, tp,label="ROC", linewidth=2)
    ax[3].set_xlabel('False positives')
    ax[3].set_ylabel('True positives')
    ax[3].set_title("ROC")
    ax[0].semilogx(history.history["lr"], history.history["loss"])
    ax[0].set_ylim(ymax=0.4)
    pd.DataFrame([history.history["auc"],history.history["val_auc"]],index=["auc","val_auc"]).T.plot(ax=ax[2])
    pd.DataFrame([history.history["loss"],history.history["val_loss"]],index=["loss","val_loss"]).T.plot(ax=ax[1])
    plt.show()


# In[ ]:


from sklearn.metrics import confusion_matrix,classification_report
def plot_cm(y_true, prediction, p=0.5):
    """
    y_true: true validation or test set labels
    prediction: prediction on val set or test set
    
    """
    cm = confusion_matrix(y_true, prediction > p)
    plt.figure(figsize=(3,3))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title('Confusion matrix @{:.2f}'.format(p))
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    
    print('\nNot Have Disease Detected (True Negatives): ', cm[0][0])
    print('Not Have Disease Incorrectly Detected (False Positives): ', cm[0][1])
    print('Disease Missed (False Negatives): ', cm[1][0])
    print('Disease Detected (True Positives): ', cm[1][1])
    print('Total Disease: ', np.sum(cm[1]))
    plt.show()


# In[ ]:


try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy() 

print("REPLICAS: ", strategy.num_replicas_in_sync)


# In[ ]:


from IPython.display import display


# In[ ]:


input_shape =  (60, features)


# In[ ]:


input_shape


# In[ ]:


lr = 0.001
def cnn(reg):
    
    inputs = keras.layers.Input(shape = input_shape)
    # reshaping  inputs layer for conv2d layers
    inputs1 = tf.keras.layers.Reshape(target_shape=(60, features,1))(inputs)
    x = tf.keras.layers.Conv2D(76, (3, 3),kernel_regularizer=tf.keras.regularizers.L2(reg), activation="relu", padding="same")(inputs1)
    x = tf.keras.layers.MaxPooling2D((3, 3), padding="same")(x)
    x = tf.keras.layers.Conv2D(76, (3, 3),kernel_regularizer=tf.keras.regularizers.L2(reg), activation="relu", padding="same")(x)
    x = tf.keras.layers.MaxPooling2D((3, 3), padding="same")(x)
    x = tf.keras.layers.Conv2DTranspose(76, (3, 3), strides=2,kernel_regularizer=tf.keras.regularizers.L2(reg), activation="relu", padding="same")(x)
    x = tf.keras.layers.Conv2DTranspose(76, (3, 3), strides=2,kernel_regularizer=tf.keras.regularizers.L2(reg), activation="relu", padding="same")(x)
    x = tf.keras.layers.Conv2D(76, (3, 3), activation="relu",kernel_regularizer=tf.keras.regularizers.L2(reg), padding="same")(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    output = keras.layers.Dense(1,activation="sigmoid")(x)
    model = tf.keras.Model(inputs,output,name="convnet")
    optimizer = keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer = optimizer , loss = "binary_crossentropy", metrics = ["AUC","acc"])
    return model
model3 = cnn(reg=0.00013)
display(model3.summary())

print("\nlearning rate decay")

epochs = 50
lr = 0.001
startDecay = 10
def scheduler(epoch, lr):
    if epoch < startDecay:
        return lr
    else:
        return lr * tf.math.exp(-0.054)


def plot_lr_decay(epochs,lr):
    x = np.arange(0,epochs)
    lrs = [] 
    lr2=lr
    for epoch in x:
        lr =  scheduler(epoch,lr)
        lrs.append(lr)
    y = np.array(lrs)
    plt.figure(figsize=(8,4))
    plt.plot(x,y)
    plt.vlines(x=startDecay-1,linestyles="--",colors="r",ymin=y[-1],ymax=lr2)
    plt.xlabel("epochs")
    plt.ylabel("learning rate")
    plt.title("learning rate decay")
callback = keras.callbacks.EarlyStopping(patience = 8, restore_best_weights = True)
lrDecay = keras.callbacks.LearningRateScheduler(scheduler)
plot_lr_decay(epochs,lr,)


# In[ ]:


def plot_valimg(valx,predy,true_y):
    """
    
    valx: validation split
    predy: predictions on valx
    true_y: true y labels
    
    """
    randInt = np.random.randint(0,valx.shape[0]-1,15)
    _,axs=plt.subplots(nrows=5,ncols=3,figsize=(18,20))
    plt.subplots_adjust(wspace=0., hspace=0.65)
    for i, ax in enumerate(axs.flat):
        pcm = ax.imshow(valx[randInt[i]].squeeze(),cmap=plt.cm.magma_r)
        predy =(predy>0.5).astype(int)
        fig.colorbar(pcm, ax=ax)
        A = valx[randInt[i]]
        if predy[randInt[i]] == true_y[randInt[i]]:
           
            ax.set_title( f"pred label: {int(predy[randInt[i]])} , \ntrue label: {true_y[randInt[i]]} , \nvar: {A.var():.3f}, std: {A.std():.3f},\nmean: {A.mean():.3f},ptp: {np.ptp(A):.3f} " ,
                         fontdict={"color":"green"})
            ax.set_xlabel("features",fontdict={"color":"green"})
            ax.set_ylabel("sequence steps",fontdict={"color":"green"})
            
        else: 
            ax.set_title( f"pred label: {int(predy[randInt[i]])} , \ntrue label: {true_y[randInt[i]]}, \nvar: {A.var():.3f}, std: {A.std():.3f},\nmean: {A.mean():.3f},ptp: {np.ptp(A):.3f}" ,
                         fontdict={"color":"red"})
            ax.set_xlabel("features",fontdict={"color":"green"})
            ax.set_ylabel("sequence steps",fontdict={"color":"green"})
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()


# In[ ]:


n_splits = 10
cv_scores = np.zeros(n_splits)
test_preds = []
batch_size = 128
kf = GroupKFold(n_splits = n_splits)
for i, (train_split, test_split) in enumerate(kf.split(train, labels, seqs.unique())):
    with strategy.scope():
        X_train, X_test = train[train_split], train[test_split]
        y_train, y_test = labels.iloc[train_split].values, labels.iloc[test_split].values
        model = cnn(reg=0.00013)
        print("\n\n"+"*"*15, f"Fold {i+1}", "*"*15)
        history = model.fit(X_train, y_train, validation_data = (X_test, y_test), epochs = epochs, batch_size = batch_size, 
                  callbacks = [callback,lrDecay])

        test_pred = model.predict(X_test).squeeze()
        cv_score = roc_auc_score(y_test, test_pred)
        cv_scores[i] = cv_score

        plot_loss_auc(history,y_test,test_pred)
        plot_valimg(X_test,test_pred,y_test)
        plot_cm(y_test,test_pred, p=0.5)

        test_preds.append(model.predict(test).squeeze())
        print(f"mean cv score: {cv_scores[0:(i+1)].mean():.5f}")



print(f"final mean cv score : {cv_scores.mean():.3f} ")


# In[ ]:


print(f"mean cv score: {cv_scores.mean():.5f}")


# In[ ]:


sample_submission["state"] = sum(test_preds)/len(test_preds)
sample_submission.to_csv("submission.csv", index=False)


# In[ ]:


get_ipython().system('head submission.csv')


# In[ ]:




