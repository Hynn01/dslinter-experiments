#!/usr/bin/env python
# coding: utf-8

# <div style="border-radius:10px;
#             border : #015a2c solid;
#             background-color:#ecfff5;
#            font-size:110%;
#            letter-spacing:0.5px;
#             text-align: center">
# 
# <center><h1 style="padding: 25px 0px; color:#015a2c; font-weight: bold; font-family: Cursive">
#  Fashion mnist <br><br>👕👚👖👜🥿👟🥾<br><br> Grayscale image analysis</h1></center>
# <center><h3 style="padding-bottom: 25px; color:#015a2c; font-weight: bold; font-style:italic; font-family: Cursive">
# With Convolutional Neural Network (CNN)</h3></center>     
# 
# </div>

# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


get_ipython().system('pip install -q -U keras-tuner')


# # Import Libraries

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt 
import random

# -----------------------------------
import tensorflow as tf
import keras
from tensorflow import keras
import keras_tuner as kt


# # Preprocessing images and Train/Test split

# In[ ]:


fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()


# <p style="font-family: Cursive; font-size:16px; padding: 25px 20px">
# The labels are an array of integers, ranging from 0 to 9.</p>
# 
# <table style="width:50%; font-family: Cursive; font-size:14px;">
#   <tr>
#     <th>Label</th>
#     <th>Class</th>
#   </tr>
#   <tr>
#     <td>0</td>
#     <td>T-shirt/top</td>
#   </tr>
#   <tr>
#     <td>1</td>
#     <td>Trouser</td>
#   </tr>
#   <tr>
#     <td>2</td>
#     <td>Pullover</td>
#   </tr>
#   <tr>
#     <td>3</td>
#     <td>Dress</td>
#   </tr>
#   <tr>
#     <td>4</td>
#     <td>Coat</td>
#   </tr>
#   <tr>
#     <td>5</td>
#     <td>Sandal</td>
#   </tr>
#   <tr>
#     <td>6</td>
#     <td>Shirt</td>
#   </tr>
#   <tr>
#     <td>7</td>
#     <td>Sneaker</td>
#   </tr>
#   <tr>
#     <td>8</td>
#     <td>Bag</td>
#   </tr>
#   <tr>
#     <td>9</td>
#     <td>Ankle boot</td>
#   </tr>
# </table>
#     

# In[ ]:


train_images  = train_images / 255.0
test_images = test_images / 255.0


# In[ ]:


plt.figure(figsize=(10,10))
for i in range(16):
    plt.subplot(4,4,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.title("Image label is: {}".format(train_labels[i]))
plt.show()


# In[ ]:


x_train = train_images.reshape(-1,28,28,1)
x_test = test_images.reshape(-1,28,28,1)


# # Hyperparameter tuning (with keras_tuner) and Build Model

# In[ ]:


early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)


# In[ ]:


def build_model(hp):
    model = keras.Sequential([
        
    # First conv_block
    keras.layers.Conv2D(
        filters = hp.Choice('conv_1_filter', values=[16, 32, 64, 128]),
        kernel_size=hp.Choice('conv_1_kernel', values = [3,4]),
        activation='relu',
        input_shape=(28,28,1)),
    keras.layers.MaxPooling2D((2,2)),
    
    # Second conv_block
    keras.layers.Conv2D(
        filters = hp.Choice('conv_2_filter', values=[16, 32, 64, 128]),
        kernel_size=hp.Choice('conv_2_kernel', values = [3,4]),
        activation='relu'),
    keras.layers.MaxPooling2D((2,2)),
    
    # --------------------------------
    keras.layers.Flatten(),
    keras.layers.Dense(units = hp.Choice('units', values=[16, 32, 64, 128, 256]),
                       activation='relu'),
    keras.layers.Dropout(hp.Float('dropout', 0, 0.5, step=0.1, default=0.5)),
        
    # --------------------------------
    keras.layers.Dense(10)
    ])

    model.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', 
                                                            values=[1e-1, 1e-2, 1e-3, 1e-4])),
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    return model


# In[ ]:


tuner = kt.Hyperband(build_model,
                     objective="val_accuracy",
                     max_epochs=5,
                     factor=3,
                     hyperband_iterations=3)


# In[ ]:


tuner.search_space_summary()


# In[ ]:


tuner.search(x_train,train_labels, epochs=3, validation_split=0.2)


# In[ ]:


best_hps = best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"""conv_1_filter is {best_hps.get('conv_1_filter')}""")
print(f"""conv_1_kernel is {best_hps.get('conv_1_kernel')}""")
print(f"""conv_2_filter is {best_hps.get('conv_2_filter')}""")
print(f"""conv_2_kernel is {best_hps.get('conv_2_kernel')}""")
print("-------------------------------------------------")
print(f"""units is {best_hps.get('units')}""")
print(f"""learning_rate is {best_hps.get('learning_rate')}""")
print(f"""dropout is {best_hps.get('dropout')}""")


# In[ ]:


model = tuner.hypermodel.build(best_hps)
history = model.fit(x_train, train_labels, 
                    epochs=50, validation_split=0.2)

val_acc_per_epoch = history.history['val_accuracy']
best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
print('Best epoch: %d' % (best_epoch,))


# In[ ]:


hypermodel = tuner.hypermodel.build(best_hps)

history = hypermodel.fit(x_train, train_labels, 
                         epochs=best_epoch, 
                         validation_split=0.2, 
                         callbacks=[early_stop])


# In[ ]:


hypermodel.summary()


# In[ ]:


keras.utils.plot_model(hypermodel, show_shapes=True)


# <div style="border-radius:10px;
#             background-color:#ffffff;
#             border-style: solid;
#             border-color: #015a2c;
#             letter-spacing:0.5px;
#             font-family: Cursive; 
#             font-size:16px;
#             padding: 10px;">
# 
# <p>To get a feel for what kind of features our CNN has learned, one fun thing to do is to visualize how an input gets transformed as it goes through the model. <br>We can see below how the pixels highlighted turn to increasingly abstract and compact representations, especially at the second max_pooling2d grid.</p>
# ref: <a href="https://github.com/https-deeplearning-ai/tensorflow-1-public/blob/main/C1/W4/ungraded_labs/C1_W4_Lab_1_image_generator_no_validation.ipynb">Visualizing Intermediate Representations</a>
# </div>

# In[ ]:


successive_outputs = [layer.output for layer in hypermodel.layers[1:]]
visualization_model = keras.models.Model(inputs = hypermodel.input, outputs = successive_outputs)
index = 20
plt.imshow(train_images[index], cmap=plt.cm.binary)

x = train_images[index]
x = x.reshape((1,) + x.shape)
x /= 255
successive_feature_maps = visualization_model.predict(x)
layer_names = [layer.name for layer in hypermodel.layers[1:]]

for layer_name, feature_map in zip(layer_names, successive_feature_maps):
    if len(feature_map.shape) == 4:
        n_features = feature_map.shape[-1] 
        size = feature_map.shape[1]
        display_grid = np.zeros((size, size * n_features))
        for i in range(n_features):
            x = feature_map[0, :, :, i]
            x -= x.mean()
            x /= x.std()
            x *= 64
            x += 128
            x = np.clip(x, 0, 255).astype('uint8')
            display_grid[:, i * size : (i + 1) * size] = x

    scale = 20. / n_features
    plt.figure(figsize=(scale * n_features, scale))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')


# # Evaluation and Prediction

# In[ ]:


eval_result = hypermodel.evaluate(x_test, test_labels)
print("[test loss, test accuracy]:", eval_result)


# In[ ]:


pred = hypermodel.predict(x_test)

print("Prediction is -> {}".format(pred[12]))
print("Actual value is -> {}".format(test_labels[12]))
print("The highest value for label is {}".format(np.argmax(pred[12])))


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss'] 

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()


# # References
# * https://www.tensorflow.org/tutorials/keras/classification
# * https://blog.tensorflow.org/2020/01/hyperparameter-tuning-with-keras-tuner.html
# * https://www.simplilearn.com/tutorials/deep-learning-tutorial/keras-tuner
# * https://github.com/https-deeplearning-ai/tensorflow-1-public/blob/main/C1/W4/ungraded_labs/C1_W4_Lab_1_image_generator_no_validation.ipynb
# * https://www.analyticsvidhya.com/blog/2021/06/create-convolutional-neural-network-model-and-optimize-using-keras-tuner-deep-learning/

# <div style="border-radius:10px;
#             background-color:#ffffff;
#             border-style: solid;
#             border-color: #015a2c;
#             letter-spacing:0.5px;">
# 
# <center><h4 style="padding: 5px 0px; color:#015a2c; font-weight: bold; font-family: Cursive">
#     Thanks for your attention and for reviewing my notebook.🙌 <br><br>Please write your comments for me.📝</h4></center>
# <center><h4 style="padding: 5px 0px; color:#015a2c; font-weight: bold; font-family: Cursive">
# If you liked my work and found it useful, please upvote. Thank you🙏</h4></center>
# </div>
