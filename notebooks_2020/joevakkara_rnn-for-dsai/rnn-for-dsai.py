#!/usr/bin/env python
# coding: utf-8

# # RNN Prediction

# ## Imports

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import datetime
from sklearn.preprocessing import MinMaxScaler


# ## We need to import several things from Keras.

# In[ ]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, GRU, Embedding
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from tensorflow.keras.backend import square, mean


# ## This was developed using Python 3.6 (Anaconda) and package versions:

# In[ ]:


tf.__version__


# In[ ]:


tf.keras.__version__


# In[ ]:


pd.__version__


# ## Load the pre-processed Data

# In[ ]:


get_ipython().system('ls "../input/ca-data"')


# In[ ]:


path = '../input/ca-data/'
ca1_data = pd.read_csv(path+"CA1_ext.csv")
ca2_data = pd.read_csv(path+"CA2_ext.csv")
ca3_data = pd.read_csv(path+"CA_3_ext.csv")
ca4_data = pd.read_csv(path+"CA4_ext.csv")
tx1_data = pd.read_csv(path+"TX_1_ext.csv")
tx2_data = pd.read_csv(path+"TX_2_ext.csv")
tx3_data = pd.read_csv(path+"TX_3_ext.csv")
wi1_data = pd.read_csv(path+"WI_1_ext.csv")
wi2_data = pd.read_csv(path+"WI_2_ext.csv")
wi3_data = pd.read_csv(path+"WI_3_ext.csv")


# In[ ]:


data = {}
data["CA_1"] = ca1_data[["Hobbie_revenue","House_revenue","Foods_revenue"]]
data["CA_2"] = ca2_data[["Hobbie_revenue","House_revenue","Foods_revenue"]]
data["CA_3"] = ca3_data[["Hobbie_revenue","House_revenue","Foods_revenue"]]
data["CA_4"] = ca4_data[["Hobbie_revenue","House_revenue","Foods_revenue"]]
data["TX_1"] = tx1_data[["Hobbie_revenue","House_revenue","Foods_revenue"]]
data["TX_2"] = tx2_data[["Hobbie_revenue","House_revenue","Foods_revenue"]]
data["TX_3"] = tx3_data[["Hobbie_revenue","House_revenue","Foods_revenue"]]
data["WI_1"] = wi1_data[["Hobbie_revenue","House_revenue","Foods_revenue"]]
data["WI_2"] = wi2_data[["Hobbie_revenue","House_revenue","Foods_revenue"]]
data["WI_3"] = wi3_data[["Hobbie_revenue","House_revenue","Foods_revenue"]]
data


# In[ ]:


data["CA_1"].head()


# List of the cities used in the data-set.

# In[ ]:


listofstore = ["CA_1","CA_2","CA_3","CA_4","TX_1","TX_2","TX_3","WI_1","WI_2","WI_3"]
listofstore


# In[ ]:


data_temp = data["CA_1"].join(data["CA_2"], lsuffix='_CA_1', rsuffix='_CA_2')
for store in listofstore[2:]:
    data_temp1 = data_temp.join(data[store], lsuffix='', rsuffix=store)
    data_temp1 = data_temp1.rename(columns={"Hobbie_revenue": "Hobbie_revenue_"+store,"House_revenue": "House_revenue_"+store,"Foods_revenue": "Foods_revenue_"+store})
    data_temp = data_temp1
    
#data_temp = data_temp1.join(data["TX_1"], lsuffix='', rsuffix='_TX_1')

data_df = data_temp1.copy()


# These are the top rows of the data-set.

# In[ ]:


data_df.head()


# There are 3*10 input-signals in the data-set. There are 1913 rows

# In[ ]:


data_df.values.shape


# ### Add Data
# 
# We can add some input-signals to the data that may help our model in making predictions.

# In[ ]:


import datetime
numdays = 1913
base = datetime.datetime(2011, 1, 29)
date_list = [base + datetime.timedelta(days=x) for x in range(numdays)]


# In[ ]:


from datetime import datetime
dayofyearlist = [i.timetuple().tm_yday for i in date_list]


# In[ ]:


data_df["Dayofyear"] = dayofyearlist


# In[ ]:


data_df


# ### Target Data for Prediction

# In[ ]:


target_store = 'CA_1'


# We will try and predict these signals.

# In[ ]:


target_names = ['Hobbie_revenue', 'House_revenue', 'Foods_revenue']


# The following is the number of time-steps that we will shift the target-data. Our data-set is sampled to have an observation for each day, so there are 30 observations for a month.
# 
# We want to predict the Revenue 1 month into the future, we shift the data 30 time-steps

# In[ ]:


shift_months = 1
shift_steps = shift_months * 30  # Number of days.


# In[ ]:


data_targets = data[target_store][target_names].shift(-shift_steps)


# In[ ]:


data[target_store][target_names].head(shift_steps + 5)


# In[ ]:


data_targets.head(5)


# In[ ]:


data_targets.tail()


# ### NumPy Arrays
# 
# We now convert the Pandas data-frames to NumPy arrays that can be input to the neural network. We also remove the last part of the numpy arrays, because the target-data has `NaN` for the shifted period, and we only want to have valid data and we need the same array-shapes for the input- and output-data.
# 
# These are the input-signals:

# In[ ]:


data_df.values


# In[ ]:


x_data = data_df.values[0:-shift_steps]


# In[ ]:


print(type(x_data))
print("Shape:", x_data.shape)


# These are the output-signals (or target-signals):

# In[ ]:


y_data = data_targets.values[:-shift_steps]
y_data


# In[ ]:


print(type(y_data))
print("Shape:", y_data.shape)


# This is the number of observations (aka. data-points or samples) in the data-set:

# In[ ]:


num_data = len(x_data)
num_data


# This is the fraction of the data-set that will be used for the training-set:

# In[ ]:


train_split = 0.9


# This is the number of observations in the training-set:

# In[ ]:


num_train = int(train_split * num_data)
num_train


# This is the number of observations in the test-set:

# In[ ]:


num_test = num_data - num_train
num_test


# These are the input-signals for the training- and test-sets:

# In[ ]:


x_train = x_data[0:num_train]
x_test = x_data[num_train:]
len(x_train) + len(x_test)


# These are the output-signals for the training- and test-sets:

# In[ ]:


y_train = y_data[0:num_train]
y_test = y_data[num_train:]
len(y_train) + len(y_test)


# This is the number of input-signals:

# In[ ]:


num_x_signals = x_data.shape[1]
num_x_signals


# This is the number of output-signals:

# In[ ]:


num_y_signals = y_data.shape[1]
num_y_signals


# ### Scaled Data
# 
# The data-set contains a wide range of values:

# In[ ]:


print("Min:", np.min(x_train))
print("Max:", np.max(x_train))


# The neural network works best on values roughly between -1 and 1, so we need to scale the data before it is being input to the neural network. We can use `scikit-learn` for this.
# 
# We first create a scaler-object for the input-signals.

# In[ ]:


x_scaler = MinMaxScaler()


# We then detect the range of values from the training-data and scale the training-data.

# In[ ]:


x_train_scaled = x_scaler.fit_transform(x_train)


# Apart from a small rounding-error, the data has been scaled to be between 0 and 1.

# In[ ]:


print("Min:", np.min(x_train_scaled))
print("Max:", np.max(x_train_scaled))


# We use the same scaler-object for the input-signals in the test-set.

# In[ ]:


x_test_scaled = x_scaler.transform(x_test)


# The target-data comes from the same data-set as the input-signals, because it is the weather-data for one of the cities that is merely time-shifted. But the target-data could be from a different source with different value-ranges, so we create a separate scaler-object for the target-data.

# In[ ]:


y_scaler = MinMaxScaler()
y_train_scaled = y_scaler.fit_transform(y_train)
y_test_scaled = y_scaler.transform(y_test)


# ## Data Generator
# 
# The data-set has now been prepared as 2-dimensional numpy arrays. The training-data has almost large observations, consisting of 20 input-signals and 3 output-signals.
# 
# These are the array-shapes of the input and output data:

# In[ ]:


print(x_train_scaled.shape)
print(y_train_scaled.shape)


# Instead of training the Recurrent Neural Network on the complete sequences of large observations, we will use the following function to create a batch of shorter sub-sequences picked at random from the training-data.

# In[ ]:


def batch_generator(batch_size, sequence_length):
    """
    Generator function for creating random batches of training-data.
    """

    # Infinite loop.
    while True:
        # Allocate a new array for the batch of input-signals.
        x_shape = (batch_size, sequence_length, num_x_signals)
        x_batch = np.zeros(shape=x_shape, dtype=np.float16)

        # Allocate a new array for the batch of output-signals.
        y_shape = (batch_size, sequence_length, num_y_signals)
        y_batch = np.zeros(shape=y_shape, dtype=np.float16)

        # Fill the batch with random sequences of data.
        for i in range(batch_size):
            # Get a random start-index.
            # This points somewhere into the training-data.
            idx = np.random.randint(num_train - sequence_length)
            
            # Copy the sequences of data starting at this index.
            x_batch[i] = x_train_scaled[idx:idx+sequence_length]
            y_batch[i] = y_train_scaled[idx:idx+sequence_length]
        
        yield (x_batch, y_batch)


# We will use a large batch-size so as to keep the GPU near 100% work-load. You may have to adjust this number depending on your GPU, its RAM and your choice of `sequence_length` below.

# In[ ]:


batch_size = 256


# In[ ]:


sequence_length = 30 * 6
sequence_length


# We then create the batch-generator.

# In[ ]:


generator = batch_generator(batch_size=batch_size,
                            sequence_length=sequence_length)


# We can then test the batch-generator to see if it works.

# In[ ]:


x_batch, y_batch = next(generator)


# This gives us a random batch of 256 sequences, each sequence having 180 observations, and each observation having 31 input-signals and 3 output-signals.

# In[ ]:


print(x_batch.shape)
print(y_batch.shape)


# We can plot one of the 20 input-signals as an example.

# In[ ]:


batch = 0   # First sequence in the batch.
signal = 0  # First signal from the 20 input-signals.
seq = x_batch[batch, :, signal]
plt.plot(seq)


# We can also plot one of the output-signals that we want the model to learn how to predict given all those 20 input signals.

# In[ ]:


seq = y_batch[batch, :, signal]
plt.plot(seq)


# ### Validation Set
# 
# The neural network trains quickly so we can easily run many training epochs. But then there is a risk of overfitting the model to the training-set so it does not generalize well to unseen data. We will therefore monitor the model's performance on the test-set after each epoch and only save the model's weights if the performance is improved on the test-set.
# 
# The batch-generator randomly selects a batch of short sequences from the training-data and uses that during training. But for the validation-data we will instead run through the entire sequence from the test-set and measure the prediction accuracy on that entire sequence.

# In[ ]:


validation_data = (np.expand_dims(x_test_scaled, axis=0),
                   np.expand_dims(y_test_scaled, axis=0))


# ## Create the Recurrent Neural Network
# 
# We are now ready to create the Recurrent Neural Network (RNN). We will use the Keras API for this because of its simplicity.

# In[ ]:


model = Sequential()


# We can now add a Gated Recurrent Unit (GRU) to the network. This will have 512 outputs for each time-step in the sequence.
# 
# Note that because this is the first layer in the model, Keras needs to know the shape of its input, which is a batch of sequences of arbitrary length (indicated by `None`), where each observation has a number of input-signals (`num_x_signals`).

# In[ ]:


model.add(GRU(units=512,
              return_sequences=True,
              input_shape=(None, num_x_signals,)))


# The GRU outputs a batch of sequences of 512 values. We want to predict 3 output-signals, so we add a fully-connected (or dense) layer which maps 512 values down to only 3 values.
# 
# The output-signals in the data-set have been limited to be between 0 and 1 using a scaler-object. So we also limit the output of the neural network using the Sigmoid activation function, which squashes the output to be between 0 and 1.

# In[ ]:


model.add(Dense(num_y_signals, activation='sigmoid'))


# A problem with using the Sigmoid activation function, is that we can now only output values in the same range as the training-data.
# 
# We can use a linear activation function on the output instead. This allows for the output to take on arbitrary values. It might work with the standard initialization for a simple network architecture, but for more complicated network architectures e.g. with more layers, it might be necessary to initialize the weights with smaller values to avoid `NaN` values during training. You may need to experiment with this to get it working.

# In[ ]:


if False:
    from tensorflow.python.keras.initializers import RandomUniform

    # Maybe use lower init-ranges.
    init = RandomUniform(minval=-0.05, maxval=0.05)

    model.add(Dense(num_y_signals,
                    activation='linear',
                    kernel_initializer=init))


# ### Loss Function
# 
# We will use Mean Squared Error (MSE) as the loss-function that will be minimized. This measures how closely the model's output matches the true output signals.
# 
# However, at the beginning of a sequence, the model has only seen input-signals for a few time-steps, so its generated output may be very inaccurate. Using the loss-value for the early time-steps may cause the model to distort its later output. We therefore give the model a "warmup-period" of 30 time-steps where we don't use its accuracy in the loss-function, in hope of improving the accuracy for later time-steps.

# In[ ]:


warmup_steps = 30


# In[ ]:


def loss_mse_warmup(y_true, y_pred):
    """
    Calculate the Mean Squared Error between y_true and y_pred,
    but ignore the beginning "warmup" part of the sequences.
    
    y_true is the desired output.
    y_pred is the model's output.
    """

    # The shape of both input tensors are:
    # [batch_size, sequence_length, num_y_signals].

    # Ignore the "warmup" parts of the sequences
    # by taking slices of the tensors.
    y_true_slice = y_true[:, warmup_steps:, :]
    y_pred_slice = y_pred[:, warmup_steps:, :]

    # These sliced tensors both have this shape:
    # [batch_size, sequence_length - warmup_steps, num_y_signals]

    # Calculat the Mean Squared Error and use it as loss.
    mse = mean(square(y_true_slice - y_pred_slice))
    
    return mse


# ### Compile Model
# 
# This is the optimizer and the beginning learning-rate that we will use.

# In[ ]:


optimizer = RMSprop(lr=1e-3)


# We then compile the Keras model so it is ready for training.

# In[ ]:


model.compile(loss=loss_mse_warmup, optimizer=optimizer)


# This is a very small model with only two layers. The output shape of `(None, None, 3)` means that the model will output a batch with an arbitrary number of sequences, each of which has an arbitrary number of observations, and each observation has 3 signals. This corresponds to the 3 target signals we want to predict.

# In[ ]:


model.summary()


# ### Callback Functions
# 
# During training we want to save checkpoints and log the progress to TensorBoard so we create the appropriate callbacks for Keras.
# 
# This is the callback for writing checkpoints during training.

# In[ ]:


path_checkpoint = '23_checkpoint.keras'
callback_checkpoint = ModelCheckpoint(filepath=path_checkpoint,
                                      monitor='val_loss',
                                      verbose=1,
                                      save_weights_only=True,
                                      save_best_only=True)


# This is the callback for stopping the optimization when performance worsens on the validation-set.

# In[ ]:


callback_early_stopping = EarlyStopping(monitor='val_loss',
                                        patience=5, verbose=1)


# This is the callback for writing the TensorBoard log during training.

# In[ ]:


callback_tensorboard = TensorBoard(log_dir='./23_logs/',
                                   histogram_freq=0,
                                   write_graph=False)


# This callback reduces the learning-rate for the optimizer if the validation-loss has not improved since the last epoch (as indicated by `patience=0`). The learning-rate will be reduced by multiplying it with the given factor. We set a start learning-rate of 1e-3 above, so multiplying it by 0.1 gives a learning-rate of 1e-4. We don't want the learning-rate to go any lower than this.

# In[ ]:


callback_reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                       factor=0.1,
                                       min_lr=1e-4,
                                       patience=0,
                                       verbose=1)


# In[ ]:


callbacks = [callback_early_stopping,
             callback_checkpoint,
             callback_tensorboard,
             callback_reduce_lr]


# ## Train the Recurrent Neural Network
# 
# We can now train the neural network.
# 

# In[ ]:


get_ipython().run_cell_magic('time', '', 'model.fit(x=generator,\n          epochs=30,\n          steps_per_epoch=100,\n          validation_data=validation_data,\n          callbacks=callbacks)')


# ### Load Checkpoint
# 
# Because we use early-stopping when training the model, it is possible that the model's performance has worsened on the test-set for several epochs before training was stopped. We therefore reload the last saved checkpoint, which should have the best performance on the test-set.

# In[ ]:


try:
    model.load_weights(path_checkpoint)
except Exception as error:
    print("Error trying to load checkpoint.")
    print(error)


# ## Performance on Test-Set
# 
# We can now evaluate the model's performance on the test-set. This function expects a batch of data, but we will just use one long time-series for the test-set, so we just expand the array-dimensionality to create a batch with that one sequence.

# In[ ]:


result = model.evaluate(x=np.expand_dims(x_test_scaled, axis=0),
                        y=np.expand_dims(y_test_scaled, axis=0))


# In[ ]:


print("loss (test-set):", result)


# In[ ]:


# If you have several metrics you can use this instead.
if False:
    for res, metric in zip(result, model.metrics_names):
        print("{0}: {1:.3e}".format(metric, res))


# ## Generate Predictions
# 
# This helper-function plots the predicted and true output-signals.

# In[ ]:


def plot_comparison(start_idx, length=100, train=True):
    """
    Plot the predicted and true output-signals.
    
    :param start_idx: Start-index for the time-series.
    :param length: Sequence-length to process and plot.
    :param train: Boolean whether to use training- or test-set.
    """
    
    if train:
        # Use training-data.
        x = x_train_scaled
        y_true = y_train
    else:
        # Use test-data.
        x = x_test_scaled
        y_true = y_test
    
    # End-index for the sequences.
    end_idx = start_idx + length
    
    # Select the sequences from the given start-index and
    # of the given length.
    x = x[start_idx:end_idx]
    y_true = y_true[start_idx:end_idx]
    
    # Input-signals for the model.
    x = np.expand_dims(x, axis=0)

    # Use the model to predict the output-signals.
    y_pred = model.predict(x)
    
    # The output of the model is between 0 and 1.
    # Do an inverse map to get it back to the scale
    # of the original data-set.
    y_pred_rescaled = y_scaler.inverse_transform(y_pred[0])
    
    #print(loss_mse_warmup(y_true, y_pred))
    
    # For each output-signal.
    for signal in range(len(target_names)):
        # Get the output-signal predicted by the model.
        signal_pred = y_pred_rescaled[:, signal]
        
        # Get the true output-signal from the data-set.
        signal_true = y_true[:, signal]
        

        # Make the plotting-canvas bigger.
        plt.figure(figsize=(15,5))
        
        # Plot and compare the two signals.
        plt.plot(signal_true, label='true')
        plt.plot(signal_pred, label='pred')
        
        # Plot grey box for warmup-period.
        p = plt.axvspan(0, warmup_steps, facecolor='black', alpha=0.15)
        
        # Plot labels etc.
        plt.ylabel(target_names[signal])
        plt.legend()
        plt.show()


# We can now plot an example of predicted output-signals. 
# 
# The prediction is not very accurate for the first 30-50 time-steps because the model has seen very little input-data at this point.
# The model generates a single time-step of output data for each time-step of the input-data, so when the model has only run for a few time-steps, it knows very little of the history of the input-signals and cannot make an accurate prediction. The model needs to "warm up" by processing perhaps 30-50 time-steps before its predicted output-signals can be used.
# 
# That is why we ignore this "warmup-period" of 50 time-steps when calculating the mean-squared-error in the loss-function. The "warmup-period" is shown as a grey box in these plots.

# In[ ]:


plot_comparison(start_idx=1000, length=500, train=True)


# The model was able to predict the overall oscillations quite well but the peaks were sometimes inaccurate. 

# In[ ]:


data["CA_4"]['Hobbie_revenue'][1000:1000+500].plot();


# ### Example from Test-Set
# 
# Now consider an example from the test-set. The model has not seen this data during training.
# 
# The temperature is predicted reasonably well, although the peaks are sometimes inaccurate.
# 
# The wind-speed has not been predicted so well. The daily oscillation-frequency seems to match, but the center-level and the peaks are quite inaccurate. A guess would be that the wind-speed is difficult to predict from the given input data, so the model has merely learnt to output sinusoidal oscillations in the daily frequency and approximately at the right center-level.
# 
# The atmospheric pressure is predicted reasonably well, except for a lag and a more noisy signal than the true time-series.

# In[ ]:


plot_comparison(start_idx=10, length=500, train=False)


# ## Conclusion
# 
# Used a Recurrent Neural Network to predict several time-series from a number of input-signals. We used revenue-data for 10 stores to predict next months revenue for one of the stores.
