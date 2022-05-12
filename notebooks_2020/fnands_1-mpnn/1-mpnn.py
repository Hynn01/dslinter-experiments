#!/usr/bin/env python
# coding: utf-8

# ## Message Passing Neural Network
# 
# So, as many of you might have surmised by now the dataset for this challenge is essentially the QM9 dataset with some new values calculated for it. 
# 
# The first thing I though of when seeing this challenge was the [Gilmer paper](https://arxiv.org/abs/1704.01212), as it uses the QM9 dataset. ([see this talk](https://vimeo.com/238221016))
# 
# The major difference in this challenge is that we are asked to calulate bond properties (thus edges in a graph) as opposed to bulk properties in the paper. 
# 
# Here the model is laid out in a modular way so the parts can easily be replaced
# 

# In[ ]:


# Internet needs to be on
get_ipython().system('pip install tensorflow-gpu==2.0a0')


# In[ ]:


import tensorflow as tf
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.utils import shuffle
import os
print(os.listdir("../input"))


# In[ ]:


# Make sure tf 2.0 alpha has been installed
print(tf.__version__)


# In[ ]:


#is it using the gpu?
tf.test.is_gpu_available(
    cuda_only=False,
    min_cuda_compute_capability=None
)


# In[ ]:


datadir = "../input/"

nodes_train     = np.load(datadir + "champs-basic-graph/nodes_train.npz" )['arr_0']
in_edges_train  = np.load(datadir + "champs-basic-graph/in_edges_train.npz")['arr_0']
out_edges_train = np.load(datadir + "champs-basic-graph/out_edges_train.npz" )['arr_0']

nodes_test     = np.load(datadir + "champs-basic-graph/nodes_test.npz" )['arr_0']
in_edges_test  = np.load(datadir + "champs-basic-graph/in_edges_test.npz")['arr_0']


# In[ ]:


out_labels = out_edges_train.reshape(-1,out_edges_train.shape[1]*out_edges_train.shape[2],1)
in_edges_train = in_edges_train.reshape(-1,in_edges_train.shape[1]*in_edges_train.shape[2],in_edges_train.shape[3])
in_edges_test  = in_edges_test.reshape(-1,in_edges_test.shape[1]*in_edges_test.shape[2],in_edges_test.shape[3])


# In[ ]:


nodes_train, in_edges_train, out_labels = shuffle(nodes_train, in_edges_train, out_labels)


# ## Message passer
# Define the message passer like the Gilmer paper
# 
# Use a NN to embed edges as matrices, then matrix multiply with nodes.

# In[ ]:



class Message_Passer_NNM(tf.keras.layers.Layer):
    def __init__(self, node_dim):
        super(Message_Passer_NNM, self).__init__()
        self.node_dim = node_dim
        self.nn = tf.keras.layers.Dense(units=self.node_dim*self.node_dim, activation = tf.nn.relu)
      
    def call(self, node_j, edge_ij):
        
        # Embed the edge as a matrix
        A = self.nn(edge_ij)
        
        # Reshape so matrix mult can be done
        A = tf.reshape(A, [-1, self.node_dim, self.node_dim])
        node_j = tf.reshape(node_j, [-1, self.node_dim, 1])
        
        # Multiply edge matrix by node and shape into message list
        messages = tf.linalg.matmul(A, node_j)
        messages = tf.reshape(messages, [-1, tf.shape(edge_ij)[1], self.node_dim])

        return messages


# ## Aggregator
# 
# Define the message aggregator (just sum)  
# Probably overkill to have it as its own layer, but good if you want to replace it with something more complex
# 

# In[ ]:


class Message_Agg(tf.keras.layers.Layer):
    def __init__(self):
        super(Message_Agg, self).__init__()
    
    def call(self, messages):
        return tf.math.reduce_sum(messages, 2)


# ## Update function
# 
# Define the Update function (a GRU)  
# The GRU basically runs over a sequence of length 2, i.e. [ old state, agged_messages ]

# In[ ]:


class Update_Func_GRU(tf.keras.layers.Layer):
    def __init__(self, state_dim):
        super(Update_Func_GRU, self).__init__()
        self.concat_layer = tf.keras.layers.Concatenate(axis=1)
        self.GRU = tf.keras.layers.GRU(state_dim)
        
    def call(self, old_state, agg_messages):
    
        # Remember node dim
        n_nodes  = tf.shape(old_state)[1]
        node_dim = tf.shape(old_state)[2]
        
        # Reshape so GRU can be applied, concat so old_state and messages are in sequence
        old_state = tf.reshape(old_state, [-1, 1, tf.shape(old_state)[-1]])
        agg_messages = tf.reshape(agg_messages, [-1, 1, tf.shape(agg_messages)[-1]])
        concat = self.concat_layer([old_state, agg_messages])
        
        # Apply GRU and then reshape so it can be returned
        activation = self.GRU(concat)
        activation = tf.reshape(activation, [-1, n_nodes, node_dim])
        
        return activation


# ## Output layer
# 
# This is where the model diverges with the paper.   
# As the paper predicts bulk properties, but we are interested in edges, we need something different.   
# 
# Here the each edge is concatenated to it's two nodes and a MLP is used to regress the scalar coupling for each edge

# In[ ]:


# Define the final output layer 
class Edge_Regressor(tf.keras.layers.Layer):
    def __init__(self, intermediate_dim):
        super(Edge_Regressor, self).__init__()
        self.concat_layer = tf.keras.layers.Concatenate()
        self.hidden_layer_1 = tf.keras.layers.Dense(units=intermediate_dim, activation=tf.nn.relu)
        self.hidden_layer_2 = tf.keras.layers.Dense(units=intermediate_dim, activation=tf.nn.relu)
        self.output_layer = tf.keras.layers.Dense(units=1, activation=None)

        
    def call(self, nodes, edges):
            
        # Remember node dims
        n_nodes  = tf.shape(nodes)[1]
        node_dim = tf.shape(nodes)[2]
        
        # Tile and reshape to match edges
        state_i = tf.reshape(tf.tile(nodes, [1, 1, n_nodes]),[-1,n_nodes*n_nodes, node_dim ])
        state_j = tf.tile(nodes, [1, n_nodes, 1])
        
        # concat edges and nodes and apply MLP
        concat = self.concat_layer([state_i, edges, state_j])
        activation_1 = self.hidden_layer_1(concat)  
        activation_2 = self.hidden_layer_2(activation_1)

        return self.output_layer(activation_2)


# ## Message passing layer
# 
# Put all of the above together to make a message passing layer which does one round of message passing and node updating

# In[ ]:


# Define a single message passing layer
class MP_Layer(tf.keras.layers.Layer):
    def __init__(self, state_dim):
        super(MP_Layer, self).__init__(self)
        self.message_passers  = Message_Passer_NNM(node_dim = state_dim) 
        self.message_aggs    = Message_Agg()
        self.update_functions = Update_Func_GRU(state_dim = state_dim)
        
        self.state_dim = state_dim         

    def call(self, nodes, edges, mask):
      
        n_nodes  = tf.shape(nodes)[1]
        node_dim = tf.shape(nodes)[2]
        
        state_j = tf.tile(nodes, [1, n_nodes, 1])

        messages  = self.message_passers(state_j, edges)

        # Do this to ignore messages from non-existant nodes
        masked =  tf.math.multiply(messages, mask)
        
        masked = tf.reshape(masked, [tf.shape(messages)[0], n_nodes, n_nodes, node_dim])

        agg_m = self.message_aggs(masked)
        
        updated_nodes = self.update_functions(nodes, agg_m)
        
        nodes_out = updated_nodes
        # Batch norm seems not to work. 
        #nodes_out = self.batch_norm(updated_nodes)
        
        return nodes_out


# ## Put it all together to form a MPNN
# 
# Defines the full mpnn that does T message passing steps, where T is a hyperparameter.   
# As in the paper, the same MP layer is re-used, but this is not a requirement. 

# In[ ]:


adj_input = tf.keras.Input(shape=(None,), name='adj_input')
nod_input = tf.keras.Input(shape=(None,), name='nod_input')
class MPNN(tf.keras.Model):
    def __init__(self, out_int_dim, state_dim, T):
        super(MPNN, self).__init__(self)   
        self.T = T
        self.embed = tf.keras.layers.Dense(units=state_dim, activation=tf.nn.relu)
        self.MP = MP_Layer( state_dim)     
        self.edge_regressor  = Edge_Regressor(out_int_dim)
        #self.batch_norm = tf.keras.layers.BatchNormalization() 

        
    def call(self, inputs =  [adj_input, nod_input]):
      
      
        nodes            = inputs['nod_input']
        edges            = inputs['adj_input']

        # Get distances, and create mask wherever 0 (i.e. non-existant nodes)
        # This also masks node self-interactions...
        # This assumes distance is last
        len_edges = tf.shape(edges)[-1]
        
        _, x = tf.split(edges, [len_edges -1, 1], 2)
        mask =  tf.where(tf.equal(x, 0), x, tf.ones_like(x))
        
        # Embed node to be of the chosen node dimension (you can also just pad)
        nodes = self.embed(nodes) 
        
        #nodes = self.batch_norm(nodes)
        # Run the T message passing steps
        for mp in range(self.T):
            nodes =  self.MP(nodes, edges, mask)
        
        # Regress the output values
        con_edges = self.edge_regressor(nodes, edges)
        
        
        return con_edges
        


# ## Define the loss functions. 
# 
# Here the losses are MSE, MAE and LMAE.  
# (**note**: that for LMAE, as the values have been scaled down values will be much smaller than for unscaled values)

# In[ ]:


def mse(orig , preds):
 
    # Mask values for which no scalar coupling exists
    mask  = tf.where(tf.equal(orig, 0), orig, tf.ones_like(orig))

    nums  = tf.boolean_mask(orig,  mask)
    preds = tf.boolean_mask(preds,  mask)


    reconstruction_error = tf.reduce_mean(tf.square(tf.subtract(nums, preds)))


    return reconstruction_error


# In[ ]:


def log_mse(orig , preds):
 
    # Mask values for which no scalar coupling exists
    mask  = tf.where(tf.equal(orig, 0), orig, tf.ones_like(orig))

    nums  = tf.boolean_mask(orig,  mask)
    preds = tf.boolean_mask(preds,  mask)


    reconstruction_error = tf.math.log(tf.reduce_mean(tf.square(tf.subtract(nums, preds))))


    return reconstruction_error


# In[ ]:


def mae(orig , preds):
 
    # Mask values for which no scalar coupling exists
    mask  = tf.where(tf.equal(orig, 0), orig, tf.ones_like(orig))

    nums  = tf.boolean_mask(orig,  mask)
    preds = tf.boolean_mask(preds,  mask)


    reconstruction_error = tf.reduce_mean(tf.abs(tf.subtract(nums, preds)))


    return reconstruction_error


# In[ ]:


def log_mae(orig , preds):
 
    # Mask values for which no scalar coupling exists
    mask  = tf.where(tf.equal(orig, 0), orig, tf.ones_like(orig))

    nums  = tf.boolean_mask(orig,  mask)
    preds = tf.boolean_mask(preds,  mask)

    reconstruction_error = tf.math.log(tf.reduce_mean(tf.abs(tf.subtract(nums, preds))))

    return reconstruction_error


# ## Define some callbacks, the initial learning rate and the optimizer

# In[ ]:


learning_rate = 0.001
def step_decay(epoch):
    initial_lrate = learning_rate
    drop = 0.1
    epochs_drop = 20.0
    lrate = initial_lrate * np.power(drop,  
           np.floor((epoch)/epochs_drop))
    tf.print("Learning rate: ", lrate)
    return lrate

lrate = tf.keras.callbacks.LearningRateScheduler(step_decay)
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience = 15, restore_best_weights=True)

#lrate  =  tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1,
#                              patience=5, min_lr=0.00001, verbose = 1)

opt = tf.optimizers.Adam(learning_rate=learning_rate)


# ## Finally create the model, and compile

# In[ ]:


mpnn = MPNN(out_int_dim = 512, state_dim = 128, T = 4)
mpnn.compile(opt, log_mae, metrics = [mae, log_mse])


# Define some hyperparameters

# In[ ]:


train_size = int(len(out_labels)*0.8)
batch_size = 16
epochs = 25


# Call once as tensorflow likes this

# In[ ]:


mpnn.call({'adj_input' : in_edges_train[:10], 'nod_input': nodes_train[:10]})


# ## Let the learning begin!

# In[ ]:


mpnn.fit({'adj_input' : in_edges_train[:train_size], 'nod_input': nodes_train[:train_size]}, y = out_labels[:train_size], batch_size = batch_size, epochs = epochs, 
         callbacks = [lrate, stop_early], use_multiprocessing = True, initial_epoch = 0, verbose = 2, 
         validation_data = ({'adj_input' : in_edges_train[train_size:], 'nod_input': nodes_train[train_size:]},out_labels[train_size:]) )


# ## Predict on the test set

# In[ ]:


preds = mpnn.predict({'adj_input' : in_edges_test, 'nod_input': nodes_test})


# In[ ]:


np.save("preds_kernel.npy" , preds)


# # Prediction done!
# 
# Now rescale outputs and create submission.csv

# In[ ]:


train = pd.read_csv(datadir + "champs-scalar-coupling/train.csv")
test = pd.read_csv(datadir + "champs-scalar-coupling/test.csv")

test_group = test.groupby('molecule_name')

scale_min  = train['scalar_coupling_constant'].min()
scale_max = train['scalar_coupling_constant'].max()
scale_mid = (scale_max + scale_min)/2
scale_norm = scale_max - scale_mid


# In[ ]:


def make_outs(test_group, preds):
    i = 0
    x = np.array([])
    for test_gp, preds in zip(test_group, preds):
        if (not i%1000):
            print(i)

        gp = test_gp[1]
        
        x = np.append(x, (preds[gp['atom_index_0'].values, gp['atom_index_1'].values] + preds[gp['atom_index_1'].values, gp['atom_index_0'].values])/2.0)
        
        i = i+1
    return x


# In[ ]:


max_size = 29
preds = preds.reshape((-1,max_size, max_size))


# In[ ]:


out_unscaled = make_outs(test_group, preds)


# In[ ]:


test['scalar_coupling_constant'] = out_unscaled
test['scalar_coupling_constant'] = test['scalar_coupling_constant']*scale_norm + scale_mid
test[['id','scalar_coupling_constant']].to_csv('submission.csv', index=False)

