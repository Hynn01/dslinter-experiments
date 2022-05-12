#!/usr/bin/env python
# coding: utf-8
# Table of contents
1. [Introduction](#introduction)
    1. [Source code](#sourcecode)
    2. [Code housekeeping](#housekeeping)
2. [Intuitive description of neural network](#intuitive)
    1. [Fitting a neural network](#fitting)
3. [Visualising the hidden layer](#visualisation_hidden)
    1. [Effect of the regularisation parameter](#regparam)
4. [Visualising the output layer](#visualisation_output)
5. [Comparison with scikit-learn neural network](#sklearn)
6. [Conclusion](#conclusion) 
7. [References](#references)
# <a id="introduction"></a>
# # 1. Introduction 
# Some years ago I completed Andrew Ng's [Machine Learning course on coursera](https://www.coursera.org/learn/machine-learning), an improved and updated version of which has now migrated to [deeplearning.ai](https://www.deeplearning.ai/program/machine-learning-specialization/). One part of the coursera course that stood out for me was the Neural Networks module. In it, code for a neural network was set up to identify hand-written digits (which is more or less the same dataset as this kaggle competition), and I've been meaning to try the learnings and code out on the digit recognition competition here on kaggle.
# 
# My main motivations for creating this notebook were to:
# * Port the Octave code of the neural network model from the coursera course into python, taking advantage of some of python's computational efficiencies and aligning the functional interface with the scikit-learn estimator API,
# * Provide an intuitive description of a simple (single-layer) neural network and how we can use it for the Digit Recognizer competition,
# * Compare the performance of the pedagogical implementation of a neural network from coursera to scikit-learn's operational implementation (i.e. [sklearn.neural_network.MLPClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html)), and
# * Publish a nicely formatted kaggle notebook that may be of interest to other people. (Shubham Singh's [Create Beautiful Notebooks : Formatting Tutorial](https://www.kaggle.com/code/shubhamksingh/create-beautiful-notebooks-formatting-tutorial) helped with this!)
# 
# <a id="sourcecode"></a>
# ## 1A. Source code 
# The neural network code itself is located in a [script file](https://www.kaggle.com/code/nnjjpp/neuralnetworkclassifier). In order to use it, it needs to be added as a "utility script" through the File menu of your notebook. Then it can be imported like a regular python module:
# <pre>
# import neuralnetworkclassifier as nn
# neural_network = nn.NeuralNetwork()
# </pre>
# 
# I endeavoured to create the neural network class similarly to a [scikit-learn estimator](https://scikit-learn.org/stable/developers/develop.html), so that things like the `fit()` and `predict()` method are used in the same way to estimators like RandomForestClassifier etc. Briefly, the main components of the code are the `predict()` method, the `cost()` method, which computes the error and the partial derivatives of the loss function for a given set of parameters, and the `fit()` method, which finds optimal parameters for the model using [scipy's conjugate gradient algorithm](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_cg.html). Rather than using loops, or even matrix multiplication directly, I have implemented the calculations using [numpy's einsum](https://numpy.org/doc/stable/reference/generated/numpy.einsum.html), which is generally a super-fast way of organising matrix multiplications and summation.
# 
# <a id="housekeeping"></a>
# ## 1B. Housekeeping 
# First let's load some modules and the data (and print a sample of some of the data). I also normalise the features (pixel alpha channels) to lie between 0 and 1, although that is not necessary for a neural network.

# In[ ]:


import os
import numpy as np 
import pandas as pd 
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import time

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
from sklearn.preprocessing import OneHotEncoder
one = OneHotEncoder()
y = one.fit_transform(np.array(train['label']).reshape(-1,1)).todense()
X = train.drop(['label'],axis=1)

# Normalise features so that values lie between 0 and 1, not strictly necessary:
X /= 255

test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')


# Next, set some parameters that we use in the model fitting. Note that I found optimal regularisation parameters using cross-validation in a separate (private) notebook. 

# In[ ]:


## PARAMETERS:

LAM = 5.373 # Cross validation estimate from https://www.kaggle.com/code/nnjjpp/fork-of-minimal-neural-network-mine
ALPHA = 0.1428 # Cross validation estimate from https://www.kaggle.com/code/nnjjpp/optimise-reg-parameter-sklearn
N_PERCEPTRONS = 12 # Number of nodes in hidden layer 
N_ITER = 120 # Number of iterations for fitting the model
N_SPLITS = 3 # Number of splits for cross validation

PRINT_ITERATIONS = 60 # Print fitting progress after this many iterations


# The input data in the Digit Recognizer competition are 28 x 28 grids of pixels, with element of the matrix originally having a value between 0 and 255 representing pixel intensity, which we normalised in the code cell above to lie between 0 and 1. Plotting a sample of the data using seaborn's heatmap plotting function shows some examples:

# In[ ]:


plt.figure(figsize=(15,3))
plt.plot(1,1)
Xc = {i: X.loc[train['label'] == i,:] for i in range(10)}
for i in range(20):
    plt.subplot(2,10,i+1)
    sns.heatmap(np.reshape(np.array(Xc[i%10].iloc[i//10,:]), (28,28)),
                xticklabels=False,yticklabels=False,cbar=False,cmap='binary')


# <a id="intuitive"></a>
# # 2. Intuitive description of neural network 
# There are plenty of fantastic descriptions of neural networks around the web (not least of which is the coursera course); Efron and Hastie (2021, chapter 18) have all the relevant equations (some of which was adapted from Andrew Ng's notes). Here, I provide a brief, intuitive description of a neural network, and how it relates to the digit recognition problem. 
# 
# In one sentence, a neural network searches for rules (or patterns in the data) identifying particular digits and combines them nonlinearly to produce classification probabilities.
# 
# If we were to try and formulate how we might differentiate hand-drawn digits, we might start with the following rules:
# 
# * A '1' has a vertical line in the middle of the image, and, usually, a lack of horizontal lines at the top and bottom,
# * A '7' has a vertical line at the top and no vertical line at the bottom
# 
# and so on.
# 
# Luckily we don't have to code up these rules (this is machine learning after all!) - fitting a neural network lets the data determine the common pixel patterns and how these relate to the classes (digit types). The fascinating part about neural networks is that, unlike some other machine learning algorithms, we can have a look at the internals of the model (i.e. the hidden and output layers) and directly gain some insight into how the algorithm determined these rules (more on this later...)
# 
# In this workbook (and the associated source code), we only look at a single layer neural network. In practice, multiple hidden layers are normally used. Efron and Hastie (2021) recommend two hidden layers for the digit recognition problem, yet common wisdom seems to suggest that [no more than two hidden layers are necessary in any problem](https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw). 
# 
# If we begin with the input data as a $(n_r, n_f+1)$ sized matrix $X$, where $n_r$ is the number of records, in this example the number of digit images in the dataset, and $n_f$ is the number of features (pixels) - the $+1$ is for an intercept or *bias* term - then the matrix $W$ corresponding to the hidden layer has $n_f+1$ rows and $n_l$ columns, where $n_l$ is the number of labels (i.e. digits we are trying to predict, in this case 10). Then
# 
# $$Z=XW$$
# 
# gives a linear prediction, to which we then apply an activation function:
# 
# $$A=f(Z)$$
# 
# Here we have $f$ as a [sigmoid function](https://en.wikipedia.org/wiki/Sigmoid_function), which converts the linear predictions into probabilistic predictions lying between zero and one. The output layer (and any additional hidden layers) can be implemented in a similar way by adding a bias column (intercept) to $A$ and using it as input to the next layer. Better and more detailed descriptions of the mathematics can be found in one of the references [below](#references).
# 
# ![Single layer neural network schematic showing input layer, hidden layer and output layer](https://upload.wikimedia.org/wikipedia/commons/9/99/Neural_network_example.svg)
# 
# <cite>User:Wiso, Public domain, via Wikimedia Commons</cite>
# 
# In the Digit Recognizer competition, the input layer is a flattened image (i.e. a vector of pixels), the hidden layers are specific combinations of pixels and the output layer (shown in the schematic above as one node, but in our example would consist of ten nodes - one for each type of digit) are class probabilities.
# 
# <a id="fitting"></a>
# ## 2A. Fitting a neural network 
# Neural networks are typically fitted using an iterative method to minimise errors (i.e. given that we know the actual class of each image, we can iteratively improve the parameters of the model so that the class probabilities match the known class as closely as possible). For each iteration, the method calculates class probabilities (in the "feed forward" step) and calculates the errors as a cost function. Then the partial derivatives of the error (i.e. each parameter's contribution to the error) are calculated in the "back propagation" step. The loss and the loss gradient are then sent to an iterative optimisation algorithm (e.g. gradient descent function minimisation), which finds progressively better parameters (weights) that minimise the errors. 
# 
# Note that, in operational implementations of a neural network,
# 
# * Sigmoid functions are rarely used now as they are too expensive to compute, and better alternatives exist (such as the [relu curve](https://en.wikipedia.org/wiki/Rectifier_(neural_networks))).
# * Conjugate gradient iteration is not typically used either nowadays; better and faster iterative procedures are used. For example, scikit-learn MLPClassifier uses as a default the [Adam](https://en.wikipedia.org/wiki/Stochastic_gradient_descent#Adam) method.

# In[ ]:


get_ipython().run_cell_magic('time', '', "import neuralnetworkclassifier as nn\nfrom sklearn.model_selection import StratifiedKFold\nfrom sklearn.metrics import accuracy_score\n\ncv = StratifiedKFold(n_splits = N_SPLITS, \n                    shuffle = True, \n                    random_state = 1)\ntimes = []\nprint('Fitting model with stratified 3-fold split:')\n\nfor foldnum, (train_ix, test_ix) in enumerate(cv.split(X,train['label'])):\n    model = nn.NeuralNetwork(hidden_layer_size=N_PERCEPTRONS,Lam=LAM,\n                                 n_iter=N_ITER,print_progress_mod = PRINT_ITERATIONS)\n\n    X_train = X.iloc[train_ix,:]\n    y_train = train['label'][train_ix]\n    X_test = X.iloc[test_ix,:]\n    y_test = train['label'][test_ix]\n\n    tt = time.time()\n    # Fit the model:\n    model.fit(X_train, y_train, save_retall=True*(foldnum==N_SPLITS-1)) \n    # For the final fold, return all the output from the conjugate gradient procedure\n    # In particular, this gives the parameters for the model at intermediate\n    # iterations, and is used for comparison to the scikit-learn implementation below\n    \n    times.append(time.time() - tt)\n    y_pred = model.predict(X_test)\n    tt = time.time()\n\n    cost = model.cost(model.parameters['vparams'], \n                                X_train,\n                                y[train_ix], \n                                LAM)[0]\n    \n    print(f'Fold {foldnum+1:d}: Value of cost function = {cost:.3f}, ' \\\n          f'accuracy on test set = {accuracy_score(y_test, y_pred):.3f}, ' \\\n          f'time taken = {times[-1]:.1f}s')\n    ")


# <a id="visualisation_hidden"></a>
# # 3. Visualising the hidden layer
# 
# As mentioned previously, the hidden layer, being a matrix of weights, can be easily visualised, which can be useful for the interpretation of image processing applications, but is probably less useful for other machine learning problems. Specifically, each node in the hidden layer corresponds to a pattern of pixels that discriminates between different digits. Each node in the output layer relates the patterns in the hidden layer to different digits, either positively (e.g. the presence of the 'x' in the middle of the image should give us reasonable confidence that the digit is an '8'), or negatively (e.g. vertical lines at the top and/or bottom of the image suggest the digit is definitely not a '4'). An additional hidden layer picks out, and combines, features from the previous hidden layer. In our example, the initial hidden layer identifies pen strokes, and an additional hidden layer would identify [combinations of pen strokes](https://stats.stackexchange.com/questions/63152/what-does-the-hidden-layer-in-a-neural-network-compute).

# In[ ]:



plt.figure(figsize=(17.5,8.75))
plot_nc = int(np.ceil(np.sqrt(N_PERCEPTRONS)))

for n in range(N_PERCEPTRONS):
    # plot hidden layer from fitted model:
    plt.subplot(plot_nc, 2*plot_nc + 1, (n//plot_nc)*(2*plot_nc+1) + (n % plot_nc) + 1)
    sns.heatmap(np.reshape(np.array(model.parameters['Theta1'][n,1:]),(28,28)),
                xticklabels=False,yticklabels=False,cbar=False,cmap="RdGy",
                center=0)


# Here, positive weights in the hidden layer nodes are displayed as greyish pixels, and negative weights in the hidden layer are shaded red. Each node looks like part of a pen stroke, although some are more identifiable than others.

# <a id="regparam"></a>
# # 3A. Effect of the regularisation parameter 

# The main hyperparameter of a neural network is the regularisation parameter. This acts as a penalty to the model parameters in the layers of the neural network. The larger the parameter is, the smoother the hidden layer becomes. I fitted the neural network with smaller than optimal (left panel, below) and larger than optimal (right panel) regularisation parameter. (Note that the optimal value of the parameter is quite small so there is only a small difference between the previous hidden layer and the hidden layer with smaller regularisation parameter.)

# In[ ]:


get_ipython().run_cell_magic('time', '', "model_large_Lambda = nn.NeuralNetwork(hidden_layer_size=N_PERCEPTRONS,Lam=LAM*50,\n                                      n_iter=N_ITER,print_progress_mod = -1)\nmodel_large_Lambda.fit(X,train['label'])\nmodel_small_Lambda = nn.NeuralNetwork(hidden_layer_size=N_PERCEPTRONS,Lam=LAM/50,\n                                      n_iter=N_ITER,print_progress_mod = -1)\nmodel_small_Lambda.fit(X,train['label'])\ngcf = plt.figure(figsize=(25,8.75))\n#axs = gcf.subplots(nrows=3, ncols=13)\nfor n in range(N_PERCEPTRONS):\n    plt.subplot(3, 9, (n//4)*(2*4+1) + (n % 4) + 1)\n    sns.heatmap(np.reshape(np.array(model_small_Lambda.parameters['Theta1'][n,1:]),(28,28)),\n                xticklabels=False,yticklabels=False,cbar=False,\n                cmap='RdGy',center=0)\n    plt.subplot(3, 9, (n//4)*(2*plot_nc+1) + (n % 4) + 4+2)\n    sns.heatmap(np.reshape(np.array(model_large_Lambda.parameters['Theta1'][n,1:]),(28,28)),\n                xticklabels=False,yticklabels=False,cbar=False,\n                cmap='RdGy',center=0)")


# The hidden layer with the smaller value of the regularisation parameter is fitted too much to the peculiarities of the training dataset, and conversely the hidden layer with the larger value of the regularisation parameter pays less attention to individual differences in the training data.
# 
# An underfitted model (i.e. a too smooth hidden layer) from a too large regularisation parameter has high bias and low variance. This means the model doesn't take advantage of enough differences between digits in the training set and would produce a very similar model based on a different training dataset. 
# 
# An overfitted model (i.e. a variable hidden layer) from a too small regularisation parameter, in contrast, has low bias and high variance. It is fitted to the specific differences within the training dataset and would produce a different model when fed alternative training data.
# 
# Both underfitted and overfitted models struggle to accurately predict new data that is unseen by the fitting process (i.e. the test dataset). This is summarised in the bias-variance tradeoff figure from the [wikipedia article](https://en.wikipedia.org/wiki/Bias%E2%80%93variance_tradeoff):
# 
# ![Bias-variance tradeoff](https://upload.wikimedia.org/wikipedia/commons/9/9f/Bias_and_variance_contributing_to_total_error.svg)
# 
# <cite>User:Bigbossfarin, CC0, via Wikimedia Commons</cite>
# 
# Here, larger values of the regularisation parameter produce models that are not complex enough (left-hand side of the figure), and smaller values produce overly complex models (right-hand side of the figure). For prediction, we need to optimise the regularisation parameter to reduce predictive uncertainty, and cross-validation is used for this. I found optimal values of the regularisation parameter beforehand (not shown) and used these as input parameters to the model (i.e. the parameter `LAM`). For demonstration purposes, however, the underfitted model produces more interpretable figures, and I use the larger-than-optimal value of the regularisation parameter below for visualising the hidden and output layers.

# <a id="visualisation_output"></a>
# # 4. Visualising the output layer 
# 
# The purpose of the output layer is to combine weighted sums of the hidden layer and relate them to the labels seen during training. In some sense, an image is compared to each of the nodes in the hidden layer and a similarity score is obtained. These similarity scores are then added either positively or negatively and the probability of being each kind of digit is obtained.

# In[ ]:



gcf=plt.figure(figsize=(25,25/11*13))
axs = gcf.subplots(nrows=N_PERCEPTRONS+1, ncols=11)
gs = axs[1, 1].get_gridspec()
axs[0,0].remove()
for col in range(1,11):
    for ax in axs[:, col]:
        ax.remove()

for i in range(10): # Plot example digits in column headers of figure
    plt.subplot(N_PERCEPTRONS+1, 11, i+2)
    sns.heatmap(np.reshape(np.array(Xc[i%10].iloc[16,:]), (28,28)),
                xticklabels=False,yticklabels=False,cbar=False,cmap='binary')

for n in range(N_PERCEPTRONS):
    # plot hidden layer from the neural network with larger Lambda parameter, which as mentioned above
    # is not good for prediction, but better for making more interpretable figures.
    plt.subplot(N_PERCEPTRONS+1, 11, 11*(n+1) + 1)
    sns.heatmap(np.reshape(np.array(model_large_Lambda.parameters['Theta1'][n,1:]),(28,28)),
                xticklabels=False,yticklabels=False,cbar=False,
                cmap='RdGy', center=0)
    axlong = gcf.add_subplot(gs[n+1, 1:])
    sns.heatmap(model_large_Lambda.parameters['Theta2'][:,(n+1)][np.newaxis,:],
         xticklabels=False,yticklabels=False,cbar=False,center=0,cmap='Spectral')


# Here nodes in the hidden layers that look like parts of each digit have large, positive values in the output layer (visualised as dark red squares above), and hidden layers that when activated suggest the image is *not* a particular digit have large, negative values in the output layer (visualised as dark purple squares). Nodes providing little evidence either for or against particular digits are lighter, green/yellow colours. The hidden layer nodes are not always recognisable as digits or parts of digits, because of the variety of shapes and positions of the input data. However, a few recognisable features can be picked out. Perhaps the easiest to see is the hidden layers associated with ones. Nodes providing positive evidence for a one tend to have vertical strokes in the middle of the image, and nodes providing evidence that the digit is not a one tend to have vertical strokes at the top and bottom of the image.
# 
# An additional hidden layer between the hidden and output layers would work by combining features in the hidden layer in a non-linear fashion, which are then activated by the output layer.

# <a id=sklearn></a>
# # 5. Comparison with scikit-learn neural network 
# 
# Scikit-learn has an implementation of a neural network for classification, namely [`MLPClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html). There are a few differences compared to the implementation here, most notably the activation function and the optimisation procedure. Obviously, much work has gone into optimising the code, and here I compare the two implementations for speed and accuracy. I fit the scikit-learn implementation with identical layer sizes and iterations and compare the time taken and the accuracy on training and test sets.

# In[ ]:


get_ipython().run_cell_magic('time', '', "if False:\n    # Refit the neural network, calculating the accuracies at each iteration:\n    model = nn.NeuralNetwork(hidden_layer_size=N_PERCEPTRONS,Lam=LAM,\n                                     n_iter=N_ITER,print_progress_mod = -1)\n    model.fit(X_train, y_train, save_retall=True)\n\ntrain_accuracies = []\ntest_accuracies = []\n# In the last fold of fitting, output from the conjugate gradient method\n# was saved in the fmin_cg_retall attribute, which I use to calculate\n# accuracies at each iteration:\nfor i,vp in enumerate(model.fmin_cg_retall[1]):\n    model.parameters['vparams'] = vp\n    model.unravel_parameters(28*28, 10)\n    train_y_pred_step = model.predict(X_train)\n    train_accuracies.append(accuracy_score(y_train, train_y_pred_step))\n    test_y_pred_step = model.predict(X_test)\n    test_accuracies.append(accuracy_score(y_test, test_y_pred_step))\n\n# Reset the model to have the final parameter values:\nmodel.parameters['vparams'] = model.fmin_cg_retall[0]\nmodel.unravel_parameters(28*28, 10)\n\n\n# First fit the scikit-learn model with 3-fold cv to get the time \n# taken for the optimisation.\nfrom sklearn.neural_network import MLPClassifier \n\nsklearn_times = []\nprint('Fitting sklearn model with stratified 3-fold split:')\nfor foldnum, (train_ix, test_ix) in enumerate(cv.split(X,train['label'])):\n    sklearn_model = sklearn_nn = MLPClassifier(hidden_layer_sizes = (N_PERCEPTRONS,),\n                           alpha = ALPHA,\n                           max_iter = N_ITER,\n                           random_state = 1,verbose = False)\n\n    X_train = X.iloc[train_ix,:]\n    y_train = train['label'][train_ix]\n    X_test = X.iloc[test_ix,:]\n    y_test = train['label'][test_ix]\n\n    tt = time.time()\n    sklearn_model.fit(X_train, y_train)\n    sklearn_times.append(time.time() - tt)\n    sklearn_pred = sklearn_model.predict(X_test)\n\n    print(f'Fold {foldnum+1:d}: Value of cost function = {sklearn_model.loss_:.3f}, ' \\\n          f'accuracy on test set = {accuracy_score(y_test, sklearn_pred):.3f}, ' \\\n          f'time taken = {sklearn_times[-1]:.1f}s')\n\n# As far as I am aware, parameters cannot be easily obtained for each\n# iteration from the scikit-learn implementation. To get around this,\n# I refit the model using the partial_fit() method.\n# This carries out one iteration of the optimisation procedure at a time, which we\n# can use to calculate predictions within a loop:\n\nsklearn_partial_fit = MLPClassifier(hidden_layer_sizes = (N_PERCEPTRONS,),\n                                    alpha = ALPHA, \n                                    max_iter = N_ITER,\n                                    random_state = 1,\n                                    verbose=False)\n\nsklearn_train_accuracies = []\nsklearn_test_accuracies = []\ntt = time.time()\nunique_y = np.unique(train['label']) # Needed for first iteration of partial_fit() method\nfor i in range(N_ITER):\n    sklearn_partial_fit.partial_fit(X_train, y_train, classes=unique_y)\n    train_y_pred_step = sklearn_partial_fit.predict(X_train)\n    sklearn_train_accuracies.append(accuracy_score(y_train, train_y_pred_step))\n    test_y_pred_step = sklearn_partial_fit.predict(X_test)\n    sklearn_test_accuracies.append(accuracy_score(y_test, test_y_pred_step))")


# In[ ]:


print(f'Average time taken to fit {N_ITER} iterations for scikit-learn implementation:       {np.mean(sklearn_times):.1f}s')
print(f'Average time taken to fit {N_ITER} iterations for neuralnetworkclass implementation: {np.mean(times):.1f}s')
print('\n'*3)


matplotlib.rcParams.update({'font.size': 16})

fig = plt.figure(figsize=(14.5,7))
for n in range(N_PERCEPTRONS):
    # plot hidden layer from the scikit-learn implementation of neural network:
    fig.add_subplot(3, 8, (n%4)+1+8*(n//4))
    sns.heatmap(np.reshape(np.array(sklearn_partial_fit.coefs_[0][:,n]),(28,28)),
                xticklabels=False,yticklabels=False,cbar=False,cmap="RdGy",center=0)
ax=fig.add_subplot(1,2,1)
ax.axis('off')
ax.set_title('Hidden layers for scikit-learn implementation')

    
sklearn_colour='#66c2a5'
nnc_colour='#fc8d62' # Colours courtesy of colorbrewer2.org
ax = fig.add_subplot(1,2,2)
fig.subplots_adjust(left=None, bottom=None, right=1.2, top=None, wspace=None, hspace=None)
ax.set_title('Accuracies after each iteration')
ax.plot(sklearn_train_accuracies,linestyle='-',color=sklearn_colour,lw=3)
ax.plot(sklearn_test_accuracies,linestyle='--',color=sklearn_colour,lw=3)
ax.plot(train_accuracies,linestyle='-', color=nnc_colour,lw=3)
ax.plot(test_accuracies,linestyle='--',color=nnc_colour,lw=3)

_=ax.set_ylim([0.75,1])
_=plt.legend(['scikit-learn - training set',
              'scikit-learn - test set',
              'neuralnetworkclassifier - training set',
              'neuralnetworkclassifier - test set'],
            loc=10)
_=ax.set_ylabel('Accuracy')
_=ax.set_xlabel('Number of iterations')


# Comparing the neural network implementation in `neuralnetworkclassifier` and scikit-learn, we see that the hand-coded implementation takes 50% longer to compute 150 iterations, and while the accuracy ends up being similar, scikit-learn approaches the maximum accuracy values (particularly for the test-set accuracy) much quicker. This is most likely due to the improved optimisation procedure (i.e. adam rather than conjugate gradient). 

# # 6. Conclusion<a id="conclusion"></a>
# 
# We learnt that:
# 
# * The neural network from Andrew Ng's coursera course works, and is qualitatively similar in operation to scikit-learn's implementation of a neural network (i.e. `MLPClassifier`)
# * A neural network consists of an input layer (the data), one or more hidden layers (which picks out features of the input data), and an output layer (which takes similarity to hidden layer nodes as evidence for or against a particular digit).
# * The regularisation parameter controls the complexity of the fitted model, and can be optimised to reduce predictive error through the bias-variance tradeoff.
# * The scikit-learn implementation is faster per iteration, and converges to better predictions much quicker. (This is not surprising as the coursera implementation is pedagogical in nature, and a lot of work has gone into scikit-learn to create production-quality and optimised code.)

# # 7. References<a id="references"></a>
# Efron, B. and Hastie, T. (2021) Computer Age Statistical Inference: Algorithms, evidence and data science. Cambridge University Press, UK, Student Edition.
# 
# Ma, T. Avati, A., Katanforoosh, K. Ng, A. (2020) Deep Learning - CS229 Lecture Notes, Stanford University, 2020. https://cs229.stanford.edu/notes2020spring/
# 
# Coursera (2012) https://www.coursera.org/learn/machine-learning
# 
# deeplearning.ai (2022) https://www.deeplearning.ai/program/machine-learning-specialization/
# 
# FAtBalloon (https://stats.stackexchange.com/users/17679/fatballoon), What does the hidden layer in a neural network compute?, URL (version: 2017-03-03): https://stats.stackexchange.com/q/63152
