#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/slyofzero/ML-algorithms-from-SCRATCH/blob/main/Logistic_Regression_from_Scratch.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# <a href = "https://ml-concepts.com">
# <img src = "https://drive.google.com/uc?export=view&id=178rD8AqMa8xsEKLYw2IEpswftKKKHznZ">
# </a>

# #**AIM** - 
# 
# To create a Machine Learning algorithm that uses Logistic Regression, from
# scratch.
# 
# ---

# Logistic Regression is a type of **linear model** that's mostly used for **binary classification** but can also be used for **multi-class classification**. If the term linear model seems something familiar, then that might be because Linear Regression is also a type of a linear model.
# 
# To proceed with this notebook you firstly have to make sure that you understand ML concepts like Linear Regression, Cost Function, and Gradient Descent and mathematical concepts like Logarithm and Matrices. If you don't, then the links below can help you out.
# 
# 1. [Linear Regression](https://medium.com/ml-concepts/linear-regression-101-f4c27fb7a586)
# 
# 2. [Gradient Descent](https://medium.com/ml-concepts/gradient-descent-6a449eae1095)
# 
# ---

# In[ ]:


# !mkdir ~/.kaggle
# !cp /content/drive/MyDrive/C.S/Kaggle/kaggle.json ~/.kaggle/kaggle.json
# !kaggle datasets download -d dragonheir/logistic-regression


# In[ ]:


# !unzip /content/logistic-regression.zip


# In[ ]:


# Importing neccessary modules.
import numpy as np
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go

import warnings
warnings.filterwarnings("ignore")


# In[ ]:


# Loading the data.
df = pd.read_csv("/content/Social_Network_Ads.csv")
df.drop(columns = ["User ID", "Gender"], inplace = True)
df = df.rename(columns = {"Age":"Feature 1", "EstimatedSalary":"Feature 2", "Purchased":"Target"})
df.head()


# In[ ]:


# Scaling the data.
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df.iloc[:, :-1] = pd.DataFrame(scaler.fit_transform(df.iloc[:, :-1]), columns = df.columns[:-1])
df.head()


# In[ ]:


# Plotting the target values.
fig = px.scatter(data_frame = df, x = "Feature 1", y = "Target", color = "Target", color_continuous_scale = ["red", "blue"])
fig.show()


# As you can see from the plot above, the data we have can only have two values 0 and 1, no other. If we can plot a line right in the middle of the graph then we'll be able to seperate the two classes easily.
# 
# Let's do this using Linear Regression.

# In[ ]:


# Using Linear Regression to seperate the classes.
from sklearn.linear_model import LinearRegression

features = df[["Feature 1", "Feature 2"]].values.reshape(df.shape[0], -1)
target = df["Target"].values.reshape(df.shape[0], -1)

linreg = LinearRegression()
linreg.fit(features, target)

beta_0 = linreg.intercept_
betas = linreg.coef_.ravel()


# In[ ]:


# Plotting the line.
pred = betas[0] * df["Feature 1"] + beta_0

fig = px.scatter(data_frame = df, x = "Feature 1", y = "Target", color = "Target", color_continuous_scale = ["red", "blue"])
fig.add_traces(px.line(data_frame = df, x = "Feature 1", y = pred, color_discrete_sequence = ["orange"]).data)
fig.show()


# As you can see, the predicted values from the line above...can range from -$∞$ all the way to +$∞$. Which is not what we want. We want the predictions to be limited from 0 to 1 only. That way we'll be able to have the probability of something belonging to class 1 (in our case it'll be the probability of that person buying insurance). If the prediction is 0.5 then there would be a 50% chance of that person buying insurance. If the prediction is 0.9, then there would be a 90% chance of that person buying insurance.
# 
# Lucky for us, there already exists a function which does the exact same thing. This function is called the **Sigmoid Function**.
# 
# ---

# #**Sigmoid Function**
# 
# The sigmoid function looks something like this,
# 
# $$h(x) = \frac{1}{1 + e ^ {-x}}$$
# 
# It takes in any series and gives out that series in the terms of probabilities, which restricts it from 0 to 1. Let's take an example of this.

# In[ ]:


# Creating the sigmoid function.
def sigmoid(series):
  return 1 / (1 + np.exp(-series))


# In[ ]:


# Using the sigmoid function over a series.
np.random.seed(42)
series = np.random.normal(0, 10, 100)

fig1 = px.scatter(x = series, y = np.linspace(series.min(), series.max(), 100), title = "Original Series")
fig1.show()

fig2 = px.scatter(x = series, y = sigmoid(series), title = "Sigmoidal Series")
fig2.show()


# For Logistic Regression, we would be inputing $\beta_0 + \beta_1 x_1 + \beta_2 x_2$ into the sigmoid function. Which would the equation look something like this.
# 
# $$p = \frac{1}{1 + e ^ {-(\beta_0 + \beta_1 x_1 + \beta_2 x_2)}}$$
# 
# This equation can be written in the terms of matrices.
# 
# $$p = \frac{1}{1 + e^{-BX^{T}}}$$
# 
# Where -
# 
# 
# -  $B$ is the matrix with all the regression coefficients
# 
#     $$B = [\beta_0, \beta_1, \beta_2]$$
# 
# -  $X$ is the matrix with all the feature values with an added column with 1s.
# 
#   $$X = \begin{bmatrix} 1 & x_{1,1} & x_{2,1} \\ 1 & x_{1,2} & x_{2,2}\\ 1 & x_{1,3} & x_{2,3} \\ \vdots & \vdots & \vdots \\ 1 & x_{1,1000} & x_{2,1000} \end{bmatrix}$$
# 
# The sigmoid function can help us in differentiating two classes but only when we have the equation of the ideal line to pass into the function.
# 
# How can we get the equation of the ideal line? It's simple. By minimzing the cost function for Logistic Regression.
# 
# ---

# #**Cost Function**
# 
# Just like Linear Regression had MSE as its cost function, Logistic Regression has one too. So let's derive it.

# #**Likelihood Function**
# 
# So...we know that Logistic Regression is used for binary classification. Meaning the predictions can only be 0 or 1 (Either it belongs to a class, or it doesn't). So suppose, the probability of something belonging to class 1 is $p$, then probability of it belonging to class 0 would be $1 - p$.
# 
# \begin{split}
# & P(y = 1) = p \\
# & P(y = 0) = 1 - p
# \end{split}
# 
# We can combine these two equations into something like this.
# 
# \begin{split}
# & P(y) = p^y(1 - p)^{1-y} \\
# \end{split}
# 
# If we substitute $y$ with $1$ we get the following .
# 
# \begin{split}
# P(y = 1) 
# & = p^1 (1 - p)^{1 - 1} \\
# & ⇒ p (1 - p)^0 \\
# & ⇒ p
# \end{split}
# 
# If we substitute $y$ with $0$ we get the following .
# 
# \begin{split}
# P(y = 1) 
# & = p^0 (1 - p)^{1 - 0} \\
# & ⇒ p^0 (1 - p)^1 \\
# & ⇒ 1 - p
# \end{split}
# 
# \
# 
# This equation is called the **likelihood function**, and it can give us the likelihood of one item belonging to a class. To get the likelihood function of all the items in a series, we just can just multiply the likelihood of all the items.
# 
# \begin{split}
# P(y) 
# & = p_1^{y_1}(1 - p_1)^{1-y_1} × p_2^{y_2}(1 - p_2)^{1-y_2} × … × p_n^{y_n}(1 - p_n)^{1-y_n} \\[5px]
# & ⇒ \prod_{i = 1}^n \space p_i^{y_i}(1 - p_i)^{1-y_i}
# \end{split}

# #**Log Likelihood Function**
# 
# When we start applying it on a series, the likelihood function would have huge numbers in it. This would complexify our calculations. So to tackle this problem we can take the log of this function.
# 
# \begin{split}
# P(y) 
# & = p_1^{y_1}(1 - p_1)^{1-y_1} × p_2^{y_2}(1 - p_2)^{1-y_2} × … × p_n^{y_n}(1 - p_n)^{1-y_n} \\[5px]
# & = log(p_1^{y_1}(1 - p_1)^{1-y_1} × p_2^{y_2}(1 - p_2)^{1-y_2} × … × p_n^{y_n}(1 - p_n)^{1-y_n}) \\[5px]
# & = log(p_1^{y_1}(1 - p_1)^{1-y_1}) + log(p_2^{y_2}(1 - p_2)^{1-y_2}) + … + log(p_n^{y_n}(1 - p_n)^{1-y_n}) \\[5px]
# & = (log(p_1^{y_1}) + log(1 - p_1)^{1-y_1}) + (log(p_2^{y_2}) + log(1 - p_2)^{1-y_2}) + … + (log(p_n^{y_n}) + log(1 - p_n)^{1-y_n}) \\[5px]
# & = (y_1 \space log \space p_1 + (1-y_1)log(1 - p_1)) + (y_2 \space log \space p_2 + (1-y_2)log(1 - p_2)) + … + (y_n \space log \space p_n + (1-y_n)log(1 - p_n)) \\[5px]
# & = \sum_{i = 1}^{n} (y_i \space log \space p_i + (1-y_i)log(1 - p_i)
# \end{split}
# 
# \
# 
# This function takes in the values of $p_i$ and $1 - p_i$ which range from 0 to 1 (it takes in probabilities).
# 
# Let's plot a log of numbers that fall between 0 and 1.

# In[ ]:


# Log of numbers ranging from 0 to 1.
series = np.linspace(0, 1, 100)
px.scatter(x = series, y = np.log(series))


# As you can see the log of numbers from 0 to 1 is in negative. Meaning the whole function $P(y)$ would be negative for all the inputs. So we would multiply $-1$ with $P(y)$ to fix this.
# 
# And one more thing. $\sum_{i = 1}^{n} (y_i \space log \space p_i + (1-y_i)log(1 - p_i)$ gives us the sum of all errors and not the mean. So to fix this we can divide the whole equation by $n$ to get the mean of all errors.
# 
# $$J = -\frac1n \sum_{i = 1}^{n} (y_i \space log \space p_i + (1-y_i)log(1 - p_i)$$
# 
# And to avoid overfitting, let's add pennalisation to the equation just the way we added it to cost function for Ridge Regression.
# 
# 
# $$J = -\frac1n \sum_{i = 1}^{n} (y_i \space log \space p_i + (1-y_i)log(1 - p_i) + \frac\lambda n (\sum_{i = 1}^n \beta_i ^ 2)$$
# 
# This function we have here is also called as the **Regularized Cost Function** and can help us in getting the error values for certain value of $\beta$s.

# #**Gradient Descent**
# 
# Now that we have our Cost Function all we need to do is find the minimum value of it to get the best predictions. And we can do this by applying partial differentation on the function.
# 
# According to the Convergence Theorem, the ideal $\beta$ value can be calculated using the equation below.
# 
# $$\beta_n = \beta_n - \left(\frac{\partial J}{\partial \beta_n}\right) * L$$
# 
# All we need to do is find the value of $\frac{\partial J}{\partial \beta_n}$ for each $\beta$ and we are good to go.
# 
# \
# 
# \begin{align}
# J &= - \frac{1}{n} \sum_{i = 1} ^{n} (y_i \log p_i  + (1 - y_i) \log(1 - p_i)) + \frac{\lambda}{n}(\beta_1^2 + \beta_2^2 + \dots + \beta_n^2)\\
# \frac{\partial J}{\partial \beta_0} &=  - \frac{1}{n} \sum_{i = 1} ^{n} \left( y_i \times \frac{1}{p_i} \times \frac{\partial p_i}{\partial \beta_0} + (1 - y_i) \times \frac{-1}{1 - p_i} \times \frac{\partial p_i}{\partial \beta_0} \right) + \frac{\lambda}{n}\times 0
# \end{align}
# 
# \
# 
# We know that
# 
# \begin{align}
# p_i &= \frac{1}{1 + e^{-(\beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_n x_n)}} \\
# \therefore \frac{\partial p_i}{\partial \beta_0} &= - \left( \frac{1}{1 + e^{-(\beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_n x_n)}} \right)^2 (0 + e^{-(\beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_n x_n)})(-1 + 0) \\
# &= \frac{e^{-(\beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_n x_n)}}{(1 + e^{-(\beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_n x_n)})^2} \\
# \end{align}
# 
# \
# 
# On adding $-1$ and $1$ to the above equation, we get
# 
# \begin{align}
# \therefore \frac{\partial p_i}{\partial \beta_0} &= \frac{1 + e^{-(\beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_n x_n)} - 1}{(1 + e^{-(\beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_n x_n)})^2} \\
# &= \frac{1}{1 + e^{-(\beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_n x_n)}} - \frac{1}{(1 + e^{-(\beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_n x_n)})^2} \\
# &= p_i - p_i^2 \\
# &= p_i(1 - p_i)
# \end{align}
# 
# \
# 
# On substituting $\frac{\partial p_i}{\partial \beta_0}$ in the derivative of the cost function with respect to $\beta_0$, we get
# 
# \begin{align}
# \frac{\partial J}{\partial \beta_0} &=  - \frac{1}{n} \sum_{i = 1} ^{n} \left( y_i \times \frac{1}{p_i} \times p_i(1 - p_i) + (1 - y_i) \times \frac{-1}{1 - p_i} \times p_i(1 - p_i) \right) \\
# &=  - \frac{1}{n} \sum_{i = 1} ^{n} \left( y_i (1 - p_i) - (1 - y_i) p_i\right) \\
# &=  - \frac{1}{n} \sum_{i = 1} ^{n} (y_i  - y_i p_i - p_i + y_i p_i) \\
# \Rightarrow \frac{\partial J}{\partial \beta_0} &=  \frac{1}{n} \sum_{i = 1} ^{n} (p_i  - y_i)
# \end{align}
# 
# \
# 
# Similarly, if you differentiate $J$ with respect to to $\beta_1$, you will get 
# 
# \begin{equation}
# \frac{\partial J}{\partial \beta_1} =  \frac{1}{n} \sum_{i = 1} ^{n} (p_i  - y_i)x_1 + \frac{\lambda}{n} \beta_1 \\
# \end{equation}
# 
# \
# 
# In general, for $\beta_n$ you will get
# 
# \begin{equation}
# \frac{\partial J}{\partial \beta_n} =  \frac{1}{n} \sum_{i = 1} ^{n} (p_i  - y_i)x_n + \frac{\lambda}{n} \beta_n \\
# \end{equation}

# Now let's code this into Python.

# In[ ]:


# Creating a function for the cost function.
def reg_cost_function(betas, x, y, penn_const):
  n = x.shape[0]
  series = np.matmul(betas, np.transpose(x))
  p = sigmoid(series)
  y = y.reshape(1, n)

  class_0 = y * np.log(p)
  class_1 = (1 - y) * np.log(1 - p)

  cf = (-1 / n) * np.sum(class_0 + class_1)
  pennalize = penn_const/(2 * n) * np.sum(betas[:, 1:]**2)

  grad = np.zeros((1, betas.shape[1]))

  for i in range(betas.shape[1]):
    grad[0, i] = (1/n) * np.matmul(p - y, x[:, i])

  return (cf + pennalize, grad)


# In[ ]:


# Creating a function for gradient descent.
def gradient_descent(X, y, beta, learn_rate, num_iters, penn_const):
  m = X.shape[0]
  cost_func_values = []
  
  for i in range(num_iters):
    cost, grad = reg_cost_function(beta, X, y, penn_const)
    beta[0][0] = beta[0][0] - learn_rate * grad[0][0]
    beta[0][1] = beta[0][1] - learn_rate * (grad[0][1] + penn_const * beta[0][1] / m)
    beta[0][2] = beta[0][2] - learn_rate * (grad[0][2] + penn_const * beta[0][2] / m)
    cost_func_values.append(cost)

  return beta, cost_func_values


# Now that we have both the function ready, let's use them to seperate the two classes.

# In[ ]:


# Seperating the features and targets.
features_array = df.iloc[:, :-1].values
target_array = df.iloc[:, -1].values


# In[ ]:


# Getting the function inputs ready.
x = np.append(np.ones((features_array.shape[0], 1)), features_array, axis = 1)
y = target_array
betas = np.zeros((1, x.shape[1]))


# In[ ]:


# Applying the functions.
results = gradient_descent(x, y, betas, 0.001, 10000, 10)


# In[ ]:


# Checking the beta values.
betas = results[0].ravel()
betas


# Now that we have the ideal beta values for the ideal line, let's use them to plot a line and pass it inside the sigmoid function.

# In[ ]:


# Plotting the new line for seperation.
pred = betas[1] * df["Feature 1"] + betas[0]

fig = px.scatter(data_frame = df, x = "Feature 1", y = "Target", color = "Target", color_continuous_scale = ["red", "blue"])
fig.add_traces(px.scatter(data_frame = df, x = "Feature 1", y = sigmoid(pred), color_discrete_sequence = ["orange"]).data)
fig.show()


# As you can see, the issue of having predictions ranging from $- ∞$ to $+ ∞$ is solved here. 
# 
# Now let's plot a decision boundary to seperate the two classes. But wait...what is a decision boundary? 

# #**Decision Boundary**
# 
# In the simplest terms, a decision boundary is just a line that can help us in identifying which point belongs to which class. The image below can help you understand a decision boundary much more clearly.
# 
# \
# 
# <center><img src = "https://www.jeremyjordan.me/content/images/2017/06/Screen-Shot-2017-06-10-at-9.41.25-AM.png" width = 50%></center>
# 
# \
# 
# Here the blue line seperates the two classes which are represented as green and red dots. Any point to the left of the decision boundary belongs to the class represented with the red dots. Any point to the right belongs to the classs represented with the green dots. That's all what a decision boundary does.
# 
# \
# 
# It can be calculated using the equation of the straight line itself. The equation of the straight line in the general form can be given as this -
# 
# $$ax + by + c = 0$$
# 
# Where,
# 
# - a is the coefficient of x,
# - b is the coefficient of y.
# - c is some arbitrary constant.
# 
# Using this equation we can assume that the equation of the decision boundary is -
# 
# $$\beta_0 + \beta_1 x_1 + \beta_2 x_2 = 0$$
# 
# Where,
# 
# - $x_1$ is the $1^{st}$ feature variable.
# - $x_2$ is the $2^{nd}$ feature variable.
# 
# If we are able to calculate $x_2$ values for certain $x_1$ values, then we would be able to plot our decision boundary. This can be done this way.
# 
# \begin{split}
# & \beta_0 + \beta_1 x_1 + \beta_2 x_2 = 0 \\[5px]
# & \Rightarrow \beta_2 x_2 = -(\beta_0 + \beta_1 x_1) \\[5px]
# & \Rightarrow x_2 = - \left(\frac{\beta_0 + \beta_1 x_1}{\beta_2}\right)
# \end{split}
# 
# Let's code this using Python.
# 
# 

# In[ ]:


# Using the betas to plot a decision boundary.
pred = (-betas[0] -betas[1]*df["Feature 1"])/betas[2]

fig = px.scatter(data_frame = df[df["Target"] == 0], x = "Feature 1", y = "Feature 2")
fig.add_traces(px.scatter(data_frame = df[df["Target"] == 1], x = "Feature 1", y = "Feature 2", color_discrete_sequence = "red", color = "Target").data)
fig.add_traces(px.line(x = df["Feature 1"], y = pred, color_discrete_sequence = ["orange"]).data)
fig


# As you can see, the decision boundary we plotted is able to seperate the two classes. Most of the points to the left of the boundary belong to the class 0, while most of the points to the right of the boundary belong to class 1.

# We can also get the predictions for each feature variable by just mutliplying the $B$ and $X^T$ matrices. Once we have these predictions, we'll have to convert them into a sigmoidal series which would give us the probabilty of of an item belonging to class 1. In our example class 1 is $y = 1$, or the case in which a person bought something.

# In[ ]:


# Getting the predictions.
features_array = np.append(np.ones((features_array.shape[0], 1)), features_array, axis = 1)
best_fit_line = np.matmul(betas, np.transpose(features_array))
sigmoid_outputs = sigmoid(best_fit_line)


# In[ ]:


# Assigning classes to each prediction.
y_pred = np.array([1 if output >= 0.5 else 0 for output in sigmoid_outputs])
y_pred


# Now that we have assigned each item a class based upon the predictions, let's check the accuracy of our predictions.

# In[ ]:


# Checking the accuracy.
correct_preds_count = np.unique((y == y_pred), return_counts = True)[1][1]
accuracy = (correct_preds_count/y.shape[0]) * 100
accuracy


# Our model gave us $84.25 \%$ accuracy. That's a good score!!
# 
# ---

# We have successfully created an ML algorithm that uses Logistic Regression.
# 
# #END OF THE NOTEBOOK
