#!/usr/bin/env python
# coding: utf-8

# In this tutorial, you'll learn how to create advanced **scatter plots**.
# 
# # Set up the notebook
# 
# As always, we begin by setting up the coding environment.  (_This code is hidden, but you can un-hide it by clicking on the "Code" button immediately below this text, on the right._)

# In[ ]:



import pandas as pd
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
print("Setup Complete")


# # Load and examine the data
# 
# We'll work with a (_synthetic_) dataset of insurance charges, to see if we can understand why some customers pay more than others.  
# 
# ![tut3_insurance](https://i.imgur.com/1nmy2YO.png)
# 
# If you like, you can read more about the dataset [here](https://www.kaggle.com/mirichoi0218/insurance/home).

# In[ ]:


# Path of the file to read
insurance_filepath = "../input/insurance.csv"

# Read the file into a variable insurance_data
insurance_data = pd.read_csv(insurance_filepath)


# As always, we check that the dataset loaded properly by printing the first five rows.

# In[ ]:


insurance_data.head()


# # Scatter plots
# 
# To create a simple **scatter plot**, we use the `sns.scatterplot` command and specify the values for:
# - the horizontal x-axis (`x=insurance_data['bmi']`), and 
# - the vertical y-axis (`y=insurance_data['charges']`).

# In[ ]:


sns.scatterplot(x=insurance_data['bmi'], y=insurance_data['charges'])


# The scatterplot above suggests that [body mass index](https://en.wikipedia.org/wiki/Body_mass_index) (BMI) and insurance charges are **positively correlated**, where customers with higher BMI typically also tend to pay more in insurance costs.  (_This pattern makes sense, since high BMI is typically associated with higher risk of chronic disease._)
# 
# To double-check the strength of this relationship, you might like to add a **regression line**, or the line that best fits the data.  We do this by changing the command to `sns.regplot`.

# In[ ]:


sns.regplot(x=insurance_data['bmi'], y=insurance_data['charges'])


# # Color-coded scatter plots
# 
# We can use scatter plots to display the relationships between (_not two, but..._) three variables!  One way of doing this is by color-coding the points.  
# 
# For instance, to understand how smoking affects the relationship between BMI and insurance costs, we can color-code the points by `'smoker'`, and plot the other two columns (`'bmi'`, `'charges'`) on the axes.

# In[ ]:


sns.scatterplot(x=insurance_data['bmi'], y=insurance_data['charges'], hue=insurance_data['smoker'])


# This scatter plot shows that while nonsmokers to tend to pay slightly more with increasing BMI, smokers pay MUCH more.
# 
# To further emphasize this fact, we can use the `sns.lmplot` command to add two regression lines, corresponding to smokers and nonsmokers.  (_You'll notice that the regression line for smokers has a much steeper slope, relative to the line for nonsmokers!_)

# In[ ]:


sns.lmplot(x="bmi", y="charges", hue="smoker", data=insurance_data)


# The `sns.lmplot` command above works slightly differently than the commands you have learned about so far:
# - Instead of setting `x=insurance_data['bmi']` to select the `'bmi'` column in `insurance_data`, we set `x="bmi"` to specify the name of the column only.  
# - Similarly, `y="charges"` and `hue="smoker"` also contain the names of columns.  
# - We specify the dataset with `data=insurance_data`.
# 
# Finally, there's one more plot that you'll learn about, that might look slightly different from how you're used to seeing scatter plots.  Usually, we use scatter plots to highlight the relationship between two continuous variables (like `"bmi"` and `"charges"`).  However, we can adapt the design of the scatter plot to feature a categorical variable (like `"smoker"`) on one of the main axes.  We'll refer to this plot type as a **categorical scatter plot**, and we build it with the `sns.swarmplot` command.

# In[ ]:


sns.swarmplot(x=insurance_data['smoker'],
              y=insurance_data['charges'])


# Among other things, this plot shows us that:
# - on average, non-smokers are charged less than smokers, and
# - the customers who pay the most are smokers; whereas the customers who pay the least are non-smokers.
# 
# # What's next?
# 
# Apply your new skills to solve a real-world scenario with a **[coding exercise](https://www.kaggle.com/kernels/fork/2951535)**!

# ---
# 
# 
# 
# 
# *Have questions or comments? Visit the [course discussion forum](https://www.kaggle.com/learn/data-visualization/discussion) to chat with other learners.*
