#!/usr/bin/env python
# coding: utf-8

# Congratulations for making it to the end of the course!
# 
# In this final tutorial, you'll learn an efficient workflow that you can use to continue creating your own stunning data visualizations on the Kaggle website.

# ## Workflow
# 
# Begin by navigating to the site for Kaggle Notebooks:
# > https://www.kaggle.com/code
# 
# Then, in the top left corner, click on **[+ New Notebook]**.
# 
# ![tut7_new_kernel](https://i.imgur.com/kw9cct2.png)
# 
# This opens a notebook.  As a first step, check the language of the notebook by selecting **File > Language**.  If it's not Python, change the language to Python now.
# 
# ![tut7_default_lang](https://i.imgur.com/FcQhCjF.png)
# 
# The notebook should hvae some default code.  **_Please erase this code, and replace it with the code in the cell below._**  (_This is the same code that you used in all of the exercises to set up your Python environment._)

# In[ ]:


import pandas as pd
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
print("Setup Complete")


# The next step is to attach a dataset, before writing code to visualize it.  (_You learned how to do that in the previous tutorial._) 
# 
# Then, once you have generated a figure, you need only save it as an image file that you can easily add to your presentations!

# ---
# 
# 
# 
# 
# *Have questions or comments? Visit the [course discussion forum](https://www.kaggle.com/learn/data-visualization/discussion) to chat with other learners.*
