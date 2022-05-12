#!/usr/bin/env python
# coding: utf-8

# # Importing libraries

# In[ ]:


import numpy as np
import pandas as pd

import missingno

import matplotlib.pyplot as plt
import plotly.express as px

import warnings
warnings.filterwarnings('ignore')

styles = [dict(selector='caption', props=[('text-align', 'center'), ('font-size', '150%'), ('color', '#FFFFFF'), ('background-color' , '#6DA3C7')])] # For stylizing dataframes


# # Loading and preparing data

# In[ ]:


df = pd.read_csv('../input/lca-programs-h1b-h1b1-e3-visa-petitions/LCA_FY_2022.csv')

print(f'\nNumber of Visa petitions records available for 2022 (Q1) alone : \33[91m{df.shape[0]}\33[0m')
print(f'\nNumber of features present : \33[91m{df.shape[1]}\n')

df.head(3).style.set_caption('LCA_FY_2022 Data Schema').set_table_styles(styles)


# # Dealing Missing Values
# ## Exploring the pattern of missing values

# In[ ]:


df_na = pd.DataFrame(df.isna().sum()).reset_index().rename(columns = {'index': 'Feature Name', 0 : 'No.of Missing values'})

missing_val_columns = df_na[df_na['No.of Missing values'] != 0]['Feature Name']

for column in missing_val_columns:
    missing_count = df[column].isnull().sum()
    per_value = missing_count * 100 / df.shape[0]
    print(f'\33[94m{column}\33[0m is having {missing_count} missing values ie, \33[31m{per_value:.4f}%\33[0m of total data\n')
    
df_na.style.set_caption('Missing Value Count').set_table_styles(styles).background_gradient(cmap = 'Blues')


# In[ ]:


missingno.dendrogram(df)
plt.show()


# <div class="alert alert-warning" role="alert">
#    <font color='red'><strong>Observations</strong></font>
#     <ul>
#        <li>It's a clear indication that the missing values are from columns <strong>Worksite</strong> and <strong>Employer_Location</strong>.
#        <li>The dendrogram shows there is a correlation in missing value to the <strong>Employer_Country</strong>.
#         <li><strong>Lets try to go some more deeper</strong>.
#     </ul>
# </div>

# In[ ]:


df[df.Worksite.isnull()].reset_index().style.set_caption("'Worksite' - Missing values").set_table_styles(styles)


# **<font color='red'>Observations</font>**
# - Shows that if missing value is *Worksite*, then all the *Employer_Country* are *United States Of America*.
# - None of the *Employer_Location* are missing here.
# </div>

# In[ ]:


df_na_Employer_Location = df[df.Employer_Location.isnull()]
df_na_Employer_Location.reset_index(drop=True).style.set_caption("'Employer_Location' - Missing values").set_table_styles(styles)


# **<font color='red'>Observations</font>**
# 
# - Shows that if missing value is **Employer_Location**, then all the **Employer_Country** are different or unique.
# - Let's evaluate **Employer_Location** missing values based on **Employer_Country**.

# In[ ]:


# https://stackoverflow.com/a/39452138
class colors:
    '''Colors class:
    Reset all colors with colors.reset
    Two subclasses fg for foreground and bg for background.
    Use as colors.subclass.colorname.
    i.e. colors.fg.red or colors.bg.green
    Also, the generic bold, disable, underline, reverse, strikethrough,
    and invisible work with the main class
    i.e. colors.bold
    '''
    reset='\033[0m'
    bold='\033[01m'
    disable='\033[02m'
    underline='\033[04m'
    reverse='\033[07m'
    strikethrough='\033[09m'
    invisible='\033[08m'
    class fg:
        black='\033[30m'
        red='\033[31m'
        green='\033[32m'
        orange='\033[33m'
        blue='\033[34m'
        purple='\033[35m'
        cyan='\033[36m'
        lightgrey='\033[37m'
        darkgrey='\033[90m'
        lightred='\033[91m'
        lightgreen='\033[92m'
        yellow='\033[93m'
        lightblue='\033[94m'
        pink='\033[95m'
        lightcyan='\033[96m'
    class bg:
        black='\033[40m'
        red='\033[41m'
        green='\033[42m'
        orange='\033[43m'
        blue='\033[44m'
        purple='\033[45m'
        cyan='\033[46m'
        lightgrey='\033[47m'


# In[ ]:


missing_Employer_Location_Country = df_na_Employer_Location.Employer_Country.unique()

print('\33[1m\33[91mCountry and number of total records present in the whole 2022 dataset\33[0m\n')
count = 0
for country in missing_Employer_Location_Country:
            
    print(f'{colors.fg.lightblue}{country:25}\33[0m : \33[91m{df[df.Employer_Country == country].shape[0]}\33[0m')


# **<font color='red'>Observations</font>**
# 
# - If *Employer_Country* <font color='red'>is not</font> *United States Of America*, then the total number of missing values in *Employer_Location* is : **<font color='red'>18</font>**
# - This also shows simply dropping missing values we are loosing much more information.

# <div class="alert alert-info" role="alert">
#    <font color='red'><strong>Observations</strong></font>
#  <ul>
#   <li>Assuming the <strong><em>Employer_Location</em></strong> does not have much more impact on visa application and pay of the employee.</li>
#    <ul>
#    <li>So dropping the entire <strong><em>Employer_Location</em></strong> column.</li>
#    <li>Even by dropping we will preserve the other valuble details.</li>
#    </ul>
#   <li>For <strong><em>Worksite</em></strong> column, there are only <strong>4</strong> values are missing.</li>
#      <ul>
#   <li>Since for all these <strong><em>Employer_Country</em></strong> is USA, we are filling <code>NaN</code> value as <strong>USA</strong>.</li>
# </ul> 
# </ul> 
# 
# </div>

# ## Handling missing values

# In[ ]:


df.drop('Employer_Location', axis = 1, inplace = True) # Dropping 'Employer_Location' column
df.Worksite.fillna('USA', inplace = True) # Filling missing values by substituting value as 'USA'


# # Case Status
# 
# - Let’s check the ***Case_Status*** to see the distribution of it.
# - Since there are no missing values present in this we can directly plot a **histogram** to see the distributions.
# - This ***Case_Status*** contains records of **H-1B**, **H-1B1 Chile**, **E-3 Australian**, **H-1B1 Singapore** visas.

# In[ ]:


fig = px.histogram(data_frame = df, x = 'Case_Status', color_discrete_sequence=['#6DA3C7'])

fig.update_layout(
    title_text = 'Status associated with the each Visa application - H-1B, H-1B1 Chile, E-3 Australian, H-1B1 Singapore',
    xaxis_title_text = 'Case Status',
    yaxis_title_text = 'Number of applications',
    hoverlabel_bgcolor = px.colors.qualitative.Plotly[4])

fig.show()


# In[ ]:


df_Case_Status = pd.DataFrame(df.Case_Status.value_counts()).reset_index().rename(columns = {'index' : 'Case Status', 'Case_Status' : 'Count'})

df_Case_Status['Contribution to total applications (%)'] = df_Case_Status.Count *100 / df_Case_Status.Count.sum()

df_Case_Status.style.highlight_max(subset = ['Count', 'Contribution to total applications (%)'], color = '#BBFBB2').highlight_min(subset = ['Count', 'Contribution to total applications (%)'], color = '#F2B2FB').set_caption('Application Status and % contribution').set_table_styles(styles)


# <div class="alert alert-info" role="alert">
#    <font color='red'><strong>Observations</strong></font>
# <ul>
#  <li>Most of the applications are got <strong>Certified</strong> are <font color = 'red'>112073</font>.</li>
#  <li>For a smaller number of applications listed as <strong>Certified - Withdrawn</strong> are <font color = 'red'>8017</font>.</li>
#  <li>The number of applicatoins <strong>Withdrawn</strong> are <font color = 'red'>2068</font>.</li>
#  <li>The <strong>Denied</strong> number of applications are only <font color = 'red'>450</font>.</li>
#     <li>Only <font color = 'red'>0.367%</font> of the applications are got rejected and more than <font color = 'red'>91%</font> are got accepted.
#  </ul>
# </div>

# ## Visa_Class & Case_Status

# In[ ]:


fig = px.histogram(data_frame = df, x = 'Visa_Class', color_discrete_sequence=['#6DA3C7'])

fig.update_layout(
    title_text = 'Types of Visa and number of applicants',
    xaxis_title_text = 'Visa Class',
    yaxis_title_text = 'Number of applicants',
    hoverlabel_bgcolor = px.colors.qualitative.Plotly[4])

fig.show()


# In[ ]:


fig = px.histogram(x = 'Visa_Class', color = 'Case_Status', data_frame = df[df.Visa_Class != 'H-1B'])

fig.update_layout(
    title_text = 'Types of Visa (H-1B1 Chile, E-3 Australian & H-1B1 Singapore) and number of applicants',
    xaxis_title_text = 'Visa Class',
    yaxis_title_text = 'Number of applicants',
    hoverlabel_bgcolor = px.colors.qualitative.Plotly[4])

fig.show()


# In[ ]:


per_column_name, visa_counts = [], []

for status in df.Case_Status.unique():
    visa_counts.append(list(df[df.Case_Status == status].Visa_Class.value_counts()))

df_2 = pd.DataFrame(np.transpose(visa_counts))
df_2.columns = df.Visa_Class.unique()
df_2.insert(0, 'Status', df.Case_Status.unique())
df_2.reset_index()

for sum_ , idx in zip(df.Visa_Class.unique(), [2, 4, 6, 8]):
    
    col_name = f'% Status ' + sum_
    per_column_name.append(col_name)
    
    df_2.insert(idx, col_name, df_2[sum_] * 100 / df_2[sum_].sum())
    
df_2.style.set_caption('Visa Classes and Case Status Count').set_table_styles(styles).highlight_max(subset = per_column_name, color = '#BBFBB2').highlight_min(subset = per_column_name, color = '#F2B2FB')


# <div class="alert alert-info" role="alert">
#    <font color='red'><strong>Observations</strong></font>
# <ul>
#  <li>The most filed visa application is for <strong>H-1B</strong> and morethan <font color = 'red'>96%</font> of those accepted.</li>
#  <li>The most <em>Certified</em> visa is <strong>H-1B1 Chile</strong> with morethan <font color = 'red'>97%</font>.</li>
#  <li>The <em>least</em> number of applicants were for <strong>E-3 Australian</strong> and for this the <em>Certification</em> and <em>denial</em> was higher with values <font color = 'red'>~79%</font> and <font color = 'red'>~4%</font> respectively.</li>
#  </ul>
# </div>

# # SOC_Title

# In[ ]:


df.head(2)


# ### **<center><font color='#EF10DF'>🚧 Notebook is still under construction 🛠</font></center>**
# 
# ### **<center><font color = '#EF10DF'>If you liked this Notebook so-far, please do a <font color='green'>upvote</font> 👍</font></center>**
