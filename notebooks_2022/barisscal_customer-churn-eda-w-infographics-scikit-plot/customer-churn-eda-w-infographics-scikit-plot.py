#!/usr/bin/env python
# coding: utf-8

# # Credit Card Customers EDA & Churn Prediction
# 
# In this notebook I made Data Analysis on Credit Card Customers dataset and trained a Machine Learning algorithm. You will find informative infographics about churn and classification metrics in the notebook. You will find graphs created with the scikit-plot library that measures the performance of the model I trained.
#  
# I wish you pleasant reading.
# 
# **Note:** Please open the infographics in a new tab or hide the 'Table of Content' section on the right for easy reading.

# In[ ]:


import numpy as np 
import pandas as pd 
import math
pd.set_option('display.max_columns', None)
import seaborn as sns; sns.set()
import scikitplot as skplt
import matplotlib.style as style
style.use("fivethirtyeight")
import matplotlib.pyplot as plt
import matplotlib.style as style
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.gridspec as grid_spec
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')

colors = ["#8ecae6","#219ebc","#023047","#ffb703","#fb8500"]
sns.palplot(sns.color_palette(colors));
plt.show()


# In[ ]:


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


first_read = pd.read_csv("/kaggle/input/credit-card-customers/BankChurners.csv")
df = first_read.copy()


# # Information About Dataset

# In[ ]:


df.drop(['Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1', 'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2'], inplace = True, axis = 1)


# In[ ]:


def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Info #####################")
    print(dataframe.info())
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.1, 0.25, 0.50, 0.75, 0.95, 0.99, 1]).T)


# In[ ]:


def column_information(dataframe, cat_th=10, car_th=20):
    
    #object
    categoric_columns = [col for col in dataframe.columns if dataframe[col].dtypes == "object"]
    
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    
    categoric_columns = categoric_columns + num_but_cat
    categoric_columns = [col for col in categoric_columns if col not in cat_but_car]

    #num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]
    
    #float64
    float64_columns = [col for col in dataframe.columns if dataframe[col].dtypes == "float64"]
    
    #int64
    int64_columns = [col for col in dataframe.columns if dataframe[col].dtypes == "int64"]
    

    print(f"# of Samples: {dataframe.shape[0]}")
    print(f"# of Columns: {dataframe.shape[1]}")
    print(f'# of Categoric Columns: {len(categoric_columns)}')
    print(f'Name of Categoric Columns: {(categoric_columns)}')
    print(f'# of float64 Numeric Columns: {len(float64_columns)}')
    print(f'Name of float64 Numeric Columns: {(float64_columns)}')
    print(f'# of int64 Numeric Columns: {len(num_cols)}')
    print(f'Name of int64 Numeric Columns: {(num_cols)}')
    print(f'# of Total Numeric Columns: {len(int64_columns)}')
    print(f'# of Categorical But Cardinal Columns: {len(cat_but_car)}')
    print(f'Name of Categorical But Cardinal Columns: {(cat_but_car)}')
    print(f'# of Numerical But Categoric Columns: {len(num_but_cat)}')
    print(f'Name of Numerical But Categoric Columns: {(num_but_cat)}')

    return categoric_columns, float64_columns, int64_columns


# In[ ]:


check_df(df)


# In[ ]:


categoric_columns, float64_columns, int64_columns = column_information(df)


# ## Variable Descriptions

# <html>
# <head>
# <style>
# table {
#   font-family: arial, sans-serif;
#   border-collapse: collapse;
#   width: 100%;
# }
# 
# td, th {
#   border: 1px solid #dddddd;
#   text-align: left;
#   padding: 8px;
# }
# 
# tr:nth-child(even) {
#   background-color: #dddddd;
# }
# </style>
# </head>
# <body>
#     
# <table>
#     <tr>
#         <th>COLUMN NAME</th>
#         <th>DESCRIPTION </th>
#     </tr>
#     <tr>
#         <td>CLIENTNUM</td>
#         <td>Client number. Unique identifier for the customer holding the account </td>
#     </tr>
#     <tr>
#         <td>Attrition_Flag</td>
#         <td>Internal event (customer activity) variable - if the account is closed then 1 else 0 </td>
#     </tr>
#     <tr>
#         <td>Customer_Age</td>
#         <td>Demographic variable - Customer's Age in Years </td>
#     </tr>
#     <tr>
#         <td>Gender</td>
#         <td>Demographic variable - M=Male, F=Female </td>
#     </tr>
#     <tr>
#         <td>Dependent_count</td>
#         <td>Demographic variable - Number of dependents </td>
#     </tr>
#     <tr>
#         <td>Education_Level</td>
#         <td>Demographic variable - Educational Qualification of the account holder (example: high school, college graduate, etc.) </td>
#     </tr>
#     <tr>
#         <td>Marital_Status</td>
#         <td>Demographic variable - Married, Single, Divorced, Unknown </td>
#     </tr>
#     <tr>
#         <td>Income_Category</td>
#         <td>Demographic variable - Annual Income Category of the account holder (&lt; $40K, $40K - 60K, $60K - $80K, $80K-$120K, &gt; $120K, Unknown) </td>
#     </tr>
#     <tr>
#         <td>Card_Category</td>
#         <td>Product Variable - Type of Card (Blue, Silver, Gold, Platinum) </td>
#     </tr>
#     <tr>
#         <td>Months_on_book</td>
#         <td>Period of relationship with bank </td>
#     </tr>
#     <tr>
#         <td>Total_Relationship_Count</td>
#         <td>Total no. of products held by the customer </td>
#     </tr>
#     <tr>
#         <td>Months_Inactive_12_mon</td>
#         <td>No. of Contacts in the last 12 months </td>
#     </tr>
#     <tr>
#         <td>Contacts_Count_12_mon</td>
#         <td>No. of Contacts in the last 12 months </td>
#     </tr>
#     <tr>
#         <td>Credit_Limit</td>
#         <td>Credit Limit on the Credit Card </td>
#     </tr>
#     <tr>
#         <td>Total_Revolving_Bal</td>
#         <td>Total Revolving Balance on the Credit Card </td>
#     </tr>
#     <tr>
#         <td>Avg_Open_To_Buy</td>
#         <td>Open to Buy Credit Line (Average of last 12 months) </td>
#     </tr>
#     <tr>
#         <td>Total_Amt_Chng_Q4_Q1</td>
#         <td>Change in Transaction Amount (Q4 over Q1) </td>
#     </tr>
#     <tr>
#         <td>Total_Trans_Amt</td>
#         <td>Total Transaction Amount (Last 12 months) </td>
#     </tr>
#     <tr>
#         <td>Total_Trans_Ct</td>
#         <td>Total Transaction Count (Last 12 months) </td>
#     </tr>
#     <tr>
#         <td>Total_Ct_Chng_Q4_Q1</td>
#         <td>Change in Transaction Count (Q4 over Q1) </td>
#     </tr>
#     <tr>
#         <td>Avg_Utilization_Ratio</td>
#         <td>Average Card Utilization Ratio </td>
#     </tr>
# </table>
# 
# </body>
# </html>

# ### Univariate Variable Analysis
# 
# #### Categoric Variables

# In[ ]:


def bar_plot(variable):
    # get feature
    var = df[variable]
    # count number of categorical variable(value/sample)
    varValue = var.value_counts()
    
    # visualize
    plt.figure(figsize = (9,3))
    plt.bar(varValue.index, varValue)
    plt.xticks(varValue.index, varValue.index.values)
    plt.ylabel("Frequency")
    plt.title(variable)
    plt.show()
    print("{}:\n{}".format(variable,varValue))


# In[ ]:


for c in categoric_columns:
    bar_plot(c)


# #### Numeric Variables (int64)

# In[ ]:


def plot_hist(variable):
    plt.figure(figsize = (9,3))
    plt.hist(df[variable], bins = 50)
    plt.xlabel(variable)
    plt.ylabel("Frequency")
    plt.title("{} distribution with hist".format(variable))
    plt.show()


# In[ ]:


for n in int64_columns:
    plot_hist(n)


# #### Numeric Variables (float64)

# In[ ]:


for n in float64_columns:
    plot_hist(n)


# # Exploratory Data Analysis

# In[ ]:


color_palette=["gray","#0e4f66"]
background_color = '#fafafa'
style.use("fivethirtyeight")
fig = plt.figure(figsize=(60,20), dpi=250)
fig.patch.set_facecolor(background_color) # figure background color
gs = fig.add_gridspec(1,2)
gs.update(wspace=.0, hspace=0)
ax0 = fig.add_subplot(gs[0, 0])
ax0.tick_params(axis='y', which='both', labelleft='off', labelright='on')


labels = df['Attrition_Flag'].value_counts().index
sizes = df['Attrition_Flag'].value_counts().values
explode = [0,.1]
ax0 = plt.pie(sizes, labels=labels, shadow = True, explode = explode, startangle=360, colors = colors, autopct='%1.1f%%',textprops={'fontsize': 18, 'fontfamily': 'serif', 'fontweight':'bold', 'fontsize': 20});
plt.title('Distribution of Customers by Churn',color = 'black',fontsize = 40, style = 'normal',fontweight='bold', fontfamily='serif')



fig.text(0.5, 0.87, 'Customer Churn', fontsize=55, fontweight='bold', fontfamily='serif',color='#323232')

fig.text(0.5, 0.80, 'What is Churn?', fontsize=40, fontweight='bold', fontfamily='serif',color='#323232')
fig.text(0.5, 0.65, '''
Customer churn analysis is the process of reviewing the purchasing behavior of your customers, identifying the profiles of customers who are likely to quit working with you, 
and predicting those who are likely to leave (Churn). Customer churn refers to the percentage of customers who stop using your company's product or service over a period of time. 
The churn rate is calculated by dividing the number of customers lost during that period by the number of customers acquired at the beginning of that period.

Considering that finding a new customer is much more costly than retaining your existing customers, it can be seen how important a customer churn analysis is. 
This analysis has now become a tool frequently used by strategic decision-making and planning officials.'''
, fontsize=25, fontweight='light', fontfamily='serif',color='#323232')


fig.text(0.5, 0.55, 'Types of Customer Churn', fontsize=40, fontweight='bold', fontfamily='serif',color='#323232')
fig.text(0.5, 0.35, '''
It consists of two types depending on whether it is voluntary or involuntary in terms of the circumstances and conditions that cause the loss of customers. 
Voluntary losses; is the situation in which a customer leaves the current business and buys or prefers the same good or service from another business. 
Involuntary losses; It is the situation of abandonment caused by force majeure or undesirable reasons that are not counted in line with the customers' own wishes and demands. 
The important part here is about the reasons why the person who voluntarily parted ways with us and chose the competitor changed his choice. 
However, if various risk measures are taken in involuntary situations, a solution process can be created that can prevent this situation. 
Types of involuntary loss are often ignored in statistical and data mining studies. The main reason for this is the inability to prevent the loss of customers. 
In case of canceling the subscription, it may occur that we are in less contact with the customers and cannot determine the desired situation, 
or it may be closed by the legal representatives in case of the customer's death. Closing the account may be voluntarily, or it may be due to compelling reasons.'''
, fontsize=25, fontweight='light', fontfamily='serif',color='#323232')

fig.text(0.5, 0.25, 'How is Customer Churn Calculated?', fontsize=40, fontweight='bold', fontfamily='serif',color='#323232')
fig.text(0.5, 0.07, '''
The ratio of the difference between the number of customers at the beginning of the period and the number of customers at the end of the period to 
the number of customers at the beginning of the period allows us to find the “Periodic Loss Rate”. We can also calculate these periods in different 
periods according to the nature and frequency of the work. We also need to take into account the loss of income per capita. With these calculations, 
necessary interpretations should be made, goods and services should be diversified accordingly, analyzes should be made and determined for needs,
and communication channels and methods should be reviewed.'''
, fontsize=25, fontweight='light', fontfamily='serif',color='#323232')



import matplotlib.lines as lines
l1 = lines.Line2D([0.48, 0.48], [0.1, 0.9], transform=fig.transFigure, figure=fig,color='black',lw=5)
fig.lines.extend([l1])

plt.show()


# In[ ]:


color_palette=["gray","#0e4f66"]
background_color = '#fafafa'
#style.use("fivethirtyeight")
fig = plt.figure(figsize=(60,50), dpi=250)
fig.patch.set_facecolor(background_color) # figure background color
gs = fig.add_gridspec(6,6)
gs.update(wspace=.6, hspace=.3)
ax0 = fig.add_subplot(gs[0, 0])
ax1 = fig.add_subplot(gs[0, 1])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[1, 1])
ax4 = fig.add_subplot(gs[2, 0])
ax5 = fig.add_subplot(gs[2, 1])
ax6 = fig.add_subplot(gs[3, 0])
ax7 = fig.add_subplot(gs[3, 1])
ax8 = fig.add_subplot(gs[4, 0])

ax0 = sns.countplot(ax = ax0, x="Attrition_Flag", hue = 'Gender', data=df, palette = colors)
ax0.set_xlabel('Churned', fontsize = 16, fontfamily = 'serif')
ax0.set_ylabel('Count', fontsize = 16, fontfamily = 'serif')
ax0.set_title('Distribution of Churned Customers',fontfamily = 'serif', fontsize=20, fontweight = 'bold', loc='center')
ax0.grid(color='black', linestyle=':', axis='y', zorder=0,  dashes=(1,5))

ax1 = sns.countplot(ax = ax1, x="Attrition_Flag", hue = 'Education_Level', data=df, palette = colors)
ax1.set_xlabel('Churned', fontsize = 16, fontfamily = 'serif')
ax1.set_ylabel('Count', fontsize = 16, fontfamily = 'serif')
ax1.set_title('Distribution of Customers',fontfamily = 'serif', fontsize=20, fontweight = 'bold', loc='center')
ax1.grid(color='black', linestyle=':', axis='y', zorder=0,  dashes=(1,5))


ax2 = sns.countplot(ax = ax2, x="Attrition_Flag", hue = 'Marital_Status', data=df, palette = colors)
ax2.set_xlabel('Churned', fontsize = 16, fontfamily = 'serif')
ax2.set_ylabel('Count', fontsize = 16, fontfamily = 'serif')
ax2.set_title('Distribution of Churned Customers',fontfamily = 'serif', fontsize=20, fontweight = 'bold', loc='center')
ax2.grid(color='black', linestyle=':', axis='y', zorder=0,  dashes=(1,5))


ax3 = sns.countplot(ax = ax3, x="Attrition_Flag", hue = 'Income_Category', data=df, palette = colors)
ax3.set_xlabel('Churned', fontsize = 16, fontfamily = 'serif')
ax3.set_ylabel('Count', fontsize = 16, fontfamily = 'serif')
ax3.set_title('Distribution of Churned Customers',fontfamily = 'serif', fontsize=20, fontweight = 'bold', loc='center')
ax3.grid(color='black', linestyle=':', axis='y', zorder=0,  dashes=(1,5))



ax4 = sns.countplot(ax = ax4, x="Attrition_Flag", hue = 'Card_Category', data=df, palette = colors)
ax4.set_xlabel('Churned', fontsize = 16, fontfamily = 'serif')
ax4.set_ylabel('Count', fontsize = 16, fontfamily = 'serif')
ax4.set_title('Distribution of Churned Customers',fontfamily = 'serif', fontsize=20, fontweight = 'bold', loc='center')
ax4.grid(color='black', linestyle=':', axis='y', zorder=0,  dashes=(1,5))


ax5 = sns.countplot(ax = ax5, x="Attrition_Flag", hue = 'Dependent_count', data=df, palette = colors)
ax5.set_xlabel('Churned', fontsize = 16, fontfamily = 'serif')
ax5.set_ylabel('Count', fontsize = 16, fontfamily = 'serif')
ax5.set_title('Distribution of Churned Customers',fontfamily = 'serif', fontsize=20, fontweight = 'bold', loc='center')
ax5.grid(color='black', linestyle=':', axis='y', zorder=0,  dashes=(1,5))



ax6 = sns.countplot(ax = ax6, x="Attrition_Flag", hue = 'Total_Relationship_Count', data=df, palette = colors)
ax6.set_xlabel('Churned', fontsize = 16, fontfamily = 'serif')
ax6.set_ylabel('Count', fontsize = 16, fontfamily = 'serif')
ax6.set_title('Distribution of Churned Customers',fontfamily = 'serif', fontsize=20, fontweight = 'bold', loc='center')
ax6.grid(color='black', linestyle=':', axis='y', zorder=0,  dashes=(1,5))


ax7 = sns.countplot(ax = ax7, x="Attrition_Flag", hue = 'Months_Inactive_12_mon', data=df, palette = colors)
ax7.set_xlabel('Churned', fontsize = 16, fontfamily = 'serif')
ax7.set_ylabel('Count', fontsize = 16, fontfamily = 'serif')
ax7.set_title('Distribution of Churned Customers',fontfamily = 'serif', fontsize=20, fontweight = 'bold', loc='center')
ax7.grid(color='black', linestyle=':', axis='y', zorder=0,  dashes=(1,5))


ax8 = sns.countplot(ax = ax8, x="Attrition_Flag", hue = 'Contacts_Count_12_mon', data=df, palette = colors)
ax8.set_xlabel('Churned', fontsize = 16, fontfamily = 'serif')
ax8.set_ylabel('Count', fontsize = 16, fontfamily = 'serif')
ax8.set_title('Distribution of Churned Customers',fontfamily = 'serif', fontsize=20, fontweight = 'bold', loc='center')
ax8.grid(color='black', linestyle=':', axis='y', zorder=0,  dashes=(1,5))

plt.show()


# In[ ]:


color_palette=["gray","#0e4f66"]
background_color = '#fafafa'
#style.use("fivethirtyeight")
fig = plt.figure(figsize=(20,15), dpi=250)
fig.patch.set_facecolor(background_color) # figure background color
gs = fig.add_gridspec(2,2)
gs.update(wspace=.6, hspace=.3)
ax0 = fig.add_subplot(gs[0, 0])
ax1 = fig.add_subplot(gs[0, 1])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[1, 1])
ax0.tick_params(axis='y', which='both', labelleft='off', labelright='on')


ax0 = sns.countplot(ax = ax0, x="Gender", data=df, palette = colors)
ax0.set_xlabel('Gender', fontsize = 16, fontfamily = 'serif')
ax0.set_ylabel('Count', fontsize = 16, fontfamily = 'serif')
ax0.set_title('Distribution of Genders',fontfamily = 'serif', fontsize=20, fontweight = 'bold', loc='center')


ax1 = sns.countplot(ax = ax1, x="Gender", hue = 'Attrition_Flag', data=df, palette = colors)
ax1.set_xlabel('Gender', fontsize = 16, fontfamily = 'serif')
ax1.set_ylabel('Count', fontsize = 16, fontfamily = 'serif')
ax1.set_title('Distribution of Genders',fontfamily = 'serif', fontsize=20, fontweight = 'bold', loc='center')


ax2 = sns.countplot(ax = ax2, x="Gender", hue = 'Education_Level', data=df, palette = colors)
ax2.set_xlabel('Gender', fontsize = 16, fontfamily = 'serif')
ax2.set_ylabel('Count', fontsize = 16, fontfamily = 'serif')
ax2.set_title('Distribution of Genders',fontfamily = 'serif', fontsize=20, fontweight = 'bold', loc='center')


ax3 = sns.countplot(ax = ax3, x="Gender", hue = 'Marital_Status', data=df, palette = colors)
ax3.set_xlabel('Gender', fontsize = 16, fontfamily = 'serif')
ax3.set_ylabel('Count', fontsize = 16, fontfamily = 'serif')
ax3.set_title('Distribution of Genders',fontfamily = 'serif', fontsize=20, fontweight = 'bold', loc='center')


plt.show()


# In[ ]:


categoric_columns, float64_columns, int64_columns


# In[ ]:


float64_columns


# In[ ]:


color_palette=["gray","#0e4f66"]
background_color = '#fafafa'
#style.use("fivethirtyeight")
fig = plt.figure(figsize=(20,15), dpi=250)
fig.patch.set_facecolor(background_color) # figure background color
gs = fig.add_gridspec(2,2)
gs.update(wspace=.6, hspace=.3)
ax0 = fig.add_subplot(gs[0, 0])
ax1 = fig.add_subplot(gs[0, 1])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[1, 1])
ax0.tick_params(axis='y', which='both', labelleft='off', labelright='on')


ax0 = sns.boxplot(ax = ax0, x="Attrition_Flag", y = 'Total_Trans_Amt', data=df, palette = colors)
ax0.set_xlabel('Attrition_Flag', fontsize = 16, fontfamily = 'serif')
ax0.set_ylabel('')
ax0.set_title('Total_Trans_Amt',fontfamily = 'serif', fontsize=20, fontweight = 'bold', loc='center')


ax1 = sns.boxplot(ax = ax1, x="Attrition_Flag", y = 'Total_Trans_Ct', data=df, palette = colors)
ax1.set_xlabel('Attrition_Flag', fontsize = 16, fontfamily = 'serif')
ax1.set_ylabel('')
ax1.set_title('Total_Trans_Ct',fontfamily = 'serif', fontsize=20, fontweight = 'bold', loc='center')


ax2 = sns.boxplot(ax = ax2, x="Attrition_Flag", y = 'Total_Ct_Chng_Q4_Q1', data=df, palette = colors)
ax2.set_xlabel('Attrition_Flag', fontsize = 16, fontfamily = 'serif')
ax2.set_ylabel('')
ax2.set_title('Total_Ct_Chng_Q4_Q1',fontfamily = 'serif', fontsize=20, fontweight = 'bold', loc='center')


ax3 = sns.boxplot(ax = ax3, x="Attrition_Flag", y = 'Avg_Utilization_Ratio', data=df, palette = colors)
ax3.set_xlabel('Attrition_Flag', fontsize = 16, fontfamily = 'serif')
ax3.set_ylabel('')
ax3.set_title('Avg_Utilization_Ratio',fontfamily = 'serif', fontsize=20, fontweight = 'bold', loc='center')


plt.show()


# # Correlation

# In[ ]:


plt.figure(figsize=(20, 10))
style.use("fivethirtyeight")
mask = np.triu(np.ones_like(df.corr(), dtype=np.bool))
heatmap = sns.heatmap(df.corr(), mask=mask,annot=True, cmap=colors, linewidths = 2)
heatmap.set_title('Correlation of Features', fontdict={'fontsize':30, 'fontfamily' : 'serif', 'fontweight' : 'bold'}, pad=16);


# # Cardinality

# In[ ]:


for column in categoric_columns:
    print('Column: {} - Unique Values: {}'.format(column, df[column].unique()))


# # Missing Values

# In[ ]:


import missingno as mno
mno.matrix(df, figsize = (20, 6))


# # Encoding
# 
# ## Label Encoding

# In[ ]:


label_encoding = []
one_hot = []

for x in categoric_columns:
    a = df[x].unique()
    print(f'Unique Values for {x}: ', df[x].unique())
    if(len(a) == 2):
        label_encoding.append(x)
    else:
        one_hot.append(x)


# In[ ]:


for y in label_encoding:
    var = df[y].unique()
    y_mapping = {var[0]: 0, var[1]: 1}
    df[y] = df[y].map(y_mapping)


# ## One-Hot Encoding

# In[ ]:


for i in range(0, len(one_hot)):
    df[f'{one_hot[i]}'] = pd.Categorical(df[f'{one_hot[i]}'])
    dummies = pd.get_dummies(df[f'{one_hot[i]}'], prefix = f'{one_hot[i]}_encoded', drop_first=True)
    df.drop([f'{one_hot[i]}'], axis=1, inplace=True)
    df = pd.concat([df, dummies], axis=1)


# In[ ]:


df


# # Train - Test Split

# In[ ]:


from sklearn.model_selection import train_test_split

X = df.drop(['CLIENTNUM','Attrition_Flag'], axis = 1)
y = df['Attrition_Flag']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 101)

print("##################### Length #####################")
print(f'Total # of sample in whole dataset: {len(X_train)+len(X_test)}')
print(f'Total # of sample in train dataset: {len(X_train)}')
print(f'Total # of sample in test dataset: {len(X_test)}')

print("##################### Shape #####################")
print(f'Shape of train dataset: {X_train.shape}')
print(f'Shape of test dataset: {X_test.shape}')

print("##################### Percantage #####################")
print(f'Percentage of train dataset: {round((len(X_train)/(len(X_train)+len(X_test)))*100,2)}%')
print(f'Percentage of validation dataset: {round((len(X_test)/(len(X_train)+len(X_test)))*100,2)}%')


# # Anomaly Detection

# In[ ]:


from collections import Counter

def detect_outliers(df,features):
    outlier_indices = []
    
    for c in features:
        # 1st quartile
        Q1 = np.percentile(df[c],25)
        # 3rd quartile
        Q3 = np.percentile(df[c],75)
        # IQR
        IQR = Q3 - Q1
        # Outlier step
        outlier_step = IQR * 1.5
        # detect outlier and their indeces
        outlier_list_col = df[(df[c] < Q1 - outlier_step) | (df[c] > Q3 + outlier_step)].index
        # store indeces
        outlier_indices.extend(outlier_list_col)
    
    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(i for i, v in outlier_indices.items() if v > 2)
    
    return multiple_outliers


# In[ ]:


temp_df =  pd.concat([X_train, y_train], axis=1)


# In[ ]:


categoric_columns, float64_columns, int64_columns


# In[ ]:


temp_df.loc[detect_outliers(temp_df,float64_columns)]


# In[ ]:


# drop outliers
temp_df = temp_df.drop(detect_outliers(temp_df,float64_columns),axis = 0).reset_index(drop = True)
temp_df


# In[ ]:


X_train = temp_df.drop(['Attrition_Flag'], axis = 1)
y_train = temp_df.Attrition_Flag

print("##################### Length #####################")
print(f'Total # of sample in whole dataset: {len(X_train)+len(X_test)}')
print(f'Total # of sample in train dataset: {len(X_train)}')
print(f'Total # of sample in test dataset: {len(X_test)}')

print("##################### Shape #####################")
print(f'Shape of train dataset: {X_train.shape}')
print(f'Shape of test dataset: {X_test.shape}')

print("##################### Percantage #####################")
print(f'Percentage of train dataset: {round((len(X_train)/(len(X_train)+len(X_test)))*100,2)}%')
print(f'Percentage of validation dataset: {round((len(X_test)/(len(X_train)+len(X_test)))*100,2)}%')


# # Scaling

# In[ ]:


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# # GradientBoostingClassifier

# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

gbc_model = GradientBoostingClassifier()
gbc_model.fit(X_train, y_train)

train_score = gbc_model.score(X_train, y_train)
print(f'Train score of trained model: {train_score*100}')

test_score = gbc_model.score(X_test, y_test)
print(f'Test score of trained model: {test_score*100}')

y_predictions = gbc_model.predict(X_test)
y_proba = gbc_model.predict_proba(X_test)
conf_matrix = confusion_matrix(y_predictions, y_test)

print(f'Confussion matrix: \n{conf_matrix}\n')

sns.heatmap(conf_matrix, annot=True, color = colors)

tn = conf_matrix[0,0]
fp = conf_matrix[0,1]
tp = conf_matrix[1,1]
fn = conf_matrix[1,0]

total = tn + fp + tp + fn
real_positive = tp + fn
real_negative = tn + fp

accuracy  = (tp + tn) / total # Accuracy Rate
precision = tp / (tp + fp) # Positive Predictive Value
recall    = tp / (tp + fn) # True Positive Rate
f1score  = 2 * precision * recall / (precision + recall)
specificity = tn / (tn + fp) # True Negative Rate
error_rate = (fp + fn) / total # Missclassification Rate
prevalence = real_positive / total
miss_rate = fn / real_positive # False Negative Rate
fall_out = fp / real_negative # False Positive Rate


print('Evaluation Metrics:')
print(f'Accuracy    : {accuracy}')
print(f'Precision   : {precision}')
print(f'Recall      : {recall}')
print(f'F1 score    : {f1score}')
print(f'Specificity : {specificity}')
print(f'Error Rate  : {error_rate}')
print(f'Prevalence  : {prevalence}')
print(f'Miss Rate   : {miss_rate}')
print(f'Fall Out    : {fall_out}')

print("") 
print(f'Classification Report: \n{classification_report(y_predictions, y_test)}\n')
print("")


# In[ ]:


color_palette=["gray","#0e4f66"]
background_color = '#fafafa'
#style.use("fivethirtyeight")
fig = plt.figure(figsize=(40,20), dpi=500)
fig.patch.set_facecolor(background_color) # figure background color
gs = fig.add_gridspec(3,3)
gs.update(wspace=.6, hspace=.3)
ax0 = fig.add_subplot(gs[0, 0])
ax1 = fig.add_subplot(gs[0, 1])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[1, 1])
ax4 = fig.add_subplot(gs[2, 0])
ax5 = fig.add_subplot(gs[2, 1])


ax0 = skplt.metrics.plot_roc( y_test, y_proba, ax = ax0, title = 'ROC Vurve for GBC',figsize = (10,6), title_fontsize = 20, text_fontsize = 15)

ax1 = skplt.metrics.plot_precision_recall(y_test, y_proba, ax = ax1, title = 'PR Curve for GBC',figsize = (10,6), title_fontsize = 20, text_fontsize = 15)

ax2 = skplt.metrics.plot_cumulative_gain(y_test, y_proba, ax = ax2, title = 'Cumulative Gains Chart for GBC',figsize = (10,6), title_fontsize = 20, text_fontsize = 15)

ax3 = skplt.metrics.plot_lift_curve(y_test, y_proba, ax = ax3, title = 'Lift Curve for GBC',figsize = (10,6), title_fontsize = 20, text_fontsize = 15)

ax4 = skplt.estimators.plot_learning_curve(gbc_model, X_train, y_train, ax = ax4, title = 'Learning Curve for GBC',figsize = (10,6), title_fontsize = 20, text_fontsize = 15)

ax5 = skplt.estimators.plot_feature_importances(gbc_model, feature_names=X.columns, ax = ax5, x_tick_rotation = 90, title = 'Feature Importance',figsize = (10,6), title_fontsize = 20, text_fontsize = 15 )


fig.text(0.7, 0.85, 'ROC Curve', fontsize=45, fontweight='bold', fontfamily='serif',color='#323232')
fig.text(0.7, 0.75, '''On the ROC curve, the true positive rate (Sensitivity) is plotted as a function of the false positive rate (Specificity) of a parameter for different cut-off points. 
Each point on the ROC curve represents a sensitivity/specificity pair corresponding to a certain decision threshold. 
The area under the ROC curve (AUC) is a measure of how well a parameter can distinguish between two groups.'''
, fontsize=30, fontweight='light', fontfamily='serif',color='#323232')


fig.text(0.7, 0.68, 'PR Curve', fontsize=45, fontweight='bold', fontfamily='serif',color='#323232')
# https://www.geeksforgeeks.org/precision-recall-curve-ml/
fig.text(0.7, 0.55, '''A PR curve is simply a graph with Precision values on the y-axis and Recall values on the x-axis. 
In other words, the PR curve contains TP/(TP+FN) on the y-axis and TP/(TP+FP) on the x-axis.
It is important to note that Precision is also called the Positive Predictive Value (PPV).
Recall is also called Sensitivity, Hit Rate or True Positive Rate (TPR).'''
, fontsize=30, fontweight='light', fontfamily='serif',color='#323232')

fig.text(0.7, 0.48, 'Cumulative Gains Chart & Lift Curve', fontsize=45, fontweight='bold', fontfamily='serif',color='#323232')
# https://www.geeksforgeeks.org/understanding-gain-chart-and-lift-chart/
fig.text(0.7, 0.20, '''The gain chart and lift chart are two measures that are used for Measuring the benefits of using the model and are used 
in business contexts such as target marketing. It’s not just restricted to marketing analysis. It can also be used in other domains such as risk modeling, 
supply chain analytics, etc. In other words, 
Gain and Lift charts are two approaches used while solving classification problems with imbalanced data sets.
The gain and lift chart is obtained using the following steps:
1 - Predict the probability Y = 1 (positive) using the LR model and arrange the observation in the decreasing order of predicted probability [i.e., P(Y = 1)].
2 - Divide the data sets into deciles. Calculate the number of positives (Y = 1) in each decile and the cumulative number of positives up to a decile.
3 - Gain is the ratio between the cumulative number of positive observations up to a decile to the total number of positive observations in the data. 
The gain chart is a chart drawn between the gain on the vertical axis and the decile on the horizontal axis.
''', fontsize=30, fontweight='light', fontfamily='serif',color='#323232')


fig.text(0.7, 0.15, 'Learning Curve', fontsize=45, fontweight='bold', fontfamily='serif',color='#323232')
# https://en.wikipedia.org/wiki/Learning_curve_(machine_learning)
fig.text(0.7, 0.00, '''In machine learning, a learning curve (or training curve) plots the optimal value of a model's loss function for a training set against this loss function evaluated on a 
validation data set with same parameters as produced the optimal function. It is a tool to find out how much a machine model benefits from adding 
more training data and whether the estimator suffers more from a variance error or a bias error. If both the validation score and the training score converge to a value 
that is too low with increasing size of the training set, it will not benefit much from more training data.
'''
, fontsize=30, fontweight='light', fontfamily='serif',color='#323232')

fig.text(0.7, -0.05, 'Feature Importance', fontsize=45, fontweight='bold', fontfamily='serif',color='#323232')
# https://machinelearningmastery.com/calculate-feature-importance-with-python/
fig.text(0.7, -0.25, '''
Feature importance refers to a class of techniques for assigning scores to input features to a predictive model that indicates the relative 
importance of each feature when making a prediction.Feature importance scores can be calculated for problems that involve predicting 
a numerical value, called regression, and those problems that involve predicting a class label, called classification.
The scores are useful and can be used in a range of situations in a predictive modeling problem, such as:
- Better understanding the data.
- Better understanding a model.
- Reducing the number of input features.
'''
, fontsize=30, fontweight='light', fontfamily='serif',color='#323232')


import matplotlib.lines as lines
l1 = lines.Line2D([0.65, 0.65], [-0.2, 0.91], transform=fig.transFigure, figure=fig,color='black',lw=5)
fig.lines.extend([l1])

plt.show()


# # Conclusion
# 
# In this notebook, I examined Credit Card Customers Dataset. Firstly, I made Exploratory Data Analysis, Visualization, then I applied Gradient Boosting algorithm to this dataset.
# 
# If you liked this notebook, please let me know :)
# 
# Thank you for your time.
