#!/usr/bin/env python
# coding: utf-8

# ## Question answering with Data #1
# 
# **Author**: [Sayak Paul](https://sites.google.com/view/spsayakpaul)
# 
# In this study, I explore a dataset which contains information of students who got graduated from a certain college in the year 2017. I ask several questions (in an unordered fashion) to the dataset which are practical according to me and eventually find their answers. I also try to present these answers in a nice graphical way. 

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mp 
import numpy as np
from collections import Counter
import re
import string
import warnings
warnings.filterwarnings("ignore")

# Set jupyter's max row display
pd.set_option('display.max_row', 1000)

# Set jupyter's max column width to 50
pd.set_option('display.max_columns', 50)

plt.style.use('ggplot')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


data_2017 = pd.read_excel('../input/2017_Batch.xlsx')
data_2017.head()


# ## 1. What is the total number of students this dataset has? 

# In[ ]:


data_2017.shape


# * 670 - The number of students. This is 1 less than 671 because 671 contains the header row as well. 
# * 64 - Columns

# ## 2. What kind of information does this dataset convey? 

# In[ ]:


list(data_2017)


# ## 3. Are there any missing values in the dataset? 

# In[ ]:


data_2017.isna().sum()


# ## 4. How many departments are there in the dataset?

# In[ ]:


# Rename the long name without loss in its meaning
data_2017.rename(columns={'DEPARTMENT (ABBR.)': 'Department'}, inplace=True)


# In[ ]:


set(data_2017.Department.values)


# * AEIE - Applied Electronics and Instrumentation Engg.
# * BME - Biomedical Engg.
# * CE - Civil Engg.
# * CSE - Computer Science and Engg.
# * ECE - Electronics and Communication Engg.
# * EE - Electrical Engg.
# * IT - Information Technology
# * ME - Mechanical Engg.

# ## 5. What is the student count per department?

# In[ ]:


data_2017.groupby('Department')['SL.NO.'].count().reset_index(name='No. of students')


# In[ ]:


subplot = data_2017.groupby('Department')['SL.NO.'].count().plot(kind='barh',figsize=(12,10))
subplot.set_ylabel('')
subplot.set_title('Number of Students per Department', fontsize = 20)
for i in subplot.patches:
    # get_width pulls left or right; get_y pushes up or down
    subplot.text(i.get_width()+.3, i.get_y()+.2,             str(i.get_width()), fontsize=15,)


# ## 6. What is the highest semester grade obtained by a student from a particular department?

# In[ ]:


data_2017.groupby('Department')['SEM AVG'].max().reset_index(name = 'Highest Average Semester Grade')


# In[ ]:


subplot = data_2017.groupby('Department')['SEM AVG'].max().plot(kind='barh',figsize=(12,8))
subplot.set_ylabel('')
subplot.set_title('Highest Average Semester Grade per Department', fontsize = 20)
for i in subplot.patches:
    # get_width pulls left or right; get_y pushes up or down
    subplot.text(i.get_width()+.1, i.get_y()+.1,             str(round(i.get_width(),2)), fontsize=13)


# In[ ]:


data_2017[data_2017['Department'] == 'IT'][['SEM AVG']].head()


# ## 7. Students from different states took admission in this college. What is the state-wise student count?

# In[ ]:


data_2017['PERMANENT LOCATION (STATE)'].value_counts()


# For the same state **West Bengal**, we have several variants like WB, WEST-BENGAL and so on. There are even cases where several spaces are appended before the word. Let's try to give it a proper shape. Otherwise, the numbers will be faulty. We will stick to the name `WEST BENGAL` and will replace the other variants accordingly. We have this problem for other states as well. 
# 
# **Quick observation**: A candidate has even given India as his/her state)

# In[ ]:


# Manual engineering but okay! Can be written in an efficient manner using regex
# The correct entries have also been specified. This is to avoid the entry of NaNs. 
data_2017['PERMANENT LOCATION (STATE)'] = data_2017['PERMANENT LOCATION (STATE)'] .map({'WESTBENGAL':'WEST BENGAL',
'WEST BENGAL':'WEST BENGAL', 'West Bengal':'WEST BENGAL', 'WEST-BENGAL':'WEST BENGAL',
' WEST BENGAL':'WEST BENGAL', 'WB':'WEST BENGAL','West bengal':'WEST BENGAL', 'WEST BINGAL': 'WEST BENGAL',
'               WEST BENGAL':'WEST BENGAL', '                WEST BENGAL':'WEST BENGAL','WEST BENGAL.':'WEST BENGAL',
'               JHARKHAND':'JHARKHAND','BIHAR':'BIHAR','TRIPURA':'TRIPURA','JHARKHAND':'JHARKHAND','UTTARAKHAND':'UTTARAKHAND',
'DELHI':'DELHI','INDIA':'INDIA'})


# In[ ]:


no_students_state_wise = data_2017['PERMANENT LOCATION (STATE)'].value_counts().reset_index(name='Number of students')
no_students_state_wise.rename(columns={'index':'State'},inplace=True)
no_students_state_wise


# Looks much more tidy now! Can we plot this? Of course!

# In[ ]:


subplot = data_2017['PERMANENT LOCATION (STATE)'].value_counts().plot(kind='barh',figsize=(12,8))
subplot.set_ylabel('')
subplot.set_title('State-wise Distribution of Students', fontsize = 20)
for i in subplot.patches:
    # get_width pulls left or right; get_y pushes up or down
    subplot.text(i.get_width()+.1, i.get_y()+.1,             str(round(i.get_width(),2)), fontsize=13)


# The value **India** is really becoming intolerable to my eyes. I am going drop the row corresponding to it. 

# In[ ]:


data_2017[data_2017['PERMANENT LOCATION (STATE)'] == 'INDIA']


# In[ ]:


data_2017.drop(index=249, inplace=True)


# In[ ]:


data_2017[data_2017['PERMANENT LOCATION (STATE)'] == 'INDIA']


# The intended record has been successfully deleted. 

# ## 8. How did the toppers perform in their high school examinations?

# In[ ]:


toppers = data_2017.groupby('Department')['SEM AVG'].transform(max) == data_2017['SEM AVG']
data_2017[toppers][['Department', 'SEM AVG',
                    'ACTUAL % OF CLASS XII','NAME OF BOARD/COUNCIL - CLASS XII']]


# (WEST BENGAL BOARD OF HIGHER SECONDARY EXAMINATION and WBCHSE are the same.)
# 
# We see that toppers were pretty good in terms of marks in their high-school examinations. Now, this has something to do with the **BOARD/COUNCIL** to which their schools were affiliated. Because, getting 90% from WBCHSE was actually a lot more harder than getting 90% (or above) from any other boards back then.

# ## 9.  How the semester grades of the toppers have changed over time?

# In[ ]:


data_2017[toppers][['Department','SEM 1', 'SEM 2', 'SEM 3', 'SEM 4', 'SEM 5']]


# In[ ]:


color=['#81c784','#0288d1','#0288d1','#9575cd','#f44336','#ffb74d','#6d4c41','#f50057']

fig, ax = plt.subplots(figsize=(10,10))
ax.set_ylabel('Semester-wise Grades',fontsize=14)
ax.set_title('Changes in grades of the toppers over time')
for i,c in zip(range(len(data_2017[toppers][['Department','Department','SEM 1', 'SEM 2', 'SEM 3', 'SEM 4', 'SEM 5']])),color):
    ax.plot(['SEM 1', 'SEM 2', 'SEM 3', 'SEM 4', 'SEM 5'], 
            data_2017[toppers][['SEM 1', 'SEM 2', 'SEM 3', 'SEM 4', 'SEM 5']].iloc[i], 
            color=c, linewidth=3,label=data_2017[toppers]['Department'].iloc[i])
    
ax.legend(loc='lower right')


# Looks like someone from the department of **IT** has really improved a lot. Kudos to the individual for that!

# ## 10. Students gave different entrance examinations to get their admissions. How many different entrance examinations are there in the dataset?
# 

# In[ ]:


data_2017['NAME OF JOINT ENTRANCE (WBJEE/JEE-MAINS/JELET ETC.)'].value_counts()


# We have inconsistency in the data. We have dealt with a similar kind inconsistency moments back. 

# In[ ]:


data_2017['NAME OF JOINT ENTRANCE (WBJEE/JEE-MAINS/JELET ETC.)'] = data_2017['NAME OF JOINT ENTRANCE (WBJEE/JEE-MAINS/JELET ETC.)'] .map({'WBJEE':'WBJEE',
'JELET':'JELET', 'JEE-MAINS':'JEE-MAINS','DE-CENTRALISED ADMISSION':'DE-CENTRALISED ADMISSION',
'WBJEE & JEE-MAINS':'WBJEE & JEE-MAINS'})


# In[ ]:


data_2017['NAME OF JOINT ENTRANCE (WBJEE/JEE-MAINS/JELET ETC.)'].value_counts()


# In[ ]:


data_2017[data_2017['NAME OF JOINT ENTRANCE (WBJEE/JEE-MAINS/JELET ETC.)'].isna()==True][['Department','NAME OF JOINT ENTRANCE (WBJEE/JEE-MAINS/JELET ETC.)']]


# It turns out some the students did not specify their entrance examination details. So, we will have to ignore them. From manual inspection of the dataset I found the departments of the following students. 

# In[ ]:


data_2017.ix[130, 'NAME OF JOINT ENTRANCE (WBJEE/JEE-MAINS/JELET ETC.)'] = 'JELET'
data_2017.ix[581, 'NAME OF JOINT ENTRANCE (WBJEE/JEE-MAINS/JELET ETC.)'] = 'WBJEE'
data_2017.ix[586, 'NAME OF JOINT ENTRANCE (WBJEE/JEE-MAINS/JELET ETC.)'] = 'WBJEE'
data_2017.ix[617, 'NAME OF JOINT ENTRANCE (WBJEE/JEE-MAINS/JELET ETC.)'] = 'WBJEE'


# In[ ]:


data_2017['NAME OF JOINT ENTRANCE (WBJEE/JEE-MAINS/JELET ETC.)'].value_counts()


# There are total 84 **lateral** candidates. 

# > A number of students have got backlogs throughout their coursework. What is the total number of backlogs in each of the departments?

# ## 11. What is the total number of backlogs in each of the departments?

# In[ ]:


data_2017[data_2017['IF YES, MENTION NUMBER OF BACKLOG(S)'].isna()==False]        [['IF YES, MENTION NUMBER OF BACKLOG(S)']].sample(20)


# The noise is real here. Some students have specified the subjects in which they got their backlogs instead of specifying numbers. Some specified 'N/A', '-', 'NO', 'NIL' and so on to denote that they did not get any backlogs. While the instruction for non-backlog candidates was to not specify anything and leave the field as it is. 
# 
# There are many ways to fix this. We will explore a one or two - 

# In[ ]:


data_2017[data_2017['IF YES, MENTION NUMBER OF BACKLOG(S)'].str.isalnum()==True]    [['IF YES, MENTION NUMBER OF BACKLOG(S)']]


# In[ ]:


data_2017[data_2017['IF YES, MENTION NUMBER OF BACKLOG(S)'].str.match('^[a-zA-Z0-9]+$')==False]        [['IF YES, MENTION NUMBER OF BACKLOG(S)']]


# In[ ]:


data_2017.iloc[33]['IF YES, MENTION NUMBER OF BACKLOG(S)']


# A bit of manual engineering needed here - 

# In[ ]:


data_2017.ix[76, 'IF YES, MENTION NUMBER OF BACKLOG(S)'] = 1
data_2017.ix[77, 'IF YES, MENTION NUMBER OF BACKLOG(S)'] = 1
data_2017.ix[562, 'IF YES, MENTION NUMBER OF BACKLOG(S)'] = 1
data_2017.ix[33, 'IF YES, MENTION NUMBER OF BACKLOG(S)'] = 5
data_2017.ix[74, 'IF YES, MENTION NUMBER OF BACKLOG(S)'] = 1
data_2017.ix[211, 'IF YES, MENTION NUMBER OF BACKLOG(S)'] = 3
data_2017.ix[212, 'IF YES, MENTION NUMBER OF BACKLOG(S)'] = 1


# In[ ]:


data_2017.iloc[212][['STUDENT\'S FULL NAME','IF YES, MENTION NUMBER OF BACKLOG(S)']]


# The manual engineering part is not at all the best of options as it is not scalable. As the data is small, we could go for it. But there are in deed efficient to resolve this type of problems. 
# 
# Now we need to run a bunch of tests to be sure that we have dealt with the backlog noise in a proper manner. 

# In[ ]:


data_2017[data_2017['IF YES, MENTION NUMBER OF BACKLOG(S)'].isna()==False]        [['IF YES, MENTION NUMBER OF BACKLOG(S)']].sample(20)


# In the above cell, we randomly sampled 20 rows. We can see there is an instance where `IF YES, MENTION NUMBER OF BACKLOG(S)` is `True`. We cannot take this type of instances into the account for calculating the number of backlog students per department. 
# 
# And this type of instances as well - 

# In[ ]:


data_2017[data_2017['IF YES, MENTION NUMBER OF BACKLOG(S)'].str.match('^[a-zA-Z0-9]+$')==False]        [['IF YES, MENTION NUMBER OF BACKLOG(S)']]


# In[ ]:


data_2017[data_2017['IF YES, MENTION NUMBER OF BACKLOG(S)'].str.match('^[\d]')==False]        [['IF YES, MENTION NUMBER OF BACKLOG(S)']]


# So we discard this instances to determine the actual number of backlog candidates from each department. 

# In[ ]:


false_backlog = data_2017.index[data_2017['IF YES, MENTION NUMBER OF BACKLOG(S)'].str.match('^[\d]')==False].tolist()


# In[ ]:


noisy_backlogs = data_2017.index[data_2017['IF YES, MENTION NUMBER OF BACKLOG(S)'].isna()==False].tolist()


# In[ ]:


noisy_backlogs = set(noisy_backlogs) - set(false_backlog)


# In[ ]:


data_2017.ix[noisy_backlogs][['Department','IF YES, MENTION NUMBER OF BACKLOG(S)']]


# In[ ]:


noisy_backlogs_df = data_2017.ix[noisy_backlogs]                [['Department','IF YES, MENTION NUMBER OF BACKLOG(S)']]
label_counts = Counter(noisy_backlogs_df['Department'].values)
label_counts.most_common()


# In[ ]:


plot_df = pd.DataFrame(label_counts.most_common(),columns=['Department','Backlog Count'])
plot_df


# In[ ]:


subplot = plot_df.groupby('Department')['Backlog Count'].sum().plot(kind='barh',figsize=(10,8))
subplot.set_title('Number of Backlogs per Department')
for i in subplot.patches:
    # get_width pulls left or right; get_y pushes up or down
    subplot.text(i.get_width()+.20, i.get_y()+.1,             str(i.get_width()), fontsize=13)


# So, the department of **BME** is good enough. But there is another factor to be considered here - **Number of students per department**.

# In[ ]:


no_stu_dept_wise = data_2017.groupby('Department')['SL.NO.'].count().reset_index(name='Student Count')


# In[ ]:


no_stu_dept_wise.merge(plot_df,on='Department')


# In[ ]:


stu_backlog_count= no_stu_dept_wise.merge(plot_df,on='Department')
subplot = stu_backlog_count.groupby('Department').sum().plot(kind='barh',figsize=(10,8))
subplot.set_title('Number of backlogs w.r.t number of students per department ')
for i in subplot.patches:
    # get_width pulls left or right; get_y pushes up or down
    subplot.text(i.get_width()+.5, i.get_y()+.1,             str(i.get_width()), fontsize=13)


# We can talk in percentages as well - 

# In[ ]:


stu_backlog_count['Backlog percentage'] = round((stu_backlog_count['Backlog Count'] /                                     stu_backlog_count['Student Count'])*100.0,2)
stu_backlog_count


# The story is much more clear now. Time for another question. 

# ## 12. How are grades for each semester distributed?
# 
# I want to know what fraction of students belonged to what grades' range after the first semester.  

# In[ ]:


sem1_df = pd.DataFrame(data_2017['SEM 1'].value_counts().reset_index(name='Student Count'))
sem1_df.rename(columns={'index':'Grade'},inplace=True)
sem1_df.head(10)


# In[ ]:


plt.scatter(sem1_df['Grade'],sem1_df['Student Count'])
plt.xlabel('Grades obtained in Semester 1')
plt.ylabel('Number of students')
plt.title('Grades vs. Number of students for Sem 1')
plt.show()


# The plot tells us that there are total 4 candidates who got exactly 9 in their first semester. Let's verify this - 

# In[ ]:


len(data_2017[data_2017['SEM 1']==9])


# True enough! Let's now see what is the total number of students who got 9 or above - 

# In[ ]:


len(data_2017[data_2017['SEM 1']>=9])


# So, it turned out that the number of 9 pointers is not that high after the first semester. This happens for a lot of reasons - 
# * It takes sometime for a candidate to move out from his school phase and adapt to college life in a fast fashion.
# * There are subjects which are absolutely not related to particular disciplines. For example - during first semester some universities make it compulsory to study subjects like Engineering Mechanics which is absolutely not related to anything in disciplines like Information Technology. So candidates do not take much interest to study subjects like this. 
# 
# Let's now find out if the number of 9 pointers increases in the second semester or not. 

# In[ ]:


sem2_df = pd.DataFrame(data_2017['SEM 2'].value_counts().reset_index(name='Student Count'))
sem2_df.rename(columns={'index':'Grade'},inplace=True)
sem2_df.head(10)


# In[ ]:


plt.scatter(sem2_df['Grade'],sem2_df['Student Count'])
plt.xlabel('Grades obtained in second semester')
plt.ylabel('Number of students')
plt.title('Grades vs. Number of students for Sem 2')
plt.show()


# In[ ]:


len(data_2017[data_2017['SEM 2']>=9])


# > Let's now find out if the number of 9 pointers increases in the second semester or not.
# 
# - Yes, certainly!

# Now, I want to plot this trend till the fifth semester. 

# In[ ]:


# nine_pointers = []
# sems = ['SEM 1', 'SEM 2', 'SEM 3', 'SEM 4', 'SEM 5']
# for sem in sems:
#     print(len(data_2017[data_2017[sem]>=9]))
# nine_pointers


# In[ ]:


# len(data_2017[data_2017['SEM 3']>=9])


# In[ ]:


data_2017['SEM 3']


# (The above two cells are commented out intentionally because the errors would hamper the Kernel to be committed. )
# The errors are because in some of the entries for `SEM 3` there are NaNs. For this, the column values are interpreted as strings not numbers. So we can replace these NaNs with zeros or we can drop them. 

# In[ ]:


sem3_without_na = data_2017['SEM 3'].dropna()
sem3_without_na = pd.to_numeric(sem3_without_na, errors = 'coerce')


# In[ ]:


sem3_without_na.sample(20)


# In[ ]:


len(sem3_without_na[lambda x: x>=9])


# Now it will work. So, for plotting, we will have to find out the number of students that got greater or equal to 9 for a given semester and plot it accordingly. 
# 

# In[ ]:


sem_1 = len(data_2017[data_2017['SEM 1']>=9])
sem_2 = len(data_2017[data_2017['SEM 2']>=9])
sem_3 = len(sem3_without_na[lambda x: x>=9])
sem_4 = len(data_2017[data_2017['SEM 4']>=9])
sem_5 = len(data_2017[data_2017['SEM 5']>=9])

nine_pointers_number = [sem_1, sem_2, sem_3, sem_4, sem_5]


# In[ ]:


sems = ['SEM 1', 'SEM 2', 'SEM 3', 'SEM 4', 'SEM 5']
fig, ax = plt.subplots()
plt.barh(sems,nine_pointers_number)
for i, v in enumerate(nine_pointers_number):
    ax.text(v+0.5, i, str(v), color='red', fontweight='bold')
plt.xlabel('Number of students who got 9 points')
plt.title('Count of 9 pointers in each semester')


# The value kept on increasing till fourth semester. 

# Many students have specified the computer programming languages that they know (`COMPUTER LANGUAGES KNOWN` column). I want to find out the language known by highest number of students. 

# ## 13. What is the programming language that is known by the highest number of students?

# In[ ]:


prog_langs = pd.DataFrame(data_2017['COMPUTER LANGUAGES KNOWN'].value_counts().reset_index(name = 'Count'))
prog_langs.rename(columns={'index':'Programming Language'},inplace = True)
prog_langs.head(10)


# The noise is real here, again. Although we can easily see that **C programming language** is the answer to my question. But what if I want to see the fourth highest language? Good amount of data cleaning is required here. 

# In[ ]:


prog_langs_cleaned =     prog_langs.set_index('Count')    ['Programming Language'].str.split(',', expand=True).stack()    .reset_index('Count')


# In[ ]:


prog_langs_cleaned.rename(columns={0:'Programming Language'},inplace=True)
prog_langs_cleaned.head()


# We split the data w.r.t comma only but there can be other delimiters as well. So, finding that out will be another task in itself and it refers to the domain of text cleaning. Also, There are values which do not really conform to being computer programming languages for example - data structures, ms-office and so on. We cannot include such values. This can be for faulty data collection. 

# In[ ]:


prog_langs_cleaned['Programming Language'].unique()


# We need some functionality that would treat words like C PROGRAMMING and C-PROGRAMMING one and the same. 
# We can use **Levenshtein distance** to incorporate similarity measurement to extract the similar words. For example Programming C and C Programming means essentially the same in this context. But we can end this question here to not go out of the scope. 

# With this question I end my analysis. Feel free to download the dataset and come up with your own study and let me know about that. 
# 
# Thank you for your time :-)
