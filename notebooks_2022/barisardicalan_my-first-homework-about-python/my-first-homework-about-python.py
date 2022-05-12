#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns 
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# ## 1)INTRODUCTION TO PYTHON (RANDOM EXAMPLES FOR MY HOMEWORK)

# In[ ]:


data = pd.read_csv("/kaggle/input/2022-ukraine-russian-war/russia_losses_equipment.csv")
data.head()


# In[ ]:


data.info()


# In[ ]:


data.corr()


# In[ ]:


#correlation map
f,ax = plt.subplots(figsize=(12, 12)) # Made (12) because when u slide down to the correlation map you can see the whole dataset in one piece.
sns.heatmap(data.corr(), annot=True, linewidths=2, fmt= '.1f',ax=ax)
plt.show()


# In[ ]:


# I want to try Solarized Light stylesheet , Pek anlayamadım hocam ne nereye gidiyor internetten bakarak yapmaya çalıştım biraz değiştirdim.
np.random.seed(1000)

x = np.linspace(600,10)
with plt.style.context('Solarize_Light2'):
    plt.plot(x, np.sin(x) + x + np.random.randn(50))
    plt.plot(x, np.sin(x) + 10 * x + np.random.randn(50))
    # Number of accent colors in the color scheme
    plt.title('Aircraft & Helicopter - Line')
    plt.xlabel('Aircraft', fontsize=16)
    plt.ylabel('Helicopter', fontsize=16)

plt.show()


# In[ ]:


data.head(20) # it is almost 61 days i just wanted to show you the first 20 data


# In[ ]:


data.columns


# In[ ]:


# Line Plot
# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line
data.aircraft.plot(kind = 'line', color = 'g',label = 'aircraft',linewidth=5,alpha = 0.9,grid = True,linestyle = '-.')
data.tank.plot(color = 'r',label = 'tank',linewidth=5, alpha = 0.9,grid = True,linestyle = ':')
data.helicopter.plot(color = 'b',label = 'helicopter',linewidth=5, alpha = 0.9,grid = True,linestyle = '--')
plt.legend(loc='upper left')     # legend = puts label into plot
plt.xlabel('Plane')              # label = name of label
plt.ylabel('Vehicle')
plt.title('Line Plot')            # title = title of plot
data.plot(figsize=(15,10)) # figsize = making it bigger (Hocam line plot büyüdü tamam güzel ama ; diğer arkadaşları çıkarmayı başaramadım.)
plt.show()


# In[ ]:


data.columns
plt.scatter(data.aircraft,data.tank,data.helicopter, color ="blue",alpha=0.5)


# In[ ]:


# Scatter Plot 
# x = Aircraft, y = Helicopter
data.plot(kind='scatter', x='aircraft', y='helicopter',alpha = 0.5,color = 'red')
plt.xlabel('Aircraft')              # label = name of label
plt.ylabel('Helicopter')
plt.title('Aircraft & Helicopter Scatter Plot')            # title = title of plot


# In[ ]:


# Histogram
# bins = number of bar in figure
data.aircraft.plot(kind = 'hist',bins = 25,figsize = (10,10))
plt.show()


# In[ ]:


# Histogram
# bins = number of bar in figure
data.helicopter.plot(color = 'green' , kind = 'hist',bins = 25,figsize = (10,10))
plt.show()


# In[ ]:


# Histogram
# bins = number of bar in figure
data.tank.plot(color = 'red' , kind = 'hist',bins = 25,figsize = (10,10))
plt.show()


# In[ ]:


# clf() = cleans it up again you can start a fresh
data.aircraft.plot(kind = 'hist',bins = 50)
plt.clf()
# We cannot see plot due to clf()


# In[ ]:


#create dictionary and look its keys and values
dictionary = {'aircraft' : 'Mig-29' ,'helicopter' : 'Mi-24' , 'tank':'T-14'}
print(dictionary.keys())
print(dictionary.values())


# In[ ]:


# Keys have to be immutable objects like string, boolean, float, integer or tubles
# List is not immutable
# Keys are unique
dictionary['aircraft'] = "Mig-29"    # update existing entry
print(dictionary)
dictionary['helicopter'] = "Mi-24"       # Add new entry
print(dictionary)
del dictionary['tank']              # remove entry with key 'tank'
print(dictionary)
print('helicopter' in dictionary)        # check include or not
dictionary.clear()   # remove all entries in dict
print(dictionary)


# In[ ]:


# 1 - Filtering Pandas data frame
x = data['aircraft']>20    #  aircraft more than 20
data[x]
data[np.logical_and(data['aircraft']>10, data['helicopter']<500)] # aircraft more than 10 & helicopter less than 500


# In[ ]:


# 2 - Filtering pandas with logical_and
# There are 6 days who have higher aircraft than 175 & higher helicopter than 150
data[np.logical_and(data['aircraft']>175, data['helicopter']>150)]


# In[ ]:


# This is also same with previous code line. Therefore we can also use '&' for filtering.
data[(data['aircraft']>20) & (data['helicopter']>10)]


# In[ ]:


dictionary = {'aircraft':'Mig-29','helicopter':'Mi-24','tank' : 'T-14'}

for key,value in dictionary.items():

    print(value," : ",key)


# ## 2)DATA SCIENCE TOOLBOX (RANDOM EXAMPLES)

# In[ ]:


def tuple_ex():
    x = (1,2,3)
    return x
a,b,c = tuple_ex()
print(a,b,c)


# In[ ]:


# Different types of tuples

# Empty tuple
my_tuple = ()
print(my_tuple)

# Tuple having integers
my_tuple = (1, 2, 3)
print(my_tuple)

# tuple with mixed datatypes
my_tuple = (1, "Hello", 3.4)
print(my_tuple)

# nested tuple
my_tuple = ("mouse", [8, 4, 6], (1, 2, 3))
print(my_tuple)

#
my_tuple = year_born
year_born = ("Paris Hilton",1981)
print(my_tuple)
year_born[0]
year_born[1]


# In[ ]:


b = ("Barış",30,"Python")
b = (name,age,studies)
name, age,studies


# In[ ]:


x = 3

def f():
    x = 4
    return x 
    
print(x)
print(f())


# In[ ]:


x = 3
def f():
    y = 3*x
    return y

print(x)
print(f())


# In[ ]:


import builtins
dir(builtins)


# In[ ]:


def square():
    
    x = 1
    y = 2
    z = x+y 
    return z

print(square())


# In[ ]:


def square():
    def add():
        x = 1
        y = 2
        z = x+y
        return z
    return add()**3
print(square())


# In[ ]:


def f(a,b=1,c=2):
    y = a+b+c
    return y 
print(f(10))
 
print(f(3,2,1))


# In[ ]:


def f(*args):
    for i in args:
        print(i)

f(1)
print("")
f(1,2,3,4)

def f(**kwargs):
    for key, value in kwargs.items():
        print(key," ",value)
        
f(country = 'spain',capital='madrid',population=123456)


# In[ ]:


square = lambda x:x**2
print(square(5))
tot = lambda x,y,z:x+y+z
print(tot(2,3,5))


# In[ ]:


v1 = lambda x : ((x+5)/5)**5
print(v1(5))


# In[ ]:


number_list = [4,5,6]
y = map(lambda x:x**5,number_list)
print(list(y))


# In[ ]:


name = 'messi'
it = iter(name)
print(next(it)) # print next iteration
print(*it) # print remaining iteration


# In[ ]:


liste = 'Barış'
it = iter(liste)
print(next(it))
print(*it)


# In[ ]:


list1 = [5,6,7,8]
list2 = [9,10,11,12]
z = zip (list1,list2)
print(z)
print(z_list)


# In[ ]:


un_zip = zip(z_list)
print(un_list1)
print(un_list2)
print(type(z_list))


# In[ ]:


num1 = [4,5,6]
num2 = [i*2 for i in num1]
print(num2)


# In[ ]:


num1 = [4,8,12]
num2 = [i**2 if i == 8 else i-4 if i < 8 else i+4 for i in num1]
print(num2)


# In[ ]:


# lets return pokemon csv and make one more list comprehension example
# lets classify pokemons whether they have high or low speed. Our threshold is average speed.
threshold = sum(data.aircraft)/len(data.aircraft)
data["field_artillery"] = ["high" if i > threshold else "low" for i in data.aircraft]
data.loc[:10,["field_artillery","aircraft"]] # we will learn loc more detailed later


# ## 3) CLEANING DATA

# In[ ]:


data = pd.read_csv("/kaggle/input/2022-ukraine-russian-war/russia_losses_equipment.csv")
data.head()


# In[ ]:


data.value_counts()


# In[ ]:


# tail shows last 5 Rows
data.tail()


# In[ ]:


data.columns


# In[ ]:


data.info()


# In[ ]:


print(data.value_counts(dropna =False))


# In[ ]:


1,2,3,4,500


# In[ ]:


data.describe()


# In[ ]:


data.boxplot(column='aircraft',by = 'helicopter', figsize = (15,15))


# In[ ]:


# Firs of all , I a create new data from pokemons data to explain melt more easily.
data_new = data.head()
data_new


# In[ ]:


# lets melt
# id_vars = what we do not wish to melt
# value_vars = what we want to melt
melted = pd.melt(frame=data_new,id_vars = 'date', value_vars= ['aircraft','helicopter'])
melted


# In[ ]:


# Index is name
# I want to make that columns are variable
# Finally values in columns are value
melted.pivot(index = 'date', columns = 'variable',values='value')


# In[ ]:


# Firstly lets create 2 data frame
data1 = data.head()
data2= data.tail()
conc_data_row = pd.concat([data1,data2],axis =0,ignore_index =True) # axis = 0 : adds dataframes in row
conc_data_row


# In[ ]:


data1 = data['aircraft'].head()
data2= data['helicopter'].head()
conc_data_col = pd.concat([data1,data2],axis =1) # axis = 1 : adds dataframes in column
conc_data_col


# In[ ]:


data.dtypes


# In[ ]:


# lets convert object(str) to categorical and int to float.
data['aircraft'] = data['aircraft'].astype('category')
data['helicopter'] = data['helicopter'].astype('float')


# In[ ]:


data.dtypes


# In[ ]:


data.info()


# In[ ]:


data["aircraft"].value_counts(dropna =False)


# In[ ]:


# Lets drop nan values
data1=data   # also we will use data to fill missing value so I assign it to data1 variable
data1["aircraft"].dropna(inplace = True)  # inplace = True means we do not assign it to new variable. Changes automatically assigned to data
# So does it work ?


# In[ ]:


#  Lets check with assert statement
# Assert statement:
assert 1==1 # return nothing because it is true


# In[ ]:


# In order to run all code, we need to make this line comment
# assert 1==2 # return error because it is false
assert 1 == 2


# In[ ]:


assert  data['aircraft'].notnull().all() # returns nothing because we drop nan values


# In[ ]:


data["tank"].fillna('tank',inplace = True)


# In[ ]:


assert  data['aircraft'].notnull().all() # returns nothing because we do not have nan values


# ## 4) PANDAS FOUNDATION

# In[ ]:


# data frames from dictionary
vehicle = ["aircraft","helicopter","tank"]
quantity = ["50","60","40"]
list_lable = ["vehicle","quantity"]
list_col = [vehicle,quantity]
zipped ) list(zip)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




