#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot
import seaborn as sns
import numpy as np


#Reading the data through pandas
df = pd.read_csv('/kaggle/input/2022-ukraine-russian-war/russia_losses_equipment.csv')
df.head()


# In[ ]:


#checking for null data
df.isna().sum()


# In[ ]:


#filling null data by zero
df.fillna(0, inplace=True)
df['special equipment']= df['special equipment'].astype(int)
df['mobile SRBM system']= df['mobile SRBM system'].astype(int)
df['cruise missiles']= df['cruise missiles'].astype(int)
df['vehicles and fuel tanks']= df['vehicles and fuel tanks'].astype(int)


# In[ ]:


df.isna().sum()


# In[ ]:


i=0
while i<=64:
    df["vehicles and fuel tanks"][i]=df["military auto"][i]+df['fuel tank'][i]
    i+=1

df = df.drop(['military auto', 'fuel tank'], 1)


# In[ ]:


i=0
while i<=64:
    df["cruise missiles"][i]=df["mobile SRBM system"][i]+df['cruise missiles'][i]
    i+=1

df= df.drop("mobile SRBM system", 1)
df_subcopy=df.copy()
df


# In[ ]:


#cumulative describe
#df.describe()
#today = date.today()
#yesterday= today - timedelta(days = 1)

needed_list=["aircraft","helicopter","tank","APC","field artillery","MRL","drone","naval ship","anti-aircraft warfare","special equipment","vehicles and fuel tanks","cruise missiles"]
print('\033[1m'"The Equipment Losses till ",df['date'].iloc[-1]," are listed Below :"'\033[0m')
Total_loss_of_equipment=0
for i in needed_list:
    Total_loss_of_equipment = Total_loss_of_equipment +df[i].max()
    print(i,'\033[1m',df[i].max(),'\033[0m')

print('\033[1m'"The total loss of equipments : ",Total_loss_of_equipment,'\033[0m')


# **Total Equipments possessed by Russia before the war brokeout**
# 
# *Total Military Equipments of Russia : Took from https://armedforces.eu/*
# 
# * Aircraft : 5552
# * Helicopter : 1724
# * Tank : 12270
# * Artillery : 18497
# * APC : 26831
# * MRL : 4359
# * NAVAL SHIP : 664
# * Special Equip : 1070
# * Drones: 2000
# 

# In[ ]:


#changing the dataframe from cumulative to daily change
needed_list=["aircraft","helicopter","tank","APC","field artillery","MRL","drone","naval ship","anti-aircraft warfare","special equipment","vehicles and fuel tanks","cruise missiles"]
for (columnName, columnData)  in df.iteritems():
    if columnName in needed_list:
        i=0
        new_list=[]
        new_list.append(columnData[0])
        while i<len(columnData)-1:
            n=columnData[i+1]- columnData[i]
            new_list.append(n)
            i=i+1
        print(new_list)
        
        # Drop that column
        df.drop(columnName, axis = 1, inplace = True)

        # Put whatever series you want in its place
        df[columnName] = new_list


# In[ ]:


#daily change head
df.head()


# In[ ]:


#today = date.today()
#yesterday= today - timedelta(days = 1)
needed_list=["aircraft","helicopter","tank","APC","field artillery","MRL","drone","naval ship","anti-aircraft warfare","special equipment","vehicles and fuel tanks","cruise missiles"]
print('\033[1m'"Equipment Loss today (",df['date'].iloc[-1],") : are listed below "'\033[0m')
for i in needed_list:
    print(i,'\033[1m',df[i].iloc[-1],'\033[0m')


# In[ ]:


#correlation matrix for daily change
df_corr=df.drop('day',1)

correlation_mat = df_corr.corr() 
top_corr_features = correlation_mat.index
plt.figure(figsize=(20,20))
#plot heat map
g=sns.heatmap(df[top_corr_features].corr(),annot=True,cmap="RdYlGn")


# In[ ]:


#Normal Plotting for day vs equipment under daily change
#needed_list=["aircraft","helicopter","tank","APC","field artillery","MRL","military auto","fuel tank","drone","naval ship","anti-aircraft warfare","special equipment","mobile SRBM system"]
#for (columnName_1, columnData_1)  in df.iteritems():
 #   for (columnName_2, columnData_2)  in df.iteritems():
  #      if columnName_1=="day" and columnName_2 in needed_list:
   #         plt.stem(columnData_1,columnData_2)
    #        plt.xlabel(columnName_1)
     #       plt.ylabel(columnName_2)
      #      plt.show()

#df1 = pd.read_csv('/kaggle/input/2022-ukraine-russian-war/russia_losses_equipment.csv')
needed_list=["aircraft","helicopter","tank","APC","field artillery","MRL","drone","naval ship","anti-aircraft warfare","special equipment","vehicles and fuel tanks","cruise missiles"]
#Merged Plotting for both cumulative and daily change
for (columnName_1, columnData_1)  in df.iteritems():
    for (columnName_2, columnData_2)  in df.iteritems():
        if columnName_1=="day" and columnName_2 in needed_list:
            data=[df[columnName_1],df[columnName_2],df_subcopy[columnName_2]]
            headers=["day","daily_aircraft_count","cumulative aircraft count"]
            plot(pd.concat(data,axis=1,keys=headers).set_index('day'))
            plt.xlabel("Days")
            plt.ylabel("count of destroyed")
            plt.title("cumulative "+columnName_2+" count vs daily "+columnName_2+" count")
            plt.show()
            
#All Feature PairGrid Plot - Seaborn
#import seaborn as sns
#g = sns.PairGrid(df)
#g.map(sns.scatterplot)


# In[ ]:


#First 25 days vs last remaining days  -  a bar to explore insights

#plt.barh(df.day[24],df.aircraft[:24].cumsum())
#plt.show()


bar1_main=[]
bar2_main=[]
for (columnName, columnData)  in df.iteritems():
    X_axis = np.arange(len(needed_list))
    if columnName in needed_list:
        bar1=columnData[:39].sum()
        bar2=columnData[39:].sum()
        bar1_main.append(bar1)
        bar2_main.append(bar2)
        
plt.bar(X_axis - 0.2, bar1_main , 0.4, label = 'first 40 Days')
plt.bar(X_axis + 0.2, bar2_main , 0.4, label = 'After 40 Days')
plot_name=["aircraft","helicopter","tank","APC","artillery","MRL","drone","ship","anti-aircraft","special_equip","vehi & fuel tank","cruise missiles"]
plt.xticks(X_axis, plot_name)
fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)
plt.legend()
plt.xlabel("Features",labelpad=10)
plt.ylabel("Equipments Destroyed by ukraine",labelpad=10)
plt.show()


# In[ ]:


#Daily Total Loss of equipments
#new_df=pd.read_csv('/kaggle/input/2022-ukraine-russian-war/russia_losses_equipment.csv')
total_loss_day_basis = df.copy()
total_loss_day_basis.drop(columns={'date'}, inplace=True)
total_loss_day_basis.set_index('day', inplace=True)
total_loss_day_basis['Daily equipment loss'] = pd.DataFrame(total_loss_day_basis.sum(axis=1))
total_loss_day_basis['Daily equipment loss'].plot(figsize=(16,6),kind="area")
plt.xlabel('days')
plt.ylabel('equipment count loss')
plt.show()


# In[ ]:


#Reading the data through pandas
df_personnel = pd.read_csv('/kaggle/input/2022-ukraine-russian-war/russia_losses_personnel.csv')
df_personnel.head()


# In[ ]:


df_personnel.describe()


# In[ ]:


initial = 0
Data = []

for i in (df_personnel['POW'].values):
    value = i - initial
    Data.append(value)
    initial = i

df_personnel['Daily increase in POW'] = Data
df_personnel_structured = df_personnel[['day', 'POW', 'Daily increase in POW']].set_index('day')
df_personnel_structured.rename(columns={'POW':'Daily total POW'}, inplace=True)
#df_personnel_structured
df_personnel_structured.plot(figsize=(16,6))
plt.xlabel('Days')
plt.ylabel('Prisoners count')
plt.title("Daily Total POW VS Daily Increase in POW")
plt.show()
        


# In[ ]:


#df_personnel_structured.describe()

#today = date.today()
#yesterday= today - timedelta(days = 1)
yesterday_personnel_loss=df_personnel_structured["Daily increase in POW"].iloc[-1]
print('\033[1m'"Total Personnel Losses ",df_personnel['POW'].max(),'\033[0m')
print('\033[1m'"Personnel Loss Yesterday (",df_personnel['date'].iloc[-1],") : ",yesterday_personnel_loss,'\033[0m')

