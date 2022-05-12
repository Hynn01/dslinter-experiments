#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go 
import plotly.offline as py
py.init_notebook_mode(connected=True) 


from sklearn.preprocessing import StandardScaler


# In[ ]:





# In[ ]:


train_data = pd.read_csv("/kaggle/input/titanic/train.csv")


# In[ ]:


train_data.info()


# In[ ]:


train_data.head(10)


# In[ ]:


# train_data.columns


# In[ ]:


train_data.shape


# In[ ]:


train_data.describe().T


# In[ ]:





# In[ ]:





# In[ ]:


train_data.isna().sum()


# In[ ]:


train_data.isna().sum()


# ### Pie plot of the target column distribution:

# In[ ]:


trace = go.Pie(labels = train_data["Survived"].value_counts().keys().tolist(),
               values =  train_data["Survived"].value_counts().values.tolist(),
               marker = dict(colors = ['royalblue','lime'],
                             line = dict(color = "white", width =  1.3)
                            ),
               rotation = 90,
               hoverinfo = "label+value+text",
               hole = .5
              )
layout = go.Layout(dict(title = "Customer churn Telco dataset ",
                        plot_bgcolor = "orange",
                        paper_bgcolor = "white",
                       )
                  )
data = [trace]
fig = go.Figure(data = data, layout = layout)
py.iplot(fig)


# In[ ]:


train_data.groupby("Survived").size()


# #### First i will make a copy of the original dataset and manipulate the copy. The original will stay intact

# In[ ]:


train_data_tidy = train_data.copy()


# #### Verifying Cabin missing values:
# 

# In[ ]:


percent_missing_values_cabin = train_data.Cabin.isna().sum()/len(train_data.Cabin)*100
print(f'Percent missing values: {round(percent_missing_values_cabin,2)} %.')
# percent_missing_values_cabin


# It is safe to remove the column because less than 30% of the data is existing.

# In[ ]:


train_data_tidy.drop("Cabin",axis = 1, inplace= True)


# #### Analyzing Age column:

# 19.87% of the total Age column values are missing. From the missing values 58% are not survived from Pclass 3.
# I will fill the missing values with the mean of the survived from Pclass 3. I will plot the new distribution of the Age column.

# In[ ]:





# In[ ]:


train_data_tidy.Age.isnull().sum()


# In[ ]:


percent_missing_values_age = train_data.Age.isnull().sum()/len(train_data.Age)*100
print(f'Percent missing values: {round(percent_missing_values_age,2)} %.')
# percent_missing_values_cabin


# In[ ]:


missing_age_values = train_data_tidy[np.isnan(train_data_tidy.Age)]
missing_age_values


# In[ ]:


plt.hist(train_data_tidy.Age,bins="fd")
plt.xlabel("Age of the passengers")
plt.ylabel("Count")
plt.show()


# In[ ]:


""" x -> Missing Age values , not survived, from 3d class""" 

x = train_data_tidy[(train_data_tidy.Age.isnull()) & (train_data_tidy.Survived==0) & (train_data.Pclass ==3)]


# In[ ]:


print(f'Percent of  the missing age values who aren`t survived and who are from 3d Pclass : {round(len(x)/len(missing_age_values),2)*100} % .')

# round(len(x)/len(missing_age_values),2)


# In[ ]:


x_present = train_data_tidy[(train_data_tidy.Age.notnull()) & (train_data_tidy.Survived==0) & (train_data.Pclass ==3)]


# In[ ]:


x_present


# In[ ]:


x_present.Age.mean()


# In[ ]:


mean_age_of_not_survived = (train_data_tidy[(train_data_tidy.Age.notna()) & (train_data_tidy.Survived == 0)]).mean()
mean_age_of_not_survived


# In[ ]:


train_data_tidy.Age=train_data_tidy.Age.fillna(mean_age_of_not_survived[3])


# In[ ]:


train_data.Age.mean()


# In[ ]:


train_data_tidy.Age.mode()


# In[ ]:


train_data_tidy.Age.isna().sum()


# In[ ]:


plt.hist(train_data_tidy.Age,bins ="fd")
plt.xlabel("Age of the passengers")
plt.ylabel("Count")
plt.show()


# #### Analysing  Pclass column

# Column contains 3 classes.Imbalanced distribution. 3d class represents 55.11% from all the passengers. Only 47 male susrvived from 3d class passengers. For differnce first class represents 24.24% of the total prassengers. 45 male survived from first class.

# Column contains 3 classes.Imbalanced distribution

# In[ ]:


train_data_tidy.Pclass.unique()


# In[ ]:


plt.hist(train_data_tidy.Pclass)
plt.xlabel("Class")
plt.ylabel("Count")
plt.show()


# In[ ]:


pd.pivot_table(train_data_tidy,  values = "Survived", columns = "Sex", index = "Pclass")


# In[ ]:


third_class_passengers = len(train_data_tidy[train_data_tidy.Pclass == 3]) / len(train_data_tidy.Pclass)*100
print(f'Percent of the passengers in 3d class vs all: {round(third_class_passengers,2)}%.' )


# In[ ]:


survivde_male_3d_class = train_data_tidy[(train_data_tidy.Pclass==3) & (train_data_tidy.Sex =="male") & (train_data_tidy.Survived == 1)].shape
print(f"Number of survived male from 3d class {survivde_male_3d_class[0]}.")


# In[ ]:


first_class_passengers = len(train_data_tidy[train_data_tidy.Pclass == 1]) / len(train_data_tidy.Pclass)*100
print(f'Percent of the passengers in 3d class vs all: {round(first_class_passengers,2)}%.' )


# In[ ]:


survivde_male_3d_class = train_data_tidy[(train_data_tidy.Pclass==1) & (train_data_tidy.Sex =="male") & (train_data_tidy.Survived == 1)].shape
print(f"Number of survived male from 1d class {survivde_male_3d_class[0]}.")


# #### PassengerId column and Name column

# There are no missing values in both PassengerI and Name. I will keep a part Name column and i will remove it from the dataset. The original dataset will stay intact.

# In[ ]:


print(f'Number of missing value in PassengerId column {train_data_tidy.PassengerId.isnull().sum()}.' )


# In[ ]:


print(f'Number of missing value in Name column {train_data_tidy.Name.isnull().sum()}.' )


# In[ ]:


name_column = train_data_tidy.Name
train_data_tidy.drop("Name", axis=1,inplace=True)


# In[ ]:


train_data_tidy.head(5)


# In[ ]:


"""Last verification if any values is missing""" 

if len(train_data_tidy.PassengerId) == train_data_tidy.shape[0]:
    print("No missing values.")


# #### Analysing Sex  column

# Highly imbalanced data. Number of female passenger of board: 314 with 74.2 % survival rate. Number of male passengers of board: 577 with survival rate 18.89 %. Sex column is highly correlated with the Survival. Female survival rate is much times higher, the main reason it the rescue orders "female and kids"first. This observation is confirmed by th fact that the number distribution of the survived female is almost the same in the 3 Pclass.

# In[ ]:


plt.hist(train_data_tidy.Sex)
plt.xlabel("Male/Femae")
plt.ylabel("Count")
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:


pd.pivot_table(train_data_tidy,values = "Fare", columns="Sex", index= "Survived",aggfunc="count")


# In[ ]:


print(f'The total number of male passengers onboard : {train_data_tidy[train_data_tidy["Sex"] == "male"].shape[0]}.' )


# In[ ]:


train_data_tidy[train_data_tidy["Sex"] == "female"].shape[0]


# In[ ]:


print(f'The total number of female passengers onboard : {train_data_tidy[train_data_tidy["Sex"] == "female"].shape[0]}.' )


# In[ ]:


total_number_survived_male = train_data_tidy[(train_data_tidy["Sex"] == "male") & (train_data_tidy["Survived"]==1)].shape[0]
print(f'Total number of survived male {total_number_survived_male}.')


# In[ ]:


total_number_survived_female = train_data_tidy[(train_data_tidy["Sex"] == "female") & (train_data_tidy["Survived"]==1)].shape[0]
print(f'Total number of survived female {total_number_survived_female}.')


# In[ ]:


print(f'Survival rate among female passengers: {round(total_number_survived_female /(train_data_tidy[train_data_tidy["Sex"] == "female"].shape[0])*100,2)} %. ')


# In[ ]:


round(total_number_survived_male/(train_data_tidy[train_data_tidy["Sex"] == "male"].shape[0])*100,2)


# In[ ]:


print(f'Survival rate among male passengers: {round(total_number_survived_male/(train_data_tidy[train_data_tidy["Sex"] == "male"].shape[0])*100,2)} %. ')


# In[ ]:





# In[ ]:


total_number_survived_female_1st_class = train_data_tidy[(train_data_tidy["Sex"] == "female") & (train_data_tidy["Survived"]==1) & (train_data_tidy["Pclass"]==1)].shape[0]


# In[ ]:


print(f"Number of survived female from 1st class: {total_number_survived_female_1st_class }.")


# In[ ]:


total_number_survived_female_3d_class = train_data_tidy[(train_data_tidy["Sex"] == "female") & (train_data_tidy["Survived"]==1) & (train_data_tidy["Pclass"]==3)].shape[0]
print(f"Number of survived female from 3d class: {total_number_survived_female_3d_class}.")


# In[ ]:


pd.pivot_table(train_data_tidy,values= "Fare" ,index = "Survived", columns = ["Sex", "Pclass"], aggfunc="count")


# #### Analysing Sibsp / siblings  on board  and Parch/ Number of Parents/Children Aboard columns

# In[ ]:


print(f"Mean sibling member of bord {round(train_data_tidy.SibSp.mean(),2)}")
print(f"Mean parents and children familly members of bord {round(train_data_tidy.Parch.mean(),2)}")


# In[ ]:


train_data_tidy.SibSp.unique()


# In[ ]:


train_data_tidy.Parch.unique()


# In[ ]:


train_data_tidy.SibSp.mean()


# In[ ]:


train_data_tidy.Parch.mean()


# #### Analysing Ticket column:

# No missing values in the columns. Only 76% of the passenger own a single tickets. The reason might be that the siblings,children and parents have a combinated ticket. This column will be irrelevant for the model, I will drop it.

# In[ ]:


len(train_data_tidy.Ticket.unique())


# In[ ]:


train_data_tidy.Ticket.isna().sum()


# In[ ]:


print(f'Percent of the unique ticket over all the passengers on board {round(len(train_data_tidy.Ticket.unique()) / len(train_data_tidy)*100)}%.')


# In[ ]:


train_data_tidy.drop("Ticket",axis=1,inplace=True)


# #### Analysing Fare column:

# Survival enverselly propotionnal to the Fare rate. Only 27 customers paid Fare more that 150 USA and 68.97% of them survived. 838 passengers had paid Fare between 0 and 150 USA, only 36.14% of them survived.

# In[ ]:


train_data_tidy.Fare.mean()


# In[ ]:


train_data_tidy.Fare.min()


# In[ ]:


train_data_tidy.Fare.max()


# In[ ]:


plt.hist(train_data_tidy.Fare)
plt.xlabel("Fare")
plt.ylabel("Count")
plt.show()


# In[ ]:


print(f'Number of passengers paid Fare more than 150 USD is {train_data_tidy[train_data_tidy["Fare"] > 150].shape[0]}.')


# In[ ]:


print(f'Number of passengers paid Fare between 0 and 100 USD is {train_data_tidy[train_data_tidy["Fare"] <= 100].shape[0]}.')


# In[ ]:


print(f'Number of passenger paid Fare more that 150 USD who survived is {train_data_tidy[(train_data_tidy["Fare"] > 150) & (train_data_tidy["Survived"] ==1)].shape[0]}.')


# In[ ]:


survival_ratio_up_150_fare = round(train_data_tidy[(train_data_tidy["Fare"] > 150) & (train_data_tidy["Survived"] ==1)].shape[0]/train_data_tidy[train_data_tidy["Fare"] > 150].shape[0] *100,2)


# In[ ]:


print(f"Ration btw passengers with fare more that 150 USA and the survived among them is {survival_ratio_up_150_fare} %.")


# In[ ]:


print(f'Number of  female passengers paid Fare more that 150 USD who survived is {train_data_tidy[(train_data_tidy["Fare"] > 150) & (train_data_tidy["Survived"] ==1) & (train_data_tidy["Sex"]=="female")].shape[0]}.')


# In[ ]:


train_data_tidy[(train_data_tidy["Fare"] < 150) & (train_data_tidy["Survived"] == 0)].shape[0] 


# In[ ]:


ratio_fare_less_150_survival = train_data_tidy[(train_data_tidy["Fare"] <150) & train_data["Survived"] ==1].shape[0]/len(train_data_tidy["Survived"] == 1)*100


# In[ ]:


print(f'Ratio btw passengers who did survived and passengers among them with Fare rate less than 150 :{round(ratio_fare_less_150_survival,2)}%.')


# ####  Analysis of Embarked column:

# Imbalanced data. Almost half of all the passenger embarked from "S" and the deadrate is inversely proportional.8% of the passengers embarked from 'S' did`t survived vs 8% from embarkment 'C' and 5% from 'Q'. This is highly correlated with the Fare rate.

# In[ ]:


train_data_tidy.Embarked.unique()


# In[ ]:


train_data_tidy.Embarked.isna().sum()


# In[ ]:


nan_embarked_values = train_data_tidy[train_data_tidy["Embarked"].isna()]
nan_embarked_values


# In[ ]:





# In[ ]:


x = train_data_tidy[(train_data_tidy["Survived"] == 1) & (train_data_tidy["Sex"]=="female") & (train_data_tidy["Fare"] < 100 )]
print(f'Majority of  the female passenger who survived and paid Fare less than  100 USD embarked from {x.Embarked.mode()}.')


# In[ ]:


train_data_tidy["Embarked"].fillna("S", inplace=True)


# In[ ]:


train_data_tidy["Embarked"].unique()


# In[ ]:


train_data_tidy["Embarked"].isna().sum()


# In[ ]:


plt.hist(train_data_tidy["Embarked"])
plt.xlabel("From")
plt.ylabel("Count")
plt.show()


# In[ ]:


no_survival_rate_embarkement_C = train_data_tidy[(train_data_tidy["Survived"] == 0) &(train_data_tidy["Embarked"] == "C")].shape[0]/len(train_data_tidy["Embarked"] == "C")*100
no_survival_rate_embarkement_Q = train_data_tidy[(train_data_tidy["Survived"] == 0) &(train_data_tidy["Embarked"] == "Q")].shape[0]/len(train_data_tidy["Embarked"] == "Q")*100
no_survival_rate_embarkement_S= train_data_tidy[(train_data_tidy["Survived"] == 0) &(train_data_tidy["Embarked"] == "S")].shape[0]/len(train_data_tidy["Embarked"] == "S")*100
print(f"{round(no_survival_rate_embarkement_S)}% of the passengers embarked from 'S' did`t survived vs {round(no_survival_rate_embarkement_C )}% from embarkment 'C' and {round(no_survival_rate_embarkement_Q)}% from 'Q'.")


# ## Plot of all the relations between each column and the target "Survived"

# In[ ]:


for i, j in enumerate(train_data_tidy.drop(["PassengerId", "Age", "Fare"],axis=1)):
    plt.figure(i)
    sns.countplot(data=train_data, x= j, hue="Survived")


# In[ ]:


train_data_tidy.info()


# # Preparing the dataset from the prediction model

# In[ ]:


train_data_modeling = train_data_tidy.copy()


# In[ ]:


train_data_tidy.to_csv("train_data_tidy.csv")


# In[ ]:


train_data_modeling[["Pclass", "SibSp","Parch" , "Embarked", "Sex"]] = train_data_modeling[["Pclass", "SibSp","Parch" , "Embarked","Sex"]].astype("category")


# In[ ]:


train_data_modeling.info()


# In[ ]:


train_data_modeling.Fare.mean()


# In[ ]:


train_data_modeling= pd.get_dummies(train_data_modeling)


# In[ ]:


train_data_modeling


# In[ ]:


scaler= StandardScaler()
train_data_modeling[["Fare", "Age"]] = scaler.fit_transform(train_data_modeling[["Fare", "Age"]] )


# In[ ]:


train_data_modeling.head(20)


# In[ ]:


train_data_modeling.to_csv("train_data_modeling.csv", index=None)


# In[ ]:


train_data_modeling.columns


# In[ ]:


train_data_modeling.Fare.mean()


# In[ ]:




