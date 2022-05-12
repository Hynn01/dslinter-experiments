#!/usr/bin/env python
# coding: utf-8

# **Import all necessary library**

# In[ ]:


import pandas as pd 
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import plotly as py
import plotly.graph_objs as go


# Import Dataset

# In[ ]:


train = pd.read_csv("/kaggle/input/spaceship-titanic/train.csv")
train.head()


# In[ ]:


test = pd.read_csv("/kaggle/input/spaceship-titanic/test.csv")
test.head()


# Data Preprocessing

# In[ ]:


train.shape


# In[ ]:


test.shape


# In[ ]:


train.info()


# In[ ]:


train.isnull().sum()


# In[ ]:


test.isnull().sum()


# In[ ]:


train['Cabin'].unique()


# In[ ]:


hp = train['Cabin'].value_counts()
hp


# In[ ]:


train['HomePlanet'].fillna(method='bfill', inplace=True)
train['CryoSleep'].fillna('False', inplace=True)
train['Cabin'].fillna(method='bfill', inplace=True)
train['Destination'].fillna('PSO J318.5-22', inplace=True)
train['Age'].fillna(method='bfill', inplace=True)
train['VIP'].fillna('False', inplace=True)
train['RoomService'].fillna(method='ffill', inplace=True)
train['FoodCourt'].fillna(method='ffill', inplace=True)
train['ShoppingMall'].fillna(method='ffill', inplace=True)
train['Spa'].fillna(method='ffill', inplace=True)
train['VRDeck'].fillna(method='ffill', inplace=True)
train['Name'].fillna('ZZZ', inplace=True)

test['HomePlanet'].fillna(method='bfill', inplace=True)
test['CryoSleep'].fillna('False', inplace=True)
test['Cabin'].fillna(method='bfill', inplace=True)
test['Destination'].fillna('PSO J318.5-22', inplace=True)
test['Age'].fillna(method='bfill', inplace=True)
test['VIP'].fillna('False', inplace=True)
test['RoomService'].fillna(method='ffill', inplace=True)
test['FoodCourt'].fillna(method='ffill', inplace=True)
test['ShoppingMall'].fillna(method='ffill', inplace=True)
test['Spa'].fillna(method='ffill', inplace=True)
test['VRDeck'].fillna(method='ffill', inplace=True)
test['Name'].fillna('ZZZ', inplace=True)


# In[ ]:


train.info()


# In[ ]:


test.info()


# In[ ]:


train.isnull().sum()


# In[ ]:


test.isnull().sum()


# **EDA**

# In[ ]:


train.head()


# In[ ]:


# bar graph on planet distribution

train['HomePlanet'].unique()


# In[ ]:


ev = train['HomePlanet'].value_counts()
ev


# In[ ]:


fig = plt.figure(figsize = (8, 4))

# Creating a Bar plot
plt.bar(ev.index, ev, color='purple', width=0.3)

plt.title('HomePlanet Distribution')
plt.xlabel('HomePlanet')
plt.ylabel('Total Passenger')
plt.show()


# In[ ]:


# Plot a histogram on Age

plt.hist(train['Age'])


# In[ ]:


# people_count = train.groupby('HomePlanet')['Transported'].count()
people_count = train.pivot_table(index='HomePlanet',columns='Transported', aggfunc='count')['Destination']

people_count


# In[ ]:


people_count.index


# In[ ]:


people_count.iloc[:,0]


# In[ ]:


yes = train[train['Transported'] == True]
no = train[train['Transported'] == False]


# In[ ]:


yes.head()


# In[ ]:


yes_home = yes['HomePlanet'].value_counts()
yes_home


# In[ ]:


no_home = no['HomePlanet'].value_counts()
no_home


# In[ ]:


total_home = train['HomePlanet'].value_counts()
total_home


# In[ ]:


total_ppl = pd.DataFrame({
    'HomePlanet': total_home.index,
    'total Passenger': total_home.values,
    'Transported' : yes_home.values,
    'not_Transported' : no_home.values,
}, columns = ['HomePlanet', 'total Passenger', 'Transported', 'not_Transported'])

total_ppl


# In[ ]:


total_ppl.set_index('HomePlanet', inplace=True)


# In[ ]:


total_ppl


# In[ ]:


trace1 = go.Bar(
    y=total_ppl['Transported'].values,
    x=total_ppl.index,
    marker_color='indianred',
    name='Transported Passenger'
)

trace2 = go.Bar(
    y=total_ppl['not_Transported'].values,
    x=total_ppl.index,
    marker_color='lightsalmon',
    name='not_Transported Passenger'

)


data = [trace1, trace2]
layout = go.Layout(
    barmode='stack',
    title="Transported people by HomePlanet",
    xaxis={
        'title':"HomePlanet",
    },
    yaxis={
        'title':"Total Transported Passenger",
    }
)

fig = go.Figure(data=data, layout=layout)
py.offline.iplot(fig)


# In[ ]:


# Plot a piechart on transported people

Transported = train[train['Transported'] == True]
notTransported = train[train['Transported'] == False]

print("Transported: ", len(Transported))
print("Not_Transported: ", len(notTransported))


# In[ ]:


ppl_Transported = pd.DataFrame([ len(Transported), len(notTransported)], index=['Transported', 'Not_Transported'])
ppl_Transported.plot(kind = 'pie', subplots = True, figsize=(16,8), autopct = '%1.1f%%' )


# In[ ]:


# circle plot on destination place

train['Destination'].unique()


# In[ ]:


des = train['Destination'].value_counts()
des


# In[ ]:


des_pass = pd.DataFrame({
    'Destination' : des.index,
    'Total_Passenger' : des.values
},columns = ['Destination', 'Total_Passenger'])

des_pass


# In[ ]:


des_pass.set_index('Destination', inplace=True)


# In[ ]:


circle=plt.Circle( (0,0), 0.7, color='white')

plt.pie(des_pass['Total_Passenger'], labels= des_pass.index)
p=plt.gcf()
p.gca().add_artist(circle)
plt.title('Total passenger by destination place')


# In[ ]:


total_ppl


# In[ ]:


# Find transported and not_transported homeplanet wise

def transpeople(start,end):
    # set width of bar
    barWidth = 0.20
    
    #set height of bar
    bars1 = total_ppl['total Passenger'][start:end]
    bars2 = total_ppl['Transported'][start:end]
    bars3 = total_ppl['not_Transported'][start:end]
    
    
    # set position of bar on X axis
    r1 = np.arange(bars1.size)
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]
    
     # Make the plot
    plt.bar(r1, bars1, color='#36688D', width=barWidth, edgecolor='white', label='Total Passenger')
    plt.bar(r2, bars2, color='#F3CD05', width=barWidth, edgecolor='white', label='Transported')
    plt.bar(r3, bars3, color='#F49F05', width=barWidth, edgecolor='white', label='not_Transported')
    
    # Add xticks on the middle of the group bars
    plt.xticks([r + barWidth for r in range(len(bars1))], total_ppl.index[start:end])

    # Create legend & Show graphic
    plt.legend()


# In[ ]:


fig = plt.figure(figsize=(25,15))

plt.subplot(311)
transpeople(0,3)


# In[ ]:


#  Drop unnecessary columns

train.drop('PassengerId', axis=1, inplace=True)
train.drop('Name', axis=1, inplace=True)
# train.drop('Cabin', axis=1, inplace=True)

test.drop('PassengerId', axis=1, inplace=True)
test.drop('Name', axis=1, inplace=True)
# test.drop('Cabin', axis=1, inplace=True)


# In[ ]:


train


# In[ ]:


train[['cabinn', 'a', 'b']] = train['Cabin'].str.split('/', expand=True)

train.head()


# In[ ]:


test[['cabinn', 'a', 'b']] = train['Cabin'].str.split('/', expand=True)

test.head()


# Perform Label Encoding

# In[ ]:


from sklearn import preprocessing

label_encoder = preprocessing.LabelEncoder()
train['encoded_HomePlanet'] = label_encoder.fit_transform(train['HomePlanet'])
train['encoded_Cabinn'] = label_encoder.fit_transform(train['cabinn'])
train['encoded_Destination'] = label_encoder.fit_transform(train['Destination'])
train['encoded_Transported'] = label_encoder.fit_transform(train['Transported'])

train.head()


# In[ ]:


def encode_bool_train(x):
    if x == True:
        return 1
    else:
        return 0

train['CryoSleep'] = train['CryoSleep'].apply(encode_bool_train)
train['VIP'] = train['VIP'].apply(encode_bool_train)


print(encode_bool_train)


# In[ ]:


train.head()


# In[ ]:


label_encoder = preprocessing.LabelEncoder()
test['encoded_HomePlanet'] = label_encoder.fit_transform(test['HomePlanet'])
test['encoded_Cabinn'] = label_encoder.fit_transform(test['cabinn'])
test['encoded_Destination'] = label_encoder.fit_transform(test['Destination'])

test.head()


# In[ ]:


def encode_bool_test(x):
    if x == True:
        return 1
    else:
        return 0

test['CryoSleep'] = test['CryoSleep'].apply(encode_bool_test)
test['VIP'] = test['VIP'].apply(encode_bool_test)


print(encode_bool_test)


# In[ ]:


test.head()


# In[ ]:


# Drop unnecessary column

train.drop('HomePlanet', axis=1, inplace=True)
train.drop('Cabin', axis=1, inplace=True)
train.drop('cabinn', axis=1, inplace=True)
train.drop('Destination', axis=1, inplace=True)
train.drop('Transported', axis=1, inplace=True)
train.drop('a', axis=1, inplace=True)
train.drop('b', axis=1, inplace=True)


test.drop('HomePlanet', axis=1, inplace=True)
test.drop('Cabin', axis=1, inplace=True)
test.drop('cabinn', axis=1, inplace=True)
test.drop('Destination', axis=1, inplace=True)
test.drop('a', axis=1, inplace=True)
test.drop('b', axis=1, inplace=True)


# In[ ]:


# train.drop('CryoSleep', axis=1, inplace=True)
# train.drop('VIP', axis=1, inplace=True)

# test.drop('CryoSleep', axis=1, inplace=True)
# test.drop('VIP', axis=1, inplace=True)


# In[ ]:


train.head()


# In[ ]:


test.head()


# Perform Normalization [0-100]

# In[ ]:


from sklearn.preprocessing import MinMaxScaler

scaling = MinMaxScaler()


# In[ ]:


cols_to_norm = ['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']
train[cols_to_norm] = train[cols_to_norm].apply(lambda x: ((x - x.min()) / (x.max() - x.min())*100 ))

test[cols_to_norm] = test[cols_to_norm].apply(lambda x: ((x - x.min()) / (x.max() - x.min())*100 ))


# In[ ]:


train.head()


# In[ ]:


test.head()


# Feature Selection (correlation)

# In[ ]:


x = train.drop('encoded_Transported', axis=1)
y = train['encoded_Transported']


# In[ ]:


train.head()


# In[ ]:


test['encoded_Cabinn'].unique()


# In[ ]:


x.head()


# In[ ]:


# now, plot the data

plt.figure(figsize=(12,8))
ax = sns.heatmap(x.corr(), annot=True)
plt.show()


# In[ ]:


def correlation(dataset, threshold):
    col_corr = set()  # Set of all the names of correlated columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold: # we are interested in absolute coeff value
                colname = corr_matrix.columns[i]  # getting the name of column
                col_corr.add(colname)
    return col_corr


# In[ ]:


corr_features = correlation(x, 0.3)
len(set(corr_features))


# In[ ]:


corr_features


# In[ ]:


X_corr = x.drop(corr_features,axis=1)
X_corr


# In[ ]:


test_data = test.drop(corr_features, axis=1)
test_data


# **Splitting**

# In[ ]:


from sklearn import tree

from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(X_corr,y,test_size =0.3)

# print the data
x_train


# In[ ]:


print('x_train: ', x_train.shape)
print('x_test: ', x_test.shape)
print('y_train: ', y_train.shape)
print('y_test: ', y_test.shape)


# **Model selection and implimentation (Decision Tree)**

# In[ ]:


# clf = tree.DecisionTreeClassifier(criterion='entropy')
# clf.fit(x_train, y_train)


# In[ ]:


# clf = tree.DecisionTreeClassifier()
# clf.fit(x_train, y_train)


# In[ ]:


from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()

clf.fit(x_train,y_train)


# In[ ]:


# from sklearn.svm import SVC
# clf = SVC()

# clf.fit(x_train,y_train)


# In[ ]:


pred = clf.predict(x_test)


# In[ ]:


y_test


# In[ ]:


clf.score(x_test,y_test)


# **Confusion Metrix**

# In[ ]:


from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, pred)
cm


# In[ ]:


# now, visualize confusion metrix

ax = sns.heatmap(cm, annot=True, cmap='Blues')

ax.set_title('Confusion Matrix with labels\n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');


## Display the visualization of the Confusion Matrix.
plt.show()


# In[ ]:


fnl = clf.predict(test_data)
fnl


# In[ ]:


fnl = pd.DataFrame(fnl,columns=['Transported'])
fnl


# In[ ]:


def transformTo_bool(x):
    if x < 0.4:
        return False
    else:
        return True

fnl['Transported'] = fnl['Transported'].apply(transformTo_bool)

print(fnl)


# In[ ]:


fnl


# In[ ]:


test = pd.read_csv("/kaggle/input/spaceship-titanic/test.csv")
test.head()


# In[ ]:


fnl = pd.DataFrame(fnl,columns=['Transported'])
sub = pd.concat([test['PassengerId'],fnl['Transported']],axis=1)

sub.set_index('PassengerId',inplace=True)

sub.to_csv("submission.csv")

# submission.to_csv('submission.csv', header=True, index=False)


# In[ ]:


dff = pd.read_csv('submission.csv')
dff


# In[ ]:




