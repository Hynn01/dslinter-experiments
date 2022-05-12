#!/usr/bin/env python
# coding: utf-8

# * 作业说明：本次作业选择题目为titanic
# * 小组成员：陶仕林 林峰印 郝晨阳 戴尡鹏 
# * 本部分由 郝晨阳 完成
# * homework-titanic-ml版本对特征进行提升提高结果正确率，这份代码参考其他博客，通过构建随机森林模型，并结合小组同学的改进，进一步提高了结果正确率。最终分数为0.83732
# 

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd#数据分析
import numpy as np#科学计算
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

train=pd.read_csv(r"../input/titanic/train.csv")
test=pd.read_csv(r"../input/titanic/test.csv")
PassengerId=test['PassengerId']
all_data = pd.concat([train, test], ignore_index = True)


# 首先导入数据包与数据集，查看总体查看数据train如下

# In[ ]:


train.head()


# 我们看到，总共有12列，其中Survived字段表示的是该乘客是否获救，其余都是乘客的个人信息，包括：
# PassengerID（ID）
# Survived(存活与否)
# Pclass（客舱等级，较为重要）
# Name（姓名，可提取出更多信息）
# Sex（性别，较为重要）
# Age（年龄，较为重要）
# Parch（直系亲友）
# SibSp（旁系）
# Ticket（票编号）
# Fare（票价）
# Cabin（客舱编号）
# Embarked（上船的港口编号）

# In[ ]:


train.info()


# 通过上面数据分析发现有些列的特征是有空值的，比如Cabin只有204名乘客的是已知的。
# 

# In[ ]:


train.describe()


# 通过查看具体数据数值情况，得到数值型数据的一些分布。
# mean字段告诉我们，大概0.383838的人最后获救了，2/3等舱的人数比1等舱要多，平均乘客年龄大概是29.7岁等。

# 接下来利用统计学数据与绘图来进行数据的初步分析。

# In[ ]:


train['Survived'].value_counts()


# 可以看到891名乘客中只有342名乘客survived

# 接下来依次查看乘客的各属性与结果之间的关系

# In[ ]:


sns.barplot(x="Sex", y="Survived", data=train)


# sex影响：女性存活率远高于男性

# In[ ]:


sns.barplot(x="Pclass", y="Survived", data=train)


# Pclass影响：社会等级越高，存活率越高。

# In[ ]:


sns.barplot(x="SibSp", y="Survived", data=train)


# SibSp：配偶及兄弟姐妹数适中的乘客存活率越高

# In[ ]:


facet = sns.FacetGrid(train, hue="Survived",aspect=2)
facet.map(sns.kdeplot,'Age',shade= True)
facet.set(xlim=(0, train['Age'].max()))
facet.add_legend()
plt.xlabel('Age') 
plt.ylabel('density') 


# 从不同生还情况的密度图可以看出，在年龄15岁的左侧，生还率有明显差别，密度图非交叉区域面积非常大，但在其他年龄段，则差别不是很明显，认为是随机所致，因此可以考虑将此年龄偏小的区域分离出来。

# In[ ]:


sns.countplot('Embarked',hue='Survived',data=train)


# Embarked登港港口影响：C地的生存率更高,这个也应该保留为模型特征.

# 以上的简单分析可以看到：
# 获救的人不到半数
# 不同舱位/乘客等级获救概率不同
# 年龄对获救概率也有影响

# In[ ]:


all_data['Title'] = all_data['Name'].apply(lambda x:x.split(',')[1].split('.')[0].strip())
Title_Dict = {}
Title_Dict.update(dict.fromkeys(['Capt', 'Col', 'Major', 'Dr', 'Rev'], 'Officer'))
Title_Dict.update(dict.fromkeys(['Don', 'Sir', 'the Countess', 'Dona', 'Lady'], 'Royalty'))
Title_Dict.update(dict.fromkeys(['Mme', 'Ms', 'Mrs'], 'Mrs'))
Title_Dict.update(dict.fromkeys(['Mlle', 'Miss'], 'Miss'))
Title_Dict.update(dict.fromkeys(['Mr'], 'Mr'))
Title_Dict.update(dict.fromkeys(['Master','Jonkheer'], 'Master'))

all_data['Title'] = all_data['Title'].map(Title_Dict)
sns.barplot(x="Title", y="Survived", data=all_data)


# Title Feature(New)：不同称呼的乘客幸存率不同
# 新增Title特征，从姓名中提取乘客的称呼，归纳为六类

# In[ ]:


all_data['FamilySize']=all_data['SibSp']+all_data['Parch']+1
sns.barplot(x="FamilySize", y="Survived", data=all_data)


# FamilyLabel Feature(New)：家庭人数为2到4的乘客幸存率较高
# 新增FamilyLabel特征，先计算FamilySize=Parch+SibSp+1，然后把FamilySize分为三类。

# In[ ]:


def Fam_label(s):
    if (s >= 2) & (s <= 4):
        return 2
    elif ((s > 4) & (s <= 7)) | (s == 1):
        return 1
    elif (s > 7):
        return 0
all_data['FamilyLabel']=all_data['FamilySize'].apply(Fam_label)
sns.barplot(x="FamilyLabel", y="Survived", data=all_data)


# 按生存率把FamilySize分为三类，构成FamilyLabel特征。

# In[ ]:


all_data['Cabin'] = all_data['Cabin'].fillna('Unknown')
all_data['Deck']=all_data['Cabin'].str.get(0)
sns.barplot(x="Deck", y="Survived", data=all_data)


# Deck Feature(New)：不同甲板的乘客幸存率不同
# 新增Deck特征，先把Cabin空缺值填充为'Unknown'，再提取Cabin中的首字母构成乘客的甲板号

# In[ ]:


Ticket_Count = dict(all_data['Ticket'].value_counts())
all_data['TicketGroup'] = all_data['Ticket'].apply(lambda x:Ticket_Count[x])
sns.barplot(x='TicketGroup', y='Survived', data=all_data)


# TicketGroup Feature(New)：与2至4人共票号的乘客幸存率较高
# 新增TicketGroup特征，统计每个乘客的共票号数。

# In[ ]:


def Ticket_Label(s):
    if (s >= 2) & (s <= 4):
        return 2
    elif ((s > 4) & (s <= 8)) | (s == 1):
        return 1
    elif (s > 8):
        return 0

all_data['TicketGroup'] = all_data['TicketGroup'].apply(Ticket_Label)
sns.barplot(x='TicketGroup', y='Survived', data=all_data)


# 按生存率把TicketGroup分为三类。

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
age_df = all_data[['Age', 'Pclass','Sex','Title']]
age_df=pd.get_dummies(age_df)
known_age = age_df[age_df.Age.notnull()].values
unknown_age = age_df[age_df.Age.isnull()].values
y = known_age[:, 0]
X = known_age[:, 1:]
rfr = RandomForestRegressor(random_state=0, n_estimators=100, n_jobs=-1)
rfr.fit(X, y)
predictedAges = rfr.predict(unknown_age[:, 1::])
all_data.loc[ (all_data.Age.isnull()), 'Age' ] = predictedAges 


# Age Feature：Age缺失量为263，缺失量较大，用Sex, Title, Pclass三个特征构建随机森林模型，填充年龄缺失值。

# In[ ]:


all_data[all_data['Embarked'].isnull()]


# Embarked Feature：Embarked缺失量为2，缺失Embarked信息的乘客的Pclass均为1，且Fare均为80，因为Embarked为C且Pclass为1的乘客的Fare中位数为80，所以缺失值填充为C。

# In[ ]:


all_data.groupby(by=["Pclass","Embarked"]).Fare.median()


# In[ ]:


all_data['Embarked'] = all_data['Embarked'].fillna('C')


# In[ ]:


all_data[all_data['Fare'].isnull()]


# Fare Feature：Fare缺失量为1，缺失Fare信息的乘客的Embarked为S，Pclass为3，所以用Embarked为S，Pclass为3的乘客的Fare中位数填充。

# In[ ]:


fare=all_data[(all_data['Embarked'] == "S") & (all_data['Pclass'] == 3)].Fare.median()
all_data['Fare']=all_data['Fare'].fillna(fare)


# 2）同组识别：把姓氏相同的乘客划分为同一组，从人数大于一的组中分别提取出每组的妇女儿童和成年男性。

# In[ ]:


all_data['Surname']=all_data['Name'].apply(lambda x:x.split(',')[0].strip())
Surname_Count = dict(all_data['Surname'].value_counts())
all_data['FamilyGroup'] = all_data['Surname'].apply(lambda x:Surname_Count[x])
Female_Child_Group=all_data.loc[(all_data['FamilyGroup']>=2) & ((all_data['Age']<=12) | (all_data['Sex']=='female'))]
Male_Adult_Group=all_data.loc[(all_data['FamilyGroup']>=2) & (all_data['Age']>12) & (all_data['Sex']=='male')]


# In[ ]:


Female_Child=pd.DataFrame(Female_Child_Group.groupby('Surname')['Survived'].mean().value_counts())
Female_Child.columns=['GroupCount']
Female_Child


# 发现绝大部分女性和儿童组的平均存活率都为1或0，即同组的女性和儿童要么全部幸存，要么全部遇难。

# In[ ]:


sns.barplot(x=Female_Child.index, y=Female_Child["GroupCount"]).set_xlabel('AverageSurvived')


# In[ ]:


Male_Adult=pd.DataFrame(Male_Adult_Group.groupby('Surname')['Survived'].mean().value_counts())
Male_Adult.columns=['GroupCount']
Male_Adult


# 绝大部分成年男性组的平均存活率也为1或0。

# 因为普遍规律是女性和儿童幸存率高，成年男性幸存较低，所以我们把不符合普遍规律的反常组选出来单独处理。把女性和儿童组中幸存率为0的组设置为遇难组，把成年男性组中存活率为1的设置为幸存组，推测处于遇难组的女性和儿童幸存的可能性较低，处于幸存组的成年男性幸存的可能性较高。

# In[ ]:


Female_Child_Group=Female_Child_Group.groupby('Surname')['Survived'].mean()
Dead_List=set(Female_Child_Group[Female_Child_Group.apply(lambda x:x==0)].index)
print(Dead_List)
Male_Adult_List=Male_Adult_Group.groupby('Surname')['Survived'].mean()
Survived_List=set(Male_Adult_List[Male_Adult_List.apply(lambda x:x==1)].index)
print(Survived_List)


# 为了使处于这两种反常组中的样本能够被正确分类，对测试集中处于反常组中的样本的Age，Title，Sex进行惩罚修改。

# In[ ]:


train=all_data.loc[all_data['Survived'].notnull()]
test=all_data.loc[all_data['Survived'].isnull()]
test.loc[(test['Surname'].apply(lambda x:x in Dead_List)),'Sex'] = 'male'
test.loc[(test['Surname'].apply(lambda x:x in Dead_List)),'Age'] = 60
test.loc[(test['Surname'].apply(lambda x:x in Dead_List)),'Title'] = 'Mr'
test.loc[(test['Surname'].apply(lambda x:x in Survived_List)),'Sex'] = 'female'
test.loc[(test['Surname'].apply(lambda x:x in Survived_List)),'Age'] = 5
test.loc[(test['Surname'].apply(lambda x:x in Survived_List)),'Title'] = 'Miss'


# 3)特征转换：选取特征，转换为数值变量，划分训练集和测试集。

# In[ ]:


all_data=pd.concat([train, test])
all_data=all_data[['Survived','Pclass','Sex','Age','Fare','Embarked','Title','FamilyLabel','Deck','TicketGroup']]
all_data=pd.get_dummies(all_data)
train=all_data[all_data['Survived'].notnull()]
test=all_data[all_data['Survived'].isnull()].drop('Survived',axis=1)
X = train.values[:,1:]
y = train.values[:,0]


# 1）参数优化

# In[ ]:


from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest

pipe=Pipeline([('select',SelectKBest(k=20)), 
               ('classify', RandomForestClassifier(random_state = 10, max_features = 'sqrt'))])

param_test = {'classify__n_estimators':list(range(20,50,2)), 
              'classify__max_depth':list(range(3,60,3))}
gsearch = GridSearchCV(estimator = pipe, param_grid = param_test, scoring='roc_auc', cv=10)
gsearch.fit(X,y)
print(gsearch.best_params_, gsearch.best_score_)


# 2）训练模型

# In[ ]:


from sklearn.pipeline import make_pipeline
select = SelectKBest(k = 20)
clf = RandomForestClassifier(random_state = 10, warm_start = True, 
                                  n_estimators = 26,
                                  max_depth = 6, 
                                  max_features = 'sqrt')
pipeline = make_pipeline(select, clf)
pipeline.fit(X, y)


# 3）交叉验证

# In[ ]:


from sklearn.model_selection import cross_val_score
from sklearn import  metrics
cv_score = cross_val_score(pipeline, X, y, cv= 10)
print("CV Score : Mean - %.7g | Std - %.7g " % (np.mean(cv_score), np.std(cv_score)))


# 预测：

# In[ ]:


predictions = pipeline.predict(test)
submission = pd.DataFrame({"PassengerId": PassengerId, "Survived": predictions.astype(np.int32)})
submission.to_csv(r"submission1.csv", index=False)


# In[ ]:




