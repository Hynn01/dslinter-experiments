import pandas as pd
pd.options.mode.chained_assignment = None
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
file=pd.read_csv('/kaggle/input/covid19-corona-virus-india-dataset/patients_data.csv')
data=file[['p_id','date_announced','age_bracket','detected_state','status_change_date','current_status']].dropna().set_index('p_id')
data=data[pd.to_numeric(data['age_bracket'], errors='coerce').notnull()]
data['age_bracket']=pd.to_numeric(data['age_bracket'])
data['date_announced']= pd.to_datetime(data['date_announced'],format='%d/%m/%Y')
data['status_change_date']= pd.to_datetime(data['status_change_date'],format='%d/%m/%Y')
########preprossing the data and encoding string to int#################3
state=data['detected_state'].unique()
le=preprocessing.LabelEncoder()
le.fit(state)
le.classes_
data['detected_state']=le.transform(data['detected_state'])
###########
_data=data[data['current_status'].isin(['Recovered','Deceased'])]
_data['Days']=(_data['status_change_date']-_data['date_announced']).dt.days
_data=_data.drop(['status_change_date', 'date_announced'],axis=1)
x=_data.drop(['current_status'],1)
label=_data['current_status'].unique()
le.fit(label)
le.classes_
_data['current_status']=le.transform(_data['current_status'])
y=_data['current_status']
print(_data.info())
#####################
predict_data=data[data['current_status'].isin(['Hospitalized'])]
predict_data['Days']=(pd.to_datetime('now')-predict_data['date_announced']).dt.days
predict_data=predict_data.drop(['status_change_date', 'date_announced','current_status'],axis=1)
###############################
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
model=DecisionTreeClassifier(criterion='gini')
model.fit(x_train,y_train)
predict_train=model.predict(x_train)
##########################
predict_train=model.predict(x_test)
accuracy_test=accuracy_score(y_test, predict_train)
##########
print('Accuracy score on test set: ', accuracy_test)
output=model.predict(predict_data)
############### decoding i.e. converting int to string
output=list(le.inverse_transform(output))
le.fit(state)
predict_data['detected_state']=list(le.inverse_transform(predict_data['detected_state']))
predict_data['prediction']=output
print("prediction Output: ",predict_data)
predict_data.to_csv('prediction.csv')
cnt_recovered=0
cnt_deceased=0
for x in predict_data['prediction']:
    if x=='Recovered':
        cnt_recovered+=1
    else:
        cnt_deceased+=1
print("The no of Recovered patients: ", cnt_recovered)
print("The no of Deceased patients: ",cnt_deceased)