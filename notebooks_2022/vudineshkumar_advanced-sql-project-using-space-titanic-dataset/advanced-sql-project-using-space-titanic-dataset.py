#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # Hi Kaggle Family, 
# #### This NB is a very unique approach to dive deep on the new Space Titanic Competition dataset. Here we will be using only SQL to crunch on our data points and derive key inferences. Please feel free to drop a comments for any doubts/ clarifications related to this effort

# In[ ]:


train=pd.read_csv(r'/kaggle/input/spaceship-titanic/train.csv') ## read the pandas file
train.head() ## head allows us to pick the 1st 5 rows of a data frame


# In[ ]:


# import sqlalchemy and create a sqlite engine
from sqlalchemy import create_engine
engine = create_engine('sqlite://', echo=False)

# export the dataframe as a table 'playstore' to the sqlite engine
train.to_sql("spacetitanic", con =engine) ## salary dataset is the table name


# In[ ]:


## select all

sql='''

Select * from spacetitanic
limit 5


''';

    
df_sql = pd.read_sql_query(sql,con=engine)
df_sql.head()


# # Query 1 on Family And Groups
# 
# #### In this dive deep we will access on the number of groups (Similar 1st 4 passenger ID) and number of Family members (Folks with same last name) in each group

# In[ ]:


##  Approach to Query 1 

## In this use case we will derive the total number of groups present in the ship and the total number of family members in each group.
## this will help us to understand the survival rate of the folks per family.

##notes total number of family groups that are in the space titanic ship


#1. sort the data based on passenger_id

# 2.compare current row and row minus 1 and check on the gggg

# for example:- 
# 0001_01 and 0001_02 are from the same group, 
# 0002_02 and 0003_02 are not from the same group



# Result of this will be a column that will have a group number for the folks within the group

## to check if they are in the same family then, us last name
## if last name is same then, same family, else it is a different family









# In[ ]:


## Query 1.a code

## groups yes or no

sql='''



Select
passengerid,
Name,
SUBSTRING(PassengerId, 1, 4) as group_id,
case when SUBSTRING(PassengerId, 1, 4)=lead(SUBSTRING(PassengerId, 1, 4)) over (order by PassengerId asc) then 'yes' else 'no' end as groups
from spacetitanic
order by PassengerId asc

''';

    
df_sql = pd.read_sql_query(sql,con=engine)
df_sql.head(10)


# In[ ]:


## Query 1.b code

## family yes or no

# pos is the position at which space is happening between the 2 names

sql='''

select *,
substr(name, 1, pos-1) AS first_name,
substr(name, pos+1) AS last_name,
case when substr(name, pos+1)=lead(substr(name,pos+1)) over (order by substr(name,pos+1) desc) then 'yes' else 'no' end as family
from
(

select * from
(

Select
passengerid,
Name,
instr(name,' ') as pos,
SUBSTRING(PassengerId, 1, 4) as group_id,
case when SUBSTRING(PassengerId, 1, 4)=lead(SUBSTRING(PassengerId, 1, 4)) over (order by PassengerId asc) then 'yes' else 'no' end as groups
from spacetitanic
order by PassengerId asc

) groups

where groups ='yes'

) family

order by substr(name,pos+1) desc 
''';

    
df_sql = pd.read_sql_query(sql,con=engine)
df_sql.head(10)


# In[ ]:


## 1. c using 1.a and b let's do a count of total groups and total family.



sql='''

select * from
(
select * from
(
select
'Count of groups in the ship : -' as Description,count (distinct group_id) as value

from
(Select
passengerid,
Name,
SUBSTRING(PassengerId, 1, 4) as group_id,
case when SUBSTRING(PassengerId, 1, 4)=lead(SUBSTRING(PassengerId, 1, 4)) over (order by PassengerId asc) then 'yes' else 'no' end as groups
from spacetitanic
order by PassengerId asc
) groups

where groups = 'yes'

) group_final

union all

select * from 
(

select
'Count of families in the ship'as Description,count (distinct last_name) as value
from
(

select *,
substr(name, 1, pos-1) AS first_name,
substr(name, pos+1) AS last_name,
case when substr(name, pos+1)=lead(substr(name,pos+1)) over (order by substr(name,pos+1) desc) then 'yes' else 'no' end as family
from
(

select * from
(

Select
passengerid,
Name,
instr(name,' ') as pos,
SUBSTRING(PassengerId, 1, 4) as group_id,
case when SUBSTRING(PassengerId, 1, 4)=lead(SUBSTRING(PassengerId, 1, 4)) over (order by PassengerId asc) then 'yes' else 'no' end as groups
from spacetitanic
order by PassengerId asc

) groups

where groups ='yes'

) family

order by substr(name,pos+1) desc 
) family_final



) family_final2

) view_1

union all

select * from

(
select 
'Count of Single persons in the ship' as description ,value -lead(value) over (order by value desc) as value

from
(
select * from
(
select
'Count of groups in the ship : -' as Description,count (distinct group_id) as value

from
(Select
passengerid,
Name,
SUBSTRING(PassengerId, 1, 4) as group_id,
case when SUBSTRING(PassengerId, 1, 4)=lead(SUBSTRING(PassengerId, 1, 4)) over (order by PassengerId asc) then 'yes' else 'no' end as groups
from spacetitanic
order by PassengerId asc
) groups

where groups = 'yes'

) group_final

union all

select * from 
(

select
'Count of families in the ship'as Description,count (distinct last_name) as value
from
(

select *,
substr(name, 1, pos-1) AS first_name,
substr(name, pos+1) AS last_name,
case when substr(name, pos+1)=lead(substr(name,pos+1)) over (order by substr(name,pos+1) desc) then 'yes' else 'no' end as family
from
(

select * from
(

Select
passengerid,
Name,
instr(name,' ') as pos,
SUBSTRING(PassengerId, 1, 4) as group_id,
case when SUBSTRING(PassengerId, 1, 4)=lead(SUBSTRING(PassengerId, 1, 4)) over (order by PassengerId asc) then 'yes' else 'no' end as groups
from spacetitanic
order by PassengerId asc

) groups

where groups ='yes'

) family

order by substr(name,pos+1) desc 
) family_final



) family_final2

) view
 
 limit 1
 
) view_2

 
''';

    
df_sql = pd.read_sql_query(sql,con=engine)
df_sql.head(10)


# #  Inference for Family and Groups Query 1
# 
# #### There are less single group travellers in the ship as compared to multi group travellers, most of the multi group travellers are familes

# # Query 2 on Who is a VIP?
# 
# #### In this we will access on multipole parameters and identify if the traveller is a VIP or not, we already have a column made available for VIP with a YES/No classification. But the goal of this analyis is to study on why these travellers are classified as VIPs and use this analysis to fill the blank values of the VIP column

# In[ ]:


### Approach to Query 2

## 2.a  Create an expense column and do a sum of all the amount each traveller has spent during the journey
## 2.b  based on expenses in 2.a we will identify the avg expenses of all the travellers in 2.b
## 2.c based on avg expenses derived from 2.b we can create a column to understand if the traveller has spent more than the average expenses or not
## 2.d based on the results derived in 2c, we can add a filter for VIP and get an understanding of avg expenditure of VIPs 
## 2.f we will attempt to fill the blank values for vip classification and understand if they are in the vip bucket or not
## 2.f.1 we will fill the blank values from the filter list in 2.f
## 2.f.2 based on the list we cleand in 2f.1, we will remove the vip column which has null values and keep the colum we need vip 2 rename vip
## 2.f.3 based on the clean values of 2f2, we will create a new table based on 2 views
# 1st view, we will filter out all the blank values from the original table and store it as table a
#  2nd view we will use the table dervied from 2f2 and store it as table b
# we will do a union for table 1 and table 2 , this will have all list of vips available

## logic is we will remove the blank values in the original table, and add our cleaned version to it.
## 2g. based on the final table derived from 2f3, we will do a count of vips who spent less than the overall avg and didnt get transported
## 2h. We will create a view, based on the output vip_master_table  we prepared in 2fd


# In[ ]:


## 2.a  Create an expense column and do a sum of all the amount each traveller has spent during the journey

sql='''

Select *,
(RoomService + FoodCourt +ShoppingMall + Spa+ VRDeck) as expenses

from spacetitanic
limit 5


''';

    
df_sql = pd.read_sql_query(sql,con=engine)
df_sql.head()


# In[ ]:


## 2.b  based on expenses in 2.a we will identify the avg expenses of all the travellers in 2.b
sql='''

select 

'The average expenses of all the travellers in the ship is :- ' as Description,
round(avg(all_travellers.expenses),2) as Avg
from

(
Select *,
(RoomService + FoodCourt +ShoppingMall + Spa+ VRDeck) as expenses

from spacetitanic


) all_travellers

union all

select 

'The average expenses of VIP travellers in the ship is :- ' as Description,
round(avg(vip_travellers.expenses),2) as Avg
from

(
Select *,
(RoomService + FoodCourt +ShoppingMall + Spa+ VRDeck) as expenses
from spacetitanic
where vip= 1


) vip_travellers

union all

select 

'The average expenses of non VIP travellers in the ship is :- ' as Description,
round(avg(non_vip_travellers.expenses),2) as Avg
from

(
Select *,
(RoomService + FoodCourt +ShoppingMall + Spa+ VRDeck) as expenses
from spacetitanic
where vip= 0


) non_vip_travellers





''';

    
df_sql = pd.read_sql_query(sql,con=engine)
df_sql.head()


# In[ ]:


## 2.c based on avg expenses derived from 2.b we can create a column to understand if the traveller has spent more than the average expenses or not


sql='''

Select *,
(RoomService + FoodCourt +ShoppingMall + Spa+ VRDeck) as expenses,
case 
when RoomService + FoodCourt +ShoppingMall + Spa+ VRDeck > 1484.60 
then 'Spent more than avg' else 'Spent less than avg' end as expenditure_rate


from spacetitanic


''';

    
df_sql = pd.read_sql_query(sql,con=engine)
df_sql.head()


# In[ ]:


## 2.d based on the results derived in 2c, we can add a filter for VIP and get an understanding of avg expenditure of VIPs 

sql='''

select

sum(case when a.expenditure_rate='Vip who Spent more than avg' then 1 else 0 end) as VIP_counts_Spent_more_than_avg,
sum(case when a.expenditure_rate='Vip who Spent less than avg' then 1 else 0 end) as VIP_counts_Spent_less_than_avg
from
(
Select *,
(RoomService + FoodCourt +ShoppingMall + Spa+ VRDeck) as expenses,
case 
when RoomService + FoodCourt +ShoppingMall + Spa+ VRDeck > 4599.74 
then 'Vip who Spent more than avg' else 'Vip who Spent less than avg' end as expenditure_rate


from spacetitanic
where vip = 1

) a
''';

    
df_sql = pd.read_sql_query(sql,con=engine)
df_sql.head()


# In[ ]:


## 2.f we will attempt to fill the blank values for vip classification and understand if they are in the vip bucket or not

sql='''


select 
'Count of Vip null values before filling' as Description,
count(*) as value
from
(

Select *,
(RoomService + FoodCourt +ShoppingMall + Spa+ VRDeck) as expenses
from spacetitanic
where vip IS NULL
) a

''';

    
df_sql = pd.read_sql_query(sql,con=engine)
df_sql.head()


# In[ ]:


## 2.f.1 we will fill the blank values from the filter list in 2.f
sql='''



Select *,
(RoomService + FoodCourt +ShoppingMall + Spa+ VRDeck) as expenses,
case when vip IS NULL and RoomService + FoodCourt +ShoppingMall + Spa+ VRDeck > 4599.74 then 1 else 0 end as VIP_2
from spacetitanic
where vip IS NULL

''';

    
df_sql = pd.read_sql_query(sql,con=engine)
df_sql.head()


# In[ ]:


## 2.f.2 based on the list we cleand in 2f.1, we will remove the vip column which has null values and keep the colum we need vip 2 rename vip
sql='''

Select
a.PassengerId,
a.HomePlanet,
a.CryoSleep,
a.cabin,
a.Destination,
a.Age,
a.VIP_2 as VIP,
a.RoomService,
a.FoodCourt,
a.ShoppingMall,
a.Spa,
a.VRDeck,
a.Name,
a.Transported,
a.expenses
from
(
select *,
(RoomService + FoodCourt +ShoppingMall + Spa+ VRDeck) as expenses,
case when vip IS NULL and RoomService + FoodCourt +ShoppingMall + Spa+ VRDeck > 4599.74 then 1 else 0 end as VIP_2
from spacetitanic
where vip IS NULL

) a


''';

    
df_sql = pd.read_sql_query(sql,con=engine)
df_sql.head()


# In[ ]:


## 2.f.3 based on the clean values of 2f2, we will create a new table based on 2 views
# 1st view, we will filter out all the blank values from the original table and store it as table a
#  2nd view we will use the table dervied from 2f2 and store it as table b
# we will do a union for table 1 and table 2 , this will have all list of vips available

## logic is we will remove the blank values in the original table, and add our cleaned version to it.

sql='''

select 
*
from
(
select * from
(Select
PassengerId,
HomePlanet,
CryoSleep,
Cabin,
Destination,
Age,
VIP,
RoomService,
FoodCourt,
ShoppingMall,
Spa,
VRDeck,
Name,
Transported,
(RoomService + FoodCourt +ShoppingMall + Spa+ VRDeck) as expenses
from spacetitanic
where vip is not null
) a


union all

select * from 

(
Select
fill_vip.PassengerId,
fill_vip.HomePlanet,
fill_vip.CryoSleep,
fill_vip.cabin,
fill_vip.Destination,
fill_vip.Age,
fill_vip.VIP_2 as VIP,
fill_vip.RoomService,
fill_vip.FoodCourt,
fill_vip.ShoppingMall,
fill_vip.Spa,
fill_vip.VRDeck,
fill_vip.Name,
fill_vip.Transported,
fill_vip.expenses
from
(
select *,
(RoomService + FoodCourt +ShoppingMall + Spa+ VRDeck) as expenses,
case when vip IS NULL and RoomService + FoodCourt +ShoppingMall + Spa+ VRDeck > 4599.74 then 1 else 0 end as VIP_2
from spacetitanic
where vip IS NULL

) fill_vip

) b

) vip_master_table

''';

    
df_sql = pd.read_sql_query(sql,con=engine)
df_sql.head()



# In[ ]:


## 2g. based on the final table derived from 2f3, we will do a count of vips who spent less than the overall avg and didnt get transported

sql='''

select
*

from
(
select * from
(Select
PassengerId,
HomePlanet,
CryoSleep,
Cabin,
Destination,
Age,
VIP,
RoomService,
FoodCourt,
ShoppingMall,
Spa,
VRDeck,
Name,
Transported,
(RoomService + FoodCourt +ShoppingMall + Spa+ VRDeck) as expenses
from spacetitanic
where vip is not null
) a


union all

select * from 

(
Select
fill_vip.PassengerId,
fill_vip.HomePlanet,
fill_vip.CryoSleep,
fill_vip.cabin,
fill_vip.Destination,
fill_vip.Age,
fill_vip.VIP_2 as VIP,
fill_vip.RoomService,
fill_vip.FoodCourt,
fill_vip.ShoppingMall,
fill_vip.Spa,
fill_vip.VRDeck,
fill_vip.Name,
fill_vip.Transported,
fill_vip.expenses
from
(
select *,
(RoomService + FoodCourt +ShoppingMall + Spa+ VRDeck) as expenses,
case when vip IS NULL and RoomService + FoodCourt +ShoppingMall + Spa+ VRDeck > 4599.74 then 1 else 0 end as VIP_2
from spacetitanic
where vip IS NULL

) fill_vip

) b

) vip_master_table
''';

    
df_sql = pd.read_sql_query(sql,con=engine)
df_sql.head()



# In[ ]:


## 2h. We will create a view, based on the output vip_master_table  we prepared in 2fd

sql='''

select * from
(
select
'Total counts of travellers' as Description,
count(*) as Value

from
(
select * from
(Select
PassengerId,
HomePlanet,
CryoSleep,
Cabin,
Destination,
Age,
VIP,
RoomService,
FoodCourt,
ShoppingMall,
Spa,
VRDeck,
Name,
Transported,
(RoomService + FoodCourt +ShoppingMall + Spa+ VRDeck) as expenses
from spacetitanic
where vip is not null
) a


union all

select * from 

(
Select
fill_vip.PassengerId,
fill_vip.HomePlanet,
fill_vip.CryoSleep,
fill_vip.cabin,
fill_vip.Destination,
fill_vip.Age,
fill_vip.VIP_2 as VIP,
fill_vip.RoomService,
fill_vip.FoodCourt,
fill_vip.ShoppingMall,
fill_vip.Spa,
fill_vip.VRDeck,
fill_vip.Name,
fill_vip.Transported,
fill_vip.expenses
from
(
select *,
(RoomService + FoodCourt +ShoppingMall + Spa+ VRDeck) as expenses,
case when vip IS NULL and RoomService + FoodCourt +ShoppingMall + Spa+ VRDeck > 4599.74 then 1 else 0 end as VIP_2
from spacetitanic
where vip IS NULL

) fill_vip

) b

) vip_master_table

) total_count_of_travellers_view



union all

select * from
(
select
'Total counts of VIP travellers' as Description,
count(*) as Value

from
(
select * from
(Select
PassengerId,
HomePlanet,
CryoSleep,
Cabin,
Destination,
Age,
VIP,
RoomService,
FoodCourt,
ShoppingMall,
Spa,
VRDeck,
Name,
Transported,
(RoomService + FoodCourt +ShoppingMall + Spa+ VRDeck) as expenses
from spacetitanic
where vip is not null
) a


union all

select * from 

(
Select
fill_vip.PassengerId,
fill_vip.HomePlanet,
fill_vip.CryoSleep,
fill_vip.cabin,
fill_vip.Destination,
fill_vip.Age,
fill_vip.VIP_2 as VIP,
fill_vip.RoomService,
fill_vip.FoodCourt,
fill_vip.ShoppingMall,
fill_vip.Spa,
fill_vip.VRDeck,
fill_vip.Name,
fill_vip.Transported,
fill_vip.expenses
from
(
select *,
(RoomService + FoodCourt +ShoppingMall + Spa+ VRDeck) as expenses,
case when vip IS NULL and RoomService + FoodCourt +ShoppingMall + Spa+ VRDeck > 4599.74 then 1 else 0 end as VIP_2
from spacetitanic
where vip IS NULL

) fill_vip

) b

) vip_master_table


where vip_master_table.VIP=1
) total_count_of_vip_travellers_view


union all

select
'Average Expenditure of all the travellers' as Description
,round(avg(expense_view.expenses),2) as Value

from
(
select
*
from
(
select * from
(Select
PassengerId,
HomePlanet,
CryoSleep,
Cabin,
Destination,
Age,
VIP,
RoomService,
FoodCourt,
ShoppingMall,
Spa,
VRDeck,
Name,
Transported,
(RoomService + FoodCourt +ShoppingMall + Spa+ VRDeck) as expenses
from spacetitanic
where vip is not null
) a


union all

select * from 

(
Select
fill_vip.PassengerId,
fill_vip.HomePlanet,
fill_vip.CryoSleep,
fill_vip.cabin,
fill_vip.Destination,
fill_vip.Age,
fill_vip.VIP_2 as VIP,
fill_vip.RoomService,
fill_vip.FoodCourt,
fill_vip.ShoppingMall,
fill_vip.Spa,
fill_vip.VRDeck,
fill_vip.Name,
fill_vip.Transported,
fill_vip.expenses
from
(
select *,
(RoomService + FoodCourt +ShoppingMall + Spa+ VRDeck) as expenses,
case when vip IS NULL and RoomService + FoodCourt +ShoppingMall + Spa+ VRDeck > 4599.74 then 1 else 0 end as VIP_2
from spacetitanic
where vip IS NULL

) fill_vip

) b

) vip_master_table

) expense_view


union all

select
'VIPs who spent less than avg and got transported' as Description,
count(*) as Value

from
(
select
*
from
(
select * from
(Select
PassengerId,
HomePlanet,
CryoSleep,
Cabin,
Destination,
Age,
VIP,
RoomService,
FoodCourt,
ShoppingMall,
Spa,
VRDeck,
Name,
Transported,
(RoomService + FoodCourt +ShoppingMall + Spa+ VRDeck) as expenses
from spacetitanic
where vip is not null
) a


union all

select * from 

(
Select
fill_vip.PassengerId,
fill_vip.HomePlanet,
fill_vip.CryoSleep,
fill_vip.cabin,
fill_vip.Destination,
fill_vip.Age,
fill_vip.VIP_2 as VIP,
fill_vip.RoomService,
fill_vip.FoodCourt,
fill_vip.ShoppingMall,
fill_vip.Spa,
fill_vip.VRDeck,
fill_vip.Name,
fill_vip.Transported,
fill_vip.expenses
from
(
select *,
(RoomService + FoodCourt +ShoppingMall + Spa+ VRDeck) as expenses,
case when vip IS NULL and RoomService + FoodCourt +ShoppingMall + Spa+ VRDeck > 4599.74 then 1 else 0 end as VIP_2
from spacetitanic
where vip IS NULL

) fill_vip

) b

) vip_master_table

where vip_master_table.vip=1 and vip_master_table.expenses <=1484.6
) VIP_lessthanavg_transported



union all


select
'Count of VIPs who got transported' as Description,
sum(case when vip_transported.vip=1 and vip_transported.Transported=1 then 1 else 0 end) as Value

from
(
select
*
from
(
select * from
(Select
PassengerId,
HomePlanet,
CryoSleep,
Cabin,
Destination,
Age,
VIP,
RoomService,
FoodCourt,
ShoppingMall,
Spa,
VRDeck,
Name,
Transported,
(RoomService + FoodCourt +ShoppingMall + Spa+ VRDeck) as expenses
from spacetitanic
where vip is not null
) a


union all

select * from 

(
Select
fill_vip.PassengerId,
fill_vip.HomePlanet,
fill_vip.CryoSleep,
fill_vip.cabin,
fill_vip.Destination,
fill_vip.Age,
fill_vip.VIP_2 as VIP,
fill_vip.RoomService,
fill_vip.FoodCourt,
fill_vip.ShoppingMall,
fill_vip.Spa,
fill_vip.VRDeck,
fill_vip.Name,
fill_vip.Transported,
fill_vip.expenses
from
(
select *,
(RoomService + FoodCourt +ShoppingMall + Spa+ VRDeck) as expenses,
case when vip IS NULL and RoomService + FoodCourt +ShoppingMall + Spa+ VRDeck > 4599.74 then 1 else 0 end as VIP_2
from spacetitanic
where vip IS NULL

) fill_vip

) b

) vip_master_table

) VIP_transported

union all

select 
'VIP transported (spent less than avg)/VIPs total' as Description,
cast(100*round(round(55,4)/round(80,2),4) as varchar)||'%' as Value



''';

    
df_sql = pd.read_sql_query(sql,con=engine)
df_sql.head(6)



# # Inference for VIP query 2
# 
# #### We can infer that over 68.75% VIPs who spent less than the average expenses of all travellers(Included VIPs and non-VIPs in the ship got transported

# # Query 3 Are CryoSleepers Impacted big time?
# 
# ####  In this we will check if the hibernators have any impact in being transported or not

# In[ ]:


## Approach to Query 3

## in this we will take the total hibernators and check how many got transported or not


## 3.a Fill the blank values of cryo sleeprs, the logic we use is if total expenses is 0, then the traveller is a cryo sleeper

## 3b. using 3.a, we will create a table that will gives the total list of travellers who are cryo sleeping

## 3c using 3b, we will create a table that will give us total list of hibernators who got transported

## 3.d We will divide 3a. and 3b. to get a percentage understanding hibernators who got transported vs total hibernators


# In[ ]:


## 3.a Fill the blank values of cryo sleeprs, the logic we use is if total expenses is 0, then the traveller is a cryo sleeper

sql='''

select 
cryosleep_clean.PassengerId,
cryosleep_clean.HomePlanet,
cryosleep_clean.Cabin,
cryosleep_clean.Destination,
cryosleep_clean.Age,
cryosleep_clean.CryoSleep2 as CryoSleep,
cryosleep_clean.RoomService,
cryosleep_clean.FoodCourt,
cryosleep_clean.ShoppingMall,
cryosleep_clean.Spa,
cryosleep_clean.VRDeck,
cryosleep_clean.Name,
cryosleep_clean.Transported,
cryosleep_clean.expenses
from

(

select
*,
case when cryosleep_null.expenses=0 then 1 else 0 end as CryoSleep2

from
(
Select *,
(RoomService + FoodCourt +ShoppingMall + Spa+ VRDeck) as expenses
from spacetitanic
) cryosleep_null
where cryosleep_null.cryosleep is null

) cryosleep_clean

union all

select * from

(

select
a.PassengerId,
a.HomePlanet,
a.Cabin,
a.Destination,
a.Age,
a.CryoSleep,
a.RoomService,
a.FoodCourt,
a.ShoppingMall,
a.Spa,
a.VRDeck,
a.Name,
a.Transported,
a.expenses
from
(
Select *,
(RoomService + FoodCourt +ShoppingMall + Spa+ VRDeck) as expenses

from spacetitanic
where CryoSleep is not null

) a




)cryosleep_withnull


''';

    
df_sql = pd.read_sql_query(sql,con=engine)
df_sql.head()


# In[ ]:


## 3b. using 3.a, we will create a table that will gives the total list of travellers who are cryo sleeping

sql='''

select 
'Total Cryo Sleeprs' as Description,
count(*) as Value
from
(
select 
cryosleep_clean.PassengerId,
cryosleep_clean.HomePlanet,
cryosleep_clean.Cabin,
cryosleep_clean.Destination,
cryosleep_clean.Age,
cryosleep_clean.CryoSleep2 as CryoSleep,
cryosleep_clean.RoomService,
cryosleep_clean.FoodCourt,
cryosleep_clean.ShoppingMall,
cryosleep_clean.Spa,
cryosleep_clean.VRDeck,
cryosleep_clean.Name,
cryosleep_clean.Transported,
cryosleep_clean.expenses
from

(

select
*,
case when cryosleep_null.expenses=0 then 1 else 0 end as CryoSleep2

from
(
Select *,
(RoomService + FoodCourt +ShoppingMall + Spa+ VRDeck) as expenses
from spacetitanic
) cryosleep_null
where cryosleep_null.cryosleep is null

) cryosleep_clean

union all

select * from

(

select
a.PassengerId,
a.HomePlanet,
a.Cabin,
a.Destination,
a.Age,
a.CryoSleep,
a.RoomService,
a.FoodCourt,
a.ShoppingMall,
a.Spa,
a.VRDeck,
a.Name,
a.Transported,
a.expenses
from
(
Select *,
(RoomService + FoodCourt +ShoppingMall + Spa+ VRDeck) as expenses

from spacetitanic
where CryoSleep is not null

) a




)cryosleep_withnull

) total_cryosleepers

where total_cryosleepers.CryoSleep=1

''';

    
df_sql = pd.read_sql_query(sql,con=engine)
df_sql.head()



# In[ ]:


## 3c using 3b, we will create a table that will give us total list of hibernators who got transported
sql='''

select 
'Total Cryo Sleeprs who got transported' as Description,
count(*) as Value
from
(
select 
cryosleep_clean.PassengerId,
cryosleep_clean.HomePlanet,
cryosleep_clean.Cabin,
cryosleep_clean.Destination,
cryosleep_clean.Age,
cryosleep_clean.CryoSleep2 as CryoSleep,
cryosleep_clean.RoomService,
cryosleep_clean.FoodCourt,
cryosleep_clean.ShoppingMall,
cryosleep_clean.Spa,
cryosleep_clean.VRDeck,
cryosleep_clean.Name,
cryosleep_clean.Transported,
cryosleep_clean.expenses
from

(

select
*,
case when cryosleep_null.expenses=0 then 1 else 0 end as CryoSleep2

from
(
Select *,
(RoomService + FoodCourt +ShoppingMall + Spa+ VRDeck) as expenses
from spacetitanic
) cryosleep_null
where cryosleep_null.cryosleep is null

) cryosleep_clean

union all

select * from

(

select
a.PassengerId,
a.HomePlanet,
a.Cabin,
a.Destination,
a.Age,
a.CryoSleep,
a.RoomService,
a.FoodCourt,
a.ShoppingMall,
a.Spa,
a.VRDeck,
a.Name,
a.Transported,
a.expenses
from
(
Select *,
(RoomService + FoodCourt +ShoppingMall + Spa+ VRDeck) as expenses

from spacetitanic
where CryoSleep is not null

) a




)cryosleep_withnull

) total_cryosleepers

where total_cryosleepers.CryoSleep=1 and total_cryosleepers.Transported=1

''';

    
df_sql = pd.read_sql_query(sql,con=engine)
df_sql.head()



# In[ ]:


## 3.d We will divide 3a. and 3b. to get a percentage understanding hibernators who got transported vs total hibernators


sql='''


select *
from
(

select 
'Total Cryo Sleeprs' as Description,
count(*) as Value
from
(
select 
cryosleep_clean.PassengerId,
cryosleep_clean.HomePlanet,
cryosleep_clean.Cabin,
cryosleep_clean.Destination,
cryosleep_clean.Age,
cryosleep_clean.CryoSleep2 as CryoSleep,
cryosleep_clean.RoomService,
cryosleep_clean.FoodCourt,
cryosleep_clean.ShoppingMall,
cryosleep_clean.Spa,
cryosleep_clean.VRDeck,
cryosleep_clean.Name,
cryosleep_clean.Transported,
cryosleep_clean.expenses
from

(

select
*,
case when cryosleep_null.expenses=0 then 1 else 0 end as CryoSleep2

from
(
Select *,
(RoomService + FoodCourt +ShoppingMall + Spa+ VRDeck) as expenses
from spacetitanic
) cryosleep_null
where cryosleep_null.cryosleep is null

) cryosleep_clean

union all

select * from

(

select
a.PassengerId,
a.HomePlanet,
a.Cabin,
a.Destination,
a.Age,
a.CryoSleep,
a.RoomService,
a.FoodCourt,
a.ShoppingMall,
a.Spa,
a.VRDeck,
a.Name,
a.Transported,
a.expenses
from
(
Select *,
(RoomService + FoodCourt +ShoppingMall + Spa+ VRDeck) as expenses

from spacetitanic
where CryoSleep is not null

) a




)cryosleep_withnull

) total_cryosleepers

where total_cryosleepers.CryoSleep=1

) total_cryosleepersfinal

union all

select  * from

(
select 
'Total Cryo Sleeprs who got transported' as Description,
count(*) as Value
from
(
select 
cryosleep_clean.PassengerId,
cryosleep_clean.HomePlanet,
cryosleep_clean.Cabin,
cryosleep_clean.Destination,
cryosleep_clean.Age,
cryosleep_clean.CryoSleep2 as CryoSleep,
cryosleep_clean.RoomService,
cryosleep_clean.FoodCourt,
cryosleep_clean.ShoppingMall,
cryosleep_clean.Spa,
cryosleep_clean.VRDeck,
cryosleep_clean.Name,
cryosleep_clean.Transported,
cryosleep_clean.expenses
from

(

select
*,
case when cryosleep_null.expenses=0 then 1 else 0 end as CryoSleep2

from
(
Select *,
(RoomService + FoodCourt +ShoppingMall + Spa+ VRDeck) as expenses
from spacetitanic
) cryosleep_null
where cryosleep_null.cryosleep is null

) cryosleep_clean

union all

select * from

(

select
a.PassengerId,
a.HomePlanet,
a.Cabin,
a.Destination,
a.Age,
a.CryoSleep,
a.RoomService,
a.FoodCourt,
a.ShoppingMall,
a.Spa,
a.VRDeck,
a.Name,
a.Transported,
a.expenses
from
(
Select *,
(RoomService + FoodCourt +ShoppingMall + Spa+ VRDeck) as expenses

from spacetitanic
where CryoSleep is not null

) a




)cryosleep_withnull

) total_cryosleepers

where total_cryosleepers.CryoSleep=1 and total_cryosleepers.Transported=1



) totaltravellersfinal

union all

select 
'Percentage contribution' as description,
cast(100*round(round(2547,4)/round(3124,4),4) as varchar)||'%' as Value


''';

    
df_sql = pd.read_sql_query(sql,con=engine)
df_sql.head()



# # Inference for Cryo Sleepers Query 3
# 
# #### We can clearly say that 81.53 % sleepers got transported, owing to system troubles.

# # Query 4:- Does Age play a part here?
# 
# #### As per the original titanic films, younger travellers were preferred to be saved compared to the olders, we will dive deep in this section to identify similar trends we will check on number of travellers which are above average age who got transported and the ones below avg age who got transported.

# In[ ]:


## Query 4 approach

# 4.a is find the average age of travellers
# 4.b create a table that has all the travellers above age
# 4.c create a table that has all the travellers below age
## 4.d we will a create based on 4.b and 4.c which has 'traveller above avg age who got transported' and 'travellers below avg age who got transported'


# In[ ]:


## 4.a is find the average age of travellers
sql='''

Select 
'The average age of all travellers' as Description,
round(avg(age),2) as value from spacetitanic


''';

    
df_sql = pd.read_sql_query(sql,con=engine)
df_sql.head()


# In[ ]:


# 4.b create a table that has all the travellers above age and got transported

sql='''

Select 
'count of Travellers above average age and transported' as Description,
count(*) as value from spacetitanic
where age > 28.83 and Transported=1


''';

    
df_sql = pd.read_sql_query(sql,con=engine)
df_sql.head()


# In[ ]:


# 4.c create a table that has all the travellers below or equal age

sql='''

Select 
'count of Travellers below age and transported' as Description,
count(*) as value from spacetitanic
where age <= 28.83 and Transported=1


''';

    
df_sql = pd.read_sql_query(sql,con=engine)
df_sql.head()


# In[ ]:


## 4.d we will a create based on 4.b and 4.c which has 'traveller above avg age who got transported' and 'travellers below avg age who got transported'



sql='''

select * from 
(

Select 
'The average age of all travellers' as Description,
round(avg(age),2) as value from spacetitanic


) avg_age

union all

select * from 
(

Select 
'count of Travellers above average age and transported' as Description,
count(*) as value from spacetitanic
where age > 28.83 and Transported=1


) travellers_abv_avg_transported

union all

select * from 
(

Select 
'count of Travellers below age and transported' as Description,
count(*) as value from spacetitanic
where age <= 28.83 and Transported=1


) travellers_below_avg_transported

union all

select
'Percentage of travellers above avg who got transported' as Description,
cast(100*round(round(1833,4)/round(2455 + 1833,4),4) as varchar)||'%' as Value
''';

    
df_sql = pd.read_sql_query(sql,con=engine)
df_sql.head()


# #  Inference for Age Query 4
# 
# #### Over 42.75% of folks aboves 28 years , which is the average age of the crowd, got transported.
