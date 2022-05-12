#!/usr/bin/env python
# coding: utf-8

# # · Darwin's Finches Evolution Dataset Finches Evolution EDA and statistical study

# # imports

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import seaborn as sns
import plotly.express as px


# In[ ]:


df_2012 = pd.read_csv('../input/darwins-finches-evolution-dataset/finch_beaks_2012.csv')
df_2012_scandens = df_2012[df_2012['species'] == 'scandens'].copy()
df_2012_scandens['year'] = 2012
df_2012_scandens


# In[ ]:


df_1975 = pd.read_csv('../input/darwins-finches-evolution-dataset/finch_beaks_1975.csv')
df_1975_scandens = df_1975[df_1975['species'] == 'scandens'].copy() # to disable a warning that says (you are overriting only on a slice of the data)
df_1975_scandens.rename(columns={'Beak depth, mm' : 'bdepth','Beak length, mm' : 'blength'},inplace=True)
df_1975_scandens['year'] = 1975
df_1975_scandens


# In[ ]:


df = df_1975_scandens.append(df_2012_scandens)
df


# In[ ]:


df.describe()


# In[ ]:


df.info()


# # task 1 (EDA)

# In[ ]:


_ = sns.swarmplot(data= df , x= 'year' , y= 'bdepth')
_ = plt.xlabel('year')
_ = plt.ylabel('bdepth mm')


# ## ECDF of the two years

# In[ ]:


def ecdf(data):
    """ 
    this is a fucntion to draw the ecdf of some data
    this function takes numpy array as input and returns
    two arrays the first is the values of the x-axis and the second is the values on the y-axis
    """
    ## first sort the data and give each point a probability of 1/len(data) 
    ## if the point is repeated the probability of the point will increase (vertical points )
    ## ecdf is the sum of all the previous probabilities
    
    sorted_data = np.sort(data)
    y = np.arange(1,len(data)+1) / len(data)
    
    return sorted_data , y


# In[ ]:


bd_1975 = df[df['year'] == 1975 ] ['bdepth']
bd_2012 = df[df['year'] == 2012 ] ['bdepth']

bd_1975 = np.array(bd_1975)
bd_2012 = np.array(bd_2012)

x_1975 , y_1975 = ecdf(bd_1975)
x_2012 , y_2012 = ecdf(bd_2012)

plt.figure()
_ = plt.plot(x_1975 , y_1975 , marker= '.' , linestyle = 'none')
_ = plt.plot(x_2012 , y_2012,  marker= '.' , linestyle = 'none') # or u can use scatter
_ = plt.xlabel('b depth mm')
_ = plt.ylabel('ecdf')
_ = plt.legend(['1975' , '2012'] , loc= 'lower right')
plt.margins(0.1)
plt.show()


#  mean is larger in 2012 and the variance is higher (in 2012) !

# # task 2 (parameter estimation)
# ## we want to estimate the difference in mean with 95% CI

# #### draw bs replicates funciton (boot strap)

# In[ ]:


def boot_strap_1d(data , func):
    bs_temp = np.random.choice(data , len(data)) # this is choice with replacement
    return func(bs_temp)


# In[ ]:


def draw_bs_reps(data , func , size =1 ):
    bs = np.empty(size)
    for i in range(size):
        bs[i] = boot_strap_1d(data,func)
    return bs


# In[ ]:


## test the funcitnos above ##
n = np.array([1,2,3])
m = boot_strap_1d(n,np.mean)
m # each time you run you get a different m 


# In[ ]:


## test the function draw bs reps ##
n = np.array([1,2,3])
bs = draw_bs_reps(n,np.mean,size=10000)
plt.hist(bs)


# In[ ]:


mean_diff_obs = np.mean(bd_2012) - np.mean(bd_1975)

bs_replicates_1975 = draw_bs_reps(bd_1975 , np.mean , 10000)
bs_replicates_2012 = draw_bs_reps(bd_2012 , np.mean , 10000)

bs_diff_mean = bs_replicates_2012 - bs_replicates_1975

##compute the 95% Confindence Interval ##
## this is done using percintile ##
ci = np.percentile(bs_diff_mean , [2.5 , 97.5])
print('difference in mean observed ',mean_diff_obs , 'mm')
print('the 95% confidence interval is ' , ci , 'mm')

plt.figure()
plt.hist(bs_diff_mean)
plt.show()


# # task 3 : hypothesis test : are beaks deeper in 2012

# Your plot of the ECDF and determination of the confidence interval make it pretty clear that the beaks of G. scandens on Daphne Major have gotten deeper. But is it possible that this effect is just due to random chance? In other words, what is the probability that we would get the observed difference in mean beak depth if the means were the same?
# 
# Be careful! The hypothesis we are testing is not that the beak depths come from the same distribution. For that we could use a permutation test. The hypothesis is that the means are equal. To perform this hypothesis test, we need to shift the two data sets so that they have the same mean and then use bootstrap sampling to compute the difference of means.

# In[ ]:


mean_combined = np.mean(np.concatenate([bd_1975, bd_2012]))
mean_diff_obs = np.mean(bd_2012) - np.mean(bd_1975)


# shift the data to have the same mean (according to the null hypothesis they have the same mena)
bd_1975_shifted = bd_1975 - np.mean(bd_1975) + mean_combined
bd_2012_shifted = bd_2012 - np.mean(bd_2012) + mean_combined

# make the boot strap on each data shifted
bs_replicates_1975 = draw_bs_reps(bd_1975_shifted , np.mean , size= 10000 )
bs_replicates_2012 = draw_bs_reps(bd_2012_shifted , np.mean , size= 10000 )

bs_diff_mean = bs_replicates_2012 - bs_replicates_1975

## what is the probability that the observed mean is from the bs_diff_mean ##
p_value = np.sum(bs_diff_mean >= mean_diff_obs) / len(bs_diff_mean)
print('p value is ', p_value)


# Changing by 0.2 mm in 37 years is substantial by evolutionary standards. If it kept changing at that rate, the beak depth would double in only 400 years.
# 
# note:
# but may be they are not evolving : they are responding to change (e.g : sort of food hard to collect for small beaks so the birds with small beaks died and the ones with large beaks survived)
# 
# Now the next step: evolution. The Grants found that the offspring of the birds that survived the 1977 drought tended to be larger, with bigger beaks. So the adaptation to a changed environment led to a larger-beaked finch population in the following generation.[1]
# 
# 1 --> Peter R. Grant; Ecology and Evolution of Darwin's Finches

# # studying the relation between the beak length and depth over time

# In[ ]:


bl_1975 = df[df['year'] == 1975] ['blength']
bl_2012 = df[df['year'] == 2012] ['blength']
bl_1975 = np.array(bl_1975)
bl_2012 = np.array(bl_2012)

_ = plt.scatter(bl_1975 , bd_1975 , color = 'blue' , alpha=0.5)
_ = plt.scatter(bl_2012 , bd_2012 , color = 'red' , alpha=0.5)
_ = plt.legend(['1975' , '2012'] , loc= 'lower right')
_ = plt.xlabel('beak length')
_ = plt.ylabel('beak depth')

plt.show()


# Great work! In looking at the plot, we see that beaks got deeper (the red points are higher up in the y-direction), but not really longer. If anything, they got a bit shorter, since the red dots are to the left of the blue dots. So, it does not look like the beaks kept the same shape; they became shorter and deeper.

# ## making boot strap linear regression

# #### draw boot strap pairs linear regression function

# In[ ]:


def draw_bs_pairs_linreg(x,y,size= 1 ):
    slope = np.empty(size)
    intercept = np.empty(size)
    inds = np.arange(len(y))
    
    for i in range(size):
        bs_inds = np.random.choice (inds , len(inds))
        bs_x = x[bs_inds]
        bs_y = y[bs_inds]
        
        slope[i] , intercept[i] = np.polyfit(bs_x , bs_y , 1)
    return slope , intercept
        


# In[ ]:


## test the funcitno ##
x = np.array([1,2,3,4,5,6,7,8,9])
y = np.array([0,3,7,20,12,14,20,26,18]) 

plt.scatter(x,y)
slope , intercept = draw_bs_pairs_linreg(x,y,size =1000) # if the number of data is small poly fit will make a warning (badly conditioned)

y_dash = np.mean(slope)*x + np.mean(intercept)
plt.plot(x , y_dash)


# Perform a linear regression for both the 1975 and 2012 data. Then, perform pairs bootstrap estimates for the regression parameters. Report 95% confidence intervals on the slope and intercept of the regression line.

# In[ ]:


slope_1975 , intercept_1975 = np.polyfit(bl_1975 , bd_1975 , 1)
slope_2012 , intercept_2012 = np.polyfit(bl_2012 , bd_2012 , 1)

## performing boot strap on data to get slope and intercept
slope_bs_1975 , intercept_bs_1975 = draw_bs_pairs_linreg(bl_1975 , bd_1975 , 1000)
slope_bs_2012 , intercept_bs_2012 = draw_bs_pairs_linreg(bl_2012 , bd_2012 , 1000)

# compute the 95% percentile (confidence interval)
ci_slope_1975 = np.percentile(slope_bs_1975 , [2.5,97.5])
ci_intercept_1975 = np.percentile(intercept_bs_1975 , [2.5,97.5])

ci_slope_2012 = np.percentile(slope_bs_2012 , [2.5,97.5])
ci_intercept_2012 = np.percentile(intercept_bs_2012 , [2.5,97.5])

# Print the results
print('1975: slope =', slope_1975,
      'conf int =', ci_slope_1975)
print('1975: intercept =', intercept_1975,
      'conf int =', ci_intercept_1975)
print('2012: slope =', slope_2012,
      'conf int =', ci_slope_2012)
#print('my conf interval' , confidence_interval(slope_bs_2012 , 95))
print('2012: intercept =', intercept_2012,
      'conf int =', ci_intercept_2012)

_ = plt.scatter(bl_1975 , bd_1975)
_ = plt.plot(bl_1975 , slope_1975 * bl_1975 + intercept_1975)
_ = plt.xlabel('b length mm')
_ = plt.ylabel('b depth mm')
_ = plt.title('1975')
plt.show()


# Nicely done! It looks like they have the same slope, but different intercepts.

# ## drawing the first 100 boot strap parameters for the linear regression 

# In[ ]:


## performing boot strap on data to get slope and intercept
slope_bs_1975 , intercept_bs_1975 = draw_bs_pairs_linreg(bl_1975 , bd_1975 , 1000)
slope_bs_2012 , intercept_bs_2012 = draw_bs_pairs_linreg(bl_2012 , bd_2012 , 1000)

_ = plt.scatter(bl_1975 , bd_1975)
_ = plt.scatter(bl_2012 , bd_2012)
_ = plt.xlabel('b length mm')
_ = plt.ylabel('b depth mm')
_ = plt.legend(['1975' , '2012'])

x = np.array([10,17])

for i in range(100):
    plt.plot(x , slope_bs_1975[i] *x + intercept_bs_1975[i] , color = 'blue' , linewidth = 0.5 , alpha =0.2 )
    plt.plot(x , slope_bs_2012[i] *x + intercept_bs_2012[i] , color = 'red' , linewidth = 0.5 , alpha =0.2 )


plt.show()


# ## computing the ration of beak length to depth

# ### confidence interval funciton

# In[ ]:


def confidence_interval(data , ci= 95):
    ci_list = [(100-ci) /2 , ci + (100-ci)/2]
    confidence_interval = np.percentile(data , ci_list)
    return confidence_interval


# In[ ]:


print(confidence_interval(intercept_bs_2012 , ci=95)) # tested okay


# In[ ]:


ratio_1975 = bl_1975 / bd_1975
ratio_2012 = bl_2012 / bd_2012

mean_ratio_1975 = np.mean(ratio_1975)
mean_ratio_2012 = np.mean(ratio_2012)

## performing boot strap on the data to get CI for the mean ##
bs_mean_ratio_1975 = draw_bs_reps(ratio_1975 , func= np.mean , size = 10000)
bs_mean_ratio_2012 = draw_bs_reps(ratio_2012 , func= np.mean , size = 10000)

## get the confidence interval of 99% ##
ci_mean_ratio_1975 = confidence_interval(bs_mean_ratio_1975 , ci= 0.99)
ci_mean_ratio_2012 = confidence_interval(bs_mean_ratio_2012 , ci= 0.99)

## print the results ##
print('1975: mean ratio =', mean_ratio_1975,
      'conf int =', ci_mean_ratio_1975)
print('2012: mean ratio =', mean_ratio_2012,
      'conf int =', ci_mean_ratio_2012)


# ### plotting the range of ratios in both years

# In[ ]:


_ = plt.plot(bs_mean_ratio_1975 ,np.full((10000) , 1975) , color = 'red')
_ = plt.plot(bs_mean_ratio_2012 ,np.full((10000) , 2012) , color = 'blue')

## making the y-axis labels ##
_ = plt.yticks([1975,2012])

_ = plt.plot(mean_ratio_1975 , 1975 , 'ro'  ) #  ro : r means red and o means circle
_ = plt.plot(mean_ratio_2012 , 2012 , 'bo')
_ = plt.xlabel('beak length / beak depth')
_ = plt.ylabel('year')
## putting the legend ##
_ = plt.legend(['1975','2012'])
plt.show()
print('it\'s notable that the two ranges of the mean ratio of length over depth don\'t even overlap so the two can\'t be of the same distribution(a change happend in the ratio)' )


# the change in mean is 0.11 or 7% from 1975 to 2012 and the range of ratio is not even overlapping which means the two data sets are not likely to be of the same distribution or have the same mean (the changed happend)

# # studying the cause of the change in beak 
# may be it's because mating with the other species (fortis) then the هجين takes some attributes from fortis and inhert it to the children
# 
# we want to study how its likely for the children to inhert the parental traits in both species (fortis and scandals)

# In[ ]:


df_2012['year'] = 2012
df_1975['year'] = 1975
df_1975.rename(mapper={'Beak length, mm':'blength' , 'Beak depth, mm' : 'bdepth'} , axis = 1 , inplace = True)
df_all = df_1975.append(df_2012)
df_all


# In[ ]:


df_2012_fortis = df_2012[df_2012['species'] == 'fortis'].copy()
df_2012_fortis['year'] = 2012

df_1975_fortis = df_1975[df_1975['species'] == 'fortis'].copy()
df_1975_fortis.rename(mapper={'Beak length, mm':'blength' , 'Beak depth, mm' : 'bdepth'} , axis = 1 , inplace = True)
df_1975_fortis['year'] = 1975

df_all = df_1975.append(df_2012)

# df_fortis = df_1975_fortis.append(df_2012_fortis)
df_fortis = df_all[df_all['species'] == 'fortis']
df_scandens = df_all[df_all['species'] == 'scandens']
df_fortis.describe()


# ## visualizing the change in beak depth

# #### the evolution of the depth of the fortis

# In[ ]:


px.box(data_frame=df_fortis , y = 'bdepth' , color = 'year' )


# #### the evolution of the depth of scandens

# In[ ]:


px.box(data_frame=df_scandens , y = 'bdepth' , color = 'year' )


# #### the difference in the depth in 1975 between the two species

# In[ ]:


px.box(data_frame=df_all[df_all['year'] == 1975] , y = 'bdepth' , color = 'species' ,title= 'difference in depth in 1975' )


# #### the difference in the depth in 2012 between the two species

# In[ ]:


px.box(data_frame=df_all[df_all['year'] == 2012] , y = 'bdepth' , color = 'species' ,title='depth in 2012')


# **conclusion :**
# <br>
# 1- the depth of the **fortis** decreased from 1975 to 2012 <br>
# 2- the depth of the **scandens** increased from 1975 to 2012 <br>
# 3- the depth of the **fortis** was larger in 1975<br>
# 4- the depth range of the **scandens** is larger in 2012!! 
