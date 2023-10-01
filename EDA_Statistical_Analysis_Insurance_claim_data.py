#!/usr/bin/env python
# coding: utf-8

# In[249]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import seaborn as sns
import scipy.stats as stats 
get_ipython().run_line_magic('matplotlib', 'inline')


# 1. Import claims_data.csv and cust_data.csv which is provided to you and 
# combine the two datasets appropriately to create a 360-degree view of 
# the data. Use the same for the subsequent questions.

# In[250]:


file_path = 'C:/Users/hp/Desktop/Data Analytics/Assignments/Python/Python Foundation Case Study 3 - Insurance Claims Case Study_A4/'
cust_demo = pd.read_csv( file_path + 'cust_demographics.csv')
claims = pd.read_csv( file_path + 'claims.csv')


# In[251]:


cust_claims = pd.merge( left = cust_demo, right = claims, how = 'inner', left_on = 'CUST_ID', right_on = 'customer_id')


# In[252]:


cust_demo.head()


# In[253]:


claims.head()


# In[254]:


cust_claims.head()


# In[255]:


cust_claims.info()


# 2. Perform a data audit for the datatypes and find out if there are any 
# mismatch within the current datatypes of the columns and their 
# business significance.

# In[256]:


cust_claims['total_policy_claims']=cust_claims['total_policy_claims'].astype('object')


# In[257]:


cust_claims['DateOfBirth'] = pd.to_datetime(cust_claims['DateOfBirth'])


# In[258]:


cust_claims['DateOfBirth'] = np.where(pd.DatetimeIndex(cust_claims['DateOfBirth']).year<2000 ,cust_claims.DateOfBirth,cust_claims.DateOfBirth - pd.offsets.DateOffset(years=100))


# In[259]:


cust_claims['claim_date'] = pd.to_datetime(cust_claims['claim_date'])


# In[260]:


cust_claims.drop( columns = 'customer_id', inplace = True )


# In[261]:


cust_claims['CUST_ID']=cust_claims['CUST_ID'].astype(str)


# In[262]:


cust_claims['claim_id']=cust_claims['claim_id'].astype(str)


# 3. Convert the column claim_amount to numeric. Use the appropriate 
# modules/attributes to remove the $ sign.

# In[263]:


cust_claims['claim_amount'] = cust_claims['claim_amount'].str.replace('$','').astype(np.float64).round(2)
cust_claims


# In[264]:


def fn_cat_descriptive(x):
     # missing values calculation
    ntot = x.shape[0]
    n = x.count()
    nmiss = ntot - n
    nmiss_perc = nmiss*100/ntot
    freq = x.value_counts().sort_values(ascending = False).reset_index().iloc[0,1]
    freq_prec = freq *100/n
     # return the descripitves
    return pd.Series( [x.dtype, x.nunique(), 
                       ntot, n, nmiss, nmiss_perc, 
                       x.mode()[0],freq ,freq_prec],
                    index = ['dtype', 'cardinality',
                             'ntot', 'n', 'nmiss', 'nmiss_perc',
                             'mode', 'mode_freq','mode_perc'])


# In[265]:


cust_claims.select_dtypes(['object']).apply( lambda x: fn_cat_descriptive(x) )


# In[266]:


def fn__num_descriptive( x ):
    
    # missing values calculation
    ntot = x.shape[0]
    n = x.count()
    nmiss = ntot - n
    nmiss_perc = nmiss*100/ntot
    
    # get the lc and uc using IQR
    q1 = x.quantile(0.25)
    q3 = x.quantile(0.75)
    IQR =  q3 - q1
    lc_iqr = q1 - 1.5 * IQR
    uc_iqr = q3 + 1.5 * IQR
    
    # return the descripitves
    return pd.Series( [x.dtype, x.nunique(), 
                       ntot, n, nmiss, nmiss_perc,
                       IQR, lc_iqr, uc_iqr, 
                       x.sum(), x.mean(), x.var(), x.std(), 
                       x.min(),
                       x.quantile(0.01), x.quantile(0.05), x.quantile(0.10), 
                       x.quantile(0.25), x.quantile(0.5), x.quantile(0.75), 
                       x.quantile(0.90), x.quantile(0.95), x.quantile(0.99),
                       x.max()], 
                    index = ['dtype', 'cardinality',
                             'ntot', 'n', 'nmiss', 'nmiss_perc', 
                             'IQR', 'lc_iqr', 'uc_iqr', 
                             'sum', 'mean', 'var', 'std', 
                             'min', 
                             'p1', 'p5', 'p10',
                             'p25', 'p50', 'p75', 
                             'p90', 'p95', 'p99',
                             'max'])


# In[267]:


cust_claims.select_dtypes(['float64','int64']).apply( lambda x: fn_descriptive(x) )


# 4. Of all the injury claims, some of them have gone unreported with the 
# police. Create an alert flag (1,0) for all such claims.

# In[268]:


cust_claims['alert_flag'] = pd.Series( np.where((cust_claims.police_report == 'Unknown'), 1 , 0 ))


# 5. One customer can claim for insurance more than once and in each claim,
# multiple categories of claims can be involved. However, customer ID 
# should remain unique. 
# Retain the most recent observation and delete any duplicated records in
# the data based on the customer ID column.

# In[269]:


cust_claims.sort_values( by = ['CUST_ID','claim_date'], ascending = [ True, False], inplace = True)


# In[270]:


cust_claims.drop_duplicates( subset ='CUST_ID', keep = 'first' , inplace = True)


# 6. Check for missing values and impute the missing values with an 
# appropriate value. (mean for continuous and mode for categorical)

# In[271]:


cust_claims.total_policy_claims.fillna(cust_claims.total_policy_claims.mode()[0],inplace = True)


# In[272]:


cust_claims.claim_amount.fillna(cust_claims.claim_amount.mean(),inplace = True)


# 7. Calculate the age of customers in years. Based on the age, categorize the
# customers according to the below criteria
# Children < 18
# Youth 18-30
# Adult 30-60
# Senior > 60

# In[273]:


cust_claims['Age'] = pd.Timestamp.today().year - cust_claims['DateOfBirth'].dt.year


# In[274]:


cust_claims['Age_group'] = pd.cut(cust_claims['Age'],bins=[0,17,30,60,63],labels=['Children','Youth','Adult','Senior'],include_lowest=True)


# In[307]:


cust_claims.Age_group.value_counts()


# 8. What is the average amount claimed by the customers from various 
# segments?

# In[276]:


cust_claims.groupby('Segment')[['claim_amount']].mean().round(2)


# 9. What is the total claim amount based on incident cause for all the claims
# that have been done at least 20 days prior to 1st of October, 2018.

# In[277]:


cust_claims[cust_claims.claim_date <= pd.Timestamp( year = 2018, month = 9, day = 11 )].groupby('incident_cause')[['claim_amount']].sum().round(2)


# 10. How many adults from TX, DE and AK claimed insurance for driver 
# related issues and causes? 

# In[278]:


var_x= cust_claims[(cust_claims.Age_group =='Adult') & (cust_claims.State.isin (['TX','DE','AK']))& (cust_claims.incident_cause.isin(['Driver error','Other driver error']))]
var_x.head()


# In[279]:


print('Number of adults from TX, DE and AK, who claimed insurance for driver related issues and causes:',var_x.CUST_ID.count())


# 11. Draw a pie chart between the aggregated value of claim amount based 
# on gender and segment. Represent the claim amount as a percentage on
# the pie chart.

# In[280]:


cust_claims.groupby('gender').claim_amount.sum().plot(kind='pie',autopct='%1.1f%%')


# In[281]:


cust_claims.groupby('Segment').claim_amount.sum().plot(kind='pie',autopct='%1.1f%%')


# 12. Among males and females, which gender had claimed the most for any 
# type of driver related issues? E.g. This metric can be compared using a 
# bar chart

# In[282]:


cust_claims[cust_claims.incident_cause.isin(['Driver error','Other driver error'])].groupby('gender').gender.count().plot( kind = 'bar')


# 13. Which age group had the maximum fraudulent policy claims? Visualize 
# it on a bar chart.

# In[283]:


cust_claims.groupby('Age_group').fraudulent.count().sort_values(ascending = False).plot( kind = 'bar')


# 14. Visualize the monthly trend of the total amount that has been claimed 
# by the customers. Ensure that on the “month” axis, the month is in a 
# chronological order not alphabetical order. 

# In[284]:


cust_claims['month'] = cust_claims['claim_date'].dt.month


# In[285]:


cust_claims.groupby('month')[['claim_amount']].sum().plot( kind = 'line')
plt.xlabel('Month')
plt.ylabel('Amount')
plt.show()


# 15. What is the average claim amount for gender and age categories and 
# suitably represent the above using a facetted bar chart, one facet that 
# represents fraudulent claims and the other for non-fraudulent claims.
# Based on the conclusions from exploratory analysis as well as suitable 
# statistical tests, answer the below questions. Please include a detailed 
# write-up on the parameters taken into consideration, the Hypothesis 
# testing steps, conclusion from the p-values and the business implications of 
# the statements. 

# In[313]:


avg_claim_amount = cust_claims.groupby(['gender', 'Age_group', 'fraudulent'])['claim_amount'].mean().reset_index()
avg_claim_amount


# In[314]:


sns.catplot(data = avg_claim_amount, x='Age_group', y='claim_amount', hue='gender', col='fraudulent', kind='bar')


# 16. Is there any similarity in the amount claimed by males and females?
We will perform a two-sample t-test.
# In[83]:


cust_claims.columns


# In[315]:


# take the two samples
var_name = 'claim_amount'
male = cust_claims.loc[cust_claims.gender == 'Male',var_name ]
female = cust_claims.loc[cust_claims.gender == 'Female',var_name ]

# display the mean claim amounts
print('mean spend of male:', male.mean())
print('mean spend of female:', female.mean())

H0: The mean claim amount for males is equal to the mean claim amount for females
Ha: The mean claim amount for males is not equal to the mean claim amount for females
CI: 99%, p: 0.01
# In[316]:


# perform the test
stats.ttest_ind( male, female )

conclusion : 
p-value of 0.33601 is greater than 0.01.
we fail to reject the null hypothesis at the 99% confidence level.Business implications:
Gender is not a significant factor in predicting claim amounts for insurance companies.
# 17. Is there any relationship between age category and segment?
We will perform a chi-square test of independence
# In[54]:


cust_claims.columns


# In[317]:


# get the ob_freq_table from the dataset
obs_freq = pd.crosstab(cust_claims.Age_group, cust_claims.Segment )
obs_freq

Ho: No relationship between the two variables (Age_group and Segment)
Ha: There is association between the two variables.
CI: 95%, p: 0.05
# In[318]:


# perform the test 
stats.chi2_contingency( obs_freq )

Conclusion:
p-value of 0.9555 is greater than 0.05.
we fail to reject the null hypothesis at the 95% confidence level.Business Implications:
Marketing campaigns targeted at specific age category may not be as effective as campaigns targeted at other factors.
# 18. The current year has shown a significant rise in claim amounts as 
# compared to 2016-17 fiscal average which was $10,000.
We will perform a one-sample t-test
# In[319]:


cust_claims.columns


# In[320]:


# sample to be considered
var_name = 'claim_amount'

# values to be compared
prev_fiscal_avg = 10000
current = cust_claims.loc[:, var_name]
mean_current = current.mean()

# display the means
print('prev yr_fiscal_avg = 10,000' , '| current yr fiscal average ', mean_current  )

H0: The mean claim amount for the current year is leass than or equal to the 2016-17 fiscal average of  10,000
Ha: The mean claim amount for the current year is greater than the 2016-17 fiscal average of 10,000
CI: 99%, p: 0.01
# In[321]:


# perform the test
stats.ttest_1samp( a = current, popmean = prev_fiscal_avg )

conclusion:
p-value of 11.11e-10 is divided by 2 to get the one-tailed p-value, which is 5.56e-10.
p-value is much smaller than of 0.01.
we reject the null hypothesis at the 99% confidence level.Business conclusion:
Claim amounts has increased from the last year.
The company needs to allocate more resources to handle the increased claim amount.
# 19. Is there any difference between age groups and insurance claims?
We will  perform a one-way ANOVA test
# In[322]:


# get the num unique of age_groups
cust_claims.Age_group.nunique()


# In[323]:


# get the freq of age_groups
cust_claims.Age_group.value_counts()


# In[324]:


# declare the variable for which we want to do the analysis
var = 'claim_amount'

# filter the data based on segments
s1 = cust_claims.loc[cust_claims.Age_group == 'Adult', var ]
s2 = cust_claims.loc[cust_claims.Age_group == 'Youth', var ]
s3 = cust_claims.loc[cust_claims.Age_group == 'Senior', var ]

# display the mean of the three sample
print( 'mean of s1: ', s1.mean(), '| mean of s2: ', s2.mean(), '| mean of s3: ', s3.mean() )

Ho: There is no difference in the mean insurance claims across different age groups, means are EQUAL
Ha:  There is difference in the mean insurance claims across different age groups, means are UNEQUAL
CI: 99%, p: 0.01
# In[325]:


# perform the test
stats.f_oneway( s1, s2, s3 )

Conlcusion:
The p-value of 0.7144 is greater than the significance level of 0.01.
we fail to reject the null hypothesis at the 99% confidence level.Business Implications:
Age is not strong predictor of the amount of insurance claims.
There is no need for differential pricing based on age group.
# 20. Is there any relationship between total number of policy claims and the 
# claimed amount?
We will perform a correlation test
# In[326]:


cust_claims.columns


# In[330]:


cust_claims.plot( kind = 'scatter', x = 'claim_amount', y = 'total_policy_claims')
plt.show()


# In[331]:


cust_claims.loc[:, ['total_policy_claims','claim_amount']].corr()

Ho: No relationship between the two variables (total_policy_claims and claim_amount)
Ha: There is association between the two variables.
CI: 95%, p: 0.05
# In[332]:


# perform the test 
stats.pearsonr( cust_claims.total_policy_claims,cust_claims.claim_amount )

Conclusions: 
There is a weak negative correlation between total_policy_claims and claim_amount.
The p-value is 0.4624932766041524 , the observed correlation coefficient -0.022401566777628834 is not significant.
We fail to reject the null hypothesis at the 95% confidence level.Business Implications:
Other factors may be affecting the relationship between these variables, such as claim type or the customer demographics.
# In[ ]:




