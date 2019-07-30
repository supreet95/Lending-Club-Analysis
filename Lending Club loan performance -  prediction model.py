#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Loan Performace Analysis on shortlisted variables


# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[2]:


processed_data=pd.read_sas('D:/Data/Summer_Data/Combine2.sas7bdat')


# In[3]:


processed_data.shape


# In[4]:


processed_data.columns


# In[5]:


processed_data.describe()


# In[6]:


sns.kdeplot(processed_data['loan_amnt'], shade=True)


# In[30]:


sns.kdeplot(processed_data['annual_inc'], shade = True)


# In[62]:


processed_data['emp_length'].value_counts()


# In[35]:


processed_data['annual_inc'].describe(percentiles=[0.05,0.1,0.2,0.3,0.5,0.75,0.9])


# In[36]:


sns.kdeplot(processed_data['dti'], shade = True)


# In[41]:


sns.kdeplot(processed_data['installment'], shade = True)


# In[37]:


sns.kdeplot(processed_data['pub_rec'], shade = True)


# In[7]:


sns.kdeplot(processed_data['int_rate'], shade=True)


# In[ ]:


#bucketing interest rate into high, medium and low groups


# In[15]:


processed_data['int_buc'] = 'Low'


# In[16]:


processed_data.loc[(processed_data['int_rate']>10), 'int_buc'] = "Medium"


# In[17]:


processed_data.loc[(processed_data['int_rate']>17), 'int_buc'] = "High"


# In[18]:


processed_data['int_buc'].value_counts()


# In[2]:


#bucketing installment into H/M/L groups


# In[44]:


processed_data['installment_buc'] = 'Low'


# In[45]:


processed_data.loc[(processed_data['installment']>200), 'installment_buc'] = "Medium"


# In[46]:


processed_data.loc[(processed_data['installment']>700), 'installment_buc'] = "High"


# In[47]:


processed_data['installment_buc'].value_counts()


# In[21]:


plt.figure(figsize=(8, 5))
sns.barplot(y=processed_data.int_buc.value_counts(), x=processed_data.int_buc.value_counts().index)
plt.xticks(rotation=0)
plt.title("Loan's Distribution by Interest Rate")
plt.ylabel("Count")


# In[24]:


plt.figure(figsize=(8, 5))
sns.boxplot(y=processed_data.loan_amnt, x=processed_data.int_buc)
plt.title("Distribution by Interest Rate")
plt.ylabel("Loan Amount")


# # Debt to Income ratio analysis

# ### Preparing the data

# In[63]:


processed_data.columns


# In[65]:


processed_data['purpose'] = processed_data['purpose'].str.decode('utf-8')


# In[66]:


processed_data['purpose'].value_counts()


# In[68]:


processed_data['purpose_buc'] = 'Other'


# In[70]:


processed_data.loc[(processed_data['purpose']=='debt_consolidation'), 'purpose_buc'] = "Debt Consolidation"


# In[71]:


processed_data.loc[(processed_data['purpose']=='credit_card'), 'purpose_buc'] = "Credit Card"


# In[72]:


processed_data.loc[(processed_data['purpose']=='home_improvement'), 'purpose_buc'] = "Home Improvement"


# In[73]:


processed_data['purpose_buc'].value_counts()


# In[75]:


processed_data['home_ownership'].value_counts()


# In[81]:


processed_data['issue_year'] = processed_data['issue_d'].dt.year


# In[82]:


processed_data['issue_year'].value_counts()


# In[83]:


processed_data['dti'].describe(percentiles=[0.05,0.1,0.2,0.3,0.5,0.75,0.9])


# In[92]:


processed_data['dti_buc'] = 'High'


# In[93]:


processed_data.loc[(processed_data['dti']<23), 'dti_buc'] = "Medium"


# In[94]:


processed_data.loc[(processed_data['dti']<14), 'dti_buc'] = "Low"


# In[95]:


processed_data['dti_buc'].value_counts()


# # Crosstabs

# ### DTI vs Issued Year

# In[111]:


dti_year = pd.crosstab(processed_data["issue_year"],processed_data["dti_buc"],margins=True, normalize= 'index')


# In[112]:


dti_year


# In[115]:


stacked = dti_year.stack().reset_index().rename(columns={0:'value'})


# In[127]:


plt.figure(figsize=(12,5))
sns.barplot(x=stacked.issue_year, y=stacked.value, hue=stacked.dti_buc)
plt.xticks(rotation=0)
plt.ylabel('% of Loans')
plt.xlabel('Issue Year')
plt.title('DTI trend over years');


# ### DTI vs State

# In[135]:


dti_state = pd.crosstab(processed_data["addr_state"],processed_data["dti_buc"],margins=True, normalize= 'index')


# In[136]:


stacked = dti_state.stack().reset_index().rename(columns={0:'value'})


# In[137]:


stacked['addr_state'] = stacked['addr_state'].str.decode('utf-8')
stacked.head()


# In[141]:


plt.figure(figsize=(25,7))
sns.barplot(x=stacked.addr_state, y=stacked.value, hue=stacked.dti_buc)
plt.xticks(rotation=0)
plt.ylabel('% of Loans')
plt.xlabel('State')
plt.title('DTI trend across states');


# ### DTI vs Purpose

# In[230]:


dti_purpose = pd.crosstab(processed_data["purpose_buc"],processed_data["dti_buc"],margins=True, normalize='index')


# In[231]:


stacked = dti_purpose.stack().reset_index().rename(columns={0:'value'})


# In[232]:


stacked.head()


# In[233]:


plt.figure(figsize=(12,5))
sns.barplot(x=stacked.purpose_buc, y=stacked.value, hue=stacked.dti_buc, errwidth=10)
plt.xticks(rotation=0)
plt.ylabel('% of Loans')
plt.xlabel('Purpose of Loan')
plt.title('DTI trend for different loan purposes');


# In[279]:


dti_purpose_all = pd.crosstab(processed_data["purpose_buc"],processed_data["dti_buc"],margins=True)


# In[281]:


stacked = dti_purpose_all.stack().reset_index().rename(columns={0:'value'})


# In[282]:


plt.figure(figsize=(12,5))
sns.barplot(x=stacked.purpose_buc, y=stacked.value, hue=stacked.dti_buc, errwidth=10)
plt.xticks(rotation=0)
plt.ylabel('% of Loans')
plt.xlabel('Purpose of Loan')
plt.title('DTI trend for different loan purposes');


# In[ ]:





# ### DTI vs Loan Amount

# In[153]:


plt.figure(figsize=(15,5))
sns.lineplot(x="issue_year", y="loan_amnt", hue="dti_buc", data=processed_data)
plt.xticks(rotation=0)
plt.ylabel('Loan Amount')
plt.xlabel('Issue Year')
plt.title('Loan Amount trend split by DTI ratio')


# ### DTI vs Term

# In[154]:


dti_term = pd.crosstab(processed_data["term"],processed_data["dti_buc"],margins=True, normalize= 'index')


# In[169]:


stacked = dti_term.stack().reset_index().rename(columns={0:'value'})


# In[170]:


stacked['term'] = stacked['term'].str.decode('utf-8')
stacked.head()


# In[177]:


plt.figure(figsize=(8,5))
sns.barplot(x=stacked.term, y=stacked.value, hue=stacked.dti_buc)
plt.xticks(rotation=0)
plt.ylabel('% of Loans')
plt.xlabel('Loan Term')
plt.title('DTI trend for different loan terms');


# ### DTI vs Grade

# In[178]:


dti_grade = pd.crosstab(processed_data["grade"],processed_data["dti_buc"],margins=True, normalize= 'index')


# In[179]:


stacked = dti_grade.stack().reset_index().rename(columns={0:'value'})


# In[182]:


stacked['grade'] = stacked['grade'].str.decode('utf-8')
stacked.head()


# In[186]:


plt.figure(figsize=(15,5))
sns.barplot(x=stacked.grade, y=stacked.value, hue=stacked.dti_buc)
plt.xticks(rotation=0)
plt.ylabel('% of Loans')
plt.xlabel('Loan Grade')
plt.title('DTI trend for different loan grades');


# ### DTI vs Interest Rate

# In[187]:


dti_int = pd.crosstab(processed_data["int_buc"],processed_data["dti_buc"],margins=True, normalize= 'index')


# In[188]:


stacked = dti_int.stack().reset_index().rename(columns={0:'value'})


# In[189]:


stacked.head()


# In[193]:


plt.figure(figsize=(10,5))
sns.barplot(x=stacked.int_buc, y=stacked.value, hue=stacked.dti_buc)
plt.xticks(rotation=0)
plt.ylabel('% of Loans')
plt.xlabel('Interest Rates Bucket')
plt.title('DTI trend for different interest rates');


# ### DTI vs Verification Status

# In[194]:


dti_verification = pd.crosstab(processed_data["verification_status"],processed_data["dti_buc"],margins=True, normalize= 'index')


# In[195]:


stacked = dti_verification.stack().reset_index().rename(columns={0:'value'})


# In[197]:


stacked['verification_status'] = stacked['verification_status'].str.decode('utf-8')
stacked.head()


# In[200]:


plt.figure(figsize=(10,5))
sns.barplot(x=stacked.verification_status, y=stacked.value, hue=stacked.dti_buc)
plt.xticks(rotation=0)
plt.ylabel('% of Loans')
plt.xlabel('Verification Status')
plt.title('DTI trend over status being verified')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


# ### DTI vs Home Ownership

# In[202]:


dti_home = pd.crosstab(processed_data["home_ownership"],processed_data["dti_buc"],margins=True, normalize= 'index')


# In[204]:


stacked = dti_home.stack().reset_index().rename(columns={0:'value'})


# In[205]:


stacked['home_ownership'] = stacked['home_ownership'].str.decode('utf-8')
stacked.head()


# In[207]:


plt.figure(figsize=(10,5))
sns.barplot(x=stacked.home_ownership, y=stacked.value, hue=stacked.dti_buc)
plt.xticks(rotation=0)
plt.ylabel('% of Loans')
plt.xlabel('Home Ownership')
plt.title('DTI trend for different home ownerships')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


# ### DTI vs Loan Status

# In[212]:


processed_data['loan_status'].value_counts()


# In[211]:


processed_data['loan_status'] = processed_data['loan_status'].str.decode('utf-8')


# In[210]:


processed_data['loan_status_buc'] = 'Undesired'


# In[213]:


processed_data.loc[(processed_data['loan_status']=='Fully Paid'), 'loan_status_buc'] = "Desired"


# In[214]:


processed_data.loc[(processed_data['loan_status']=='Current'), 'loan_status_buc'] = "Pending"


# In[215]:


processed_data['loan_status_buc'].value_counts()


# In[216]:


dti_status = pd.crosstab(processed_data["loan_status_buc"],processed_data["dti_buc"],margins=True, normalize= 'index')


# In[217]:


stacked = dti_status.stack().reset_index().rename(columns={0:'value'})


# In[218]:


stacked.head()


# In[221]:


plt.figure(figsize=(10,5))
sns.barplot(x=stacked.loan_status_buc, y=stacked.value, hue=stacked.dti_buc)
plt.xticks(rotation=0)
plt.ylabel('% of Loans')
plt.xlabel('Loan status')
plt.title('DTI trend across Loan Status')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


# ## Bubble chart

# In[234]:


processed_data.shape


# In[364]:


processed_data.columns


# In[374]:


animation_df = processed_data[['id','loan_status_buc','issue_year','grade','loan_amnt', 'annual_inc']]


# In[375]:


animation_df.head()


# In[376]:


animation_rollup = animation_df.groupby(['loan_status_buc','issue_year','grade']).agg(['mean', 'count','sum'])


# In[377]:


animation_rollup.to_csv('animation_rollup_status.csv')


# In[378]:


anim_csv = pd.read_csv('D:/Analysis/Supreet/animation_rollup_status.csv')


# In[379]:


anim_csv.head()


# In[372]:


sns.set_style("white")
my_dpi=78
import matplotlib.ticker as mtick


# In[384]:


# For each year:
for i in anim_csv.issue_year.unique():
    # initialize a figure
    fig = plt.figure(figsize=(680/my_dpi, 480/my_dpi), dpi=my_dpi)
    
    # And I need to transform my categorical column (continent) in a numerical value group1->1, group2->2...
    anim_csv['loan_status_buc']=pd.Categorical(anim_csv['loan_status_buc'])


    # Change color with c and alpha. I map the color to the X axis value.
    tmp=anim_csv[ anim_csv.issue_year == i ]
    plt.scatter(tmp['grade'], tmp['inc_mean'] , s=tmp['count_loan']/20,c=tmp['loan_status_buc'].cat.codes, cmap="plasma", alpha=0.6, edgecolors="grey", linewidth=2)

    # Add titles (main and on axis)
    #plt.yscale('log')
    plt.xlabel("Grade")
    plt.ylabel("Annual Income Mean")
    plt.title("Year: "+str(i) )
    #plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    
    fmt = '${x:,.0f}'
    tick = mtick.StrMethodFormatter(fmt)
    plt.gca().yaxis.set_major_formatter(tick)
    
    plt.ylim(50000,110000)
    #plt.xlim(30, 90)


# In[ ]:





# ## LOGISTIC REGRESSION

# In[306]:


logistic_df = processed_data.loc[processed_data['loan_status_buc'] != 'Pending']


# In[307]:


logistic_df.shape


# In[312]:


logistic_df.columns


# In[308]:


logistic_df['target_var'] = 0


# In[310]:


logistic_df.loc[(logistic_df['loan_status_buc']=='Desired'), 'target_var'] = 1


# In[311]:


logistic_df.describe()


# In[313]:


logistic_df = logistic_df[['id','loan_amnt','term','int_rate','grade','annual_inc','purpose_buc','home_ownership','verification_status','dti','target_var']]


# In[314]:


logistic_df.info()


# In[319]:


cat_vars=['term','grade','purpose_buc','home_ownership','verification_status']
for var in cat_vars:
    cat_list='var'+'_'+var
    cat_list = pd.get_dummies(logistic_df[var], prefix=var)
    
    logistic_df_2=logistic_df.join(cat_list)
    logistic_df=logistic_df_2


# In[328]:


logistic_df.columns


# In[330]:


train_df = logistic_df.drop(logistic_df[['id','term','grade','purpose_buc','home_ownership','verification_status']], axis=1)


# In[332]:


train_df.columns


# In[340]:


train_df['annual_inc'].fillna(0, inplace=True)


# In[347]:


train_df.isna().sum()


# ### Model Testing

# In[387]:


cols=['loan_amnt', 'int_rate', 'annual_inc', 'dti',
       "term_b'36 months'", "term_b'60 months'", "grade_b'A'", "grade_b'B'",
       "grade_b'C'", "grade_b'D'", "grade_b'E'", "grade_b'F'", "grade_b'G'",
       'purpose_buc_Credit Card', 'purpose_buc_Debt Consolidation',
       'purpose_buc_Home Improvement', 'purpose_buc_Other',
       "home_ownership_b'ANY'", "home_ownership_b'MORTGAGE'",
       "home_ownership_b'NONE'", "home_ownership_b'OTHER'",
       "home_ownership_b'OWN'", "home_ownership_b'RENT'",
       "verification_status_b'Not Verified'",
       "verification_status_b'Source Verified'",
       "verification_status_b'Verified'"] 
X=train_df[cols]
y=train_df['target_var']


# In[388]:


import statsmodels.api as sm
logit_model=sm.Logit(y,X)
result=logit_model.fit()
print(result.summary2())


# ### Test Train Split 

# In[ ]:





# In[342]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train_df.drop('target_var',axis=1),train_df['target_var'], 
                                                    test_size=0.30, random_state=101)


# In[3]:


###Implementing the model
import statsmodels.api as sm
logit_model=sm.Logit(y,X)
result=logit_model.fit()
print(result.summary2())


# In[343]:


from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)


# In[344]:


from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))


# In[ ]:




