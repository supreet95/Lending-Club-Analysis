#!/usr/bin/env python
# coding: utf-8

# In[137]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# ### Load the loan_info dataset

# In[9]:


loan_info=pd.read_sas('D:/Input Data/SAS Datasets/loan_info.sas7bdat')


# In[10]:


loan_info.shape


# In[11]:


loan_info.head()


# In[26]:


loan_info.describe()


# In[12]:


loan_info.columns


# ### Load the extra_info_m dataset

# In[14]:


extra_info=pd.read_sas('D:/Input Data/SAS Datasets/extra_info_m.sas7bdat')


# In[15]:


extra_info.shape


# In[19]:


extra_info.columns


# In[28]:


extra_info.dtypes


# In[52]:


#Convert byte to object


# In[45]:


extra_info['url']=extra_info['url'].str.decode("utf-8")


# In[46]:


extra_info['url'][0]


# In[ ]:


#Add a new column called 'new_id' - by splitting the url string


# In[50]:


extra_info['new_id']= extra_info['url'].str.split('=').str[1]


# In[51]:


extra_info['new_id'].dtype


# In[53]:


extra_info["new_id"] = extra_info.new_id.astype(float)


# In[54]:


extra_info['new_id'].describe()
###this is same as the column 'id' in the main file


# In[59]:


###Check for null values


# In[58]:


extra_info['new_id'].isnull().sum()


# #### All statistical parameters match with column 'id' from the loan_info file. We can go ahead and merge both the files on 'id' and 'new_id'

# In[60]:


###Merging both the files


# In[61]:


loan_merge = pd.merge(loan_info,extra_info,left_on ='id', right_on ='new_id', how='outer')


# In[64]:


loan_merge.shape


# In[66]:


loan_merge.info()


# In[69]:


loan_merge.to_csv('loan_merge.csv')


# In[67]:


import warnings
warnings.filterwarnings('ignore')


# In[76]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from matplotlib.ticker import PercentFormatter


# In[1]:


### Check for missing values


# In[85]:


#percentage missing values
missing_data = loan_merge.isnull().sum()
missing_data = missing_data[missing_data.values >(0.5*len(loan_merge))]/len(loan_merge)

#plotting
plt.figure(figsize=(20,4))
ax = missing_data.plot(kind='bar')
plt.title('List of Columns & NA counts where NA values are more than 50%')

# manipulate axis format
vals = ax.get_yticks()
ax.set_yticklabels(['{:,.2%}'.format(x) for x in vals])

plt.show()


# In[92]:


missing_data.index


# In[94]:


loan_merge.drop(['desc', 'mths_since_last_delinq', 'mths_since_last_record',
       'mths_since_last_major_derog', 'annual_inc_joint', 'dti_joint',
       'verification_status_joint', 'open_acc_6m', 'open_il_6m', 'open_il_12m',
       'open_il_24m', 'mths_since_rcnt_il', 'total_bal_il', 'il_util',
       'open_rv_12m', 'open_rv_24m', 'max_bal_bc', 'all_util', 'inq_fi',
       'total_cu_tl', 'inq_last_12m'],axis=1, inplace=True)


# In[96]:


loan_merge.shape


# In[108]:


loan_merge.describe()

###policy code must be removed.


# In[106]:


levels = loan_merge.nunique()
levels = levels[levels.values==1]


# In[107]:


levels


# In[109]:


loan_merge.drop(['policy_code'], axis=1, inplace=True)


# In[110]:


loan_merge.shape


# In[111]:


loan_merge.columns

##can get rid of (member_id, url,new_id)


# In[112]:


loan_merge.drop(['member_id','url','new_id',], axis=1, inplace=True)


# In[114]:


loan_merge.shape


# In[290]:


loan_merge.total_pymnt.describe()


# In[115]:


loan_merge.to_csv('loan_merge_50.csv')


# In[113]:


loan_merge.describe()


# In[178]:


loan_merge = pd.read_csv('D:/Analysis/Supreet/loan_merge_50.csv')


# In[186]:





# In[169]:


fig, ax = plt.subplots(1, 2, figsize=(16,5))
sns.distplot(loan_merge['loan_amnt'], ax=ax[0])
ax[0].set_title("Loan Amount Distribution")
sns.distplot(loan_merge['funded_amnt'], ax=ax[1])
ax[1].set_title("Funded Amount Distribution")


# In[208]:


loan_merge['term'].head()


# In[207]:


loan_merge['term'] = loan_merge['term'].str[2:-1]


# In[ ]:





# In[209]:


##bar graph
plt.figure(figsize=(8, 5))
sns.barplot(y=loan_merge.term.value_counts(), x=loan_merge.term.value_counts().index)
plt.xticks(rotation=0)
plt.title("Loan's Term Distribution")
plt.ylabel("Count")


# In[211]:


loan_merge['grade'] = loan_merge['grade'].str[2:-1]


# In[212]:


plt.figure(figsize=(8, 5))
sns.barplot(y=loan_merge.grade.value_counts(), x=loan_merge.grade.value_counts().index)
plt.xticks(rotation=0)
plt.title("Loan's Distribution by Grade")
plt.ylabel("Count")


# In[217]:


loan_merge['sub_grade'] = loan_merge['sub_grade'].str[2:-1]


# In[218]:


plt.figure(figsize=(21, 5))
sns.barplot(y=loan_merge.sub_grade.value_counts(), x=loan_merge.sub_grade.value_counts().index)
plt.xticks(rotation=0)
plt.title("Loan's Distribution by Sub-Grade")
plt.ylabel("Count")


# In[219]:


loan_merge['purpose'] = loan_merge['purpose'].str[2:-1]


# In[259]:


plt.figure(figsize=(21, 5))
sns.barplot(y=((loan_merge.purpose.value_counts()/len(loan_merge)).round(2)), x=loan_merge.purpose.value_counts().index)
plt.xticks(rotation=0)
plt.title("Distribution by Purpose")
plt.ylabel("Percentage of Loan (%)")


# In[263]:


plt.figure(figsize=(12, 5))
sns.boxplot(y=loan_merge.loan_amnt, x=loan_merge.verification_status)
plt.title("Distribution by Verification Status")
plt.ylabel("Count")


# ### Analysis of Loan Status

# In[292]:


loan_merge['loan_status'][0]


# In[293]:


loan_merge['loan_status'].unique()


# In[271]:


loan_merge['loan_status'] = loan_merge['loan_status'].str[2:-1].head()


# In[272]:


loan_merge['credit_policy'] = loan_merge['loan_status'].str.split('.').str[0]


# In[275]:


loan_merge['credit_policy'].fillna("Meets credit policy", inplace=True)


# In[276]:


loan_merge['credit_policy'].unique()


# In[278]:


plt.figure(figsize=(12, 5))
sns.barplot(y=((loan_merge.credit_policy.value_counts()/len(loan_merge)).round(2)), x=loan_merge.credit_policy.value_counts().index)
plt.xticks(rotation=0)
plt.title("Adherence to Credit Policy")
plt.ylabel("Percentage of Loan (%)")


# In[279]:


loan_merge['credit_policy'].value_counts()


# ### Loan Status

# In[282]:


loan_merge['status_new'] = loan_merge['loan_status'].str.split(':').str[1]


# In[283]:


loan_merge['status_new'].unique()


# In[294]:


extra_info['loan_status'].unique()


# In[297]:


extra_info['loan_status_new'] = extra_info['loan_status'].str.decode('utf-8')


# In[301]:


extra_info['loan_status_new'].value_counts()


# In[ ]:




