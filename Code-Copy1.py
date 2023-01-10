#!/usr/bin/env python
# coding: utf-8

# # Data Exploration

# In[1]:


import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[2]:


os.chdir('C:/Users/popov/OneDrive/Рабочий стол/JOB/PYTHON/CARMAX')


# In[3]:


df=pd.read_csv('ShowcaseDataWinter2023.csv')


# In[6]:


print(df.shape)


# In[7]:


df.info()


# In[8]:


df.describe(include="all")


# In[9]:


df.isnull().sum() # number of missing values per column


# In[10]:


df.isnull().mean()*100 # number of missing values per column in %


# In[11]:


df.duplicated().sum() # no duplicates 


# We see that the model of the appraised vehicle is missing in 10% of all cases. Description of an appraised car an a new car are also missing in 15.5% and 11.9 % cases.

# In[10]:


df.head(10)


# ## Exploration of variables describing an appraised car

# In[123]:


ap_cols=[i for i in df.columns if 'appraisal' in i]


# In[124]:


df_ap=df[ap_cols] # 15 columns i.e. half of the columns are about the appraised car


# In[125]:


df_ap.dtypes # we have many object variables 


# In[126]:


cat_var=list(df_ap.dtypes[df_ap.dtypes==object].index)


# In[127]:


df_ap[['appraisal_offer']].sort_values(by=['appraisal_offer'])


# In[128]:


len(df_ap['appraisal_offer'].unique()) # 9 categories in the appraisal offer
# lets look at how the offer relates to the year of the car


# In[129]:


df_ap['appraisal_offer']=df_ap['appraisal_offer'].replace('$5k to $10k','$05k to $10k')
df_ap['appraisal_offer']=df_ap['appraisal_offer'].replace('$0k to $5k','$00k to $5k')


# In[130]:


df_ap['model_year_appraisal']=df_ap['model_year_appraisal'].astype('int')


# In[131]:


cross_tab_of_yr = pd.crosstab(index=df_ap['model_year_appraisal'],
                             columns=df_ap['appraisal_offer'])


# In[132]:


cross_tab_of_yr_prop = pd.crosstab(index=df_ap['model_year_appraisal'],
                             columns=df_ap['appraisal_offer'],
                                  normalize='index')


# In[133]:


labels = ['0-5', '10-15', '15-20', '20-25','25-30','30-35','35-40','40+']


# In[134]:


cross_tab_of_yr.plot(kind='bar',stacked=True,colormap='tab10',figsize=(15,9),fontsize=10)
plt.legend(labels,loc='upper left',ncol=9)
plt.xlabel('Model year')
#plt.ylabel(labels)
plt.show()


# In[23]:


cross_tab_of_yr_prop.plot(kind='bar',stacked=True,colormap='tab10',figsize=(12,9))
plt.legend(labels,loc='upper center',ncol=9)
plt.xlabel('Model year',fontsize=10)
plt.ylabel('Proportion',fontsize=10)
plt.title("Appraisal offered by Model Year, $1k")
plt.show()


# We see that almost all cars produced up to 2002 we offered the lowest appraisal offer what makes sense becasue of the car age.
# We also see that starting from the year 2011-2012 almost no cars were offered the lowest offer since they are quite new and hence are supposed to be in a decent shape.

# In[24]:


len(df_ap['mileage_appraisal'].unique()) # 21 categories in the appraisal milageb


# In[20]:


df_ap['mileage_appraisal'].value_counts()


# In[16]:


df_ap['mileage_appraisal']=df_ap['mileage_appraisal'].replace('0 to 10k miles','000 to 10k miles').replace('10k to 20k miles','010k to 20k miles')
#df_ap['appraisal_offer']=df_ap['appraisal_offer'].replace('$0k to $5k','$00k to $5k')


# In[17]:


df_ap['mileage_appraisal']=df_ap['mileage_appraisal'].replace('20k to 30k miles','020k to 030k miles').replace('30k to 40k miles','030k to 040k miles')


# In[18]:


df_ap['mileage_appraisal']=df_ap['mileage_appraisal'].replace('40k to 50k miles','040k to 050k miles').replace('50k to 60k miles','050k to 060k miles')


# In[19]:


df_ap['mileage_appraisal']=df_ap['mileage_appraisal'].replace('60k to 70k miles','060k to 070k miles').replace('70k to 80k miles','070k to 080k miles')


# In[20]:


df_ap['mileage_appraisal']=df_ap['mileage_appraisal'].replace('80k to 90k miles','080k to 090k miles').replace('90k to 100k miles','090k to 100k miles')


# In[21]:


cross_tab_of_ml_prop = pd.crosstab(index=df_ap['mileage_appraisal'],
                             columns=df_ap['appraisal_offer'],
                             normalize='index')


# In[27]:


cross_tab_of_ml_prop


# In[22]:


labels = ['0-5', '10-15', '15-20', '20-25','25-30','30-35','35-40','40+']
cross_tab_of_ml_prop.plot(kind='bar',stacked=True,colormap='tab10',figsize=(12,9))
plt.legend(labels,loc='upper center',ncol=9)
plt.xlabel('Mileage, 10k',fontsize=10)
plt.ylabel('Proportion',fontsize=10)
plt.title("Appraisal offered by Mileage")
plt.show()


# The above graph makes sense : the bigger the milage a car has the smaller offer it gets 

# ## Exploration of variables describing a purchased car

# In[23]:


pu_cols=[i for i in df.columns if 'appraisal' not in i]


# In[24]:


df_pu=df[pu_cols] # 15 columns i.e. half of the columns are about the appraised car


# In[25]:


df_pu.columns


# In[26]:


df_pu['price'].unique()


# In[27]:


df_pu['price']=df_pu['price'].replace('$00 to $15k','$00k to $15k')


# In[28]:


df_pu['price'].sort_values().value_counts()


# In[30]:


cross_tab_ml_yr_prop = pd.crosstab(index=df_pu['model_year'],
                             columns=df_pu['price'],
                                  normalize='index')


# In[31]:


cross_tab_ml_yr_prop


# In[32]:


labels2 = ['0-15', '15-20', '20-25','25-30','30-35','35-40','40-45','50-55','55-60','65-70','70+']

cross_tab_ml_yr_prop.plot(kind='bar',stacked=True,colormap='tab10',figsize=(18,9))
plt.legend(labels2,loc='upper center',ncol=9)
plt.xlabel('Model year',fontsize=10)
plt.ylabel('Proportion',fontsize=10)
plt.title("Price, $1k")
plt.show()


# We see that customers also buy older models (1991-2000 years) for very cheap prices (0-15k)

# ## Exploration of relationships between appraised car variables and a purchased car variables

# In[36]:


cols=list(set(df.columns)-set(['trim_descrip','trim_descrip_appraisal','body','body_appraisal']))
# we wont' use description columns


# In[100]:


len(cols)


# In[33]:


pu_cols_n=list(set(pu_cols)-set(['body']))
pu_cols_n


# In[37]:


ap_cols_n=list(set(ap_cols)-set(['body_appraisal']))
ap_cols_n


# In[38]:


corrmat=df[cols].corr()


# In[39]:


corrmat


# In[40]:


# returns highly correlated variables
high_c = corrmat[corrmat.abs()>0.45].index

# plots a correlation plot
f, ax = plt.subplots(figsize=(12, 9)) # determines the figure size
ax = sns.heatmap(corrmat.loc[high_c, high_c], vmax=.8, square=True, cmap="Blues") # creates a heatmap using the correlation matrixb


# In[41]:


pd.plotting.scatter_matrix(df[high_c],figsize=(15,15)) # this is very difficult to see


# In[84]:


pp=sns.pairplot(df[high_c])


# In[42]:


set1=ap_cols_n[0:5]+pu_cols_n[0:5]
g=sns.pairplot(df[set1],hue='price') # subsetting data to see relationships#


# In[5]:


set2=ap_cols_n[5:10]+pu_cols_n[5:10]
g2=sns.pairplot(df[set2]) # subsetting data to see relationships#


# In[118]:


pu_cols_n[0:5]


# In[120]:


ap_cols_n[0:5]


# In[93]:


len(pu_cols)


# In[94]:


ap_cols


# In[48]:


df['price'].sort_values().value_counts()


# In[51]:


df_cat=df[cols].select_dtypes(include=['object'])


# In[73]:


df_cat.columns


# ## Encoding categorical variables

# In[4]:


cat_pr=list(df['price'].unique()) # price categories of a purchased vehicle 


# In[5]:


categories_price=pd.Categorical(df['price'], categories=cat_pr.sort(), ordered=True)


# In[6]:


labels, unique= pd.factorize(categories_price, sort=True)
df['price_cat']=labels


# In[7]:


df.sort_values(by=['price_cat'])


# In[8]:


cat_ap_of=list(df['appraisal_offer'].unique()) # price categories of a purchased vehicle 


# In[9]:


df['appraisal_offer']=df['appraisal_offer'].replace('$5k to $10k','$05k to $10k')
df['appraisal_offer']=df['appraisal_offer'].replace('$0k to $5k','$00k to $5k')


# In[10]:


cat_ap_of.sort()


# In[11]:


categories_ap_of=pd.Categorical(df['appraisal_offer'], categories=cat_ap_of.sort(), ordered=True)
labels, unique= pd.factorize(categories_ap_of, sort=True)
df['appraisal_offer_cat']=labels


# In[12]:


df.sort_values(by=['appraisal_offer_cat']).tail(10)


# In[13]:


df['mileage']=['0'+i if i!='100k+ miles' else i for i in df['mileage']]


# In[14]:


df['mileage']=df['mileage'].replace('05k to 10k miles','005k to 10k miles')
df['mileage']=df['mileage'].replace('00 to 5k miles','000 to 5k miles')


# In[15]:


cat_mil=list(df['mileage'].unique()) # mileage categories of a purchased/appraised vehicle 


# In[16]:


cat_mil.sort()
cat_mil


# In[17]:


categories_mil=pd.Categorical(df['mileage'], categories=cat_mil.sort(), ordered=True)
labels, unique= pd.factorize(categories_mil, sort=True)
df['mileage_cat']=labels


# In[18]:


print(df['mileage_appraisal'].unique())


# In[19]:


l1=['0 to 10k miles', '10k to 20k miles','20k to 30k miles', '30k to 40k miles','40k to 50k miles','50k to 60k miles','60k to 70k miles'
   , '70k to 80k miles','80k to 90k miles','90k to 100k miles']


# In[20]:


l1


# In[21]:


df['mileage_appraisal']=['0'+i if i in l1 else i for i in df['mileage_appraisal']]


# In[22]:


df['mileage_appraisal']=df['mileage_appraisal'].replace('00 to 10k miles','000 to 10k miles')


# In[23]:


cat_mil_ap=list(df['mileage_appraisal'].unique()) # mileage categories of a purchased/appraised vehicle 
cat_mil_ap.sort()


# In[24]:


#df['mileage_appraisal']=df['mileage_appraisal'].replace('05k to 10k miles','005k to 10k miles')
#df['mileage_appraisal']=df['mileage_appraisal'].replace('00 to 5k miles','000 to 5k miles')
#cat_mil_ap=list(df['mileage_appraisal'].unique()) # mileage categories of a purchased/appraised vehicle 
#cat_mil_ap.sort()
categories_mil_ap=pd.Categorical(df['mileage_appraisal'], categories=cat_mil_ap.sort(), ordered=True)
labels, unique= pd.factorize(categories_mil_ap, sort=True)
df['mileage_appraisal_cat']=labels


# In[25]:


df['engine'].unique()


# In[26]:


df['engine']=[i.replace('L','') for i in df['engine']]


# In[27]:


df['engine']=df['engine'].astype('float')


# In[28]:


df['engine_appraisal']=[i.replace('L','') for i in df['engine_appraisal']]


# In[29]:


df['engine_appraisal']=df['engine_appraisal'].astype('float')


# In[30]:


df['engine_appraisal'].mean()


# In[31]:


df.columns


# In[32]:


cols=list(set(df.columns)-set(['trim_descrip','trim_descrip_appraisal','body','body_appraisal','price','appraisal_offer','mileage',
                             'mileage_appraisal']))


# In[326]:


##cols=list(set(df.columns)-set(['trim_descrip','trim_descrip_appraisal','body','body_appraisal','price','appraisal_offer','mileage',
                             'mileage_appraisal','color','color_appraisal','model','model_appraisal','make','make_appraisal' ]))


# In[33]:


df[cols].dtypes # make, model and color are still non numerics


# In[328]:


#cols_target=['price_cat','model_year','mileage_cat','horsepower','mpg_city','fuel_capacity','mpg_highway','engine','cylinders']


# In[36]:


df['color'].value_counts() # there are just 4 unknown new car colors which we can drop for simplicity


# We should replace unknown values with None 

# In[35]:


df['color']=df['color'].replace('Unknown',None)


# In[37]:


len(df['color'].value_counts())


# In[88]:


sns.set(style="white") # background color
ax = sns.barplot(x=df['color'].value_counts(normalize=True).values, 
                 y=df['color'].value_counts().index, 
                ) 
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
sns.despine(left=True,bottom=True)
plt.show()


# Even though we have 15 color categories we could encode just 5 the most frequnt colors and treat all omitted categories as other in our model.

# In[84]:


df['color_appraisal'].value_counts()


# In[81]:


df['color_appraisal']=df['color_appraisal'].replace('Unknown',None)


# In[85]:


df['color_appraisal'].value_counts(normalize=True)*100 


# In[86]:


sns.set(style="white") # background color
ax = sns.barplot(x=df['color_appraisal'].value_counts(normalize=True).values, 
                 y=df['color_appraisal'].value_counts().index, 
                ) 
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
sns.despine(left=True,bottom=True)
plt.show()


# We can create dummy variables jsut for Black, White, Gray, Silver and Blue colors which are the most frequent 

# In[94]:


df['Black']=[1 if i=='Black' else 0 for i in df['color']]


# In[99]:


df['White']=[1 if i=='White' else 0 for i in df['color']]
df['Gray']=[1 if i=='Gray' else 0 for i in df['color']]
df['Silver']=[1 if i=='Silver' else 0 for i in df['color']]
df['Blue']=[1 if i=='Blue' else 0 for i in df['color']]


# In[101]:


df['Black_appraisal']=[1 if i=='Black' else 0 for i in df['color_appraisal']]
df['White_appraisal']=[1 if i=='White' else 0 for i in df['color_appraisal']]
df['Gray_appraisal']=[1 if i=='Gray' else 0 for i in df['color_appraisal']]
df['Silver_appraisal']=[1 if i=='Silver' else 0 for i in df['color_appraisal']]
df['Blue_appraisal']=[1 if i=='Blue' else 0 for i in df['color_appraisal']]


# In[102]:


df


# In[ ]:


#df_new=pd.get_dummies(df_new, columns=['make','make_appraisal','color','color_appraisal','model','model_appraisal'], drop_first=True)


# In[106]:


df['make'].value_counts()[0:5]


# In[108]:


df['make'].value_counts(normalize=True)[0:5]*100


# In[103]:


sns.set(style="white") # background color
ax = sns.barplot(x=df['make'].value_counts(normalize=True).values, 
                 y=df['make'].value_counts().index, 
                ) 
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
sns.despine(left=True,bottom=True)
plt.show()


# In[105]:


sns.set(style="white") # background color
ax = sns.barplot(x=df['make_appraisal'].value_counts(normalize=True).values, 
                 y=df['make_appraisal'].value_counts().index, 
                ) 
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
sns.despine(left=True,bottom=True)
plt.show()


# In[109]:


df['make_appraisal'].value_counts()[0:5]


# In[110]:


df['make_appraisal'].value_counts(normalize=True)[0:5]*100


# In[111]:


df['make'].value_counts(normalize=True)[0:5]*100


# Even though we have many make categories we could encode just 5 the most frequnt ones and treat all omitted categories as other in our model.For both appraised car and a new car AIH,HXQ,KQZ,LTX and ARU are the msot frequent

# In[ ]:


df['Black_appraisal']=[1 if i=='Black' else 0 for i in df['color_appraisal']]
df['White_appraisal']=[1 if i=='White' else 0 for i in df['color_appraisal']]
df['Gray_appraisal']=[1 if i=='Gray' else 0 for i in df['color_appraisal']]
df['Silver_appraisal']=[1 if i=='Silver' else 0 for i in df['color_appraisal']]
df['Blue_appraisal']=[1 if i=='Blue' else 0 for i in df['color_appraisal']]


# In[114]:


fr=list(df['make_appraisal'].value_counts()[0:5].index)
fr


# In[122]:


for each in fr:
    df[each]=[1 if i==each else 0 for i in df['make']]


# In[130]:


for each in fr:
    df[each+'_appraisal']=[1 if i==each else 0 for i in df['make_appraisal']]


# In[133]:





# In[138]:


df['model'].value_counts(normalize=True)[0:5]*100 # there are too many low frequency categories


# In[139]:


df.columns


# In[156]:


cols=list(set(df.columns)-set(['trim_descrip','trim_descrip_appraisal','body','body_appraisal','price','appraisal_offer','mileage',
                             'mileage_appraisal','color','color_appraisal','make','make_appraisal','model','model_appraisal']))


# In[157]:


df_final=df[cols]


# In[158]:


df_final.dtypes


# In[159]:


len(df_final.columns)


# In[161]:


corrmat=df_final.corr()


# In[162]:


# returns highly correlated variables
high_c = corrmat[corrmat.abs()>0.5].index

# plots a correlation plot
f, ax = plt.subplots(figsize=(20, 20)) # determines the figure size
ax = sns.heatmap(corrmat.loc[high_c, high_c], vmax=.8, square=True, cmap="Blues") # creates a heatmap using the correlation matrixb


# ## Modeling features of the newly purchased vehicle

# In[200]:


df_final=df_final.dropna()


# In[202]:


input_var=[]
for i in df_final.columns:
    if 'appraisal' in i:
        input_var.append(i)


# In[203]:


X=df_final[input_var]


# In[204]:


target_vr=list(set(df_final.columns)-set(input_var))


# In[205]:


X.isnull().mean()*100 # number of missing values per column in %

Lets predict price of a newly purchased vehicle 
# ### Analysing price as our target variable

# In[206]:


print(df_final['price_cat'].describe())
print('\n')
ax = sns.distplot(df_final['price_cat']) #checking the distribution of the output variable


# In[207]:


Y=df_final['price_cat']


# In[210]:


sum(Y.isnull())


# In[211]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42) # testsize is what % of original dataset for testing
# random test referd to seeds


# In[189]:


from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix


# In[212]:


import scipy.stats as stats

from statsmodels.miscmodels.ordinal_model import OrderedModel


# In[214]:


mod_lg=OrderedModel(Y_train,X_train,distr='logit')
res_lg=mod_lg.fit(method='bfgs',disp=False)
res_lg.summary()


# In[190]:


# Decision Trees
dt = DecisionTreeClassifier(criterion = 'gini', splitter='best', max_depth=15) # max_depth is how high the tree is going to be
dt.fit(X_train, Y_train)


# In[193]:


X


# In[69]:


threshold = 10  # Remove items less than or equal to threshold
for col in df:
    vc = df['model'].value_counts()
    vals_to_remove = vc[vc <= threshold].index.values
    df['model'].loc[df['model'].isin(vals_to_remove)] = None


# In[70]:


print(df['model'].value_counts()) # we have many low frequency categories lets replace them with None


# In[71]:


print(df['model_appraisal'].value_counts()) # we have many low frequency categories lets replace them with None


# In[72]:


threshold = 10  # Remove items less than or equal to threshold
for col in df:
    vc = df['model_appraisal'].value_counts()
    vals_to_remove = vc[vc <= threshold].index.values
    df['model_appraisal'].loc[df['model_appraisal'].isin(vals_to_remove)] = None


# In[74]:


df['model_appraisal'].value_counts(normalize=True)*100 


# In[77]:


df_new=df[cols]


# In[78]:


df_new=pd.get_dummies(df_new, columns=['make','make_appraisal','color','color_appraisal','model','model_appraisal'], drop_first=True)


# In[82]:


df[cols].isnull().mean()*100 # number of missing values per column in %


# In[83]:


df['model_appraisal']


# In[80]:


df_new.isnull().mean()*100 # number of missing values per column in %


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




