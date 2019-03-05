#!/usr/bin/env python
# coding: utf-8

# <img src="images/celo.png" style="width:1000px;height:150px;">
# 
# 
# This was a Kaggle competition, which aimed to develop machine learning algorithms to identify and serve the most relevant opportunities for individuals, revealing a sign of customer loyalty. Improve customer’s experience by proposing strategies to help Elo reduce unwanted campaigns to groups of customers is also explored. (https://www.kaggle.com/c/elo-merchant-category-recommendation ). 
# 
# In this report/code a very detailed solution is presented. Tools used are presented and key strategic data analysis functions are explained. The solution has some functions/strategies also used by Chau Ngoc Huynh’s kernel. The intention is to make the explanation more accessible to people who are starting to study Data Science / Machine Learning. We believe that through this file some people can learn a bit how to manipulate data, start a contact with one Machine Learning model, and get used to search the libraries and the methods used.

# In[2]:


import numpy as np
import pandas as pd
import datetime
import gc
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle
import warnings
warnings.filterwarnings('ignore')
np.random.seed(4590)


# ### If you are interested in knowing a bit more about each of the libraries and modules used, please check summary below:
# 
# <img src="images/introduction.png" style="width:1000px;height:650px;">
# 
# 
# 
# ### Links/References available at :
# - [NumPy](https://docs.scipy.org/doc/numpy-1.12.0/reference/index.html)
# - [Pandas](pandas.pydata.org/pandas-docs/stable/) 
# - [datetime](https://docs.python.org/3/library/datetime.html#module-datetime)
# - [gc](https://docs.python.org/2/library/gc.html)
# - [matplotlib.pyplot](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.html#module-matplotlib.pyplot)
# - [Seaborn](https://seaborn.pydata.org/) 
# - [LightGBM](https://lightgbm.readthedocs.io/en/latest/genindex.html)
# - [Scikit-learn](https://scikit-learn.org/stable/)
# - [StratifiedKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html), 
# - [Mean Squared Error](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html). 

# **Converting "Comma Separated Values" (CSV) file into a DataFrame:**

# In[3]:


df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
df_hist_trans = pd.read_csv('historical_transactions.csv')
df_new_merchant_trans = pd.read_csv('new_merchant_transactions.csv')


# **Applying `df.head(n)`, which will return the first "n" rows. `IPython.display` is used in order to make possible to see all information in only one cell:**

# In[3]:


display(df_train.head(2))
display(df_hist_trans.head(2))
display(df_test.head(2))
display(df_new_merchant_trans.head(2))


# **The `df.info( )` method is useful to get a data description from a data frame, in particular, the total number of rows, each attribute’s type and number of non-null values:**

# In[4]:


display(df_train.info())
display(df_hist_trans.info())
display(df_test.info())
display(df_new_merchant_trans.info())


# **At this point, an analysis over missing values in each data frame is performed. An anonymous function, also known as $\lambda $ *(lambda)* functions was elaborated to point it out the quantity of null values in each data set. The syntax of Lambda function in python is `lambda arguments: expression`. In our case, we applied:**
# 
# ```python
# df.apply(lambda x: sum(x.isnull()), axis=0)
# ```
# 
# **Ps. It is possible to use different methods to measure null values (missing values) in a data frame, which means that it is not necessary to create a *lambda* function for that. However, in our study we did elaborated that mainly because we are aiming to have a simple and understandable code.**

# In[5]:


dic = {'df_train': df_train, 'df_test': df_test, 'df_hist_trans': df_hist_trans, 'df_new_merchant_trans': df_new_merchant_trans }
for name, df in dic.items():
    print("-------" + name + "-------")
    print(df.apply(lambda x: sum(x.isnull()), axis=0)) #axis=0 define que a função deve ser aplicada em cada coluna


# **In the data frame `df_test`, only one value was lost, located in the "first_active_month" column. Approach followed in this case was remove the row that have this missing value. For more information regarding function used to do that, please see:** [pandas.DataFrame.dropna](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.dropna.html).

# In[4]:


# drop rows with missing values
df_test.dropna(inplace=True)


# **For the other data frames (that presented a significant amount of missing values per data set), replace the missing value followed a different approach. The mode (most frequent value in a range) of each collum took the place of the missing values. In order to do that, it is first necessary to know the most frequent value in each case, `df[columns].mode()` was used and results are presented below:**

# In[6]:


dic = {'df_hist_trans': df_hist_trans, 'df_new_merchant_trans': df_new_merchant_trans}
for name, df in dic.items():
    for columns in ('category_2', 'category_3','merchant_id' ):
        print('The value that appears most often in column ' + columns + ' of the ' + name)
        print(df[columns].mode())


# **Now, `fillna` method is applied to fill up the columns. In order to do that, the mode and data set have to be mention in the code:**

# In[5]:


for df in [df_hist_trans,df_new_merchant_trans]:
    df['category_2'].fillna(1.0,inplace=True)
    df['category_3'].fillna('A',inplace=True)
    df['merchant_id'].fillna('M_ID_00a6ca8a8a',inplace=True)


# **An analysis over the "target" column in the `df_train` dataframe is performed. The aim of this analysis is to verify the existence of outliers. A simple method to do that is plot its histogram, in order to do that we used `plot.hist`:**

# In[7]:


df_train['target'].plot.hist(grid=True, bins=20, rwidth=0.9, color='k')
plt.title('Target Frequency')
plt.xlabel('Target')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.75)


# It was possible to see that the interval $[-30,-18]$ does not contain any value. On the other hand, it was observed values up to 18. Therefore, it is possible to justify that any other value less than -18 is an outlier and furthermore, a collum is created (outlier collum) and those values are set as 1.
# 
# 
# Analysis can be check using the piece of code below:
# 
# ```python
# 
# In [] len([1 for i in df_train['target'] if i>-30 and i<-18])
# Out[] 0
# In []len([1 for i in df_train['target'] if i>18]) 
# Out[] 0 
# ```

# In[6]:


df_train['outliers'] = 0
df_train.loc[df_train['target'] < -18, 'outliers'] = 1
df_train['outliers'].value_counts()


# **Features 1, 2 and 3 are not well described in the data set provided by the company. We only know that they are categorical variables defined by integer numbers and they are respectively $[1,5]$, $[1,3]$ and $[0,1]$. Therefore, one interesting thing to do at this stage is to add these "categorical" information for a more relevante variable (or at least give them a weight). Therefore, the following pice of code does perform a group of those features (by its mean) and add each of the three collums at train and test data frame with this new value.**  

# In[7]:


for f in ['feature_1','feature_2','feature_3']:
    order_label = df_train.groupby([f])['target'].mean()
    df_train[f+'_groupby_target'] = df_train[f].map(order_label)
    df_test[f+'_groupby_target'] = df_test[f].map(order_label)


# **Feature Eng. is now performed. First, data/time features are analyzed and created. In the `df_train` and `df_test`, the "first active month" feature is presented as month/year. Therefore, two new collums are added (month and year). On the other hand, for `df_hist_trans` and `df_new_merchant_trans` the "purchase date" feature is a full date/time cell and six new collums are created, information such as year, weekend and hour is now presented separately from others:**

# In[8]:


for df in [df_train,df_test]:
    df['first_active_month'] =  pd.to_datetime(df['first_active_month'])
    df['activation_month'] = df['first_active_month'].dt.month
    df['activation_year'] = df['first_active_month'].dt.year
    del df['first_active_month']
    
for df in [df_hist_trans,df_new_merchant_trans]:
    df['purchase_date'] = pd.to_datetime(df['purchase_date']) 
    df['year_of_purchase'] = df['purchase_date'].dt.year
    df['month_of_purchase'] = df['purchase_date'].dt.month
    df['week_of_year_purchase'] = df['purchase_date'].dt.weekofyear
    df['day_of_week_purchase'] = df['purchase_date'].dt.dayofweek
    df['weekend_purchase'] = (df['purchase_date'].dt.weekday >=5).astype(int)
    df['hour_of_purchase'] = df['purchase_date'].dt.hour
    df['purchase_date'] = df['purchase_date'].dt.dayofyear


# **Other categorical features, such as "authorized flag" and "category 1" that has a nominal observation as an informatios ("yes" and "no") are convert to numerical data ("0" and "1"), and feture "cagegory 3", which has nominal values ("A","B" and "C") will be replaced respectively by values "1, 2 and 3":**

# In[9]:


for df in [df_hist_trans,df_new_merchant_trans]:
    df['authorized_flag'] = df['authorized_flag'].map({'Y':1, 'N':0})
    df['category_1'] = df['category_1'].map({'Y':1, 'N':0})
    df['category_3'] = df['category_3'].map({'A':1, 'B':2, 'C':3})


# **The next two pieces of code intends to create more features through of some features that we have. The idea below was based on ideas presented on kaggle's kernel [My first kernel](https://www.kaggle.com/chauhuynh/my-first-kernel-3-699 ).**
# 
# 1. Step
# Create a function that the arggument is a dictionary to create feature names
# 
# 2. Step
# Define new features
# 
# 3. Step
# Create more features based on features that we just defined
# 
# 4. Step
# Merge all the dataframes (df_hist_trans,df_new_merchant_trans) into test and train (df_train and df_test)
# 
# 5. Step
# Delete the dataframes that will not be use (df_hist_trans,df_new_merchant_trans) 

# In[10]:


def get_new_columns(dic):
    return [k + '_' + value for k in dic.keys() for value in dic[k]]


# In[11]:


dic = {}

dic['purchase_amount'] = ['sum','max','min','mean','var']
dic['installments'] = ['mean','var']
dic['month_lag'] = ['max','min','mean','var']
dic['card_id'] = ['size']

new_columns = get_new_columns(dic)
new_columns.insert(0, 'card_id')
#     df_group = df
#     df_group = df_hist_group.groupby('card_id').agg(dic)
#     df_group.columns = new_columns

#     df_group.reset_index(drop=False,inplace=True)   
#     df = df.merge(df_group,on='card_id',how='left')
#     del df['purchase_date']
#     df_train = df_train.merge(df,on='card_id',how='left')
#     df_test = df_test.merge(df,on='card_id',how='left')
#     del df;gc.collect()


# In[12]:


df_temp = df_hist_trans.groupby('card_id').agg(dic)
df_temp.reset_index(level=0, inplace=True)
df_temp.columns = new_columns
df_temp = df_hist_trans.merge(df_temp,on='card_id',how='left')
df_train = df_train.merge(df_temp,on='card_id',how='left')
df_test = df_test.merge(df_temp,on='card_id',how='left')
del df_temp ;gc.collect()


# In[ ]:


df_temp = df_new_merchant_trans.groupby('card_id').agg(dic)
df_temp.reset_index(level=0, inplace=True)
df_temp.columns = new_columns
df_temp = df_new_merchant_trans.merge(df_temp,on='card_id',how='left')
df_train = df_train.merge(df_temp,on='card_id',how='left')
df_test = df_test.merge(df_temp,on='card_id',how='left')
del df_temp ;gc.collect()


# **In the following lines of code, target (as a feature) is removed from train data set:**

# In[ ]:


target = df_train['target']
del df_train['target']


# **The following pieces of code does present the definition of a correlation matrix between variables. And all variables that present correlation over than 0.9 is removed from train and test data set. After this process the matrix is plotted one more time to verify that process work it out:**

# In[ ]:


fig, (axis1) = plt.subplots(1,1,figsize=(20,15))
# df.corr() Compute pairwise correlation of columns, excluding NA/null values.
# sns.heatmap() Plot rectangular data as a color-encoded matrix.
sns.heatmap(df_train.corr(),annot=True,linewidths=0.001,cmap="YlGnBu")


# In[ ]:


def feature_correlated(df_in, threshold):
   df_corr = df_in.corr(method='pearson', min_periods=1)

   # np.ones() Return a new array of given shape and type, filled with ones.
   # np.tril() Return a copy of an array with elements above the diagonal equal False.
   # df.mask() Replace values where the condition is False.
    
   serie_not_correlated = ~(df_corr.mask(np.tril(np.ones(df_corr.shape, dtype=bool))).abs() > threshold).any()
   column_corr_idx = serie_not_correlated.loc[serie_not_correlated == True].index
   df_out = df_in[column_corr_idx]
   return df_out


# In[ ]:


df_train = feature_correlated(df_train, 0.9)
df_train_columns = [c for c in df_train.columns if c not in ['card_id','target','outliers']]


# In[ ]:


fig, (axis1) = plt.subplots(1,1,figsize=(20,15))
sns.heatmap(df_train.corr(),annot=True,linewidths=1,cmap="YlGnBu")


# **`LightGBM` is the Model selected to train and apply ML at the test data set. It is supervised learning recommended to problems that has large data sets.**
# 
# **If `LightGBM` is new for you and you would like to know about it one good reference is the article [What is LightGBM](https://medium.com/@pushkarmandot/https-medium-com-pushkarmandot-what-is-lightgbm-how-to-implement-it-how-to-fine-tune-the-parameters-60347819b7fc) by Pushkar Mandot.**

# In[ ]:


# Parameters https://lightgbm.readthedocs.io/en/latest/Parameters.html
param = {'num_leaves': 31,                # defauld = 31
         'min_data_in_leaf': 100, 
         'objective':'regression',
         'max_depth': -1,                 # default = -1, This parameter is used to handle model overfitting
         'learning_rate': 0.01,           # The learning parameter 
         "min_child_samples": 20,
         "boosting": "gbdt",              #  gbdt: traditional Gradient Boosting Decision Tree
         "feature_fraction": 0.9,         #  LightGBM will select 90% of parameters randomly in each iteration for building trees.
         "bagging_freq": 1,
         "bagging_fraction": 0.9 ,        # 90% of data will be used for each iteration(It is used to speed up the training and avoid overfitting.)
         "bagging_seed": 11,
         "max_cat_group": 64
         "metric": 'mse',                 # mean squared error
         "lambda_l1": 0.1,
         "verbosity": -1,
         "nthread": 4,
         "random_state": 42}
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  # Provides train/test indices to split data in train/test sets.
oof = np.zeros(len(df_train))
predictions = np.zeros(len(df_test))
feature_importance_df = pd.DataFrame()
  
for fold_, (trn_idx, val_idx) in enumerate(folds.split(df_train,df_train['outliers'].values)):
    print("fold {}".format(fold_))
    trn_data = lgb.Dataset(df_train.iloc[trn_idx][df_train_columns], label=target.iloc[trn_idx])#, categorical_feature=categorical_feats)
    val_data = lgb.Dataset(df_train.iloc[val_idx][df_train_columns], label=target.iloc[val_idx])#, categorical_feature=categorical_feats)

    num_round = 10000
    clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=100, early_stopping_rounds = 100)
    oof[val_idx] = clf.predict(df_train.iloc[val_idx][df_train_columns], num_iteration=clf.best_iteration)
    
    fold_importance_df = pd.DataFrame()
    fold_importance_df["Feature"] = df_train_columns
    fold_importance_df["importance"] = clf.feature_importance()
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    
    predictions += clf.predict(df_test[df_train_columns], num_iteration=clf.best_iteration) / folds.n_splits

np.sqrt(mean_squared_error(oof, target))


# In[ ]:




