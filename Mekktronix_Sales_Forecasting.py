
# coding: utf-8

# ## Import libraries

# In[84]:


import numpy as np
import pandas as pd


# ## Path to data

# In[85]:


PATH = 'dataset'
PATH_TO_train_data = PATH + '/' + 'yds_train2018.csv'
PATH_TO_test_data = PATH + '/' + 'yds_test2018.csv'
PATH_TO_promotional_expense = PATH + '/' + 'promotional_expense.csv'
PATH_TO_holidays = PATH + '/' + 'holidays.xlsx'


# In[86]:


train_data = pd.read_csv(PATH_TO_train_data)


# In[87]:


train_data.head()


# ## Drop Merchant_ID and S_No as there are not required.

# In[88]:


train_data.drop(columns=['Merchant_ID', 'S_No'], inplace=True)


# In[89]:


train_data.shape


# In[90]:


train_data.head()


# ## Group by (Year, Month, Product_ID and Country) and add Sales for each group.

# In[91]:


train_data = train_data.groupby(['Year', 'Month', 'Product_ID', 'Country']).Sales.sum().reset_index()


# In[92]:


train_data.shape


# In[93]:


train_data.head()


# ## Read Expense data and rename Product_ID

# In[94]:


promotional_expense_data = pd.read_csv(PATH_TO_promotional_expense)
promotional_expense_data.rename(columns={'Product_Type':'Product_ID'}, inplace=True)
promotional_expense_data.head()


# ## Merge train data and Promotional Expense data as the correlation between thwm is very high and it will be used to predict accurately.

# In[95]:


train_sales_and_expense_data = pd.merge(train_data, promotional_expense_data, on=['Year', 'Month', 'Country', 'Product_ID'])


# In[96]:


train_sales_and_expense_data.head()


# In[97]:


train_sales_and_expense_data['Sales'].corr(train_sales_and_expense_data['Expense_Price'])


# ## Convert Country data into categorical values/one hot representation.

# In[98]:


# train_sales_and_expense_data['Country_num'] = train_sales_and_expense_data.Country.map({'Argentina': 0, 'Belgium': 1, 'Columbia': 2, 'Denmark': 3, 'England': 4, 'Finland':5})
train_sales_and_expense_data = pd.get_dummies(train_sales_and_expense_data, columns=['Country'])


# In[99]:


# train_sales_and_expense_data.drop(columns='Country', inplace=True)


# In[100]:


train_sales_and_expense_data.head()


# In[101]:


X_train = train_sales_and_expense_data.drop(columns='Sales')


# In[102]:


X_train.head()


# In[103]:


X_train.shape


# In[104]:


y_train = train_sales_and_expense_data.Sales


# In[105]:


y_train.head()


# In[106]:


y_train.shape


# ## Modelling
# ### Tried different algorithms on the train data.

# In[107]:


from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost


# In[108]:


# svr = SVR(kernel='linear', C=1e3)
# model = svr.fit(X_train, y_train)


# In[109]:


# lreg = LinearRegression(normalize=True)
# model = lreg.fit(X_train, y_train)


# In[110]:


# ridgeReg = Ridge(alpha=1, normalize=True)
# model = ridgeReg.fit(X_train, y_train)


# In[111]:


# lassoReg = Lasso(alpha=0.01, normalize=True)
# model = lassoReg.fit(X_train,y_train)


# In[112]:


# ENreg = ElasticNet(alpha=1, l1_ratio=0.5, normalize=False)
# model = ENreg.fit(X_train,y_train)


# In[113]:


# dtReg = DecisionTreeRegressor()
# model = dtReg.fit(X_train, y_train)


# In[114]:


# rfReg = RandomForestRegressor()
# model = rfReg.fit(X_train, y_train)


# In[141]:


xgb = xgboost.XGBRegressor(n_estimators=1500, learning_rate=0.05, gamma=0, subsample=0.75,
                           colsample_bytree=1, max_depth=15)
model = xgb.fit(X_train, y_train)


# In[142]:


# svr = SVR(kernel='rbf', C=1, gamma=0.1)
# model = svr.fit(X_train, y_train)

# svr = SVR(kernel='poly', C=1e3, degree=2)
# model = svr.fit(X_train, y_train)


# In[143]:


model


# In[144]:


y_train_pred = model.predict(X_train)


# In[145]:


compare = pd.DataFrame({'y_train': y_train, 'y_train_pred': y_train_pred})
compare.head()


# ## Find correlation between y_train and predicted values on train data.

# In[146]:


compare['y_train'].corr(compare['y_train_pred'])


# In[147]:


l = (compare['y_train_pred'] - compare['y_train']).abs()
l.count()


# ## Fins SMAPE score for y_train and predicted value.

# In[150]:


num = (compare['y_train'] - compare['y_train_pred']).abs()
den = (compare['y_train'] + compare['y_train_pred'].abs())
count = num.count()
out = (num/den).sum()
SMAPE = (out/count)*100
SMAPE


# ## Preprocess the test data.
# ### Merge expense data with test data.
# ### Convert Country column into categorical values/one hot representation.

# In[151]:


promotional_expense_data.head()


# In[152]:


test_data = pd.read_csv(PATH_TO_test_data)
test_data.drop(['S_No', 'Sales'], axis=1, inplace=True)
test_sales_and_expense_data = pd.merge(test_data, promotional_expense_data, on=['Year', 'Month', 'Country', 'Product_ID'], how='left')

# test_sales_and_expense_data['Country_num'] = test_sales_and_expense_data.Country.map({'Argentina': 0, 'Belgium': 1, 'Columbia': 2, 'Denmark': 3, 'England': 4, 'Finland':5})
# test_sales_and_expense_data.drop(columns='Country', inplace=True)
test_sales_and_expense_data = pd.get_dummies(test_sales_and_expense_data, columns=['Country'])
X_test = test_sales_and_expense_data
X_test = X_test.fillna(0)
X_test.head()


# In[153]:


X_test.shape


# In[154]:


y_predict = model.predict(X_test)


# In[155]:


y_predict.shape


# In[156]:


y_predict


# In[157]:


y_predict_df = pd.DataFrame({'Sales': y_predict})


# In[158]:


y_predict_df.tail()


# In[159]:


output_df = pd.read_csv(PATH_TO_test_data)
s_no = output_df.S_No
output_df.drop('S_No', axis=1, inplace=True)


# In[160]:


output_df.head()


# In[161]:


output_df.Sales = y_predict_df.abs()


# In[162]:


output_df.head()


# In[163]:


output_df = pd.concat([s_no, output_df], axis=1)


# ## Save data in a csv file.

# In[164]:


output_df.to_csv('yds_submission2018.csv', index=False)


# In[165]:


output_df.shape


# In[166]:


output_df.head()

