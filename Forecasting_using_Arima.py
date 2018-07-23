
# coding: utf-8

# ## Import libraries

# In[378]:


import numpy as np
import pandas as pd
import statsmodels as sm
import statsmodels.tsa.api as smt
import statsmodels.formula.api as smf
import itertools


# In[379]:


import statsmodels


# In[380]:


statsmodels.__version__


# ## Path to data files

# In[381]:


PATH = 'dataset'
PATH_TO_train_data = PATH + '/' + 'yds_train2018.csv'
PATH_TO_test_data = PATH + '/' + 'yds_test2018.csv'
PATH_TO_promotional_expense = PATH + '/' + 'promotional_expense.csv'
PATH_TO_holidays = PATH + '/' + 'holidays.xlsx'


# In[382]:


train_data = pd.read_csv(PATH_TO_train_data)


# ## Drop Merchant_ID and S_No as they are niot required.

# In[383]:


train_data.drop(columns=['Merchant_ID', 'S_No'], inplace=True)


# In[384]:


train_data = train_data.groupby(['Year', 'Month', 'Product_ID', 'Country']).Sales.sum().reset_index()


# In[385]:


train_data.head()


# ## Find unique combinations of Country and Product_ID

# In[386]:


find_unique_country_pro = pd.DataFrame({'Country': train_data.Country, 'Product_ID': train_data.Product_ID})


# In[387]:


find_unique_country_pro.head()


# In[388]:


unique = find_unique_country_pro.drop_duplicates()
unique.sort_values(['Country'])


# ## Function for ARIMA model.
# ### Takes country, id and series data as input.
# ### It tries different values of pdq values and takes th best value out of them.
# ### Using those values, it predicts the next 36 data points.
# ### There are 11 different combinations. Apply sarimax on all combinations.

# In[389]:


import sys
warnings.filterwarnings("ignore") # specify to ignore warning messages

def sarimax(series_data, country, id):
    # Define the p, d and q parameters to take any value between 0 and 2
    p = d = q = range(0, 4)

    # Generate all different combinations of p, d and q triplets
    pdq = list(itertools.product(p, d, q))

    # Generate all different combinations of seasonal p, q and q triplets
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
    
    
    best_aic = np.inf
    best_pdq = None
    best_seasonal_pdq = None
    temp_model = None

    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                temp_model = sm.tsa.statespace.sarimax.SARIMAX(series_data, order=param, seasonal_order=param_seasonal)
                results = temp_model.fit()
                if results.aic < best_aic:
                    best_aic = results.aic
                    best_pdq = param
                    best_seasonal_pdq = param_seasonal
            except:
                continue
    best_model = sm.tsa.statespace.sarimax.SARIMAX(series_data,
                                      order=best_pdq,
                                      seasonal_order=best_seasonal_pdq, enforce_invertibility=False)
    best_results = best_model.fit()
    n_steps = 36
    pred_uc_99 = best_results.get_forecast(steps=36, alpha=0.01)
    df = pred_uc_99.conf_int()
    
    df['forcast'] = df['lower Sales'] * 0.5 + df['upper Sales'] * 0.5
    df['Country'] = country
    df['Product_ID'] = id
    
    return best_pdq, best_seasonal_pdq, df
    print("Best SARIMAX{}x{}12 model - AIC:{}".format(best_pdq, best_seasonal_pdq, best_aic))


# In[390]:


df.head()


# In[391]:


df.shape


# In[392]:


x = pd.concat([df, df]).shape


# In[393]:


argentina_1 = train_data.loc[train_data.Country == 'Argentina', :]
argentina_1 = argentina_1.sort_values(['Year', 'Month']).loc[argentina_1.Product_ID == 1]
argentina_1['Day'] = 1
argentina_1['Date'] = pd.to_datetime(argentina_1[['Year', 'Month', 'Day']])
argentina_1.head()

sales = argentina_1.Sales
sales.index = range(sales.count())
date = argentina_1.Date
date.index = range(date.count())
argentina_1_series = pd.DataFrame({'Date': date, 'Sales': sales})
argentina_1_series.head()

argentina_1_series.set_index('Date', inplace=True)

argentina_1_series = pd.Series(argentina_1_series.Sales, index=argentina_1_series.index)
argentina_1_series.head()


# In[394]:


best_pdq, best_seasonal_pdq, df = sarimax(argentina_1_series, 'Argentina', 1)
main_df = df


# In[395]:


main_df.shape


# In[396]:


argentina_2 = train_data.loc[train_data.Country == 'Argentina', :]
argentina_2 = argentina_2.sort_values(['Year', 'Month']).loc[argentina_2.Product_ID == 2]
argentina_2['Day'] = 1
argentina_2['Date'] = pd.to_datetime(argentina_2[['Year', 'Month', 'Day']])
argentina_2.head()

sales = argentina_2.Sales
sales.index = range(sales.count())
date = argentina_2.Date
date.index = range(date.count())
argentina_2_series = pd.DataFrame({'Date': date, 'Sales': sales})
argentina_2_series.head()

argentina_2_series.set_index('Date', inplace=True)

argentina_2_series = pd.Series(argentina_2_series.Sales, index=argentina_2_series.index)
argentina_2_series.head()


# In[397]:


best_pdq, best_seasonal_pdq, df = sarimax(argentina_2_series, 'Argentina', 2)


# In[398]:


main_df = pd.concat([main_df, df])


# In[399]:


main_df.shape


# In[400]:


argentina_3 = train_data.loc[train_data.Country == 'Argentina', :]
argentina_3 = argentina_3.sort_values(['Year', 'Month']).loc[argentina_3.Product_ID == 3]
argentina_3['Day'] = 1
argentina_3['Date'] = pd.to_datetime(argentina_3[['Year', 'Month', 'Day']])
argentina_3.head()

sales = argentina_3.Sales
sales.index = range(sales.count())
date = argentina_3.Date
date.index = range(date.count())
argentina_3_series = pd.DataFrame({'Date': date, 'Sales': sales})
argentina_3_series.head()

argentina_3_series.set_index('Date', inplace=True)

argentina_3_series = pd.Series(argentina_3_series.Sales, index=argentina_3_series.index)
argentina_3_series.head()


# In[401]:


best_pdq, best_seasonal_pdq, df = sarimax(argentina_3_series, 'Argentina', 3)


# In[402]:


main_df = pd.concat([main_df, df])


# In[403]:


main_df.shape


# In[404]:


belgium_2 = train_data.loc[train_data.Country == 'Belgium', :]
belgium_2 = belgium_2.sort_values(['Year', 'Month']).loc[belgium_2.Product_ID == 2]
belgium_2['Day'] = 1
belgium_2['Date'] = pd.to_datetime(belgium_2[['Year', 'Month', 'Day']])
belgium_2.head()

sales = belgium_2.Sales
sales.index = range(sales.count())
date = belgium_2.Date
date.index = range(date.count())
belgium_2_series = pd.DataFrame({'Date': date, 'Sales': sales})
belgium_2_series.head()

belgium_2_series.set_index('Date', inplace=True)

belgium_2_series = pd.Series(belgium_2_series.Sales, index=belgium_2_series.index)
belgium_2_series.head()


# In[405]:


best_pdq, best_seasonal_pdq, df = sarimax(belgium_2_series, 'Belgium', 2)


# In[406]:


main_df = pd.concat([main_df, df])


# In[407]:


main_df.shape


# In[408]:


columbia_1 = train_data.loc[train_data.Country == 'Columbia', :]
columbia_1 = columbia_1.sort_values(['Year', 'Month']).loc[columbia_1.Product_ID == 1]
columbia_1['Day'] = 1
columbia_1['Date'] = pd.to_datetime(columbia_1[['Year', 'Month', 'Day']])
columbia_1.head()

sales = columbia_1.Sales
sales.index = range(sales.count())
date = columbia_1.Date
date.index = range(date.count())
columbia_1_series = pd.DataFrame({'Date': date, 'Sales': sales})
columbia_1_series.head()

columbia_1_series.set_index('Date', inplace=True)

columbia_1_series = pd.Series(columbia_1_series.Sales, index=columbia_1_series.index)
columbia_1_series.head()


# In[409]:


best_pdq, best_seasonal_pdq, df = sarimax(columbia_1_series, 'Columbia', 1)


# In[249]:


main_df = pd.concat([main_df, df])


# In[250]:


main_df.shape


# In[251]:


columbia_2 = train_data.loc[train_data.Country == 'Columbia', :]
columbia_2 = columbia_2.sort_values(['Year', 'Month']).loc[columbia_2.Product_ID == 2]
columbia_2['Day'] = 1
columbia_2['Date'] = pd.to_datetime(columbia_2[['Year', 'Month', 'Day']])
columbia_2.head()

sales = columbia_2.Sales
sales.index = range(sales.count())
date = columbia_2.Date
date.index = range(date.count())
columbia_2_series = pd.DataFrame({'Date': date, 'Sales': sales})
columbia_2_series.head()

columbia_2_series.set_index('Date', inplace=True)

columbia_2_series = pd.Series(columbia_2_series.Sales, index=columbia_2_series.index)
columbia_2_series.head()


# In[252]:


best_pdq, best_seasonal_pdq, df = sarimax(columbia_2_series, 'Columbia', 2)


# In[253]:


main_df = pd.concat([main_df, df])


# In[254]:


main_df.shape


# In[255]:


columbia_3 = train_data.loc[train_data.Country == 'Columbia', :]
columbia_3 = columbia_3.sort_values(['Year', 'Month']).loc[columbia_3.Product_ID == 3]
columbia_3['Day'] = 1
columbia_3['Date'] = pd.to_datetime(columbia_3[['Year', 'Month', 'Day']])
columbia_3.head()

sales = columbia_3.Sales
sales.index = range(sales.count())
date = columbia_3.Date
date.index = range(date.count())
columbia_3_series = pd.DataFrame({'Date': date, 'Sales': sales})
columbia_3_series.head()

columbia_3_series.set_index('Date', inplace=True)

columbia_3_series = pd.Series(columbia_3_series.Sales, index=columbia_3_series.index)
columbia_3_series.head()


# In[256]:


best_pdq, best_seasonal_pdq, df = sarimax(columbia_3_series, 'Columbia', 3)


# In[257]:


main_df = pd.concat([main_df, df])


# In[258]:


main_df.shape


# In[259]:


denmark_2 = train_data.loc[train_data.Country == 'Denmark', :]
denmark_2 = denmark_2.sort_values(['Year', 'Month']).loc[denmark_2.Product_ID == 2]
denmark_2['Day'] = 1
denmark_2['Date'] = pd.to_datetime(denmark_2[['Year', 'Month', 'Day']])
denmark_2.head()

sales = denmark_2.Sales
sales.index = range(sales.count())
date = denmark_2.Date
date.index = range(date.count())
denmark_2_series = pd.DataFrame({'Date': date, 'Sales': sales})
denmark_2_series.head()

denmark_2_series.set_index('Date', inplace=True)

denmark_2_series = pd.Series(denmark_2_series.Sales, index=denmark_2_series.index)
denmark_2_series.head()


# In[260]:


best_pdq, best_seasonal_pdq, df = sarimax(denmark_2_series, 'Denmark', 2)


# In[261]:


main_df = pd.concat([main_df, df])


# In[262]:


main_df.shape


# In[263]:


england_4 = train_data.loc[train_data.Country == 'England', :]
england_4 = england_4.sort_values(['Year', 'Month']).loc[england_4.Product_ID == 4]
england_4['Day'] = 1
england_4['Date'] = pd.to_datetime(england_4[['Year', 'Month', 'Day']])
england_4.head()

sales = england_4.Sales
sales.index = range(sales.count())
date = england_4.Date
date.index = range(date.count())
england_4_series = pd.DataFrame({'Date': date, 'Sales': sales})
england_4_series.head()

england_4_series.set_index('Date', inplace=True)

england_4_series = pd.Series(england_4_series.Sales, index=england_4_series.index)
england_4_series.head()


# In[264]:


best_pdq, best_seasonal_pdq, df = sarimax(england_4_series, 'England', 4)


# In[265]:


main_df = pd.concat([main_df, df])


# In[266]:


main_df.shape


# In[267]:


england_5 = train_data.loc[train_data.Country == 'England', :]
england_5 = england_5.sort_values(['Year', 'Month']).loc[england_5.Product_ID == 5]
england_5['Day'] = 1
england_5['Date'] = pd.to_datetime(england_5[['Year', 'Month', 'Day']])
england_5.head()

sales = england_5.Sales
sales.index = range(sales.count())
date = england_5.Date
date.index = range(date.count())
england_5_series = pd.DataFrame({'Date': date, 'Sales': sales})
england_5_series.head()

england_5_series.set_index('Date', inplace=True)

england_5_series = pd.Series(england_5_series.Sales, index=england_5_series.index)
england_5_series.head()


# In[268]:


best_pdq, best_seasonal_pdq, df = sarimax(england_5_series, 'England', 5)


# In[269]:


main_df = pd.concat([main_df, df])


# In[270]:


main_df.shape


# In[271]:


finland_4 = train_data.loc[train_data.Country == 'Finland', :]
finland_4 = finland_4.sort_values(['Year', 'Month']).loc[finland_4.Product_ID == 4]
finland_4['Day'] = 1
finland_4['Date'] = pd.to_datetime(finland_4[['Year', 'Month', 'Day']])
finland_4.head()

sales = finland_4.Sales
sales.index = range(sales.count())
date = finland_4.Date
date.index = range(date.count())
finland_4_series = pd.DataFrame({'Date': date, 'Sales': sales})

finland_4_series.set_index('Date', inplace=True)

finland_4_series = pd.Series(finland_4_series.Sales, index=finland_4_series.index)
finland_4_series.head()


# In[272]:


best_pdq, best_seasonal_pdq, df = sarimax(finland_4_series, 'Finland', 4)


# In[273]:


main_df = pd.concat([main_df, df])


# ## 'main_df' contains all the predicted values for all the combinations of Country and Product_ID

# In[274]:


main_df.shape


# In[276]:


# main_df.to_csv('main_df.csv')


# ## Read test data and preprocess.

# In[281]:


test_data = pd.read_csv(PATH_TO_test_data)
test_data.drop(['S_No', 'Sales'], axis=1, inplace=True)
test_data['Day'] = 1
test_data['Date'] = pd.to_datetime(test_data[['Year', 'Month', 'Day']])


# In[283]:


test_data.head()


# In[358]:


duplicate_test_data = test_data


# In[359]:


duplicate_test_data.head()


# In[360]:


duplicate_test_data.shape


# In[361]:


duplicate_main_df = main_df


# In[362]:


duplicate_main_df = duplicate_main_df.reset_index()
duplicate_main_df.rename(columns = {'index': 'Date'}, inplace=True)


# In[363]:


duplicate_main_df.head()


# ## Merge test data and main dataframe.

# In[364]:


duplicate_test_data = pd.merge(duplicate_test_data, duplicate_main_df, on=['Product_ID', 'Country', 'Date'], how='left')


# In[365]:


duplicate_test_data.head()


# In[366]:


duplicate_test_data.drop(columns=['Day', 'Date', 'lower Sales', 'upper Sales'], inplace=True)


# In[367]:


duplicate_test_data.head()


# In[368]:


duplicate_test_data.shape


# In[369]:


duplicate_test_data = duplicate_test_data.rename(columns={'forcast': 'Sales'})


# In[370]:


duplicate_test_data.head()


# In[371]:


test_data_s_no = pd.read_csv(PATH_TO_test_data)


# In[372]:


test_data_s_no.head()


# In[373]:


duplicate_test_data['S_No'] = test_data_s_no.S_No


# In[374]:


duplicate_test_data['Sales'] = duplicate_test_data['Sales'].abs()


# In[375]:


duplicate_test_data = duplicate_test_data[['S_No', 'Year', 'Month', 'Product_ID', 'Country', 'Sales']]


# In[376]:


duplicate_test_data.head()


# In[377]:


duplicate_test_data.to_csv('yds_submission2018.csv', index=False)

