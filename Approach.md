How to run:
1. Run the whole Jupyter notebook(Mekktronix_Sales_Forecasting.ipynb). This will produce the output file.
2. There is also another file(Arima.ipynb), which contains the implementation of Arima country wise.

Approach:


I) Mekktronix Sales Forecating:

1. Grouped train data by (Year, Month, Product id, Country) as week is not important. Test set only depends on the month.
2. So, sales grouped by the above features are added and stored.
3. Converted country into categorical values(0-5). Also tried with one hot vector representation of countries.
4. Merged Expense_Price data with train data using (Year, Month, Product id, Country) features and assigned zero to the other values.
5. Sales is used as target value and all the other values are used as train features.

Model:
Tried 7 algorithms using this approach. xgBoost worked well for predicting 'Sales'(output variable).

II) Arima:

1. Grouped the data based on Country and Product_ID columns. There are around 11 unique combination of values.
2. Applied Arima on each combination. Tried different values of p, d, q values. It considers the best set values after trying different combinations of pdq values.
3. Stored the next 36 Sales points for all combinations of Country and Product_ID.
3. Predicted Sales on the test data.

Tools: Python
IDE: Jupyter notebook.
