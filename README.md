




Cleaning script explanation: 
What This Script Does

This Python script cleans and transforms stock data stored in an Excel file. It is specifically made for the dataset located at: dataset/stocks_clean2.xlsx (sheet: 'ark').

How It Works

1. Reads the Excel file
   The script loads the stock dataset from the specified Excel file using pandas.

2. Applies IHS transformation
   For selected columns, it applies an inverse hyperbolic sine (IHS) transformation. This is useful for normalizing data that may include zeros or negative values. It works similarly to a log-transform but is more flexible.

   It transforms these columns if they exist:
   - croic
   - roic
   - one_year_momentum
   - stock_volatility_inverted
   - income_quality
   - ebitda_yield

   Each of these becomes a new column with _ihs added to the name (for example, roic_ihs).

3. Winsorizes some variables
   The script caps extreme values in the IHS-transformed columns by setting a floor at the 1st percentile and a ceiling at the 99th percentile. This reduces the impact of extreme outliers.

4. Applies z-score capping
   For momentum_ihs, any value more than 3 standard deviations away from the mean is replaced with a missing value (NaN). This is another way to reduce outlier influence.

5. Saves the cleaned data
   The cleaned data is saved to a new Excel file in the same folder, with '_cleaned' added to the filename.

Why This Was Done

The purpose of this cleaning process is to improve the quality of the dataset used for stock return prediction models. Some of the raw financial metrics can be highly skewed or contain extreme values that distort statistical modeling. Applying transformations and outlier treatments ensures that the input data is more normally distributed and stable, which helps improve model performance and interpretability.

Example Output

If the original file is stocks_clean2.xlsx, the cleaned file will be saved as stocks_clean2_cleaned.xlsx in the same directory.

Summary

- Input: 'dataset/stocks_clean2.xlsx', sheet 'ark'
- Output: 'dataset/stocks_clean2_cleaned.xlsx'
- Helps reduce skewness and outliers in the data
- Improves modeling quality and consistency
