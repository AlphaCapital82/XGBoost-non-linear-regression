Diagnostics script: What This Script Does

This Python script checks the quality of your cleaned stock dataset before running any models. It looks at missing data, normality, outliers, and multicollinearity. The results are saved in a report called 'diagnostics_report.txt'.

It uses the file: dataset/stocks_clean2_cleaned.xlsx, sheet: 'ark'.

Step-by-Step Overview

1. Loads the Excel file
   It reads the Excel file using pandas and makes sure decimal numbers are formatted correctly.

2. Checks for missing values
   It counts how many missing values (empty cells) are in each column. This helps you see if any important data is missing.

3. Shows basic statistics
   For each variable, it shows the average, standard deviation, minimum, maximum, and percentiles. This gives a quick overview of the data.

4. Measures skewness
   It checks if the numbers in a column are spread evenly or if they are pulled more to one side (left or right). Strong skew may mean you need to transform the data.

5. Tests for normal distribution
   It runs a test (Shapiro-Wilk) to check if each column looks like a normal distribution. A low score means the data is probably not normal.

6. Checks for multicollinearity (VIF)
   It looks at whether any of the columns are strongly related to each other. If two variables are almost the same, it can mess up regression results.

7. Finds outliers
   It shows how many extreme values are found in each column using two methods:
   - IQR (based on percentiles)
   - Z-score (based on standard deviations)

Why This Is Useful

This script helps you understand problems in your data before you use it in a model. It tells you if you have missing values, strange numbers, or highly related columns. That way you can clean or adjust your data to get better results.

Output

The report is saved as: diagnostics_report.txt





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



