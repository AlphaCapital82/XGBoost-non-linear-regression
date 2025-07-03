Diagnostics script:
What This Script Does

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


XGBoost script explanation:

What This Script Does

This Python script builds a stock return prediction model using XGBoost. It uses financial and macroeconomic variables along with five interaction terms involving momentum. The script performs model training, tuning, evaluation, and feature interpretation. It is based on the cleaned dataset: stocks_clean2_cleaned.xlsx (sheet: 'ark').

Step-by-Step Summary

1. Loads and prepares the data
   The script reads the Excel file, removes rows without stock return, converts all input variables to numeric, and adds interaction terms that combine momentum with other variables.

2. Splits data into training and testing
   Training is done on all data from year 2020 and earlier. Testing is done on data after 2020.

3. Defines inputs and target
   Features include original variables plus interaction terms. The target is one-year stock return. Missing values are filled with column averages from training data.

4. Fits an initial XGBoost model
   A basic XGBoost model is trained using fixed parameters. Performance is measured using RMSE (error) and R-squared (fit).

5. Shows feature importance (initial model)
   The most important features are displayed based on gain (how much each variable improved the model).

6. Tunes the model using GridSearchCV
   A wide set of XGBoost parameters are tested using cross-validation to find the best combination.

7. Trains a tuned XGBoost model with early stopping
   The best parameters from tuning are used to train a new model with early stopping to avoid overfitting.

8. Evaluates the tuned model
   RMSE and R-squared are reported for both training and test data. The script also prints the generalization gap (difference between train and test error).

9. Displays feature importance (tuned model)
   Shows which features contributed most to the tuned model using a bar chart based on gain.

10. Generates Partial Dependence Plots (PDP)
    PDPs are created to visualize how specific variables (and combinations) affect the predicted stock return. This includes:
    - One-way PDPs for variables like momentum, roic, book-to-market, etc.
    - Two-way PDPs showing interactions like momentum × VIX and momentum × book-to-market.

Why This Is Useful

This script helps build and interpret a machine learning model for predicting stock returns. It uses interaction terms, hyperparameter tuning, and diagnostic plots to make the model both powerful and explainable.

Output

- Model evaluation (printout)
- Feature importance chart
- Partial dependence plots (for individual and paired variables)

