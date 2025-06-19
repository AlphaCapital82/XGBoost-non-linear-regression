import pandas as pd
import numpy as np
import os
from scipy.stats import shapiro
from statsmodels.stats.outliers_influence import variance_inflation_factor

# --- INPUT ---
excel_path = "dataset/stocks_clean2_cleaned.xlsx"
sheet_name = "ark"
report_path = "diagnostics_report.txt"

# --- Variableliste ---
features = [
     "book_to_market", "vix_index_inverted",
    "high_yield_spread_inverted", "debt_ratio_directional",
    "croic_ihs", "roic_ihs", "ebitda_ihs", "momentum_ihs",
    "volatility_ihs", "income_quality_ihs"
]

# --- Hjelpefunksjoner ---
def calculate_vif(df, variables):
    vif_data = pd.DataFrame()
    vif_data["feature"] = variables
    vif_data["VIF"] = [variance_inflation_factor(df[variables].values, i)
                       for i in range(len(variables))]
    return vif_data

def detect_outliers(df, column):
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    iqr_outliers = ((df[column] < (q1 - 1.5 * iqr)) | (df[column] > (q3 + 1.5 * iqr))).sum()
    z_scores = ((df[column] - df[column].mean()) / df[column].std()).abs()
    z_outliers = (z_scores > 3).sum()
    return iqr_outliers, z_outliers

# --- Last inn data ---
df = pd.read_excel(excel_path, sheet_name=sheet_name)

# --- FIX: Konverter desimalformatering ---
for col in features:
    df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", "."), errors="coerce")
df["stock_return"] = pd.to_numeric(df["stock_return"].astype(str).str.replace(",", "."), errors="coerce")


with open(report_path, "w", encoding="utf-8") as f:

    f.write("=== DIAGNOSTICS REPORT ===\n\n")

    # 1. Missing values
    f.write("1. Missing Values:\n")
    f.write(df[["stock_return"] + features].isnull().sum().to_string())
    f.write("\n\n")

    # 2. Summary Statistics
    f.write("2. Summary Statistics:\n")
    f.write(df[features].describe().to_string())
    f.write("\n\n")

    # 3. Skewness
    f.write("3. Skewness:\n")
    for var in features:
        skew_val = df[var].skew()
        f.write(f"{var}: {skew_val:.6f}\n")
    f.write("\n")

    # 4. Normality (Shapiro-Wilk)
    f.write("4. Normality (Shapiro-Wilk p-values):\n")
    for var in features:
        try:
            pval = shapiro(df[var].sample(n=min(5000, df[var].shape[0]), random_state=1))[1]
            f.write(f"{var}: {pval:.3e}\n")
        except Exception as e:
            f.write(f"{var}: Error ({e})\n")
    f.write("\n")

    # 5. VIF
    f.write("5. VIF:\n")
    vif_df = calculate_vif(df.dropna(subset=features), features)
    f.write(vif_df.to_string(index=False))
    f.write("\n\n")

    # 6. Outlier Summary
    f.write("6. Outlier Summary:\n")
    f.write(f"{'Variable':>22}  {'IQR-outliers':>13}  {'Z-outliers':>10}\n")
    for var in features:
        iqr_outliers, z_outliers = detect_outliers(df, var)
        f.write(f"{var:>22}  {iqr_outliers:13}  {z_outliers:10}\n")
