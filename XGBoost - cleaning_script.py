import pandas as pd
import numpy as np
import os

# --- INPUT ---
excel_path = "dataset/stocks_clean2.xlsx"
sheet_name = "ark"

# --- TRANSFORM FUNCTIONS ---
def ihs_transform(x):
    return np.log(x + np.sqrt(x**2 + 1))

def z_score_cap(series, z_thresh=3.0):
    z_scores = (series - series.mean()) / series.std()
    return series.where(z_scores.abs() <= z_thresh, np.nan)

def winsorize_series(series, lower_quantile=0.01, upper_quantile=0.99):
    lower = series.quantile(lower_quantile)
    upper = series.quantile(upper_quantile)
    return series.clip(lower, upper)

# --- MAIN ---
def clean_data(df):
    # IHS-transformasjon
    if 'croic' in df.columns:
        df['croic_ihs'] = ihs_transform(df['croic'])
    if 'roic' in df.columns:
        df['roic_ihs'] = ihs_transform(df['roic'])
    if 'one_year_momentum' in df.columns:
        df['momentum_ihs'] = ihs_transform(df['one_year_momentum'])
    if 'stock_volatility_inverted' in df.columns:
        df['volatility_ihs'] = ihs_transform(df['stock_volatility_inverted'])
    if 'income_quality' in df.columns:
        df['income_quality_ihs'] = ihs_transform(df['income_quality'])
    if 'ebitda_yield' in df.columns:
        df['ebitda_ihs'] = ihs_transform(df['ebitda_yield'])

    # Winsorizing etter IHS
    winsorize_cols = ['croic_ihs', 'roic_ihs', 'ebitda_ihs', 'income_quality_ihs', 'volatility_ihs']
    for col in winsorize_cols:
        if col in df.columns:
            df[col] = winsorize_series(df[col])

    # Z-score capping på øvrige (valgfritt hvis winsor er brukt)
    z_cap_cols = ['momentum_ihs']
    for col in z_cap_cols:
        if col in df.columns:
            df[col] = z_score_cap(df[col])

    return df

# --- EXECUTE ---
if __name__ == "__main__":
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    cleaned_df = clean_data(df)

    output_path = excel_path.replace(".xlsx", "_cleaned.xlsx")
    cleaned_df.to_excel(output_path, index=False)
    print(f"✅ Saved cleaned file to: {output_path}")

