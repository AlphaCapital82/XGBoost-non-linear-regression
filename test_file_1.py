# === Step 1: Load and Clean Data ===
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import XGBRegressor, plot_importance
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

# Load Excel
df = pd.read_excel("dataset/stocks_clean2_cleaned.xlsx", sheet_name="ark")
df = df.dropna(subset=["stock_return"])  # Drop rows with missing target

# Convert problematic columns to numeric
columns_to_convert = [
    'croic_ihs', 'roic_ihs', 'ebitda_ihs', 'book_to_market',
    'vix_index_inverted', 'momentum_ihs',
    'volatility_ihs', 'high_yield_spread_inverted',
    'income_quality_ihs', 'debt_ratio_directional'
]
for col in columns_to_convert:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# === Step 2: Define Train/Test Split by Year ===
train_df = df[df['year'] <= 2020]
test_df = df[df['year'] > 2020]

# === Step 3: Define Features and Target ===
features = columns_to_convert
X_train = train_df[features].fillna(train_df[features].mean())
X_test = test_df[features].fillna(train_df[features].mean())
y_train = train_df['stock_return']
y_test = test_df['stock_return']

# === Step 4: Initial Model ===
initial_model = XGBRegressor(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
initial_model.fit(X_train, y_train)

# === Step 5: Evaluate Initial Model ===
y_pred = initial_model.predict(X_test)
print("Initial RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("Initial R-squared:", r2_score(y_test, y_pred))

# === Step 6: Feature Importance (Initial) ===
plot_importance(initial_model, importance_type='gain', max_num_features=10)
plt.title("Feature Importance (Initial Model)")
plt.tight_layout()
plt.show()

# === Step 7: Grid Search for Hyperparameters ===
param_grid = {
    'max_depth': [6, 7, 8, 9],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [80, 100, 120],
    'min_child_weight': [1, 3, 5],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'reg_alpha': [0, 0.1, 0.5],
    'reg_lambda': [1, 5, 10],
    'gamma': [0, 0.1],
    'subsample': [0.6, 0.8],
}


grid = GridSearchCV(
    XGBRegressor(random_state=42),
    param_grid,
    cv=4,
    scoring='neg_mean_squared_error',
    verbose=1,
    n_jobs=-1
)
grid.fit(X_train, y_train)
print("Best Parameters:", grid.best_params_)
best_model_manual = XGBRegressor(**grid.best_params_, random_state=42)

# === Step 8: Train Tuned Model with Early Stopping ===
# Convert to DMatrix for xgb.train
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

xgb_params = {k: v for k, v in grid.best_params_.items() if k != 'n_estimators'}
xgb_params.update({
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "seed": 42
})

booster = xgb.train(
    xgb_params,
    dtrain,
    num_boost_round=500,
    evals=[(dtrain, "train"), (dtest, "eval")],
    early_stopping_rounds=10,
    verbose_eval=False
)

# === Step 9: Evaluate Tuned Model ===
y_pred_best = booster.predict(dtest)
y_train_pred_best = booster.predict(dtrain)

rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_best))
r2_test = r2_score(y_test, y_pred_best)
rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred_best))
r2_train = r2_score(y_train, y_train_pred_best)

print("\n--- Tuned Model Evaluation (Booster) ---")
print("Test RMSE:", rmse_test)
print("Test R-squared:", r2_test)
print("Train RMSE:", rmse_train)
print("Train R-squared:", r2_train)
print("Generalization Gap (RMSE):", abs(rmse_train - rmse_test))

# === Step 10: Feature Importance from Booster ===
booster_feature_scores = booster.get_score(importance_type='gain')
sorted_scores = dict(sorted(booster_feature_scores.items(), key=lambda item: item[1], reverse=True))

print("\nFeature Importance (Gain):")
for feat, score in sorted_scores.items():
    print(f"{feat}: {score:.2f}")

plt.scatter(y_test, y_pred_best, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Stock Return")
plt.ylabel("Predicted Stock Return")
plt.title("Actual vs Predicted Returns")
plt.tight_layout()
plt.show()

results_df = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred_best
})
print(results_df.head(10))

import seaborn as sns

residuals = y_test - y_pred_best
sns.histplot(residuals, kde=True)
plt.title("Residual Distribution")
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt

# Sort and create DataFrame
importance_df = pd.DataFrame(sorted_scores.items(), columns=["Feature", "Gain"]).sort_values("Gain", ascending=False)

# Plot without deprecated palette usage
plt.figure(figsize=(10, 6))
sns.barplot(data=importance_df, x="Gain", y="Feature", hue="Feature", dodge=False, legend=False)
plt.title("Feature Importance (Gain, Booster Model)")
plt.xlabel("Total Gain")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()
