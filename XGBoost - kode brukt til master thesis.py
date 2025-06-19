
# Lag en tekstfil som inneholder den oppdaterte Python-koden med:
# - fem interaksjonsledd for momentum
# - smartere hyperparameter-tuning

# === Step 1: Load and Clean Data ===
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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

# Add interaction terms
df['interaction_momentum_croic'] = df['momentum_ihs'] * df['croic_ihs']
df['interaction_momentum_volatility'] = df['momentum_ihs'] * df['volatility_ihs']
df['interaction_momentum_hyspread'] = df['momentum_ihs'] * df['high_yield_spread_inverted']
df['interaction_momentum_vix'] = df['momentum_ihs'] * df['vix_index_inverted']
df['interaction_momentum_btm'] = df['momentum_ihs'] * df['book_to_market']

# === Step 2: Define Train/Test Split by Year ===
train_df = df[df['year'] <= 2020]
test_df = df[df['year'] > 2020]

# === Step 3: Define Features and Target ===
momentum_interactions = [
    'interaction_momentum_croic',
    'interaction_momentum_volatility',
    'interaction_momentum_hyspread',
    'interaction_momentum_vix',
    'interaction_momentum_btm'
]
features = columns_to_convert + momentum_interactions
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

# === Step 7: Smart Grid Search for Hyperparameters ===
param_grid = {
    'n_estimators': [150, 200],
    'learning_rate': [0.01, 0.05],
    'max_depth': [4, 6],
    'min_child_weight': [1, 3],
    'gamma': [0, 0.1],
    'subsample': [0.7, 0.8],
    'colsample_bytree': [0.7, 0.8],
    'reg_alpha': [0, 0.1],
    'reg_lambda': [1, 3]
}

grid = GridSearchCV(
    XGBRegressor(random_state=42),
    param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    verbose=1,
    n_jobs=-1
)
grid.fit(X_train, y_train)
print("Best Parameters:", grid.best_params_)
best_model_manual = XGBRegressor(**grid.best_params_, random_state=42)

# === Step 8: Train Tuned Model with Early Stopping ===
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
    num_boost_round=grid.best_params_['n_estimators'],
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

print("\\n--- Tuned Model Evaluation (Booster) ---")
print("Test RMSE:", rmse_test)
print("Test R-squared:", r2_test)
print("Train RMSE:", rmse_train)
print("Train R-squared:", r2_train)
print("Generalization Gap (RMSE):", abs(rmse_train - rmse_test))

# === Step 10: Feature Importance from Booster ===
booster_feature_scores = booster.get_score(importance_type='gain')
sorted_scores = dict(sorted(booster_feature_scores.items(), key=lambda item: item[1], reverse=True))

print("\\nFeature Importance (Gain):")
for feat, score in sorted_scores.items():
    print(f"{feat}: {score:.2f}")

plt.figure(figsize=(10, 6))
importance_df = pd.DataFrame(sorted_scores.items(), columns=["Feature", "Gain"]).sort_values("Gain", ascending=False)
sns.barplot(data=importance_df, x="Gain", y="Feature", hue="Feature", dodge=False, legend=False)
plt.title("Feature Importance (Gain, Booster Model)")
plt.xlabel("Total Gain")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()


# === Step 11: Partial Dependence Plots (1D and 2D) ===
from sklearn.inspection import PartialDependenceDisplay
import matplotlib.pyplot as plt

# Refit the sklearn-compatible model for PDP
best_model_manual.fit(X_train, y_train)

# --- One-way PDP for stock return drivers ---
features_1d = ['momentum_ihs', 'roic_ihs', 'book_to_market', 'debt_ratio_directional', 'croic_ihs',
               'high_yield_spread_inverted',
               'volatility_ihs', 'income_quality_ihs', 'vix_index_inverted',
               'ebitda_ihs', 'vix_index_inverted'
               ]
for feature in features_1d:
    fig, ax = plt.subplots(figsize=(7, 4))
    PartialDependenceDisplay.from_estimator(best_model_manual, X_train, [feature], ax=ax)
    plt.title(f"Partial Dependence Plot: {feature}")
    plt.tight_layout()
    plt.show()

# --- Two-way PDP for interaction terms (e.g., momentum × VIX) ---
features_2d = [
    ('momentum_ihs', 'vix_index_inverted'),
               ('momentum_ihs', 'book_to_market')
               ]
for feature_pair in features_2d:
    fig = plt.figure(figsize=(8, 6))
    PartialDependenceDisplay.from_estimator(best_model_manual, X_train, [feature_pair])
    plt.title(f"2D PDP: {feature_pair[0]} × {feature_pair[1]}")
    plt.tight_layout()
    plt.show()
