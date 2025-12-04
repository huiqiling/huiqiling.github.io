<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

## Cryptocurrency Price Movement Prediction Using Time-Series Machine Learning

This project explores whether short-term cryptocurrency price changes can be predicted using simple machine-learning models and basic time-series features. Using daily data for Bitcoin, Ethereum, and ten other cryptocurrencies, I built lag features, volatility measures, and rolling averages, then tested Logistic Regression, Random Forest, and XGBoost on 1, 3, 5, and 7-day prediction windows.

The models show small but consistent level of predictability for the 1 and 3 day horizons (with accuracy reaching about 0.56), while performance drops off quickly for longer forecasts. Tree-based models do slightly better for very short-term predictions, and Logistic Regression ends up being more stable for the 5 and 7 day horizons. Overall, the findings match the common idea that crypto prices can show short-lived patterns, but become much harder to predict as the time horizon gets longer.

***

## Introduction 

![](assets/IMG/figure0_intro.png){: width="600" }

Cryptocurrency markets change quickly and are very volatile. Past research shows that Bitcoin and other coins often move in ways that are hard to predict. This project asks: can recent price changes help us guess short-term direction?

I use daily price data for 12 major cryptocurrencies. The goal is not to predict exact prices, but to see if machine learning can classify next-day and multi-day price direction using simple features. Stablecoins are not included because their prices are fixed and do not show real market changes.

***

## Data

### Data Overview

The data comes from Kaggle's *Cryptocurrency Historical Prices*. It has daily open, close, high, low, and volume for many coins.

I use 12 major cryptocurrencies: Bitcoin, Ethereum, Binance Coin, XRP, Litecoin, Cardano, Polkadot, Solana, Uniswap, ChainLink, Dogecoin, and Monero. Each coin is in a separate CSV file. I combine them and sort by date for feature engineering.

Stablecoins (like USDT, USDC, Tether, Wrapped Bitcoin) are not used. Their prices are fixed to fiat currencies, so they do not show real market changes. Including them would lower volatility and make prediction less meaningful.

<div style="max-height: 400px; overflow-y: auto; border: 1px solid #ddd; padding: 10px; margin: 10px 0;">

```python
# 1. Environment and Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc

# Load selected pure cryptocurrency files (exclude stablecoins)
coin_files = [
    ("coin_Bitcoin.csv", "BTC"),
    ("coin_Ethereum.csv", "ETH"),
    ("coin_BinanceCoin.csv", "BNB"),
    ("coin_XRP.csv", "XRP"),
    ("coin_Litecoin.csv", "LTC"),
    ("coin_Cardano.csv", "ADA"),
    ("coin_Polkadot.csv", "DOT"),
    ("coin_Solana.csv", "SOL"),
    ("coin_Uniswap.csv", "UNI"),
    ("coin_ChainLink.csv", "LINK"),
    ("coin_Dogecoin.csv", "DOGE"),
    ("coin_Monero.csv", "XMR"),
]

dfs = []
for file, label in coin_files:
    df_coin = pd.read_csv(file)
    df_coin["coin"] = label
    dfs.append(df_coin)
df = pd.concat(dfs, ignore_index=True)

df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(["coin", "Date"])

print("Dataset Summary Statistics:")
display(df.describe())

print("First 5 Rows of Dataset:")
display(df.head())
```

</div>

### Feature Engineering

To enable effective machine learning on cryptocurrency price data, a variety of features are engineered to capture different aspects of market behavior. The full set of features includes:

- **Date**: The date of each observation (used for time alignment, not as a predictor).
- **Close**: The closing price (used for target construction and reference, not as a predictor).
- **Volume**: Daily trading volume, reflecting market activity.
- **Marketcap**: Total market capitalization, indicating the overall value of the cryptocurrency.
- **Lag Features**: Previous closing prices (e.g., lag_1, lag_7) to capture short-term memory and trends.
- **Returns and Momentum**: *n-day return* (e.g., ret_3, ret_7) is the simple return over a fixed window

<p>
\[
\mathrm{ret}_n = \frac{\mathrm{Close}_t - \mathrm{Close}_{t-n}}{\mathrm{Close}_{t-n}}
\]
</p>

- **Moving Averages**: Simple moving average (SMA)

<p>
\[
\mathrm{ma}_n(t) = \frac{1}{n} \sum_{i=0}^{n-1} \mathrm{Close}_{t-i}
\]
</p>

- **Volatility**: Rolling standard deviation of returns (volatility):

<p>
\[
\mathrm{volatility}_n(t) = \mathrm{std}(\mathrm{return}_{t-n+1}, \ldots, \mathrm{return}_t)
\]
</p>

- **Volume Change**: Change of daily trading volume

<p>
\[
\mathrm{volume\_change}_t = \frac{\mathrm{Volume}_t - \mathrm{Volume}_{t-1}}{\mathrm{Volume}_{t-1}}
\]
</p>

- **Streaks**: Consecutive up/down days (up_streak, down_streak) to capture persistent trends or reversals.
- **Rolling Max/Min and Quantiles**: Rolling max, rolling min, and rolling quantile (e.g., 90th percentile) for detecting price extremes.
- **Multi-horizon targets**: Binary labels for whether the price goes up in 1, 3, 5, or 7 days.

<div style="max-height: 400px; overflow-y: auto; border: 1px solid #ddd; padding: 10px; margin: 10px 0;">

```python
# Create time-series features for modeling
feature_df = df.copy()
feature_df = feature_df.sort_values(['coin', 'Date'])

# Lag features (keep lag_1 and lag_7 to reduce multicollinearity)
feature_df['lag_1'] = feature_df.groupby('coin')['Close'].shift(1)
feature_df['lag_7'] = feature_df.groupby('coin')['Close'].shift(7)

# Daily return (percentage)
feature_df['return'] = feature_df.groupby('coin')['Close'].pct_change()

# 3-day and 7-day returns
feature_df['ret_3'] = feature_df.groupby('coin')['Close'].pct_change(3)
feature_df['ret_7'] = feature_df.groupby('coin')['Close'].pct_change(7)

# Moving averages (7-day and 30-day)
feature_df['ma_7'] = feature_df.groupby('coin')['Close'].transform(lambda x: x.rolling(window=7).mean())
feature_df['ma_30'] = feature_df.groupby('coin')['Close'].transform(lambda x: x.rolling(window=30).mean())

# Exponential moving averages (EMA)
feature_df['ema_7'] = feature_df.groupby('coin')['Close'].transform(lambda x: x.ewm(span=7, adjust=False).mean())
feature_df['ema_30'] = feature_df.groupby('coin')['Close'].transform(lambda x: x.ewm(span=30, adjust=False).mean())

# Rolling volatility (30-day std of returns)
feature_df['volatility'] = feature_df.groupby('coin')['return'].transform(lambda x: x.rolling(window=30).std())

# Volume change relative to previous day (percentage)
feature_df['volume_change'] = feature_df.groupby('coin')['Volume'].pct_change()

# Consecutive up/down days
feature_df['up_streak'] = feature_df.groupby('coin')['return'].transform(
    lambda x: (x > 0).astype(int).groupby((x <= 0).cumsum()).cumcount())
feature_df['down_streak'] = feature_df.groupby('coin')['return'].transform(
    lambda x: (x < 0).astype(int).groupby((x >= 0).cumsum()).cumcount())

# 7-day max/min close
feature_df['max_7'] = feature_df.groupby('coin')['Close'].transform(lambda x: x.rolling(window=7).max())
feature_df['min_7'] = feature_df.groupby('coin')['Close'].transform(lambda x: x.rolling(window=7).min())

# 14-day cumulative return
feature_df['momentum_14'] = feature_df.groupby('coin')['Close'].pct_change(14)

# 90th percentile over 30 days
feature_df['q90_30'] = feature_df.groupby('coin')['Close'].transform(lambda x: x.rolling(window=30).quantile(0.9))

print("All Features in Dataset:")
for col in feature_df.columns:
    print(col)
display(feature_df.head())
```

</div>

<div style="max-height: 300px; overflow-y: auto; border: 1px solid #ddd; padding: 10px; margin: 10px 0;">

```python
# Multi-horizon target engineering 
horizons = [1, 3, 5, 7]
for h in horizons:
    feature_df[f'target_{h}'] = (feature_df['Close'].shift(-h) > feature_df['Close']).astype(int)

# Remove the last max(horizons) rows to avoid lookahead bias
feature_df = feature_df.iloc[:-max(horizons)].reset_index(drop=True)

# Drop any remaining NaN values
feature_df = feature_df.dropna().reset_index(drop=True)

# Show new features and targets
print('Features:', feature_df.columns.tolist())
display(feature_df.head())
```

</div>

**Output:**
```
Features: ['Date', 'Close', 'Volume', 'Marketcap', 'lag_1', 'lag_7', 'return', 'ret_3', 'ret_7', 
'ma_7', 'ma_30', 'ema_7', 'ema_30', 'volatility', 'volume_change', 'up_streak', 'down_streak', 
'max_7', 'min_7', 'momentum_14', 'q90_30', 'target_1', 'target_3', 'target_5', 'target_7']
```

### Exploratory Data Analysis

#### Figure 1: Closing Price Trends for 12 Major Cryptocurrencies (Log Scale)

![](assets/IMG/figure1_price_trends.png){: width="700" }

*Figure 1: Closing Price Trends for 12 Major Cryptocurrencies (Log Scale)*

<div style="max-height: 300px; overflow-y: auto; border: 1px solid #ddd; padding: 10px; margin: 10px 0;">

```python
# Figure 1: Closing Price Trends for 12 Major Cryptocurrencies (Log Scale)
plt.figure(figsize=(10, 5))
for coin, group in df.groupby('coin'):
    plt.plot(group['Date'], group['Close'], label=coin)
plt.yscale('log')
plt.title("Figure 1: Closing Price Trends for 12 Major Cryptocurrencies (Log Scale)")
plt.xlabel("Date")
plt.ylabel("Closing Price (USD, log scale)")
plt.legend()
plt.grid(alpha=0.3, which='both', linestyle='--')
plt.savefig('assets/IMG/figure1_price_trends.png', dpi=150, bbox_inches='tight')
plt.show()
```

</div>

**Explanation:** This plot shows the long-term price evolution of 12 major cryptocurrencies on a logarithmic scale. Using a log scale makes it possible to compare assets with prices that span several orders of magnitude. We can see clear market cycles—periods of rapid appreciation followed by extended drawdowns. Bitcoin remains the most stable and highest-valued asset, while other coins such as Ethereum, Binance Coin, and Solana exhibit significant growth and volatility.

#### Figure 2: Distribution of Daily Returns for 12 Major Cryptocurrencies

![](assets/IMG/figure2_daily_returns.png){: width="600" }

*Figure 2: Distribution of Daily Returns for 12 Major Cryptocurrencies (Log Y, Density)*

<div style="max-height: 300px; overflow-y: auto; border: 1px solid #ddd; padding: 10px; margin: 10px 0;">

```python
# Figure 2: Distribution of Daily Returns for 12 Major Cryptocurrencies (Log Y, Density)
plt.figure(figsize=(8, 4))
for coin, group in df.groupby('coin'):
    returns = group['Close'].pct_change()
    plt.hist(returns, bins=80, alpha=0.5, label=coin, density=True)
plt.yscale('log')
plt.title("Figure 2: Distribution of Daily Returns for 12 Major Cryptocurrencies (Log Y, Density)")
plt.xlabel("Daily Return")
plt.ylabel("Density (log scale)")
plt.legend()
plt.savefig('assets/IMG/figure2_daily_returns.png', dpi=150, bbox_inches='tight')
plt.show()
```

</div>

**Explanation:** This figure shows the distribution of daily returns for the 12 selected cryptocurrencies on a log-scaled y-axis. Most returns cluster around zero, but the distributions have long tails, indicating that large price jumps occur frequently.

#### Figure 3: Daily Trading Volume for 12 Major Cryptocurrencies

![](assets/IMG/figure3_trading_volume.png){: width="700" }

*Figure 3: Daily Trading Volume for 12 Major Cryptocurrencies (Log Scale)*

<div style="max-height: 300px; overflow-y: auto; border: 1px solid #ddd; padding: 10px; margin: 10px 0;">

```python
# Figure 3: Daily Trading Volume for 12 Major Cryptocurrencies (Log Scale)
plt.figure(figsize=(12, 5))
for coin, group in df.groupby('coin'):
    plt.plot(group['Date'], group['Volume'], label=coin)
plt.yscale('log')
plt.title("Figure 3: Daily Trading Volume for 12 Major Cryptocurrencies (Log Scale)")
plt.xlabel("Date")
plt.ylabel("Volume (log scale)")
plt.legend()
plt.grid(alpha=0.3, which='both', linestyle='--')
plt.savefig('assets/IMG/figure3_trading_volume.png', dpi=150, bbox_inches='tight')
plt.show()
```

</div>

**Explanation:** This figure shows the daily trading volume of cryptocurrencies on a logarithmic scale. Across all coins, trading activity rises over time, with noticeable peaks during major market cycles.

#### Figure 4: Feature vs Target Relationship

![](assets/IMG/figure4_feature_target.png){: width="100%" }

*Figure 4: Feature Distributions by Next-Day Target (Boxplots)*

<div style="max-height: 350px; overflow-y: auto; border: 1px solid #ddd; padding: 10px; margin: 10px 0;">

```python
# Figure 4: Feature Distributions by Next-Day Target (Target_1) 
import seaborn as sns

# Automatically get all numeric features (excluding targets, Close, Date)
all_features = [col for col in feature_df.columns 
                if col not in ['Date', 'Close', 'target', 'target_1', 'target_3', 'target_5', 'target_7']]

fig, axes = plt.subplots(nrows=1, ncols=len(all_features), figsize=(3*len(all_features), 6))
if len(all_features) == 1:
    axes = [axes]
for i, feature in enumerate(all_features):
    sns.boxplot(x='target_1', y=feature, data=feature_df, ax=axes[i], showfliers=False)
    axes[i].set_title(feature)
    axes[i].set_xlabel('target_1')
plt.tight_layout()
plt.suptitle('Figure 4: Feature Distributions by Next-Day Target (Boxplots)', y=1.03, fontsize=16)
plt.savefig('assets/IMG/figure4_feature_target.png', dpi=150, bbox_inches='tight')
plt.show()
```

</div>

**Explanation:** This figure shows the distribution of each engineered feature grouped by the short-term target (target_1, i.e., next-day up/down). Most features show only weak separation between up and down days, which reflects how noisy and hard to predict short-term cryptocurrency movements are.

#### Figure 5: Feature Correlation Heatmap

![](assets/IMG/figure5_correlation_heatmap.png){: width="600" }

*Figure 5: Feature Correlation Heatmap for Engineered Predictors*

<div style="max-height: 350px; overflow-y: auto; border: 1px solid #ddd; padding: 10px; margin: 10px 0;">

```python
# Figure 5: Feature Correlation Heatmap for Engineered Predictors
import seaborn as sns
plt.figure(figsize=(10, 8)) 
# Get all features (no target, Close, Date)
all_features = [col for col in feature_df.columns 
                if col not in ['Date', 'Close', 'target', 'target_1', 'target_3', 'target_5', 'target_7']]
corr = feature_df[all_features].corr()

sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', square=True)
plt.title('Figure 5: Feature Correlation Heatmap for Engineered Predictors')
plt.savefig('assets/IMG/figure5_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()
```

</div>

**Explanation:** The correlation heatmap shows that price-related features (lag_1, lag_7, ma_7, ma_30) are highly correlated with each other. In contrast, return is almost uncorrelated with these trend features. Volatility and volume_change show very low correlations with all other variables, suggesting that they represent different dimensions of market behavior.

***

## Modelling

### Model Selection and Training Data Preparation

The exploratory analysis shows that daily price movements are hard to separate using simple patterns. Features for up and down days overlap a lot, and the short-term signals in the data are fairly weak. Because of this, the goal is not to build a perfect classifier, but to see how different models deal with noisy financial data.

Logistic Regression is used as a baseline since it shows how well a basic linear model can perform. Random Forest and XGBoost are added as more flexible models that can capture nonlinear patterns and interactions between features.

<div style="max-height: 300px; overflow-y: auto; border: 1px solid #ddd; padding: 10px; margin: 10px 0;">

```python
# Train-test split 
split_idx = int(len(feature_df) * 0.8)
X = feature_df.select_dtypes(include=[np.number]).drop(
    columns=['target_1', 'target_3', 'target_5', 'target_7'], errors='ignore')
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y1_train, y1_test = feature_df['target_1'].iloc[:split_idx], feature_df['target_1'].iloc[split_idx:]
y3_train, y3_test = feature_df['target_3'].iloc[:split_idx], feature_df['target_3'].iloc[split_idx:]
y5_train, y5_test = feature_df['target_5'].iloc[:split_idx], feature_df['target_5'].iloc[split_idx:]
y7_train, y7_test = feature_df['target_7'].iloc[:split_idx], feature_df['target_7'].iloc[split_idx:]
print(f"Train set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")
```

</div>

**Output:**
```
Train set: 16603 samples
Test set: 4151 samples
```

### Model Training and Evaluation

#### Logistic Regression

Logistic Regression serves as a baseline for predicting next-day price movement. As a simple linear model, it shows how much predictability exists in the data.

<div style="max-height: 400px; overflow-y: auto; border: 1px solid #ddd; padding: 10px; margin: 10px 0;">

```python
# Multi-horizon Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

horizons = [1, 3, 5, 7]
results_logreg = []

for h in horizons:
    target_col = f'target_{h}'
    y = feature_df[target_col]
    split_idx = int(len(feature_df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    logreg = LogisticRegression(max_iter=1000, random_state=42)
    logreg.fit(X_train, y_train)
    y_pred_logreg = logreg.predict(X_test)
    acc_logreg = accuracy_score(y_test, y_pred_logreg)
    print(f"Logistic Regression Accuracy (horizon {h}): {acc_logreg:.4f}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_logreg))
    results_logreg.append({'horizon': h, 'model': 'LogReg', 'acc': acc_logreg})
```

</div>

**Output:**
```
Logistic Regression Accuracy (horizon 1): 0.5312
Confusion Matrix:
 [[1237  912]
 [1034  968]]
Logistic Regression Accuracy (horizon 3): 0.5317
Confusion Matrix:
 [[1423  758]
 [1186  784]]
Logistic Regression Accuracy (horizon 5): 0.5584
Confusion Matrix:
 [[1433  774]
 [1059  885]]
Logistic Regression Accuracy (horizon 7): 0.5560
```

#### Random Forest

Random Forest combines many decision trees to capture nonlinear patterns and feature interactions.

<div style="max-height: 500px; overflow-y: auto; border: 1px solid #ddd; padding: 10px; margin: 10px 0;">

```python
# Random Forest: All Horizons
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Horizon 1
rf1 = RandomForestClassifier(n_estimators=260, max_depth=6, min_samples_leaf=1, random_state=42)
rf1.fit(X_train, y1_train)
y_pred_rf1 = rf1.predict(X_test)
acc_rf1 = accuracy_score(y1_test, y_pred_rf1)
print(f"Random Forest (horizon 1) Accuracy: {acc_rf1:.4f}")
print("Confusion Matrix:\n", confusion_matrix(y1_test, y_pred_rf1))

# Horizon 3
rf3 = RandomForestClassifier(n_estimators=270, max_depth=6, min_samples_leaf=1, random_state=42) 
rf3.fit(X_train, y3_train)
y_pred_rf3 = rf3.predict(X_test)
acc_rf3 = accuracy_score(y3_test, y_pred_rf3)
print(f"Random Forest (horizon 3) Accuracy: {acc_rf3:.4f}")

# Horizon 5
rf5 = RandomForestClassifier(n_estimators=250, max_depth=6, min_samples_leaf=1, random_state=42)
rf5.fit(X_train, y5_train)
y_pred_rf5 = rf5.predict(X_test)
acc_rf5 = accuracy_score(y5_test, y_pred_rf5)
print(f"Random Forest (horizon 5) Accuracy: {acc_rf5:.4f}")

# Horizon 7
rf7 = RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_leaf=1, random_state=42)
rf7.fit(X_train, y7_train)
y_pred_rf7 = rf7.predict(X_test)
acc_rf7 = accuracy_score(y7_test, y_pred_rf7)
print(f"Random Forest (horizon 7) Accuracy: {acc_rf7:.4f}")
```

</div>

**Output:**
```
Random Forest (horizon 1) Accuracy: 0.5616
Random Forest (horizon 3) Accuracy: 0.5437
Random Forest (horizon 5) Accuracy: 0.5283
Random Forest (horizon 7) Accuracy: 0.5298
```

#### XGBoost (Extreme Gradient Boosting)

XGBoost is a gradient boosting method that builds decision trees sequentially to improve predictive accuracy.

<div style="max-height: 500px; overflow-y: auto; border: 1px solid #ddd; padding: 10px; margin: 10px 0;">

```python
# XGBoost: Four horizons
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Horizon 1
xgb1 = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.06, 
                     eval_metric='logloss', random_state=42, use_label_encoder=False)
xgb1.fit(X_train, y1_train)
y_pred_xgb1 = xgb1.predict(X_test)
acc_xgb1 = accuracy_score(y1_test, y_pred_xgb1)
print(f"XGBoost (horizon 1) Accuracy: {acc_xgb1:.4f}")

# Horizon 3
xgb3 = XGBClassifier(n_estimators=148, max_depth=4, learning_rate=0.03, 
                     eval_metric='logloss', random_state=42, use_label_encoder=False)
xgb3.fit(X_train, y3_train)
y_pred_xgb3 = xgb3.predict(X_test)
acc_xgb3 = accuracy_score(y3_test, y_pred_xgb3)
print(f"XGBoost (horizon 3) Accuracy: {acc_xgb3:.4f}")

# Horizon 5
xgb5 = XGBClassifier(n_estimators=265, max_depth=5, learning_rate=0.02, 
                     eval_metric='logloss', random_state=42, use_label_encoder=False)
xgb5.fit(X_train, y5_train)
y_pred_xgb5 = xgb5.predict(X_test)
acc_xgb5 = accuracy_score(y5_test, y_pred_xgb5)
print(f"XGBoost (horizon 5) Accuracy: {acc_xgb5:.4f}")

# Horizon 7
xgb7 = XGBClassifier(n_estimators=380, max_depth=4, learning_rate=0.05, 
                     eval_metric='logloss', random_state=42, use_label_encoder=False)
xgb7.fit(X_train, y7_train)
y_pred_xgb7 = xgb7.predict(X_test)
acc_xgb7 = accuracy_score(y7_test, y_pred_xgb7)
print(f"XGBoost (horizon 7) Accuracy: {acc_xgb7:.4f}")
```

</div>

**Output:**
```
XGBoost (horizon 1) Accuracy: 0.5606
XGBoost (horizon 3) Accuracy: 0.5526
XGBoost (horizon 5) Accuracy: 0.5338
XGBoost (horizon 7) Accuracy: 0.5240
```

***

## Results

### Overall Model Performance

| Horizon | LogReg | RandomForest | XGBoost |
|---------|--------|--------------|---------|
| 1       | 0.5312 | 0.5616       | 0.5606  |
| 3       | 0.5317 | 0.5437       | 0.5526  |
| 5       | 0.5584 | 0.5283       | 0.5338  |
| 7       | 0.5560 | 0.5298       | 0.5240  |

![](assets/IMG/figure6_model_comparison.png){: width="600" }

*Figure 6: Model Performance Comparison Across Forecast Horizons*

<div style="max-height: 400px; overflow-y: auto; border: 1px solid #ddd; padding: 10px; margin: 10px 0;">

```python
# Combined results table for all three models across all horizons
combined_results = pd.concat([results_df_logreg, results_df_rf, results_df_xg], ignore_index=True)
combined_pivot = combined_results.pivot(index='horizon', columns='model', values='acc')

print("Combined Model Performance Across All Horizons:")
display(combined_pivot)

# Visualization
fig, ax = plt.subplots(figsize=(8, 4))
for model in combined_pivot.columns:
    ax.plot(combined_pivot.index, combined_pivot[model], marker='o', label=model, linewidth=2)

ax.set_xlabel('Forecast Horizon (days)', fontsize=12)
ax.set_ylabel('Accuracy', fontsize=12)
ax.set_title('Model Performance Comparison Across Forecast Horizons', fontsize=14)
ax.legend(fontsize=11)
ax.grid(alpha=0.3, linestyle='--')
ax.set_xticks([1, 3, 5, 7])
plt.tight_layout()
plt.savefig('assets/IMG/figure6_model_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
```

</div>

**Explanation**: All three models (Logistic Regression, Random Forest, XGBoost) were evaluated on four forecast horizons (1, 3, 5, and 7 days). Overall performance ranges from 0.52 to 0.56, which is expected given the noisy and volatile nature of crypto markets.

### Performance by Horizon

- **1-day horizon:** Random Forest and XGBoost achieve the highest accuracy at this horizon, and both of them are better than Logistic Regression. This is consistent with the idea that most usable structure exists in the very short term.

- **3-day horizon:** Accuracy begins to decline for both Random Forest and XGBoost, and the performance gap between models narrows. XGBoost remains slightly ahead.

- **5-day and 7-day horizons:** For these longer horizons, the accuracy of Random Forest and XGBoost drops toward random-guess levels. Interestingly, Logistic Regression performs better than the nonlinear models at this stage.

### Feature Importance (Horizon 1)

![](assets/IMG/figure7_feature_importance.png){: width="700" }

*Figure 7: Feature Importance for Random Forest and XGBoost (Horizon 1)*

<div style="max-height: 400px; overflow-y: auto; border: 1px solid #ddd; padding: 10px; margin: 10px 0;">

```python
# Feature Importance Plots for Random Forest and XGBoost (Horizon 1 Example)
importances_rf = rf1.feature_importances_
importances_xgb = xgb1.feature_importances_
feature_names = X_train.columns

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Random Forest Feature Importance (Left)
rf_imp = pd.Series(importances_rf, index=feature_names).sort_values(ascending=False)
rf_imp.head(15).iloc[::-1].plot(kind='barh', color='darkblue', ax=axes[0])
axes[0].set_title('Random Forest Feature Importance (Horizon 1)')
axes[0].set_xlabel('Importance')

# XGBoost Feature Importance (Right)
xgb_imp = pd.Series(importances_xgb, index=feature_names).sort_values(ascending=False)
xgb_imp.head(15).iloc[::-1].plot(kind='barh', color='orange', ax=axes[1])
axes[1].set_title('XGBoost Feature Importance (Horizon 1)')
axes[1].set_xlabel('Importance')

plt.tight_layout()
plt.savefig('assets/IMG/figure7_feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()
```

</div>

**Explanation (Random Forest):** For the 1-day horizon, Random Forest assigns the highest importance to very short-term price changes, particularly the 3-day return (ret_3) and the most recent daily return. Volume-related features also appear near the top.

**Explanation (XGBoost):** XGBoost shows a similar pattern to Random Forest. Compared with Random Forest, XGBoost places slightly more weight on individual lag features (e.g., lag_1) and longer-horizon moving averages.

**Summary:** Across both models, short-term returns consistently dominate feature importance, highlighting the limited but detectable short-horizon momentum effects in crypto markets.

***

## Discussion

### Predictability in Crypto Markets

The results show that cryptocurrency returns contain weak but usable short-term structure, with Random Forest and XGBoost achieving the best performance on very short horizons (1–3 days) with accuracies of 53–56%. However, as the forecast horizon extends to 5–7 days, model accuracy rapidly approaches the random-guessing baseline (~50%).

This horizon-dependent decay aligns with prior findings that crypto markets are highly volatile and dominated by noise and speculative behavior, which erodes predictability beyond the very short term (Chu et al., 2017).

### Model Comparison

Across all horizons, the three models perform fairly similarly. Nonlinear models (Random Forest and XGBoost) do slightly better at short horizons, but the improvement is small. For longer horizons, their performance drops and becomes similar to—or sometimes worse than—Logistic Regression.

Overall, this means that crypto price movements do not contain enough stable nonlinear patterns for complex models to take advantage of.

### Feature Importance Insights

Feature importance results from both tree-based models reveal that:

- Short-term returns (e.g., daily return, 3-day return) carry the most predictive information
- Volume and volatility contribute moderately
- Longer-term moving averages and slow trend indicators have little impact

This suggests that crypto markets exhibit very short-lived momentum, which quickly decays.

### Interpretation of Horizon Decay

The rapid decline in model accuracy for longer horizons reflects how quickly useful information fades in cryptocurrency markets. While very short-term patterns can sometimes be captured by models, these signals are easily overwhelmed as the horizon extends. Crypto prices are heavily influenced by factors that introduce unpredictability, including:

- Market sentiment, which can shift abruptly due to social media, investor mood, or crowd behavior
- Speculative trading and sudden news events
- Regulatory announcements and macroeconomic factors

***

## Conclusion

This project shows that short-term cryptocurrency price movements contain only limited predictive structure. Random Forest and XGBoost perform slightly better than Logistic Regression for 1–3 day horizons, suggesting that recent returns and volume provide weak nonlinear signals. For 5–7 day horizons, accuracy falls to near-random levels, and Logistic Regression performs just as well, indicating that longer-term signals are extremely weak and mostly linear. Feature importance results confirm that short-term returns dominate, while slower trend indicators contribute little. Overall, only very recent price behavior carries modest predictive value in crypto markets.

***

## References

[1] Rajkumar, S. (2018). *Cryptocurrency Historical Prices*. Kaggle. https://www.kaggle.com/datasets/sudalairajkumar/cryptocurrencypricehistory

[2] Urquhart, A. (2016). The inefficiency of Bitcoin. *Economics Letters, 148*, 80–82.

[3] Chu, J., Chan, S., Nadarajah, S., & Osterrieder, J. (2017). GARCH modelling of cryptocurrencies. *Journal of Risk and Financial Management, 10*(4), 17.

[4] Corbet, S., Lucey, B., Urquhart, A., & Yarovaya, L. (2019). Cryptocurrencies as a financial asset: A systematic analysis. *International Review of Financial Analysis, 62*, 182–199.

[back](./)
