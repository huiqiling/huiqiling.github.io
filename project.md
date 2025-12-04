<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

## Cryptocurrency Price Movement Prediction Using Time-Series Machine Learning

This project explores whether short-term cryptocurrency price changes can be predicted using simple machine-learning models and basic time-series features. Using daily data for Bitcoin, Ethereum, and ten other cryptocurrencies, I built lag features, volatility measures, and rolling averages, then tested Logistic Regression, Random Forest, and XGBoost on 1, 3, 5, and 7-day prediction windows.

The models show small but consistent level of predictability for the 1 and 3 day horizons (with accuracy reaching about 0.56), while performance drops off quickly for longer forecasts. Tree-based models do slightly better for very short-term predictions, and Logistic Regression ends up being more stable for the 5 and 7 day horizons. Overall, the findings match the common idea that crypto prices can show short-lived patterns, but become much harder to predict as the time horizon gets longer.

***

## Introduction 

Cryptocurrency markets change quickly and are very volatile. Past research shows that Bitcoin and other coins often move in ways that are hard to predict. This project asks: can recent price changes help us guess short-term direction?

I use daily price data for 12 major cryptocurrencies. The goal is not to predict exact prices, but to see if machine learning can classify next-day and multi-day price direction using simple features. Stablecoins are not included because their prices are fixed and do not show real market changes.

***

## Data

### Data Overview

The data comes from Kaggle's *Cryptocurrency Historical Prices*. It has daily open, close, high, low, and volume for many coins.

I use 12 major cryptocurrencies: Bitcoin, Ethereum, Binance Coin, XRP, Litecoin, Cardano, Polkadot, Solana, Uniswap, ChainLink, Dogecoin, and Monero. Each coin is in a separate CSV file. I combine them and sort by date for feature engineering.

Stablecoins (like USDT, USDC, Tether, Wrapped Bitcoin) are not used. Their prices are fixed to fiat currencies, so they do not show real market changes. Including them would lower volatility and make prediction less meaningful.

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

### Exploratory Data Analysis

#### Figure 1: Closing Price Trends for 12 Major Cryptocurrencies (Log Scale)

![](assets/IMG/figure1_price_trends.png){: width="700" }

*Figure 1: Closing Price Trends for 12 Major Cryptocurrencies (Log Scale)*

This plot shows the long-term price evolution of 12 major cryptocurrencies on a logarithmic scale. Using a log scale makes it possible to compare assets with prices that span several orders of magnitude. We can see clear market cycles—periods of rapid appreciation followed by extended drawdowns. Bitcoin remains the most stable and highest-valued asset, while other coins such as Ethereum, Binance Coin, and Solana exhibit significant growth and volatility.

#### Figure 2: Distribution of Daily Returns for 12 Major Cryptocurrencies

![](assets/IMG/figure2_daily_returns.png){: width="600" }

*Figure 2: Distribution of Daily Returns for 12 Major Cryptocurrencies (Log Y, Density)*

This figure shows the distribution of daily returns for the 12 selected cryptocurrencies on a log-scaled y-axis. Most returns cluster around zero, but the distributions have long tails, indicating that large price jumps occur frequently.

#### Figure 3: Daily Trading Volume for 12 Major Cryptocurrencies

![](assets/IMG/figure3_trading_volume.png){: width="700" }

*Figure 3: Daily Trading Volume for 12 Major Cryptocurrencies (Log Scale)*

This figure shows the daily trading volume of cryptocurrencies on a logarithmic scale. Across all coins, trading activity rises over time, with noticeable peaks during major market cycles.

#### Figure 4: Feature vs Target Relationship

![](assets/IMG/figure4_feature_target.png){: width="100%" }

*Figure 4: Feature Distributions by Next-Day Target (Boxplots)*

This figure shows the distribution of each engineered feature grouped by the short-term target (target_1, i.e., next-day up/down). Most features show only weak separation between up and down days, which reflects how noisy and hard to predict short-term cryptocurrency movements are.

#### Figure 5: Feature Correlation Heatmap

![](assets/IMG/figure5_correlation_heatmap.png){: width="600" }

*Figure 5: Feature Correlation Heatmap for Engineered Predictors*

The correlation heatmap shows that price-related features (lag_1, lag_7, ma_7, ma_30) are highly correlated with each other. In contrast, return is almost uncorrelated with these trend features. Volatility and volume_change show very low correlations with all other variables, suggesting that they represent different dimensions of market behavior.

***

## Modelling

### Model Selection and Training Data Preparation

The exploratory analysis shows that daily price movements are hard to separate using simple patterns. Features for up and down days overlap a lot, and the short-term signals in the data are fairly weak. Because of this, the goal is not to build a perfect classifier, but to see how different models deal with noisy financial data.

Logistic Regression is used as a baseline since it shows how well a basic linear model can perform. Random Forest and XGBoost are added as more flexible models that can capture nonlinear patterns and interactions between features.

The dataset is split into 80% training set (16,603 samples) and 20% test set (4,151 samples).

### Model Training and Evaluation

#### Logistic Regression

Logistic Regression serves as a baseline for predicting next-day price movement. As a simple linear model, it shows how much predictability exists in the data.

| Horizon | Accuracy |
|---------|----------|
| 1       | 0.5312   |
| 3       | 0.5317   |
| 5       | 0.5584   |
| 7       | 0.5560   |

#### Random Forest

Random Forest combines many decision trees to capture nonlinear patterns and feature interactions.

| Horizon | Accuracy |
|---------|----------|
| 1       | 0.5616   |
| 3       | 0.5437   |
| 5       | 0.5283   |
| 7       | 0.5298   |

#### XGBoost (Extreme Gradient Boosting)

XGBoost is a gradient boosting method that builds decision trees sequentially to improve predictive accuracy.

| Horizon | Accuracy |
|---------|----------|
| 1       | 0.5606   |
| 3       | 0.5526   |
| 5       | 0.5338   |
| 7       | 0.5240   |

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

All three models (Logistic Regression, Random Forest, XGBoost) were evaluated on four forecast horizons (1, 3, 5, and 7 days). Overall performance ranges from 0.52 to 0.56, which is expected given the noisy and volatile nature of crypto markets.

### Performance by Horizon

- **1-day horizon:** Random Forest and XGBoost achieve the highest accuracy at this horizon, and both of them are better than Logistic Regression. This is consistent with the idea that most usable structure exists in the very short term.

- **3-day horizon:** Accuracy begins to decline for both Random Forest and XGBoost, and the performance gap between models narrows. XGBoost remains slightly ahead.

- **5-day and 7-day horizons:** For these longer horizons, the accuracy of Random Forest and XGBoost drops toward random-guess levels. Interestingly, Logistic Regression performs better than the nonlinear models at this stage.

### Feature Importance (Horizon 1)

![](assets/IMG/figure7_feature_importance.png){: width="700" }

*Figure 7: Feature Importance for Random Forest and XGBoost (Horizon 1)*

**Random Forest:** For the 1-day horizon, Random Forest assigns the highest importance to very short-term price changes, particularly the 3-day return (ret_3) and the most recent daily return. Volume-related features also appear near the top.

**XGBoost:** XGBoost shows a similar pattern to Random Forest. Compared with Random Forest, XGBoost places slightly more weight on individual lag features (e.g., lag_1) and longer-horizon moving averages.

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
