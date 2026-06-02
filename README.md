## Project Overview

The objective is to build a statistically robust allocation strategy that maps financial signals into dynamic portfolio weights.

For each asset, the model uses predictive signals such as:
- momentum;
- short-term reversal;
- realized volatility;
- volatility ratio;
- drawdown;
- trend strength.

The model estimates the relationship between these signals and future returns using Bayesian linear regression.

---

## Results

The strategy achieved:

| Metric | Value |
|---|---:|
| Bootstrap mean Sharpe | ~0.81 |
| Bootstrap median Sharpe | ~0.81 |
| 5% bootstrap Sharpe | ~0.42 |
| 95% bootstrap Sharpe | ~1.21 |
| Probability of negative Sharpe | ~0.0% |

These results suggest that the strategy produced a positive and statistically robust out-of-sample Sharpe under block-bootstrap resampling.



---



## Disclaimer

This project is for research and educational purposes only. It is not investment advice and should not be used for live trading without additional validation, transaction-cost modeling, liquidity analysis and risk controls.
