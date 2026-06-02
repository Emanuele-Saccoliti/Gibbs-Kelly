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
Instead of producing only point estimates, it uses **Gibbs sampling** to generate posterior distributions of the predictive coefficients and residual return variance.

These posterior samples are then converted into posterior predictive moments:

$$
m_{1,t} = \mathbb{E}[\mu_t \mid \mathcal{D}_t],
\qquad
m_{2,t} = \mathbb{E}[\sigma_t^2 + \mu_t^2 \mid \mathcal{D}_t].
$$



The strategy uses these moments to generate dynamic fractional Kelly-style portfolio weights:

$$
w_t =
\operatorname{clip}
\left(
\lambda \frac{m_{1,t}}{m_{2,t}},
-w_{\max},
w_{\max}
\right).
$$

This means the model increases exposure when the posterior expected return is large relative to predictive risk, and reduces exposure when uncertainty or volatility is high. The leverage constraint $w_{\max}$ prevents the strategy from taking unrealistically large positions.

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
