## Project Overview

The objective is to build a statistically robust allocation strategy that maps financial signals into dynamic portfolio weights.

For each asset, the model uses predictive signals such as:
- momentum;
- short-term mean reversion;
- realized volatility;
- volatility ratio;
- drawdown;
- trend strength.



---



## Model

For predictors $z_t$ and next-period return $x_{t+1}$, the model is:

```math
x_{t+1}
=
\alpha + z_t^\top \beta + \varepsilon_{t+1},
\qquad
\varepsilon_{t+1} \sim \mathcal{N}(0,\sigma^2).
```

Instead of estimating only point coefficients, the model estimates the posterior distribution:

```math
p(\alpha,\beta,\sigma^2 \mid \mathcal{D}_t).
```

Posterior samples are obtained using a custom Gibbs sampler.

For posterior draw $m$:

```math
\mu_t^{(m)} = \alpha^{(m)} + z_t^\top\beta^{(m)}.
```

The posterior expected return is:

```math
m_{1,t} = \mathbb{E}[\mu_t \mid \mathcal{D}_t] \approx \frac{1}{M} \sum_{m=1}^{M} \mu_t^{(m)}.
```

The posterior predictive second moment is:

```math
m_{2,t} = \mathbb{E}[\sigma_t^2+\mu_t^2 \mid \mathcal{D}_t]
\approx \frac{1}{M} \sum_{m=1}^{M} \left[ \sigma^{2,(m)} + \left(\mu_t^{(m)}\right)^2 \right].
```

---

## Allocation Rule

The strategy uses a fractional Kelly-style allocation rule:

```math
w_t
=
\text{clip}
\left(
\lambda
\frac{m_{1,t}}{m_{2,t}},
-w_{\max},
w_{\max}
\right).
```

where:

- $m_{1,t}$ is the posterior expected return;
- $m_{2,t}$ is the posterior predictive second moment;
- $\lambda$ is a fractional Kelly multiplier;
- $w_{\max}$ is the maximum absolute leverage.

This allocation rule increases exposure when the posterior expected return is large relative to predictive risk, and reduces exposure when uncertainty or volatility is high. The leverage constraint $w_{\max}$ prevents the strategy from taking unrealistically large positions.

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
