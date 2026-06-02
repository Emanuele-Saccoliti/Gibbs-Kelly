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

## Model

For predictors \(z_t\) and next-period return \(x_{t+1}\), the model is:

\[
x_{t+1}
=
\alpha + z_t^\top \beta + \varepsilon_{t+1},
\qquad
\varepsilon_{t+1} \sim \mathcal{N}(0,\sigma^2).
\]

Instead of estimating only point coefficients, the model estimates the posterior distribution:

\[
p(\alpha,\beta,\sigma^2 \mid \mathcal{D}_t).
\]

Posterior samples are obtained using a custom Gibbs sampler.

For posterior draw \(m\):

\[
\mu_t^{(m)}
=
\alpha^{(m)}
+
z_t^\top\beta^{(m)}.
\]

The posterior expected return is:

\[
m_{1,t}
=
\mathbb{E}[\mu_t \mid \mathcal{D}_t]
\approx
\frac{1}{M}
\sum_{m=1}^{M}
\mu_t^{(m)}.
\]

The posterior predictive second moment is:

\[
m_{2,t}
=
\mathbb{E}[\sigma_t^2+\mu_t^2 \mid \mathcal{D}_t]
\approx
\frac{1}{M}
\sum_{m=1}^{M}
\left[
\sigma^{2,(m)}
+
\left(\mu_t^{(m)}\right)^2
\right].
\]

---

## Allocation Rule

The strategy uses a fractional Kelly-style allocation rule:

\[
w_t
=
\operatorname{clip}
\left(
\lambda
\frac{m_{1,t}}{m_{2,t}},
-w_{\max},
w_{\max}
\right).
\]

where:

- \(m_{1,t}\) is the posterior expected return;
- \(m_{2,t}\) is the posterior predictive second moment;
- \(\lambda\) is a fractional Kelly multiplier;
- \(w_{\max}\) is the maximum absolute leverage.

This allocation rule increases exposure when the posterior expected return is large relative to predictive risk, and reduces exposure when uncertainty or volatility is high.

---

## Why Bayesian?

The Bayesian approach is useful because it does not treat estimated coefficients as fixed.

Instead, the model accounts for uncertainty in:

- predictive coefficients;
- return variance;
- posterior expected returns;
- position sizing.

This helps avoid overconfident allocation decisions based on noisy financial time-series signals.

---

## Features

### Bayesian Modeling

- Bayesian linear regression with Normal-Inverse-Gamma prior;
- custom Gibbs sampler;
- posterior coefficient distributions;
- posterior predictive moments;
- posterior probability of positive expected return.

### Signal Engineering

- momentum signals;
- short-term reversal signals;
- realized volatility;
- volatility ratio;
- drawdown;
- trend strength.

### Allocation

- posterior predictive return estimation;
- fractional Kelly-style sizing;
- leverage constraints;
- dynamic asset-level weights.

### Backtesting

- rolling out-of-sample evaluation;
- purged walk-forward validation;
- asset-level attribution;
- equal-weight portfolio aggregation;
- benchmark comparison.

### Robustness Analysis

- block-bootstrap Sharpe inference;
- lower-tail Sharpe estimates;
- probability of negative Sharpe;
- drawdown analysis;
- hit-rate diagnostics;
- exposure and turnover monitoring.

---

## Validation Framework

The project avoids random train/test splits.

Instead, the model uses a time-series validation design:

\[
\text{past training data}
\quad \rightarrow \quad
\text{embargo}
\quad \rightarrow \quad
\text{future test data}.
\]

At each walk-forward fold:
1. The Bayesian model is fitted only on past data.
2. Posterior predictive moments are computed on the future test window.
3. Portfolio weights are generated out-of-sample.
4. Strategy returns are recorded.
5. The process rolls forward through time.

---

## Block-Bootstrap Sharpe Inference

Because financial returns are serially dependent, the project does not rely only on the realized Sharpe ratio.

Given strategy returns \((s_t)_{t=1}^{T}\), contiguous blocks are resampled to produce bootstrap strategy paths. For each bootstrap sample \(b\), the annualized Sharpe ratio is:

\[
\operatorname{SR}^{(b)}
=
\sqrt{252}
\frac{
\overline{s}^{(b)}
}{
\operatorname{std}(s^{(b)})
}.
\]

The final report includes:

\[
\mathbb{E}[\operatorname{SR}^{(b)}],
\qquad
Q_{5\%}(\operatorname{SR}^{(b)}),
\qquad
Q_{50\%}(\operatorname{SR}^{(b)}),
\qquad
Q_{95\%}(\operatorname{SR}^{(b)}),
\]

and

\[
\mathbb{P}(\operatorname{SR}^{(b)} < 0).
\]

This provides a more robust estimate of whether the observed Sharpe ratio is statistically meaningful.

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
