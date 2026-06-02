
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from dataclasses import dataclass, replace
from pathlib import Path
from scipy.stats import invgamma
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = [
	'mom_20',
	'mom_60',
	'rev_5',
	'vol_20',
	'vol_60',
	'volratio_20_60',
	'dd_60',
	'trend_20_60',
]


MARKET_MIN_N_SAMPLES = 3000
MARKET_MIN_BURN_IN = 1000
ALLOWED_TARGET_HORIZONS = (5, 20)


def build_price_signals(price, target_horizon=5):
	'''
	Build causal daily signals and the cumulative forward return target.

	price: pandas Series indexed by date
	target_horizon: forecast horizon in trading days, either 5 or 20
	'''
	if target_horizon not in ALLOWED_TARGET_HORIZONS:
		raise Exception('target_horizon must be either 5 or 20 trading days')

	price = pd.Series(price, dtype=float).dropna().sort_index()
	log_price = np.log(price)
	log_return = log_price.diff()

	df = pd.DataFrame(index=price.index)
	df['price'] = price
	df['log_return'] = log_return
	df['mom_20'] = log_price.diff(20)
	df['mom_60'] = log_price.diff(60)
	df['rev_5'] = -log_price.diff(5)
	df['vol_20'] = log_return.rolling(20).std(ddof=0)
	df['vol_60'] = log_return.rolling(60).std(ddof=0)
	df['volratio_20_60'] = df['vol_20'] / df['vol_60'].clip(lower=1e-12)
	df['dd_60'] = price / price.rolling(60).max() - 1
	df['trend_20_60'] = np.log(price.rolling(20).mean() / price.rolling(60).mean())
	df['next_day_return'] = log_return.shift(-1)
	df['target_return'] = log_price.shift(-target_horizon) - log_price
	return df.dropna(subset=FEATURE_COLUMNS + ['next_day_return', 'target_return']).copy()


class BayesianMultiSignalModel(object):
	'''
		The model estimates:
			x[t + h] = alpha + z[t].T beta + epsilon[t + h]
			epsilon[t + h] ~ StudentT(nu, 0, sigma2)

		Gibbs samples are converted into posterior predictive moments:
			m1 = E[mu | data]
			m2 = E[sigma2 * nu / (nu - 2) + mu**2 | data]

	The final fractional Kelly-style weight is:
		w = clip(kelly_fraction * m1 / m2, -max_w, max_w)
	'''

	def __init__(
		self,
		max_w=1.0,
		kelly_fraction=0.25,
		n_samples=3000,
		burn_in=1000,
			theta_prior_scale=10.0,
			a0=2.0,
			b0=1e-4,
			student_t_df=5.0,
			random_state=42,
	):
		if max_w is not None and max_w <= 0:
			raise Exception('max_w must be positive or None')
		if kelly_fraction <= 0:
			raise Exception('kelly_fraction must be positive')
		if n_samples <= 0 or burn_in < 0:
			raise Exception('n_samples must be positive and burn_in cannot be negative')
		if theta_prior_scale <= 0 or a0 <= 0 or b0 <= 0:
			raise Exception('prior hyperparameters must be positive')
		if student_t_df <= 2:
			raise Exception('student_t_df must be greater than 2')

		self.max_w = max_w
		self.kelly_fraction = float(kelly_fraction)
		self.n_samples = int(n_samples)
		self.burn_in = int(burn_in)
		self.theta_prior_scale = float(theta_prior_scale)
		self.a0 = float(a0)
		self.b0 = float(b0)
		self.student_t_df = float(student_t_df)
		self.random_state = random_state

		self.scaler = None
		self.theta_samples = None
		self.sigma2_samples = None
		self.theta_posterior_mean = None
		self.sigma2_posterior_mean = None
		self.latent_precision_posterior_mean = None

		self.inference_s = None
		self.inference_s_fees = None
		self.inference_lev = None
		self.inference_prob_positive = None
		self.bootstrap_sharpe_samples = None

	@staticmethod
	def _as_matrix(z):
		z = np.asarray(z, dtype=float)
		if z.ndim == 1:
			z = z[:, None]
		if z.ndim != 2:
			raise Exception('z must be a vector or a 2D signal matrix')
		if not np.all(np.isfinite(z)):
			raise Exception('z must contain only finite values')
		return z

	@staticmethod
	def _as_vector(x):
		x = np.asarray(x, dtype=float).reshape(-1)
		if x.ndim != 1 or not np.all(np.isfinite(x)):
			raise Exception('x must be a finite vector')
		return x

	@staticmethod
	def _design_matrix(z):
		z = BayesianMultiSignalModel._as_matrix(z)
		return np.column_stack((np.ones(z.shape[0]), z))

	def _check_fitted(self):
		if self.theta_samples is None or self.sigma2_samples is None:
			raise Exception('Fit the model before requesting posterior quantities')

	def _fit_standardized(self, z, x, random_state=None):
		z = self._as_matrix(z)
		x = self._as_vector(x)
		if z.shape[0] != x.size:
			raise Exception('z and x must contain the same number of observations')

		design = self._design_matrix(z)
		n_obs, n_theta = design.shape
		prior_precision = np.eye(n_theta) / np.power(self.theta_prior_scale, 2)
		rng = np.random.default_rng(self.random_state if random_state is None else random_state)
		theta = np.linalg.lstsq(design, x, rcond=None)[0]
		residuals = x - design @ theta
		sigma2 = max(float(np.mean(np.power(residuals, 2))), 1e-12)
		latent_precision = np.ones(n_obs)

		theta_samples = np.empty((self.n_samples, n_theta))
		sigma2_samples = np.empty(self.n_samples)
		latent_precision_sum = np.zeros(n_obs)
		posterior_shape = self.a0 + 0.5 * (n_obs + n_theta)

		for draw in range(self.n_samples + self.burn_in):
			weighted_design = latent_precision[:, None] * design
			posterior_precision = design.T @ weighted_design + prior_precision
			posterior_covariance_base = np.linalg.inv(posterior_precision)
			posterior_mean = np.linalg.solve(
				posterior_precision,
				design.T @ (latent_precision * x),
			)
			theta = rng.multivariate_normal(
				mean=posterior_mean,
				cov=sigma2 * posterior_covariance_base,
			)
			residuals = x - design @ theta
			prior_quadratic = theta @ prior_precision @ theta
			posterior_scale = self.b0 + 0.5 * (
				np.sum(latent_precision * np.power(residuals, 2)) + prior_quadratic
			)
			sigma2 = float(
				invgamma.rvs(
					a=posterior_shape,
					scale=posterior_scale,
					random_state=rng,
				)
			)
			latent_precision = rng.gamma(
				shape=0.5 * (self.student_t_df + 1),
				scale=2.0 / (
					self.student_t_df + np.power(residuals, 2) / sigma2
				),
			)

			if draw >= self.burn_in:
				saved_draw = draw - self.burn_in
				theta_samples[saved_draw] = theta
				sigma2_samples[saved_draw] = sigma2
				latent_precision_sum += latent_precision

		return {
			'theta_samples': theta_samples,
			'sigma2_samples': sigma2_samples,
			'theta_posterior_mean': np.mean(theta_samples, axis=0),
			'sigma2_posterior_mean': float(np.mean(sigma2_samples)),
			'latent_precision_posterior_mean': latent_precision_sum / self.n_samples,
		}

	def _set_posterior(self, posterior):
		self.theta_samples = posterior['theta_samples']
		self.sigma2_samples = posterior['sigma2_samples']
		self.theta_posterior_mean = posterior['theta_posterior_mean']
		self.sigma2_posterior_mean = posterior['sigma2_posterior_mean']
		self.latent_precision_posterior_mean = posterior['latent_precision_posterior_mean']

	def _student_t_variance_samples(self):
		return self.sigma2_samples * self.student_t_df / (self.student_t_df - 2)

	def _posterior_mean_samples_standardized(self, z):
		self._check_fitted()
		return self._design_matrix(z) @ self.theta_samples.T

	def _posterior_predictive_moments_standardized(self, z):
		mu_samples = self._posterior_mean_samples_standardized(z)
		m1 = np.mean(mu_samples, axis=1)
		m2 = np.mean(self._student_t_variance_samples()[None, :] + np.power(mu_samples, 2), axis=1)
		return m1, m2

	def _posterior_direction_probability_standardized(self, z):
		mu_samples = self._posterior_mean_samples_standardized(z)
		return np.mean(mu_samples > 0, axis=1)

	def _weights_standardized(self, z):
		m1, m2 = self._posterior_predictive_moments_standardized(z)
		w = self.kelly_fraction * m1 / np.maximum(m2, 1e-12)
		if self.max_w is not None:
			w = np.clip(w, -self.max_w, self.max_w)
		return w

	def fit(self, z, x):
		'''
		z: numpy (n, p) array of signals, or numpy (n,) array
		x: numpy (n,) array of future returns
		'''
		z = self._as_matrix(z)
		x = self._as_vector(x)
		if z.shape[0] != x.size:
			raise Exception('z and x must contain the same number of observations')
		self.scaler = StandardScaler()
		z_standardized = self.scaler.fit_transform(z)
		self._set_posterior(self._fit_standardized(z_standardized, x))
		return self

	def posterior_mean_samples(self, z):
		self._check_fitted()
		z_standardized = self.scaler.transform(self._as_matrix(z))
		return self._posterior_mean_samples_standardized(z_standardized)

	def posterior_predictive_moments(self, z):
		mu_samples = self.posterior_mean_samples(z)
		m1 = np.mean(mu_samples, axis=1)
		m2 = np.mean(self._student_t_variance_samples()[None, :] + np.power(mu_samples, 2), axis=1)
		return m1, m2

	def posterior_direction_probability(self, z):
		return np.mean(self.posterior_mean_samples(z) > 0, axis=1)

	def predict(self, z):
		m1, _ = self.posterior_predictive_moments(z)
		return m1

	def get_weight(self, z):
		m1, m2 = self.posterior_predictive_moments(z)
		w = self.kelly_fraction * m1 / np.maximum(m2, 1e-12)
		if self.max_w is not None:
			w = np.clip(w, -self.max_w, self.max_w)
		return w

	def summary(self, feature_names=None):
		self._check_fitted()
		n_features = self.theta_samples.shape[1] - 1
		if feature_names is None:
			feature_names = ['signal_' + str(i + 1) for i in range(n_features)]
		if len(feature_names) != n_features:
			raise Exception('feature_names must match the number of fitted signals')

		return pd.DataFrame({
			'parameter': ['intercept'] + list(feature_names),
			'posterior_mean': np.mean(self.theta_samples, axis=0),
			'posterior_std': np.std(self.theta_samples, axis=0),
			'credible_2.5pct': np.quantile(self.theta_samples, 0.025, axis=0),
			'credible_97.5pct': np.quantile(self.theta_samples, 0.975, axis=0),
			'prob_positive': np.mean(self.theta_samples > 0, axis=0),
		})

	def view(self, feature_names=None):
		print('** Bayesian Multi-Signal Model **')
		print(self.summary(feature_names).to_string(index=False))
		print()
		print('-> posterior sigma2 mean =', self.sigma2_posterior_mean)
		print('-> student_t_df =', self.student_t_df)
		print('-> max_w =', self.max_w)
		print('-> kelly_fraction =', self.kelly_fraction)
		print()

	@staticmethod
	def stationary_bootstrap_sharpe(s, n_boot=1000, avg_block_size=22, random_state=None):
		s = np.asarray(s, dtype=float).reshape(-1)
		s = s[np.isfinite(s)]
		l = s.size
		if l < 2:
			raise Exception('s must contain at least 2 observations')
		if n_boot <= 0:
			raise Exception('n_boot must be positive')
		if avg_block_size <= 0:
			raise Exception('avg_block_size must be positive')

		rng = np.random.default_rng(random_state)
		restart_probability = 1.0 / float(avg_block_size)
		boot_samples = np.empty(n_boot)

		for b in range(n_boot):
			idx = np.empty(l, dtype=int)
			idx[0] = rng.integers(0, l)

			for t in range(1, l):
				if rng.random() < restart_probability:
					idx[t] = rng.integers(0, l)
				else:
					idx[t] = (idx[t - 1] + 1) % l

			s_boot = s[idx]
			std_boot = np.std(s_boot)
			if std_boot > 0:
				boot_samples[b] = np.mean(s_boot) / std_boot
			else:
				boot_samples[b] = np.nan

		return boot_samples[np.isfinite(boot_samples)]

	def inference_rolling(
		self,
		z,
		x,
		k_folds=8,
		n_paths=1,
		cv_burn_fraction=0.1,
		min_train_size=50,
		n_boot=5000,
		pct_fee=0,
		sr_mult=np.sqrt(250),
		view=True,
		avg_block_size=22,
	):
		'''
		Walk-forward evaluation.

		n_paths is retained for interface compatibility. Posterior Gibbs draws
		already describe parameter uncertainty, so one realized strategy path
		is evaluated for each test period.
		'''
		z = self._as_matrix(z)
		x = self._as_vector(x)
		if z.shape[0] != x.size:
			raise Exception('z and x must contain the same number of observations')

		idx = np.arange(x.size, dtype=int)
		folds_idx = np.array_split(idx, k_folds)
		s = np.full(x.size, np.nan)
		lev = np.full(x.size, np.nan)
		prob_positive = np.full(x.size, np.nan)

		cv_theta = []
		cv_sigma2 = []

		for i, test_idx in enumerate(folds_idx):
			if test_idx.size == 0:
				continue

			embargo_size = int(cv_burn_fraction * test_idx.size)
			train_end = test_idx[0] - embargo_size
			if train_end < min_train_size:
				continue

			train_idx = np.arange(train_end, dtype=int)
			scaler = StandardScaler()
			z_train = scaler.fit_transform(z[train_idx])
			z_test = scaler.transform(z[test_idx])
			posterior = self._fit_standardized(
				z_train,
				x[train_idx],
				random_state=None if self.random_state is None else self.random_state + i,
			)

			self.scaler = scaler
			self._set_posterior(posterior)
			m1, m2 = self._posterior_predictive_moments_standardized(z_test)
			w = self.kelly_fraction * m1 / np.maximum(m2, 1e-12)
			if self.max_w is not None:
				w = np.clip(w, -self.max_w, self.max_w)

			lev[test_idx] = w
			s[test_idx] = x[test_idx] * w
			prob_positive[test_idx] = self._posterior_direction_probability_standardized(z_test)
			cv_theta.append(posterior['theta_samples'])
			cv_sigma2.append(posterior['sigma2_samples'])

		valid_rows = np.isfinite(s)
		s = s[valid_rows]
		lev = lev[valid_rows]
		prob_positive = prob_positive[valid_rows]
		if s.size == 0:
			raise Exception('No valid folds: Increase sample size or reduce min_train_size/the embargo.')

		s_fees = s - np.abs(lev) * pct_fee
		b_samples = self.stationary_bootstrap_sharpe(
			s,
			n_boot=n_boot,
			avg_block_size=avg_block_size,
			random_state=self.random_state,
		)
		b_samples *= sr_mult
		valid = np.quantile(b_samples, 0.05) > 0

		self.inference_s = np.copy(s)
		self.inference_s_fees = np.copy(s_fees)
		self.inference_lev = np.copy(lev)
		self.inference_prob_positive = np.copy(prob_positive)
		self.bootstrap_sharpe_samples = np.copy(b_samples)

		if view:
			cv_theta = np.vstack(cv_theta)
			cv_sigma2 = np.hstack(cv_sigma2)
			sharpe = sr_mult * np.mean(s) / np.std(s)
			sharpe_fees = sr_mult * np.mean(s_fees) / np.std(s_fees)

			print('-> ACCEPT STRATEGY' if valid else '-> REJECT STRATEGY')
			print('** Summary **')
			print('Return: ', np.power(sr_mult, 2) * np.mean(s))
			print('Standard deviation: ', sr_mult * np.std(s))
			print('Sharpe: ', sharpe)
			print()
			print('Return w/ fees: ', np.power(sr_mult, 2) * np.mean(s_fees))
			print('Standard deviation w/ fees: ', sr_mult * np.std(s_fees))
			print('Sharpe w /fees: ', sharpe_fees)
			print('Bootstrap 5% Sharpe: ', np.quantile(b_samples, 0.05))
			print('Bootstrap P(Sharpe < 0): ', np.mean(b_samples < 0))
			print('**')

			plt.title('Equity curves')
			plt.plot(np.cumsum(s), color='g')
			plt.grid(True)
			plt.show()

			plt.title('Equity curves w/ fees')
			plt.plot(np.cumsum(s_fees), color='g')
			plt.grid(True)
			plt.show()

			plt.title('Leverage')
			plt.plot(lev, color='r')
			plt.grid(True)
			plt.show()

			plt.title('Posterior probability of positive conditional return')
			plt.plot(prob_positive, color='b')
			plt.axhline(0.5, color='k', linestyle='--')
			plt.ylim(0, 1)
			plt.grid(True)
			plt.show()

			plt.title('SR stationary-bootstrap distribution')
			plt.hist(b_samples, density=True)
			plt.axvline(0, color='k', linestyle='--')
			plt.grid(True)
			plt.show()

			plt.title('Strategy returns distribution')
			plt.hist(s, bins=100, density=True)
			plt.grid(True)
			plt.show()

			plt.title('Intercept posterior distribution')
			plt.hist(cv_theta[:, 0], density=True)
			plt.grid(True)
			plt.show()

			fig, axes = plt.subplots(cv_theta.shape[1] - 1, 1, figsize=(8, 2.5 * (cv_theta.shape[1] - 1)))
			axes = np.atleast_1d(axes)
			for i, ax in enumerate(axes):
				ax.hist(cv_theta[:, i + 1], density=True)
				ax.set_title('Beta ' + str(i + 1) + ' posterior distribution')
				ax.grid(True)
			plt.tight_layout()
			plt.show()

			plt.title('Scale posterior distribution')
			plt.hist(np.sqrt(cv_sigma2), density=True)
			plt.grid(True)
			plt.show()

		return valid

	def inference(self, z, x, **kwargs):
		return self.inference_rolling(z, x, **kwargs)

	def save(self, filepath):
		if '.pkl' not in filepath:
			filepath += '.pkl'
		with open(filepath, 'wb') as f:
			pickle.dump(self.__dict__, f, pickle.HIGHEST_PROTOCOL)

	def load(self, filepath):
		with open(filepath, 'rb') as f:
			self.__dict__.update(pickle.load(f))
		return self


@dataclass(frozen=True)
class MarketBacktestConfig:
	tickers: tuple = (
		'SPY',
		'QQQ',
		'TLT',
		'GLD',
		'HYG',
		'BTC-USD',
	)
	start_date: str = '2015-01-01'
	end_date: str = None
	target_horizon: int = 5
	n_samples: int = 3000
	burn_in: int = 1000
	theta_prior_scale: float = 10.0
	a0: float = 2.0
	b0: float = 1e-4
	student_t_df: float = 5.0
	random_state: int = 42
	kelly_fraction: float = 0.25
	max_w: float = 1.0
	min_train_size: int = 500
	test_size: int = 125
	embargo_size: int = 5
	n_boot: int = 2000
	avg_block_size: float = 22
	periods_per_year: int = 252
	fee_bps: float = 0.0
	results_dir: str = 'results_market'
	show_plots: bool = True


def quick_market_config():
	return MarketBacktestConfig(
		tickers=('SPY', 'QQQ'),
		start_date='2022-01-01',
		n_samples=3000,
		burn_in=1000,
		min_train_size=250,
		test_size=125,
		n_boot=300,
		results_dir='results_market_quick',
	)


def full_market_config():
	return MarketBacktestConfig(
		start_date='2010-01-01',
		n_samples=3000,
		burn_in=1000,
		n_boot=5000,
		results_dir='results_market_full',
	)


def _ticker_seed(ticker):
	return sum((i + 1) * ord(character) for i, character in enumerate(ticker))


def validate_market_config(config):
	if config.target_horizon not in ALLOWED_TARGET_HORIZONS:
		raise Exception('target_horizon must be either 5 or 20 trading days')
	if config.n_samples < MARKET_MIN_N_SAMPLES:
		raise Exception('n_samples must be at least ' + str(MARKET_MIN_N_SAMPLES))
	if config.burn_in < MARKET_MIN_BURN_IN:
		raise Exception('burn_in must be at least ' + str(MARKET_MIN_BURN_IN))
	if config.student_t_df <= 2:
		raise Exception('student_t_df must be greater than 2')
	return config


def _extract_yahoo_price(raw, ticker):
	if raw.empty:
		raise Exception('No Yahoo Finance data returned for ' + ticker)

	if isinstance(raw.columns, pd.MultiIndex):
		for candidate in ('Adj Close', 'Close'):
			if candidate in raw.columns.get_level_values(0):
				selected = raw[candidate]
				if isinstance(selected, pd.DataFrame):
					if ticker in selected.columns:
						selected = selected[ticker]
					else:
						selected = selected.iloc[:, 0]
				return selected.rename('price')
	else:
		for candidate in ('Adj Close', 'Close'):
			if candidate in raw.columns:
				return raw[candidate].rename('price')

	raise Exception('Neither adjusted close nor close is available for ' + ticker)


def download_yahoo_prices(config):
	import yfinance as yf

	prices = {}
	for ticker in config.tickers:
		print('Downloading', ticker, 'from Yahoo Finance...')
		raw = yf.download(
			ticker,
			start=config.start_date,
			end=config.end_date,
			auto_adjust=False,
			progress=False,
			threads=False,
		)
		price = _extract_yahoo_price(raw, ticker).dropna().astype(float)
		price.index = pd.to_datetime(price.index).tz_localize(None)
		if price.empty:
			raise Exception('No usable Yahoo Finance prices returned for ' + ticker)
		prices[ticker] = price
		print('->', len(price), 'daily prices')
	return prices


def walk_forward_splits(n_obs, min_train_size, test_size, embargo_size):
	if min_train_size <= 0 or test_size <= 0 or embargo_size < 0:
		raise Exception('Split sizes must be positive and embargo_size cannot be negative')

	fold = 0
	while True:
		train_end = min_train_size + fold * test_size
		test_start = train_end + embargo_size
		if test_start >= n_obs:
			break
		test_end = min(test_start + test_size, n_obs)
		yield fold, np.arange(train_end, dtype=int), np.arange(test_start, test_end, dtype=int)
		fold += 1


def run_market_asset_backtest(price, ticker, config):
	df = build_price_signals(price, target_horizon=config.target_horizon)
	z = df[FEATURE_COLUMNS].to_numpy(dtype=float)
	x = df['target_return'].to_numpy(dtype=float)
	next_day_return = df['next_day_return'].to_numpy(dtype=float)
	fold_results = []
	effective_embargo_size = max(config.embargo_size, config.target_horizon)

	for fold, train_idx, test_idx in walk_forward_splits(
		len(df),
		config.min_train_size,
		config.test_size,
		effective_embargo_size,
	):
		print(
			'->',
			ticker,
			'fold',
			fold,
			'train',
			len(train_idx),
			'test',
			len(test_idx),
		)
		model = BayesianMultiSignalModel(
			max_w=config.max_w,
			kelly_fraction=config.kelly_fraction,
			n_samples=config.n_samples,
			burn_in=config.burn_in,
			theta_prior_scale=config.theta_prior_scale,
			a0=config.a0,
			b0=config.b0,
			student_t_df=config.student_t_df,
			random_state=config.random_state + _ticker_seed(ticker) + fold,
		)
		model.fit(z[train_idx], x[train_idx])
		m1, m2 = model.posterior_predictive_moments(z[test_idx])
		prob_positive = model.posterior_direction_probability(z[test_idx])
		weight = model.get_weight(z[test_idx])

		fold_results.append(pd.DataFrame({
			'date': df.index[test_idx],
			'ticker': ticker,
			'fold': fold,
			'target_horizon': config.target_horizon,
			'student_t_df': config.student_t_df,
			'target_return': x[test_idx],
			'next_day_return': next_day_return[test_idx],
			'posterior_m1': m1,
			'posterior_m2': m2,
			'posterior_prob_positive': prob_positive,
			'weight': weight,
			'strategy_return_gross': weight * next_day_return[test_idx],
		}))

	if len(fold_results) == 0:
		raise Exception(
			'No valid folds for '
			+ ticker
			+ ': increase the history or reduce min_train_size'
		)

	result = pd.concat(fold_results, ignore_index=True).sort_values('date')
	result['turnover'] = result['weight'].diff().abs()
	result.loc[result.index[0], 'turnover'] = abs(result.loc[result.index[0], 'weight'])
	result['transaction_cost'] = result['turnover'] * config.fee_bps / 10000
	result['strategy_return_net'] = (
		result['strategy_return_gross'] - result['transaction_cost']
	)
	return result


def run_market_backtest(config, prices=None):
	validate_market_config(config)
	if prices is None:
		prices = download_yahoo_prices(config)

	results = []
	for ticker in config.tickers:
		if ticker not in prices:
			raise Exception('Missing prices for ' + ticker)
		print()
		print('Backtesting', ticker)
		results.append(run_market_asset_backtest(prices[ticker], ticker, config))

	return (
		pd.concat(results, ignore_index=True)
		.sort_values(['date', 'ticker'])
		.reset_index(drop=True)
	)


def aggregate_market_portfolio(results):
	return (
		results.groupby('date', sort=True)
		.agg(
			model_target_return=('target_return', 'mean'),
			buy_and_hold_return=('next_day_return', 'mean'),
			strategy_return_gross=('strategy_return_gross', 'mean'),
			strategy_return_net=('strategy_return_net', 'mean'),
			gross_exposure=('weight', lambda value: np.mean(np.abs(value))),
			net_exposure=('weight', 'mean'),
			turnover=('turnover', 'mean'),
			transaction_cost=('transaction_cost', 'mean'),
			active_assets=('ticker', 'nunique'),
		)
		.reset_index()
	)


def _finite_vector(values):
	values = np.asarray(values, dtype=float).reshape(-1)
	return values[np.isfinite(values)]


def annualized_return(values, periods_per_year=252):
	values = _finite_vector(values)
	return float(periods_per_year * np.mean(values))


def annualized_volatility(values, periods_per_year=252):
	values = _finite_vector(values)
	return float(np.sqrt(periods_per_year) * np.std(values))


def sharpe_ratio(values, periods_per_year=252):
	volatility = annualized_volatility(values, periods_per_year)
	if volatility <= 0:
		return np.nan
	return annualized_return(values, periods_per_year) / volatility


def max_drawdown(values):
	values = _finite_vector(values)
	equity = np.exp(np.cumsum(values))
	return float(np.min(equity / np.maximum.accumulate(equity) - 1))


def summarize_returns(name, values, periods_per_year=252):
	values = _finite_vector(values)
	return {
		'strategy': name,
		'annualized_return': annualized_return(values, periods_per_year),
		'annualized_volatility': annualized_volatility(values, periods_per_year),
		'sharpe': sharpe_ratio(values, periods_per_year),
		'max_drawdown': max_drawdown(values),
		'hit_rate': float(np.mean(values > 0)),
		'n_observations': len(values),
	}


def summarize_bootstrap(bootstrap_samples):
	bootstrap_samples = _finite_vector(bootstrap_samples)
	return pd.DataFrame([{
		'boot_mean_sharpe': float(np.mean(bootstrap_samples)),
		'boot_5pct_sharpe': float(np.quantile(bootstrap_samples, 0.05)),
		'boot_50pct_sharpe': float(np.quantile(bootstrap_samples, 0.50)),
		'boot_95pct_sharpe': float(np.quantile(bootstrap_samples, 0.95)),
		'prob_sharpe_below_zero': float(np.mean(bootstrap_samples < 0)),
	}])


def build_per_asset_sharpe_report(results, portfolio, config):
	full_portfolio_sharpe = sharpe_ratio(
		portfolio['strategy_return_net'],
		config.periods_per_year,
	)
	active_assets_by_date = results.groupby('date')['ticker'].nunique()
	portfolio_dates = pd.Index(portfolio['date'])
	rows = []

	for ticker, ticker_results in results.groupby('ticker'):
		ticker_results = ticker_results.sort_values('date').copy()
		bootstrap_samples = BayesianMultiSignalModel.stationary_bootstrap_sharpe(
			ticker_results['strategy_return_net'],
			n_boot=config.n_boot,
			avg_block_size=config.avg_block_size,
			random_state=config.random_state + _ticker_seed(ticker),
		)
		bootstrap_samples *= np.sqrt(config.periods_per_year)
		bootstrap = summarize_bootstrap(bootstrap_samples).iloc[0]

		remaining_results = results.loc[results['ticker'] != ticker]
		if remaining_results.empty:
			leave_one_out_sharpe = np.nan
		else:
			leave_one_out_returns = (
				remaining_results.groupby('date', sort=True)['strategy_return_net']
				.mean()
			)
			leave_one_out_sharpe = sharpe_ratio(
				leave_one_out_returns,
				config.periods_per_year,
			)

		contribution_by_date = (
			ticker_results.set_index('date')['strategy_return_net']
			/ active_assets_by_date
		)
		annualized_return_contribution = config.periods_per_year * (
			contribution_by_date.reindex(portfolio_dates, fill_value=0).sum()
			/ len(portfolio_dates)
		)

		rows.append({
			'ticker': ticker,
			'likelihood': 'student_t',
			'student_t_df': config.student_t_df,
			'n_observations': len(ticker_results),
			'annualized_return_contribution': float(annualized_return_contribution),
			'annualized_return_gross': annualized_return(
				ticker_results['strategy_return_gross'],
				config.periods_per_year,
			),
			'annualized_return_net': annualized_return(
				ticker_results['strategy_return_net'],
				config.periods_per_year,
			),
			'annualized_volatility_net': annualized_volatility(
				ticker_results['strategy_return_net'],
				config.periods_per_year,
			),
			'sharpe_gross': sharpe_ratio(
				ticker_results['strategy_return_gross'],
				config.periods_per_year,
			),
			'sharpe_net': sharpe_ratio(
				ticker_results['strategy_return_net'],
				config.periods_per_year,
			),
			'buy_and_hold_sharpe': sharpe_ratio(
				ticker_results['next_day_return'],
				config.periods_per_year,
			),
			'max_drawdown_net': max_drawdown(ticker_results['strategy_return_net']),
			'hit_rate_net': float(np.mean(ticker_results['strategy_return_net'] > 0)),
			'average_abs_exposure': float(np.mean(np.abs(ticker_results['weight']))),
			'average_turnover': float(np.mean(ticker_results['turnover'])),
			'boot_5pct_sharpe_net': float(bootstrap['boot_5pct_sharpe']),
			'boot_50pct_sharpe_net': float(bootstrap['boot_50pct_sharpe']),
			'prob_sharpe_below_zero_net': float(bootstrap['prob_sharpe_below_zero']),
			'leave_one_out_portfolio_sharpe_net': float(leave_one_out_sharpe),
			'sharpe_change_without_asset': float(
				leave_one_out_sharpe - full_portfolio_sharpe
			),
		})

	return (
		pd.DataFrame(rows)
		.sort_values('sharpe_net', ascending=False)
		.reset_index(drop=True)
	)


def _finish_plot(filepath, show_plots):
	plt.tight_layout()
	plt.savefig(filepath, dpi=150)
	if show_plots:
		plt.show()
	else:
		plt.close()


def make_market_plots(results, portfolio, bootstrap_samples, config):
	results_dir = Path(config.results_dir)
	results_dir.mkdir(parents=True, exist_ok=True)

	plt.figure(figsize=(11, 5))
	gross_equity = np.exp(portfolio['strategy_return_gross'].cumsum())
	net_equity = np.exp(portfolio['strategy_return_net'].cumsum())
	if np.allclose(gross_equity, net_equity):
		plt.plot(
			portfolio['date'],
			gross_equity,
			label='Bayesian Kelly gross = net (fee_bps=0)',
			color='tab:blue',
		)
	else:
		plt.plot(
			portfolio['date'],
			gross_equity,
			label='Bayesian Kelly gross',
			color='tab:blue',
		)
		plt.plot(
			portfolio['date'],
			net_equity,
			label='Bayesian Kelly net',
			color='tab:orange',
			linestyle='--',
		)
	plt.plot(
		portfolio['date'],
		np.exp(portfolio['buy_and_hold_return'].cumsum()),
		label='Equal-weight buy and hold',
		alpha=0.8,
	)
	plt.title('Real-market portfolio equity curves')
	plt.ylabel('Growth of 1')
	plt.grid(True)
	plt.legend()
	_finish_plot(results_dir / 'market_equity_curve.png', config.show_plots)

	plt.figure(figsize=(11, 5))
	weight_pivot = results.pivot(index='date', columns='ticker', values='weight')
	plt.plot(weight_pivot.index, weight_pivot)
	plt.title('Bayesian Kelly weights by asset')
	plt.ylabel('Weight')
	plt.grid(True)
	plt.legend(weight_pivot.columns, ncol=2)
	_finish_plot(results_dir / 'market_weights_by_asset.png', config.show_plots)

	diagnostic_ticker = 'SPY' if 'SPY' in set(results['ticker']) else results['ticker'].iloc[0]
	diagnostic = results.loc[results['ticker'] == diagnostic_ticker]
	plt.figure(figsize=(11, 4))
	plt.plot(diagnostic['date'], diagnostic['posterior_prob_positive'])
	plt.axhline(0.5, color='k', linestyle='--')
	plt.ylim(0, 1)
	plt.title(diagnostic_ticker + ' posterior probability of positive conditional return')
	plt.ylabel('P(mu > 0 | data)')
	plt.grid(True)
	_finish_plot(results_dir / 'market_spy_posterior_confidence.png', config.show_plots)

	plt.figure(figsize=(9, 4))
	plt.hist(bootstrap_samples, bins=40, density=True)
	plt.axvline(0, color='k', linestyle='--')
	plt.title('Real-market portfolio stationary-bootstrap Sharpe')
	plt.xlabel('Annualized Sharpe')
	plt.grid(True)
	_finish_plot(results_dir / 'market_bootstrap_sharpe.png', config.show_plots)

	plt.figure(figsize=(9, 4))
	plt.hist(portfolio['strategy_return_net'], bins=100, density=True)
	plt.title('Real-market portfolio net returns distribution')
	plt.grid(True)
	_finish_plot(results_dir / 'market_returns_distribution.png', config.show_plots)


def save_market_report(results, portfolio, config):
	results_dir = Path(config.results_dir)
	results_dir.mkdir(parents=True, exist_ok=True)

	results.to_csv(results_dir / 'market_backtest_results.csv', index=False)
	portfolio.to_csv(results_dir / 'market_portfolio_returns.csv', index=False)

	portfolio_summary = pd.DataFrame([
		summarize_returns(
			'buy_and_hold_equal_weight',
			portfolio['buy_and_hold_return'],
			config.periods_per_year,
		),
		summarize_returns(
			'bayesian_kelly_gross',
			portfolio['strategy_return_gross'],
			config.periods_per_year,
		),
		summarize_returns(
			'bayesian_kelly_net',
			portfolio['strategy_return_net'],
			config.periods_per_year,
		),
	])
	portfolio_summary.insert(1, 'likelihood', 'student_t')
	portfolio_summary.insert(2, 'student_t_df', config.student_t_df)
	portfolio_summary.insert(3, 'target_horizon', config.target_horizon)
	portfolio_summary.to_csv(results_dir / 'market_portfolio_summary.csv', index=False)

	per_asset_summary = build_per_asset_sharpe_report(results, portfolio, config)
	per_asset_summary.to_csv(results_dir / 'market_per_asset_summary.csv', index=False)
	per_asset_summary.to_csv(results_dir / 'market_internal_sharpe_by_asset.csv', index=False)

	bootstrap_samples = BayesianMultiSignalModel.stationary_bootstrap_sharpe(
		portfolio['strategy_return_net'],
		n_boot=config.n_boot,
		avg_block_size=config.avg_block_size,
		random_state=config.random_state,
	)
	bootstrap_samples *= np.sqrt(config.periods_per_year)
	bootstrap_summary = summarize_bootstrap(bootstrap_samples)
	bootstrap_summary.to_csv(results_dir / 'market_bootstrap_sharpe_summary.csv', index=False)

	make_market_plots(results, portfolio, bootstrap_samples, config)

	print()
	print('Likelihood: Student-t with nu =', config.student_t_df)
	print('Forecast target horizon:', config.target_horizon, 'trading days')
	if config.fee_bps == 0:
		print('Transaction costs: fee_bps=0, so gross and net equity curves coincide.')
	print()
	print('** Real-Market Portfolio Summary **')
	print(portfolio_summary.to_string(index=False))
	print()
	print('** Real-Market Bootstrap Sharpe Summary **')
	print(bootstrap_summary.to_string(index=False))
	print()
	print('** Internal Sharpe Test By Asset **')
	print(
		per_asset_summary[[
			'ticker',
			'annualized_return_contribution',
			'sharpe_net',
			'buy_and_hold_sharpe',
			'boot_5pct_sharpe_net',
			'prob_sharpe_below_zero_net',
			'sharpe_change_without_asset',
		]].to_string(index=False)
	)
	print()
	print('Interpretation: sharpe_change_without_asset > 0 means the asset reduced portfolio Sharpe.')
	print()
	print('Saved results in:', results_dir.resolve())
	return {
		'portfolio_summary': portfolio_summary,
		'per_asset_summary': per_asset_summary,
		'bootstrap_summary': bootstrap_summary,
		'bootstrap_samples': bootstrap_samples,
	}


def run_real_market(config=None, prices=None):
	config = MarketBacktestConfig() if config is None else config
	results = run_market_backtest(config, prices=prices)
	portfolio = aggregate_market_portfolio(results)
	report = save_market_report(results, portfolio, config)
	return {
		'config': config,
		'results': results,
		'portfolio': portfolio,
		**report,
	}


def parse_command_line():
	parser = argparse.ArgumentParser(
		description='Bayesian multi-signal Kelly-style real-market backtest'
	)
	mode = parser.add_mutually_exclusive_group()
	mode.add_argument('--quick', action='store_true', help='run a fast real-market smoke test')
	mode.add_argument('--full', action='store_true', help='run the exhaustive 2010-present configuration')
	mode.add_argument('--synthetic', action='store_true', help='run the original synthetic demonstration')
	parser.add_argument('--no-show', action='store_true', help='save plots without opening windows')
	parser.add_argument('--results-dir', help='override the output directory')
	parser.add_argument('--start-date', help='override the Yahoo Finance start date')
	parser.add_argument('--end-date', help='override the Yahoo Finance end date')
	parser.add_argument('--tickers', nargs='+', help='override Yahoo Finance ticker symbols')
	parser.add_argument('--target-horizon', type=int, choices=ALLOWED_TARGET_HORIZONS, help='forecast horizon in trading days')
	parser.add_argument('--fee-bps', type=float, help='transaction cost per unit turnover in basis points')
	parser.add_argument('--n-samples', type=int, help='override saved Gibbs draws per fold')
	parser.add_argument('--burn-in', type=int, help='override Gibbs burn-in draws per fold')
	parser.add_argument('--student-t-df', type=float, help='override Student-t degrees of freedom; must be greater than 2')
	parser.add_argument('--avg-block-size', type=float, help='override stationary-bootstrap average block length')
	return parser.parse_args()


def market_config_from_args(args):
	config = full_market_config() if args.full else MarketBacktestConfig()
	if args.quick:
		config = quick_market_config()

	overrides = {}
	for argument, attribute in (
		(args.results_dir, 'results_dir'),
		(args.start_date, 'start_date'),
		(args.end_date, 'end_date'),
		(args.target_horizon, 'target_horizon'),
		(args.fee_bps, 'fee_bps'),
		(args.n_samples, 'n_samples'),
		(args.burn_in, 'burn_in'),
		(args.student_t_df, 'student_t_df'),
		(args.avg_block_size, 'avg_block_size'),
	):
		if argument is not None:
			overrides[attribute] = argument
	if args.tickers is not None:
		overrides['tickers'] = tuple(args.tickers)
	if args.no_show:
		overrides['show_plots'] = False
	return replace(config, **overrides)


def test_bayesian(view=True):
	rng = np.random.default_rng(42)
	n = 1400
	z = rng.normal(0, 1, size=(n, 3))
	true_beta = np.array([0.04, -0.025, 0.0])
	x = 0.002 + z @ true_beta + rng.normal(0, 0.05, size=n)

	model = BayesianMultiSignalModel(
		max_w=1,
		kelly_fraction=0.25,
		n_samples=300,
		burn_in=100,
		random_state=42,
	)
	model.inference_rolling(
		z,
		x,
		k_folds=7,
		min_train_size=300,
		n_boot=500,
		sr_mult=1,
		view=view,
		avg_block_size=22,
	)
	model.fit(z, x)
	model.view(['signal_1', 'signal_2', 'irrelevant_signal'])

	z_query = rng.normal(0, 1, size=(3, 3))
	print('Test predict')
	print(model.predict(z_query))
	print('Test weights')
	print(model.get_weight(z_query))
	print('Test P(mu > 0 | data)')
	print(model.posterior_direction_probability(z_query))

	filepath = 'test_bayesian_model.pkl'
	model.save(filepath)
	tmp = BayesianMultiSignalModel().load(filepath)
	assert np.allclose(model.get_weight(z_query), tmp.get_weight(z_query))
	os.remove(filepath)
	return model

if __name__=='__main__':
	args = parse_command_line()
	if args.synthetic:
		test_bayesian(view=not args.no_show)
	else:
		run_real_market(market_config_from_args(args))
