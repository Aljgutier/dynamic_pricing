from dataclasses import dataclass, field
import numpy as np
import pymc as pm
import random
import arviz as az

import contextlib
from pathlib import Path
import logging

# TS Dynamic Pricing Model Class


@dataclass
class PymcTsModel:
    m_a_linear: float
    sigma_a_linear: float
    m_v: float
    sigma_v: float
    lower_v: float
    CV_d: float
    p0: float
    y0: float
    price_min: float
    price_max: float
    F: float
    c: float
    sigma_log_fixed: float = None
    m_a_linear_fixed: float = None
    log_file: str = "tm_model_log.txt"
    verbose: bool = True

    random_seed: int = 42

    model: pm.Model = field(init=False)
    trace: object = field(init=False, default=None)

    # Optional: keep handles to pm.Data containers
    p_data: pm.Data = field(init=False)
    y_data: pm.Data = field(init=False)
    # y_log_data: pm.Data = field(init=False)

    def __post_init__(self):

        # quiet PyMC logging!!!!
        if not self.verbose:
            logging.getLogger("pymc").setLevel(logging.ERROR)
            logging.getLogger("pytensor").setLevel(logging.ERROR)

        # histories live outside the PyMC model context
        self.traces = []
        self.price_history = []
        self.demand_history = []

        # build model + take initial posterior
        self.init_ts_pricing_model()

        # store initial trace + initial observation
        self.traces.append(self.trace)
        self.price_history.append(self.p0)
        self.demand_history.append(self.y0)

    @contextlib.contextmanager
    def _redirect_io(self):
        if self.verbose:
            yield
            return

        Path(self.log_file).parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_file, "a", buffering=1) as f, contextlib.redirect_stdout(
            f
        ), contextlib.redirect_stderr(f):
            yield

    def init_ts_pricing_model(
        self,
        draws: int = 1000,
        tune: int = 2000,
        chains: int = 2,
        target_accept: float = 0.99,
    ):
        with pm.Model() as model:
            # ---- Prior for a (LogNormal) ----
            cv2_a = (self.sigma_a_linear / self.m_a_linear) ** 2
            sigma_loga = np.sqrt(np.log1p(cv2_a))
            mu_loga = np.log(self.m_a_linear) - 0.5 * sigma_loga**2

            # ---- Prior for v (truncated normal) ----
            v = pm.TruncatedNormal(
                "v",
                mu=self.m_v,
                sigma=self.sigma_v,
                lower=self.lower_v,
            )

            # ---- a (learned or fixed) ----
            if self.m_a_linear_fixed is None:
                a = pm.LogNormal("a", mu=mu_loga, sigma=sigma_loga)
            else:
                a = pm.Deterministic(
                    "a",
                    0 * v + self.m_a_linear_fixed,
                )
            loga = pm.Deterministic("loga", pm.math.log(a))

            # ---- Prior for sigma_log (log-space noise, learned or fixed) ----
            if self.sigma_log_fixed is None:
                sigma_log_scale = np.sqrt(np.log1p(self.CV_d**2))
                sigma_log = pm.HalfNormal(
                    "sigma_log",
                    sigma=sigma_log_scale,
                )
            else:
                sigma_log = pm.Deterministic(
                    "sigma_log",
                    0 * v + self.sigma_log_fixed,
                )
            # cv useful when sigma_log not constant
            cv = pm.Deterministic("cv", pm.math.sqrt(pm.math.exp(sigma_log**2) - 1))

            # ---- Mutable data containers ----
            self.p_data = pm.Data("p_data", np.array([self.p0], dtype=np.float64))
            self.y_data = pm.Data("y_data", np.array([self.y0], dtype=np.float64))

            # ---- Log-space likelihood (multiplicative noise) ----
            mu_log = loga - v * pm.math.log(self.p_data)
            pm.LogNormal("D_obs", mu=mu_log, sigma=sigma_log, observed=self.y_data)

            self.model = model

            with self._redirect_io():
                self.trace = pm.sample(
                    draws=draws,
                    tune=tune,
                    chains=chains,
                    target_accept=target_accept,
                    random_seed=self.random_seed,
                    progressbar=self.verbose,
                )

        return self.trace

    # bump
    def price_anchor_deviation(
        self,
        percent_change=0.1,
        anchor_p_opt_flag=True,
        max_local_price_change_tf=False,
    ):
        """
        Apply a +/- price change alternating up or down from a base price (p_opt or latest price history point).

        Args:
            percent_change (float): maximum percent change (default 0.10)

            anchor_p_opt_flag (bool): if True, use global optimal price estimate as base; if False, use latest price history point as base

            max_change_tf (bool): if True, apply the full max_percent_change; if False, apply a random percent change uniformly drawn from [0, max_percent_change]

        Returns:
            float: new price constrained to global bounds and rounded to 2 decimals
        """

        if anchor_p_opt_flag:
            # global optimal price estimate
            _p = self.price_global_popt_expected_profit()
        else:
            # adjust to max amount from latest offer
            _p = self.price_history[
                -1
            ]  # adjust on price_history as tighten is better ,

        # initialize direction if first call
        if not hasattr(self, "_last_direction"):
            self._last_direction = np.random.choice([-1, 1])
        else:
            self._last_direction *= -1  # alternate direction

        if max_local_price_change_tf:
            # max change
            _percent_change = percent_change
        else:
            # random
            _percent_change = np.random.uniform(0, percent_change)

        # print(f'\n...  _p = {_p}, percent_change = {round(percent_change,4)}, direction = {self._last_direction}')

        new_price = _p * (1 + self._last_direction * _percent_change)

        # enforce global bounds
        new_price = float(np.clip(new_price, self.price_min, self.price_max))

        return round(new_price, 2)

    def _flatten_posterior_draws(self, var_name: str) -> np.ndarray:
        """
        Return posterior draws for `var_name` as a 1D numpy array.
        Assumes init_ts_pricing_model() has already been run.
        """
        return self.trace.posterior[var_name].values.reshape(-1)

    # Offer Price
    def price_max_profit_sample(
        self, max_price_change=None, Npgrid=40, K=1, limit_price_change=True
    ):
        """
        Price that maximizes local profit given a price change constraint and K parameter sample (defaults to K = 1).
        - parameters a and v are drawn from the posterior trace (K samples)
        - optimization is inside the allowed price change region ("trust region") only
        - price that maximizes the profit given the sample parameters a, v (i.e., v is a new demand curve)

        Args:
            max_price_change (float)): should be a percent (<= 1)
            Npgrid (int, optional): Number of price grid points. Defaults to 40

        Returns:
            float: price that otimizes the profit given parameter sample(s)
        """

        # Note

        price_min = self.price_history[-1] * (1 - max_price_change)
        price_max = self.price_history[-1] * (1 + max_price_change)

        # Optimize profit ONLY inside allowed price change region

        if limit_price_change:
            price_min = max(price_min, self.price_min)
            price_max = min(price_max, self.price_max)
        else:
            # global bounds
            price_min = self.price_min
            price_max = self.price_max

        p_grid = np.linspace(price_min, price_max, Npgrid)
        idx = np.random.randint(self.trace.posterior["a"].values.size, size=K)
        a_s = self.trace.posterior["a"].values.reshape(-1)[idx]  # sample
        v_s = self.trace.posterior["v"].values.reshape(-1)[idx]  # sample

        if K == 1:
            a1 = float(a_s[0])
            v1 = float(v_s[0])
            profit_local = np.array(
                [(p - self.c) * a1 * (p ** (-v1)) - self.F for p in p_grid]
            )
        else:
            profit_local = np.array(
                [np.mean((p - self.c) * a_s * p ** (-v_s)) - self.F for p in p_grid]
            )

        price = p_grid[int(np.argmax(profit_local))]

        price = max(price, self.price_min)  # stay within global bounds
        price = min(price, self.price_max)  # stay within global bounds

        return round(price, 2)

    def price_global_popt_expected_profit(self, Npgrid=200) -> float:
        """
        Bayes action: p that maximizes posterior expected profit.
        Uses posterior draws of (a, v). (sigma_log cancels in expectation if you use median demand;
        """
        a_s = self._flatten_posterior_draws("a")
        v_s = self._flatten_posterior_draws("v")

        p_grid = np.linspace(self.price_min, self.price_max, Npgrid)

        # expected profit under posterior
        exp_profit = np.array(
            [np.mean((p - self.c) * a_s * (p ** (-v_s)) - self.F) for p in p_grid]
        )

        return float(p_grid[int(np.argmax(exp_profit))])

    def sampler_health(
        self, idata, rhat_thresh=1.01, ess_thresh=100, print_warning=True
    ):
        """
        Extracts the algorithm convergence health statistics

        Args:
            idata (np.array): the trace, inference data
            rhat_thresh (float, optional): Threshold for comparing model's . Defaults to 1.01.
            ess_thresh (float, optional): Threshold for minimum effective sample size. Defaults to 100.
            print_warning (bool, optional): If True, prints a compact warning message when thresholds are violated.

        Returns:
            dict: Dictionary of convergence diagnostics including divergences, rhat, and ESS metrics.

        Note about Rhat threshold:
        Rhat = 1 only in the infinite-sample limit with perfectly mixed chains.
        With finite draws, Monte Carlo noise almost always gives
        Rhat >1.
        Empirical and theoretical work (Vehtari et al., 2021) shows that
        *      ≤ 1.01 → acceptable for most applied work
        *     Rhat 1.01–1.05 → warning
        *     > 1.05 → unreliable
        That’s why PyMC/ArviZ and the literature flag 1.01, not 1.00, as the practical cutoff.

        Note about ESS:
        Effective Sample Size (ESS) measures the number of independent samples after accounting
        for autocorrelation in the MCMC chains. Low ESS indicates poor mixing or strong posterior
        correlation. Bulk ESS reflects central tendency; tail ESS reflects tail exploration.
        """

        divergences = int(idata.sample_stats["diverging"].values.sum())

        # Only evaluate rhat on sampled parameters
        # var_names = ["a", "v", "sigma_log"]

        sigma_fixed = (
            self.sigma_log_fixed is not None
        )  # True if sigma_log_fixed is not None
        a_fixed = (
            self.m_a_linear_fixed is not None
        )  # true if m_a_linear fixed is not None

        if sigma_fixed and a_fixed:
            var_names = ["v"]
        elif sigma_fixed and not a_fixed:
            var_names = ["a", "v"]
        elif not sigma_fixed and a_fixed:
            var_names = ["sigma_log", "v"]
        else:
            var_names = ["a", "v", "sigma_log"]

        rhat = az.rhat(idata, var_names=var_names).to_array()
        ess_bulk = az.ess(idata, var_names=var_names, method="bulk").to_array()
        ess_tail = az.ess(idata, var_names=var_names, method="tail").to_array()

        result = {
            "divergences": divergences,
            "max_rhat": float(rhat.max()),
            "n_rhat_gt": int((rhat > rhat_thresh).sum()),
            "min_ess_bulk": float(ess_bulk.min()),
            "min_ess_tail": float(ess_tail.min()),
            "n_ess_bulk_lt": int((ess_bulk < ess_thresh).sum()),
            "n_ess_tail_lt": int((ess_tail < ess_thresh).sum()),
        }

        # ---- Compact warning print ----
        if print_warning:
            if (
                result["divergences"] > 0
                or result["max_rhat"] > rhat_thresh
                or result["min_ess_bulk"] < ess_thresh
                or result["min_ess_tail"] < ess_thresh
            ):
                print(
                    f", WARNING div={result['divergences']}, "
                    f"rhat={result['max_rhat']:.3f}, "
                    f"ess_bulk={result['min_ess_bulk']:.1f}, "
                    f"ess_tail={result['min_ess_tail']:.1f}",
                    end="",
                )

        return result

    def model_update(self, price, demand, draws=1000, tune=2000, **kwargs):

        self.price_history.append(price)
        self.demand_history.append(demand)

        # quiet this function down!!!!
        # capture the verbose model output and redirect to log_file ..

        # update data
        with self.model:
            # Posterior Update
            pm.set_data(
                {
                    "p_data": np.asarray(self.price_history, dtype=np.float64),
                    "y_data": np.asarray(self.demand_history, dtype=np.float64),
                }
            )

        # update model
        with self._redirect_io():
            with self.model:
                self.trace = pm.sample(
                    draws=draws,
                    tune=tune,
                    target_accept=0.99,
                    chains=2,
                    init="adapt_diag",
                    random_seed=self.random_seed + len(self.price_history),
                    progressbar=self.verbose,
                    **kwargs,
                )

        return self.trace
