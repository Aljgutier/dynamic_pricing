"""
Demand Estimation Module

- fit_demand_estimator
- vhat_calculator
- get_demand
- select_demand_model
- aggregate_price_demand
- mab_price_optimization_loop
"""

from sklearn.linear_model import (
    HuberRegressor,
    LinearRegression,
    RidgeCV,
)

from scipy.optimize import least_squares


import numpy as np

if __package__:
    # Package-style import (e.g., from py_modules.demand_estimation import ...)
    from .demand_elasticity_profit import (  # pylint: disable=relative-beyond-top-level
        demand_logit,
        demand_constant_elasticity,
        demand_linear,
    )
else:
    # Top-level import (e.g., when py_modules is added to sys.path in notebooks)
    from demand_elasticity_profit import (
        demand_logit,
        demand_constant_elasticity,
        demand_linear,
    )


# ----------------- fit with tilt penalty ------------------#


def _fit_augmented_penalties(
    features,
    target,
    p,
    sample_weights=None,
    tilt_penalty=0.0,
    b2_lambda=0.0,
):
    """
    Fit a regression model using augmented weighted least squares with optional residual-tilt and beta2 penalties.

     The fit minimizes

         weighted RSS
         + tilt_penalty * (
             mean_residual_high
             - mean_residual_low
         )**2
         + b2_lambda * beta2**2

     where the low and high residual means are calculated over the
     lower and upper thirds of the observed price range.

     The tilt penalty therefore favors equal residual error at the
     low- and high-price ends. A common positive or negative residual
     level is not penalized as tilt; the ordinary RSS term continues
     to control the overall level of fit.

     This is solved exactly as an augmented weighted least-squares
     problem; no iterative numerical optimizer is required.

     Parameters
     ----------
     features : ndarray
         Regression design matrix excluding the intercept.

     target : ndarray
         Regression response.

         For linear demand:
             target = d

         For log-quadratic demand:
             target = log(d)

     p : array-like
         Observed prices.

     sample_weights : array-like or None, default=None
         Optional regression sample weights.

     tilt_penalty : float, default=1.0
         Strength of the residual-tilt penalty.


     b2_lambda : float, default=0.0
         Quadratic-coefficient regularization strength.

         Adds

             b2_lambda * beta2**2

         to the weighted least-squares objective.

         A value of 0 disables beta2 regularization.

     Returns
     -------
     model : LinearRegression
         Model object containing the fitted intercept and coefficients.

     tilt_error : float
         Magnitude of the low-to-high residual tilt, calculated as

             abs(
                 mean_residual_high
                 - mean_residual_low
             )
    """

    features = np.asarray(
        features,
        dtype=float,
    )

    target = np.asarray(
        target,
        dtype=float,
    )

    p = np.asarray(
        p,
        dtype=float,
    )

    N = len(target)

    # ---------- Observation weights ----------#

    if sample_weights is None:

        weights = np.ones(
            N,
            dtype=float,
        )

    else:

        weights = np.asarray(
            sample_weights,
            dtype=float,
        )

    # ---------- Add intercept column ----------#

    X = np.column_stack(
        (
            np.ones(N),
            features,
        )
    )

    # ---------- Low / high price regions ----------#

    p_low_cut = np.quantile(
        p,
        1.0 / 3.0,
    )

    p_high_cut = np.quantile(
        p,
        2.0 / 3.0,
    )

    low_mask = p <= p_low_cut
    high_mask = p >= p_high_cut

    # ---------- Construct low/high tilt vectors ----------#
    #
    # residual_low  = a_low  @ residual
    # residual_high = a_high @ residual
    #
    # The tilt penalty minimizes the low-to-high
    # difference in residual means:
    #
    #     abs(residual_high - residual_low)
    #
    # which is equivalent in the least-squares objective
    # to penalizing:
    #
    #     (residual_high - residual_low)**2

    a_low = np.zeros(
        N,
        dtype=float,
    )

    a_high = np.zeros(
        N,
        dtype=float,
    )

    weight_low = np.sum(weights[low_mask])

    weight_high = np.sum(weights[high_mask])

    if weight_low <= 0 or weight_high <= 0:

        raise ValueError("Low and high price regions must have positive weight.")

    a_low[low_mask] = weights[low_mask] / weight_low

    a_high[high_mask] = weights[high_mask] / weight_high

    # ---------- Weighted least squares ----------#

    sqrt_weights = np.sqrt(weights)

    X_weighted = X * sqrt_weights[:, None]

    y_weighted = target * sqrt_weights

    # ---------- Low-to-high tilt penalty ----------#

    total_weight = np.sum(weights)

    sqrt_tilt_penalty = np.sqrt(tilt_penalty * total_weight)

    # Enforce equality between the weighted residual means
    # at the high- and low-price ends:
    #
    #     residual_high - residual_low -> 0
    #
    # where
    #
    #     residual = target - X @ theta

    a_tilt = a_high - a_low

    X_tilt = sqrt_tilt_penalty * (a_tilt @ X)

    y_tilt = sqrt_tilt_penalty * (a_tilt @ target)

    X_augmented = np.vstack(
        (
            X_weighted,
            X_tilt,
        )
    )

    y_augmented = np.concatenate(
        (
            y_weighted,
            [y_tilt],
        )
    )

    # ---------- Beta2-only regularization ----------#
    #
    # Adds
    #
    #     b2_lambda * beta2**2
    #
    # to the least-squares objective.
    #
    # X includes the intercept:
    #
    #     theta = [beta0, beta1, beta2, ...]

    if b2_lambda > 0:

        if X.shape[1] < 3:

            raise ValueError("b2_lambda requires a quadratic basis.")

        X_b2 = np.zeros(
            X.shape[1],
            dtype=float,
        )

        X_b2[2] = np.sqrt(b2_lambda)

        X_augmented = np.vstack(
            (
                X_augmented,
                X_b2,
            )
        )

        y_augmented = np.concatenate(
            (
                y_augmented,
                [0.0],
            )
        )

    # ---------- Exact least-squares solution ----------#

    theta, _, _, _ = np.linalg.lstsq(
        X_augmented,
        y_augmented,
        rcond=None,
    )

    # ---------- Store solution in sklearn model ----------#

    model = LinearRegression()

    model.intercept_ = float(theta[0])

    model.coef_ = np.asarray(
        theta[1:],
        dtype=float,
    )

    model.n_features_in_ = features.shape[1]

    # ---------- Final residual tilt diagnostic ----------#

    target_hat = model.predict(features)

    residual = target - target_hat

    tilt_error_low = float(a_low @ residual)

    tilt_error_high = float(a_high @ residual)

    tilt_error = float(abs(tilt_error_high - tilt_error_low))

    return model, tilt_error


# ---------- LogQL linear-penalty fit ------------------#


def _fit_log_quadratic_level_lp(
    features,
    demand,
    sample_weights=None,
    b2_lambda=0.0,
):
    """
    Fit the LogQL demand model

        d_hat = exp(
            beta0
            + beta1 * p
            + beta2 * p**2
            + gamma' * x
        )

    using demand-level residuals.

    The fit minimizes

        sum_i w_i * (d_i - d_hat_i)**2
        + b2_lambda * beta2**2.

    beta0 and beta1 are not regularized.
    """

    features = np.asarray(
        features,
        dtype=float,
    )

    demand = np.asarray(
        demand,
        dtype=float,
    )

    N = len(demand)

    if sample_weights is None:

        weights = np.ones(
            N,
            dtype=float,
        )

    else:

        weights = np.asarray(
            sample_weights,
            dtype=float,
        )

    sqrt_weights = np.sqrt(weights)

    # ---------- Add intercept column ----------#

    X = np.column_stack(
        (
            np.ones(N),
            features,
        )
    )

    # ---------- Initial estimate ----------#
    #
    # Start from ordinary LogQL:
    #
    #     log(d) = X @ theta

    theta0, _, _, _ = np.linalg.lstsq(
        X,
        np.log(demand),
        rcond=None,
    )

    # ---------- Residual function ----------#

    def residual_function(theta):

        log_d_hat = X @ theta

        # Numerical safety during nonlinear optimization.
        log_d_hat = np.clip(
            log_d_hat,
            -50.0,
            50.0,
        )

        d_hat = np.exp(log_d_hat)

        residual_data = sqrt_weights * (demand - d_hat)

        # Beta2-only regularization.
        if b2_lambda > 0:

            residual_b2 = np.array(
                [np.sqrt(b2_lambda) * theta[2]],
                dtype=float,
            )

            return np.concatenate(
                (
                    residual_data,
                    residual_b2,
                )
            )

        return residual_data

    # ---------- Nonlinear least-squares fit ----------#

    result = least_squares(
        residual_function,
        theta0,
    )

    if not result.success:

        raise ValueError("log_quadratic_level_lp fit failed: " f"{result.message}")

    theta = result.x

    # ---------- Store in sklearn-like model ----------#
    #
    # This preserves the existing downstream interface:
    #
    #     model.predict(features)
    #     model.intercept_
    #     model.coef_

    model = LinearRegression()

    model.intercept_ = float(theta[0])

    model.coef_ = np.asarray(
        theta[1:],
        dtype=float,
    )

    model.n_features_in_ = features.shape[1]

    return model


# ---------------  demand estimator -----------------#
def fit_demand_estimator(
    p,
    d,
    x=None,
    model_type="log_quadratic",
    sample_weights=None,
    tilt_penalty_linear=0.0,
    tilt_penalty_log_level=0.0,
    tilt_penalty_log_log=0.0,
    tilt_penalty_log_quadratic=0.0,
    tilt_penalty_log_quadratic_level=0.0,
    b2_lambda_q=0.0,
    b2_lambda_lql=0.0,
    b2_lambda_lql_lp=0.0,
    ridge_cv_linear=False,
    ridge_cv_q=False,
    ridge_cv_lql=False,
    huber=False,
    huber_epsilon=1.35,
):
    """
    Fits several demand models. Supported models include linear, log-level, log-log, and log-quadratic.

    The function returns a callable demand prediction function and a dictionary containing the fitted regression model and supporting information.

    Process for selecting the best model is handled by the select_demand_model() function, which evaluates multiple candidate models and selects the one with the smallest combined selection score as follows:

    * See the select_model funtion for selectng the bset model for providing a good elasticity estimate, which in turn can be used to estimate profit.
    * Or, the best model can simply be selected based on the lowest in-sample fit error (e.g., log-RMSE) without regard to elasticity.


    Supported demand models
    -----------------------

    linear:
        d = beta_0 + beta_1 * p + gamma' x

    log_level:
        log(d) = beta_0 + beta_1 * p + gamma' x

    log_log:
        log(d) = beta_0 + beta_1 * log(p) + gamma' x

        This model is mathematically equivalent to a constant-elasticity
        demand model

            d = A * p^(-v),

        where
            A = exp(beta_0)
            v = -beta_1.

        Consequently, the implied price elasticity is constant and equal
        to -beta_1.

    quadratic:
        d = beta_0
        + beta_1 * p
        + beta_2 * p^2
        + gamma' x

        This generalizes the linear demand model by allowing
        curvature directly in demand levels.

        The implied price elasticity is

            v(p) = -p * (beta_1 + 2 * beta_2 * p) / d(p).

        As beta_2 approaches zero, the quadratic model approaches
        the linear demand model.

        Regularization strength applied only to the beta2 coefficient
        of the "quadratic" demand model.

    log_quadratic:
        log(d) = beta_0
            + beta_1 * log(p)
            + beta_2 * log(p)^2
            + gamma' x

        This generalizes the constant-elasticity model by allowing the
        elasticity to vary smoothly with price:

            v(p) = -(beta_1 + 2 * beta_2 * log(p)).

    log_quadratic_level:
        log(d) = beta_0
            + beta_1 * p
            + beta_2 * p^2
            + gamma' x

        This allows log-demand curvature directly in price levels.

        The implied price elasticity is

            v(p) = -p * (beta_1 + 2 * beta_2 * p).

    log_quadratic_level_lp:
        d = exp(
            beta_0
            + beta_1 * p
            + beta_2 * p^2
            + gamma' x
        )

        This has the same functional form as
        "log_quadratic_level", but model fitting is
        performed using demand-level residuals.

        With beta2 regularization, the fit minimizes

            weighted demand-level RSS
            + b2_lambda_lql_lp * beta2**2.

        The implied price elasticity is

            v(p) = -p * (beta_1 + 2 * beta_2 * p).


    Parameters
    ----------
    p : array-like
        Historical prices.

    d : array-like
        Historical demand observations.

    x : array-like or None, default=None
        Optional contextual variables. Each row corresponds to one
        observation.

    model_type : {"linear","quadratic" , "log_level", "log_log", "log_quadratic", "log_quadratic_level", "log_quadratic_level_lp"},
        default="log_quadratic"

    sample_weights : array-like or None, default=None
        Optional regression sample weights.

    alpha : float, default=0.0
        Ridge regularization parameter.
        alpha = 0 uses ordinary least squares.

    b2_lambda_q : float, default=0.0
        If b2_lambda_q > 0, beta_2 is regularized toward zero.

        As beta_2 approaches zero, the quadratic model approaches
        the linear demand model.

        Regularization strength applied only to the beta2 coefficient
        of the "quadratic" demand model.

        The quadratic fit minimizes

            weighted RSS + b2_lambda * beta2**2.

        beta0 and beta1 are not regularized.

        A value of 0 disables beta2 regularization.

    b2_lambda_lql : float, default=0.0
        Beta2-only regularization strength for the
        "log_quadratic_level" demand model.

        The log-quadratic-level fit minimizes

            weighted log-demand RSS
            + b2_lambda_lql * beta2**2.

        beta0 and beta1 are not regularized.

        A value of 0 disables beta2 regularization
        for the log-quadratic-level model.

    b2_lambda_lql_lp : float, default=0.0
        Beta2-only regularization strength for the
        "log_quadratic_level_lp" demand model.

        The fit minimizes

            weighted demand-level RSS
            + b2_lambda_lql_lp * beta2**2.

        beta0 and beta1 are not regularized.

        A value of 0 disables beta2 regularization
        for the log-quadratic-level LP model.

    Returns
    -------
    dhat_model : callable
        Demand prediction function
            dhat_model(p_new, x_new=None)

    model_info : dict
        Dictionary containing the fitted regression model and
        supporting information.
    """

    p = np.asarray(p, dtype=float)
    d = np.asarray(d, dtype=float)

    # ----------------- validate inputs --------------
    if (ridge_cv_linear or ridge_cv_q or ridge_cv_lql) and huber:

        raise ValueError("RidgeCV and huber cannot currently " "be used together.")

    # Quadratic cannot use its beta2 penalty and RidgeCV
    # at the same time.
    if b2_lambda_q > 0 and ridge_cv_q:

        raise ValueError(
            "b2_lambda_q and ridge_cv_q cannot " "currently be used together."
        )

    # LogQL cannot use its beta2 penalty and RidgeCV
    # at the same time.
    if b2_lambda_lql > 0 and ridge_cv_lql:

        raise ValueError(
            "b2_lambda_lql and ridge_cv_lql cannot " "currently be used together."
        )

    if (b2_lambda_q > 0 or b2_lambda_lql > 0 or b2_lambda_lql_lp > 0) and huber:
        raise ValueError(
            "beta2 regularization and huber " "cannot currently be used together."
        )

    # beta2 lamnda validation
    if b2_lambda_q < 0:
        raise ValueError("b2_lambda_q must be non-negative.")

    if b2_lambda_lql < 0:
        raise ValueError("b2_lambda_lql must be non-negative.")

    if b2_lambda_lql_lp < 0:

        raise ValueError("b2_lambda_lql_lp must be non-negative.")

    if len(p) != len(d):
        raise ValueError("p and d must have the same length.")

    if np.any(p <= 0):
        raise ValueError("Prices must be positive.")

    if model_type in (
        "log_level",
        "log_log",
        "log_quadratic",
        "log_quadratic_level",
        "log_quadratic_level_lp",
    ):
        if np.any(d <= 0):
            raise ValueError("Demand must be positive for logarithmic models.")

    # minimum observations
    if model_type in (
        "quadratic",
        "log_quadratic",
        "log_quadratic_level",
        "log_quadratic_level_lp",
    ):
        if len(p) < 3:
            raise ValueError(
                f"At least three observations are required for model_type='{model_type}'."
            )

    elif model_type in ("linear", "log_level", "log_log"):
        if len(p) < 2:
            raise ValueError(
                f"At least two observations are required "
                f"for model_type='{model_type}'."
            )
    else:
        raise ValueError(f"Unsupported model_type: {model_type}.")

    # --------Helper: construct regression response ----------------------

    def _response(d_values):
        """
        Construct the regression response variable.
        """

        if model_type in ("linear", "quadratic"):
            return d_values

        return np.log(d_values)

    def _price_basis(p_values):
        """
        Construct the regression basis functions associated with
        price.
        """

        p_values = np.asarray(p_values, dtype=float)  # price values

        if np.any(p_values <= 0):
            raise ValueError("Prices must be positive.")

        if model_type in ("linear", "log_level"):

            return p_values.reshape(-1, 1)

        elif model_type == "quadratic":

            return np.column_stack(
                (
                    p_values,
                    p_values**2,
                )
            )

        elif model_type == "log_log":

            log_p = np.log(p_values).reshape(-1, 1)

            return log_p

        elif model_type == "log_quadratic":

            log_p = np.log(p_values)

            return np.column_stack(
                (
                    log_p,
                    log_p**2,
                )
            )

        elif model_type in (
            "log_quadratic_level",
            "log_quadratic_level_lp",
        ):

            return np.column_stack(
                (
                    p_values,
                    p_values**2,
                )
            )

        raise ValueError(f"Unsupported model_type: {model_type}.")

    # ----- Helper: transform regression output back to demand  ------------ %

    def _inverse_response(y):
        """
        Transform regression predictions back to demand.
        """

        if model_type in ("linear", "quadratic"):
            # demand cannot be negative
            return np.maximum(y, 0.0)

        return np.exp(y)

    # --------Construct regression response and price basis  ------------#

    target = _response(d)
    price_basis = _price_basis(p)

    # ------- Optional contextual variables -----------------#

    n_x_features = 0

    if x is not None:
        x = np.asarray(x, dtype=float)

        if x.ndim == 1:
            x = x.reshape(-1, 1)

        if len(x) != len(p):
            raise ValueError("x must have the same number of rows as p.")

        n_x_features = x.shape[1]

        features = np.column_stack(
            (
                price_basis,
                x,
            )
        )

    else:

        features = price_basis

    # ---------- Select RidgeCV by model ----------#

    if model_type == "linear":

        ridge_cv_model = ridge_cv_linear

    elif model_type == "quadratic":

        ridge_cv_model = ridge_cv_q

    elif model_type == "log_quadratic_level":

        ridge_cv_model = ridge_cv_lql

    else:

        ridge_cv_model = False

    # ---------- Select regression estimator ----------#

    if huber:

        model = HuberRegressor(
            epsilon=huber_epsilon,
            alpha=0.0,
            max_iter=1000,
        )

    elif ridge_cv_model:

        model = RidgeCV(alphas=np.logspace(-4, 2, 25))

    else:

        model = LinearRegression()

    # ---------- Select and fit regression estimator ----------#

    if model_type == "linear":
        tilt_penalty = tilt_penalty_linear

    elif model_type == "log_level":
        tilt_penalty = tilt_penalty_log_level

    elif model_type == "log_log":
        tilt_penalty = tilt_penalty_log_log

    elif model_type == "log_quadratic":
        tilt_penalty = tilt_penalty_log_quadratic

    elif model_type in (
        "log_quadratic_level",
        "log_quadratic_level_lp",
    ):
        tilt_penalty = tilt_penalty_log_quadratic_level

    else:
        tilt_penalty = 0.0

    # ---------- Select beta2 regularization ----------#

    if model_type == "quadratic":

        b2_lambda_model = b2_lambda_q

    elif model_type == "log_quadratic_level":

        b2_lambda_model = b2_lambda_lql

    elif model_type == "log_quadratic_level_lp":
        b2_lambda_model = b2_lambda_lql_lp

    else:
        b2_lambda_model = 0.0

    tilt_error = np.nan

    if model_type == "log_quadratic_level_lp":

        if huber:

            raise ValueError(
                "log_quadratic_level_lp does not " "currently support huber."
            )

        if tilt_penalty > 0:

            raise ValueError(
                "log_quadratic_level_lp does not " "currently support tilt_penalty."
            )

        model = _fit_log_quadratic_level_lp(
            features=features,
            demand=d,
            sample_weights=sample_weights,
            b2_lambda=b2_lambda_model,
        )

    elif (
        (tilt_penalty > 0 or b2_lambda_model > 0)
        and model_type
        in (
            "linear",
            "quadratic",
            "log_level",
            "log_log",
            "log_quadratic",
            "log_quadratic_level",
        )
        and not ridge_cv_model
        and not huber
    ):

        model, tilt_error = _fit_augmented_penalties(
            features=features,
            target=target,
            p=p,
            sample_weights=sample_weights,
            tilt_penalty=tilt_penalty,
            b2_lambda=b2_lambda_model,
        )

    else:

        if huber:
            model = HuberRegressor(
                epsilon=huber_epsilon,
                alpha=0.0,
                max_iter=1000,
            )

        elif ridge_cv_model:
            model = RidgeCV(alphas=np.logspace(-4, 2, 25))

        else:

            model = LinearRegression()

        model.fit(
            features,
            target,
            sample_weight=sample_weights,
        )

    # ---------- Final residual tilt diagnostics ----------#

    tilt_error = np.nan
    tilt_error_low = np.nan
    tilt_error_high = np.nan
    tilt_imbalance = np.nan

    target_hat = model.predict(features)
    residual = target - target_hat

    p_low_cut = np.quantile(
        p,
        1.0 / 3.0,
    )

    p_high_cut = np.quantile(
        p,
        2.0 / 3.0,
    )

    low_mask = p <= p_low_cut
    high_mask = p >= p_high_cut

    if sample_weights is None:

        weights = np.ones(
            len(p),
            dtype=float,
        )

    else:

        weights = np.asarray(
            sample_weights,
            dtype=float,
        )

    # weighted mean residual on low-price side
    tilt_error_low = float(
        np.average(
            residual[low_mask],
            weights=weights[low_mask],
        )
    )

    # weighted mean residual on high-price side
    tilt_error_high = float(
        np.average(
            residual[high_mask],
            weights=weights[high_mask],
        )
    )

    # magnitude imbalance between low- and high-price residual tilt
    tilt_imbalance = float(abs(abs(tilt_error_high) - abs(tilt_error_low)))

    # low-to-high residual slope tilt
    tilt_error = float(abs(tilt_error_high - tilt_error_low))

    # ------  Demand prediction function -------------------

    def dhat_model(p_new, x_new=None):

        scalar_input = np.isscalar(p_new)
        p_new = np.atleast_1d(np.asarray(p_new, dtype=float))

        price_basis_new = _price_basis(p_new)

        # ---------- Optional contextual feature variables -----------

        if n_x_features > 0:

            if x_new is None:
                raise ValueError(
                    "x_new is required because the model "
                    "was fitted with contextual variables."
                )

            x_new = np.asarray(x_new, dtype=float)

            if x_new.ndim == 1:

                if len(p_new) == 1:
                    x_new = x_new.reshape(1, -1)
                else:
                    x_new = x_new.reshape(-1, 1)

            if len(x_new) != len(p_new):
                raise ValueError("x_new must have the same number of rows as p_new.")

            features_new = np.column_stack(
                (
                    price_basis_new,
                    x_new,
                )
            )

        else:

            features_new = price_basis_new

        # ------------ Predict demand -----------------#

        prediction = model.predict(features_new)

        d_hat = _inverse_response(prediction)

        if scalar_input:
            return float(d_hat[0])

        return d_hat

    # -----Store model information --------------------#

    model_info = {
        "dhat_model": dhat_model,  # fitted demand prediction function
        "model": model,  # fitted regression model
        "model_type": model_type,  # model specification
        "intercept": model.intercept_,  # fitted regression coefficients
        "coef": model.coef_,
        "p": p,  # training data
        "d": d,
        "n_x_features": n_x_features,  # model configuration
        "ridge_cv": ridge_cv_model,  # RidgeCV flag
        "huber": huber,  # huber
        "b2_lambda_model": float(b2_lambda_model),
        "huber_epsilon": huber_epsilon,
        "tilt_penalty": tilt_penalty,  # log-quadratic tilt
        "tilt_error": tilt_error,
        "tilt_error_low": tilt_error_low,
        "tilt_error_high": tilt_error_high,
        "tilt_imbalance": tilt_imbalance,
    }

    return dhat_model, model_info


# --------------- vhat calculator ------------------#
def vhat_calculator(model_info, p):
    """
    Compute the price elasticity implied by a fitted demand model.

    Supported demand models
    -----------------------

    linear:
        d = beta_0 + beta_1 * p

    log_level:
        log(d) = beta_0 + beta_1 * p

    log_log:
        log(d) = beta_0 + beta_1 * log(p)

    log_quadratic:
        log(d) = beta_0 + beta_1 * log(p)
                        + beta_2 * log(p)^2


    log_quadratic_level:
        log(d) = beta_0 + beta_1 * p
                        + beta_2 * p^2

    Parameters
    ----------
    model_info : dict
        Dictionary returned by fit_demand_estimator() or
        piecewise_demand_fit().

    p : float or array-like
        Price(s) at which elasticity is to be evaluated.

    Returns
    -------
    float or ndarray
        Estimated elasticity evaluated at p.
    """

    model_type = model_info["model_type"]

    scalar_input = np.isscalar(p)

    p = np.atleast_1d(
        np.asarray(
            p,
            dtype=float,
        )
    )

    if np.any(p <= 0):
        raise ValueError("Prices must be positive.")

    # ---------- Linear  ----------#

    def _vhat_linear():

        beta1 = model_info["coef"][0]

        d_hat = model_info["dhat_model"](p)

        d_hat = np.maximum(
            d_hat,
            1e-12,
        )

        return -(p / d_hat) * beta1

    # ---------- Quadratic ----------#

    def _vhat_quadratic():

        beta1 = model_info["coef"][0]
        beta2 = model_info["coef"][1]

        d_hat = model_info["dhat_model"](p)

        d_hat = np.maximum(
            d_hat,
            1e-12,
        )

        d_1prime = beta1 + 2.0 * beta2 * p

        return -(p / d_hat) * d_1prime

    # ---------- Log-level----------#

    def _vhat_log_level():

        beta1 = model_info["coef"][0]

        return -beta1 * p

    # ---------- Log-log ----------#

    def _vhat_log_log():

        beta1 = model_info["coef"][0]

        return np.full_like(
            p,
            -beta1,
            dtype=float,
        )

    # ---------- Log-quadratic ----------#

    def _vhat_log_quadratic():

        beta1 = model_info["coef"][0]
        beta2 = model_info["coef"][1]

        return -(beta1 + 2.0 * beta2 * np.log(p))

    # ---------- Log-quadratic-level ----------#

    def _vhat_log_quadratic_level():

        beta1 = model_info["coef"][0]
        beta2 = model_info["coef"][1]

        return -p * (beta1 + 2.0 * beta2 * p)

    # ---------- Model dispatch ----------#

    vhat_functions = {
        "linear": _vhat_linear,
        "quadratic": _vhat_quadratic,
        "log_level": _vhat_log_level,
        "log_log": _vhat_log_log,
        "log_quadratic": _vhat_log_quadratic,
        "log_quadratic_level": _vhat_log_quadratic_level,
        "log_quadratic_level_lp": _vhat_log_quadratic_level,
    }

    if model_type not in vhat_functions:
        raise ValueError(f"Unsupported model_type: {model_type}")

    vhat = vhat_functions[model_type]()

    if scalar_input:
        return float(vhat[0])

    return vhat


# ---------------- get_demand ------------------#


def get_demand(demand_type, demand_params, p, sigma_log=0):

    scalar_input = np.isscalar(p)

    if demand_type == "constant_elasticity":
        d = demand_constant_elasticity(
            p,
            demand_params["A"],
            demand_params["v"],
            sigma_log=sigma_log,
        )

    elif demand_type == "linear":
        d = demand_linear(
            p,
            demand_params["A"],
            demand_params["b"],
            sigma_log=sigma_log,
        )

    elif demand_type == "logit":
        d = demand_logit(
            p,
            demand_params["A"],
            demand_params["a"],
            demand_params["b"],
            sigma_log=sigma_log,
        )

    else:
        raise ValueError(f"Invalid demand type: {demand_type}")

    if scalar_input:
        return float(d[0])

    return d


# --------------------- select_demand_model ------------------#
def select_demand_model(
    p,
    d,
    candidate_model_types,
    price_min,
    price_max,
    c,
    F=0.0,
    sample_weights=None,
    ridge_cv_linear=False,
    ridge_cv_q=False,
    ridge_cv_lql=False,
    huber=False,
    huber_epsilon=1.35,
    b2_lambda_q=0.0,
    b2_lambda_lql=0.0,
    b2_lambda_lql_lp=0.0,
    minimum_elasticity=1.0,
    selection_criterion="rss",
    primary_model_type=None,
    boundary_buffer=0.1,
    Ngrid=500,
    tilt_penalty_linear=0.0,
    tilt_penalty_log_level=0.0,
    tilt_penalty_log_log=0.0,
    tilt_penalty_log_quadratic=0.0,
    tilt_penalty_log_quadratic_level=0.0,
    p_opt_linear_delta_frac_max=0.20,
    logq_aic_improvement_min=2.0,
    linear_tilt_error_max=np.inf,
    linear_tilt_imbalance_max=np.inf,
    thr_rss_diff_max=np.inf,
    thr_rss_max=np.inf,
):
    """
    Select the best demand model from a set of candidate demand
    estimators.

    Each candidate model is fitted, the implied profit-maximizing
    price is computed, and the model is scored using the selected
    information criterion (AIC or BIC). Models whose estimated
    optimal price lies within a specified boundary buffer of the
    price limits are rejected whenever possible. The remaining
    model with the lowest information criterion is selected.

    Supported candidate models
    --------------------------

    linear:
        d = beta_0 + beta_1 * p

    quadratic:
        d = beta_0 + beta_1 * p + beta_2 * p^2

    log_level:
        log(d) = beta_0 + beta_1 * p

    log_log:
        log(d) = beta_0 + beta_1 * log(p)

    log_quadratic:
        log(d) = beta_0 + beta_1 * log(p)
                        + beta_2 * log(p)^2

    log_quadratic_level:
        log(d) = beta_0 + beta_1 * p
                        + beta_2 * p^2

    Parameters
    ----------
    p : array-like
        Historical prices.

    d : array-like
        Historical demand observations.

    price_min : float
        Minimum allowable price.

    price_max : float
        Maximum allowable price.

    c : float
        Variable unit cost.

    F : float
        Fixed cost.

    candidate_model_types : list of str
        Demand-model types to fit and consider during model selection.

        The "linear" model is required because it serves as the
        reference and safety fallback model.

        At least one additional supported model must also be supplied.

        Supported model types are:
            - "linear"
            - "quadratic"
            - "log_level"
            - "log_log"
            - "log_quadratic"
            - "log_quadratic_level"

        Examples:

            ["linear", "log_quadratic"]

            ["linear", "log_log"]

            ["linear", "log_log", "log_quadratic"]

            ["linear", "log_quadratic_level"]

        If both "log_log" and "log_quadratic" are present, the
        log-quadratic model is required to provide sufficient AIC
        improvement over log-log according to
        logq_aic_improvement_min.

        If log-quadratic is present without log-log, this additional
        complexity comparison is not applied.

    selection_criteria : {"AIC", "BIC"}, default="AIC"
        Information criterion used for model selection.

    boundary_buffer : float, default=0.0
        Distance from the price boundaries within which a model
        is considered invalid if another valid candidate exists.

    sample_weights : array-like or None, default=None
        Optional regression sample weights.

    ridge_cv_linear : bool, default=False
        If True, fit the "linear" model using RidgeCV.

    ridge_cv_q : bool, default=False
        If True, fit the "quadratic" model using RidgeCV.

        Cannot currently be combined with
        b2_lambda_q > 0.

    ridge_cv_lql : bool, default=False
        If True, fit the "log_quadratic_level"
        model using RidgeCV.

        Cannot currently be combined with
        b2_lambda_lql > 0.

    b2_lambda_q : float, default=0.0
        Beta2-only regularization strength for the
        "quadratic" demand model.

        A value of 0 disables beta2 regularization
        for the quadratic model.

    b2_lambda_lql : float, default=0.0
        Beta2-only regularization strength for the
        "log_quadratic_level" demand model.

        A value of 0 disables beta2 regularization
        for the log-quadratic-level model.

    Neither beta2 regularization option can currently
    be combined with ridge_cv or huber.

    huber : bool, default=False
        If True, use Huber regression instead of ordinary least
        squares for supported regression models.

    primary_model_type : str or None, optional
        Optional candidate model to use as the initial model-selection
        choice instead of selecting the initial winner by the configured
        fit criterion.

        If None, the initial model is selected normally using
        selection_criterion.

        If specified, primary_model_type must be included in
        candidate_model_types and must be a valid fitted model.

        The primary-model override affects only the initial model choice.
        The existing divergence-from-linear and boundary safeguards are
        still applied, so the selected model may still fall back to
        linear.

        Defaults to None.


    linear_tilt_error_max : float, optional
        Maximum allowable low-to-high residual tilt for linear
        to be accepted as the initial model-selection winner.

        Residual tilt is defined as

            abs(
                tilt_error_high
                - tilt_error_low
            )

        This measures the difference between the weighted mean
        residuals at the high- and low-price ends.

        Defaults to np.inf, which disables this guard.

    linear_tilt_imbalance_max : float, optional
        Maximum allowable imbalance between the magnitudes of the
        low- and high-price residual tilt for linear to be accepted
        as the initial model-selection winner. Defined as

            abs(
                abs(tilt_error_high)
                - abs(tilt_error_low)
            )

        Defaults to np.inf, which disables this guard.

    Returns
    -------
    dhat_model : callable
        Selected demand prediction function.

    model_info : dict
        Dictionary describing the selected fitted demand model.
    """

    # ---------- Validate  ----------#

    if thr_rss_diff_max < 0:
        raise ValueError("thr_rss_diff_max must be non-negative.")

    if thr_rss_max < 0:
        raise ValueError("thr_rss_max must be non-negative.")

    if (ridge_cv_linear or ridge_cv_q or ridge_cv_lql) and huber:

        raise ValueError("RidgeCV and huber cannot currently " "be used together.")

    if b2_lambda_q > 0 and ridge_cv_q:

        raise ValueError(
            "b2_lambda_q and ridge_cv_q cannot " "currently be used together."
        )

    if b2_lambda_lql > 0 and ridge_cv_lql:

        raise ValueError(
            "b2_lambda_lql and ridge_cv_lql cannot " "currently be used together."
        )

    if (b2_lambda_q > 0 or b2_lambda_lql > 0 or b2_lambda_lql_lp > 0) and huber:
        raise ValueError(
            "beta2 regularization and huber " "cannot currently be used together."
        )

    if b2_lambda_q < 0:
        raise ValueError("b2_lambda_q must be non-negative.")

    if b2_lambda_lql < 0:
        raise ValueError("b2_lambda_lql must be non-negative.")

    if b2_lambda_lql_lp < 0:

        raise ValueError("b2_lambda_lql_lp must be " "non-negative.")

    valid_selection_criteria = {
        "rss",
        "log_rmse",
        "aic",
        "aicc",
        "bic",
    }

    if selection_criterion not in valid_selection_criteria:
        raise ValueError(
            "selection_criterion must be one of "
            "{'rss', 'log_rmse', 'aic', 'aicc', 'bic'}."
        )

    p = np.asarray(p, dtype=float)
    d = np.asarray(d, dtype=float)

    p_grid = np.linspace(
        price_min,
        price_max,
        Ngrid,
    )

    # ---------- Validate candidate models ----------#

    supported_model_types = {
        "linear",
        "quadratic",
        "log_level",
        "log_log",
        "log_quadratic",
        "log_quadratic_level",
        "log_quadratic_level_lp",
    }

    candidate_model_types = list(candidate_model_types)

    unsupported_model_types = set(candidate_model_types) - supported_model_types

    if unsupported_model_types:
        raise ValueError(
            "Unsupported candidate model type(s): " f"{sorted(unsupported_model_types)}"
        )

    # Linear is required as the diagnostic reference model.
    if "linear" not in candidate_model_types:
        raise ValueError("candidate_model_types must include 'linear'.")

    # Quadratic is required as the conservative default model.
    if "quadratic" not in candidate_model_types:
        raise ValueError("candidate_model_types must include 'quadratic'.")

    # ---------- Validate primary model ----------#

    if (
        primary_model_type is not None
        and primary_model_type not in candidate_model_types
    ):
        raise ValueError(
            "primary_model_type must be None or one of "
            "candidate_model_types. "
            f"Got primary_model_type={primary_model_type!r}."
        )

    # ---------- Ensure prices are above unit cost ----------#

    p_grid = p_grid[p_grid > c]

    if len(p_grid) == 0:
        raise ValueError("price_max must be greater than unit cost c.")

    price_range = price_max - price_min
    boundary_distance = boundary_buffer * price_range

    fitted_models = {}
    model_diagnostics = {}

    # initialize some model diagnostics
    beta1 = np.nan
    beta2 = np.nan
    log_quad_vhat_range = np.nan
    log_quad_vhat_mid = np.nan
    log_log_vhat = np.nan
    log_models_vhat_delta = np.nan

    # --------- Fit candidate models and compute diagnostics ---------#

    for model_type in candidate_model_types:
        try:
            # ---------- Fit candidate model ----------#
            dhat_model_i, model_info_i = fit_demand_estimator(
                p=p,
                d=d,
                x=None,
                ridge_cv_linear=ridge_cv_linear,
                ridge_cv_q=ridge_cv_q,
                ridge_cv_lql=ridge_cv_lql,
                huber=huber,
                huber_epsilon=huber_epsilon,
                model_type=model_type,
                sample_weights=sample_weights,
                tilt_penalty_linear=tilt_penalty_linear,
                tilt_penalty_log_level=tilt_penalty_log_level,
                tilt_penalty_log_log=tilt_penalty_log_log,
                tilt_penalty_log_quadratic=tilt_penalty_log_quadratic,
                tilt_penalty_log_quadratic_level=tilt_penalty_log_quadratic_level,
                b2_lambda_q=b2_lambda_q,
                b2_lambda_lql=b2_lambda_lql,
                b2_lambda_lql_lp=b2_lambda_lql_lp,
            )

            # ---------- Log-demand residual diagnostics ----------#

            dhat = np.asarray(
                dhat_model_i(p),
                dtype=float,
            )

            predictions_valid = (
                np.all(np.isfinite(dhat))
                and np.all(dhat > 0)
                and np.all(np.isfinite(d))
                and np.all(d > 0)
            )

            # ---------- Number of fitted parameters ----------#

            n_parameters = len(model_info_i["coef"]) + 1

            if not predictions_valid:

                fit_error = np.inf

                rss = np.inf
                rss_log_low = np.inf
                rss_log_high = np.inf

                rss_level = np.inf
                rss_level_low = np.inf
                rss_level_high = np.inf

                aic = np.inf
                aicc = np.inf
                bic = np.inf

            else:

                # ---------- Residuals on both scales ----------#

                residual_log = np.log(d) - np.log(dhat)

                residual_level = d - dhat

                # ---------- Low / high price regions ----------#

                p_low_cut = np.quantile(
                    p,
                    1.0 / 3.0,
                )

                p_high_cut = np.quantile(
                    p,
                    2.0 / 3.0,
                )

                low_mask = p <= p_low_cut
                high_mask = p >= p_high_cut

                if sample_weights is None:

                    # ---------- Log-demand scale ----------#

                    fit_error = np.sqrt(np.mean(residual_log**2))

                    rss = np.sum(residual_log**2)

                    rss_log_low = np.sum(residual_log[low_mask] ** 2)

                    rss_log_high = np.sum(residual_log[high_mask] ** 2)

                    # ---------- Demand-level scale ----------#

                    rss_level = np.sum(residual_level**2)

                    rss_level_low = np.sum(residual_level[low_mask] ** 2)

                    rss_level_high = np.sum(residual_level[high_mask] ** 2)

                else:

                    weights = np.asarray(
                        sample_weights,
                        dtype=float,
                    )

                    # ---------- Log-demand scale ----------#

                    fit_error = np.sqrt(
                        np.average(
                            residual_log**2,
                            weights=weights,
                        )
                    )

                    rss = np.sum(weights * residual_log**2)

                    rss_log_low = np.sum(
                        weights[low_mask] * residual_log[low_mask] ** 2
                    )

                    rss_log_high = np.sum(
                        weights[high_mask] * residual_log[high_mask] ** 2
                    )

                    # ---------- Demand-level scale ----------#

                    rss_level = np.sum(weights * residual_level**2)

                    rss_level_low = np.sum(
                        weights[low_mask] * residual_level[low_mask] ** 2
                    )

                    rss_level_high = np.sum(
                        weights[high_mask] * residual_level[high_mask] ** 2
                    )

                # ---------- Numerical safeguards ----------#

                rss = max(
                    float(rss),
                    1e-12,
                )

                rss_log_low = max(
                    float(rss_log_low),
                    1e-12,
                )

                rss_log_high = max(
                    float(rss_log_high),
                    1e-12,
                )

                rss_level = max(
                    float(rss_level),
                    1e-12,
                )

                rss_level_low = max(
                    float(rss_level_low),
                    1e-12,
                )

                rss_level_high = max(
                    float(rss_level_high),
                    1e-12,
                )

                # ---------- Information criteria ----------#
                #
                # Keep AIC/AICc/BIC based on the existing
                # common log-demand RSS.

                n_observations = len(residual_log)

                aic = n_observations * np.log(rss / n_observations) + 2.0 * n_parameters

                if n_observations > n_parameters + 1:

                    aicc = aic + (2.0 * n_parameters * (n_parameters + 1)) / (
                        n_observations - n_parameters - 1
                    )

                else:

                    aicc = np.inf

                bic = n_observations * np.log(
                    rss / n_observations
                ) + n_parameters * np.log(n_observations)

            # ---------- model-selection score ----------#

            if selection_criterion == "rss":
                model_fit_score = rss

            elif selection_criterion == "log_rmse":
                model_fit_score = fit_error

            elif selection_criterion == "aic":
                model_fit_score = aic

            elif selection_criterion == "aicc":
                model_fit_score = aicc

            else:
                model_fit_score = bic

            # ---------- Estimate profit-maximizing price ----------#
            d_grid_hat = np.asarray(
                dhat_model_i(p_grid),
                dtype=float,
            )

            profit_grid_hat = (p_grid - c) * d_grid_hat - F

            valid_grid = (
                np.isfinite(d_grid_hat)
                & np.isfinite(profit_grid_hat)
                & (d_grid_hat > 0)
            )

            if not np.any(valid_grid):
                raise ValueError("No valid demand predictions over price grid.")

            profit_grid_valid = np.where(
                valid_grid,
                profit_grid_hat,
                -np.inf,
            )

            i_opt = int(np.argmax(profit_grid_valid))
            p_opt_hat = float(p_grid[i_opt])
            d_opt_hat = float(d_grid_hat[i_opt])
            profit_opt_hat = float(profit_grid_hat[i_opt])

            # ----------elasticity at p_opt_hat ----------#
            vhat_opt = float(
                np.asarray(
                    vhat_calculator(
                        model_info=model_info_i,
                        p=p_opt_hat,
                    )
                ).reshape(-1)[0]
            )

            inv_umar = p_opt_hat / (p_opt_hat - c)
            optimality_gap = abs(vhat_opt - inv_umar)

            # elasticity agreement diagnostic ----------#

            vhat_compare = vhat_opt

            # boundary diagnostic ----------#
            at_boundary = (
                p_opt_hat <= price_min + boundary_distance
                or p_opt_hat >= price_max - boundary_distance
            )

            # minimum elasticity check ----------#

            elasticity_valid = np.isfinite(vhat_opt) and vhat_opt >= minimum_elasticity

            # model-selection score ----------#

            if np.isfinite(model_fit_score) and elasticity_valid:
                selection_score = model_fit_score

            else:
                selection_score = np.inf

            # ---------- RidgeCV-selected alpha ----------#

            if "model" in model_info_i and hasattr(model_info_i["model"], "alpha_"):
                selected_alpha = float(model_info_i["model"].alpha_)
            else:
                selected_alpha = 0.0

            # ---------- Log-model structural diagnostics ----------#

            beta1 = np.nan
            beta2 = np.nan
            log_quad_vhat_range = np.nan

            # local curvature diagnostics evaluated at each model's own p_opt_hat
            logq_1prime = np.nan
            logq_2prime = np.nan
            logq_dvhat_dp = np.nan
            logq_d2d_over_d = np.nan
            logq_d2d_over_dd = np.nan

            logql_1prime = np.nan
            logql_2prime = np.nan
            logql_dvhat_dp = np.nan
            logql_d2d_over_d = np.nan
            logql_d2d_over_dd = np.nan

            ql_1prime = np.nan
            ql_2prime = np.nan
            ql_dvhat_dp = np.nan
            ql_d2d_over_d = np.nan
            ql_d2d_over_dd = np.nan

            if model_type == "log_level":
                beta1 = float(model_info_i["coef"][0])

            elif model_type == "quadratic":
                beta1 = float(model_info_i["coef"][0])

                beta2 = float(model_info_i["coef"][1])

                p_ref = p_opt_hat

                d_ref = float(dhat_model_i(p_ref))

                ql_1prime = beta1 + 2.0 * beta2 * p_ref

                ql_2prime = 2.0 * beta2

                # dv/dp
                if d_ref > 1e-12:

                    ql_dvhat_dp = (
                        -(ql_1prime / d_ref)
                        - p_ref * (ql_2prime / d_ref)
                        + p_ref * (ql_1prime / d_ref) ** 2
                    )

                    # D'' / D
                    ql_d2d_over_d = ql_2prime / d_ref

                # D'' / D'
                if abs(ql_1prime) > 1e-12:

                    ql_d2d_over_dd = ql_2prime / ql_1prime

            elif model_type == "log_log":

                beta1_log_log = float(model_info_i["coef"][0])
                log_log_vhat = -beta1_log_log

            elif model_type == "log_quadratic":

                beta1 = float(model_info_i["coef"][0])

                beta2 = float(model_info_i["coef"][1])

                # vhat_range diagnostic ... elasticity change
                # total change in fitted elasticity across price range
                log_quad_vhat_range = (
                    2.0 * abs(beta2) * abs(np.log(price_max / price_min))
                )

                # elasticity at geometric midpoint of the allowable price range
                p_mid = np.sqrt(price_min * price_max)

                log_quad_vhat_mid = -(beta1 + 2.0 * beta2 * np.log(p_mid))

                # ---------- LogQ local curvature diagnostics ----------#
                #
                # g(p) = log(d)
                #
                # with
                #
                # g(p) =
                #     beta0
                #     + beta1 * log(p)
                #     + beta2 * log(p)^2
                #
                # evaluated at the LogQ model's own p_opt_hat

                p_ref = p_opt_hat

                log_p_ref = np.log(p_ref)

                h_z = beta1 + 2.0 * beta2 * log_p_ref

                # g'(p)
                logq_1prime = h_z / p_ref

                # g''(p)
                logq_2prime = (2.0 * beta2 - h_z) / p_ref**2

                # dvhat / dp
                logq_dvhat_dp = -2.0 * beta2 / p_ref

                # D'' / D
                logq_d2d_over_d = logq_1prime**2 + logq_2prime

                # D'' / D'
                if abs(logq_1prime) > 1e-12:

                    logq_d2d_over_dd = logq_d2d_over_d / logq_1prime

            elif model_type in (
                "log_quadratic_level",
                "log_quadratic_level_lp",
            ):

                beta1 = float(model_info_i["coef"][0])
                beta2 = float(model_info_i["coef"][1])

                # ---------- LogQL local curvature diagnostics ----------#
                # g(p) = log(d) = beta0 + beta1*p + beta2*p^2
                # evaluated at the LogQL model's own p_opt_hat

                p_ref = p_opt_hat

                logql_1prime = beta1 + 2.0 * beta2 * p_ref

                logql_2prime = 2.0 * beta2

                # dv/dp
                logql_dvhat_dp = -logql_1prime - p_ref * logql_2prime

                # D'' / D
                logql_d2d_over_d = logql_1prime**2 + logql_2prime

                # D'' / D'
                if abs(logql_1prime) > 1e-12:

                    logql_d2d_over_dd = logql_d2d_over_d / logql_1prime

            # --------- Store model diagnostics ----------#

            model_diagnostics[model_type] = {
                "fit_error": float(fit_error),
                # total RSS
                "rss": float(rss),
                "rss_level": float(rss_level),
                # regional log-space RSS
                "rss_log_low": float(rss_log_low),
                "rss_log_high": float(rss_log_high),
                # regional level-demand RSS
                "rss_level_low": float(rss_level_low),
                "rss_level_high": float(rss_level_high),
                "n_parameters": int(n_parameters),
                "aic": float(aic),
                "aicc": float(aicc),
                "bic": float(bic),
                "selection_criterion": selection_criterion,
                "selection_score": float(selection_score),
                "optimality_gap": float(optimality_gap),
                "p_opt_hat": float(p_opt_hat),
                "d_opt_hat": float(d_opt_hat),
                "profit_opt_hat": float(profit_opt_hat),
                "vhat_opt": float(vhat_opt),
                "vhat_compare": float(vhat_compare),
                "inv_umar": float(inv_umar),
                "at_boundary": bool(at_boundary),
                "elasticity_valid": bool(elasticity_valid),
                "selected_alpha": float(selected_alpha),
                "tilt_penalty": float(model_info_i.get("tilt_penalty", 0.0)),
                "tilt_error": float(model_info_i.get("tilt_error", np.nan)),
                "tilt_error_low": float(
                    model_info_i.get(
                        "tilt_error_low",
                        np.nan,
                    )
                ),
                "tilt_error_high": float(
                    model_info_i.get(
                        "tilt_error_high",
                        np.nan,
                    )
                ),
                "tilt_imbalance": float(
                    model_info_i.get(
                        "tilt_imbalance",
                        np.nan,
                    )
                ),
                "beta1": beta1,
                "beta2": beta2,
                "log_quad_vhat_range": log_quad_vhat_range,
                # b2 regularization
                "b2_lambda_model": float(model_info_i.get("b2_lambda_model", 0.0)),
                # LogQ local curvature diagnostics
                "logq_1prime": float(logq_1prime),
                "logq_2prime": float(logq_2prime),
                "logq_dvhat_dp": float(logq_dvhat_dp),
                "logq_d2d_over_d": float(logq_d2d_over_d),
                "logq_d2d_over_dd": float(logq_d2d_over_dd),
                # LogQL local curvature diagnostics
                "logql_1prime": float(logql_1prime),
                "logql_2prime": float(logql_2prime),
                "logql_dvhat_dp": float(logql_dvhat_dp),
                "logql_d2d_over_d": float(logql_d2d_over_d),
                "logql_d2d_over_dd": float(logql_d2d_over_dd),
                # Quadratic local curvature diagnostics
                "ql_1prime": float(ql_1prime),
                "ql_2prime": float(ql_2prime),
                "ql_dvhat_dp": float(ql_dvhat_dp),
                "ql_d2d_over_d": float(ql_d2d_over_d),
                "ql_d2d_over_dd": float(ql_d2d_over_dd),
                # error message
                "error_message": None,
            }

            fitted_models[model_type] = {
                "dhat_model": dhat_model_i,
                "model_info": model_info_i,
            }

        except Exception as exc:  # pylint: disable=broad-exception-caught

            model_diagnostics[model_type] = {
                "fit_error": np.inf,
                "rss": np.inf,
                "rss_level": np.inf,
                "rss_log_low": np.inf,
                "rss_log_high": np.inf,
                "rss_level_low": np.inf,
                "rss_level_high": np.inf,
                "n_parameters": 0,
                "aic": np.inf,
                "aicc": np.inf,
                "bic": np.inf,
                "selection_criterion": selection_criterion,
                "selection_score": np.inf,
                "optimality_gap": np.inf,
                "p_opt_hat": np.nan,
                "d_opt_hat": np.nan,
                "profit_opt_hat": np.nan,
                "vhat_opt": np.nan,
                "inv_umar": np.nan,
                "at_boundary": False,
                "elasticity_valid": False,
                "selected_alpha": np.nan,
                "error_message": str(exc),
                "vhat_compare": np.nan,
                "vhat_delta": np.nan,
                "vhat_at_linear_p_opt": np.nan,
                "vhat_delta_from_linear": np.nan,
                "tilt_penalty": np.nan,
                "tilt_error": np.nan,
                "tilt_error_low": np.nan,
                "tilt_error_high": np.nan,
                "tilt_imbalance": np.nan,
                "beta1": np.nan,
                "beta2": np.nan,
                "log_quad_vhat_range": np.nan,
                "logq_1prime": np.nan,
                "logq_2prime": np.nan,
                "logq_dvhat_dp": np.nan,
                "logq_d2d_over_d": np.nan,
                "logq_d2d_over_dd": np.nan,
                "logql_1prime": np.nan,
                "logql_2prime": np.nan,
                "logql_dvhat_dp": np.nan,
                "logql_d2d_over_d": np.nan,
                "logql_d2d_over_dd": np.nan,
                "b2_lambda_model": np.nan,
                "ql_1prime": np.nan,
                "ql_2prime": np.nan,
                "ql_dvhat_dp": np.nan,
                "ql_d2d_over_d": np.nan,
                "ql_d2d_over_dd": np.nan,
            }

        #### end of model fitting loop

    m_log_log = np.nan
    m_log_quadratic = np.nan
    markup_pct_delta = np.nan

    # ---- model diagnostics ... after loop----#

    # the reason for a model not being in model_diagnostics is if it failed to fit and raised an exception, which is caught above.

    if "log_log" in model_diagnostics and "log_quadratic" in model_diagnostics:
        # eps ... epsilon ... elasticity at the estimated optimum price for each model
        eps_log_log = model_diagnostics["log_log"]["vhat_opt"]
        eps_log_quadratic = model_diagnostics["log_quadratic"]["vhat_opt"]

        if eps_log_log > 1.0:
            m_log_log = eps_log_log / (eps_log_log - 1.0)

        # ... eps  = epsilon (elasticity)
        if eps_log_quadratic > 1.0:
            m_log_quadratic = eps_log_quadratic / (eps_log_quadratic - 1.0)

        if np.isfinite(m_log_log) and np.isfinite(m_log_quadratic):
            markup_pct_delta = 100.0 * abs(m_log_quadratic - m_log_log) / m_log_log

    if np.isfinite(log_quad_vhat_mid) and np.isfinite(log_log_vhat):
        log_models_vhat_delta = abs(log_quad_vhat_mid - log_log_vhat)

    else:
        log_models_vhat_delta = np.nan

    if "log_quadratic" in model_diagnostics:
        model_diagnostics["log_quadratic"]["log_quad_vhat_mid"] = float(
            log_quad_vhat_mid
        )

        model_diagnostics["log_quadratic"]["log_models_vhat_delta"] = float(
            log_models_vhat_delta
        )

    if "log_log" in model_diagnostics:
        model_diagnostics["log_log"]["log_log_vhat"] = float(log_log_vhat)

        model_diagnostics["log_log"]["log_models_vhat_delta"] = float(
            log_models_vhat_delta
        )

    # log-log vs log-quadratic AIC improvement
    logq_vs_loglog_aic_improvement = np.nan

    if "log_log" in model_diagnostics and "log_quadratic" in model_diagnostics:
        logq_vs_loglog_aic_improvement = (
            model_diagnostics["log_log"]["aic"]
            - model_diagnostics["log_quadratic"]["aic"]
        )

    #  log model diagnostics so they are available for reporting and plotting
    for model_type in ("log_log", "log_quadratic"):

        if model_type in model_diagnostics:
            model_diagnostics[model_type].update(
                {
                    "log_quad_vhat_mid": float(log_quad_vhat_mid),
                    "log_log_vhat": float(log_log_vhat),
                    "log_models_vhat_delta": float(log_models_vhat_delta),
                    "m_log_log": float(m_log_log),
                    "m_log_quadratic": float(m_log_quadratic),
                    "markup_pct_delta": float(markup_pct_delta),
                    "logq_vs_loglog_aic_improvement": float(
                        logq_vs_loglog_aic_improvement
                    ),
                }
            )

    # ---------- Identify valid candidate models ----------#

    valid_models = [
        model_type
        for model_type, diagnostics in model_diagnostics.items()
        if (
            diagnostics["elasticity_valid"]
            and np.isfinite(diagnostics["selection_score"])
        )
    ]

    if len(valid_models) == 0:

        error_messages = {
            model_type: diagnostics["error_message"]
            for model_type, diagnostics in model_diagnostics.items()
        }

        raise ValueError(
            "None of the candidate models could be selected. "
            f"Errors: {error_messages}"
        )
    # ---------- log-model challengers ----------#

    log_model_types = (
        "log_log",
        "log_quadratic",
        "log_quadratic_level",
    )

    valid_log_models = [
        model_type
        for model_type in log_model_types
        if (
            model_type in model_diagnostics
            and model_diagnostics[model_type]["elasticity_valid"]
            and np.isfinite(model_diagnostics[model_type]["aic"])
        )
    ]

    best_log_model_type = None

    if valid_log_models:

        best_log_model_type = min(
            valid_log_models,
            key=lambda model_type: model_diagnostics[model_type]["aic"],
        )

    # ------- calculate diagnostics relative to linear model ----- #

    p_opt_linear = model_diagnostics["linear"]["p_opt_hat"]

    price_range = price_max - price_min

    # elasticity of the linear model evaluated at its own p_opt_hat
    vhat_linear_at_linear_p_opt = vhat_calculator(
        model_info=fitted_models["linear"]["model_info"],
        p=p_opt_linear,
    )

    for model_type in candidate_model_types:

        p_opt_model = model_diagnostics[model_type]["p_opt_hat"]

        # p_opt divergence from linear
        model_diagnostics[model_type]["p_opt_linear_delta_frac"] = (
            abs(p_opt_model - p_opt_linear) / price_range
        )

        # elasticity evaluated at the linear model's p_opt_hat
        if model_type in fitted_models:

            vhat_at_linear_p_opt = vhat_calculator(
                model_info=fitted_models[model_type]["model_info"],
                p=p_opt_linear,
            )

            vhat_delta_from_linear = vhat_at_linear_p_opt - vhat_linear_at_linear_p_opt

        else:

            vhat_at_linear_p_opt = np.nan
            vhat_delta_from_linear = np.nan

        model_diagnostics[model_type]["vhat_at_linear_p_opt"] = float(
            vhat_at_linear_p_opt
        )

        model_diagnostics[model_type]["vhat_delta_from_linear"] = float(
            vhat_delta_from_linear
        )

    # ---------- Linear RSS log-model trust diagnostic ----------#
    #
    # Quadratic is the conservative default model.
    #
    # A log-family model is trusted only when:
    #
    # 1. the total linear-model log-RSS is below threshold, and
    # 2. the absolute difference between low-side and high-side
    #    linear log-RSS is below threshold.

    linear_rss = model_diagnostics["linear"]["rss"]

    linear_rss_log_low = model_diagnostics["linear"]["rss_log_low"]

    linear_rss_log_high = model_diagnostics["linear"]["rss_log_high"]

    linear_rss_log_diff = abs(linear_rss_log_high - linear_rss_log_low)

    log_model_rss_ok = (
        np.isfinite(linear_rss)
        and np.isfinite(linear_rss_log_diff)
        and linear_rss_log_diff < thr_rss_diff_max
        and linear_rss < thr_rss_max
    )

    # ============= Select model =============#

    # Quadratic is the conservative default model.
    selected_model_type = "quadratic"

    # ---------- Optional log-model challenger ----------#
    #
    # A log-family model can replace quadratic only when:
    #
    # 1. at least one valid log model is available, and
    # 2. the linear reference RSS diagnostics indicate
    #    that the additional curvature can be trusted.
    #
    # If multiple valid log models are available,
    # choose the one with the lowest AIC.

    if best_log_model_type is not None and log_model_rss_ok:
        selected_model_type = best_log_model_type

    if (
        selected_model_type != "quadratic"
        and model_diagnostics[selected_model_type]["p_opt_linear_delta_frac"]
        > p_opt_linear_delta_frac_max
    ):
        selected_model_type = "quadratic"

    if (
        selected_model_type != "quadratic"
        and model_diagnostics[selected_model_type]["at_boundary"]
    ):
        selected_model_type = "quadratic"

    #### end model selecton

    # ---------- Selection diagnostics ----------#

    for model_type in candidate_model_types:

        model_diagnostics[model_type]["thr_rss_diff_max"] = float(thr_rss_diff_max)
        model_diagnostics[model_type]["thr_rss_max"] = float(thr_rss_max)
        model_diagnostics[model_type]["linear_rss_log_diff"] = float(
            linear_rss_log_diff
        )
        model_diagnostics[model_type]["logq_aic_improvement"] = float(
            logq_vs_loglog_aic_improvement
        )
        model_diagnostics[model_type]["log_model_rss_ok"] = bool(log_model_rss_ok)
        model_diagnostics[model_type]["best_log_model_type"] = best_log_model_type

    dhat_model = fitted_models[selected_model_type]["dhat_model"]
    model_info = fitted_models[selected_model_type]["model_info"]
    model_info["selection_diagnostics"] = model_diagnostics[selected_model_type]

    return (
        dhat_model,
        model_info,
        selected_model_type,
        model_diagnostics,
        fitted_models,
    )


# --------------------- aggregate_price_demand ------------------#


def aggregate_price_demand(
    p,
    d,
    price_decimals=10,
):
    """
    Aggregate repeated prices for a log-demand regression.

    Demand is aggregated using its geometric mean, and the
    number of observations at each price is returned for use
    as regression sample weights.

    Parameters
    ----------
    p : array-like
        Historical prices.

    d : array-like
        Positive historical demand observations.

    price_decimals : int, default=10
        Number of decimal places used to identify repeated prices.

    Returns
    -------
    p_unique : ndarray
        Sorted unique prices.

    d_geo_mean : ndarray
        Geometric mean demand at each unique price.

    counts : ndarray
        Number of observations at each unique price.
    """

    p = np.asarray(p, dtype=float)
    d = np.asarray(d, dtype=float)

    if len(p) != len(d):
        raise ValueError("p and d must have the same length.")

    if np.any(p <= 0):
        raise ValueError("Prices must be positive.")

    if np.any(d <= 0):
        raise ValueError("Demand must be positive.")

    p_group = np.round(p, decimals=price_decimals)

    p_unique = np.unique(p_group)

    d_geo_mean = np.empty(len(p_unique))
    counts = np.empty(len(p_unique), dtype=int)

    for j, price in enumerate(p_unique):
        mask = p_group == price

        d_geo_mean[j] = np.exp(np.mean(np.log(d[mask])))

        counts[j] = np.sum(mask)

    return p_unique, d_geo_mean, counts


# ---------------- price_filter_local_window -----------------#


def price_filter_local_window(
    p_fit,
    d_fit,
    counts,
    p_opt_hat,
    price_delta,
    price_delta_plus,
    min_fit_obs=3,
):
    """
    Restrict aggregated demand-fit observations to a local price
    window around the current estimated optimal price.

    The local price window is

        p_opt_hat * (1 - local_window_frac)
        <= p <=
        p_opt_hat * (1 + local_window_frac)

    where

        local_window_frac = price_delta + price_delta_plus.

    Parameters
    ----------
    p_fit : array-like
        Aggregated historical prices.

    d_fit : array-like
        Aggregated historical demand corresponding to p_fit.

    counts : array-like
        Number of observations represented by each aggregated
        price-demand point.

    p_opt_hat : float
        Current estimated profit-maximizing price used as the
        center of the local price window.

    price_delta : float
        Fractional local exploration distance around p_opt_hat.

    price_delta_plus : float
        Additional fractional width added to price_delta when
        defining the local fitting window.

    min_fit_obs : int, optional
        Minimum number of price points required after filtering.
        If fewer points remain, the original unfiltered data are
        returned. Defaults to 3.

    Returns
    -------
    p_fit_filtered : ndarray
        Filtered prices.

    d_fit_filtered : ndarray
        Filtered demand observations.

    counts_filtered : ndarray
        Filtered observation counts.
    """

    p_fit = np.asarray(p_fit, dtype=float)
    d_fit = np.asarray(d_fit, dtype=float)
    counts = np.asarray(counts, dtype=float)

    if not np.isfinite(p_opt_hat):
        return p_fit, d_fit, counts

    local_window_frac = price_delta + price_delta_plus

    price_window_min = p_opt_hat * (1.0 - local_window_frac)

    price_window_max = p_opt_hat * (1.0 + local_window_frac)

    window_mask = (p_fit >= price_window_min) & (p_fit <= price_window_max)

    # Keep the full data set if filtering would leave
    # too few price points to fit the candidate models.
    if np.sum(window_mask) < min_fit_obs:
        return p_fit, d_fit, counts

    return (
        p_fit[window_mask],
        d_fit[window_mask],
        counts[window_mask],
    )


# ------------------ mab_price_optimization_loop -----------------#
def mab_price_optimization_loop(
    price_min: float,
    price_max: float,
    c: float,
    F: float,
    p_opt: float,
    demand_type: str,
    demand_params: dict,
    sigma_log: float,
    N_loop=20,
    N_price_repeat=1,
    price_repeat_aggregation=False,
    ridge_cv_linear=False,
    ridge_cv_q=False,
    ridge_cv_lql=False,
    huber=False,
    huber_epsilon=1.35,
    b2_lambda_q=0.0,
    b2_lambda_lql=0.0,
    b2_lambda_lql_lp=0.0,
    candidate_model_types=None,
    primary_model_type=None,
    N_initial_prices=10,
    initial_prices=None,
    initial_demand=None,
    price_delta=0.1,
    price_delta_plus=0.05,
    price_local_window=False,
    boundary_buffer=0.02,
    selection_criterion="rss",
    tilt_penalty_linear=0.0,
    tilt_penalty_log_level=0.0,
    tilt_penalty_log_log=0.0,
    tilt_penalty_log_quadratic=0.0,
    tilt_penalty_log_quadratic_level=0.0,
    linear_tilt_error_max=np.inf,
    linear_tilt_imbalance_max=np.inf,
    thr_rss_diff_max=np.inf,
    thr_rss_max=np.inf,
    logq_aic_improvement_min=2.0,
    p_opt_linear_delta_frac_max=0.20,
    verbose=2,
):
    """Multi-armed bandit (MAB) price optimization loop.

        This function simulates a price optimization process using a multi-armed bandit approach.
        It iteratively selects prices, observes simulated demand, and updates the estimated profit-maximizing price.
        The function keeps track of price and demand histories, fits demand models, and selects the optimal price based on the estimated demand models.

        The loop consists of an initial exploration phase followed by local price
        experimentation around the estimated profit-maximizing price. After the
        initial prices, prices are sampled in +/- pairs around a fixed p_opt_hat
        anchor. The anchor is set at the beginning of each pair and remains fixed
        until both the positive and negative price_delta observations are completed.
        Each price is repeated N_price_repeat times.

        The function returns a dictionary containing the selected demand model information, including the fitted model, model diagnostics, and the selected model type.


    Args:
        price_min (float): Minimum allowable price.
        price_max (float): Maximum allowable price.
        c (float): Unit variable cost.
        F (float): Fixed cost.
        p_opt (float): True or theoretical optimal price used for evaluation.
        demand_type (str): Demand model type. Supported values include "linear",
            "constant_elasticity", and "logit".
        demand_params (dict): Parameters for the selected demand model.
        sigma_log (float): Standard deviation of the log-normal noise added to the
            demand simulation.
        N_loop (int, optional): Number of iterations of the MAB price optimization loop.
            Defaults to 20.

        N_price_repeat : int, optional
            Number of observations at each newly selected price after the
            initial exploration. When initial_prices is generated automatically,
            it also determines the number of repetitions of each initial price.
            When initial_prices is supplied explicitly, that sequence is used
            exactly as provided. Defaults to 1.

        price_repeat_aggregation : bool, optional
            Aggregate repeated-price demand observations before model fitting.

            If False, retain all individual price-demand observations.
            If True, replace repeated observations at each price with one
            mean-demand observation weighted by the repeat count.

            Defaults to False.

        candidate_model_types : list[str] or None, optional
            Candidate demand model types to evaluate. If None, defaults to
            ["linear", "log_log", "log_quadratic", "log_quadratic_level"]. Additional supported
            models, such as "log_level" and "log_quadratic_level", may optionally be included.

        primary_model_type : str or None, optional
            Optional candidate model to use as the initial model choice
            during each demand-model selection step.

            If None, the initial model is selected normally using the
            configured selection criterion.

            If specified, the model must be included in
            candidate_model_types. Existing divergence and boundary
            safeguards remain active and may still cause fallback to
            linear.

            Defaults to None.

        N_initial_prices : int, optional
            Number of distinct evenly spaced prices used during automatic
            initial exploration. Ignored when initial_prices is provided.
            Defaults to 10.

        initial_prices : list of float or None, optional
            Explicit sequence of prices to use during the initial exploration
            phase.

            If None, initial exploration prices are generated automatically
            using N_initial_prices evenly spaced prices over
            [price_min, price_max], with each price repeated
            N_price_repeat times.

            If provided, the supplied sequence is used exactly as the initial
            exploration path and N_initial_prices is ignored. N_price_repeat
            is not used to construct the supplied initial sequence, although
            it continues to control price repetition later in the MAB loop.

            All supplied prices must lie within [price_min, price_max].

        initial_demand : list or None, default=None
            Optional list of observed demand values corresponding to `initial_prices`.
            If provided, these demand values are used directly instead of generating
            demand observations with `get_demand()`. Must have the same length as
            `initial_prices`. If None, demand is generated normally from the specified
            demand model and noise level.

        price_delta : float, optional
            Fractional price perturbation around the estimated profit-maximizing
            price after the initial exploration.

            At the beginning of each local exploration pair, the current
            p_opt_hat is saved as a fixed anchor. Prices are then sampled at:

                p = p_opt_anchor * (1 + price_delta)
                p = p_opt_anchor * (1 - price_delta)

            Both sides therefore use the same p_opt anchor. Each price is repeated
            N_price_repeat times before switching sides. After both sides are
            completed, the latest p_opt_hat becomes the anchor for the next pair.
            Defaults to 0.1.

        price_delta_plus : float, optional

            Additional fractional width used when defining the optional
            local fitting window.

            When price_local_window is True, aggregated historical
            price-demand observations are retained for model fitting when

                p_opt_hat * (1 - local_window_frac)
                <= p_fit <=
                p_opt_hat * (1 + local_window_frac)

            where

                local_window_frac = price_delta + price_delta_plus.

            For example, with price_delta=0.10 and
            price_delta_plus=0.05, the fitting window extends
            approximately +/-15% around the current p_opt_hat.

            Defaults to 0.05.

        price_local_window : bool, optional
            If True, after the initial exploration phase the aggregated
            historical price-demand observations used for model fitting
            are restricted to a local window around the current
            estimated optimal price.

            Historical observations remain in p_history and d_history;
            only the observations used for the current demand-model fit
            are filtered.

            The local window is defined using

                price_delta + price_delta_plus

            around the current p_opt_hat.

            If filtering would leave fewer than the minimum required
            number of unique price points, the full unfiltered data are
            used instead.

            Defaults to False.

        ridge_cv_linear : bool, default=False
            If True, fit the "linear" model using RidgeCV.

        ridge_cv_q : bool, default=False
            If True, fit the "quadratic" model using RidgeCV.

            Cannot currently be combined with b2_lambda_q > 0.

        ridge_cv_lql : bool, default=False
            If True, fit the "log_quadratic_level" model using RidgeCV.


            Cannot currently be combined with b2_lambda_lql > 0.

        b2_lambda_q : float, optional
            Beta2-only regularization strength for the
            "quadratic" demand model.

            Defaults to 0.0.

        b2_lambda_lql : float, optional
            Beta2-only regularization strength for the
            "log_quadratic_level" demand model.

            Defaults to 0.0.

        b2_lambda_lql_lp : float, optional
            Beta2-only regularization strength for
            "log_quadratic_level_lp".

            Defaults to 0.0.

    Returns:
        dict: Dictionary containing the simulation outcomes, including:
            - p_history
            - d_history
            - selected_model
            - selected_model_type
            - dhat_model
            - model_info
            - model_fit_errors
            - p_opt_hat_history
            - p_opt_vhat_history
            - p_opt_hat_error_history
            - d_opt_hat_history
            - d_sigma0_history
            - vhat_opt_history
            - inv_umar_history
            - profit_opt_history
    """
    # ---------- Asertions -----------------#

    if (ridge_cv_linear or ridge_cv_q or ridge_cv_lql) and huber:

        raise ValueError("RidgeCV and huber cannot currently " "be used together.")

    if not 0 <= price_delta < 1:
        raise ValueError("price_delta must satisfy 0 <= price_delta < 1.")

    if not 0 <= price_delta_plus < 1:
        raise ValueError("price_delta_plus must satisfy " "0 <= price_delta_plus < 1.")

    if price_delta + price_delta_plus >= 1:
        raise ValueError("price_delta + price_delta_plus must be less than 1.")

    if not isinstance(price_local_window, (bool, np.bool_)):
        raise ValueError("price_local_window must be True or False.")

    if not isinstance(
        price_repeat_aggregation,
        (bool, np.bool_),
    ):
        raise ValueError("price_repeat_aggregation must be True or False.")

    # ---------- candidate_model_types -----------------#

    if candidate_model_types is None:

        candidate_model_types = [
            "linear",
            "log_log",
            "log_quadratic",
        ]

    # --------- Price Exploration Initialization --------

    if initial_prices is None:

        # Construct evenly spaced initial exploration prices.
        # Each price is repeated N_price_repeat times.
        prices_initial = np.linspace(
            start=price_min,
            stop=price_max,
            num=N_initial_prices,
        ).tolist()

        prices_initial = [
            round(p, 2) for p in prices_initial for _ in range(N_price_repeat)
        ]

    else:
        # Use user-supplied initial price sequence exactly as provided.
        # N_initial_prices is ignored.
        prices_initial = [float(p) for p in initial_prices]

    # ---------- Validate initial prices ----------#

    if len(prices_initial) == 0:
        raise ValueError("initial_prices must contain at least one price.")

    if any(p < price_min or p > price_max for p in prices_initial):
        raise ValueError("All initial_prices must lie within [price_min, price_max].")

    if N_loop < len(prices_initial):
        raise ValueError(
            f"N_loop={N_loop} must be >= the number of "
            f"initial exploration observations "
            f"({len(prices_initial)})."
        )

    # ---------- Validate initial demand ----------#

    if initial_demand is not None:
        if len(initial_demand) != len(prices_initial):
            raise ValueError(
                f"initial_demand length ({len(initial_demand)}) must match "
                f"the number of initial prices ({len(prices_initial)})."
            )
        if any(d < 0 for d in initial_demand):
            raise ValueError("All initial_demand values must be non-negative.")

    # --------- Price Grid used to maximize estimated profit ----------#

    p_grid = np.maximum(
        np.linspace(price_min, price_max, 500),
        c + 0.02,
    )

    # ---- min fit observations log_quadratic requires at least three unique prices ---- #

    min_fit_obs = 3
    # min_fit_obs = max(min_fit_obs, len(prices_initial))

    # ------------Histories -------------------------#

    p_history = []
    d_history = []

    selected_model_history = []
    model_fit_error_history = []

    dhat_model = None
    model_info = None
    selected_model_type = None

    p_opt_hat_history = []
    p_opt_vhat_history = []
    d_opt_hat_history = []
    vhat_opt_history = []
    inv_umar_history = []
    profit_opt_history = []
    p_opt_hat_error_history = []
    d_sigma0_history = []

    m_log_log_history = []
    m_log_quadratic_history = []
    markup_pct_delta_history = []

    model_diagnostics_history = []
    fitted_models_history = []

    # ---- Initial values used by the price-control logic ------#

    # Each local exploration pair uses a fixed p_opt anchor:
    # first +price_delta, then -price_delta.
    price_direction = 1
    p_opt_anchor = np.nan

    p_opt_hat = np.nan
    vhat_opt_i = np.nan
    inv_umar_i = np.nan

    p_opt_vhat_i = np.nan
    p_opt_hat_error = np.nan
    d_opt_hat = np.nan
    profit_opt_hat = np.nan
    di_sigma0_i = np.nan

    # ======== MAB loop ===============================#

    repeat_count = 0
    prev_pi = None

    for i in range(N_loop):

        if verbose > 0:
            print(f"i = {i}", end=", ")

        # ---------- Price selection ----------#

        if i < len(prices_initial):

            # Use the initial exploration path exactly as constructed
            # or supplied by the user.
            pi = prices_initial[i]

        else:

            # ---------- Select local exploration price ----------#
            if repeat_count == 0 or repeat_count >= N_price_repeat:
                if not np.isfinite(p_opt_hat):
                    raise ValueError(
                        "p_opt_hat must be finite after the initial "
                        "exploration before local price exploration begins."
                    )

                # Start of a new +/- exploration pair:
                # freeze the latest p_opt_hat as the pair anchor.
                if price_direction == 1:

                    p_opt_anchor = float(p_opt_hat)

                # Both sides of the pair are centered on the same anchor.
                pi = p_opt_anchor * (1.0 + price_direction * price_delta)

                pi = float(
                    np.clip(
                        round(pi, 2),
                        price_min,
                        price_max,
                    )
                )

                # Alternate direction for the next price block.
                price_direction *= -1

                # First observation at this new price.
                repeat_count = 1

            else:

                # Repeat the previously selected local exploration price.
                pi = prev_pi

                repeat_count += 1

        prev_pi = pi

        # -------- Simulated observed demand ------------------#

        if initial_demand is not None and i < len(initial_demand):

            di = float(initial_demand[i])

        else:

            di = round(
                get_demand(
                    demand_type,
                    demand_params,
                    pi,
                    sigma_log=sigma_log,
                ),
                1,
            )

        # Noiseless demand is used only as a simulation diagnostic.
        # It is not used for model fitting or model selection.
        di_sigma0_i = round(
            get_demand(
                demand_type,
                demand_params,
                pi,
                sigma_log=0,
            ),
            1,
        )
        if verbose > 1:
            print(
                f"pi = {pi}, " f"di = {di}, " f"di_sigma0_i = {di_sigma0_i}",
                end=", ",
            )

        d_sigma0_history.append(di_sigma0_i)

        # -----------Update price and demand history ------------------#

        p_history.append(pi)
        d_history.append(di)

        # ------ Optionally aggregate repeated prices ----------------#

        if price_repeat_aggregation:

            p_fit, d_fit, counts = aggregate_price_demand(
                p=p_history,
                d=d_history,
            )

            p_fit = np.asarray(
                p_fit,
                dtype=float,
            )

            d_fit = np.asarray(
                d_fit,
                dtype=float,
            )

            counts = np.asarray(
                counts,
                dtype=float,
            )

        else:

            # Keep every individual price-demand observation.
            p_fit = np.asarray(
                p_history,
                dtype=float,
            )

            d_fit = np.asarray(
                d_history,
                dtype=float,
            )

            # Each observation represents one sample.
            counts = np.ones(
                len(p_fit),
                dtype=float,
            )
        # ---------- Optional local price window ----------#

        if price_local_window and i >= len(prices_initial) and np.isfinite(p_opt_hat):

            p_fit, d_fit, counts = price_filter_local_window(
                p_fit=p_fit,
                d_fit=d_fit,
                counts=counts,
                p_opt_hat=p_opt_hat,
                price_delta=price_delta,
                price_delta_plus=price_delta_plus,
                min_fit_obs=min_fit_obs,
            )

        # ---------- Sample Weights ----------#
        #
        # With aggregation, each mean-demand point is weighted by
        # the number of observations represented at that price.
        # Without aggregation, each individual observation has
        # unit weight.
        sample_weights = counts

        # ---- Wait until enough unique prices are available -------#

        N_unique_prices = len(np.unique(p_fit))

        if N_unique_prices < min_fit_obs:

            if verbose > 1:
                print("selected_model = none, p_opt_hat = nan")

            selected_model_history.append(None)
            model_fit_error_history.append(None)
            p_opt_hat_history.append(np.nan)
            p_opt_vhat_history.append(np.nan)
            d_opt_hat_history.append(np.nan)
            vhat_opt_history.append(np.nan)
            inv_umar_history.append(np.nan)
            profit_opt_history.append(np.nan)
            p_opt_hat_error_history.append(np.nan)

            continue

        # ------ Fit and select the demand model -------------

        try:
            (
                dhat_model,
                model_info,
                selected_model_type,
                model_diagnostics,
                fitted_models,
            ) = select_demand_model(
                p=p_fit,
                d=d_fit,
                candidate_model_types=candidate_model_types,
                primary_model_type=primary_model_type,
                price_min=price_min,
                price_max=price_max,
                c=c,
                F=F,
                sample_weights=sample_weights,
                boundary_buffer=boundary_buffer,
                selection_criterion=selection_criterion,
                ridge_cv_linear=ridge_cv_linear,
                ridge_cv_q=ridge_cv_q,
                ridge_cv_lql=ridge_cv_lql,
                huber=huber,
                huber_epsilon=huber_epsilon,
                b2_lambda_q=b2_lambda_q,
                b2_lambda_lql=b2_lambda_lql,
                b2_lambda_lql_lp=b2_lambda_lql_lp,
                tilt_penalty_linear=tilt_penalty_linear,
                tilt_penalty_log_level=tilt_penalty_log_level,
                tilt_penalty_log_log=tilt_penalty_log_log,
                tilt_penalty_log_quadratic=tilt_penalty_log_quadratic,
                tilt_penalty_log_quadratic_level=tilt_penalty_log_quadratic_level,
                linear_tilt_error_max=linear_tilt_error_max,
                thr_rss_diff_max=thr_rss_diff_max,
                thr_rss_max=thr_rss_max,
                linear_tilt_imbalance_max=linear_tilt_imbalance_max,
                logq_aic_improvement_min=logq_aic_improvement_min,
                p_opt_linear_delta_frac_max=p_opt_linear_delta_frac_max,
            )

        except ValueError as exc:

            if verbose > 1:
                print(f"selected_model = none, " f"model selection failed: {exc}")

            selected_model_history.append(None)
            model_fit_error_history.append(None)
            p_opt_hat_history.append(np.nan)
            p_opt_vhat_history.append(np.nan)
            d_opt_hat_history.append(np.nan)
            vhat_opt_history.append(np.nan)
            inv_umar_history.append(np.nan)
            profit_opt_history.append(np.nan)
            p_opt_hat_error_history.append(np.nan)

            continue

        _selected_model_error = model_diagnostics[selected_model_type]["fit_error"]

        m_log_log = model_diagnostics[selected_model_type].get(
            "m_log_log",
            np.nan,
        )

        m_log_quadratic = model_diagnostics[selected_model_type].get(
            "m_log_quadratic",
            np.nan,
        )

        markup_pct_delta = model_diagnostics[selected_model_type].get(
            "markup_pct_delta",
            np.nan,
        )

        # ---Estimate demand and profit over the price grid ---------------------#

        d_grid_hat = np.asarray(
            dhat_model(p_grid),
            dtype=float,
        )

        profit_grid_hat = (p_grid - c) * d_grid_hat - F

        # Ignore any invalid grid predictions
        valid_grid = (
            np.isfinite(d_grid_hat) & np.isfinite(profit_grid_hat) & (d_grid_hat >= 0)
        )

        if not np.any(valid_grid):
            print(
                f"selected_model = {selected_model_type}, "
                "no valid profit-grid predictions"
            )
            print(f"d_grid_hat = {d_grid_hat}")

            selected_model_history.append(selected_model_type)

            model_fit_error_history.append(
                model_diagnostics[selected_model_type]["fit_error"]
            )

            p_opt_hat_history.append(np.nan)
            p_opt_vhat_history.append(np.nan)
            d_opt_hat_history.append(np.nan)
            vhat_opt_history.append(np.nan)
            inv_umar_history.append(np.nan)
            profit_opt_history.append(np.nan)
            p_opt_hat_error_history.append(np.nan)

            continue

        profit_grid_valid = np.where(
            valid_grid,
            profit_grid_hat,
            -np.inf,
        )

        i_opt = int(np.argmax(profit_grid_valid))

        # ------- Estimated optimum ----------------#

        p_opt_hat = round(
            float(p_grid[i_opt]),
            2,
        )
        d_opt_hat = round(
            float(d_grid_hat[i_opt]),
            1,
        )
        profit_opt_hat = round(
            float(profit_grid_hat[i_opt]),
            2,
        )

        # ---- Elasticity at the estimated optimum  -----#

        vhat_opt_i = round(
            float(
                np.asarray(
                    vhat_calculator(
                        model_info=model_info,
                        p=p_opt_hat,
                    )
                ).reshape(-1)[0]
            ),
            2,
        )

        inv_umar_i = round(
            p_opt_hat / (p_opt_hat - c),
            2,
        )  # inverse unit margine

        # price = c * vhat / (vhat - 1) ...  constant elasticity assumption
        if not np.isfinite(vhat_opt_i) or np.isclose(vhat_opt_i, 1.0):
            p_opt_vhat_i = np.nan
        else:
            p_opt_vhat_i = round(
                c * vhat_opt_i / (vhat_opt_i - 1.0),
                2,
            )

        # ------- price  vs true price error -----------------#
        # p_opt is the known true optimum in the simulation.
        # It is not used by model fitting or model selection.

        p_opt_hat_error = round(
            abs(p_opt_hat - p_opt),
            2,
        )

        # histories

        selected_model_history.append(selected_model_type)
        model_fit_error_history.append(
            model_diagnostics[selected_model_type]["fit_error"]
        )

        p_opt_hat_history.append(p_opt_hat)
        p_opt_vhat_history.append(p_opt_vhat_i)
        d_opt_hat_history.append(d_opt_hat)
        vhat_opt_history.append(vhat_opt_i)
        inv_umar_history.append(inv_umar_i)
        profit_opt_history.append(profit_opt_hat)
        p_opt_hat_error_history.append(p_opt_hat_error)

        m_log_log_history.append(m_log_log)
        m_log_quadratic_history.append(m_log_quadratic)
        markup_pct_delta_history.append(markup_pct_delta)
        model_diagnostics_history.append(model_diagnostics)
        fitted_models_history.append(fitted_models)

        # ------------- loop verbose prints ----------------#
        if verbose > 1 and i < N_loop - 1:
            print(
                f"selected_model = {selected_model_type}",
                f", model_fit_error = {round(model_diagnostics[selected_model_type]["fit_error"],2)}",
                f", p_opt_hat = {p_opt_hat}",
                f", p_opt_vhat_i = {p_opt_vhat_i}",
                f", p_opt_hat_error = {p_opt_hat_error}",
                f", vhat_opt_i = {vhat_opt_i}",
                f", inv_umar_i = {inv_umar_i}",
                f", d_opt_hat = {d_opt_hat}",
                f", d_sigma0_i = {di_sigma0_i}",
                f", profit_opt_hat = {profit_opt_hat}",
            )

    # afer loop results
    if verbose > 0:
        print(
            f"selected_model = {selected_model_type}",
            f", p_opt_hat = {p_opt_hat}",
            f", p_opt_vhat_i = {p_opt_vhat_i}",
            f", p_opt_hat_error = {p_opt_hat_error}",
            f", vhat_opt_i = {vhat_opt_i}",
            f", inv_umar_i = {inv_umar_i}",
            f", d_opt_hat = {d_opt_hat}",
            f", d_sigma0_i = {di_sigma0_i}",
            f", profit_opt_hat = {profit_opt_hat}",
        )

    results_dict = {
        # observed MAB path
        "p_history": p_history,
        "d_history": d_history,
        "d_sigma0_history": d_sigma0_history,
        # selected-model path
        "selected_model": selected_model_history,
        "p_opt_hat_history": p_opt_hat_history,
        "d_opt_hat_history": d_opt_hat_history,
        "p_opt_hat_error_history": p_opt_hat_error_history,
        # final selected model
        "selected_model_type": selected_model_type,
        "dhat_model": dhat_model,
        "model_info": model_info,
        # all models
        "model_diagnostics": model_diagnostics,
        "fitted_models": fitted_models,
        # optional iteration-by-iteration deep diagnostics
        "model_diagnostics_history": model_diagnostics_history,
        "fitted_models_history": fitted_models_history,
    }

    return results_dict
