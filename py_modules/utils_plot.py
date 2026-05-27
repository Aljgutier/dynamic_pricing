# utils_plot.py
"""
plot utilities
   - plost_hdi_width_over_time
   - _draws
   - _overlay_kde_series_colored
   - plot_kde_densities_at_prices
"""

from typing import Sequence
import arviz as az
import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import Normalize


# -------- plot_hdi_width_over_time ------- #
def plot_hdi_width_over_time(
    traces: Sequence[az.InferenceData],
    ax,
    var_names: Sequence[str] = ("a", "v", "sigma_log"),
    hdi_prob: float = 0.95,
    overlay_mean: bool = False,
    mean_on_secondary_axis: bool = True,
    title: str = None,
    xlabel: str = None,
    ylabel: str = None,
    linestyle: str = "-",
    secondary_linestyle: str = "--",
    linewidth: float = 1.6,
    secondary_linewidth: float = 1.3,
    line_color: str = "#013bff",
    secondary_line_color: str = "black",
    marker: str = None,
    secondary_marker: str = None,
    background_color: str = "#bcc7d6",
    show_grid: bool = True,
    grid_alpha: float = 0.6,
    grid_color: str = "white",
    y_lims: tuple[float, float] = None,
    y_lims2: tuple[float, float] = None,
) -> dict[str, list[tuple[float, float]]]:
    """
    Plot HDI width across iterations with optional posterior mean overlay.
    """

    steps = np.arange(1, len(traces) + 1)

    var_hdi_bounds: dict[str, list[tuple[float, float]]] = {
        var: [] for var in var_names
    }

    if background_color is not None:
        ax.set_facecolor(background_color)

    for var in var_names:
        widths: list[float] = []
        means: list[float] = []

        for idata in traces:
            posterior_da = idata.posterior[var]  # posterior variable
            # hdi_da = az.hdi(posterior_da, hdi_prob=hdi_prob)  # compute HDI
            hdi_da = az.hdi(posterior_da, hdi_prob)

            if isinstance(hdi_da, np.ndarray):
                vals = hdi_da.ravel()

            elif hasattr(hdi_da, "to_array"):  # Dataset case
                vals = hdi_da.to_array().values.ravel()

            else:  # DataArray case
                try:
                    low = float(hdi_da.sel(hdi="lower").values)
                    high = float(hdi_da.sel(hdi="higher").values)
                    vals = None
                except Exception:
                    vals = hdi_da.values.ravel()

            if vals is not None:
                low = float(np.min(vals))
                high = float(np.max(vals))

            var_hdi_bounds[var].append((low, high))
            widths.append(high - low)

            if overlay_mean:
                means.append(float(poster ior_da.mean().item()))

        # --- primary HDI width line ---
        line1 = ax.plot(
            steps,
            widths,
            marker=marker,
            linestyle=linestyle,
            linewidth=linewidth,
            color=line_color,
            label=f"{var} HDI width",
        )

        ax.set_title(
            title
            if title is not None
            else f"{var}: posterior tightening over time (HDI {int(hdi_prob*100)}%)"
        )
        ax.set_xlabel(xlabel if xlabel is not None else "step t")
        ax.set_ylabel(
            ylabel if ylabel is not None else "HDI width",
            color=line_color,
        )
        ax.tick_params(axis="y", labelcolor=line_color)

        if show_grid:
            ax.set_axisbelow(True)
            ax.grid(True, color=grid_color, alpha=grid_alpha)
        else:
            ax.grid(False)

        lines = list(line1)
        labels = [l.get_label() for l in lines]

        if y_lims is not None:
            ax.set_ylim(y_lims)

        # --- overlay posterior mean ---
        if overlay_mean:
            if mean_on_secondary_axis:
                ax2 = ax.twinx()

                line2 = ax2.plot(
                    steps,
                    means,
                    marker=secondary_marker,
                    linestyle=secondary_linestyle,
                    linewidth=secondary_linewidth,
                    color=secondary_line_color,
                    label=f"{var} posterior mean",
                )

                ax2.set_ylabel("posterior mean", color=secondary_line_color)
                ax2.tick_params(axis="y", labelcolor=secondary_line_color)

                lines += list(line2)
                labels += [l.get_label() for l in line2]

            else:
                line2 = ax.plot(
                    steps,
                    means,
                    marker=secondary_marker,
                    linestyle=secondary_linestyle,
                    linewidth=secondary_linewidth,
                    color=secondary_line_color,
                    label=f"{var} posterior mean",
                )

                lines += list(line2)
                labels += [l.get_label() for l in line2]

        ax.legend(lines, labels, loc="best")

        if y_lims2 is not None and overlay_mean and mean_on_secondary_axis:
            ax2.set_ylim(y_lims2)

    return var_hdi_bounds


# -------- _draws ------- #
def _draws(idata, var_name: str) -> np.ndarray:
    """
    Flatten posterior draws for a scalar variable from an ArviZ InferenceData.
    """
    return np.asarray(idata.posterior[var_name]).reshape(-1)


# -------- _overlay_kde_series_colored ------- #
def _overlay_kde_series_colored(
    series,
    ax,
    title: str,
    xlabel: str,
    bw_method=None,
    quantile_clip=(0.002, 0.998),
    cmap_name="viridis",
    show_colorbar=True,
    colors=None,
    title_override=None,
    xlabel_override=None,
    ylabel_override=None,
    background_color=None,
    linewidth=2.0,
    linewidth_progression=False,
    linewidth_min=1.0,
    linewidth_max=3.0,
    reverse_linewidth_progression=False,
    show_grid=True,
    grid_alpha=1.0,
    grid_color="white",
):
    # LINE WIDTHS
    ts = np.array([t for t, _ in series])
    norm = Normalize(vmin=ts.min(), vmax=ts.max())
    cmap = mpl.colormaps.get_cmap(cmap_name)

    n_series = len(series)
    if linewidth_progression and n_series > 1:
        linewidths = np.linspace(linewidth_min, linewidth_max, n_series)
        if reverse_linewidth_progression:
            linewidths = linewidths[::-1]
    else:
        linewidths = np.full(n_series, linewidth)

    # BACKGROUND COLOR
    if background_color is not None:
        ax.set_facecolor(background_color)

    all_s = np.concatenate([s for _, s in series])
    lo, hi = np.quantile(all_s, list(quantile_clip))

    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo, hi = float(np.min(all_s)), float(np.max(all_s))

    pad = 0.02 * (hi - lo) if hi > lo else 1.0
    grid = np.linspace(lo - pad, hi + pad, 400)

    ts = np.array([t for t, _ in series])
    norm = Normalize(vmin=ts.min(), vmax=ts.max())
    cmap = mpl.colormaps.get_cmap(cmap_name)

    for i, (t, s) in enumerate(series):
        linewidth = (linewidths[i],)
        s = np.asarray(s)
        s = s[np.isfinite(s)]
        if len(s) < 2:
            continue

        kde = gaussian_kde(s, bw_method=bw_method)

        if colors is not None:
            color = colors[i]
        else:
            color = cmap(norm(t))

        ax.plot(
            grid,
            kde(grid),
            color=color,
            alpha=0.9,
            linewidth=linewidths[i],
        )

    if show_colorbar:
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        ax.figure.colorbar(sm, ax=ax, label="step $i$")

    ax.set_title(title_override if title_override is not None else title)
    ax.set_xlabel(xlabel_override if xlabel_override is not None else xlabel)
    ax.set_ylabel(ylabel_override if ylabel_override is not None else "density")

    if show_grid:
        ax.set_axisbelow(True)
        ax.grid(True, color=grid_color, alpha=grid_alpha)
    else:
        ax.grid(False)
    return ax


# ------ plot_kde_densities_at_prices ----- #
def plot_kde_densities_at_prices(
    traces,
    prices_history,
    kde_plot_variable: str,
    variable_cost: float,
    fixed_cost: float,
    ax,
    price_min: float = None,
    price_max: float = None,
    Npgrid: int = 200,
    closed_form_optimal_price: bool = True,
    bw_method=None,
    cmap_name="viridis",
    t_max: int = None,
    t_min: int = 1,
    t_indices=None,
    show_colorbar: bool = True,
    colors=None,
    title=None,
    xlabel=None,
    ylabel=None,
    background_color=None,
    linewidth=2.0,
    linewidth_progression=False,
    linewidth_min=1.0,
    linewidth_max=3.0,
    reverse_linewidth_progression=False,
    show_grid=True,
    grid_alpha=0.6,
    grid_color="white",
):
    """
    Overlay KDEs across TS iterations for posterior-implied quantities.
    """
    valid_modes = {"demand", "revenue", "profit", "optimal_price"}
    if kde_plot_variable not in valid_modes:
        raise ValueError(
            "kde_plot_variable must be one of "
            "'demand', 'revenue', 'profit', or 'optimal_price'"
        )

    if len(traces) == 0:
        raise ValueError("traces must not be empty")

    if kde_plot_variable != "optimal_price":
        if len(traces) != len(prices_history):
            raise ValueError(
                "len(traces) must equal len(prices_history): "
                "one trace + one price per step"
            )

    if t_max is None:
        t_max = len(traces)
    t_max = min(t_max, len(traces))

    if kde_plot_variable == "optimal_price":
        if price_min is None or price_max is None:
            raise ValueError(
                "price_min and price_max must be provided when "
                "kde_plot_variable='optimal_price'"
            )
        if price_min >= price_max:
            raise ValueError("price_min must be less than price_max")

    demand_series = []
    revenue_series = []
    profit_series = []
    optimal_price_series = []

    # Needed only for grid-based optimal_price mode
    p_grid = None
    if kde_plot_variable == "optimal_price" and not closed_form_optimal_price:
        p_grid = np.linspace(price_min, price_max, Npgrid)

    for t, idata in enumerate(traces[:t_max], start=1):
        if t < t_min:
            continue
        if t_indices is not None and t not in t_indices:
            continue

        a_s = _draws(idata, "a")
        v_s = _draws(idata, "v")

        if kde_plot_variable in {"demand", "revenue", "profit"}:
            p_t = prices_history[t - 1]

            D_s = a_s * (p_t ** (-v_s))
            R_s = p_t * D_s
            Pi_s = (p_t - variable_cost) * D_s - fixed_cost

            demand_series.append((t, D_s))
            revenue_series.append((t, R_s))
            profit_series.append((t, Pi_s))

        elif kde_plot_variable == "optimal_price":
            if closed_form_optimal_price:
                p_star_s = (v_s / (v_s - 1.0)) * variable_cost
                p_star_s = np.clip(p_star_s, price_min, price_max)
            else:
                profit_grid = (
                    (p_grid[None, :] - variable_cost)
                    * a_s[:, None]
                    * (p_grid[None, :] ** (-v_s[:, None]))
                )
                idx_star = np.argmax(profit_grid, axis=1)
                p_star_s = p_grid[idx_star]

            optimal_price_series.append((t, p_star_s))

    if kde_plot_variable == "demand":
        _overlay_kde_series_colored(
            demand_series,
            ax=ax,
            title="Posterior density of demand at step-specific offered price $p_t$",
            xlabel="Demand $D(p_t)$",
            bw_method=bw_method,
            cmap_name=cmap_name,
            show_colorbar=show_colorbar,
            colors=colors,
            title_override=title,
            xlabel_override=xlabel,
            ylabel_override=ylabel,
            background_color=background_color,
            linewidth=linewidth,
            linewidth_progression=linewidth_progression,
            linewidth_min=linewidth_min,
            linewidth_max=linewidth_max,
            reverse_linewidth_progression=reverse_linewidth_progression,
            show_grid=show_grid,
            grid_alpha=grid_alpha,
            grid_color=grid_color,
        )

    elif kde_plot_variable == "revenue":
        _overlay_kde_series_colored(
            revenue_series,
            ax=ax,
            title="Posterior density of revenue at step-specific offered price $p_t$",
            xlabel="Revenue $R(p_i)$",
            bw_method=bw_method,
            cmap_name=cmap_name,
            show_colorbar=show_colorbar,
            colors=colors,
            title_override=title,
            xlabel_override=xlabel,
            ylabel_override=ylabel,
            background_color=background_color,
            linewidth=linewidth,
            linewidth_progression=linewidth_progression,
            linewidth_min=linewidth_min,
            linewidth_max=linewidth_max,
            reverse_linewidth_progression=reverse_linewidth_progression,
            show_grid=show_grid,
            grid_alpha=grid_alpha,
            grid_color=grid_color,
        )

    elif kde_plot_variable == "profit":
        _overlay_kde_series_colored(
            profit_series,
            ax=ax,
            title="Posterior density of profit at step-specific offered price $p_t$",
            xlabel="$Profit(p)$",
            bw_method=bw_method,
            cmap_name=cmap_name,
            show_colorbar=show_colorbar,
            colors=colors,
            title_override=title,
            xlabel_override=xlabel,
            ylabel_override=ylabel,
            background_color=background_color,
            linewidth=linewidth,
            linewidth_progression=linewidth_progression,
            linewidth_min=linewidth_min,
            linewidth_max=linewidth_max,
            reverse_linewidth_progression=reverse_linewidth_progression,
            show_grid=show_grid,
            grid_alpha=grid_alpha,
            grid_color=grid_color,
        )

    elif kde_plot_variable == "optimal_price":
        _overlay_kde_series_colored(
            optimal_price_series,
            ax=ax,
            title="Posterior density of draw-specific optimal price $p_s^*$",
            xlabel="Optimal price $p_s^*$",
            bw_method=bw_method,
            cmap_name=cmap_name,
            show_colorbar=show_colorbar,
            colors=colors,
            title_override=title,
            xlabel_override=xlabel,
            ylabel_override=ylabel,
            background_color=background_color,
            linewidth=linewidth,
            linewidth_progression=linewidth_progression,
            linewidth_min=linewidth_min,
            linewidth_max=linewidth_max,
            reverse_linewidth_progression=reverse_linewidth_progression,
            show_grid=show_grid,
            grid_alpha=grid_alpha,
            grid_color=grid_color,
        )

    return ax
