# Import libraries
from pathlib import Path
import cvxpy as cp
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import json

from demand_xgboost import evaluate_xg_boost


#------------------------------------------------------------------------------------------------
# Demand Response (cvxpy)
#------------------------------------------------------------------------------------------------
def D_cp(pi, l, y, E):
    """
    Demand response function in cvxpy format

    Parameters
    ----------
    pi: Distribution prices (NOK/kWh)
    l: Day-ahead energy prices (NOK/kWh)
    y: Baseline demand with no distribution price (kWh)
    E: Price elasticity of demand

    Returns
    -------
    D: Demand after the price response, y * ((pi + l) / l)^E (kWh)
    """
    return cp.multiply((pi + l)**E, cp.multiply(l**(-E), y))


#------------------------------------------------------------------------------------------------
# Demand Response (NumPy)
#------------------------------------------------------------------------------------------------
def D_np(pi, l, y, E):
    """
    Demand response function in numpy format, for simulation and visualization

    Parameters
    ----------
    pi: Distribution prices (NOK/kWh)
    l: Day-ahead energy prices (NOK/kWh)
    y: Baseline demand with no distribution price (kWh)
    E: Price elasticity of demand

    Returns
    -------
    D: Demand after the price response, y * ((pi + l) / l)^E (kWh)
    """
    pi = np.asarray(pi, dtype=float).reshape(-1, 1)
    l = np.asarray(l, dtype=float).reshape(-1, 1)
    y = np.asarray(y, dtype=float)
    return (((pi + l)**E) * (l**(-E))) * y


#------------------------------------------------------------------------------------------------
# Demand Response Step with Shifting
#------------------------------------------------------------------------------------------------
def D_fill_step_np(D_current, pi_prev, pi_next, l, y, E, t, fill):
    """
    Shifts reduced demand to the cheapest hour, in response to a small increase in price

    Parameters
    ----------
    D_current: Demand before the price increase (kWh)
    pi_prev: Old distribution prices (NOK/kWh)
    pi_next: New distribution prices (after the increase) (NOK/kWh)
    l: Day-ahead energy prices (NOK/kWh)
    y: Baseline demand with no distribution price (kWh)
    E: Price elasticity of demand
    t: Index of the hour whose price is increased
    fill: If True, add the reduced demand to the hour with the lowest total price pi_next + l (demand shifting)

    Returns
    -------
    D_next: Demand after the price increase (kWh)
    """
    D_next = D_current.copy()

    D_before = D_np(pi_prev, l, y, E)
    D_after = D_np(pi_next, l, y, E)

    D_reduced = D_before[t, :] - D_after[t, :]

    # valley = np.argmin(D_next, axis=0)
    valley = np.argmin(pi_next + l, axis=0)

    D_next[t, np.arange(D_next.shape[1])] -= D_reduced
    if fill:
        D_next[valley, np.arange(D_next.shape[1])] += D_reduced

    return D_next


#------------------------------------------------------------------------------------------------
# Demand Response with Shifting
#------------------------------------------------------------------------------------------------
def D_fill_np(pi_end, l, y, E, fill, pi_step=0.01):
    """
    Simulates demand under a full tariff by repeatedly applying D_fill_step_np

    Parameters
    ----------
    pi_end: Final distribution prices (NOK/kWh)
    l: Energy (day-ahead spot) prices (NOK/kWh)
    y: Baseline demand with no distribution price (kWh)
    E: Price elasticity of demand
    fill: If True, shift demand reduction to the cheapest hour
    pi_step: Price increment (NOK/kWh)

    Returns
    -------
    D_current: Demand after the tariff is applied (kWh)
    """
    T = len(pi_end)
    pi_end = np.asarray(pi_end, dtype=float).reshape(-1, 1)
    l = np.asarray(l, dtype=float).reshape(-1, 1)
    y = np.asarray(y, dtype=float)
    
    # initialize pi and D_current
    pi = np.zeros(pi_end.shape)
    D_current = y.copy()
    for t in range(T):
        while pi[t] < pi_end[t]:
            pi_prev = pi.copy()
            pi[t] += pi_step
            D_current = D_fill_step_np(D_current, pi_prev, pi, l, y, E, t, fill)

    return D_current


#------------------------------------------------------------------------------------------------
# Convex Relaxation (Overshoot Revenue Bound)
#------------------------------------------------------------------------------------------------
def convex_relaxation_overshoot(l, D_scenarios, T, N, R, E, beta=6/7):
    """
    Convex relaxation algorithm to solve the demand reduction problem - replaces the revenue constraint with the hyperplane through each hour's revenue intercept a_t, which slightly overshoots the revenue target. Solved with SCIP.

    Parameters
    ----------
    l: Day-ahead energy prices (NOK/kWh)
    D_scenarios: Baseline demand scenarios (kWh)
    T: Number of timesteps (h)
    N: Number of scenarios
    R: Revenue bound (NOK)
    E: Price elasticity of demand
    beta: CVaR confidence level

    Returns
    -------
    pi: Optimal distribution prices (NOK/kWh)
    alpha: CVaR threshold (kWh)
    M: Epigraph of peak demand CVaR (kWh)
    """

    # Decision variables
    pi = cp.Variable((T, 1), nonneg=True) # distribution prices
    alpha = cp.Variable() # CVaR threshold variable
    M = cp.Variable((N, 1), nonneg=True)

    # Define constraints
    con = []
    # vector of pi_t intercepts
    a = ((N * R * l**E).reshape(-1, 1) / D_scenarios.sum(axis=1).reshape(-1, 1)) ** (1/(1 + E))

    # CVaR epigraph constraints for scenario peak demand
    con.append(M >= cp.reshape(cp.max(D_cp(pi, l, D_scenarios, E), axis=0), (N, 1), order="C") - alpha)

    # Constant revenue approximation using pi_t intercepts to define a hyperplane
    con.append(cp.sum(pi / a) == 1)
        
    obj = cp.Minimize(alpha + (1 / (N * (1 - beta))) * cp.sum(M))
    
    prob = cp.Problem(obj, con)
    prob.solve(solver="SCIP")

    return pi.value, alpha.value, M.value


#------------------------------------------------------------------------------------------------
# Numerical Algorithm
#------------------------------------------------------------------------------------------------
# Numerical algorithm - increasing pi until meeting revenue target
def greedy_algo(l, D_scenarios, T, N, R, E, fill, beta, pi_step=0.01):
    """
    Numerical algorithm to solve the demand reduction problem - repeatedly adds pi_step to the hour that most reduces the CVaR of peak demand, until average revenue exceeds R.

    Parameters
    ----------
    l: Day-ahead energy prices (NOK/kWh)
    D_scenarios: Baseline demand scenarios (kWh)
    T: Number of timesteps (h)
    N: Number of scenarios
    R: Revenue bound (NOK)
    E: Price elasticity of demand
    fill: If True, shift reduced demand to the cheapest hour
    beta: CVaR confidence level
    pi_step: Price increment (NOK/kWh)

    Returns
    -------
    pi: Distribution prices (NOK/kWh)
    """

    # initialize pi, D, R
    pi = np.zeros((T, 1))
    D_current = D_np(pi, l, D_scenarios, E)
    R_current = (1 / N) * np.sum(pi  * D_current)
    cdf = np.arange(1, N + 1) / N
    
    while R_current <= R:   
            
        cvar_per_hour = np.zeros(T)
        
        for t in range(T):
            # get peaks if we add pi_step to hour t
            pi_temp = pi.copy()
            pi_temp[t] = pi_temp[t] + pi_step
            D_temp = D_fill_step_np(D_current, pi, pi_temp, l, D_scenarios, E, t, fill)
            peak_s = np.max(D_temp, axis=0)
            
            # sort peaks in ascending order and calculate alpha
            sorted_peaks = np.sort(peak_s)
            alpha_idx = min(cdf.searchsorted(beta), N - 1)
            alpha_val = sorted_peaks[alpha_idx]
            
            # calculate CVaR for hour t
            M_s = np.maximum(sorted_peaks - alpha_val, 0)
            cvar_objective = alpha_val + (1 / (1 - beta)) * np.mean(M_s)
            
            cvar_per_hour[t] = cvar_objective
            
        # time step selection that minimizes CVaR
        best_t = np.argmin(cvar_per_hour)
        pi_old = pi.copy()
        pi[best_t] += pi_step

        D_current = D_fill_step_np(D_current, pi_old, pi, l, D_scenarios, E, best_t, fill)
        R_current = (1 / N) * np.sum(pi  * D_current)

    return pi


#------------------------------------------------------------------------------------------------
# TOU Benchmark
#------------------------------------------------------------------------------------------------
def tou_strategy(T, tariff, start_hour, end_hour):
    """
    Time-of-use benchmark applies a flat distribution price over a fixed window of hours.

    Parameters
    ----------
    T: Number of timesteps (h)
    tariff: Distribution price during the window (NOK/kWh)
    start_hour: First hour of the window
    end_hour: Last hour of the window

    Returns
    -------
    pi: Distribution prices (NOK/kWh)
    """
    pi = np.zeros((T, 1))
    pi[start_hour-1:end_hour] = tariff
    return pi


#------------------------------------------------------------------------------------------------
# Prepare Parameters
#------------------------------------------------------------------------------------------------
def prep_params(data, scenarios, community, date):
    """
    Formats parameters for demand response functions

    Parameters
    ----------
    data: Residential energy consumption data from load_iflex_aggregate
    scenarios: Demand scenarios in raw format
    community: Community (iFlex group) name
    date: Date to select

    Returns
    -------
    T: Number of timesteps (h)
    N: Number of scenarios
    df_scenarios: Demand scenarios (kWh)
    D_scenarios: Demand scenarios in a time x scenario array (kWh)
    D: Baseline demand (kWh)
    l: Day-ahead prices (NOK/kWh)
    """
    date = pd.Timestamp(date).normalize()
    comm_scenarios = scenarios.loc[scenarios["Group"] == community].copy()
    if "Date" in comm_scenarios.columns:
        scenario_dates = pd.to_datetime(comm_scenarios["Date"]).dt.normalize()
        comm_scenarios = comm_scenarios.loc[scenario_dates == date]
    if comm_scenarios.empty:
        raise ValueError(f"No scenarios found for community {community} on {date.date()}")

    df_scenarios = (
        comm_scenarios.pivot_table(index="Hour", columns="scenario_id", values="scenario_demand", aggfunc="first")
        .sort_index()
    )
    T = len(df_scenarios.index)
    N = len(df_scenarios.columns)
    if T == 0 or N == 0:
        raise ValueError(f"Scenario matrix is empty for community {community} on {date.date()}")
    D_scenarios = df_scenarios.to_numpy(dtype=float)

    date_mask = pd.to_datetime(data["Date"]).dt.normalize().eq(date)
    original_df = (
        data.loc[(data["Group"] == community) & date_mask, ["Hour", "Energy_Price_NOK_kWh", "demand"]]
        .drop_duplicates("Hour")
        .set_index("Hour")
        .sort_index()
    )
    missing_hours = df_scenarios.index.difference(original_df.index)
    if len(missing_hours) > 0:
        raise ValueError(f"Missing energy prices for {community} hours: {missing_hours.tolist()}")
    original_df = original_df.reindex(df_scenarios.index)
    D = original_df["demand"].to_numpy(dtype=float).reshape(-1,1)
    l = original_df["Energy_Price_NOK_kWh"].to_numpy(dtype=float).reshape(-1,1)

    return T, N, df_scenarios, D_scenarios, D, l 


#------------------------------------------------------------------------------------------------
# Run Algorithms
#------------------------------------------------------------------------------------------------
def run_algos(data, communities, scenarios, date, R, E, fill=False, beta=6/7, pi_step=0.01):
    """
    Runs numerical and convex relaxation algorithms for each community on one date.

    Parameters
    ----------
    data: Residential energy consumption data from load_iflex_aggregate
    communities: Community names to run
    scenarios: Demand scenarios in raw format
    date: Date to optimize
    R: Revenue bound (NOK)
    E: Price elasticity of demand
    fill: If True, perform demand shifting in the numerical algorithm
    beta: CVaR confidence level
    pi_step: Price increment for the numerical algorithm (NOK/kWh)

    Returns
    -------
    results: Resulting pi by community, hour, and algorithm (NOK/kWh)
    """
    cvar_result_frames = []
    scenario_groups = set(scenarios["Group"].dropna().unique())
    for community in [group for group in communities if group in scenario_groups]:
        T, N, df_scenarios, D_scenarios, _, l = prep_params(data, scenarios, community, date)

        if len(l) != T:
            raise ValueError(f"Price vector and scenario matrix length mismatch for {community}")

        # apply peak shaving algorithms
        pi_greedy = greedy_algo(l, D_scenarios, T, N, R, E, fill=fill, beta=beta, pi_step=pi_step)
        pi_relaxed_overshoot, _, _ = convex_relaxation_overshoot(l, D_scenarios, T, N, R, E, beta=beta)

        df_greedy = pd.DataFrame(
            {
                "Group": community,
                "Hour": df_scenarios.index.to_numpy(),
                "Method": "greedy",
                "pi": pi_greedy.flatten()
            }
        )
        df_relaxed_overshoot = pd.DataFrame(
            {
                "Group": community,
                "Hour": df_scenarios.index.to_numpy(),
                "Method": "relaxed_overshoot",
                "pi": pi_relaxed_overshoot.flatten()
            }
        )
    
        cvar_result_frames.append(df_greedy)
        cvar_result_frames.append(df_relaxed_overshoot)

    if not cvar_result_frames:
        return pd.DataFrame(columns=["Group", "Hour", "pi_greedy", "pi_relaxed_overshoot"])
    return pd.concat(cvar_result_frames, ignore_index=True)


#------------------------------------------------------------------------------------------------
# Run Benchmarks
#------------------------------------------------------------------------------------------------
def run_benchmarks(data, communities, date, R, E, tariff, start_hour, end_hour, fill=False, beta=0, pi_step=0.01):
    """
    Computes benchmark tariffs for each community on one date, using the numerical algorithm to perform perfect foresight.

    Parameters
    ----------
    data: Residential energy consumption data from load_iflex_aggregate
    communities: Community names to run
    date: Date to optimize
    R: Revenue bound (NOK)
    E: Price elasticity of demand
    tariff: TOU distribution price (NOK/kWh)
    start_hour: First hour of the TOU window
    end_hour: Last hour of the TOU window
    fill: If True, perform demand shifting in the numerical algorithm
    beta: CVaR confidence level
    pi_step: Price increment for the numerical algorithm (NOK/kWh)

    Returns
    -------
    results: Resulting pi by community, hour, and algorithm (NOK/kWh)
    """
    date = pd.Timestamp(date).normalize()
    N = 1
    cvar_result_frames = []
    for community in communities:
        l = data.loc[(data["Group"] == community) & (pd.to_datetime(data["Date"]).dt.normalize() == date), "Energy_Price_NOK_kWh"].to_numpy(dtype=float).reshape(-1, 1)
        D = data.loc[(data["Group"] == community) & (pd.to_datetime(data["Date"]).dt.normalize() == date), "demand"].to_numpy(dtype=float).reshape(-1, 1)
        T = len(l)

        if len(l) != T:
            raise ValueError(f"Price vector and scenario matrix length mismatch for {community}")

        # apply peak shaving algorithms
        pi_foresight_greedy = greedy_algo(l, D, T, N, R, E, fill=fill, beta=beta, pi_step=pi_step)
        pi_tou = tou_strategy(T, tariff=tariff, start_hour=start_hour, end_hour=end_hour)

        df_foresight = pd.DataFrame(
            {
                "Group": community,
                "Hour": np.arange(1, T + 1),
                "Method": "foresight_greedy",
                "pi": pi_foresight_greedy.flatten()
            }
        )
        df_tou = pd.DataFrame(
            {
                "Group": community,
                "Hour": np.arange(1, T + 1),
                "Method": "tou",
                "pi": pi_tou.flatten()
            }
        )
    
        cvar_result_frames.append(df_foresight)
        cvar_result_frames.append(df_tou)

    if not cvar_result_frames:
        return pd.DataFrame(columns=["Group", "Hour", "pi_foresight_greedy", "pi_foresight_relaxed", "pi_tou"])
    return pd.concat(cvar_result_frames, ignore_index=True)


#------------------------------------------------------------------------------------------------
# Relabel Legend
#------------------------------------------------------------------------------------------------
def relabel_legend(labels, ax=None):
    """
    Removes the legend title and renames legend entries.

    Parameters
    ----------
    labels: Mapping from internal algorithm names to paper names
    ax: Axes whose legend to update

    Returns
    -------
    None
    """
    ax = ax or plt.gca()
    legend = ax.get_legend()
    if legend is None:
        return
    legend.set_title("")
    for text in legend.get_texts():
        text.set_text(labels.get(text.get_text(), text.get_text()))


#------------------------------------------------------------------------------------------------
# Paper Figure Size
#------------------------------------------------------------------------------------------------
def paper_figsize():
    """
    Default figure width with a 1:2 height-to-width ratio, matching the paper figures.

    Parameters
    ----------
    None

    Returns
    -------
    figsize: (width, width / 2) from the current matplotlib default width (inches)
    """
    width, _ = plt.rcParams["figure.figsize"]
    return (width, width / 2)


#------------------------------------------------------------------------------------------------
# Plot Simulation
#------------------------------------------------------------------------------------------------
# simulate demand response over time period
def plot_simulation(time, pi, N, D_scenarios, D, E, l, fill, date, community):
    """
    Plots one day's tariff solution and its effect on demand scenarios and actual demand.

    Parameters
    ----------
    time: Hours to plot on the x-axis
    pi: Distribution prices(NOK/kWh)
    N: Number of scenarios
    D_scenarios: Baseline demand scenarios (kWh)
    D: Actual baseline demand (kWh)
    E: Price elasticity of demand
    l: Day-ahead energy prices (NOK/kWh)
    fill: If True, simulate demand shifting
    date: Date to simulate
    community: Community name

    Returns
    -------
    None
    """
    cmap = sns.color_palette("flare", as_cmap=True)
    scenario_colors = cmap(np.linspace(0.0, 0.4, N))
    real_color = "firebrick"

    fig, ax1 = plt.subplots(figsize=paper_figsize())
    # pi on first axis
    ax1.bar(time, pi, color="olive", alpha=0.5)
    ax1.set_xlabel('Time (h)')
    ax1.set_ylabel("Tariff (NOK)")
    # simulated demand response on second axis
    ax2 = ax1.twinx()
    ax2.spines["right"].set_visible(True)  # keep the demand axis line on the right
    D_scenarios_response = D_fill_np(pi, l, D_scenarios, E, fill)
    for i in range(N):
        scen_color = scenario_colors[i]
        ax2.plot(time, D_scenarios_response[:, i], color=scen_color, alpha=0.7, linewidth=1, label=f"Scenario {i+1}")
        ax2.fill_between(time, D_scenarios_response[:, i], D_scenarios[:, i], color=scen_color, alpha=0.2)
        ax2.plot(time, D_scenarios[:, i], color=scen_color, alpha=0.7, linewidth=1, linestyle="--")

    D_response = D_fill_np(pi, l, D, E, fill).flatten()
    ax2.plot(time, D_response, color=real_color, label="Reality")
    ax2.fill_between(time, D_response, D.flatten(), color=real_color, alpha=0.2)

    ax2.plot(time, D, color=real_color, linestyle="--")
    ax2.set_ylabel("Demand (kWh)")
    ax2.legend(loc="upper left")
    print(f"Peak Shaving Simulation - Community {community} on {date.strftime('%Y-%m-%d')}")
    plt.show()


#------------------------------------------------------------------------------------------------
# Regret Histogram
#------------------------------------------------------------------------------------------------
def plot_regret_histogram(metrics_beta, beta, palette, labels, ymax=30):
    """
    Histogram of regret (peak demand minus perfect-foresight peak demand) for each algorithm.

    Parameters
    ----------
    metrics_beta: Metrics for one beta
    beta: CVaR confidence level
    palette: Mapping from method name to color
    labels: Mapping from method name to legend label
    ymax: Upper limit of y-axis

    Returns
    -------
    None
    """
    plt.figure(figsize=paper_figsize())
    sns.histplot(data = metrics_beta, x="regret", bins=20, hue="Method_cvar", palette=palette, element='step', fill=True, alpha=0.5)
    plt.ylim(0,ymax)
    plt.xlabel("Regret (kWh)")
    plt.ylabel("Frequency")
    relabel_legend(labels)
    print(f"Regret Distribution Across Communities and Dates, Beta = {beta:.3f}")
    plt.show()


#------------------------------------------------------------------------------------------------
# Regret vs. Original Peak Demand
#------------------------------------------------------------------------------------------------
def plot_regret_vs_original_peak(metrics_beta, beta, palette, labels):
    """
    Scatter plot of regret vs. baseline peak demand, for each algorithm.

    Parameters
    ----------
    metrics_beta: Metrics for one beta
    beta: CVaR confidence level
    palette: Mapping from method name to color
    labels: Mapping from method name to legend label

    Returns
    -------
    None
    """
    plt.figure(figsize=paper_figsize())
    sns.scatterplot(data=metrics_beta, x="peak_demand_original", y="regret", hue="Method_cvar", palette=palette, alpha=0.75)
    plt.xlabel("Demand (kWh)")
    plt.ylabel("Regret")
    relabel_legend(labels)
    print(f"Regret vs. Original Peak Demand, Beta = {beta:.3f}")
    plt.show()


#------------------------------------------------------------------------------------------------
# Revenue Histogram
#------------------------------------------------------------------------------------------------
def plot_revenue_histogram(metrics_beta, beta, R, palette, labels, height=30):
    """
    Histogram of achieved revenue for each algorithm

    Parameters
    ----------
    metrics_beta: Metrics for one beta
    beta: CVaR confidence level
    R: Revenue bound (NOK)
    palette: Mapping from method name to color
    labels: Mapping from method name to legend label
    height: Upper limit of y-axis and height of target line

    Returns
    -------
    None
    """
    plt.figure(figsize=paper_figsize())
    sns.histplot(data = metrics_beta, x="R_sum_cvar", bins=20, hue="Method_cvar", palette=palette, element='step', fill=True, alpha=0.5)
    plt.vlines(R, 0, height, color="red", linestyle="--", label="Revenue Target")
    plt.ylim(0,height)
    plt.xlabel("Revenue (NOK)")
    plt.ylabel("Frequency")
    relabel_legend(labels)
    print(f"Revenue Distribution Across Communities and Dates, Beta = {beta:.3f}")
    plt.show()


#------------------------------------------------------------------------------------------------
# Pareto Figure
#------------------------------------------------------------------------------------------------
def pareto_figure(metrics_cvar, metrics_benchmark, R, palette_cvar, palette_benchmark, labels, ymin=2, ymax=3):
    """
    Scatter plot of peak demand vs. revenue for each algorithm.

    Parameters
    ----------
    metrics_cvar: Mean peak demand and revenue per date and algorithm
    metrics_benchmark: Same for the benchmark methods
    R: Revenue bound (NOK)
    palette_cvar: Mapping from CVaR method name to color
    palette_benchmark: Mapping from benchmark method name to color
    labels: Mapping from method name to legend label
    ymin, ymax: Limits of y-axis (kWh)

    Returns
    -------
    None
    """
    overlapping = (abs(metrics_cvar["R_sum"] - metrics_benchmark["R_sum"]) < 0.01)
    metrics_cvar["R_sum_pareto"] = np.where(overlapping, metrics_cvar["R_sum"] + 0.02, metrics_cvar["R_sum"])
    plt.figure(figsize=paper_figsize())
    plt.vlines(R, ymin, ymax, color="slategrey", linestyle="--", label="Revenue Target")
    sns.scatterplot(data=metrics_benchmark, x="R_sum", y="peak_demand", hue="Method", palette=palette_benchmark)
    sns.scatterplot(data=metrics_cvar, x="R_sum_pareto", y="peak_demand", hue="Method", palette=palette_cvar)
    relabel_legend(labels)
    # Half-transparent white legend background, no border, so points behind it look paler
    legend = plt.gca().get_legend()
    legend.set_frame_on(True)
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("none")
    legend.get_frame().set_alpha(0.5)
    plt.xlabel("Revenue (NOK)")
    plt.ylabel("Peak Demand")
    plt.ylim(ymin,ymax)


#------------------------------------------------------------------------------------------------
# Test Models
#------------------------------------------------------------------------------------------------
def test_models(results, data, date, E, fill):
    """
    Simulates the tariff solutions on actual demand for one date and computes peak demand and revenue for each community and algorithm.

    Parameters
    ----------
    results: Tariffs from run_algos or run_benchmarks (NOK/kWh)
    data: Residential energy consumption data from load_iflex_aggregate
    date: Date to simulate
    E: Price elasticity of demand
    fill: If True, simulate demand shifting

    Returns
    -------
    metrics: One row per Group and Method with peak_demand (kWh) and R_sum, the achieved revenue (NOK)
    """
    date = pd.Timestamp(date).normalize()
    date_mask = pd.to_datetime(data["Date"]).dt.normalize().eq(date)
    df_test = data.loc[date_mask].copy()
    df_test = df_test.merge(results, how="left", on=["Group", "Hour"])
    if df_test["pi"].isna().any().any():
        missing = df_test.loc[df_test["pi"].isna().any(axis=1), ["Group", "Hour"]]
        raise ValueError(f"Missing optimized tariffs for {len(missing)} group-hour rows on {date.date()}")

    agg_dict = {}
    # Simulating effect of peak shaving strategies on test day, one community and method at a time
    # so that shifted demand (fill=True) moves to the cheapest hour of that community's own day
    simulated_demand = []
    for _, df_group in df_test.groupby(["Group", "Method"], observed=True, sort=False):
        l = df_group["Energy_Price_NOK_kWh"].to_numpy(dtype=float).reshape(-1, 1)
        y = df_group["demand"].to_numpy(dtype=float).reshape(-1, 1)
        D_group = D_fill_np(df_group["pi"].to_numpy(), l, y, E, fill).ravel()
        simulated_demand.append(pd.Series(D_group, index=df_group.index))

    df_test[f"demand"] = pd.concat(simulated_demand)
    df_test[f"R"] = df_test["pi"] * df_test[f"demand"]

    agg_dict[f"peak_demand"] = ("demand", "max")
    agg_dict[f"R_sum"] = ("R", "sum")

    df_metrics_grouped = (
        df_test.groupby(["Group", "Method"], observed=True).agg(**agg_dict).reset_index()
    )
    
    return df_metrics_grouped


#------------------------------------------------------------------------------------------------
# Run XGBoost
#------------------------------------------------------------------------------------------------
# get raw kernel regression results for a range of dates
def run_xgboost(data, dates, feature_cols, model_params, output_path):
    """
    Produces XGBoost predictions for each date and saves them to CSV.

    Parameters
    ----------
    data: Residential energy consumption data from load_iflex_aggregate
    dates: Dates to predict
    feature_cols: Features used by the model
    model_params: XGBoost hyperparameters
    output_path: Path of the CSV file to write

    Returns
    -------
    raw_results: XGBoost demand predictions (kWh)
    """
    frames = []
    for date in dates:
        xgb_results, _ = evaluate_xg_boost(data, [date], feature_cols, model_params)
        if not xgb_results.empty:
            frames.append(xgb_results)

    columns = ["Group", "Date", "time", "Hour", "demand", "prediction"]
    raw_results = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)
    raw_results.to_csv(output_path, index=False)
    return raw_results