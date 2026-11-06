# Import libraries
from pathlib import Path
import json
import os
import math
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_PARAM_GRID = {
    "n_estimators": [100, 250],
    "max_depth": [2, 4],
    "learning_rate": [0.05],
    "min_child_weight": [5],
}

#~~~~~~~~~ Load & Aggregate Data ~~~~~~~~~#

#------------------------------------------------------------------------------------------------
# Build Merged Data from Raw Files
#------------------------------------------------------------------------------------------------
def _build_merged_data_from_raw(RAW_DATA_PATH, eur_to_nok) -> pd.DataFrame:
    """
    Merges iFlex data and ENTSO-E prices.

    Parameters
    ----------
    RAW_DATA_PATH: Path to the raw iFlex hourly data (data_hourly.csv)
    eur_to_nok: EUR to NOK exchange rate

    Returns
    -------
    df_all: Participant-hour rows with demand, group, temperature, Energy_Price_NOK_kWh and Energy_Experiment_Price_NOK_kWh
    """
    df_data_hourly = pd.read_csv(RAW_DATA_PATH)
    df_experiment_days = pd.read_csv("experiment_days.csv")
    df_participants = pd.read_csv("participants.csv")
    df_price_signals = pd.read_csv("price_signals.csv")
    df_price_hourly = pd.concat(
        [
            pd.read_csv("entsoe/norway_all_zones_prices_2020.csv"),
            pd.read_csv("entsoe/norway_all_zones_prices_2021.csv"),
        ],
        ignore_index=True,
    )

    df_data_hourly = (
        df_data_hourly.loc[df_data_hourly["Participation_Phase"] == "Phase_2"]
        .drop(columns=["Participation_Phase"])
        .reset_index(drop=True)
    )
    df_data_hourly["From"] = pd.to_datetime(df_data_hourly["From"], format="%Y-%m-%dT%H:%M:%SZ")
    df_data_hourly["Date"] = pd.to_datetime(df_data_hourly["Date"], format="%Y-%m-%d")

    df_experiment_days = (
        df_experiment_days.loc[df_experiment_days["Participation_Phase"] == "Phase_2"]
        .drop(columns=["Participation_Phase"])
        .reset_index(drop=True)
    )
    _ = df_experiment_days.merge(
        df_price_signals,
        left_on="Price_signal",
        right_on="Price_signal",
        how="inner",
    ).loc[:, ["Date", "Hour"]].drop_duplicates().reset_index(drop=True)

    df_participants = (
        df_participants.loc[
            (df_participants["Participation_Experiment"] == "Yes")
            & (df_participants["Participation_Phase"] == "Phase_2")
            & (df_participants["Participation_status_Phase2"] != "NA"),
            ["ID", "Region", "Control_Price_Phase2", "Group_Phase2"],
        ]
        .rename(columns={"Control_Price_Phase2": "Control_Price", "Group_Phase2": "Group"})
        .reset_index(drop=True)
    )

    region_to_node = {
        "Oslo": "NO1",
        "Stavanger": "NO2",
        "Trondheim": "NO3",
        "Tromsø": "NO4",
        "Bodø": "NO4",
        "Bergen": "NO5",
    }
    df_participants["Node"] = df_participants["Region"].map(region_to_node)

    df_price_hourly = df_price_hourly.rename(columns={"timestamp": "From", "zone": "Node"})
    df_price_hourly["From"] = pd.to_datetime(df_price_hourly["From"])
    df_price_hourly["Energy_Price_NOK_kWh"] = df_price_hourly["price_eur_mwh"] * eur_to_nok / 1000
    df_price_hourly = df_price_hourly.drop_duplicates(subset=["From", "Node"], keep="first").reset_index(drop=True)

    df_timesteps = pd.DataFrame(
        {"From": pd.date_range(start=min(df_data_hourly["Date"]), end=max(df_data_hourly["Date"]), freq="h")}
    ).reset_index(names="TimeStep")

    df_all = pd.merge(df_timesteps, df_data_hourly, on=["From"], how="inner")
    df_all = pd.merge(df_all, df_participants, on=["ID"], how="inner")
    df_all = pd.merge(df_all, df_price_hourly, on=["From", "Node"], how="left")
    df_all["Energy_Price_NOK_kWh"] = df_all["Energy_Price_NOK_kWh"].interpolate(method="linear")
    df_all["Energy_Experiment_Price_NOK_kWh"] = df_all["Experiment_price_NOK_kWh"].fillna(0) + df_all["Energy_Price_NOK_kWh"]
    return df_all.sort_values(by=["From", "Group"]).reset_index(drop=True)


# iFlex_merged_data.csv (and config "aggregation_columns") use the original column
# names; map them to the names used in this module.
CSV_TO_CODE_COLUMNS = {
    "Price_NOK_kWh": "Energy_Price_NOK_kWh",
    "Power_Price_NOK_kWh": "Energy_Experiment_Price_NOK_kWh",
}
CODE_TO_CSV_COLUMNS = {code: csv for csv, code in CSV_TO_CODE_COLUMNS.items()}


#------------------------------------------------------------------------------------------------
# Load and Aggregate iFlex Data
#------------------------------------------------------------------------------------------------
def load_iflex_aggregate(price_elasticity, grid_charge, eur_to_nok, MERGED_DATA_PATH, RAW_DATA_PATH, AGGREGATION_COLUMNS, MERGED_OUTPUT_PATH=None) -> pd.DataFrame:
    """
    Load participant-hour rows and aggregate to community-hour demand. Reconstruct baseline demand without experimental price or grid charge.
    Parameters
    ----------
    price_elasticity: Price elasticity of demand
    grid_charge: Grid charge paid during the experiment (NOK/kWh)
    eur_to_nok: EUR to NOK exchange rate
    MERGED_DATA_PATH: Path to the merged participant-hour CSV
    RAW_DATA_PATH: Path to the raw iFlex hourly data
    AGGREGATION_COLUMNS: Columns to read from the merged CSV
    MERGED_OUTPUT_PATH: Where to save the merged data if it is rebuilt

    Returns
    -------
    df_agg: One row per community-hour with demand (baseline, kWh), demand_experiment (observed, kWh), prices (NOK/kWh), temperature features and sine/cosine time features
    """
    start = time.time()
    MERGED_DATA_PATH = Path(MERGED_DATA_PATH)
    if MERGED_DATA_PATH.exists():
        raw = pd.read_csv(MERGED_DATA_PATH, usecols=AGGREGATION_COLUMNS, parse_dates=["From", "Date"], low_memory=False)
        raw = raw.rename(columns=CSV_TO_CODE_COLUMNS)
        source = MERGED_DATA_PATH.name
    else:
        raw = _build_merged_data_from_raw(RAW_DATA_PATH, eur_to_nok)[[CSV_TO_CODE_COLUMNS.get(c, c) for c in AGGREGATION_COLUMNS]]
        source = "raw project files"
        if MERGED_OUTPUT_PATH is None:
            MERGED_OUTPUT_PATH = MERGED_DATA_PATH
        if MERGED_OUTPUT_PATH is not None:
            MERGED_OUTPUT_PATH = Path(MERGED_OUTPUT_PATH)
            # Save with the original CSV column names so cached and rebuilt files match.
            raw.rename(columns=CODE_TO_CSV_COLUMNS).to_csv(MERGED_OUTPUT_PATH, index=False)
            print(f"Saved merged data to {MERGED_OUTPUT_PATH.resolve()}")

    # Reconstruct pre-price-response demand. A few raw observations have zero
    # demand, so use a tiny floor before taking logs.
    demand_for_log = raw["Demand_kWh"].clip(lower=1e-9)
    total_price_for_log = (raw["Energy_Experiment_Price_NOK_kWh"] + grid_charge).clip(lower=1e-9)
    energy_price_for_log = raw["Energy_Price_NOK_kWh"].clip(lower=1e-9)
    raw["Demand_kWh_pre_price"] = np.log(demand_for_log) - price_elasticity * np.log(total_price_for_log)
    raw["Demand_kWh_pre_tariff"] = np.exp(price_elasticity * np.log(energy_price_for_log) + raw["Demand_kWh_pre_price"]) 

    df_agg = (
        raw.groupby(["TimeStep", "Date", "Hour", "From", "Group"], observed=True)
        .agg(
            Energy_Price_NOK_kWh=("Energy_Price_NOK_kWh", "mean"),
            Energy_Experiment_Price_NOK_kWh=("Energy_Experiment_Price_NOK_kWh", "mean"),
            demand=("Demand_kWh_pre_tariff", "mean"),
            demand_experiment=("Demand_kWh", "mean"),
            Price_signal=("Price_signal", "first"),
            Experiment_price_NOK_kWh=("Experiment_price_NOK_kWh", "first"),
            temperature=("Temperature", "mean"),
            Temperature24=("Temperature24", "mean"),
        )
        .reset_index()
        .rename(columns={"From": "time"})
        .sort_values(by=["time", "Group"])
        .reset_index(drop=True)
    )

    df_agg["Weekday"] = df_agg["time"].dt.weekday
    df_agg["YearDay"] = df_agg["time"].dt.dayofyear
    df_agg["hour_0"] = df_agg["Hour"] - 1
    df_agg["sin_hour"] = np.sin(2 * np.pi * df_agg["hour_0"] / 24)
    df_agg["cos_hour"] = np.cos(2 * np.pi * df_agg["hour_0"] / 24)
    df_agg["sin_weekday"] = np.sin(2 * np.pi * df_agg["Weekday"] / 7)
    df_agg["cos_weekday"] = np.cos(2 * np.pi * df_agg["Weekday"] / 7)
    df_agg["sin_yearday"] = np.sin(2 * np.pi * df_agg["YearDay"] / 366)
    df_agg["cos_yearday"] = np.cos(2 * np.pi * df_agg["YearDay"] / 366)
    df_agg["experiment_price"] = df_agg["Experiment_price_NOK_kWh"].fillna(0)
    # Temperature24 is a rolling 24-hour average, not a forecast.
    df_agg["temperature_24h_avg"] = df_agg["Temperature24"].fillna(df_agg["temperature"])

    print(f"Loaded {len(df_agg):,} community-hour rows from {source} in {time.time() - start:.1f}s")
    return df_agg


#------------------------------------------------------------------------------------------------
# XGBoost Prediction
#------------------------------------------------------------------------------------------------
def xg_boost_predict(train_df, test_df, feature_cols, model_params=None):
    """
    Fits an XGBoost model on the training rows and predicts demand for the test rows.

    Parameters
    ----------
    train_df: Training rows with feature columns and demand
    test_df: Rows to predict
    feature_cols: Features used by the model
    model_params: XGBoost hyperparameters

    Returns
    -------
    output: Predicted demand (kWh)
    """
    train = train_df.dropna(subset=feature_cols + ["demand"])
    test = test_df.dropna(subset=feature_cols)
    output = pd.DataFrame(index=test_df.index, columns=["prediction"], dtype=float)
    params = dict(
        objective="reg:squarederror", tree_method="hist", n_estimators=200,
        max_depth=3, learning_rate=0.05, min_child_weight=5,
        random_state=42, n_jobs=1,
    )
    params.update(model_params or {})
    model = XGBRegressor(**params)
    model.fit(train[feature_cols], train["demand"])
    output.loc[test.index, "prediction"] = np.maximum(0, model.predict(test[feature_cols])) # nonnegative predictions
    return output


#------------------------------------------------------------------------------------------------
# Prediction Metrics
#------------------------------------------------------------------------------------------------
def prediction_metrics(results: pd.DataFrame) -> dict[str, float]:
    """
    Computes forecast accuracy metrics.

    Parameters
    ----------
    results: Rows with actual and predicted demand (kWh)

    Returns
    -------
    metrics: Dictionary with MAE (kWh), RMSE (kWh), MAPE_% and R2
    """
    scored = results.dropna(subset=["demand", "prediction"])
    if scored.empty:
        return {"MAE": np.nan, "RMSE": np.nan, "MAPE_%": np.nan, "R2": np.nan}
    y_true = scored["demand"].to_numpy(dtype=float)
    y_pred = scored["prediction"].to_numpy(dtype=float)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "MAPE_%": float(np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-6))) * 100),
        "R2": float(r2_score(y_true, y_pred)),
    }


#------------------------------------------------------------------------------------------------
# Evaluate XGBoost
#------------------------------------------------------------------------------------------------
def evaluate_xg_boost(data, test_dates, feature_cols, model_params=None):
    """
    Evaluates XGBoost predictions. For each community and test date, trains on all earlier dates and predicts that date.

    Parameters
    ----------
    data: Residential energy consumption data from load_iflex_aggregate
    test_dates: Dates to predict
    feature_cols: Features used by the model
    model_params: XGBoost hyperparameters

    Returns
    -------
    results: Predictions with columns Group, Date, time, Hour, demand and prediction (kWh)
    metrics: Accuracy metrics for predictions
    """
    rows = []
    for _, community in data.groupby("Group", sort=True):
        community = community.sort_values("time")
        for date in test_dates:
            train = community[community["Date"] < date]
            test = community[community["Date"] == date]
            predictions = xg_boost_predict(train, test, feature_cols, model_params)
            rows.append(test[["Group", "Date", "time", "Hour", "demand"]].join(predictions))
    results = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["Group", "Date", "time", "Hour", "demand", "prediction"]
    )
    return results, prediction_metrics(results)


#------------------------------------------------------------------------------------------------
# Hourly Mean Baseline
#------------------------------------------------------------------------------------------------
def evaluate_hourly_mean_baseline(data: pd.DataFrame, test_dates: list[pd.Timestamp]) -> tuple[pd.DataFrame, dict[str, float]]:
    """
    Baseline forecast using mean demand over earlier dates.

    Parameters
    ----------
    data: Residential energy consumption data from load_iflex_aggregate
    test_dates: Dates to predict

    Returns
    -------
    results: Predictions with columns Group, Date, time, Hour, demand and prediction (kWh)
    metrics: Accuracy metrics over predictions
    """
    rows = []
    for community, community_df in data.groupby("Group", sort=True):
        community_df = community_df.sort_values("time")
        for test_date in test_dates:
            train_df = community_df[community_df["Date"] < test_date]
            test_df = community_df[community_df["Date"] == test_date]
            if train_df.empty or test_df.empty:
                continue
            hourly_mean = train_df.groupby("Hour")["demand"].mean()
            fallback = train_df["demand"].mean()
            scored = test_df[["Group", "Date", "time", "Hour", "demand"]].copy()
            scored["prediction"] = scored["Hour"].map(hourly_mean).fillna(fallback)
            rows.append(scored)

    if not rows:
        empty = pd.DataFrame(columns=["Group", "Date", "time", "Hour", "demand", "prediction"])
        return empty, prediction_metrics(empty)
    results = pd.concat(rows, ignore_index=True)
    return results, prediction_metrics(results)


#------------------------------------------------------------------------------------------------
# Hyperparameter Tuning
#------------------------------------------------------------------------------------------------
def tuning(candidate_feature_sets, candidate_params, data, tuning_dates, tuning_metric="MAE"):
    """
    Grid search over feature sets and XGBoost hyperparameters.

    Parameters
    ----------
    candidate_feature_sets: Mapping from feature set name to list of feature columns
    candidate_params: Mapping from hyperparameter name to list of candidate values
    data: Residential energy consumption data from load_iflex_aggregate
    tuning_dates: Dates used for tuning
    tuning_metric: Metric used to rank combinations ("MAE", "RMSE", "MAPE_%" or "R2")

    Returns
    -------
    results: One row per combination with feature_set, model_params, and metrics, sorted best first
    """
    rows = []
    for name, features in candidate_feature_sets.items():
        for params in ParameterGrid(candidate_params):
            _, metrics = evaluate_xg_boost(data, tuning_dates, features, params)
            rows.append({"feature_set": name, "model_params": json.dumps(params), **metrics})
    results = pd.DataFrame(rows)
    return results.sort_values(tuning_metric, ascending=tuning_metric != "R2").reset_index(drop=True)


#------------------------------------------------------------------------------------------------
# Best Parameters
#------------------------------------------------------------------------------------------------
def best_params(candidate_feature_sets, tuning_results):
    """
    Returns the best feature set and hyperparameters from the tuning results.

    Parameters
    ----------
    candidate_feature_sets: Mapping from feature set name to list of feature columns
    tuning_results: Output of tuning, sorted best first

    Returns
    -------
    name: Name of the best feature set
    features: Feature columns of the best feature set
    params: Best XGBoost hyperparameters
    """
    best = tuning_results.iloc[0]
    return best["feature_set"], candidate_feature_sets[best["feature_set"]], json.loads(best["model_params"])


#------------------------------------------------------------------------------------------------
# Final Results
#------------------------------------------------------------------------------------------------
def final_results(candidate_feature_sets, tuning_results, data, validation_dates, final_test_date):
    """
    Evaluates the XGBoost prediction using the best parameters on validation dates and the final test date, and compares it with the hourly mean baseline.

    Parameters
    ----------
    candidate_feature_sets: Mapping from feature set name to list of feature columns
    tuning_results: Output of tuning, sorted best first
    data: Residential energy consumption data from load_iflex_aggregate
    validation_dates: Dates used for validation
    final_test_date: Final test date

    Returns
    -------
    validation: Predictions on the validation dates
    validation_comparison: Metrics for the baseline and XGBoost predictions on the validation dates
    final: Predictions on the final test date
    final_comparison: Metrics for the baseline and XGBoost predictions on the final test date
    """
    name, features, params = best_params(candidate_feature_sets, tuning_results)
    validation, validation_metrics = evaluate_xg_boost(data, validation_dates, features, params)
    final, final_model_metrics = evaluate_xg_boost(data, [final_test_date], features, params)
    _, validation_baseline = evaluate_hourly_mean_baseline(data, validation_dates)
    _, final_baseline = evaluate_hourly_mean_baseline(data, [final_test_date])
    validation_comparison = pd.DataFrame([
        {"model": "hourly_mean_baseline", **validation_baseline},
        {"model": f"xgboost_{name}", **validation_metrics},
    ])
    scored = final.dropna(subset=["demand", "prediction", "lower", "upper"])
    final_comparison = pd.DataFrame([
        {"model": "hourly_mean_baseline", **final_baseline},
        {"model": f"xgboost_{name}", **final_model_metrics,
         "interval_coverage": scored["demand"].between(scored["lower"], scored["upper"]).mean(),
         "mean_interval_width": (scored["upper"] - scored["lower"]).mean()},
    ])
    return validation, validation_comparison, final, final_comparison

#~~~~~~~~~ Final Holdout Diagnostics ~~~~~~~~~#

#------------------------------------------------------------------------------------------------
# Final Metrics
#------------------------------------------------------------------------------------------------
def final_metrics(final_results: pd.DataFrame) -> pd.DataFrame:
    """
    Computes forecast accuracy metrics for each community.

    Parameters
    ----------
    final_results: Demand predictions (kWh)

    Returns
    -------
    per_group_metrics: One row per community with MAE, RMSE, MAPE_% and R2, sorted by RMSE
    """
    per_group_metrics = []
    for community, group_results in final_results.groupby("Group"):
        group_metrics = prediction_metrics(group_results)
        per_group_metrics.append({"Group": community, **group_metrics})

    per_group_metrics = pd.DataFrame(per_group_metrics).sort_values("RMSE").reset_index(drop=True)
    return per_group_metrics


#------------------------------------------------------------------------------------------------
# Save Outputs
#------------------------------------------------------------------------------------------------
def save_outputs(OUTPUT_PATH: str, 
                 TUNING_OUTPUT_PATH: str, 
                 final_results: pd.DataFrame, 
                 tuning_results: pd.DataFrame
):
    """
    Saves the final predictions and tuning results to CSV.

    Parameters
    ----------
    OUTPUT_PATH: Path of the CSV file for the final predictions
    TUNING_OUTPUT_PATH: Path of the CSV file for the tuning results
    final_kernel_results: Final predictions
    tuning_results: Tuning results

    Returns
    -------
    None
    """
    OUTPUT_PATH = Path(OUTPUT_PATH)
    final_results.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved final holdout predictions to {OUTPUT_PATH.resolve()}")

    OUTPUT_PATH = Path(TUNING_OUTPUT_PATH)
    tuning_results.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved final tuning results to {OUTPUT_PATH.resolve()}")