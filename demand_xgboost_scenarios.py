# Import libraries
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#------------------------------------------------------------------------------------------------
# Calibration Residuals
#------------------------------------------------------------------------------------------------
def make_calibration_residuals(validation_results: pd.DataFrame) -> pd.DataFrame:
    """
    Compute validation residuals used to generate scenarios.

    Parameters
    ----------
    validation_results: Demand predictions on the validation dates (kWh)

    Returns
    -------
    calibration: Residuals (demand - prediction) and absolute value of residuals (kWh)
    """
    calibration = validation_results.dropna(subset=["demand", "prediction"]).copy()
    calibration["residual"] = calibration["demand"] - calibration["prediction"]
    calibration["absolute_residual"] = calibration["residual"].abs()
    calibration["Date"] = pd.to_datetime(calibration["Date"])
    calibration["time"] = pd.to_datetime(calibration["time"])
    return calibration


#------------------------------------------------------------------------------------------------
# Residual Block Scenarios
#------------------------------------------------------------------------------------------------
def make_residual_block_scenarios(
    forecast: pd.DataFrame,
    calibration: pd.DataFrame,
    center_by_group_hour: bool = False,
) -> pd.DataFrame:
    """
    Generate demand scenarios by adding validation residuals to the demand prediction.

    Parameters
    ----------
    forecast: Demand prediction on the test date (kWh)
    calibration: Validation residuals from make_calibration_residuals
    center_by_group_hour: If True, subtract the mean residual for each community-hour first

    Returns
    -------
    scenarios: Demand scenarios
    """
    residuals = calibration.copy()
    if center_by_group_hour:
        residuals["residual"] = residuals["residual"] - residuals.groupby(["Group", "Hour"], observed=True)["residual"].transform("mean")

    block_dates = sorted(residuals["Date"].drop_duplicates())
    if not block_dates:
        return pd.DataFrame()

    scenarios = []
    for scenario_number, block_date in enumerate(block_dates, start=1):
        block = residuals.loc[
            residuals["Date"] == block_date,
            ["Group", "Hour", "residual"],
        ].rename(columns={"residual": "block_residual"})

        scenario = forecast.merge(block, on=["Group", "Hour"], how="left")
        scenario["block_residual"] = scenario["block_residual"].fillna(0.0)
        scenario["scenario_demand"] = np.maximum(0.0, scenario["prediction"] + scenario["block_residual"])
        scenario["scenario_id"] = f"block_{scenario_number:02d}_{pd.Timestamp(block_date).date()}"
        scenario["scenario_type"] = "validation_residual_block"
        scenario["block_date"] = pd.Timestamp(block_date)
        scenario["scenario_weight"] = 1.0 / len(block_dates)
        scenarios.append(scenario)

    return pd.concat(scenarios, ignore_index=True)
