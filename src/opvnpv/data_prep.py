import numpy as np
import pandas as pd
from .io_utils import safe_read_csv, region_key
from .params import EXPORT_FEE_FACTOR_NO_ACOGIDA

def load_hourly_balance(path: str, region_target: str = "ALMERIA") -> pd.DataFrame:
    df = safe_read_csv(path)
    df["time_local_hour"] = pd.to_datetime(df["time_local_hour"])
    df["region_key"] = df["region"].apply(region_key)

    df = df[df["region_key"] == region_target].copy()
    if df.empty:
        raise RuntimeError(f"No rows found for region_key == {region_target}")
    return df

def load_prices(path: str, price_col: str) -> tuple[pd.DataFrame, float]:
    prices = safe_read_csv(path)
    prices["time_local_hour"] = pd.to_datetime(prices["time_local_hour"])
    if price_col not in prices.columns:
        raise RuntimeError(f"Price column '{price_col}' not found.")

    prices = prices.sort_values("time_local_hour").copy()
    prices["p_da"] = prices[price_col] / 1000.0  # €/kWh

    if len(prices) != 8760:
        raise RuntimeError(f"Expected 8760 rows, got {len(prices)}.")

    prices["hoy"] = np.arange(len(prices))
    return prices, float(prices["p_da"].mean())

def apply_hoy_prices(df_h: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["config", "technology", "coverage_frac", "tilt_deg", "scenario"]
    df_h = df_h.sort_values(group_cols + ["time_local_hour"]).copy()
    df_h["hoy"] = df_h.groupby(group_cols).cumcount()

    df_h = df_h.merge(prices[["hoy", "p_da"]], on="hoy", how="left")
    if df_h["p_da"].isna().any():
        raise RuntimeError("HOY merge produced NaNs in p_da.")

    df_h["month_id"] = df_h["time_local_hour"].dt.to_period("M").astype(str)
    df_h["p_da_sell"] = (df_h["p_da"] * EXPORT_FEE_FACTOR_NO_ACOGIDA).clip(lower=0)
    return df_h
