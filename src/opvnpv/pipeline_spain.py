import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from .paths_local import (
    HOURLY_BALANCE_PATH,
    PRICE_SPAIN_CLEAN_PATH,
    CROP_PATH,
    OUT_SCAN,
    OUT_BEST_PER_CROP,
)

from .params import (
    DISCOUNT_RATE, PROJECT_YEARS,
    CAPACITY_RANGE,
    P_RETAIL_ES,
    EXPORT_FEE_FACTOR_NO_ACOGIDA,
)

from .io_utils import safe_read_csv, region_key
from .dispatch import dispatch_no_acogida, dispatch_compsim_cap
from .economics import battery_capex_total,  compute_pv_economics, build_cashflows_and_npv
from .crops import load_crop_table, get_crop_row



START_TIME = time.time()


def log(msg: str):
    elapsed = time.time() - START_TIME
    h = int(elapsed // 3600)
    m = int((elapsed % 3600) // 60)
    s = int(elapsed % 60)
    print(f"[{h:02d}:{m:02d}:{s:02d}] {msg}", flush=True)

def log_series(name: str, x: np.ndarray):
    x = np.asarray(x, dtype=float)
    log(f"{name}: n={x.size} | sum={x.sum():.6f} | min={x.min():.6f} | max={x.max():.6f}")
# 1) Ersetze log_input_snapshot durch diese Version (oder füge sie zusätzlich ein)

def log_input_snapshot_random(g: pd.DataFrame, n: int = 12, seed: int | None = None):
    cols = [
        "time_local_hour",
        "pv_total_kWh",
        "demand_low_kWh",
        "demand_high_kWh",
        "p_da",
        "p_da_sell",
        "month_id",
    ]
    gg = g.sort_values("time_local_hour").copy()
    sample = gg[cols].sample(n=min(n, len(gg)), random_state=seed).sort_values("time_local_hour")
    log(f"INPUT SNAPSHOT (random {len(sample)} rows, sorted)")
    print(sample.to_string(index=False), flush=True)


# ----------------------------
# 1) Load hourly balance and filter region
# ----------------------------
def load_hourly_balance(path: str, region_target: str = "ALMERIA") -> pd.DataFrame:
    df_h = safe_read_csv(path)
    df_h["time_local_hour"] = pd.to_datetime(df_h["time_local_hour"])
    df_h["region_key"] = df_h["region"].apply(region_key)

    df_h = df_h[df_h["region_key"] == region_target].copy()
    if df_h.empty:
        raise RuntimeError(f"No rows found for region_key == {region_target} in hourly balance CSV.")

    log(f"Loaded hourly rows (Spain): {len(df_h):,}")
    log(f"Time span: {df_h['time_local_hour'].min()} → {df_h['time_local_hour'].max()}")
    return df_h


# ----------------------------
# 2) Load day-ahead prices (8760) and compute p_comp
# ----------------------------
def load_prices(path: str, price_col: str) -> tuple[pd.DataFrame, float]:
    prices = safe_read_csv(path)
    prices["time_local_hour"] = pd.to_datetime(prices["time_local_hour"])

    if price_col not in prices.columns:
        raise RuntimeError(f"Price column '{price_col}' not found in price CSV.")

    prices["p_da"] = prices[price_col] / 1000.0  # €/kWh
    prices = prices.sort_values("time_local_hour").copy()

    if len(prices) != 8760:
        raise RuntimeError(f"Price file has {len(prices)} rows, expected 8760 (non-leap year hourly).")

    prices["hoy"] = np.arange(len(prices))
    p_comp = float(prices["p_da"].mean())

    log(f"Price rows: {len(prices):,}")
    log(f"p_da NaNs: {int(prices['p_da'].isna().sum())} | p_da mean: {prices['p_da'].mean()}")
    log(f"CompSim proxy price (mean DA): {p_comp:.6f} €/kWh")
    return prices, p_comp


# ----------------------------
# 3) Apply HOY mapping: broadcast same 8760h price curve to each group
# ----------------------------
def apply_hoy_prices(df_h: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    group_cols_hoy = ["config", "technology", "coverage_frac", "tilt_deg", "scenario"]
    df_h = df_h.sort_values(group_cols_hoy + ["time_local_hour"]).copy()

    # HOY per group: 0..8759
    df_h["hoy"] = df_h.groupby(group_cols_hoy).cumcount()

    df_h = df_h.merge(prices[["hoy", "p_da"]], on="hoy", how="left")
    missing = int(df_h["p_da"].isna().sum())
    log(f"Missing p_da after HOY merge: {missing}")
    if missing:
        raise RuntimeError("HOY merge produced NaNs in p_da. Check HOY construction and file lengths.")

    # month id for monthly cap
    df_h["month_id"] = df_h["time_local_hour"].dt.to_period("M").astype(str)

    # NoAcogida export price: DA * factor
    df_h["p_da_sell"] = (df_h["p_da"] * EXPORT_FEE_FACTOR_NO_ACOGIDA).clip(lower=0)
    return df_h


# ----------------------------
# 4) Run scan: objective-only NPV for each group / scheme / battery capacity
# ----------------------------
def run_scan(df_h: pd.DataFrame, crop_all: pd.DataFrame, p_comp: float) -> pd.DataFrame:
    group_cols = ["config", "region", "region_key", "tilt_deg", "technology", "scenario", "coverage_frac"]
    schemes = ["ES_NoAcogida", "ES_CompSimProxy_ExactCap"]
    crop_scenarios = ["Optimistic", "Conservative", "Pessimistic"]

    scan_rows = []
    total_groups = df_h.groupby(group_cols, sort=False).ngroups
    log(f"Total groups to process: {total_groups}")

    for gi, (keys, g) in enumerate(df_h.groupby(group_cols, sort=False)):
        g = g.copy()
        tech = g["technology"].iloc[0]
        cov = float(g["coverage_frac"].iloc[0])
        tilt = float(g["tilt_deg"].iloc[0])
        scen = str(g["scenario"].iloc[0])

        log(f"Group {gi+1}/{total_groups} | Tech={tech} | Cov={cov:.2f} | Tilt={tilt} | Scenario={scen}")
        # 2) Aufrufstelle in run_scan: statt log_input_snapshot(g, n=6) z.B. so:

        log_input_snapshot_random(g, n=12, seed=gi)   # seed=gi -> reproduzierbar pro Gruppe

        pv = g["pv_total_kWh"].to_numpy()
        p_sell = g["p_da_sell"].to_numpy()
        month_ids = g["month_id"].to_numpy()

        log("GROUP INPUTS (full year)")
        log_series("pv_total_kWh", pv)
        log_series("p_da_sell_eur_per_kWh", p_sell)

        # PV economics
        pv_kWp, pv_capex, pv_opex = compute_pv_economics(tech, cov)

        # Crop row for this shading
        shading_pct = int(round(cov * 100))
        crop_row = get_crop_row(crop_all, region_key_target="ALMERIA", shading_pct=shading_pct)
        if crop_row is None:
            log(f"WARNING: No crop data for ALMERIA shading={shading_pct}%. Skipping group.")
            continue

        for demand in ["low", "high"]:
            load = g[f"demand_{demand}_kWh"].to_numpy()
     # === 3) HIER: direkt NACH load = ... ===
            log(f"DEMAND INPUT USED: {demand}")
            log_series(f"demand_{demand}_kWh", load)
            # === 4) HIER: direkt NACH dem demand-log und VOR dem scheme-loop ===
            sc0 = np.minimum(pv, load)
            exp0 = np.clip(pv - load, 0, None)
            imp0 = np.clip(load - pv, 0, None)
            baseline_value = float(P_RETAIL_ES * sc0.sum() + np.sum(p_sell * exp0))
            log("BASELINE (no battery, greedy)")
            log(f"E_sc0_kWh: {sc0.sum():.6f} | E_exp0_kWh: {exp0.sum():.6f} | E_imp0_kWh: {imp0.sum():.6f}")
            log(f"baseline_energy_value_eur: {baseline_value:.6f}")

            for scheme in schemes:
                log(f"  Start: Scheme={scheme} | Demand={demand}")

                for C in CAPACITY_RANGE:
                    bat_capex = battery_capex_total(C)

                    # ----- DISPATCH: objective-only -----
                    _credit_vals = None
                    if scheme == "ES_NoAcogida":
                        energy_operating_value_y1, _dispatch = dispatch_no_acogida(
                            pv=pv,
                            load=load,
                            p_sell=p_sell,
                            p_retail=P_RETAIL_ES,
                            C=C,
                        )
                    elif scheme == "ES_CompSimProxy_ExactCap":
                        energy_operating_value_y1, _dispatch, _credit_vals = dispatch_compsim_cap(
                            pv=pv,
                            load=load,
                            p_retail=P_RETAIL_ES,
                            p_comp=p_comp,
                            month_ids=month_ids,
                            C=C,
                        )
                    else:
                        raise ValueError(f"Unknown scheme: {scheme}")

 # === 5) HIER: direkt NACH dem Dispatch-Aufruf (also hier, vor crop_scen loop) ===
                    sc_sum  = float(np.sum(_dispatch["sc"]))
                    ch_sum  = float(np.sum(_dispatch["ch"]))
                    dis_sum = float(np.sum(_dispatch["dis"]))
                    exp_sum = float(np.sum(_dispatch["exp"]))
                    imp_sum = float(np.sum(_dispatch["imp"]))
                    bal_err = float(np.max(np.abs(_dispatch["sc"] + _dispatch["dis"] + _dispatch["imp"] - load)))

                    log("DISPATCH RESULT")
                    log(f"scheme={scheme} | demand={demand} | C={C}")
                    log(f"energy_operating_value_eur_y1: {energy_operating_value_y1:.6f}")
                    log(f"E_sc_kWh: {sc_sum:.6f} | E_ch_kWh: {ch_sum:.6f} | E_dis_kWh: {dis_sum:.6f}")
                    log(f"E_exp_kWh: {exp_sum:.6f} | E_imp_kWh: {imp_sum:.6f}")
                    log(f"max_balance_error_kWh: {bal_err:.12e}")
                    # === 6) HIER: direkt NACH dem Dispatch-Result-Log, NUR falls credit vorhanden ===
                    if _credit_vals is not None:
                        total_credit = float(sum(_credit_vals.values()))
                        log(f"total_credit_eur: {total_credit:.6f}")
                        
                    for crop_scen in crop_scenarios:
                        # crop_row[crop_scen] is absolute annual revenue under shading scenario
                        # Baseline_Annual_Revenue_EUR is absolute baseline (no shading)
                        delta_crop_y = float(crop_row[crop_scen] - crop_row["Baseline_Annual_Revenue_EUR"])

                        npv, cf = build_cashflows_and_npv(
                            tech=tech,
                            energy_operating_value_y1=energy_operating_value_y1,
                            delta_crop_y=delta_crop_y,
                            pv_capex=pv_capex,
                            pv_opex=pv_opex,
                            bat_capex=bat_capex,
                        )

                        scan_rows.append({
                            "scheme": scheme,
                            "region": g["region"].iloc[0],
                            "technology": tech,
                            "coverage_frac": cov,
                            "tilt_deg": tilt,
                            "scenario": scen,
                            "demand_level": demand,
                            "crop_scenario": crop_scen,
                            "battery_kWh": C,

                            "NPV_eur": npv,
                            "cashflows": cf,    
                            "delta_crop_eur_y": delta_crop_y,       
                            "energy_op_value_eur_y1": energy_operating_value_y1,
                            "pv_kWp": pv_kWp,
                            "pv_capex_eur": pv_capex,
                            "bat_capex_eur": bat_capex,

                            "p_retail_eur_per_kwh": P_RETAIL_ES,
                            "p_comp_eur_per_kwh": p_comp,
                            "export_fee_factor": EXPORT_FEE_FACTOR_NO_ACOGIDA,
                        })

                log(f"  Finished: Scheme={scheme} | Demand={demand}")

    df_scan = pd.DataFrame(scan_rows)
    if df_scan.empty:
        raise RuntimeError("df_scan is empty – no results generated.")
    return df_scan


def save_outputs(df_scan: pd.DataFrame, out_scan: str, out_best: str):
    Path(os.path.dirname(out_scan)).mkdir(parents=True, exist_ok=True)

    df_scan.to_csv(out_scan, sep=";", index=False, encoding="utf-8-sig")

    best_group_cols = ["scheme", "region", "technology", "coverage_frac", "demand_level", "crop_scenario"]
    best_per_crop = (
        df_scan.sort_values("NPV_eur")
              .groupby(best_group_cols, as_index=False)
              .tail(1)
              .reset_index(drop=True)
    )
    best_per_crop.to_csv(out_best, sep=";", index=False, encoding="utf-8-sig")

    log("DONE")
    log(f"Saved full scan: {out_scan}")
    log(f"Saved best per crop: {out_best}")
    print(best_per_crop.head(10))


def main():
    # Keep local paths ONLY in paths_local.py (ignored via .gitignore)
    from .paths_local import (
        HOURLY_BALANCE_PATH,
        PRICE_SPAIN_CLEAN_PATH,
        CROP_PATH,
        OUT_DIR,
        OUT_SCAN,
        OUT_BEST_PER_CROP,
    )

    # 🔍 DEBUG: welche Pfade sind wirklich aktiv?
    log(f"OUT_SCAN = {OUT_SCAN}")
    log(f"OUT_BEST_PER_CROP = {OUT_BEST_PER_CROP}")

    log("START Spain battery NPV optimization (objective-only)")


    df_h = load_hourly_balance(HOURLY_BALANCE_PATH, region_target="ALMERIA")
    prices, p_comp = load_prices(PRICE_SPAIN_CLEAN_PATH, price_col="Day-ahead Price (EUR/MWh)")
    df_h = apply_hoy_prices(df_h, prices)

    crop_all = load_crop_table(CROP_PATH)

    df_scan = run_scan(df_h, crop_all, p_comp)
    save_outputs(df_scan, OUT_SCAN, OUT_BEST_PER_CROP)

