import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .params import DEBUG, CAPACITY_RANGE, P_RETAIL_ES, EXPORT_FEE_FACTOR_NO_ACOGIDA
from .economics import battery_capex_total

from .data_prep import load_hourly_balance, load_prices, apply_hoy_prices
from .pv_econ import compute_pv_economics
from .crops import load_crop_table, get_crop_row
from .billing import baseline_bill, bill_no_acogida, bill_compsim
from .cashflows import build_cashflows_and_npv
from .dispatch import dispatch_no_acogida, delta_energy_y1_no_acogida, dispatch_compsim_cap
try:
    from .paths_local import (
        HOURLY_BALANCE_PATH,
        PRICE_SPAIN_CLEAN_PATH,
        CROP_PATH,
        OUT_DIR,
        OUT_SCAN,
        OUT_BEST_PER_CROP,
    )
except ImportError:
    from .paths_example import (
        HOURLY_BALANCE_PATH,
        PRICE_SPAIN_CLEAN_PATH,
        CROP_PATH,
        OUT_DIR,
        OUT_SCAN,
        OUT_BEST_PER_CROP,
    )


START_TIME = time.time()

def log(msg: str):
    elapsed = time.time() - START_TIME
    h = int(elapsed // 3600); m = int((elapsed % 3600) // 60); s = int(elapsed % 60)
    print(f"[{h:02d}:{m:02d}:{s:02d}] {msg}", flush=True)

def dispatch_and_bill_y1(scheme, pv, load, p_sell, month_ids, p_comp, C):
    if scheme == "ES_NoAcogida":
        dispatch = dispatch_no_acogida(pv, load, p_sell, P_RETAIL_ES, C)
        op_y1 = delta_energy_y1_no_acogida(dispatch, p_sell, P_RETAIL_ES)
        bill_y1 = bill_no_acogida(dispatch, p_sell)
        return op_y1, bill_y1

    if scheme == "ES_CompSimProxy_ExactCap":
        op_y1, dispatch, credit_vals = dispatch_compsim_cap(pv, load, P_RETAIL_ES, p_comp, month_ids, C)
        bill_y1 = bill_compsim(dispatch, credit_vals)
        return op_y1, bill_y1

    raise ValueError(scheme)

def run_scan(df_h: pd.DataFrame, crop_all: pd.DataFrame, p_comp: float) -> pd.DataFrame:
    group_cols = ["config", "region", "region_key", "tilt_deg", "technology", "scenario", "coverage_frac"]
    schemes = ["ES_NoAcogida", "ES_CompSimProxy_ExactCap"]
    crop_scenarios = ["Optimistic", "Conservative", "Pessimistic"]

    rows = []
    total_groups = df_h.groupby(group_cols, sort=False).ngroups
    log(f"Total groups to process: {total_groups}")

    for gi, (_, g) in enumerate(df_h.groupby(group_cols, sort=False)):
        tech = g["technology"].iloc[0]
        cov = float(g["coverage_frac"].iloc[0])
        tilt = float(g["tilt_deg"].iloc[0])
        scen = str(g["scenario"].iloc[0])
        log(f"Group {gi+1}/{total_groups} | Tech={tech} | Cov={cov:.2f} | Tilt={tilt} | Scenario={scen}")

        pv = g["pv_total_kWh"].to_numpy()
        p_sell = g["p_da_sell"].to_numpy()
        month_ids = g["month_id"].to_numpy()

        pv_kWp, pv_capex, pv_opex = compute_pv_economics(tech, cov)

        shading_pct = int(round(cov * 100))
        crop_row = get_crop_row(crop_all, region_key_target="ALMERIA", shading_pct=shading_pct)
        if crop_row is None:
            log(f"WARNING: no crop data for shading={shading_pct}%. skip.")
            continue

        for demand in ["low", "high"]:
            load = g[f"demand_{demand}_kWh"].to_numpy()
            bill_base = baseline_bill(load)

            for scheme in schemes:
                log(f"  Start: Scheme={scheme} | Demand={demand}")
                for C in CAPACITY_RANGE:
                    bat_capex = battery_capex_total(C)

                    op_y1, bill_y1 = dispatch_and_bill_y1(scheme, pv, load, p_sell, month_ids, p_comp, C)
                    delta_energy_y1 = bill_base - bill_y1

                    if DEBUG and abs(op_y1 - delta_energy_y1) > 1e-3:
                        log(f"WARNING: op_y1 != bill_base-bill_y1 (diff={op_y1 - delta_energy_y1:.6f})")

                    for crop_scen in crop_scenarios:
                        delta_crop_y = float(crop_row[crop_scen] - crop_row["Baseline_Annual_Revenue_EUR"])
                        npv, _ = build_cashflows_and_npv(
                            tech=tech,
                            delta_energy_y1=delta_energy_y1,
                            delta_crop_y=delta_crop_y,
                            pv_capex=pv_capex,
                            pv_opex=pv_opex,
                            bat_capex=bat_capex,
                        )

                        rows.append({
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
                            "delta_energy_eur_y1": delta_energy_y1,
                            "delta_crop_eur_y": delta_crop_y,
                            "pv_kWp": pv_kWp,
                            "pv_capex_eur": pv_capex,
                            "bat_capex_eur": bat_capex,
                            "p_retail_eur_per_kwh": P_RETAIL_ES,
                            "p_comp_eur_per_kwh": p_comp,
                            "export_fee_factor": EXPORT_FEE_FACTOR_NO_ACOGIDA,
                            "bill_baseline_eur_y1": bill_base,
                            "bill_after_pv_eur_y1": bill_y1,
                            "op_value_eur_y1": op_y1,
                        })
                log(f"  Finished: Scheme={scheme} | Demand={demand}")

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No results generated.")
    return df

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

    log(f"Saved full scan: {out_scan}")
    log(f"Saved best per crop: {out_best}")
    print(best_per_crop.head(10))

def main():
    HOURLY_BALANCE_PATH = r"C:\Kierdorf\Python scripts\2nd OPV Paper\Top 10 Results RESkit\hourly_balance_1000m2_10_25_40perc_alltech.csv"
    PRICE_SPAIN_CLEAN_PATH = r"C:\Kierdorf\Python scripts\2nd OPV Paper\Energy Price Data\GUI_ENERGY_PRICES_2023_ES_clean.csv"
    CROP_PATH = r"C:\Kierdorf\crops\annual_revenue_scenarios_by_region.csv"

    OUT_DIR = r"C:\Kierdorf\Python scripts\2nd OPV Paper\Top 10 Results RESkit"
    OUT_SCAN = os.path.join(OUT_DIR, "ES_battery_scan_maxNPV_full_vers1.csv")
    OUT_BEST = os.path.join(OUT_DIR, "ES_battery_scan_maxNPV_best_per_crop_vers1.csv")

    log("START Spain battery NPV optimization")

    df_h = load_hourly_balance(HOURLY_BALANCE_PATH, region_target="ALMERIA")
    prices, p_comp = load_prices(PRICE_SPAIN_CLEAN_PATH, price_col="Day-ahead Price (EUR/MWh)")
    df_h = apply_hoy_prices(df_h, prices)

    crop_all = load_crop_table(CROP_PATH)

    df_scan = run_scan(df_h, crop_all, p_comp)
    save_outputs(df_scan, OUT_SCAN, OUT_BEST)

if __name__ == "__main__":
    main()
