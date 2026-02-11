# # scripts/smoke_spain.py
# import numpy as np
# import pandas as pd

# from opvnpv.paths_local import (
#     HOURLY_BALANCE_PATH,
#     PRICE_SPAIN_CLEAN_PATH,
#     CROP_PATH,
#     OUT_DIR,
# )

# from opvnpv.data_prep import load_hourly_balance, load_prices, apply_hoy_prices
# from opvnpv.dispatch import dispatch_no_acogida, dispatch_compsim_cap
# from opvnpv.params import P_RETAIL_ES


# def _log_series(name: str, x: np.ndarray):
#     x = np.asarray(x, dtype=float)
#     print(f"{name}: n={x.size} | sum={x.sum():.6f} | mean={x.mean():.6f} | min={x.min():.6f} | max={x.max():.6f}")


# def _log_head_tail_df(df: pd.DataFrame, cols: list[str], n: int = 6):
#     print("\n[HEAD]")
#     print(df[cols].head(n).to_string(index=False))
#     print("\n[TAIL]")
#     print(df[cols].tail(n).to_string(index=False))


# def main():
#     # 1) Daten laden
#     df_h = load_hourly_balance(HOURLY_BALANCE_PATH, region_target="ALMERIA")
#     prices, p_comp = load_prices(PRICE_SPAIN_CLEAN_PATH, price_col="Day-ahead Price (EUR/MWh)")
#     df_h = apply_hoy_prices(df_h, prices)

#     # 2) SMOKE: stark reduzieren
#     group_cols = ["config", "region", "region_key", "tilt_deg", "technology", "scenario", "coverage_frac"]
#     first_keys = next(iter(df_h.groupby(group_cols, sort=False).groups.keys()))
#     g = df_h.groupby(group_cols, sort=False).get_group(first_keys).copy()
#     df_h_smoke = g.sort_values("time_local_hour").head(24).copy()

#     print("\n=== SMOKE GROUP KEYS ===")
#     for k, v in zip(group_cols, first_keys):
#         print(f"{k}: {v}")

#     # 2a) Log: what exactly is being fed (PV, demand, prices)
#     cols_show = [
#         "time_local_hour",
#         "pv_total_kWh",
#         "demand_low_kWh",
#         "demand_high_kWh",
#         "p_da",
#         "p_da_sell",
#         "month_id",
#     ]
#     print("\n=== INPUT SNAPSHOT (24h) ===")
#     _log_head_tail_df(df_h_smoke, cols_show, n=8)

#     pv = df_h_smoke["pv_total_kWh"].to_numpy()
#     load_low = df_h_smoke["demand_low_kWh"].to_numpy()
#     load_high = df_h_smoke["demand_high_kWh"].to_numpy()
#     p_sell = df_h_smoke["p_da_sell"].to_numpy()

#     print("\n=== INPUT STATS (24h) ===")
#     _log_series("pv_total_kWh", pv)
#     _log_series("demand_low_kWh", load_low)
#     _log_series("demand_high_kWh", load_high)
#     _log_series("p_da_sell_eur_per_kWh", p_sell)
#     print(f"p_retail_eur_per_kWh: {P_RETAIL_ES:.6f} | p_comp_eur_per_kWh: {p_comp:.6f}")

#     # Optional: baseline (no battery) energy value from raw inputs
#     def _baseline_noacogida_value(pv_arr, load_arr, p_sell_arr):
#         sc = np.minimum(pv_arr, load_arr)
#         exp = np.clip(pv_arr - load_arr, 0, None)
#         return float(P_RETAIL_ES * sc.sum() + np.sum(p_sell_arr * exp))

#     print("\n=== BASELINE CHECK (C=0 implied) ===")
#     print(f"baseline_noacogida_value_low_eur:  {_baseline_noacogida_value(pv, load_low, p_sell):.6f}")
#     print(f"baseline_noacogida_value_high_eur: {_baseline_noacogida_value(pv, load_high, p_sell):.6f}")

#     month_ids = df_h_smoke["month_id"].to_numpy()

#     def run_dispatch(scheme: str, load: np.ndarray, C: float):
#         if scheme == "ES_NoAcogida":
#             energy_operating_value_y1, dispatch = dispatch_no_acogida(
#                 pv=pv,
#                 load=load,
#                 p_sell=p_sell,
#                 p_retail=P_RETAIL_ES,
#                 C=C,
#             )
#             credit_vals = None
#         elif scheme == "ES_CompSimProxy_ExactCap":
#             energy_operating_value_y1, dispatch, credit_vals = dispatch_compsim_cap(
#                 pv=pv,
#                 load=load,
#                 p_retail=P_RETAIL_ES,
#                 p_comp=p_comp,
#                 month_ids=month_ids,
#                 C=C,
#             )
#         else:
#             raise ValueError(f"Unknown scheme: {scheme}")
#         return energy_operating_value_y1, dispatch, credit_vals

#     for demand in ["low", "high"]:
#         load = df_h_smoke[f"demand_{demand}_kWh"].to_numpy()

#         print(f"\n=== DEMAND INPUT USED: {demand} ===")
#         _log_series(f"load_{demand}_kWh", load)

#         for scheme in ["ES_NoAcogida", "ES_CompSimProxy_ExactCap"]:
#             print(f"\n=== {scheme} | demand={demand} | 24h test ===")

#             for C in [0, 10]:
#                 energy_operating_value_y1, dispatch, credit_vals = run_dispatch(scheme, load, C)

#                 ch_sum = float(np.sum(dispatch["ch"]))
#                 dis_sum = float(np.sum(dispatch["dis"]))
#                 exp_sum = float(np.sum(dispatch["exp"]))
#                 imp_sum = float(np.sum(dispatch["imp"]))
#                 sc_sum = float(np.sum(dispatch["sc"]))

#                 print(f"\n--- C={C} kWh ---")
#                 print(f"energy_operating_value_eur_y1: {energy_operating_value_y1:.6f}")
#                 print(f"sum(sc): {sc_sum:.6f} | sum(ch): {ch_sum:.6f} | sum(dis): {dis_sum:.6f}")
#                 print(f"sum(exp): {exp_sum:.6f} | sum(imp): {imp_sum:.6f}")

#                 # quick sanity: load balance
#                 bal_err = float(np.max(np.abs(dispatch["sc"] + dispatch["dis"] + dispatch["imp"] - load)))
#                 print(f"max_balance_error_kWh: {bal_err:.12f}")

#                 if credit_vals is not None:
#                     total_credit = float(sum(credit_vals.values()))
#                     print(f"total_credit_eur: {total_credit:.6f}")

           


# if __name__ == "__main__":
#     main()
    # scripts/run_spain_one_group.py
import argparse
import numpy as np
import pandas as pd

from opvnpv.pipeline_spain import (
    load_hourly_balance,
    load_prices,
    apply_hoy_prices,
    run_scan,
    save_outputs,
    log,
)

from opvnpv.crops import load_crop_table


def pick_group(
    df_h: pd.DataFrame,
    group_cols: list[str],
    group_index: int | None,
    config: str | None,
    technology: str | None,
    scenario: str | None,
    coverage_frac: float | None,
    tilt_deg: float | None,
) -> pd.DataFrame:
    """
    Returns df filtered to exactly one group.
    If group_index is provided: pick that group in the groupby order.
    Else: apply provided filters and ensure exactly one group remains.
    """
    if group_index is not None:
        groups = list(df_h.groupby(group_cols, sort=False))
        if group_index < 0 or group_index >= len(groups):
            raise ValueError(f"group_index out of range. Got {group_index}, but there are {len(groups)} groups.")
        keys, g = groups[group_index]
        log(f"Selected group_index={group_index} keys={keys}")
        return g.copy()

    # filter-based selection
    sel = df_h.copy()

    if config is not None:
        sel = sel[sel["config"].astype(str) == str(config)]
    if technology is not None:
        sel = sel[sel["technology"].astype(str) == str(technology)]
    if scenario is not None:
        sel = sel[sel["scenario"].astype(str) == str(scenario)]
    if coverage_frac is not None:
        sel = sel[np.isclose(sel["coverage_frac"].astype(float), float(coverage_frac))]
    if tilt_deg is not None:
        sel = sel[np.isclose(sel["tilt_deg"].astype(float), float(tilt_deg))]

    if sel.empty:
        raise RuntimeError("Selection filters resulted in empty dataframe. No rows matched.")

    ng = sel.groupby(group_cols, sort=False).ngroups
    if ng != 1:
        # show candidate groups (first few)
        cand = (
            sel[group_cols]
            .drop_duplicates()
            .reset_index(drop=True)
        )
        msg = (
            f"Selection filters did not uniquely identify a single group (ngroups={ng}).\n"
            f"Candidates (first 10):\n{cand.head(10).to_string(index=False)}"
        )
        raise RuntimeError(msg)

    keys = sel[group_cols].iloc[0].to_dict()
    log(f"Selected group by filters keys={keys}")
    return sel.copy()


def main():
    parser = argparse.ArgumentParser(description="Smoke test: run full Spain pipeline but only for ONE group.")
    parser.add_argument("--region", default="ALMERIA", help="region_key target (default: ALMERIA)")

    # pick by index OR filters
    parser.add_argument("--group-index", type=int, default=None, help="0-based group index in pipeline group order")
    parser.add_argument("--config", default=None, help="config value to filter")
    parser.add_argument("--technology", default=None, help="technology to filter (e.g., opv145)")
    parser.add_argument("--scenario", default=None, help="scenario to filter (e.g., cov_10)")
    parser.add_argument("--coverage-frac", type=float, default=None, help="coverage_frac to filter (e.g., 0.10)")
    parser.add_argument("--tilt-deg", type=float, default=None, help="tilt_deg to filter (e.g., 7.2)")

    # paths: reuse your opvnpv.paths_local
    args = parser.parse_args()

    from opvnpv.paths_local import (
        HOURLY_BALANCE_PATH,
        PRICE_SPAIN_CLEAN_PATH,
        CROP_PATH,
        OUT_SCAN,
        OUT_BEST_PER_CROP,
    )

    log("START Spain ONE-GROUP full run (smoke test)")

    # 1) Load base data
    df_h = load_hourly_balance(HOURLY_BALANCE_PATH, region_target=args.region)

    # 2) Prices + HOY mapping
    prices, p_comp = load_prices(PRICE_SPAIN_CLEAN_PATH, price_col="Day-ahead Price (EUR/MWh)")
    df_h = apply_hoy_prices(df_h, prices)

    # 3) Crop data
    crop_all = load_crop_table(CROP_PATH)

    # 4) Filter down to ONE group
    # IMPORTANT: must match the grouping used inside run_scan
    group_cols = ["config", "region", "region_key", "tilt_deg", "technology", "scenario", "coverage_frac"]

    df_one = pick_group(
        df_h=df_h,
        group_cols=group_cols,
        group_index=args.group_index,
        config=args.config,
        technology=args.technology,
        scenario=args.scenario,
        coverage_frac=args.coverage_frac,
        tilt_deg=args.tilt_deg,
    )

    # 5) Run full scan, but only for this group
    df_scan = run_scan(df_one, crop_all, p_comp)

    # 6) Save outputs (optional but usually handy)
    # Note: will overwrite the same files. If you want separate names, edit paths_local or copy this script.
    save_outputs(df_scan, OUT_SCAN, OUT_BEST_PER_CROP)

    log("DONE one-group run")


if __name__ == "__main__":
    main()

