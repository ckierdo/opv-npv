# scripts/smoke_one_group_full_year_simple.py
# Minimal: ONE group, both schemes, low/high demand, sweep C, compute NPV, print best C.
# - uses NEW economic dispatch (import cost included)
# - no break-even, no cashflow debug spam

import argparse
import math
import numpy as np
import pandas as pd

from opvnpv.paths_local import HOURLY_BALANCE_PATH, PRICE_SPAIN_CLEAN_PATH, CROP_PATH
from opvnpv.data_prep import load_hourly_balance, load_prices, apply_hoy_prices

# ✅ IMPORTANT: import the NEW dispatch functions
from opvnpv.dispatch_TEST import dispatch_no_acogida_one_stage, dispatch_compsim_cap_one_stage
# if your updated file is opvnpv/dispatch.py then use:
# from opvnpv.dispatch import dispatch_no_acogida_econ, dispatch_compsim_cap_econ

from opvnpv.params import (
    P_RETAIL_ES,
    CAPACITY_RANGE,
    BAT_BASE_CAPEX_EUR_PER_KWH,
    BAT_REDUCTION_PER_4X,
)
from opvnpv.crops import load_crop_table, get_crop_row
from opvnpv.economics import compute_pv_economics, build_cashflows_and_npv


def battery_capex_total_curve(C_kWh: float, base_eur_per_kwh: float, reduction_per_4x: float) -> float:
    if C_kWh <= 0:
        return 0.0
    steps = math.log(max(float(C_kWh), 1.0), 4.0)
    factor = (1.0 - float(reduction_per_4x)) ** steps
    capex_per_kwh = float(base_eur_per_kwh) * factor
    return float(capex_per_kwh * float(C_kWh))


def pick_one_group(df_h: pd.DataFrame, group_cols: list[str], args) -> pd.DataFrame:
    use_index = args.group_index
    if use_index is None and all(
        v is None for v in [args.technology, args.coverage_frac, args.scenario, args.config, args.tilt_deg]
    ):
        use_index = 0
        print("No selection given -> defaulting to --group-index 0")

    if use_index is not None:
        groups = list(df_h.groupby(group_cols, sort=False))
        if use_index < 0 or use_index >= len(groups):
            raise ValueError(f"group_index out of range. Got {use_index}, but there are {len(groups)} groups.")
        keys, g = groups[use_index]
        print("\n=== SELECTED GROUP ===")
        for k, v in zip(group_cols, keys):
            print(f"{k}: {v}")
        return g.sort_values("time_local_hour").copy()

    sel = df_h.copy()
    if args.config is not None:
        sel = sel[sel["config"].astype(str) == str(args.config)]
    if args.technology is not None:
        sel = sel[sel["technology"].astype(str) == str(args.technology)]
    if args.scenario is not None:
        sel = sel[sel["scenario"].astype(str) == str(args.scenario)]
    if args.coverage_frac is not None:
        sel = sel[np.isclose(sel["coverage_frac"].astype(float), float(args.coverage_frac))]
    if args.tilt_deg is not None:
        sel = sel[np.isclose(sel["tilt_deg"].astype(float), float(args.tilt_deg))]

    ng = sel.groupby(group_cols, sort=False).ngroups
    if ng != 1:
        cand = sel[group_cols].drop_duplicates().reset_index(drop=True)
        raise RuntimeError(
            f"Selection did not uniquely identify a single group (ngroups={ng}). Candidates:\n"
            f"{cand.to_string(index=False)}"
        )

    keys_row = sel[group_cols].drop_duplicates().iloc[0]
    print("\n=== SELECTED GROUP ===")
    for k in group_cols:
        print(f"{k}: {keys_row[k]}")
    return sel.sort_values("time_local_hour").copy()


def main():
    ap = argparse.ArgumentParser(description="Smoke: ONE group, new economic dispatch + NPV, print best C.")
    ap.add_argument("--region", default="ALMERIA")

    ap.add_argument("--group-index", type=int, default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--technology", default=None)
    ap.add_argument("--scenario", default=None)
    ap.add_argument("--coverage-frac", type=float, default=None)
    ap.add_argument("--tilt-deg", type=float, default=None)

    ap.add_argument("--max-c", type=int, default=None)
    ap.add_argument("--quiet", action="store_true")

    ap.add_argument("--bat-base-capex-eur-per-kwh", type=float, default=None)
    ap.add_argument("--bat-reduction-per-4x", type=float, default=None)

    args = ap.parse_args()

    # Load base data
    df_h = load_hourly_balance(HOURLY_BALANCE_PATH, region_target=args.region)
    prices, p_comp = load_prices(PRICE_SPAIN_CLEAN_PATH, price_col="Day-ahead Price (EUR/MWh)")
    df_h = apply_hoy_prices(df_h, prices)

    group_cols = ["config", "region", "region_key", "tilt_deg", "technology", "scenario", "coverage_frac"]
    g = pick_one_group(df_h, group_cols, args)

    pv = g["pv_total_kWh"].to_numpy()
    p_sell = g["p_da_sell"].to_numpy()
    month_ids = g["month_id"].to_numpy()

    tech = str(g["technology"].iloc[0])
    cov = float(g["coverage_frac"].iloc[0])
    shading_pct = int(round(cov * 100))

    crop_all = load_crop_table(CROP_PATH)
    crop_row = get_crop_row(crop_all, region_key_target=args.region, shading_pct=shading_pct)
    if crop_row is None:
        raise RuntimeError(f"No crop row for region={args.region} shading={shading_pct}%")

    pv_kWp, pv_capex, pv_opex = compute_pv_economics(tech, cov)

    cap_list = list(CAPACITY_RANGE) if args.max_c is None else list(range(0, int(args.max_c) + 1))

    base_capex = BAT_BASE_CAPEX_EUR_PER_KWH if args.bat_base_capex_eur_per_kwh is None else float(args.bat_base_capex_eur_per_kwh)
    reduction = BAT_REDUCTION_PER_4X if args.bat_reduction_per_4x is None else float(args.bat_reduction_per_4x)

    def bat_capex(C: int) -> float:
        return battery_capex_total_curve(C, base_capex, reduction)

    if not args.quiet:
        print("\n=== RUN SETTINGS (short) ===")
        print(f"p_retail={P_RETAIL_ES:.3f} | p_comp={p_comp:.3f}")
        print(f"bat_base@1kWh={base_capex:.1f} €/kWh | reduction_per_4x={reduction:.3f}")
        print(f"pv_capex={pv_capex:.0f} € | pv_opex={pv_opex:.0f} €/y")

    schemes = ["ES_NoAcogida", "ES_CompSimProxy_ExactCap"]
    crop_scenarios = ["Optimistic", "Conservative", "Pessimistic"]

    rows = []

    for demand in ["low", "high"]:
        load = g[f"demand_{demand}_kWh"].to_numpy()

        for scheme in schemes:
            eov0 = None   # ← einmal pro scheme+demand, VOR for C
            for C in cap_list:
                # -------- dispatch 2 steps) --------
                if scheme == "ES_NoAcogida":
                    eov, disp = dispatch_no_acogida_one_stage(pv=pv, load=load, p_sell=p_sell, p_retail=P_RETAIL_ES, C=C)
                    credit_total = 0.0
                else:
                    eov, disp, credit_vals = dispatch_compsim_cap_one_stage(
                        pv=pv, load=load, p_retail=P_RETAIL_ES, p_comp=p_comp, month_ids=month_ids, C=C
                    )
                    credit_total = float(sum(credit_vals.values()))
        # -------- DEBUG: GENAU HIER --------
                if C == 0:
                    eov0 = eov

                dis_sum = float(np.sum(disp["dis"]))
                imp_sum = float(np.sum(disp["imp"]))
                self_sum = float(np.sum(disp["sc"] + disp["dis"]))
                exp_sum  = float(np.sum(disp["exp"]))
                sc_sum   = float(np.sum(disp["sc"]))
                


                if C in (0, 1, 2, 5, 10, cap_list[-1]):
                    print(
                        f"[{scheme}|{demand}] "
                        f"C={C:2d} "
                        f"eov={eov:8.2f} "
                        f"Δeov={eov - eov0:8.2f} "
                        f"imp={imp_sum:8.1f} "
                        f"dis={dis_sum:8.1f}"
                        f"self={self_sum:8.1f} "
                        f"exp={exp_sum:8.1f} "
                        f"sc={sc_sum:8.1f} "
                        f"credit={credit_total:8.2f}"
                    )

                # -------- NPV --------
                bat_capex_val = bat_capex(C)

                for crop_scen in crop_scenarios:
                    delta_crop_y = float(crop_row[crop_scen] - crop_row["Baseline_Annual_Revenue_EUR"])

                    npv, _cf = build_cashflows_and_npv(
                        tech=tech,
                        energy_operating_value_y1=float(eov),
                        delta_crop_y=delta_crop_y,
                        pv_capex=float(pv_capex),
                        pv_opex=float(pv_opex),
                        bat_capex=float(bat_capex_val),
                    )

                    rows.append({
                        "scheme": scheme,
                        "demand": demand,
                        "crop": crop_scen,
                        "C": int(C),
                        "NPV": float(npv),
                        "eov": float(eov),
                        "bat_capex": float(bat_capex_val),
                        "E_dis": float(np.sum(disp["dis"])),
                        "E_imp": float(np.sum(disp["imp"])),
                        "E_exp": float(np.sum(disp["exp"])),
                        "credit": float(credit_total),
                    })

    df = pd.DataFrame(rows)

    print("\n=== BEST C by NPV  ===")
    for (scheme, demand, crop), sub in df.groupby(["scheme", "demand", "crop"], sort=False):
        best = sub.loc[sub["NPV"].idxmax()]
        print(
            f"{scheme:24s} | demand={demand:4s} | crop={crop:12s} "
            f"=> best_C={int(best['C']):2d} | NPV={best['NPV']:,.2f} € | "
            f"eov={best['eov']:.2f} € | bat_capex={best['bat_capex']:,.0f} € | "
            f"E_dis={best['E_dis']:.1f} kWh | E_imp={best['E_imp']:.1f} kWh |  E_exp={best['E_exp']:.1f} kWh | self={best['eov'] - best['credit']:.2f} € | "f"self+credit={best['eov']:.2f} € (credit={best['credit']:.2f} €)"
            f"crop_delta={delta_crop_y:,.2f} €"
        )


if __name__ == "__main__":
    main()
