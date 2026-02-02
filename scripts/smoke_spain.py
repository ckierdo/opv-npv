# scripts/smoke_spain.py
import numpy as np
import pandas as pd

from opvnpv.paths_local import (
    HOURLY_BALANCE_PATH,
    PRICE_SPAIN_CLEAN_PATH,
    CROP_PATH,
    OUT_DIR,
)

from opvnpv.data_prep import load_hourly_balance, load_prices, apply_hoy_prices
from opvnpv.dispatch import dispatch_no_acogida, dispatch_compsim_cap
from opvnpv.params import P_RETAIL_ES


def main():
    # 1) Daten laden
    df_h = load_hourly_balance(HOURLY_BALANCE_PATH, region_target="ALMERIA")
    prices, p_comp = load_prices(PRICE_SPAIN_CLEAN_PATH, price_col="Day-ahead Price (EUR/MWh)")
    df_h = apply_hoy_prices(df_h, prices)

    # 2) SMOKE: stark reduzieren
    #    - nur 1 Gruppe
    #    - nur 24 Stunden (ein Tag)
    group_cols = ["config", "region", "region_key", "tilt_deg", "technology", "scenario", "coverage_frac"]
    first_keys = next(iter(df_h.groupby(group_cols, sort=False).groups.keys()))
    g = df_h.groupby(group_cols, sort=False).get_group(first_keys).copy()
    df_h_smoke = g.sort_values("time_local_hour").head(24).copy()

    # 3) Quick Dispatch Test: C=0 vs C=10, direkt im LP
    pv = df_h_smoke["pv_total_kWh"].to_numpy()
    p_sell = df_h_smoke["p_da_sell"].to_numpy()
    month_ids = df_h_smoke["month_id"].to_numpy()

    def run_dispatch(scheme: str, load: np.ndarray, C: float):
        if scheme == "ES_NoAcogida":
            energy_operating_value_y1, dispatch = dispatch_no_acogida(
                pv=pv,
                load=load,
                p_sell=p_sell,
                p_retail=P_RETAIL_ES,
                C=C,
            )
            credit_vals = None
        elif scheme == "ES_CompSimProxy_ExactCap":
            energy_operating_value_y1, dispatch, credit_vals = dispatch_compsim_cap(
                pv=pv,
                load=load,
                p_retail=P_RETAIL_ES,
                p_comp=p_comp,
                month_ids=month_ids,
                C=C,
            )
        else:
            raise ValueError(f"Unknown scheme: {scheme}")
        return energy_operating_value_y1, dispatch, credit_vals

    for demand in ["low", "high"]:
        load = df_h_smoke[f"demand_{demand}_kWh"].to_numpy()

        for scheme in ["ES_NoAcogida", "ES_CompSimProxy_ExactCap"]:
            print(f"\n=== {scheme} | demand={demand} | 24h test ===")

            for C in [0, 10]:
                energy_operating_value_y1, dispatch, credit_vals = run_dispatch(scheme, load, C)

                ch_sum = float(np.sum(dispatch["ch"]))
                dis_sum = float(np.sum(dispatch["dis"]))
                exp_sum = float(np.sum(dispatch["exp"]))
                imp_sum = float(np.sum(dispatch["imp"]))

                print(f"\n--- C={C} kWh ---")
                print(f"energy_operating_value_eur_y1: {energy_operating_value_y1:.6f}")
                print(f"sum(ch): {ch_sum:.6f} | sum(dis): {dis_sum:.6f}")
                print(f"sum(exp): {exp_sum:.6f} | sum(imp): {imp_sum:.6f}")

                if credit_vals is not None:
                    total_credit = float(sum(credit_vals.values()))
                    print(f"total_credit_eur: {total_credit:.6f}")

            print("\nInterpretation:")
            print("- Wenn energy_operating_value(C=10) == energy_operating_value(C=0) und ch/dis ~ 0: Batterie bringt in diesem Test keinen Zusatznutzen.")
            print("- Wenn energy_operating_value steigt und ch/dis > 0: Dispatch nutzt Batterie -> dann ist 'best C=0' eher Economics/NPV-getrieben.")


if __name__ == "__main__":
    main()
