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
from opvnpv.crops import load_crop_table
from opvnpv.pipeline_spain import run_scan  # nutzt deine bestehende run_scan-Logik


def main():
    # 1) Daten laden
    df_h = load_hourly_balance(HOURLY_BALANCE_PATH, region_target="ALMERIA")
    prices, p_comp = load_prices(PRICE_SPAIN_CLEAN_PATH, price_col="Day-ahead Price (EUR/MWh)")
    df_h = apply_hoy_prices(df_h, prices)
    crop_all = load_crop_table(CROP_PATH)

    # 2) SMOKE: stark reduzieren
    #    - nur 1 Gruppe
    #    - nur 24 Stunden (ein Tag)
    #    - nur ein demand ("low")
    #    - nur C=0
    group_cols = ["config", "region", "region_key", "tilt_deg", "technology", "scenario", "coverage_frac"]
    first_keys = next(iter(df_h.groupby(group_cols, sort=False).groups.keys()))
    g = df_h.groupby(group_cols, sort=False).get_group(first_keys).copy()
    g = g.sort_values("time_local_hour").head(24).copy()

    # wir bauen df_h_smoke so, dass run_scan ganz normal arbeiten kann
    df_h_smoke = g.copy()

    # 3) run_scan mit minimaler Kapazitätsrange erzwingen:
    #    => am einfachsten: temporär CAPACITY_RANGE in params klein machen,
    #       aber ohne Files ändern: wir monkeypatchen hier (nur für smoke).
    import opvnpv.params as params
    params.CAPACITY_RANGE = [0]
    # und wenn du willst: nur low-Demand in run_scan -> wir filtern high-Spalte raus
    # (run_scan greift auf demand_low_kWh / demand_high_kWh zu; wir lassen beides drin)

    df_scan = run_scan(df_h_smoke, crop_all, p_comp)

    # 4) Ausgabe: nur wenige Spalten, damit du sofort siehst ob Logik plausibel ist
    cols = [
        "scheme", "technology", "coverage_frac", "scenario", "demand_level",
        "battery_kWh", "delta_energy_eur_y1", "bill_baseline_eur_y1",
        "bill_after_pv_eur_y1", "op_value_eur_y1", "NPV_eur"
    ]
    print("\n=== SMOKE RESULT (first 24h, 1 group, C=0) ===")
    print(df_scan[cols].sort_values(["scheme", "demand_level"]).to_string(index=False))

    # 5) einfache Plausibilitätschecks
    # op_value sollte (bei deiner neuen Logik) ~ delta_energy_y1 sein
    diff = (df_scan["delta_energy_eur_y1"] - df_scan["op_value_eur_y1"]).abs().max()
    print(f"\nMax |delta_energy_y1 - op_value_y1| = {diff:.6g} (sollte ~0 sein)")


if __name__ == "__main__":
    main()
