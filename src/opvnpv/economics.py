# economics.py: functions for computing economics (CAPEX, OPEX, cashflows, NPV) for PV and battery.
import numpy as np

from .params import (
    # project
    DISCOUNT_RATE, PROJECT_YEARS,

    # pv econ
    AREA_M2, MODULE_W_M2,
    PV_OPEX_FRAC,
    CAPEX_CSI_EUR_PER_KWP, CAPEX_OPV_EUR_PER_KWP,

    # degradation
    DEG,

    # battery econ
    BAT_OPEX_FRAC, BAT_REPL_YEAR, BAT_REPL_FACTOR,
    BAT_BASE_CAPEX_EUR_PER_KWH, BAT_REDUCTION_PER_4X,
)


# ----------------------------
# Generic NPV
# ----------------------------
def npv_from_cf(cf: np.ndarray, r: float) -> float:
    """Net present value for a cashflow array cf[0..T]."""
    return float(np.sum(cf / (1.0 + r) ** np.arange(len(cf))))


# ----------------------------
# PV economics
# ----------------------------
def compute_pv_economics(tech: str, cov: float):
    """
    Returns: pv_kWp, pv_capex_eur, pv_opex_eur_per_year
    """
    pv_kWp = MODULE_W_M2[tech] * AREA_M2 * cov / 1000.0
    if tech == "polySi":
        pv_capex = CAPEX_CSI_EUR_PER_KWP * pv_kWp
    else:
        pv_capex = CAPEX_OPV_EUR_PER_KWP[tech] * pv_kWp

    pv_opex = pv_capex * PV_OPEX_FRAC
    return pv_kWp, pv_capex, pv_opex


# ----------------------------
# Battery economics
# ----------------------------
def battery_capex_total(C_kWh: float) -> float:
    """
    Simple CAPEX curve:
    - base CAPEX per kWh at 1x
    - each 4x increase reduces €/kWh by BAT_REDUCTION_PER_4X
    """
    if C_kWh <= 0:
        return 0.0

    # how many "4x steps" relative to 1 kWh
    steps = np.log(max(C_kWh, 1.0)) / np.log(4.0)
    factor = (1.0 - BAT_REDUCTION_PER_4X) ** steps
    capex_per_kwh = BAT_BASE_CAPEX_EUR_PER_KWH * factor
    return float(capex_per_kwh * C_kWh)


# ----------------------------
# Cashflows (delta-only logic)
# ----------------------------
def build_cashflows_and_npv(
    tech: str,
    energy_operating_value_y1: float,
    delta_crop_y: float,
    pv_capex: float,
    pv_opex: float,
    bat_capex: float,
):
    """
    Builds cashflows for the "incremental only" model:

    Year 0: -(PV CAPEX + Battery CAPEX)
    Year y>=1:
      + degraded(energy_operating_value_y1)
      + delta_crop_y
      - pv_opex
      - bat_opex

    Battery replacement (if enabled):
      in BAT_REPL_YEAR: subtract BAT_REPL_FACTOR * bat_capex
    """
    bat_opex = bat_capex * BAT_OPEX_FRAC

    cf = np.zeros(PROJECT_YEARS + 1, dtype=float)
    cf[0] = -(pv_capex + bat_capex)

    for year in range(1, PROJECT_YEARS + 1):
        degraded_energy = energy_operating_value_y1 * ((1.0 - DEG[tech]) ** (year - 1))
        cf[year] = degraded_energy + delta_crop_y - pv_opex - bat_opex

    if BAT_REPL_YEAR <= PROJECT_YEARS and bat_capex > 0:
        cf[BAT_REPL_YEAR] -= bat_capex * BAT_REPL_FACTOR

    return npv_from_cf(cf, DISCOUNT_RATE), cf
