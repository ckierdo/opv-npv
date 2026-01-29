import numpy as np
from .params import PROJECT_YEARS, DISCOUNT_RATE, BAT_OPEX_FRAC, BAT_REPL_YEAR, BAT_REPL_FACTOR, DEG
from .economics import npv_from_cf

def build_cashflows_and_npv(
    tech: str,
    delta_energy_y1: float,
    delta_crop_y: float,
    pv_capex: float,
    pv_opex: float,
    bat_capex: float,
):
    bat_opex = bat_capex * BAT_OPEX_FRAC

    cf = np.zeros(PROJECT_YEARS + 1, dtype=float)
    cf[0] = -(pv_capex + bat_capex)

    for year in range(1, PROJECT_YEARS + 1):
        degraded_energy = delta_energy_y1 * ((1 - DEG[tech]) ** (year - 1))
        cf[year] = degraded_energy + delta_crop_y - pv_opex - bat_opex

    if BAT_REPL_YEAR <= PROJECT_YEARS and bat_capex > 0:
        cf[BAT_REPL_YEAR] -= bat_capex * BAT_REPL_FACTOR

    return npv_from_cf(cf, DISCOUNT_RATE), cf
