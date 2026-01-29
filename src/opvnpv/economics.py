import math
import numpy as np

from .params import BASE_CAPEX_EUR_PER_KWH, REDUCTION_PER_4X

def npv_from_cf(cf: np.ndarray, r: float) -> float:
    years = np.arange(len(cf))
    return float(np.sum(cf / ((1.0 + r) ** years)))

def battery_cost_per_kwh(C: int) -> float:
    if C <= 0:
        return 0.0
    # shrinking CAPEX because of learning curve: -10% per 4x scale
    scale = math.log(C, 4)
    return BASE_CAPEX_EUR_PER_KWH * ((1.0 - REDUCTION_PER_4X) ** scale)

def battery_capex_total(C: int) -> float:
    return battery_cost_per_kwh(C) * C if C > 0 else 0.0
