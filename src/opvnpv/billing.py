import numpy as np
from .params import P_RETAIL_ES

def baseline_bill(load: np.ndarray) -> float:
    return float(np.sum(load) * P_RETAIL_ES)

def bill_no_acogida(dispatch: dict, p_sell: np.ndarray) -> float:
    import_cost = float(np.sum(dispatch["imp"]) * P_RETAIL_ES)
    export_rev = float(np.sum(dispatch["exp"] * p_sell))
    return import_cost - export_rev

def bill_compsim(dispatch: dict, credit_vals: dict) -> float:
    import_cost = float(np.sum(dispatch["imp"]) * P_RETAIL_ES)
    credit = float(sum(credit_vals.values())) if credit_vals else 0.0
    return import_cost - credit
