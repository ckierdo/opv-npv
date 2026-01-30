import numpy as np
import pandas as pd
import pulp as pl

from .params import ETA_CH, ETA_DIS, SOLVER_MSG, DEBUG


def _solve_or_raise(model: pl.LpProblem) -> None:
    """Solve LP and raise if not optimal."""
    model.solve(pl.PULP_CBC_CMD(msg=SOLVER_MSG))
    status = pl.LpStatus[model.status]
    if status != "Optimal":
        raise RuntimeError(f"Solver failed with status: {status}")


# ============================================================
# DISPATCH SOLVER: NO ACOGIDA (hourly DA export), battery-to-grid excluded
# ============================================================
def dispatch_no_acogida(pv, load, p_sell, p_retail, C):
    """
    Optimal hourly dispatch under "No Acogida".

    Objective (incremental energy benefit, Year 1, EUR):
      op_value = Σ_t [ p_retail*(sc_t + dis_t) + p_sell_t*exp_t ]

    Notes:
    - We do NOT subtract import cost here (baseline costs are treated as external).
    - Monthly caps do not apply in this scheme.
    """
    T = len(pv)
    idx = range(T)

    model = pl.LpProblem("ES_NoAcogida", pl.LpMaximize)

    sc  = pl.LpVariable.dicts("sc",  idx, lowBound=0)  # PV -> load
    ch  = pl.LpVariable.dicts("ch",  idx, lowBound=0)  # PV -> battery
    dis = pl.LpVariable.dicts("dis", idx, lowBound=0)  # battery -> load
    exp = pl.LpVariable.dicts("exp", idx, lowBound=0)  # PV -> grid
    imp = pl.LpVariable.dicts("imp", idx, lowBound=0)  # grid -> load (balance var)
    soc = pl.LpVariable.dicts("soc", idx, lowBound=0, upBound=C)

    # Objective: incremental benefit (EUR)
    model += pl.lpSum(
        p_retail * (sc[t] + dis[t]) + p_sell[t] * exp[t]
        for t in idx
    )

    for t in idx:
        model += sc[t] + ch[t] + exp[t] <= pv[t]
        model += sc[t] + dis[t] + imp[t] == load[t]

        if t == 0:
            model += soc[t] == ETA_CH * ch[t] - (1 / ETA_DIS) * dis[t]
        else:
            model += soc[t] == soc[t-1] + ETA_CH * ch[t] - (1 / ETA_DIS) * dis[t]

    model += soc[T - 1] == 0

    _solve_or_raise(model)

    op_value = float(pl.value(model.objective))

    sc_arr  = np.array([pl.value(sc[t])  for t in idx], dtype=float)
    ch_arr  = np.array([pl.value(ch[t])  for t in idx], dtype=float)
    dis_arr = np.array([pl.value(dis[t]) for t in idx], dtype=float)
    exp_arr = np.array([pl.value(exp[t]) for t in idx], dtype=float)
    imp_arr = np.array([pl.value(imp[t]) for t in idx], dtype=float)

    soc_arr = np.zeros(T, dtype=float)
    if DEBUG:
        x = 0.0
        for t in range(T):
            x = x + ETA_CH * ch_arr[t] - (1 / ETA_DIS) * dis_arr[t]
            soc_arr[t] = x

    dispatch = {"sc": sc_arr, "ch": ch_arr, "dis": dis_arr, "exp": exp_arr, "imp": imp_arr, "soc": soc_arr}
    return op_value, dispatch


# ============================================================
# DISPATCH SOLVER: COMPSIM (EXACT monthly cap inside LP)
# ============================================================
def dispatch_compsim_cap(pv, load, p_retail, p_comp, month_ids, C):
    """
    Optimal hourly dispatch under "Compensación simplificada" with monthly credit cap.

    Objective (incremental energy benefit, Year 1, EUR):
      op_value = Σ_t [ p_retail*(sc_t + dis_t) ] + Σ_m credit_m

    Monthly credit cap:
      credit_m <= p_comp   * Σ_exp_m
      credit_m <= p_retail * Σ_imp_m

    Notes:
    - Export does NOT earn direct market revenue.
    - Instead, it earns a monthly credit limited by (export value) and (monthly import cost).
    - Import cost does not need to be in the objective: import appears in constraints and in the cap.
    """
    T = len(pv)
    idx = range(T)

    months = pd.Index(month_ids).unique().tolist()
    month_to_ts = {m: np.where(month_ids == m)[0].tolist() for m in months}

    model = pl.LpProblem("ES_CompSim_ExactCap", pl.LpMaximize)

    sc  = pl.LpVariable.dicts("sc",  idx, lowBound=0)
    ch  = pl.LpVariable.dicts("ch",  idx, lowBound=0)
    dis = pl.LpVariable.dicts("dis", idx, lowBound=0)
    exp = pl.LpVariable.dicts("exp", idx, lowBound=0)
    imp = pl.LpVariable.dicts("imp", idx, lowBound=0)
    soc = pl.LpVariable.dicts("soc", idx, lowBound=0, upBound=C)

    credit = pl.LpVariable.dicts("credit", months, lowBound=0)  # EUR per month

    # Objective: retail savings + monthly credits
    model += (
        pl.lpSum(p_retail * (sc[t] + dis[t]) for t in idx)
        + pl.lpSum(credit[m] for m in months)
    )

    for t in idx:
        model += sc[t] + ch[t] + exp[t] <= pv[t]
        model += sc[t] + dis[t] + imp[t] == load[t]

        if t == 0:
            model += soc[t] == ETA_CH * ch[t] - (1 / ETA_DIS) * dis[t]
        else:
            model += soc[t] == soc[t-1] + ETA_CH * ch[t] - (1 / ETA_DIS) * dis[t]

    model += soc[T - 1] == 0

    # Monthly cap for credit
    for m in months:
        ts = month_to_ts[m]
        model += credit[m] <= p_comp   * pl.lpSum(exp[t] for t in ts)
        model += credit[m] <= p_retail * pl.lpSum(imp[t] for t in ts)

    _solve_or_raise(model)

    op_value = float(pl.value(model.objective))

    sc_arr  = np.array([pl.value(sc[t])  for t in idx], dtype=float)
    ch_arr  = np.array([pl.value(ch[t])  for t in idx], dtype=float)
    dis_arr = np.array([pl.value(dis[t]) for t in idx], dtype=float)
    exp_arr = np.array([pl.value(exp[t]) for t in idx], dtype=float)
    imp_arr = np.array([pl.value(imp[t]) for t in idx], dtype=float)

    soc_arr = np.zeros(T, dtype=float)
    if DEBUG:
        x = 0.0
        for t in range(T):
            x = x + ETA_CH * ch_arr[t] - (1 / ETA_DIS) * dis_arr[t]
            soc_arr[t] = x

    credit_vals = {m: float(pl.value(credit[m])) for m in months}

    dispatch = {"sc": sc_arr, "ch": ch_arr, "dis": dis_arr, "exp": exp_arr, "imp": imp_arr, "soc": soc_arr}
    return op_value, dispatch, credit_vals
