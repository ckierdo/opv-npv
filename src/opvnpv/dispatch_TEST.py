# opvnpv/dispatch_one_stage.py
# One-stage economic dispatch, but returns DELTA operating value vs baseline ("do nothing").
#
# Baseline: import everything -> bill0 = p_retail * sum(load)
# With system: bill1 = p_retail * sum(imp)
#
# Returned eov = (bill0 - bill1) + self_reward + export_or_credit
# so it plugs directly into your build_cashflows_and_npv(delta-only) logic.

import math
from typing import Optional, Dict, Tuple

import numpy as np
import pandas as pd
import pulp as pl

ETA_RT = 0.95
ETA_CH = math.sqrt(ETA_RT)
ETA_DIS = math.sqrt(ETA_RT)


def _solve(model: pl.LpProblem, msg: bool = False):
    solver = pl.HIGHs(msg=msg) if hasattr(pl, "HIGHs") else pl.PULP_CBC_CMD(msg=msg)
    model.solve(solver)
    status = pl.LpStatus[model.status]
    if status != "Optimal":
        raise RuntimeError(f"Solver failed with status: {status}")


def dispatch_no_acogida_one_stage(
    pv: np.ndarray,
    load: np.ndarray,
    p_sell: np.ndarray,   # €/kWh export price (already net of 5% if you want)
    p_retail: float,      # €/kWh import price
    C: int,
    solver_msg: bool = False,
    r_self: Optional[float] = None,  # €/kWh paid for self-consumed (sc+dis). default = p_retail
) -> Tuple[float, Dict[str, np.ndarray]]:
    """
    One-stage LP dispatch for NoAcogida.
    LP objective is the true annual profit of operation (excluding constant baseline),
    but returned eov is DELTA vs baseline, as required by your economics.py.

    Returns:
      eov_delta (EUR), dispatch dict with arrays sc,ch,dis,exp,imp
    """
    pv = np.asarray(pv, dtype=float)
    load = np.asarray(load, dtype=float)
    p_sell = np.asarray(p_sell, dtype=float)

    if r_self is None:
        r_self = float(p_retail)  # your current assumption: 15.7c on self-consumed

    T = len(pv)
    idx = range(T)

    m = pl.LpProblem("NoAcogida_OneStage", pl.LpMaximize)

    sc  = pl.LpVariable.dicts("sc", idx, lowBound=0)
    ch  = pl.LpVariable.dicts("ch", idx, lowBound=0)
    dis = pl.LpVariable.dicts("dis", idx, lowBound=0)
    exp = pl.LpVariable.dicts("exp", idx, lowBound=0)
    imp = pl.LpVariable.dicts("imp", idx, lowBound=0)
    soc = pl.LpVariable.dicts("soc", idx, lowBound=0, upBound=C)

    for t in idx:
        m += sc[t] + ch[t] + exp[t] <= pv[t]
        m += sc[t] + dis[t] + imp[t] == load[t]
        if t == 0:
            m += soc[t] == ETA_CH * ch[t] - (1.0 / ETA_DIS) * dis[t]
        else:
            m += soc[t] == soc[t - 1] + ETA_CH * ch[t] - (1.0 / ETA_DIS) * dis[t]

    # keep your old end condition for comparability
    m += soc[T - 1] == 0

    if C == 0:
        for t in idx:
            m += ch[t] == 0
            m += dis[t] == 0

    # ONE-STAGE ECONOMIC objective (absolute, not delta):
    # profit = self_reward + export_rev - import_cost
    # (we'll convert to DELTA after solving)
    m += pl.lpSum(
        r_self * (sc[t] + dis[t]) + p_sell[t] * exp[t] - p_retail * imp[t]
        for t in idx
    )

    _solve(m, msg=solver_msg)

    sc_arr  = np.array([pl.value(sc[t])  for t in idx], dtype=float)
    ch_arr  = np.array([pl.value(ch[t])  for t in idx], dtype=float)
    dis_arr = np.array([pl.value(dis[t]) for t in idx], dtype=float)
    exp_arr = np.array([pl.value(exp[t]) for t in idx], dtype=float)
    imp_arr = np.array([pl.value(imp[t]) for t in idx], dtype=float)

    # ---- DELTA operating value vs baseline ("do nothing") ----
    bill0 = p_retail * float(np.sum(load))        # baseline: all load imported
    bill1 = p_retail * float(np.sum(imp_arr))     # with system: remaining imports
    self_rev = float(r_self * float(np.sum(sc_arr + dis_arr)))
    exp_rev  = float(np.sum(p_sell * exp_arr))
    eov_delta = (bill0 - bill1) + self_rev + exp_rev

    dispatch = {"sc": sc_arr, "ch": ch_arr, "dis": dis_arr, "exp": exp_arr, "imp": imp_arr}
    return float(eov_delta), dispatch


def dispatch_compsim_cap_one_stage(
    pv: np.ndarray,
    load: np.ndarray,
    p_retail: float,      # €/kWh import price
    p_comp: float,        # €/kWh credit rate (e.g. 0.087)
    month_ids: np.ndarray,
    C: int,
    solver_msg: bool = False,
    r_self: Optional[float] = None,  # €/kWh paid for self-consumed (sc+dis). default = p_retail
) -> Tuple[float, Dict[str, np.ndarray], Dict[int, float]]:
    """
    One-stage LP dispatch for CompSim with monthly credit caps.
    Returns DELTA operating value vs baseline ("do nothing").

    Monthly caps (per month m):
      credit[m] <= p_comp * sum(exp in m)
      credit[m] <= p_retail * sum(imp in m)    # cannot earn more than the remaining bill in that month

    Returns:
      eov_delta (EUR), dispatch dict, credit_vals dict(month->EUR)
    """
    pv = np.asarray(pv, dtype=float)
    load = np.asarray(load, dtype=float)
    month_ids = np.asarray(month_ids)

    if r_self is None:
        r_self = float(p_retail)

    T = len(pv)
    idx = range(T)

    months = pd.Index(month_ids).unique().tolist()
    month_to_ts = {m_: np.where(month_ids == m_)[0].tolist() for m_ in months}

    m = pl.LpProblem("CompSim_OneStage", pl.LpMaximize)

    sc  = pl.LpVariable.dicts("sc", idx, lowBound=0)
    ch  = pl.LpVariable.dicts("ch", idx, lowBound=0)
    dis = pl.LpVariable.dicts("dis", idx, lowBound=0)
    exp = pl.LpVariable.dicts("exp", idx, lowBound=0)
    imp = pl.LpVariable.dicts("imp", idx, lowBound=0)
    soc = pl.LpVariable.dicts("soc", idx, lowBound=0, upBound=C)
    credit = pl.LpVariable.dicts("credit", months, lowBound=0)

    for t in idx:
        m += sc[t] + ch[t] + exp[t] <= pv[t]
        m += sc[t] + dis[t] + imp[t] == load[t]
        if t == 0:
            m += soc[t] == ETA_CH * ch[t] - (1.0 / ETA_DIS) * dis[t]
        else:
            m += soc[t] == soc[t - 1] + ETA_CH * ch[t] - (1.0 / ETA_DIS) * dis[t]

    m += soc[T - 1] == 0

    if C == 0:
        for t in idx:
            m += ch[t] == 0
            m += dis[t] == 0

    for mo in months:
        ts = month_to_ts[mo]
        m += credit[mo] <= p_comp * pl.lpSum(exp[t] for t in ts)
        m += credit[mo] <= p_retail * pl.lpSum(imp[t] for t in ts)

    # ONE-STAGE ECONOMIC objective (absolute, not delta):
    # profit = self_reward + total_credit - import_cost
    m += (
        pl.lpSum(r_self * (sc[t] + dis[t]) - p_retail * imp[t] for t in idx)
        + pl.lpSum(credit[mo] for mo in months)
    )

    _solve(m, msg=solver_msg)

    sc_arr  = np.array([pl.value(sc[t])  for t in idx], dtype=float)
    ch_arr  = np.array([pl.value(ch[t])  for t in idx], dtype=float)
    dis_arr = np.array([pl.value(dis[t]) for t in idx], dtype=float)
    exp_arr = np.array([pl.value(exp[t]) for t in idx], dtype=float)
    imp_arr = np.array([pl.value(imp[t]) for t in idx], dtype=float)

    credit_vals = {mo: float(pl.value(credit[mo])) for mo in months}
    credit_total = float(sum(credit_vals.values()))

    # ---- DELTA operating value vs baseline ("do nothing") ----
    bill0 = p_retail * float(np.sum(load))
    bill1 = p_retail * float(np.sum(imp_arr))
    self_rev = float(r_self * float(np.sum(sc_arr + dis_arr)))
    eov_delta = (bill0 - bill1) + self_rev + credit_total

    dispatch = {"sc": sc_arr, "ch": ch_arr, "dis": dis_arr, "exp": exp_arr, "imp": imp_arr}
    return float(eov_delta), dispatch, credit_vals

# opvnpv/dispatch_min_import.py
# Lexicographic dispatch:
#   Stage 1: minimize total import (battery has a reason to exist)
#   Stage 2: maximize DELTA-cashflow (avoided import costs + export/credit),
#            BUT DO NOT subtract remaining import costs (because you treat them as baseline).

# import math
# from typing import Optional

# import numpy as np
# import pandas as pd
# import pulp as pl

# # Roundtrip efficiency
# ETA_RT = 0.95
# ETA_CH = math.sqrt(ETA_RT)
# ETA_DIS = math.sqrt(ETA_RT)


# def _solve(model: pl.LpProblem, msg: bool = False):
#     model.solve(pl.PULP_CBC_CMD(msg=msg))
#     status = pl.LpStatus[model.status]
#     if status != "Optimal":
#         raise RuntimeError(f"Solver failed with status: {status}")


# def dispatch_no_acogida_lexi(
#     pv: np.ndarray,
#     load: np.ndarray,
#     p_sell: np.ndarray,
#     p_retail: float,
#     C: int,
#     solver_msg: bool = False,
#     eps_import: Optional[float] = None,
#     eps_self: Optional[float] = None,
# ):
#     """
#     3-stage lexicographic dispatch for NoAcogida:

#       Stage 1: minimize total import sum(imp)
#       Stage 2: maximize total self-consumption sum(sc + dis) subject to minimal import
#       Stage 3: maximize NoAcogida € objective subject to:
#                - minimal import
#                - maximal self-consumption

#     Note: NoAcogida € objective here stays like your ORIGINAL model:
#           p_retail*(sc+dis) + p_sell*exp
#           (Import is handled via Stage 1, not as a -cost term)

#     Hard constraint for C==0:
#       force ch[t]=0 and dis[t]=0 (otherwise solver can produce weird artifacts in debug outputs)
#     """
#     pv = np.asarray(pv, dtype=float)
#     load = np.asarray(load, dtype=float)
#     p_sell = np.asarray(p_sell, dtype=float)

#     T = len(pv)
#     idx = range(T)

#     # -----------------------
#     # Helper: build common constraints
#     # -----------------------
#     def add_common_constraints(model, sc, ch, dis, exp, imp, soc):
#         for t in idx:
#             model += sc[t] + ch[t] + exp[t] <= pv[t]
#             model += sc[t] + dis[t] + imp[t] == load[t]
#             if t == 0:
#                 model += soc[t] == ETA_CH * ch[t] - (1 / ETA_DIS) * dis[t]
#             else:
#                 model += soc[t] == soc[t - 1] + ETA_CH * ch[t] - (1 / ETA_DIS) * dis[t]
#         model += soc[T - 1] == 0

#     def add_nocharge_nodischarge_if_C0(model, ch, dis):
#         if C == 0:
#             for t in idx:
#                 model += ch[t] == 0
#                 model += dis[t] == 0

#     # -----------------------
#     # Stage 1: Min import
#     # -----------------------
#     m1 = pl.LpProblem("NoAcogida_Stage1_MinImport", pl.LpMinimize)

#     sc1 = pl.LpVariable.dicts("sc", idx, lowBound=0)
#     ch1 = pl.LpVariable.dicts("ch", idx, lowBound=0)
#     dis1 = pl.LpVariable.dicts("dis", idx, lowBound=0)
#     exp1 = pl.LpVariable.dicts("exp", idx, lowBound=0)
#     imp1 = pl.LpVariable.dicts("imp", idx, lowBound=0)
#     soc1 = pl.LpVariable.dicts("soc", idx, lowBound=0, upBound=C)

#     m1 += pl.lpSum(imp1[t] for t in idx)
#     add_common_constraints(m1, sc1, ch1, dis1, exp1, imp1, soc1)
#     add_nocharge_nodischarge_if_C0(m1, ch1, dis1)

#     _solve(m1, msg=solver_msg)
#     imp_star = float(pl.value(m1.objective))

#     if eps_import is None:
#         eps_import = max(1e-6, 1e-6 * max(1.0, imp_star))

#     # -----------------------
#     # Stage 2: Max self-consumption at minimal import
#     # -----------------------
#     m2 = pl.LpProblem("NoAcogida_Stage2_MaxSelf", pl.LpMaximize)

#     sc2 = pl.LpVariable.dicts("sc", idx, lowBound=0)
#     ch2 = pl.LpVariable.dicts("ch", idx, lowBound=0)
#     dis2 = pl.LpVariable.dicts("dis", idx, lowBound=0)
#     exp2 = pl.LpVariable.dicts("exp", idx, lowBound=0)
#     imp2 = pl.LpVariable.dicts("imp", idx, lowBound=0)
#     soc2 = pl.LpVariable.dicts("soc", idx, lowBound=0, upBound=C)

#     m2 += pl.lpSum(sc2[t] + dis2[t] for t in idx)
#     add_common_constraints(m2, sc2, ch2, dis2, exp2, imp2, soc2)
#     add_nocharge_nodischarge_if_C0(m2, ch2, dis2)

#     m2 += pl.lpSum(imp2[t] for t in idx) <= imp_star + eps_import

#     _solve(m2, msg=solver_msg)
#     self_star = float(pl.value(m2.objective))

#     if eps_self is None:
#         eps_self = max(1e-6, 1e-6 * max(1.0, self_star))

#     # -----------------------
#     # Stage 3: Max € objective at minimal import + maximal self
#     # -----------------------
#     m3 = pl.LpProblem("NoAcogida_Stage3_MaxEUR", pl.LpMaximize)

#     sc3 = pl.LpVariable.dicts("sc", idx, lowBound=0)
#     ch3 = pl.LpVariable.dicts("ch", idx, lowBound=0)
#     dis3 = pl.LpVariable.dicts("dis", idx, lowBound=0)
#     exp3 = pl.LpVariable.dicts("exp", idx, lowBound=0)
#     imp3 = pl.LpVariable.dicts("imp", idx, lowBound=0)
#     soc3 = pl.LpVariable.dicts("soc", idx, lowBound=0, upBound=C)

#     m3 += pl.lpSum(
#         p_retail * (sc3[t] + dis3[t]) + p_sell[t] * exp3[t]
#         for t in idx
#     )

#     add_common_constraints(m3, sc3, ch3, dis3, exp3, imp3, soc3)
#     add_nocharge_nodischarge_if_C0(m3, ch3, dis3)

#     # lock stage1 + stage2
#     m3 += pl.lpSum(imp3[t] for t in idx) <= imp_star + eps_import
#     m3 += pl.lpSum(sc3[t] + dis3[t] for t in idx) >= self_star - eps_self

#     _solve(m3, msg=solver_msg)

#     op_value = float(pl.value(m3.objective))

#     dispatch = {
#         "sc": np.array([pl.value(sc3[t]) for t in idx], dtype=float),
#         "ch": np.array([pl.value(ch3[t]) for t in idx], dtype=float),
#         "dis": np.array([pl.value(dis3[t]) for t in idx], dtype=float),
#         "exp": np.array([pl.value(exp3[t]) for t in idx], dtype=float),
#         "imp": np.array([pl.value(imp3[t]) for t in idx], dtype=float),
#     }
#     return op_value, dispatch


# # ============================================================
# # COMPSIM: Stage1 min import, Stage2 max self, Stage3 max € + credit
# # ============================================================
# def dispatch_compsim_cap_lexi(
#     pv: np.ndarray,
#     load: np.ndarray,
#     p_retail: float,
#     p_comp: float,
#     month_ids: np.ndarray,
#     C: int,
#     solver_msg: bool = False,
#     eps_import: Optional[float] = None,
#     eps_self: Optional[float] = None,
# ):
#     """
#     3-stage lexicographic dispatch for CompSim:

#       Stage 1: minimize total import sum(imp)
#       Stage 2: maximize total self-consumption sum(sc + dis) subject to minimal import
#       Stage 3: maximize economic objective (retail savings + credit) subject to:
#                - minimal import
#                - maximal self-consumption

#     Hard constraint for C==0:
#       force ch[t]=0 and dis[t]=0 (otherwise solver can produce weird artifacts in debug outputs)
#     """
#     pv = np.asarray(pv, dtype=float)
#     load = np.asarray(load, dtype=float)
#     month_ids = np.asarray(month_ids)

#     T = len(pv)
#     idx = range(T)

#     months = pd.Index(month_ids).unique().tolist()
#     month_to_ts = {m: np.where(month_ids == m)[0].tolist() for m in months}

#     # -----------------------
#     # Helper: build common constraints
#     # -----------------------
#     def add_common_constraints(model, sc, ch, dis, exp, imp, soc):
#         for t in idx:
#             model += sc[t] + ch[t] + exp[t] <= pv[t]
#             model += sc[t] + dis[t] + imp[t] == load[t]
#             if t == 0:
#                 model += soc[t] == ETA_CH * ch[t] - (1 / ETA_DIS) * dis[t]
#             else:
#                 model += soc[t] == soc[t - 1] + ETA_CH * ch[t] - (1 / ETA_DIS) * dis[t]
#         model += soc[T - 1] == 0

#     def add_nocharge_nodischarge_if_C0(model, ch, dis):
#         if C == 0:
#             for t in idx:
#                 model += ch[t] == 0
#                 model += dis[t] == 0

#     def add_monthly_credit_caps(model, credit, exp, imp):
#         for m in months:
#             ts = month_to_ts[m]
#             model += credit[m] <= p_comp * pl.lpSum(exp[t] for t in ts)
#             model += credit[m] <= p_retail * pl.lpSum(imp[t] for t in ts)

#     # -----------------------
#     # Stage 1: Min import
#     # -----------------------
#     m1 = pl.LpProblem("CompSim_Stage1_MinImport", pl.LpMinimize)

#     sc1 = pl.LpVariable.dicts("sc", idx, lowBound=0)
#     ch1 = pl.LpVariable.dicts("ch", idx, lowBound=0)
#     dis1 = pl.LpVariable.dicts("dis", idx, lowBound=0)
#     exp1 = pl.LpVariable.dicts("exp", idx, lowBound=0)
#     imp1 = pl.LpVariable.dicts("imp", idx, lowBound=0)
#     soc1 = pl.LpVariable.dicts("soc", idx, lowBound=0, upBound=C)
#     credit1 = pl.LpVariable.dicts("credit", months, lowBound=0)

#     m1 += pl.lpSum(imp1[t] for t in idx)
#     add_common_constraints(m1, sc1, ch1, dis1, exp1, imp1, soc1)
#     add_nocharge_nodischarge_if_C0(m1, ch1, dis1)
#     add_monthly_credit_caps(m1, credit1, exp1, imp1)

#     _solve(m1, msg=solver_msg)
#     imp_star = float(pl.value(m1.objective))

#     if eps_import is None:
#         eps_import = max(1e-6, 1e-6 * max(1.0, imp_star))

#     # -----------------------
#     # Stage 2: Max self-consumption at minimal import
#     # -----------------------
#     m2 = pl.LpProblem("CompSim_Stage2_MaxSelf", pl.LpMaximize)

#     sc2 = pl.LpVariable.dicts("sc", idx, lowBound=0)
#     ch2 = pl.LpVariable.dicts("ch", idx, lowBound=0)
#     dis2 = pl.LpVariable.dicts("dis", idx, lowBound=0)
#     exp2 = pl.LpVariable.dicts("exp", idx, lowBound=0)
#     imp2 = pl.LpVariable.dicts("imp", idx, lowBound=0)
#     soc2 = pl.LpVariable.dicts("soc", idx, lowBound=0, upBound=C)
#     credit2 = pl.LpVariable.dicts("credit", months, lowBound=0)

#     m2 += pl.lpSum(sc2[t] + dis2[t] for t in idx)  # self-consumption energy
#     add_common_constraints(m2, sc2, ch2, dis2, exp2, imp2, soc2)
#     add_nocharge_nodischarge_if_C0(m2, ch2, dis2)
#     add_monthly_credit_caps(m2, credit2, exp2, imp2)

#     # lock minimal import
#     m2 += pl.lpSum(imp2[t] for t in idx) <= imp_star + eps_import

#     _solve(m2, msg=solver_msg)
#     self_star = float(pl.value(m2.objective))

#     if eps_self is None:
#         eps_self = max(1e-6, 1e-6 * max(1.0, self_star))

#     # -----------------------
#     # Stage 3: Max economic objective at minimal import + maximal self
#     # -----------------------
#     m3 = pl.LpProblem("CompSim_Stage3_MaxEUR", pl.LpMaximize)

#     sc3 = pl.LpVariable.dicts("sc", idx, lowBound=0)
#     ch3 = pl.LpVariable.dicts("ch", idx, lowBound=0)
#     dis3 = pl.LpVariable.dicts("dis", idx, lowBound=0)
#     exp3 = pl.LpVariable.dicts("exp", idx, lowBound=0)
#     imp3 = pl.LpVariable.dicts("imp", idx, lowBound=0)
#     soc3 = pl.LpVariable.dicts("soc", idx, lowBound=0, upBound=C)
#     credit3 = pl.LpVariable.dicts("credit", months, lowBound=0)

#     # € objective (wie bei dir)
#     m3 += (
#         pl.lpSum(p_retail * (sc3[t] + dis3[t]) for t in idx)
#         + pl.lpSum(credit3[m] for m in months)
#     )

#     add_common_constraints(m3, sc3, ch3, dis3, exp3, imp3, soc3)
#     add_nocharge_nodischarge_if_C0(m3, ch3, dis3)
#     add_monthly_credit_caps(m3, credit3, exp3, imp3)

#     # lock stage1 + stage2
#     m3 += pl.lpSum(imp3[t] for t in idx) <= imp_star + eps_import
#     m3 += pl.lpSum(sc3[t] + dis3[t] for t in idx) >= self_star - eps_self

#     _solve(m3, msg=solver_msg)

#     op_value = float(pl.value(m3.objective))

#     dispatch = {
#         "sc": np.array([pl.value(sc3[t]) for t in idx], dtype=float),
#         "ch": np.array([pl.value(ch3[t]) for t in idx], dtype=float),
#         "dis": np.array([pl.value(dis3[t]) for t in idx], dtype=float),
#         "exp": np.array([pl.value(exp3[t]) for t in idx], dtype=float),
#         "imp": np.array([pl.value(imp3[t]) for t in idx], dtype=float),
#     }
#     credit_vals = {m: float(pl.value(credit3[m])) for m in months}

#     return op_value, dispatch, credit_vals
