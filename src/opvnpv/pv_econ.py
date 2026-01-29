from .params import AREA_M2, MODULE_W_M2, CAPEX_CSI_EUR_PER_KWP, CAPEX_OPV_EUR_PER_KWP, PV_OPEX_FRAC

def compute_pv_economics(tech: str, cov: float):
    pv_kWp = MODULE_W_M2[tech] * AREA_M2 * cov / 1000.0
    if tech == "polySi":
        pv_capex = CAPEX_CSI_EUR_PER_KWP * pv_kWp
    else:
        pv_capex = CAPEX_OPV_EUR_PER_KWP[tech] * pv_kWp
    pv_opex = pv_capex * PV_OPEX_FRAC
    return pv_kWp, pv_capex, pv_opex
