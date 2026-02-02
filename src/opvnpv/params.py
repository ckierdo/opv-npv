import math

# ============================================================
# DEBUG / LOGGING
# ============================================================
DEBUG = False
SOLVER_MSG = False

# ============================================================
# PARAMETERS
# ============================================================
DISCOUNT_RATE = 0.07
PROJECT_YEARS = 25
PV_OPEX_FRAC = 0.015

BAT_OPEX_FRAC = 0.015
BAT_REPL_YEAR = 15
BAT_REPL_FACTOR = 0.8

DEG = {"polySi": 0.005, "opv63": 0.01, "opv145": 0.01}

AREA_M2 = 1000.0
MODULE_W_M2 = {"polySi": 230.0, "opv63": 63.0, "opv145": 145.0}

CAPEX_CSI_EUR_PER_KWP = 970.0
CAPEX_OPV_EUR_PER_KWP = {"opv63": 3345.0, "opv145": 940.0}

ETA_RT = 0.95
ETA_CH = math.sqrt(ETA_RT)
ETA_DIS = math.sqrt(ETA_RT)

BAT_BASE_CAPEX_EUR_PER_KWH = 400.0
BAT_REDUCTION_PER_4X = 0.10

CAPACITY_RANGE = range(0, 26)  # 0..25 kWh

# Spain retail (Eurostat proxy)
P_RETAIL_ES = 0.157  # €/kWh

# Export fee modeled as factor (e.g., "factor 95" => 0.95 of DA price)
EXPORT_FEE_FACTOR_NO_ACOGIDA = 0.95
