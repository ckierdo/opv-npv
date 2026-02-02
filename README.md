# OPV–PV–Battery NPV Model  
**Spain (Almeria)**

---

## 1. Project overview

This project evaluates the **economic impact of PV and OPV installations with optional battery storage**
for agricultural sites in Spain (currently **Almeria**).

The model computes the **Net Present Value (NPV)** of PV–battery systems by combining:

- energy-related operating benefits  
- crop revenue effects due to shading  

The analysis is strictly **incremental** compared to a baseline without PV.

---

## 2. Baseline philosophy (delta-only model)

This is a **delta-only model**.

Electricity costs and revenues that would exist **without PV** are **not modeled explicitly**.
Instead, only **changes caused by the PV and battery investment** are considered.

### Included positive effects

- avoided retail imports via self-consumption  
- export revenues (No Acogida)  
- monthly compensation credits (Compensación Simplificada)  
- crop revenue changes due to shading  

### Included negative effects

- PV CAPEX and OPEX  
- battery CAPEX and OPEX  
- battery replacement costs  

There is **no full electricity bill simulation**.
The optimization objective focuses exclusively on the **energy-related operating value**.

---

## 3. Optimization structure

The model uses a **two-level optimization approach**.

---

### 3.1 Hourly dispatch optimization (inner optimization)

For a **given battery capacity C**, the model solves an **hourly linear optimization problem (LP)**.

The dispatch determines the optimal hourly energy flows:

- PV → load (self-consumption)  
- PV → battery (charging)  
- battery → load (discharging)  
- PV → grid (export)  
- grid → load (import)  

The dispatch is subject to:

- hourly energy balance constraints  
- battery state-of-charge (SOC) dynamics  
- round-trip efficiency  
- battery capacity limits  
- end-of-horizon condition: **SOC = 0**  

The dispatch optimization returns:

- optimal hourly dispatch variables  
- a single scalar value:

```text
energy_operating_value_y1   [EUR, Year 1]
```
This value represents the annual energy-related operating benefit in the first year.

⚠️ The battery cannot export to the grid (battery-to-grid excluded).
The battery is used exclusively to increase self-consumption and reduce imports.
### 3.2 Battery size optimization (outer optimization)

The battery capacity is **not optimized inside the LP**.

Instead, the model performs a **discrete scan** over a predefined capacity range:
```text
C ∈ {0, 1, 2, …, C_max}  kWh
````

For each battery size `C`:

1. the hourly dispatch LP is solved  
2. `energy_operating_value_y1` is obtained  
3. project cashflows are constructed  
4. the **Net Present Value (NPV)** is calculated  

The **optimal battery size** is the one that **maximizes NPV**.

---

## 4. Tariff schemes implemented

### 4.1 ES_NoAcogida

- PV export allowed at any time  
- export revenue at hourly day-ahead price (optionally reduced by a fee factor)  
- battery increases self-consumption by shifting PV generation to load hours  

### 4.2 ES_CompSimProxy_ExactCap

- export does **not** earn direct market revenue  
- monthly compensation credits are modeled explicitly inside the LP  
- credits are capped by:
  - exported energy value  
  - monthly import cost  
- battery operation can increase or reduce usable credits depending on load structure  

---

## 5. Economic evaluation (NPV)

Cashflows are constructed as follows.

### Year 0
```text
− PV CAPEX  
− Battery CAPEX 
```
### Year 1 - N
```
+ degraded(energy_operating_value_y1)  
+ delta_crop_revenue  
− PV OPEX  
− Battery OPEX
```
Additional features:

technology-specific degradation

battery replacement in a fixed year

discounting using a constant discount rate

The final performance metric is:
````
NPV [EUR]
````
## 6. Project structure
````
opv-npv/
│
├─ src/opvnpv/
│  ├─ dispatch.py        # hourly dispatch optimization (LP)
│  ├─ economics.py       # CAPEX, OPEX, cashflows, NPV
│  ├─ crops.py           # crop revenue lookup
│  ├─ data_prep.py       # hourly data preparation
│  ├─ pipeline_spain.py  # battery scan and NPV optimization
│  ├─ params.py          # global parameters
│  └─ paths_local.py     # local absolute paths (not committed)
│
├─ scripts/
│  ├─ run_spain.py       # full Spain optimization run
│  └─ smoke_spain.py     # fast dispatch sanity checks
│
├─ tests/
│  ├─ test_sanity.py
│  └─ test_imports.py
│
├─ pyproject.toml
└─ README.md
````
## 7. Key design decisions

battery size optimized via NPV scan, not inside the LP

energy-only objective in dispatch

no battery-to-grid export

crop effects added at cashflow level

focus on transparency and debuggability

## 8. Interpretation of results

if the optimal battery size is C = 0, the battery is technically useful but not economically viable

if the energy operating value increases with battery size but NPV decreases, CAPEX and OPEX dominate

in compensation schemes, battery operation can reduce usable credits by lowering imports

## 9. Known limitations

no battery-to-grid arbitrage

no cycle-dependent battery degradation

no power limits (only energy capacity)

perfect foresight (no uncertainty)

These simplifications are intentional to maintain model clarity.

## 10. Status

The model is stable, refactored for clarity, and ready for full-scale scenario runs.