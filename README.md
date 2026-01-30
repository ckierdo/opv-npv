\# OPV–PV–Battery NPV Model (Spain / Almeria)



\## 1. Project idea



This project evaluates the \*\*economic impact of PV and OPV installations with optional battery storage\*\*

for agricultural sites in Spain and Italy (currently: \*\*Almeria\*\*).



The model focuses on \*\*incremental value creation\*\* compared to a baseline without PV:

-avoided grid imports through self-consumption + export revenues

* two tariff schemes applied for Spain:

\- No Acogida: all Export can be sold to the grid at any time without Limit but with a charging fee/  tax 

\-  Compensación Simplificada:export is limited to  maximum of the Monthly electricity bill (monthly compensation credits)

\- crop yield changes due to shading



The result is a \*\*Net Present Value (NPV)\*\* over the project lifetime.



---



\## 2. Core modeling logic (important)



\### Baseline philosophy



This is a \*\*delta model\*\*.



Electricity costs that would exist \*without\* PV are \*\*not modeled explicitly\*\*.

Instead, only \*\*changes caused by PV / battery investment\*\* are considered:



\- \*\*Positive contributions\*\*

&nbsp; - avoided retail imports due to self-consumption

&nbsp; - export revenues (No Acogida)

&nbsp; - monthly compensation credits (CompSim)

&nbsp; - crop revenue changes



\- \*\*Negative contributions\*\*

&nbsp; - PV CAPEX / OPEX

&nbsp; - battery CAPEX / OPEX

&nbsp; - battery replacement costs



⚠️ There is \*\*no full bill simulation\*\* in the economic evaluation.

The optimization objective is the \*\*incremental energy value ("delta energy")\*\*.



---



\## 3. Optimization structure



\### Dispatch (hourly LP)



The dispatch problem determines:

\- self-consumption

\- battery charging / discharging

\- imports and exports



for a \*\*given battery capacity\*\*.



Two schemes are implemented:

\- \*\*ES\_NoAcogida\*\*: hourly export at DA price

\- \*\*ES\_CompSimProxy\_ExactCap\*\*: monthly credit capped by imports



The LP objective always returns an \*\*objective value = delta\_energy\_y1\*\*.



\### NPV calculation



The yearly delta energy value is:

\- degraded over time (technology-specific)

\- combined with crop deltas

\- discounted to compute NPV



---



\## 4. Project structure



```text

opv-npv/

│

├─ src/opvnpv/

│  ├─ dispatch.py        # hourly dispatch optimization (LP)

│  ├─ economics.py      # CAPEX, OPEX, cashflows, NPV

│  ├─ crops.py           # crop revenue lookup

│  ├─ data\_prep.py       # hourly data preparation

│  ├─ pipeline\_spain.py  # main scan logic

│  ├─ params.py          # global model parameters

│  └─ paths\_local.py     # local absolute paths (not for repo)

│

├─ scripts/

│  ├─ run\_spain.py       # full Spain scan

│  └─ smoke\_spain.py     # small test run

│

├─ tests/

│  ├─ test\_sanity.py     # basic consistency checks

│  └─ test\_imports.py

│

├─ pyproject.toml

└─ README.md



