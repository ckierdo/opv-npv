import pandas as pd
from .io_utils import safe_read_csv, region_key

def load_crop_table(path: str) -> pd.DataFrame:
    crops = safe_read_csv(path)
    crops["region_key"] = crops["Region"].apply(region_key)

    crop_pivot = (
        crops.pivot_table(
            index=["region_key", "Shading_pct"],
            columns="Yield_Scenario",
            values="Scenario_Annual_Revenue_EUR",
            aggfunc="mean"
        ).reset_index()
    )

    crop_base = crops[["region_key", "Shading_pct", "Baseline_Annual_Revenue_EUR"]].drop_duplicates()
    return crop_base.merge(crop_pivot, on=["region_key", "Shading_pct"], how="left")

def get_crop_row(crop_all: pd.DataFrame, region_key_target: str, shading_pct: int):
    sel = crop_all[(crop_all["region_key"] == region_key_target) &
                   (crop_all["Shading_pct"] == shading_pct)]
    return None if sel.empty else sel.iloc[0]
