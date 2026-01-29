import pandas as pd

def safe_read_csv(path: str) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        head = f.readline()
    sep = ";" if head.count(";") > head.count(",") else ","
    return pd.read_csv(path, sep=sep, low_memory=False)

def region_key(s: str) -> str:
    return str(s).replace("\n", " ").replace("\r", " ").upper().split("(")[0].strip()
