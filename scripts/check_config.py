from pathlib import Path
import yaml

def load_config():
    local = Path("configs/spain_almeria.local.yaml")
    example = Path("configs/spain_almeria.example.yaml")

    cfg_path = local if local.exists() else example
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    print(f"Loaded config: {cfg_path}")
    return cfg

def main():
    cfg = load_config()
    print("Project:", cfg["project"]["name"])
    print("Region:", cfg["region"]["key"])
    print("Hourly balance path:", cfg["paths"]["hourly_balance_csv"])
    print("Export fee factor:", cfg["tariffs"]["export_fee_factor"])

if __name__ == "__main__":
    main()
