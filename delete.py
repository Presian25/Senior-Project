from pathlib import Path

for model in ["EGARCH_1_1", "SARIMA", "LogHAR"]:
    for f in Path("Data/Results/forecasts").glob(f"*/*/{model}.csv"):
        f.unlink()
        print("deleted", f)