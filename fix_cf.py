import pandas as pd
d = pd.read_csv("results/val_results.csv")
d.loc[d.model == "causal_forest_400k", "model"] = "causal_forest_400k_800t"
d.to_csv("results/val_results.csv", index=False)
print(d[d.model.str.startswith("causal")][["model", "metric", "value"]])