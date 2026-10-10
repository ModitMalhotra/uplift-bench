import pandas as pd

total, sums = 0, None
for chunk in pd.read_csv("data/criteo-uplift-v2.1.csv.gz", chunksize=1_000_000):
    total += len(chunk)
    s = chunk[["treatment", "visit", "conversion", "exposure"]].sum()
    sums = s if sums is None else sums + s

print(total)
print(sums / total)