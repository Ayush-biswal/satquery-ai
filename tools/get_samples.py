import pandas as pd

DATASET_PATH = "data/BigEarthNet.txt.parquet"

df = pd.read_parquet(DATASET_PATH)

# Get benchmark VQA samples
bench = df[
    (df["split"] == "bench") &
    (df["type"].isin(["binary", "mcq"]))
]

# Keep one question per image pair
samples = bench.drop_duplicates(
    subset=["s1_name", "patch_id"]
).head(10)

print("Unique image pairs:", len(samples))

for _, row in samples.iterrows():
    print("\n" + "=" * 70)
    print("ID:", row["ID"])
    print("S1:", row["s1_name"])
    print("S2:", row["patch_id"])
    print("Type:", row["type"])
    print("Category:", row["category"])
    print("Question:", row["input"])
    print("Answer:", row["output"])