import pandas as pd

DATASET_PATH = "data/BigEarthNet.txt.parquet"

df = pd.read_parquet(DATASET_PATH)

bench = df[df["split"] == "bench"]

print("BENCHMARK ROWS:", len(bench))

print("\n--- BENCHMARK TYPES ---")
print(bench["type"].value_counts())

print("\n--- BENCHMARK CATEGORIES ---")
print(bench["category"].value_counts())

print("\n--- FIRST 10 BENCHMARK VQA SAMPLES ---")

vqa = bench[bench["type"].isin(["binary", "mcq"])]

for _, row in vqa.head(10).iterrows():
    print("\nID:", row["ID"])
    print("Type:", row["type"])
    print("Category:", row["category"])
    print("Input:", row["input"])
    print("Output:", row["output"])
    print("S1:", row["s1_name"])
    print("Patch:", row["patch_id"])