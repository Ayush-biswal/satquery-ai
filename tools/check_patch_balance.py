import pandas as pd

DATASET_PATH = "data/BigEarthNet.txt.parquet"

PATCH_NAME = "S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_32_64"

df = pd.read_parquet(DATASET_PATH)

benchmark = df[
    df["split"] == "bench"
].copy()

samples = benchmark[
    benchmark["patch_id"].astype(str).str.contains(
        PATCH_NAME,
        regex=False
    )
].copy()

print("=" * 60)
print("PATCH:", PATCH_NAME)
print("=" * 60)

print("\nTotal benchmark questions:", len(samples))

print("\nAnswer distribution:")
print(samples["output"].value_counts())

print("\nQuestions:\n")

for _, row in samples.iterrows():

    print("-" * 60)
    print("ID:", row["ID"])
    print("Type:", row["type"])
    print("Category:", row["category"])
    print("Answer:", row["output"])
    print("Question:", row["input"])

print("\n" + "=" * 60)