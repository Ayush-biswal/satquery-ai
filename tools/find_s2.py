import pandas as pd
import pystac_client
import planetary_computer


# --------------------------------------------------
# 1. Get our benchmark sample
# --------------------------------------------------

DATASET_PATH = "data/BigEarthNet.txt.parquet"

df = pd.read_parquet(DATASET_PATH)

sample = df[df["ID"] == 1459].iloc[0]

print("Benchmark sample:")
print("Question:", sample["input"])
print("Answer:", sample["output"])
print("S2 patch:", sample["patch_id"])
print("Latitude:", sample["latitude"])
print("Longitude:", sample["longitude"])


# --------------------------------------------------
# 2. Connect to Microsoft Planetary Computer
# --------------------------------------------------

catalog = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace,
)


# --------------------------------------------------
# 3. Search for the Sentinel-2 scene
# --------------------------------------------------

lat = float(sample["latitude"])
lon = float(sample["longitude"])

bbox = [
    lon - 0.01,
    lat - 0.01,
    lon + 0.01,
    lat + 0.01,
]

search = catalog.search(
    collections=["sentinel-2-l2a"],
    bbox=bbox,
    datetime="2017-06-13/2017-06-14",
    query={
        "s2:mgrs_tile": {"eq": "33UUP"}
    },
)

items = search.item_collection()

print("\nScenes found:", len(items))

for item in items:
    print("\n-----------------------------")
    print("Item:", item.id)
    print("Date:", item.datetime)
    print("Tile:", item.properties.get("s2:mgrs_tile"))
    print(
        "Cloud cover:",
        item.properties.get("eo:cloud_cover")
    )