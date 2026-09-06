import pandas as pd
import pystac_client
import planetary_computer
import requests
from pathlib import Path


DATASET_PATH = "data/BigEarthNet.txt.parquet"
OUTPUT_DIR = Path("data/images/sample_1459")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Load benchmark sample
df = pd.read_parquet(DATASET_PATH)
sample = df[df["ID"] == 1459].iloc[0]

lat = float(sample["latitude"])
lon = float(sample["longitude"])


# Connect to Planetary Computer
catalog = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace,
)


# Search for matching Sentinel-2 scene
search = catalog.search(
    collections=["sentinel-2-l2a"],
    bbox=[
        lon - 0.01,
        lat - 0.01,
        lon + 0.01,
        lat + 0.01,
    ],
    datetime="2017-06-13/2017-06-14",
    query={
        "s2:mgrs_tile": {"eq": "33UUP"}
    },
)

items = list(search.items())

if not items:
    raise RuntimeError("No Sentinel-2 scene found.")

item = items[0]

print("Using scene:")
print(item.id)

print("\nDownloading bands...")


bands = ["B02", "B03", "B04", "B08"]

for band in bands:

    asset = item.assets[band]

    output_path = OUTPUT_DIR / f"{band}.tif"

    print(f"Downloading {band}...")

    response = requests.get(asset.href, stream=True)
    response.raise_for_status()

    with open(output_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)

    print(f"Saved: {output_path}")


print("\nDone!")