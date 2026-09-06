import rasterio
from pathlib import Path

IMAGE_DIR = Path("data/images/sample_1459")

for band in ["B02", "B03", "B04", "B08"]:
    path = IMAGE_DIR / f"{band}.tif"

    with rasterio.open(path) as src:
        print("\n-----------------------------")
        print("Band:", band)
        print("Width:", src.width)
        print("Height:", src.height)
        print("CRS:", src.crs)
        print("Resolution:", src.res)
        print("Data type:", src.dtypes[0])