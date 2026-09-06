import rasterio
from rasterio.windows import Window
from pyproj import Transformer
from pathlib import Path


IMAGE_DIR = Path("data/images/sample_1459")
OUTPUT_DIR = IMAGE_DIR / "candidate_patch"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# BigEarthNet benchmark coordinates
LAT = 48.0363736865
LON = 12.840120739738445

BANDS = ["B02", "B03", "B04", "B08"]


# Open one band to get the raster CRS and transform
with rasterio.open(IMAGE_DIR / "B04.tif") as src:

    # Convert WGS84 (latitude/longitude)
    # into the raster's UTM coordinate system
    transformer = Transformer.from_crs(
        "EPSG:4326",
        src.crs,
        always_xy=True
    )

    x, y = transformer.transform(LON, LAT)

    print("UTM coordinates:")
    print("Easting:", x)
    print("Northing:", y)

    # Convert UTM coordinates to pixel coordinates
    row, col = src.index(x, y)

    print("\nPixel position:")
    print("Row:", row)
    print("Column:", col)

    patch_size = 120

    row_start = row - patch_size // 2
    col_start = col - patch_size // 2

    window = Window(
        col_start,
        row_start,
        patch_size,
        patch_size
    )

    print("\nPatch window:")
    print(window)


# Extract all bands using the same window
for band in BANDS:

    input_path = IMAGE_DIR / f"{band}.tif"
    output_path = OUTPUT_DIR / f"{band}.tif"

    with rasterio.open(input_path) as src:

        data = src.read(1, window=window)

        transform = src.window_transform(window)

        profile = src.profile.copy()

        profile.update(
            width=patch_size,
            height=patch_size,
            transform=transform
        )

        with rasterio.open(
            output_path,
            "w",
            **profile
        ) as dst:

            dst.write(data, 1)

    print("Saved:", output_path)


print("\nCandidate patch extraction complete!")