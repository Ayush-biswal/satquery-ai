import rasterio
from rasterio.transform import rowcol
from pyproj import Transformer
from pathlib import Path

IMAGE = Path("data/images/sample_1459/B04.tif")

LAT = 48.0363736865
LON = 12.840120739738445

with rasterio.open(IMAGE) as src:

    print("Raster bounds:")
    print(src.bounds)

    print("\nRaster CRS:")
    print(src.crs)

    # Convert WGS84 latitude/longitude
    # to the raster's UTM coordinate system
    transformer = Transformer.from_crs(
        "EPSG:4326",
        src.crs,
        always_xy=True
    )

    x, y = transformer.transform(LON, LAT)

    print("\nConverted coordinates:")
    print("Easting:", x)
    print("Northing:", y)

    # Convert UTM coordinates to pixel coordinates
    row, col = rowcol(src.transform, x, y)

    print("\nPixel corresponding to benchmark coordinate:")
    print("Row:", row)
    print("Column:", col)

    inside = (
        0 <= row < src.height
        and
        0 <= col < src.width
    )

    print("\nInside raster:", inside)

    if inside:

        window = rasterio.windows.Window(
            max(0, col - 5),
            max(0, row - 5),
            10,
            10
        )

        data = src.read(1, window=window)

        print("\nValues around coordinate:")
        print("Min:", data.min())
        print("Max:", data.max())
        print("Mean:", data.mean())
        print("Unique:", len(set(data.flatten())))