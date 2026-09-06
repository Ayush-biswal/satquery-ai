import rasterio
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


PATCH_DIR = Path("data/images/sample_1459/candidate_patch")
OUTPUT = Path("data/outputs/candidate_patch_rgb.png")


def normalize(image):
    low = np.percentile(image, 2)
    high = np.percentile(image, 98)

    if high <= low:
        return np.zeros_like(image, dtype=np.float32)

    image = np.clip(image, low, high)

    return (image - low) / (high - low)


def read_band(name):
    with rasterio.open(PATCH_DIR / f"{name}.tif") as src:
        return src.read(1).astype(np.float32)


# Sentinel-2 natural RGB
red = read_band("B04")
green = read_band("B03")
blue = read_band("B02")


rgb = np.dstack([
    normalize(red),
    normalize(green),
    normalize(blue)
])


plt.figure(figsize=(8, 8))
plt.imshow(rgb)
plt.title("BigEarthNet Candidate Patch")
plt.axis("off")
plt.tight_layout()


OUTPUT.parent.mkdir(parents=True, exist_ok=True)

plt.savefig(
    OUTPUT,
    dpi=200,
    bbox_inches="tight"
)

print("Saved:", OUTPUT)

plt.show()