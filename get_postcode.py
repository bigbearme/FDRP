import rasterio
import numpy as np

# load TIF
tif_path = 'Risk/risk0601.tif'
output_txt = 'PostCode.txt'

with rasterio.open(tif_path) as src:
    band = src.read(1)

    min_val = band.min()
    print("Minimum:", min_val)

    # Obtain the valid value mask
    mask = band != min_val

    # Extract row and column values
    rows, cols = np.where(mask)
    values = band[rows, cols]

    print(f"Number of effective pixels: {len(values)}")

    # Write
    with open(output_txt, 'w') as f:
        for r, c in zip(rows, cols):
            f.write(f"{r} {c}\n")

    print(f"Save: {output_txt}")

