import os
import numpy as np
import rasterio

def load_data(
    data_dir="Data",
    rainfall_dir=None,
    risk_dir="Risk",
    postcode_path="PostCode.txt",
    split_ratio=0.8,
    do_normalize=True
):

    if rainfall_dir is None:
        rainfall_dir = os.path.join(data_dir, "daily rainfall")

    # read postcodes
    with open(postcode_path, 'r') as f:
        postcodes = [tuple(map(int, line.strip().split())) for line in f.readlines()]
    N = len(postcodes)

    rainfall_files = sorted([f for f in os.listdir(rainfall_dir) if f.endswith(".tif")])
    T = len(rainfall_files)
    risk_files = sorted([f for f in os.listdir(risk_dir) if f.endswith(".tif")])
    feature_dirs = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])

    F = len(feature_dirs)

    print(f"Found: N(postcodes)={N}, T(time)={T}, features F={F}")

    # Initialize arrays
    X = np.full((N, T, F), np.nan, dtype=np.float32)
    Y = np.full((N, T, 1), np.nan, dtype=np.float32)

    for f_index, feature in enumerate(feature_dirs):
        feature_path = os.path.join(data_dir, feature)
        feature_files = sorted([f for f in os.listdir(feature_path) if f.endswith(".tif")])
        is_dynamic = len(feature_files) == T

        for t in range(T):
            file_index = t if is_dynamic else 0
            feature_file = os.path.join(feature_path, feature_files[file_index])

            with rasterio.open(feature_file) as src:
                band = src.read(1)
                nodata = src.nodata

                for n, (r, c) in enumerate(postcodes):
                    if 0 <= r < band.shape[0] and 0 <= c < band.shape[1]:
                        val = band[r, c]
                        if nodata is not None and val == nodata:
                            val = np.nan
                        if val is not None and (isinstance(val, (float, np.floating)) and val < -1e9):
                            val = np.nan
                        X[n, t, f_index] = val

    for t_index, filename in enumerate(risk_files):
        filepath = os.path.join(risk_dir, filename)
        with rasterio.open(filepath) as src:
            band = src.read(1)
            nodata = src.nodata
            for n, (r, c) in enumerate(postcodes):
                if 0 <= r < band.shape[0] and 0 <= c < band.shape[1]:
                    val = band[r, c]
                    if nodata is not None and val == nodata:
                        val = np.nan
                    if val is not None and (isinstance(val, (float, np.floating)) and val < -1e9):
                        val = np.nan
                    Y[n, t_index, 0] = val

    X[np.isnan(X)] = 0.0
    Y[np.isnan(Y)] = 0.0

    split_day = int(T * split_ratio)

    X_all = X.copy()
    Y_all = Y.copy()

    X_train_np = X_all[:, :split_day, :]
    Y_train_np = Y_all[:, :split_day, :]

    X_test_np = X_all[:, split_day:, :]
    Y_test_np = Y_all[:, split_day:, :]

    x_mean = x_std = y_mean = y_std = None
    if do_normalize:
        x_flat = X_train_np.reshape(-1, X_train_np.shape[2])
        x_mean = x_flat.mean(axis=0, keepdims=True)  # (1, F)
        x_std = x_flat.std(axis=0, keepdims=True) + 1e-6

        y_flat = Y_train_np.reshape(-1)
        y_mean = float(y_flat.mean())
        y_std = float(y_flat.std() + 1e-6)

        X_all = (X_all - x_mean.reshape(1, 1, -1)) / x_std.reshape(1, 1, -1)
        Y_all = (Y_all - y_mean) / y_std

        X_train_np = X_all[:, :split_day, :]
        Y_train_np = Y_all[:, :split_day, :]
        X_test_np = X_all[:, split_day:, :]
        Y_test_np = Y_all[:, split_day:, :]

        print("Data normalized using training mean/std.")
    else:
        print("Skipping normalization.")

    result = {
        "X_all": X_all,
        "Y_all": Y_all,
        "X_train_np": X_train_np,
        "Y_train_np": Y_train_np,
        "X_test_np": X_test_np,
        "Y_test_np": Y_test_np,
        "postcodes": postcodes,
        "rainfall_files": rainfall_files,
        "risk_files": risk_files,
        "feature_dirs": feature_dirs,
        "split_day": split_day,
        "norm": {
            "x_mean": x_mean,
            "x_std": x_std,
            "y_mean": y_mean,
            "y_std": y_std
        }
    }
    return result


if __name__ == "__main__":
    out = load_data()
    print("Loaded arrays shapes:")
    print(" X_all:", out["X_all"].shape)
    print(" Y_all:", out["Y_all"].shape)
    print(" X_train_np:", out["X_train_np"].shape)
    print(" X_test_np:", out["X_test_np"].shape)
    print(" sample feature_dirs:", out["feature_dirs"][:10])
