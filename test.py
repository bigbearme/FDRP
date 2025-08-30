import os
import numpy as np
import torch
from model import InformerModel
from sklearn.metrics import mean_squared_error, r2_score
import rasterio

from dataload import load_data

# ========== CONFIG ==========
data_dir = "Data"
rainfall_dir = os.path.join(data_dir, "daily rainfall")
risk_dir = "Risk"
postcode_path = "PostCode.txt"

seq_len = 30
label_len = 1
out_len = 1

d_model = 128
nhead = 4
e_layers = 2
d_layers = 1
dropout = 0.1

predict_batch_size = 1024
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NODATA_VALUE = -9999.0

ckpt_path = os.path.join("checkpoints", "epochckpt.pt")
output_dir = "PredictedRisk"
os.makedirs(output_dir, exist_ok=True)


# ========== UTILS ==========
def predict_batchwise(model, enc_x_all, dec_x_all, device, bs=1024):
    model.eval()
    N = enc_x_all.shape[0]
    preds = []
    with torch.no_grad():
        for i in range(0, N, bs):
            enc_b = torch.from_numpy(enc_x_all[i:i+bs]).float().to(device)
            dec_b = torch.from_numpy(dec_x_all[i:i+bs]).float().to(device)
            out_b = model(enc_b, dec_b)  # (B, out_len, 1)
            out_b = out_b.squeeze(-1).cpu().numpy()
            preds.append(out_b)
    preds = np.concatenate(preds, axis=0)
    return preds


def save_prediction_as_tif(preds, postcodes, reference_tif_path, output_path, nodata_value=NODATA_VALUE):
    with rasterio.open(reference_tif_path) as src:
        height, width = src.height, src.width
        transform = src.transform
        crs = src.crs

    out_arr = np.full((height, width), nodata_value, dtype=np.float32)
    for (r, c), val in zip(postcodes, preds):
        if 0 <= r < height and 0 <= c < width:
            if np.isnan(val):
                out_arr[r, c] = nodata_value
            else:
                out_arr[r, c] = float(val)

    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype=out_arr.dtype,
        crs=crs,
        transform=transform,
        nodata=nodata_value
    ) as dst:
        dst.write(out_arr, 1)


def evaluate_metrics(preds, true):
    preds = np.array(preds).squeeze()
    true = np.array(true).squeeze()
    mse = mean_squared_error(true, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(true, preds)
    return {"MSE": float(mse), "RMSE": float(rmse), "R2": float(r2)}


def robust_load_state_dict(model, ckpt_path, device):
    obj = torch.load(ckpt_path, map_location=device)
    if isinstance(obj, dict):
        if "model_state_dict" in obj:
            state = obj["model_state_dict"]
        elif "state_dict" in obj:
            state = obj["state_dict"]
        else:
            state = obj
    else:
        state = obj
    model.load_state_dict(state)
    return model


# ========== MAIN ==========
def main():
    print("[TEST] Start testing")
    data = load_data(
        data_dir=data_dir,
        rainfall_dir=None,
        risk_dir=risk_dir,
        postcode_path=postcode_path,
        split_ratio=0.8,
        do_normalize=True
    )

    X_all = data["X_all"]
    Y_all = data["Y_all"]
    X_test_np = data["X_test_np"]
    Y_test_np = data["Y_test_np"]
    postcodes = data["postcodes"]
    rainfall_files = data["rainfall_files"]
    split_day = data["split_day"]
    norm = data["norm"]

    y_mean = norm.get("y_mean", None)
    y_std = norm.get("y_std", None)
    is_normalized = (y_mean is not None) and (y_std is not None)

    enc_in = X_all.shape[2]
    dec_in = 1

    model = InformerModel(
        enc_in=enc_in, dec_in=dec_in, c_out=1,
        seq_len=seq_len, label_len=label_len, out_len=out_len,
        d_model=d_model, nhead=nhead, e_layers=e_layers, d_layers=d_layers, dropout=dropout
    ).to(device)

    model = robust_load_state_dict(model, ckpt_path, device)
    model.eval()
    print(f"Loaded checkpoint from: {ckpt_path}")

    reference_tif = os.path.join(rainfall_dir, rainfall_files[0])
    total_test_days = X_test_np.shape[1]
    print(f"\nStart day-by-day prediction for {total_test_days} test days...")

    for day in range(total_test_days):
        test_day_idx = split_day + day
        print(f"Predicting test day {day+1}/{total_test_days} (global time idx {test_day_idx})...")

        enc_start = test_day_idx - seq_len
        enc_end = test_day_idx
        if enc_start < 0:
            raise ValueError("Not enough history to build encoder input. Check seq_len and split_day.")

        enc_x_all = X_all[:, enc_start:enc_end, :].astype(np.float32)

        Nloc = enc_x_all.shape[0]
        dec_x_all = np.zeros((Nloc, label_len + out_len, dec_in), dtype=np.float32)

        dec_hist_start = enc_end - label_len
        dec_hist_end = enc_end
        dec_x_all[:, :label_len, 0] = Y_all[:, dec_hist_start:dec_hist_end, 0]

        preds_norm = predict_batchwise(model, enc_x_all, dec_x_all, device, bs=predict_batch_size)  # (N, out_len)

        preds_norm = preds_norm[:, 0]

        if is_normalized:
            preds = preds_norm * y_std + y_mean
        else:
            preds = preds_norm

        out_tif_name = os.path.join(output_dir, f"prediction_day{test_day_idx+1}.tif")
        save_prediction_as_tif(preds, postcodes, reference_tif, out_tif_name, nodata_value=NODATA_VALUE)

        true_norm = Y_all[:, test_day_idx, 0]
        if is_normalized:
            true_vals = true_norm * y_std + y_mean
        else:
            true_vals = true_norm

        metrics = evaluate_metrics(preds, true_vals)
        print("Metrics:", metrics)
        print("Saved:", out_tif_name)
        print("------")

    print("All done.")


if __name__ == "__main__":
    main()

