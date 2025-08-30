import os
from dataload import load_data
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import InformerModel, InformerDataset

# ========== CONFIG ==========
data_dir = "Data"
rainfall_dir = os.path.join(data_dir, "daily rainfall")
risk_dir = "Risk"
postcode_path = "PostCode.txt"
seq_len = 30
label_len = 1
out_len = 1
output_dir = "PredictedRisk"
os.makedirs(output_dir, exist_ok=True)

d_model = 128
nhead = 4
e_layers = 2
d_layers = 1
dropout = 0.1
train_epochs = 10
train_batch_size = 256
predict_batch_size = 1024
lr = 1e-3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# checkpoint path
ckpt_dir = "checkpoints"
os.makedirs(ckpt_dir, exist_ok=True)
ckpt_path = os.path.join(ckpt_dir, "epochckpt.pt")

# nodata value used when saving GeoTIFFs
NODATA_VALUE = -9999.0

# ========== DATA LOAD ==========
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
X_train_np = data["X_train_np"]
Y_train_np = data["Y_train_np"]
X_test_np = data["X_test_np"]
Y_test_np = data["Y_test_np"]
postcodes = data["postcodes"]
rainfall_files = data["rainfall_files"]
feature_dirs = data["feature_dirs"]
split_day = data["split_day"]
norm = data["norm"]

X_tensor = torch.from_numpy(X_all).float()
Y_tensor = torch.from_numpy(Y_all).float()


# ========== TRAIN ==========
def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for enc_x, dec_x, label in dataloader:
        enc_x = enc_x.to(device)
        dec_x = dec_x.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        pred = model(enc_x, dec_x)
        pred = pred.squeeze(-1)
        loss = criterion(pred, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def save_checkpoint(path, model, meta, norm):
    ckpt = {
        "model_state_dict": model.state_dict(),
        "meta": meta,
        "norm": norm
    }
    torch.save(ckpt, path)

# ========== MAIN ==========
def main():
    print("[TRAIN] Start training")
    enc_in = X_all.shape[2]
    dec_in = 1

    meta = {
        "enc_in": int(enc_in),
        "dec_in": int(dec_in),
        "c_out": 1,
        "seq_len": int(seq_len),
        "label_len": int(label_len),
        "out_len": int(out_len),
        "d_model": int(d_model),
        "nhead": int(nhead),
        "e_layers": int(e_layers),
        "d_layers": int(d_layers),
        "dropout": float(dropout),
        "split_day": int(split_day),
        "feature_dirs": feature_dirs,
        "rainfall_files": rainfall_files
    }

    model = InformerModel(
        enc_in=enc_in, dec_in=dec_in, c_out=1,
        seq_len=seq_len, label_len=label_len, out_len=out_len,
        d_model=d_model, nhead=nhead, e_layers=e_layers, d_layers=d_layers, dropout=dropout
    ).to(device)

    train_dataset = InformerDataset(X_train_np, Y_train_np, seq_len=seq_len, label_len=label_len, out_len=out_len,
                                    dec_in=dec_in)
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, drop_last=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(train_epochs):
        loss = train_epoch(model, train_loader, optimizer, criterion, device)
        print(f"[TRAIN] Epoch {epoch + 1}/{train_epochs} - Loss: {loss:.6f}")

    save_checkpoint(ckpt_path, model, meta, norm)
    print(f"[TRAIN] Saved checkpoint -> {ckpt_path}")


if __name__ == "__main__":
    main()
