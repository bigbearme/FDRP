import math
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float()
                             * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len_ = x.size(1)
        return x + self.pe[:, :seq_len_, :]


class InformerDataset(Dataset):
    def __init__(self, X_np, Y_np, seq_len=30, label_len=1, out_len=1, dec_in=1):
        self.X = X_np
        self.Y = Y_np
        self.seq_len = seq_len
        self.label_len = label_len
        self.out_len = out_len
        self.dec_in = dec_in
        self.N, self.T, self.F = X_np.shape

        max_start = self.T - (self.seq_len + self.out_len)
        if max_start < 0:
            raise ValueError("The time period is insufficient to build a sample. Please ensure T >= seq_len + out_len")
        self.samples = [(n, t) for n in range(self.N) for t in range(0, max_start + 1)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        n, t = self.samples[idx]
        enc_x = self.X[n, t : t + self.seq_len, :].astype(np.float32)
        dec_x = np.zeros((self.label_len + self.out_len, self.dec_in), dtype=np.float32)
        dec_x[:self.label_len, 0] = self.Y[n, t + self.seq_len - self.label_len : t + self.seq_len, 0]
        label = self.Y[n, t + self.seq_len : t + self.seq_len + self.out_len, 0].astype(np.float32)
        return torch.from_numpy(enc_x).float(), torch.from_numpy(dec_x).float(), torch.from_numpy(label).float()


class InformerModel(nn.Module):
    def __init__(self, enc_in, dec_in, c_out=1,
                 seq_len=30, label_len=1, out_len=1,
                 d_model=128, nhead=4, e_layers=2, d_layers=1, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.label_len = label_len
        self.out_len = out_len

        self.enc_embedding = nn.Linear(enc_in, d_model)
        self.dec_embedding = nn.Linear(dec_in, d_model)

        self.pos_enc = PositionalEncoding(d_model, max_len=2000)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=d_model*4,
            dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=e_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model, nhead, dim_feedforward=d_model*4,
            dropout=dropout, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=d_layers)

        self.proj = nn.Linear(d_model, c_out)

        self.enc_norm = nn.LayerNorm(d_model)
        self.dec_norm = nn.LayerNorm(d_model)

    def forward(self, enc_x, dec_x):
        enc = self.enc_embedding(enc_x) * math.sqrt(self.d_model)
        dec = self.dec_embedding(dec_x) * math.sqrt(self.d_model)

        enc = self.enc_norm(enc)
        dec = self.dec_norm(dec)

        enc = self.pos_enc(enc)
        dec = self.pos_enc(dec)

        memory = self.encoder(enc)
        out = self.decoder(tgt=dec, memory=memory)

        pred = self.proj(out[:, -self.out_len:, :])  # (B, out_len, c_out)
        return pred
