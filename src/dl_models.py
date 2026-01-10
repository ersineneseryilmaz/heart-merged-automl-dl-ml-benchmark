# src/dl_models.py
import argparse
import os
import time
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    precision_score, recall_score, log_loss, confusion_matrix
)

from .utils import set_seed, take_snapshot, diff_snapshot, model_file_size_mb, ensure_dir


class TabularCNN1D(nn.Module):
    def __init__(self, n_features: int, channels: int = 32, dropout: float = 0.15):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(channels, 32),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        x = x.unsqueeze(1)          # (B,1,F)
        h = self.conv(x)            # (B,C,F)
        h = self.pool(h).squeeze(-1) # (B,C)
        return self.head(h).squeeze(1)


class TabularRNNBinary(nn.Module):
    def __init__(self, n_features: int, rnn_type="gru", hidden_size=32, num_layers=1, dropout=0.0):
        super().__init__()
        rt = rnn_type.lower()
        self.rt = rt

        if rt == "rnn":
            self.rnn = nn.RNN(1, hidden_size, num_layers=num_layers, batch_first=True,
                              dropout=dropout if num_layers > 1 else 0.0)
        elif rt == "lstm":
            self.rnn = nn.LSTM(1, hidden_size, num_layers=num_layers, batch_first=True,
                               dropout=dropout if num_layers > 1 else 0.0)
        else:
            self.rnn = nn.GRU(1, hidden_size, num_layers=num_layers, batch_first=True,
                              dropout=dropout if num_layers > 1 else 0.0)

        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        x = x.unsqueeze(-1)         # (B,F,1)
        out = self.rnn(x)
        if self.rt == "lstm":
            seq_out, _ = out
        else:
            seq_out, _ = out
        last_h = seq_out[:, -1, :]
        return self.head(last_h).squeeze(1)


class AutoEncoder(nn.Module):
    def __init__(self, in_dim: int, latent_dim: int = 8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, in_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        xhat = self.decoder(z)
        return xhat


class AEClassifier(nn.Module):
    def __init__(self, encoder: nn.Module, latent_dim: int = 8):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.head(z).squeeze(1)


def compute_metrics(y_true, y_pred, y_proba):
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_proba)
    f1 = f1_score(y_true, y_pred)
    pr = precision_score(y_true, y_pred)
    rc = recall_score(y_true, y_pred)
    ll = log_loss(y_true, y_proba)
    cm = confusion_matrix(y_true, y_pred)
    return acc, auc, f1, pr, rc, ll, cm


def train_timebudget(
    model: nn.Module,
    X_tr, y_tr,
    X_val, y_val,
    X_train_full, y_train_full,
    X_test, y_test,
    budget_s: int,
    lr: float,
    batch_size: int,
    max_epochs: int,
    patience: int,
    finetune_epochs: int,
    device,
    out_path: str,
):
    snap0 = take_snapshot()
    t0 = time.time()

    model = model.to(device)
    crit = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    Xtr = torch.tensor(X_tr).to(device)
    ytr = torch.tensor(y_tr).float().to(device)
    Xva = torch.tensor(X_val).to(device)

    best_state = None
    best_f1 = -1.0
    bad = 0
    epochs_done = 0

    for epoch in range(1, max_epochs + 1):
        if time.time() - t0 >= budget_s:
            break

        model.train()
        perm = torch.randperm(Xtr.size(0), device=device)
        Xb = Xtr[perm]
        yb = ytr[perm]

        for i in range(0, Xb.size(0), batch_size):
            if time.time() - t0 >= budget_s:
                break
            xb = Xb[i:i+batch_size]
            ybb = yb[i:i+batch_size]
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, ybb)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            va_logits = model(Xva)
            va_prob = torch.sigmoid(va_logits).detach().cpu().numpy()
            va_pred = (va_prob >= 0.5).astype(int)
            va_f1 = f1_score(y_val, va_pred)

        epochs_done = epoch
        if va_f1 > best_f1 + 1e-4:
            best_f1 = va_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # finetune on full outer train (remaining time)
    Xfull = torch.tensor(X_train_full).to(device)
    yfull = torch.tensor(y_train_full).float().to(device)

    finetuned = 0
    for _ in range(finetune_epochs):
        if time.time() - t0 >= budget_s:
            break
        model.train()
        perm = torch.randperm(Xfull.size(0), device=device)
        Xb = Xfull[perm]
        yb = yfull[perm]
        for i in range(0, Xb.size(0), batch_size):
            if time.time() - t0 >= budget_s:
                break
            xb = Xb[i:i+batch_size]
            ybb = yb[i:i+batch_size]
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, ybb)
            loss.backward()
            opt.step()
        finetuned += 1

    model.eval()
    with torch.no_grad():
        Xte = torch.tensor(X_test).to(device)
        te_logits = model(Xte)
        te_prob = torch.sigmoid(te_logits).detach().cpu().numpy()
        te_pred = (te_prob >= 0.5).astype(int)

    acc, auc, f1, pr, rc, ll, cm = compute_metrics(y_test, te_pred, te_prob)

    ensure_dir(os.path.dirname(out_path) or ".")
    torch.save({
        "state_dict": model.state_dict(),
        "best_val_f1": best_f1,
        "epochs_done": epochs_done,
        "finetuned_epochs": finetuned,
    }, out_path)

    snap1 = take_snapshot()
    delta = diff_snapshot(snap0, snap1)
    size_mb = model_file_size_mb(out_path)

    return (acc, auc, f1, pr, rc, ll, cm, delta, size_mb, best_f1, epochs_done, finetuned)


def run_dl(data_path: str, mode: str, budget_s: int, seed: int, out_dir: str, model_name: str):
    ensure_dir(out_dir)
    set_seed(seed)
    torch.manual_seed(seed)

    df = pd.read_csv(data_path)
    ycol = "target"
    X = df.drop(columns=[ycol]).values.astype(np.float32)
    y = df[ycol].astype(int).values

    # outer split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    # inner split (selection)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=seed, stratify=y_train
    )

    # scale: fit only on outer train
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train).astype(np.float32)
    X_test_s = scaler.transform(X_test).astype(np.float32)
    X_tr_s = scaler.transform(X_tr).astype(np.float32)
    X_val_s = scaler.transform(X_val).astype(np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_dim = X_tr_s.shape[1]

    # mode configs
    if mode == "light":
        lr = 1e-3
        max_epochs = 60
        patience = 6
        finetune_epochs = 8
    else:
        lr = 8e-4
        max_epochs = 200
        patience = 15
        finetune_epochs = 20

    batch_size = 64

    # build model
    mname = model_name.upper()
    if mname == "CNN":
        ch = 24 if mode == "light" else 32
        model = TabularCNN1D(n_features=in_dim, channels=ch, dropout=0.15)
        out_path = os.path.join(out_dir, f"dl_cnn_{mode}_best.pt")
    elif mname in ("RNN", "GRU", "LSTM"):
        hidden = 24 if mode == "light" else 32
        rnn_type = "rnn" if mname == "RNN" else mname.lower()
        model = TabularRNNBinary(n_features=in_dim, rnn_type=rnn_type, hidden_size=hidden)
        out_path = os.path.join(out_dir, f"dl_{mname.lower()}_{mode}_best.pt")
    elif mname == "AE":
        # AE pretrain + clf
        # keep it simple but time-budgeted: pretrain within ~45% budget, then supervised within remaining
        snap0 = take_snapshot()
        t0 = time.time()

        ae = AutoEncoder(in_dim=in_dim, latent_dim=8).to(device)
        opt_ae = torch.optim.Adam(ae.parameters(), lr=lr)
        mse = nn.MSELoss()

        Xtr_t = torch.tensor(X_tr_s).to(device)

        # pretrain
        ae_epochs = 25 if mode == "light" else 70
        pretrain_limit = budget_s * 0.45
        ae.train()
        for _ in range(ae_epochs):
            if time.time() - t0 >= pretrain_limit:
                break
            perm = torch.randperm(Xtr_t.size(0), device=device)
            Xb = Xtr_t[perm]
            for i in range(0, Xb.size(0), batch_size):
                if time.time() - t0 >= pretrain_limit:
                    break
                xb = Xb[i:i+batch_size]
                opt_ae.zero_grad()
                xhat = ae(xb)
                loss = mse(xhat, xb)
                loss.backward()
                opt_ae.step()

        clf = AEClassifier(ae.encoder, latent_dim=8).to(device)
        crit = nn.BCEWithLogitsLoss()
        opt = torch.optim.Adam(clf.parameters(), lr=lr)

        Xtr = torch.tensor(X_tr_s).to(device)
        ytr = torch.tensor(y_tr).float().to(device)
        Xva = torch.tensor(X_val_s).to(device)

        best_state = None
        best_f1 = -1.0
        bad = 0
        epochs_done = 0

        clf_max_epochs = 60 if mode == "light" else 200
        while epochs_done < clf_max_epochs and (time.time() - t0) < budget_s:
            epochs_done += 1
            clf.train()
            perm = torch.randperm(Xtr.size(0), device=device)
            Xb = Xtr[perm]
            yb = ytr[perm]
            for i in range(0, Xb.size(0), batch_size):
                if time.time() - t0 >= budget_s:
                    break
                xb = Xb[i:i+batch_size]
                ybb = yb[i:i+batch_size]
                opt.zero_grad()
                logits = clf(xb)
                loss = crit(logits, ybb)
                loss.backward()
                opt.step()

            clf.eval()
            with torch.no_grad():
                va_logits = clf(Xva)
                va_prob = torch.sigmoid(va_logits).detach().cpu().numpy()
                va_pred = (va_prob >= 0.5).astype(int)
                va_f1 = f1_score(y_val, va_pred)

            if va_f1 > best_f1 + 1e-4:
                best_f1 = va_f1
                best_state = {k: v.detach().cpu().clone() for k, v in clf.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= patience:
                    break

        if best_state is not None:
            clf.load_state_dict(best_state)

        clf.eval()
        with torch.no_grad():
            Xte = torch.tensor(X_test_s).to(device)
            te_logits = clf(Xte)
            te_prob = torch.sigmoid(te_logits).detach().cpu().numpy()
            te_pred = (te_prob >= 0.5).astype(int)

        acc, auc, f1, pr, rc, ll, cm = compute_metrics(y_test, te_pred, te_prob)

        out_path = os.path.join(out_dir, f"dl_ae_{mode}_best.pt")
        torch.save({
            "ae_state": ae.state_dict(),
            "clf_state": clf.state_dict(),
            "best_val_f1": best_f1,
            "epochs_done": epochs_done,
            "seed": seed,
        }, out_path)

        snap1 = take_snapshot()
        delta = diff_snapshot(snap0, snap1)
        size_mb = model_file_size_mb(out_path)

        print(f"\n=== DL (PyTorch) AE+Head {mode.upper()} (Test) ===")
        print(f"Accuracy:  {acc*100:.2f}%")
        print(f"AUC:       {auc*100:.2f}%")
        print(f"F1 Score:  {f1*100:.2f}%")
        print(f"Precision: {pr*100:.2f}%")
        print(f"Recall:    {rc*100:.2f}%")
        print(f"LogLoss:   {ll:.2f}")
        print("\nConfusion Matrix:")
        print(cm)

        print("\n=== Resources ===")
        print(f"Runtime (wall): {delta.wall_time_s:.2f} s (Budget={budget_s}s)")
        print(f"Process RSS Δ:  {delta.proc_rss_delta_mb:.2f} MB")
        print(f"System RAM Δ:   {delta.sys_used_delta_gb:.2f} GB")
        print(f"Model size:     {size_mb:.4f} MB")
        print(f"Saved model:    {out_path}")
        return

    else:
        raise ValueError("model_name must be CNN/RNN/GRU/LSTM/AE")

    acc, auc, f1, pr, rc, ll, cm, delta, size_mb, best_val_f1, epochs_done, finetuned = train_timebudget(
        model=model,
        X_tr=X_tr_s, y_tr=y_tr,
        X_val=X_val_s, y_val=y_val,
        X_train_full=X_train_s, y_train_full=y_train,
        X_test=X_test_s, y_test=y_test,
        budget_s=budget_s,
        lr=lr,
        batch_size=batch_size,
        max_epochs=max_epochs,
        patience=patience,
        finetune_epochs=finetune_epochs,
        device=device,
        out_path=out_path
    )

    print(f"\n=== DL (PyTorch) {mname} {mode.upper()} (Test) ===")
    print(f"Accuracy:  {acc*100:.2f}%")
    print(f"AUC:       {auc*100:.2f}%")
    print(f"F1 Score:  {f1*100:.2f}%")
    print(f"Precision: {pr*100:.2f}%")
    print(f"Recall:    {rc*100:.2f}%")
    print(f"LogLoss:   {ll:.2f}")
    print("\nConfusion Matrix:")
    print(cm)

    print("\n=== Resources ===")
    print(f"Runtime (wall): {delta.wall_time_s:.2f} s (Budget={budget_s}s)")
    print(f"Process RSS Δ:  {delta.proc_rss_delta_mb:.2f} MB")
    print(f"System RAM Δ:   {delta.sys_used_delta_gb:.2f} GB")
    print(f"Model size:     {size_mb:.4f} MB")
    print(f"Saved model:    {out_path}")
    print(f"epochs_done={epochs_done} | finetuned_epochs={finetuned} | best_val_f1={best_val_f1:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--mode", choices=["light", "full"], required=True)
    ap.add_argument("--budget", type=int, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="outputs/dl")
    ap.add_argument("--model", choices=["CNN", "RNN", "GRU", "LSTM", "AE"], required=True)
    args = ap.parse_args()

    out_dir = os.path.join(args.out, args.model.lower(), args.mode)
    run_dl(args.data, args.mode, args.budget, args.seed, out_dir, args.model)


if __name__ == "__main__":
    main()
