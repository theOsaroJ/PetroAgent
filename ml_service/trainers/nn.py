import math
import numpy as np
import torch
from torch import nn as tnn
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, f1_score

class MLP(tnn.Module):
    def __init__(self, in_dim, out_dim, task):
        super().__init__()
        hidden = [max(128, in_dim*2), 256, 128]
        act = tnn.SiLU()
        layers = []
        last = in_dim
        for h in hidden:
            layers += [tnn.Linear(last, h), tnn.LayerNorm(h), act, tnn.Dropout(0.15)]
            last = h
        layers += [tnn.Linear(last, out_dim)]
        self.net = tnn.Sequential(*layers)
        self.task = task

    def forward(self, x):
        return self.net(x)

def _to_tensor(X):
    if isinstance(X, np.ndarray):
        return torch.tensor(X, dtype=torch.float32)
    return torch.tensor(np.asarray(X), dtype=torch.float32)

class TorchSklearnWrapper:
    def __init__(self, model, task):
        self.m = model
        self.task = task

    def predict_sklearn_like(self, X, task, meta):
        self.m.eval()
        with torch.no_grad():
            X_t = _to_tensor(X)
            out = self.m(X_t)
            if self.task == "regression":
                return out.squeeze(-1).cpu().numpy()
            else:
                return torch.argmax(out, dim=1).cpu().numpy()

def train_nn(splits, task="regression", meta=None, epochs=100, lr=3e-3, patience=10):
    Xtr, Xval, Xte = splits["X_train"], splits["X_val"], splits["X_test"]
    ytr, yval, yte = splits["y_train"], splits["y_val"], splits["y_test"]

    Xtr_t = _to_tensor(Xtr); Xval_t = _to_tensor(Xval); Xte_t = _to_tensor(Xte)
    if task == "regression":
        ytr_t = torch.tensor(ytr, dtype=torch.float32).view(-1, 1)
        yval_t = torch.tensor(yval, dtype=torch.float32).view(-1, 1)
        out_dim = 1
        loss_fn = tnn.SmoothL1Loss()
    else:
        # infer num classes from validation set too
        n_classes = int(max(ytr.max(), yval.max()) + 1)
        ytr_t = torch.tensor(ytr, dtype=torch.long)
        yval_t = torch.tensor(yval, dtype=torch.long)
        out_dim = n_classes
        loss_fn = tnn.CrossEntropyLoss()

    model = MLP(in_dim=Xtr_t.shape[1], out_dim=out_dim, task=task)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    best_val = math.inf
    bad = 0
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        out = model(Xtr_t)
        loss = loss_fn(out, ytr_t)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            outv = model(Xval_t)
            vloss = loss_fn(outv, yval_t).item()
        if vloss < best_val - 1e-6:
            best_val = vloss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    model.load_state_dict(best_state)
    wrapper = TorchSklearnWrapper(model, task)
    preds = wrapper.predict_sklearn_like(Xte, task, meta)
    if task == "regression":
        metrics = {"MAE": float(mean_absolute_error(yte, preds)), "R2": float(r2_score(yte, preds))}
    else:
        metrics = {"ACC": float(accuracy_score(yte, preds)), "F1": float(f1_score(yte, preds, average="macro"))}
    return wrapper, metrics
