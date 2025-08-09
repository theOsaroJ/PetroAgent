import math
import numpy as np
import torch
from torch import nn as tnn
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, f1_score
from .utils import detect_types

class TabTransformer(tnn.Module):
    def __init__(self, num_dim, cat_cardinalities, d_model=128, nhead=8, nlayers=3, dropout=0.1, task="regression", n_classes=2):
        super().__init__()
        self.task = task
        self.num_proj = tnn.Sequential(tnn.Linear(num_dim, d_model), tnn.LayerNorm(d_model), tnn.SiLU())
        self.cat_embs = tnn.ModuleList([tnn.Embedding(card+1, d_model) for card in cat_cardinalities])
        encoder_layer = tnn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, dropout=dropout, activation="gelu")
        self.encoder = tnn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        self.cls = tnn.Parameter(torch.zeros(1, 1, d_model))
        self.head = tnn.Sequential(tnn.LayerNorm(d_model), tnn.Linear(d_model, 1 if task=="regression" else n_classes))

    def forward(self, num_x, cat_x):
        B = num_x.size(0)
        num_tok = self.num_proj(num_x).unsqueeze(1)  # (B,1,d)
        cat_toks = []
        for i, emb in enumerate(self.cat_embs):
            cat_toks.append(emb(cat_x[:, i]))
        if cat_toks:
            cat_tok = torch.stack(cat_toks, dim=1)  # (B,C,d)
            seq = torch.cat([self.cls.expand(B, -1, -1), num_tok, cat_tok], dim=1)
        else:
            seq = torch.cat([self.cls.expand(B, -1, -1), num_tok], dim=1)
        enc = self.encoder(seq)[:, 0, :]  # CLS
        return self.head(enc)

def _to_tensor(X):
    # X here is raw dataframe-like (for transformer path)
    import pandas as pd
    if isinstance(X, pd.DataFrame):
        return X
    raise ValueError("Transformer expects raw DataFrame at this stage")

class TTWrapper:
    def __init__(self, model, meta, task):
        self.m = model
        self.meta = meta
        self.task = task

    def _split_num_cat(self, Xdf):
        import pandas as pd
        assert isinstance(Xdf, pd.DataFrame)
        num = torch.tensor(Xdf[self.meta["num_cols"]].to_numpy(dtype=np.float32), dtype=torch.float32) if self.meta["num_cols"] else torch.zeros((len(Xdf),1),dtype=torch.float32)
        cats = torch.zeros((len(Xdf), len(self.meta["cat_cols"])), dtype=torch.long)
        for i, c in enumerate(self.meta["cat_cols"]):
            # fit category map from training meta
            vocab = self.meta["cat_map"][c]
            cats[:, i] = torch.tensor([vocab.get(str(val), 0) for val in Xdf[c].astype(str)], dtype=torch.long)
        return num, cats

    def predict_sklearn_like(self, X, task, meta):
        import pandas as pd
        self.m.eval()
        with torch.no_grad():
            Xdf = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=meta["features"])
            num, cats = self._split_num_cat(Xdf)
            out = self.m(num, cats)
            if self.task == "regression":
                return out.squeeze(-1).cpu().numpy()
            else:
                return torch.argmax(out, dim=1).cpu().numpy()

def train_tabtransformer(splits, task="regression", meta=None, epochs=80, lr=2e-3, patience=8):
    import pandas as pd
    Xtr_df, Xval_df, Xte_df = splits["X_train"], splits["X_val"], splits["X_test"]
    # they are DataFrame here per prepare_xy(for_transformer=True)
    # Build categorical vocab
    cat_map = {}
    cat_card = []
    for c in meta["cat_cols"]:
        cats = pd.concat([Xtr_df[c], Xval_df[c]], axis=0).astype(str)
        uniq = sorted(cats.dropna().unique().tolist())
        vocab = {str(v): i+1 for i, v in enumerate(uniq)}  # 0 reserved for OOV
        cat_map[c] = vocab
        cat_card.append(len(vocab)+1)
    meta["cat_map"] = cat_map
    num_dim = len(meta["num_cols"]) if meta["num_cols"] else 1
    n_classes = None
    if task == "classification":
        n_classes = int(max(splits["y_train"].max(), splits["y_val"].max()) + 1)

    model = TabTransformer(num_dim=num_dim, cat_cardinalities=cat_card, d_model=160, nhead=8, nlayers=4, dropout=0.15, task=task, n_classes=n_classes or 2)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    if task == "regression":
        loss_fn = tnn.SmoothL1Loss()
    else:
        loss_fn = tnn.CrossEntropyLoss()

    def _prep(df):
        num = torch.tensor(df[meta["num_cols"]].to_numpy(dtype=np.float32), dtype=torch.float32) if meta["num_cols"] else torch.zeros((len(df),1),dtype=torch.float32)
        cats = torch.zeros((len(df), len(meta["cat_cols"])), dtype=torch.long)
        for i, c in enumerate(meta["cat_cols"]):
            vocab = meta["cat_map"][c]
            cats[:, i] = torch.tensor([vocab.get(str(val), 0) for val in df[c].astype(str)], dtype=torch.long)
        return num, cats

    Xtr_num, Xtr_cat = _prep(Xtr_df)
    Xval_num, Xval_cat = _prep(Xval_df)
    ytr = torch.tensor(splits["y_train"], dtype=torch.float32 if task=="regression" else torch.long)
    yval = torch.tensor(splits["y_val"], dtype=torch.float32 if task=="regression" else torch.long)

    best = math.inf
    bad = 0
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        out = model(Xtr_num, Xtr_cat)
        loss = loss_fn(out.squeeze(-1) if task=="regression" else out, ytr if task=="classification" else ytr.view(-1,1))
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            outv = model(Xval_num, Xval_cat)
            vloss = loss_fn(outv.squeeze(-1) if task=="regression" else outv, yval if task=="classification" else yval.view(-1,1)).item()

        if vloss < best - 1e-6:
            best = vloss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    model.load_state_dict(best_state)
    wrapper = TTWrapper(model, meta, task)
    preds = wrapper.predict_sklearn_like(Xte_df, task, meta)
    yte = splits["y_test"]
    if task == "regression":
        metrics = {"MAE": float(mean_absolute_error(yte, preds)), "R2": float(r2_score(yte, preds))}
    else:
        metrics = {"ACC": float(accuracy_score(yte, preds)), "F1": float(f1_score(yte, preds, average="macro"))}
    return wrapper, metrics
