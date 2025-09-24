# ============================================================
# SAS RL — Contextual τ (no frac, no caps) + Improved ŝ1 Predictor
# ============================================================
# Highlights
#   • Per-system, per-basis τ via a small contextual net (no fractions, no caps)
#   • Improved ŝ1 predictor (Res-MLP with LayerNorm, SiLU, Dropout, Skip)
#   • Clean PPO-style τ training (no .item() leaks), fixed requires_grad bug
#   • Same IO and ASF flow you’re used to; parity/min-active guards only
# ------------------------------------------------------------

import os, re, math, argparse, random, sys, csv, json, hashlib, warnings, logging
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# PySCF (quantum chemistry backend)
from pyscf import gto, scf, cc, mcscf, mp
from pyscf.scf import stability as stab

HARTREE2EV = 27.211386245988

# -------------------------- Logging --------------------------
def setup_logger(log_file: str, debug: bool):
    logger = logging.getLogger("sas_rl_ctx")
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG if debug else logging.INFO)
    ch.setFormatter(fmt)
    fh = logging.FileHandler(log_file, mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(ch); logger.addHandler(fh)
    logging.getLogger("pyscf").setLevel(logging.WARNING)
    return logger

# -------------------------- JSON helpers --------------------------
def _to_jsonable(obj):
    import numpy as _np
    if isinstance(obj, (str, int, float, bool)) or obj is None: return obj
    if isinstance(obj, (_np.integer,)):  return int(obj)
    if isinstance(obj, (_np.floating,)):
        v = float(obj)
        if not math.isfinite(v): return None
        return v
    if isinstance(obj, (_np.bool_,)):    return bool(obj)
    if isinstance(obj, (_np.ndarray,)):
        return [_to_jsonable(x) for x in obj.tolist()]
    try:
        import torch as _t
        if isinstance(obj, _t.Tensor):
            return _to_jsonable(obj.detach().cpu().numpy())
    except Exception:
        pass
    if isinstance(obj, (list, tuple)): return [_to_jsonable(x) for x in obj]
    if isinstance(obj, dict): return {str(k): _to_jsonable(v) for k, v in obj.items()}
    return obj

# -------------------------- IO --------------------------
def normalize_elem(sym: str) -> str:
    return (sym[:1].upper() + sym[1:].lower()) if sym else sym

def sanitize_name(name: str) -> str:
    name = (name or "").strip()
    name = re.split(r"[,\s;]+", name, maxsplit=1)[0]
    return name

def read_multimol_xyz(xyz_path) -> List[Dict]:
    frames = {}; order=[]
    with open(xyz_path, 'r', encoding='utf-8') as f:
        while True:
            line = f.readline()
            if not line: break
            line = line.strip()
            if not line: continue
            try: nat = int(line)
            except ValueError: continue
            raw_comment = (f.readline().rstrip("\n") or "").strip()
            name = sanitize_name(raw_comment) or f"MOL_{len(order)}"
            atoms, coords = [], []
            for _ in range(nat):
                sp, x, y, z = f.readline().split()
                atoms.append(normalize_elem(sp))
                coords.append([float(x), float(y), float(z)])
            frames[len(order)] = {'name': name, 'atoms': atoms, 'coords': np.array(coords, float)}
            order.append(len(order))
    return [frames[i] for i in order]

# -------------------------- Parse descriptors / s1 --------------------------
_mol_hdr_re      = re.compile(r"^=+\s*Molecule\s*#\s*(\d+).*?spin\s*=\s*([-\d\.]+)\s*charge\s*=\s*([-\d\.]+)\s*=+")
_mo_start_re     = re.compile(r"^-{5,}\s*MO\s*(\d+)\s*:")
_kv_energy       = re.compile(r"^\s*Energy\s*=\s*([-\d\.Ee+]+)")
_kv_scaler       = re.compile(r"^\s*(h1_scaler|g2_scaler)\s*=\s*([-\d\.Ee+]+)")
_kv_r2           = re.compile(r"^\s*⟨r²⟩\s*=\s*([-\d\.Ee+]+)")
_kv_dip          = re.compile(r"^\s*Dipole\s*=\s*\[([^\]]+)\]")
_kv_label        = re.compile(r"^\s*(Occ_label|Bond_label)\s*=\s*([-\d\.Ee+]+)")
_kv_intvec       = re.compile(r"^\s*AO_encoding\s*=\s*\[([^\]]+)\]")
_kv_apc_sum      = re.compile(r"^\s*APC_sum\s*=\s*([-\d\.Ee+]+)")
_kv_apc_avg      = re.compile(r"^\s*APC_avg\s*=\s*([-\d\.Ee+]+)")
_kv_soft_sum     = re.compile(r"^\s*soft_APC_sum\s*=\s*([-\d\.Ee+]+)")
_kv_soft_avg     = re.compile(r"^\s*soft_APC_avg\s*=\s*([-\d\.Ee+]+)")

def _parse_vec(s):
    return [float(x) for x in re.split(r"[,\s]+", s.strip()) if x]

def parse_orbital_descriptors(txt_path):
    mols = []; cur=None; cur_mo=None
    def commit_mo():
        nonlocal cur, cur_mo
        if not cur or not cur_mo: return
        row = []
        row += [cur_mo.get('energy', 0.0) or 0.0]
        row += [cur_mo.get('h1_scaler',0.0) or 0.0,
                cur_mo.get('g2_scaler',0.0) or 0.0,
                cur_mo.get('r2',0.0) or 0.0]
        dip = cur_mo.get('dipole',[0,0,0]) or [0,0,0]
        dip = (dip + [0,0,0])[:3]; row += dip
        row += [cur_mo.get('occ_label',0.0) or 0.0,
                cur_mo.get('bond_label',0.0) or 0.0]
        enc = cur_mo.get('ao_encoding',[0]*15) or [0]*15
        if len(enc)<15: enc = enc + [0]*(15-len(enc))
        row += [float(x) for x in enc[:15]]
        row += [cur_mo.get('apc_sum', 0.0) or 0.0,
                cur_mo.get('apc_avg', 0.0) or 0.0,
                cur_mo.get('soft_sum',0.0) or 0.0,
                cur_mo.get('soft_avg',0.0) or 0.0]
        cur['rows'].append(np.array(row,float)); cur_mo=None
    with open(txt_path,'r',encoding='utf-8') as f:
        for raw in f:
            line = raw.rstrip("\n")
            m = _mol_hdr_re.match(line)
            if m:
                if cur is not None:
                    commit_mo()
                    cur['node_feat'] = np.vstack(cur['rows']) if cur['rows'] else np.zeros((0,28),float)
                    del cur['rows']; mols.append(cur)
                cur={'rows':[]}; cur_mo=None; continue
            m = _mo_start_re.match(line)
            if m:
                commit_mo()
                cur_mo={'energy':None,'h1_scaler':None,'g2_scaler':None,'r2':None,'dipole':None,
                        'occ_label':None,'bond_label':None,'ao_encoding':None,
                        'apc_sum':None,'apc_avg':None,'soft_sum':None,'soft_avg':None}
                continue
            if cur is None: continue
            m=_kv_energy.match(line)
            if m: cur_mo['energy']=float(m.group(1)); continue
            m=_kv_scaler.match(line)
            if m:
                key,val=m.group(1),float(m.group(2))
                cur_mo['h1_scaler' if key=='h1_scaler' else 'g2_scaler']=val; continue
            m=_kv_r2.match(line)
            if m: cur_mo['r2']=float(m.group(1)); continue
            m=_kv_dip.match(line)
            if m: cur_mo['dipole']=_parse_vec(m.group(1)); continue
            m=_kv_label.match(line)
            if m:
                key,val=m.group(1),float(m.group(2))
                cur_mo['occ_label' if key=='Occ_label' else 'bond_label']=val; continue
            m=_kv_intvec.match(line)
            if m: cur_mo['ao_encoding']=[int(float(x)) for x in re.split(r"[,\s]+", m.group(1).strip()) if x]; continue
            m=_kv_apc_sum.match(line)
            if m: cur_mo['apc_sum']=float(m.group(1)); continue
            m=_kv_apc_avg.match(line)
            if m: cur_mo['apc_avg']=float(m.group(1)); continue
            m=_kv_soft_sum.match(line)
            if m: cur_mo['soft_sum']=float(m.group(1)); continue
            m=_kv_soft_avg.match(line)
            if m: cur_mo['soft_avg']=float(m.group(1)); continue
            if not line.strip(): commit_mo()
        if cur is not None:
            commit_mo()
            cur['node_feat']=np.vstack(cur['rows']) if cur['rows'] else np.zeros((0,28),float)
            del cur['rows']; mols.append(cur)
    return mols

# ----- s1 labels -----
_s1_mol_hdr = re.compile(r"^=+\s*Molecule\s*#\s*(\d+).*?spin\s*=\s*([-\d\.]+)\s*charge\s*=\s*([-\d\.]+)\s*=+")
_s1_line    = re.compile(r"^\s*Orbital\s+(\d+):\s*s1\s*=\s*([-\d\.Ee+]+)")
def parse_s1_entropies(txt_path, n_expected: int) -> List[np.ndarray]:
    arrs=[]; cur=None
    with open(txt_path,'r',encoding='utf-8') as f:
        for raw in f:
            line=raw.strip()
            m=_s1_mol_hdr.match(line)
            if m:
                if cur is not None:
                    maxk = max(cur['vals'].keys()) if cur['vals'] else -1
                    s1 = np.zeros(maxk+1, float)
                    for k,v in cur['vals'].items(): s1[k]=v
                    arrs.append(s1)
                cur={'vals':{}}
                continue
            m=_s1_line.match(line)
            if m and cur is not None:
                idx=int(m.group(1)); val=float(m.group(2))
                cur['vals'][idx]=val
    if cur is not None:
        maxk = max(cur['vals'].keys()) if cur['vals'] else -1
        s1 = np.zeros(maxk+1, float)
        for k,v in cur['vals'].items(): s1[k]=v
        arrs.append(s1)
    if len(arrs) != n_expected:
        raise ValueError(f"[s1] Count {len(arrs)} != expected {n_expected}")
    return arrs

# -------------------------- Dataset --------------------------
@dataclass
class MolRecord:
    idx: int
    name: str
    atoms: List[str]
    coords: np.ndarray
    E_dmrg: float
    x_feat: np.ndarray
    s1: np.ndarray
    charge: int
    spin2S: int
    basis: str
    r2: Optional[np.ndarray]
    local_id: int = -1

class S1Dataset(Dataset):
    def __init__(self, xyz_path, descriptors_txt, s1_txt, logger,
                 dmrg_path: Optional[str] = None, dmrg_unit: str = 'ev',
                 default_basis='sto-3g', charge_json: Optional[str] = None, spin_json: Optional[str] = None):
        frames = read_multimol_xyz(xyz_path)
        desc   = parse_orbital_descriptors(descriptors_txt)
        if len(frames) != len(desc):
            raise ValueError(f"Count mismatch: XYZ({len(frames)}) vs DESC({len(desc)})")
        N = len(frames)
        names = [fr['name'] for fr in frames]

        def _load_name_ints(path: str, names: List[str], what: str) -> List[int]:
            if not path or not os.path.isfile(path):
                raise ValueError(f"{what}: file not found: {path}")
            with open(path, 'r', encoding='utf-8') as fh:
                data = json.load(fh)
            if isinstance(data, list):
                if len(data) != len(names): raise ValueError(f"{what}: list len {len(data)} != dataset {len(names)}")
                return [int(v) for v in data]
            if not isinstance(data, dict): raise ValueError(f"{what}: JSON must be dict or list")
            def norm_key(s: str) -> str: return re.sub(r"[^A-Za-z0-9]+", "", (s or "")).upper()
            json_norm = {norm_key(k): int(v) for k, v in data.items()}
            names_norm = [norm_key(sanitize_name(nm)) for nm in names]
            out: List[Optional[int]] = [None] * len(names_norm)
            for i, nk in enumerate(names_norm):
                if nk in json_norm: out[i] = json_norm[nk]
            missing = [i for i,v in enumerate(out) if v is None]
            if missing:
                preview = ", ".join([names[i] for i in missing][:10])
                raise ValueError(f"{what}: missing entries for: {preview}")
            return [int(v) for v in out]

        if not charge_json or not spin_json:
            raise ValueError("Both --charge_json and --spin_json are required and must map molecule NAMES to integers.")

        charge_by_name = _load_name_ints(charge_json, names, "charge_json")
        spin_by_name   = _load_name_ints(spin_json,   names, "spin_json")

        def read_dmrg_or_zeros(path: Optional[str], n: int, unit: str) -> np.ndarray:
            if not path or (path and not os.path.isfile(path)):
                if path: logger.warning(f"[DMRG] File not found: {path}; using zeros.")
                else:    logger.warning("[DMRG] Not provided; using zeros.")
                return np.zeros(n, float)
            vals=[]
            with open(path,'r',encoding='utf-8') as f:
                for line in f:
                    s=line.strip()
                    if not s or s.startswith('#'): continue
                    vals.append(float(s.split()[0]))
            arr=np.array(vals,float)
            if len(arr)!=n: raise ValueError(f"DMRG count {len(arr)} != XYZ count {n}")
            u = unit.lower()
            if u=='ev': arr = arr / HARTREE2EV
            elif u in ('hartree','ha','au'): pass
            else: raise ValueError(f"Unknown DMRG unit: {unit}")
            return arr

        dmrg_Ha = read_dmrg_or_zeros(dmrg_path, N, dmrg_unit) if dmrg_path or dmrg_path=="" else np.zeros(N,float)
        if s1_txt and os.path.isfile(s1_txt): s1_list = parse_s1_entropies(s1_txt, N)
        else:
            s1_list = []
            for d in desc:
                nmo = int(d['node_feat'].shape[0])
                s1_list.append(np.zeros((nmo,), dtype=float))

        self.records=[]
        for i, (fr, d, s1) in enumerate(zip(frames, desc, s1_list)):
            x = d['node_feat']
            if s1.shape[0] != x.shape[0]:
                raise ValueError(f"[s1] length {s1.shape[0]} != nmo {x.shape[0]} at molecule idx={i}")
            r2 = x[:,3].astype(np.float64) if x.shape[1] >= 4 else None
            self.records.append(MolRecord(
                idx=i, name=fr['name'],
                atoms=fr['atoms'], coords=fr['coords'],
                E_dmrg=float(dmrg_Ha[i]) if (dmrg_path is not None) else 0.0,
                x_feat=x.astype(np.float32), s1=s1.astype(np.float32),
                charge=int(charge_by_name[i]), spin2S=int(abs(spin_by_name[i])),
                basis=default_basis, r2=r2
            ))

    def __len__(self): return len(self.records)
    def __getitem__(self, idx): return self.records[idx]

def reindex_local_ids(ds):
    for k, r in enumerate(ds.records): r.local_id = k
    return ds

def subset_dataset(ds_full, indices: List[int]) -> S1Dataset:
    sub = S1Dataset.__new__(S1Dataset)
    sub.records = [ds_full.records[i] for i in indices]
    reindex_local_ids(sub)
    return sub

def collate(batch: List[MolRecord]):
    B=len(batch); Ns=[len(b.s1) for b in batch]; Nmax=max(Ns)
    F = batch[0].x_feat.shape[1] if Nmax>0 else 28
    X = torch.zeros(B, Nmax, F, dtype=torch.float32)
    S1 = torch.zeros(B, Nmax, dtype=torch.float32)
    mask = torch.zeros(B, Nmax, dtype=torch.bool)
    E_dmrg = torch.zeros(B, dtype=torch.float32)
    r2 = torch.zeros(B, Nmax, dtype=torch.float32)
    metas=[]
    for k,b in enumerate(batch):
        n=len(b.s1)
        X[k,:n,:]=torch.from_numpy(b.x_feat[:n,:])
        S1[k,:n]=torch.from_numpy(b.s1[:n])
        mask[k,:n]=True
        E_dmrg[k]=b.E_dmrg
        if b.r2 is not None: r2[k,:n]=torch.from_numpy(b.r2.astype(np.float32))
        metas.append({'idx':b.idx,'name':b.name,'atoms':b.atoms,'coords':b.coords,
                      'charge':b.charge,'spin2S':b.spin2S,'basis':b.basis})
    return {'x':X,'s1':S1,'mask':mask,'E_dmrg':E_dmrg,'r2':r2,'meta':metas}

# -------------------------- Scaling --------------------------
class FeatureScaler:
    def __init__(self): self.mu=None; self.sd=None
    def fit(self, ds_train):
        xs=[rec.x_feat.astype(np.float64) for rec in ds_train.records]
        arrs = [x.reshape(-1, x.shape[-1]) for x in xs if hasattr(x, "size") and x.size > 0]
        if len(arrs) == 0:
            F = (ds_train.records[0].x_feat.shape[1] if getattr(ds_train, "records", None) and ds_train.records else 28)
            self.mu = np.zeros((F,), float); self.sd = np.ones((F,), float)
            return
        X = np.vstack(arrs)
        self.mu = X.mean(axis=0); self.sd = X.std(axis=0)
        self.sd[self.sd < 1e-12] = 1.0
    def transform(self, X: torch.Tensor) -> torch.Tensor:
        if self.mu is None or self.sd is None: return X
        mu = torch.as_tensor(self.mu, dtype=X.dtype, device=X.device).view(1,1,-1)
        sd = torch.as_tensor(self.sd, dtype=X.dtype, device=X.device).view(1,1,-1)
        return (X - mu) / sd

class TargetTransform:
    def __init__(self, kind: str = "logit01", eps: float = 1e-4):
        self.kind = kind; self.eps = float(eps)
        self.mu = None; self.sd = None; self.ymin = None; self.ymax = None
    def fit(self, ds_train):
        ys=[rec.s1.astype(np.float64).reshape(-1) for rec in ds_train.records]
        y = np.concatenate(ys, axis=0) if ys else np.zeros((0,), float)
        if y.size == 0: y = np.array([0.0, 1.0], float)
        if self.kind == "zscore":
            self.mu = float(np.mean(y)); self.sd = float(np.std(y) if np.std(y)>1e-12 else 1.0)
        elif self.kind == "logit01":
            self.ymin = float(np.min(y)); self.ymax = float(np.max(y))
            if not np.isfinite(self.ymax - self.ymin) or (self.ymax - self.ymin) < 1e-9:
                self.ymin = float(np.min(y) - 1e-3); self.ymax = float(np.max(y) + 1e-3)
    def transform_torch(self, y: torch.Tensor) -> torch.Tensor:
        if self.kind == "zscore":
            mu = float(self.mu if self.mu is not None else 0.0)
            sd = float(self.sd if (self.sd is not None and self.sd > 1e-12) else 1.0)
            return (y - mu) / sd
        if self.kind == "log1p":  return torch.log1p(y.clamp_min(0.0))
        if self.kind == "sqrt":   return torch.sqrt(y.clamp_min(0.0))
        if self.kind == "logit01":
            ymin = float(0.0 if self.ymin is None else self.ymin)
            ymax = float(1.0 if self.ymax is None else self.ymax)
            denom = max(float(ymax - ymin), 1e-12)
            y01 = (y - ymin) / denom
            y01 = y01.clamp(self.eps, 1.0 - self.eps)
            return torch.log(y01 / (1.0 - y01))
        return y
    def inverse_torch(self, t: torch.Tensor) -> torch.Tensor:
        if self.kind == "zscore":
            mu = float(self.mu if self.mu is not None else 0.0)
            sd = float(self.sd if (self.sd is not None and self.sd > 1e-12) else 1.0)
            return t * sd + mu
        if self.kind == "log1p":  return torch.expm1(t)
        if self.kind == "sqrt":   return (t.clamp_min(0.0))**2
        if self.kind == "logit01":
            ymin = float(0.0 if self.ymin is None else self.ymin)
            ymax = float(1.0 if self.ymax is None else self.ymax)
            return torch.sigmoid(t) * (ymax - ymin) + ymin
        return t
    def to_state(self) -> Dict[str, Any]:
        return {"kind": self.kind, "eps": self.eps, "mu": self.mu, "sd": self.sd, "ymin": self.ymin, "ymax": self.ymax}
    @staticmethod
    def from_state(st: Dict[str,Any]) -> "TargetTransform":
        tt = TargetTransform(kind=st.get("kind","logit01"), eps=st.get("eps",1e-4))
        tt.mu = st.get("mu", None); tt.sd = st.get("sd", None)
        tt.ymin = st.get("ymin", None); tt.ymax = st.get("ymax", None)
        return tt

# -------------------------- Improved ŝ1 predictor --------------------------
class ResBlock(nn.Module):
    def __init__(self, d_in, d_hidden, dropout=0.0):
        super().__init__()
        self.lin1 = nn.Linear(d_in, d_hidden)
        self.lin2 = nn.Linear(d_hidden, d_in)
        self.act  = nn.SiLU()
        self.norm = nn.LayerNorm(d_in)
        self.drop = nn.Dropout(dropout) if dropout>0 else nn.Identity()
        # Squeeze-Excite style gate
        self.gate = nn.Sequential(
            nn.Linear(d_in, max(8, d_in//4)),
            nn.SiLU(),
            nn.Linear(max(8, d_in//4), d_in),
            nn.Sigmoid()
        )
    def forward(self, x):
        y = self.lin1(self.norm(x))
        y = self.act(y)
        y = self.lin2(y)
        y = self.drop(y)
        g = self.gate(x)
        return x + g * y

class S1Predictor(nn.Module):
    """Token-wise MLP with residual blocks."""
    def __init__(self, in_dim: int, width: int = 384, depth: int = 6, dropout: float = 0.1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, width),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList([ResBlock(width, width*2, dropout=dropout) for _ in range(depth)])
        self.head = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, 1)
        )
    def forward(self, x):  # x: (B,N,F)
        B,N,F = x.shape
        y = self.stem(x)
        for blk in self.blocks: y = blk(y)
        out = self.head(y).view(B, N)
        return out

# -------------------------- τ policy (contextual) --------------------------
def _safe_tau_value(tau_val: float, lo: float, hi: float) -> float:
    if not np.isfinite(tau_val): return float(lo)
    return float(min(max(tau_val, lo), hi))

class TauPolicyContextual(nn.Module):
    """
    Predict a literal τ threshold from simple context:
      ctx = [N, s1_mean_pred, s1_std_pred, s1_p75, s1_p90, charge, 2S, r2_mean_pred]
    + a small basis embedding.
    Output τ in [tau_lower_hard, tau_upper_hard] via sigmoid squashing.
    """
    def __init__(self, basis_vocab: List[str], hidden: int = 64, tau_lo: float = 0.02, tau_hi: float = 0.30):
        super().__init__()
        self.basis_to_idx = {b:i for i,b in enumerate(sorted(set(basis_vocab or ["sto-3g"])))}
        self.emb = nn.Embedding(num_embeddings=len(self.basis_to_idx), embedding_dim=16)
        self.net = nn.Sequential(
            nn.Linear(8 + 16, hidden),
            nn.SiLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1)
        )
        self.tau_lo = float(tau_lo)
        self.tau_hi = float(tau_hi)

    def forward(self, ctx: torch.Tensor, basis_str: List[str]):
        # ctx: (B, 8) ; basis list length B
        idx = torch.tensor([self.basis_to_idx.get(b.lower(), 0) for b in basis_str],
                           dtype=torch.long, device=ctx.device)
        e = self.emb(idx)  # (B,16)
        h = torch.cat([ctx, e], dim=-1)
        z = self.net(h).squeeze(-1)  # (B,)
        tau01 = torch.sigmoid(z)
        tau = self.tau_lo + (self.tau_hi - self.tau_lo) * tau01
        return tau  # (B,)

# -------------------------- Chemistry helpers --------------------------
@dataclass
class ScfRef:
    mol: Any
    mf: Any
    E_scf: float
    mo_a: np.ndarray
    mo_b: np.ndarray
    occ_a: np.ndarray
    occ_b: np.ndarray
    nmo: int
    is_uhf_obj: bool
    spin2S: int

def one_shot_internal_stability(mf):
    try:
        if isinstance(mf, scf.rohf.ROHF):
            try:
                mo_i, _mo_e, stable_i, _ = stab.rohf_stability(mf, internal=True, external=False, return_status=True)
            except AttributeError:
                mo_i, _mo_e, stable_i, _ = stab.stability(mf, internal=True, external=False, return_status=True)
        elif isinstance(mf, scf.rhf.RHF):
            mo_i, _mo_e, stable_i, _ = stab.rhf_stability(mf, internal=True, external=False, return_status=True)
        else:
            try:
                mo_i, _mo_e, stable_i, _ = stab.uhf_stability(mf, internal=True, external=False, return_status=True)
            except AttributeError:
                mo_i, _mo_e, stable_i, _ = stab.stability(mf, internal=True, external=False, return_status=True)
    except Exception:
        return mf
    if bool(stable_i) or mo_i is None:
        return mf
    dm0 = mf.make_rdm1(mo_i, mf.mo_occ)
    if isinstance(mf, scf.rohf.ROHF):  mf_new = scf.ROHF(mf.mol)
    elif isinstance(mf, scf.rhf.RHF):   mf_new = scf.RHF(mf.mol)
    else:                                mf_new = scf.UHF(mf.mol)
    mf_new.conv_tol = 1e-12; mf_new.max_cycle = 60; mf_new.diis = scf.diis.CDIIS()
    mf_new.kernel(dm0=dm0)
    return mf_new

def _parity_fix_if_needed(mol, charge: int, spin2S: int, atoms, coords, basis: str, logger):
    ne = mol.nelectron
    if (ne & 1) == (spin2S & 1): return mol, spin2S
    old = spin2S
    cand1 = max(0, old - 1); cand2 = old + 1
    newS = cand1 if ((cand1 & 1) == (ne & 1)) else cand2
    logger.warning(f"[Parity fix] nelec={ne}, requested 2S={old} -> {newS} to match parity.")
    mol2 = gto.M(atom=[[a, tuple(c)] for a, c in zip(atoms, coords)],
                 basis=basis, spin=int(newS), charge=int(charge), unit='Angstrom')
    return mol2, newS

def _make_mol_signature(rec: MolRecord, basis: str) -> str:
    data = {"basis": basis, "charge": rec.charge, "spin2S": rec.spin2S,
            "atoms": [(a, [float(f"{x:.8f}") for x in xyz]) for a, xyz in zip(rec.atoms, rec.coords)]}
    blob = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()

def _make_cc_with_frozen(mf, frozen, logger):
    is_uhf = isinstance(mf, scf.uhf.UHF)
    if isinstance(mf, scf.rohf.ROHF):
        mf = mf.to_uhf(); is_uhf = True
    if is_uhf:
        mycc = cc.UCCSD(mf)
        if frozen is not None:
            if isinstance(frozen, tuple):
                mycc = mycc.set(frozen=(sorted(set(frozen[0])), sorted(set(frozen[1]))))
            else:
                F = sorted(set(frozen)); mycc = mycc.set(frozen=(F, F))
    else:
        mycc = cc.CCSD(mf)
        if frozen is not None: mycc = mycc.set(frozen=sorted(set(frozen)))
    return mycc, is_uhf

def _run_ccsd_robust(mf, logger, frozen=None):
    tries = [
        dict(conv_tol=1e-8, max_cycle=120, diis_space=10, diis_start_cycle=1),
        dict(conv_tol=3e-8, max_cycle=200, diis_space=16, diis_start_cycle=0),
    ]
    last_cc = None
    mycc, is_uhf = _make_cc_with_frozen(mf, frozen, logger)
    for t in tries:
        for k,v in t.items():
            try: setattr(mycc, k, v)
            except Exception: pass
        try:
            mycc = mycc.run()
            e = float(mycc.e_tot)
            conv = bool(getattr(mycc, "converged", True))
            if not conv:
                last_cc = mycc
                mycc, is_uhf = _make_cc_with_frozen(mf, frozen, logger)
                continue
            return e, True, mycc, is_uhf
        except Exception:
            mycc, is_uhf = _make_cc_with_frozen(mf, frozen, logger)
            continue
    try:
        emp2 = mp.MP2(mf).run()
        emp2_val = float(getattr(emp2, "e_corr", emp2))
        e_mp2 = float(mf.e_tot) + float(emp2_val)
        return e_mp2, False, last_cc if last_cc is not None else mycc, is_uhf
    except Exception:
        return float(mf.e_tot), False, last_cc if last_cc is not None else mycc, is_uhf

def _safe_triples_energy(mycc, is_unrestricted: bool, logger) -> float:
    try:
        if hasattr(mycc, "ccsd_t"): return float(mycc.ccsd_t())
    except Exception: pass
    if is_unrestricted:
        try:
            from pyscf.cc import uccsd_t as _uccsd_t
            try: return float(_uccsd_t.kernel(mycc))
            except TypeError:
                eris = mycc.ao2mo(); return float(_uccsd_t.kernel(mycc, eris))
        except Exception: pass
    try:
        from pyscf.cc import ccsd_t as _ccsd_t
        try: return float(_ccsd_t.kernel(mycc))
        except TypeError:
            eris = mycc.ao2mo(); return float(_ccsd_t.kernel(mycc, eris))
    except Exception: pass
    raise RuntimeError("CCSD(T) triples energy computation failed")

# -------------------------- Selection utilities --------------------------
def _max_allowed_count(N: int, never_all: bool) -> int:
    if N <= 1: return N
    return (N - 1) if never_all else N

def _sanitize_active_indices(act: List[int], nmo: int, logger, ctx: str) -> List[int]:
    clean = sorted({int(i) for i in act if 0 <= int(i) < nmo})
    if len(clean) != len(act): logger.warning(f"[IDX clamp] {ctx}: {act} → {clean} within [0,{nmo-1}].")
    return clean

def preferred_parity_bit(spin2S: int) -> int:
    return int(spin2S & 1)

def _apply_min_neverall(s: np.ndarray, act: List[int], min_req: int, never_all: bool, logger=None) -> List[int]:
    N = int(s.size)
    act_set = set(act)
    max_allowed = _max_allowed_count(N, never_all)
    k = len(act_set); min_req = int(min(min_req, max_allowed))
    if k > max_allowed:
        order_desc = np.argsort(-s); act_set = set(order_desc[:max_allowed])
    if k < min_req:
        order_desc = np.argsort(-s); need = min_req - k
        add = [i for i in order_desc if i not in act_set][:need]
        act_set = act_set | set(add)
    return sorted(act_set)

def _choose_with_parity_preference(s: np.ndarray, act: List[int], min_req: int, never_all: bool, prefer_bit: int) -> List[int]:
    N = int(s.size); act_set = set(act); k = len(act_set)
    max_allowed = _max_allowed_count(N, never_all)
    if (k % 2) == prefer_bit: return sorted(act_set)
    # Try drop or add minimal-impact change
    sel_vals = [(i, float(s[i])) for i in act_set]
    drop_idx = None
    if sel_vals and (k - 1) >= max(1, min_req):
        drop_idx, _ = min(sel_vals, key=lambda t: t[1])
    unsel = [i for i in range(N) if i not in act_set]
    add_idx = None
    if unsel and (k + 1) <= max_allowed:
        add_idx = max(unsel, key=lambda i: float(s[i]))
    cand = []
    if drop_idx is not None: cand.append(("drop", drop_idx))
    if add_idx is not None:  cand.append(("add", add_idx))
    if not cand: return sorted(act_set)
    move, idx = cand[0]
    if move == "drop": act_set.remove(idx)
    else:              act_set.add(idx)
    return sorted(act_set)

def _active_from_tau(s1_vec: np.ndarray, tau: float) -> List[int]:
    return [int(i) for i, v in enumerate(s1_vec) if float(v) > float(tau)]

# -------------------------- UCASCI (fixed) --------------------------
def _classify_and_fix_active(
    act_in: List[int], mf, logger, s1_hint: Optional[np.ndarray] = None, occ_eps: float = 0.5,
):
    # Ensure UHF
    if isinstance(mf, scf.uhf.UHF):
        mf_u = mf
    else:
        try:
            mf_u = mf.to_uhf()
        except Exception:
            mf_u = scf.UHF(mf.mol)
            try:
                dm = mf.make_rdm1()
                mf_u.conv_tol = 1e-12; mf_u.max_cycle = 60; mf_u.diis = scf.diis.CDIIS()
                mf_u.kernel(dm0=dm)
            except Exception:
                mf_u.kernel()

    mo_a = np.array(mf_u.mo_coeff[0], dtype=float, order="C")
    mo_b = np.array(mf_u.mo_coeff[1], dtype=float, order="C")
    occ_a = np.array(mf_u.mo_occ[0], dtype=float).ravel()
    occ_b = np.array(mf_u.mo_occ[1], dtype=float).ravel()

    nmo = mo_a.shape[1]
    act = sorted({i for i in act_in if 0 <= int(i) < nmo})
    if not act:
        raise RuntimeError("Active set invalid (empty).")

    # Force singly-occupied outside-active into active
    outside = [i for i in range(nmo) if i not in act]
    singly_out = [i for i in outside
                  if ((occ_a[i] > occ_eps) ^ (occ_b[i] > occ_eps)) or (0.5 < (occ_a[i] + occ_b[i]) < 1.5)]
    if singly_out:
        act = sorted(set(act) | set(singly_out))

    # Per-spin cores outside-active
    outside = [i for i in range(nmo) if i not in act]
    core_a = [i for i in outside if float(occ_a[i]) > occ_eps]
    core_b = [i for i in outside if float(occ_b[i]) > occ_eps]

    # Electron counts in the active space
    nalpha, nbeta = mf_u.mol.nelec  # (nα total, nβ total)
    nelecas_a = int(max(0, nalpha - len(core_a)))
    nelecas_b = int(max(0, nbeta  - len(core_b)))

    # Grow active if too small to host electrons
    def _grow_active_until_feasible(act_set):
        if s1_hint is not None and np.size(s1_hint) == nmo and np.isfinite(s1_hint).any():
            order_hint = list(np.argsort(-np.asarray(s1_hint, float)))
        else:
            order_hint = list(np.argsort(-(occ_a + occ_b)))
        while True:
            nact = len(act_set)
            if nelecas_a <= nact and nelecas_b <= nact:
                break
            for j in order_hint:
                if j not in act_set:
                    act_set.add(int(j))
                    break
            else:
                break
        return act_set

    act = sorted(_grow_active_until_feasible(set(act)))
    outside = [i for i in range(nmo) if i not in act]
    core_a = [i for i in outside if float(occ_a[i]) > occ_eps]
    core_b = [i for i in outside if float(occ_b[i]) > occ_eps]
    virt_a = [i for i in outside if i not in core_a]
    virt_b = [i for i in outside if i not in core_b]

    order_a = core_a + act + virt_a
    order_b = core_b + act + virt_b
    nelecas = (int(max(0, nalpha - len(core_a))), int(max(0, nbeta - len(core_b))))
    return mf_u, act, (core_a, core_b), (order_a, order_b), nelecas

def _run_casci_core(active_orb_idx: List[int], scfref: ScfRef, logger, max_memory_mb: int):
    try:
        mf_u, act, (core_a, core_b), (order_a, order_b), nelecas = _classify_and_fix_active(
            act_in=active_orb_idx, mf=scfref.mf, logger=logger, s1_hint=None
        )
        nact = len(act)
        mo_a_re = np.ascontiguousarray(mf_u.mo_coeff[0][:, order_a])
        mo_b_re = np.ascontiguousarray(mf_u.mo_coeff[1][:, order_b])

        mc = mcscf.UCASCI(mf_u, nact, nelecas)
        mc.ncore = (len(core_a), len(core_b))
        mc.mo_coeff = (mo_a_re, mo_b_re)
        if hasattr(mc, "fcisolver"):
            mc.fcisolver.max_memory = int(max_memory_mb)

        e = mc.kernel()[0]
        return float(e)

    except Exception as e:
        logger.debug(f"[UCASCI] failed, fallback HF: {e!r}")
        return float(scfref.mf.e_tot)

# -------------------------- CAS-CC approximations --------------------------
def run_casccsd_approx_with_cached_scf(active_orb_idx: List[int], scfref: ScfRef, logger) -> float:
    mf = scfref.mf
    occ_a, occ_b = scfref.occ_a, scfref.occ_b
    tot_occ = occ_a + occ_b; nmo = scfref.nmo
    act = sorted(i for i in active_orb_idx if 0 <= i < nmo)
    if not act: raise RuntimeError("Empty active set in CAS-CCSD.")
    try:
        if isinstance(mf, scf.uhf.UHF):
            thr = 0.5
            occ_idx_a = [i for i in range(nmo) if float(occ_a[i]) > thr]
            occ_idx_b = [i for i in range(nmo) if float(occ_b[i]) > thr]
            frz_a = [i for i in occ_idx_a if i not in act]
            frz_b = [i for i in occ_idx_b if i not in act]
            if len(occ_idx_a) > 0 and len(frz_a) == len(occ_idx_a):
                keep_a = max(occ_idx_a)
                if keep_a in frz_a: frz_a.remove(keep_a)
            if len(occ_idx_b) > 0 and len(frz_b) == len(occ_idx_b):
                keep_b = max(occ_idx_b)
                if keep_b in frz_b: frz_b.remove(keep_b)
            e_cc, conv, mycc, _ = _run_ccsd_robust(mf, logger, frozen=(sorted(set(frz_a)), sorted(set(frz_b))))
            if not conv:
                e_cc2, conv2, _, _ = _run_ccsd_robust(mf, logger, frozen=None)
                return float(e_cc2 if conv2 else e_cc)
            return float(e_cc)
        elif isinstance(mf, scf.rohf.ROHF):
            occ_idx = [i for i in range(nmo) if float(tot_occ[i]) > 1.5]
            frz = [i for i in occ_idx if i not in act]
            if len(occ_idx) > 0 and len(frz) == len(occ_idx):
                keep = max(occ_idx)
                if keep in frz: frz.remove(keep)
            e_cc, conv, mycc, _ = _run_ccsd_robust(mf, logger, frozen=(sorted(set(frz)), sorted(set(frz))))
            if not conv:
                e_cc2, conv2, _, _ = _run_ccsd_robust(mf, logger, frozen=None)
                return float(e_cc2 if conv2 else e_cc)
            return float(e_cc)
        else:
            occ_idx = [i for i in range(nmo) if float(tot_occ[i]) > 1.5]
            frz = [i for i in occ_idx if i not in act]
            if len(occ_idx) > 0 and len(frz) == len(occ_idx):
                keep = max(occ_idx)
                if keep in frz: frz.remove(keep)
            e_cc, conv, mycc, _ = _run_ccsd_robust(mf, logger, frozen=sorted(set(frz)))
            if not conv:
                e_cc2, conv2, _, _ = _run_ccsd_robust(mf, logger, frozen=None)
                return float(e_cc2 if conv2 else e_cc)
            return float(e_cc)
    except Exception:
        return float(mf.e_tot)

def run_casccsdt_approx_with_cached_scf(active_orb_idx: List[int], scfref: ScfRef, logger) -> float:
    mf = scfref.mf
    occ_a, occ_b = scfref.occ_a, scfref.occ_b
    tot_occ = occ_a + occ_b; nmo = scfref.nmo
    act = sorted(i for i in active_orb_idx if 0 <= i < nmo)
    if not act: raise RuntimeError("Empty active set in CAS-CCSD(T).")
    try:
        if isinstance(mf, scf.uhf.UHF):
            thr = 0.5
            occ_idx_a = [i for i in range(nmo) if float(occ_a[i]) > thr]
            occ_idx_b = [i for i in range(nmo) if float(occ_b[i]) > thr]
            frz_a = [i for i in occ_idx_a if i not in act]
            frz_b = [i for i in occ_idx_b if i not in act]
            if len(occ_idx_a) > 0 and len(frz_a) == len(occ_idx_a):
                keep_a = max(occ_idx_a)
                if keep_a in frz_a: frz_a.remove(keep_a)
            if len(occ_idx_b) > 0 and len(frz_b) == len(occ_idx_b):
                keep_b = max(occ_idx_b)
                if keep_b in frz_b: frz_b.remove(keep_b)
            e_cc, conv, mycc, is_uhf = _run_ccsd_robust(mf, logger, frozen=(sorted(set(frz_a)), sorted(set(frz_b))))
            if not conv:
                e_cc2, conv2, mycc2, _ = _run_ccsd_robust(mf, logger, frozen=None)
                if not conv2: return float(e_cc2)
                mycc, e_cc = mycc2, e_cc2
            try:
                e_t = _safe_triples_energy(mycc, True, logger)
                return float(e_cc + e_t)
            except Exception:
                return float(e_cc)
        elif isinstance(mf, scf.rohf.ROHF):
            occ_idx = [i for i in range(nmo) if float(tot_occ[i]) > 1.5]
            frz = [i for i in occ_idx if i not in act]
            if len(occ_idx) > 0 and len(frz) == len(occ_idx):
                keep = max(occ_idx)
                if keep in frz: frz.remove(keep)
            e_cc, conv, mycc, _ = _run_ccsd_robust(mf, logger, frozen=(sorted(set(frz)), sorted(set(frz))))
            if not conv:
                e_cc2, conv2, mycc2, _ = _run_ccsd_robust(mf, logger, frozen=None)
                if not conv2: return float(e_cc2)
                mycc, e_cc = mycc2, e_cc2
            try:
                e_t = _safe_triples_energy(mycc, True, logger)
                return float(e_cc + e_t)
            except Exception:
                return float(e_cc)
        else:
            occ_idx = [i for i in range(nmo) if float(tot_occ[i]) > 1.5]
            frz = [i for i in occ_idx if i not in act]
            if len(occ_idx) > 0 and len(frz) == len(occ_idx):
                keep = max(occ_idx)
                if keep in frz: frz.remove(keep)
            e_cc, conv, mycc, _ = _run_ccsd_robust(mf, logger, frozen=sorted(set(frz)))
            if not conv:
                e_cc2, conv2, mycc2, _ = _run_ccsd_robust(mf, logger, frozen=None)
                if not conv2: return float(e_cc2)
                mycc, e_cc = mycc2, e_cc2
            try:
                e_t = _safe_triples_energy(mycc, False, logger)
                return float(e_cc + e_t)
            except Exception:
                return float(e_cc)
    except Exception:
        return run_casccsd_approx_with_cached_scf(active_orb_idx, scfref, logger)

# -------------------------- ASF core --------------------------
def _tau_equiv_for_set(s: np.ndarray, act_idx: List[int], tau_lower: float, tau_upper: float) -> float:
    if not act_idx: return float(tau_upper)
    edge = float(np.min(s[np.array(act_idx, int)]))
    return _safe_tau_value(float(np.nextafter(edge, -np.inf)), tau_lower, tau_upper)

def compute_asf_energy_with_threshold(
    meta_item, s1_vec_np: np.ndarray, tau: float,
    scf_cache, E_ccsd_full_cache, E_ccsdt_full_cache,
    branch: str, basis: str, logger, max_memory_mb: int,
    casci_cache: Dict[Tuple[int, Tuple[int,...]], float],
    cascc_cache: Dict[Tuple[str, int, Tuple[int,...]], float],
    asf_cache: Dict[Tuple[str, int, Tuple[int,...]], float],
    min_active_default: int, never_all: bool, prefer_parity: bool,
    tau_lower_hard: float, tau_upper_hard: float,
    e_dmrg: Optional[float] = None, do_refine: bool = False, max_refine_steps: int = 0,
    progress: Optional[Tuple[int,int]] = None, log_summary: bool = True
) -> Tuple[float, List[int], float]:
    idx = meta_item['idx']; name = meta_item.get('name', f"MOL_{idx}")
    scfref: ScfRef = scf_cache[idx]

    s = np.asarray(s1_vec_np, float)
    N = s.size
    if N == 0: raise RuntimeError("No orbitals present.")

    tau = _safe_tau_value(tau, tau_lower_hard, tau_upper_hard)
    act = _active_from_tau(s, tau)
    max_allowed = _max_allowed_count(N, never_all)
    floor = max(2, min(min_active_default, max_allowed))
    act = _apply_min_neverall(s, act, floor, never_all, logger)
    if prefer_parity:
        desired_bit = preferred_parity_bit(scfref.spin2S)
        act = _choose_with_parity_preference(s, act, floor, never_all, desired_bit)

    act = _sanitize_active_indices(act, scfref.nmo, logger, "ASF-initial")
    key_casci = (idx, tuple(act))
    if key_casci in casci_cache: E_casci = casci_cache[key_casci]
    else:
        E_casci = _run_casci_core(act, scfref, logger, max_memory_mb)
        casci_cache[key_casci] = E_casci

    key_cascc = (branch, idx, tuple(act))
    if key_cascc in cascc_cache: E_cascc = cascc_cache[key_cascc]
    else:
        E_cascc = run_casccsd_approx_with_cached_scf(act, scfref, logger) if branch=='ccsd' \
                  else run_casccsdt_approx_with_cached_scf(act, scfref, logger)
        cascc_cache[key_cascc] = E_cascc

    E_full = E_ccsd_full_cache[idx] if branch == 'ccsd' else E_ccsdt_full_cache[idx]
    key_asf = (branch, idx, tuple(act))
    if key_asf in asf_cache: E_asf = asf_cache[key_asf]
    else:
        E_asf = float(E_full + (E_casci - E_cascc))
        asf_cache[key_asf] = E_asf

    tau_used = _tau_equiv_for_set(s, act, tau_lower_hard, tau_upper_hard)
    if log_summary:
        k, total = progress if progress else (None, None)
        tag = f"[{k}/{total}]" if (k is not None and total is not None) else ""
        label_cc = "E_CASCCSD" if branch=="ccsd" else "E_CASCCSD(T)"
        logger.info(
            f"[ASF] {tag} idx={idx} name={name} | branch={branch} | nmo={scfref.nmo} | n_act={len(act)} "
            f"| tau_in={tau:.6f} | tau_used={tau_used:.6f} | E_CASCI={E_casci:+.10f} | {label_cc}={E_cascc:+.10f} | E_full={E_full:+.10f} | E_ASF={E_asf:+.10f}"
        )
    return float(E_asf), list(act), float(tau_used)

# -------------------------- PPO utils --------------------------
class EMABaseline:
    def __init__(self, beta=0.9): self.beta=beta; self.val=None
    def update(self, x: torch.Tensor) -> torch.Tensor:
        # x is scalar tensor
        if self.val is None: self.val = x.detach()
        else: self.val = self.beta*self.val + (1-self.beta)*x.detach()
        return self.val

# -------------------------- CC cache build --------------------------
def build_cache_from_xyz_one_stab(
    ds, basis: str, logger,
    persist_cc_path: Optional[str] = None, skip_triples: bool = False
):
    scf_cache: Dict[int, ScfRef] = {}
    E_ccsd_full_cache: Dict[int, float] = {}
    E_ccsdt_full_cache: Dict[int, float] = {}
    scf_converged_map: Dict[int, bool] = {}
    ccsd_converged_map: Dict[int, bool] = {}

    persist_map: Dict[str, Dict[str, Any]] = {}
    if persist_cc_path and os.path.isfile(persist_cc_path):
        try:
            with open(persist_cc_path, "r", encoding="utf-8") as f:
                persist_map = json.load(f)
            logger.info(f"[CC Persist] Loaded CC cache: {persist_cc_path} (entries={len(persist_map)})")
        except Exception as e:
            logger.warning(f"[CC Persist] Failed to load {persist_cc_path}: {e!r}")

    updated = False

    total = len(ds.records)
    k = 0

    for rec in ds.records:
        k += 1
        logger.info(f"[HF/CC cache] [{k}/{total}] Building for idx={rec.idx} name={rec.name}")
        mol = gto.M(atom=[[a, tuple(c)] for a, c in zip(rec.atoms, rec.coords)],
                    basis=basis, spin=int(rec.spin2S), charge=int(rec.charge), unit='Angstrom')
        mol, spin2S = _parity_fix_if_needed(mol, rec.charge, rec.spin2S, rec.atoms, rec.coords, basis, logger)

        mf = scf.ROHF(mol) if spin2S > 0 else scf.RHF(mol)
        mf.conv_tol = 1e-9; mf.max_cycle = 120; mf.diis = scf.diis.CDIIS()
        mf.kernel()
        if not getattr(mf, "converged", False):
            try:
                mf = mf.newton(); mf.conv_tol = 1e-12; mf.max_cycle = 60; mf.kernel()
            except Exception:
                pass
        mf = one_shot_internal_stability(mf)
        scf_converged_map[rec.idx] = bool(getattr(mf, "converged", False))

        if isinstance(mf, scf.uhf.UHF):
            mo_a = np.array(mf.mo_coeff[0], float, order="C")
            mo_b = np.array(mf.mo_coeff[1], float, order="C")
            occ_a = np.array(mf.mo_occ[0], float); occ_b = np.array(mf.mo_occ[1], float)
            is_uhf = True
        else:
            mo = np.array(mf.mo_coeff, float, order="C")
            mo_a = mo.copy(); mo_b = mo.copy()
            mo_occ = np.array(mf.mo_occ, float)
            if mo_occ.ndim == 1:
                occ_a = mo_occ.copy(); occ_b = np.zeros_like(mo_occ)
            else:
                occ_a = np.array(mo_occ[...,0]).ravel()
                occ_b = np.array(mo_occ[...,1]).ravel()
            is_uhf = False
        nmo = int(mo_a.shape[1])

        scf_cache[rec.idx] = ScfRef(
            mol=mol, mf=mf, E_scf=float(mf.e_tot),
            mo_a=mo_a, mo_b=mo_b, occ_a=occ_a, occ_b=occ_b,
            nmo=nmo, is_uhf_obj=is_uhf, spin2S=int(spin2S)
        )

        key = _make_mol_signature(rec, basis)
        used_persist = False
        if key in persist_map:
            try:
                entry = persist_map[key]
                E_cc = float(entry["E_ccsd"])
                E_cct = float(entry.get("E_ccsdt", E_cc))
                converged = bool(entry.get("ccsd_converged", True))
                E_ccsd_full_cache[rec.idx] = E_cc
                E_ccsdt_full_cache[rec.idx] = E_cct
                used_persist = True
                ccsd_converged_map[rec.idx] = converged
                logger.info(f"[CC Persist] Using cached CC energies for idx={rec.idx} name={rec.name}")
            except Exception:
                used_persist = False

        if not used_persist:
            E_cc, cc_converged, mycc_last, _ = _run_ccsd_robust(mf, logger, frozen=None)
            ccsd_converged_map[rec.idx] = bool(cc_converged)
            E_ccsd_full_cache[rec.idx] = float(E_cc)
            if skip_triples or (not cc_converged):
                E_cct = float(E_cc)
            else:
                try:
                    e_t = _safe_triples_energy(mycc_last, isinstance(mf, scf.uhf.UHF) or isinstance(mf, scf.rohf.ROHF), logger)
                    E_cct = float(E_cc) + float(e_t)
                except Exception:
                    E_cct = float(E_cc)
            E_ccsdt_full_cache[rec.idx] = float(E_cct)

            if persist_cc_path:
                persist_map[key] = {
                    "E_ccsd": E_ccsd_full_cache[rec.idx],
                    "E_ccsdt": E_ccsdt_full_cache[rec.idx],
                    "ccsd_converged": bool(cc_converged),
                    "name": rec.name
                }
                updated = True

        logger.info(f"[HF cache] idx={rec.idx:3d} name={rec.name:<12} "
                    f"SCF_converged={scf_converged_map[rec.idx]}  CCSD_converged={ccsd_converged_map[rec.idx]}  "
                    f"basis={basis} nmo={nmo}  E(HF)={mf.e_tot:+.10f} Ha  "
                    f"E(CCSD)={E_ccsd_full_cache[rec.idx]:+.10f} Ha  "
                    f"E[CCSD(T)]={E_ccsdt_full_cache[rec.idx]:+.10f} Ha")

    if persist_cc_path and updated:
        try:
            tmp = persist_cc_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(persist_map, f)
            os.replace(tmp, persist_cc_path)
            logger.info(f"[CC Persist] Updated CC cache: {persist_cc_path} (entries={len(persist_map)})")
        except Exception as e:
            logger.warning(f"[CC Persist] Failed to write {persist_cc_path}: {e!r}")

    return scf_cache, E_ccsd_full_cache, E_ccsdt_full_cache, scf_converged_map, ccsd_converged_map

# -------------------------- Trainer --------------------------
def _dump_pre_asf(path: str, idx: int, name: str, branch: str, nmo: int, tau_in: float, tau_used: float, s1_vec: np.ndarray, act_idx: List[int]):
    if not path: return
    new = not os.path.exists(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["idx","name","branch","nmo","tau_in","tau_used","mo_index","s1_pred","selected_by_tau"])
        act_set = set(int(i) for i in act_idx)
        for m in range(int(nmo)):
            s1m = float(s1_vec[m]) if m < len(s1_vec) else float('nan')
            w.writerow([idx, name, branch, nmo, tau_in, tau_used, m, s1m, int(m in act_set)])

class Trainer:
    def __init__(self, args, logger):
        self.args = args; self.logger = logger
        if args.torch_compile and hasattr(torch, "compile"):
            try: torch._dynamo.config.suppress_errors = True
            except Exception: pass
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model: Optional[S1Predictor] = None
        self.tau_policy = {
            "ccsd":  TauPolicyContextual(basis_vocab=[args.basis], hidden=args.tau_ctx_hidden, tau_lo=args.tau_lower_hard, tau_hi=args.tau_upper_hard).to(self.device),
            "ccsdt": TauPolicyContextual(basis_vocab=[args.basis], hidden=args.tau_ctx_hidden, tau_lo=args.tau_lower_hard, tau_hi=args.tau_upper_hard).to(self.device),
        }
        self.scaler = FeatureScaler(); self.ytf = TargetTransform(kind=args.y_transform, eps=args.y_eps)
        self.casci_cache: Dict[Tuple[int, Tuple[int,...]], float] = {}
        self.cascc_cache: Dict[Tuple[str, int, Tuple[int,...]], float] = {}
        self.asf_cache: Dict[Tuple[str, int, Tuple[int,...]], float] = {}

    def save_checkpoint(self, path: str):
        ck = {
            "model": self.model.state_dict(),
            "tau_ccsd": self.tau_policy["ccsd"].state_dict(),
            "tau_ccsdt": self.tau_policy["ccsdt"].state_dict(),
            "x_mu": self.scaler.mu.tolist() if self.scaler.mu is not None else None,
            "x_std": self.scaler.sd.tolist() if self.scaler.sd is not None else None,
            "y_transform": self.ytf.to_state(),
            "version": "sas_ctx_tau_1",
        }
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(ck, path)
        self.logger.info(f"[SAVE] Checkpoint written to {os.path.abspath(path)}")

    def load_checkpoint_if_any(self, path: str):
        if not path or not os.path.isfile(path): return False
        ck = torch.load(path, map_location=self.device)
        if 'model' in ck: self.model.load_state_dict(ck['model'])
        if 'tau_ccsd' in ck: self.tau_policy["ccsd"].load_state_dict(ck['tau_ccsd'])
        if 'tau_ccsdt' in ck: self.tau_policy["ccsdt"].load_state_dict(ck['tau_ccsdt'])
        if 'x_mu' in ck and 'x_std' in ck and ck['x_mu'] is not None:
            self.scaler.mu = np.array(ck['x_mu'], dtype=float); self.scaler.sd = np.array(ck['x_std'], dtype=float)
        if 'y_transform' in ck: self.ytf = TargetTransform.from_state(ck['y_transform'])
        self.logger.info(f"[LOAD] Restored from {path}")
        return True

    @torch.no_grad()
    def _predict_s1_np(self, X: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
        Xn = self.scaler.transform(X)
        t = self.model(Xn)[0, mask[0]]
        y = self.ytf.inverse_torch(t).detach().cpu().numpy()
        y = np.nan_to_num(y, nan=0.0, posinf=np.finfo(np.float32).max/4, neginf=0.0)
        return y

    def build_data_and_cache(self):
        self.ds_full = S1Dataset(
            xyz_path=self.args.xyz, descriptors_txt=self.args.desc_txt, s1_txt=self.args.s1_txt, logger=self.logger,
            dmrg_path=(self.args.dmrg if self.args.mode in ("train","evaluate") else None),
            dmrg_unit=self.args.dmrg_unit, default_basis=self.args.basis,
            charge_json=(self.args.charge_json or None), spin_json=(self.args.spin_json or None)
        )
        self.logger.info("Building HF/CC cache ...")
        self.scf_cache, self.E_ccsd_full_cache, self.E_ccsdt_full_cache, \
        self.scf_converged_map, self.ccsd_converged_map = build_cache_from_xyz_one_stab(
            self.ds_full, self.args.basis, self.logger,
            persist_cc_path=self.args.persist_cc_cache if self.args.persist_cc_cache else None,
            skip_triples=self.args.skip_triples
        )

        if self.args.mode == "deploy":
            self.ds_train = self.ds_full
            self.ds_test  = subset_dataset(self.ds_full, []) if len(self.ds_full.records) > 0 else self.ds_full
            self.scaler.fit(self.ds_train)
            self.logger.info(f"[Deploy] Using full dataset: {len(self.ds_train)} molecules; no split; no y-transform fit.")
            return

        all_idx = list(range(len(self.ds_full.records)))
        if not self.args.keep_nonconverged:
            keep_idx = [i for i in all_idx if self.scf_converged_map.get(i, False) and self.ccsd_converged_map.get(i, False)]
            drop_idx = [i for i in all_idx if i not in keep_idx]
            if drop_idx:
                preview = ", ".join(f"{i}:{self.ds_full.records[i].name}" for i in drop_idx[:10])
                self.logger.warning(f"[FILTER] Dropping {len(drop_idx)} non-converged from train/eval: {preview}" + (" ..." if len(drop_idx) > 10 else ""))
            if not keep_idx: raise RuntimeError("All molecules failed convergence; nothing left for training/evaluation.")
            eligible_idx = keep_idx
        else:
            eligible_idx = all_idx
        n = len(eligible_idx); idx = list(eligible_idx); random.Random(self.args.split_seed).shuffle(idx)
        n_test = max(1, int(round(self.args.test_fraction * n)))
        test_idx = sorted(idx[:n_test]); train_idx = sorted(idx[n_test:]) or [test_idx.pop()]
        self.ds_train = subset_dataset(self.ds_full, train_idx)
        self.ds_test  = subset_dataset(self.ds_full, test_idx)
        self.logger.info(f"[Split] train={len(self.ds_train)}  test={len(self.ds_test)}  (eligible={n}/{len(self.ds_full)})")
        self.scaler.fit(self.ds_train); self.ytf.fit(self.ds_train)

    def build_model(self):
        F = self.ds_full.records[0].x_feat.shape[1]
        self.model = S1Predictor(in_dim=F, width=self.args.hidden, depth=self.args.depth, dropout=self.args.dropout).to(self.device)
        if self.args.torch_compile and hasattr(torch, "compile"):
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
                self.logger.info("[Torch] Using torch.compile for predictor")
            except Exception:
                self.logger.info("[Torch] torch.compile unavailable; continuing")
        if self.args.load_model and os.path.isfile(self.args.load_model):
            self.load_checkpoint_if_any(self.args.load_model)

    @torch.no_grad()
    def _compute_s1_metrics(self, ds, split_name: str, epoch: int):
        self.model.eval()
        dl = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate)
        yh_all, yt_all = [], []
        for batch in dl:
            x = batch['x'].to(self.device)
            s1 = batch['s1'].to(self.device)
            mask = batch['mask'].to(self.device)
            y_pred = self._predict_s1_np(x, mask)
            yh_all.append(y_pred.reshape(-1))
            yt_all.append(s1[mask].detach().cpu().numpy().reshape(-1))
        y_pred = np.concatenate(yh_all, axis=0) if yh_all else np.zeros((0,), float)
        y_true = np.concatenate(yt_all, axis=0) if yt_all else np.zeros((0,), float)
        n = y_true.size
        if n == 0: mae = rmse = r2 = float('nan')
        else:
            diff = y_pred - y_true
            mae = float(np.mean(np.abs(diff)))
            rmse = float(np.sqrt(np.mean(diff**2)))
            ss_res = float(np.sum(diff**2))
            mu = float(np.mean(y_true))
            ss_tot = float(np.sum((y_true - mu)**2))
            r2 = float(1.0 - ss_res/ss_tot) if ss_tot > 1e-16 else float('nan')
        self.logger.info(f"[s1 metrics] epoch={epoch:03d} split={split_name:<5} | N={n}  MAE={mae:.6f}  RMSE={rmse:.6f}  R2={r2:.6f}")
        if self.args.s1_metrics_csv:
            new_file = not os.path.exists(self.args.s1_metrics_csv)
            os.makedirs(os.path.dirname(self.args.s1_metrics_csv) or ".", exist_ok=True)
            with open(self.args.s1_metrics_csv, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                if new_file: w.writerow(["epoch","split","N","MAE","RMSE","R2","y_transform"])
                w.writerow([epoch, split_name, n, mae, rmse, r2, self.args.y_transform])
        return mae, rmse, r2, n

    def train_predictor(self):
        dl = DataLoader(self.ds_train, batch_size=self.args.batch_size, shuffle=True, collate_fn=collate)
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, self.args.epochs))
        loss_fn = nn.SmoothL1Loss(beta=0.05)
        self._compute_s1_metrics(self.ds_train, "train", epoch=0)
        _, best_rmse, _, _ = self._compute_s1_metrics(self.ds_test,  "test",  epoch=0)
        best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
        patience = 0
        for ep in range(1, self.args.epochs+1):
            self.model.train()
            tot, nseen = 0.0, 0
            for batch in dl:
                x = batch['x'].to(self.device)
                s1 = batch['s1'].to(self.device)
                mask = batch['mask'].to(self.device)
                x = self.scaler.transform(x)
                t_true = self.ytf.transform_torch(s1)
                t_pred = self.model(x)
                loss = loss_fn(t_pred[mask], t_true[mask])
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(self.model.parameters(), 5.0); opt.step()
                tot += float(loss.item()) * int(mask.sum().item()); nseen += int(mask.sum().item())
            sched.step()
            self.logger.info(f"[Predictor] ep={ep:03d} | SmoothL1(transformed s1)={tot/max(1,nseen):.6f}")
            self._compute_s1_metrics(self.ds_train, "train", epoch=ep)
            _, rmse_val, _, _ = self._compute_s1_metrics(self.ds_test,  "test",  epoch=ep)
            if rmse_val < best_rmse - 1e-10:
                best_rmse = rmse_val
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                patience = 0
            else:
                patience += 1
                if patience >= self.args.early_stop_patience:
                    self.logger.info(f"[Predictor] Early stop at epoch {ep}; best test RMSE={best_rmse:.6f}")
                    break
        self.model.load_state_dict(best_state)

    def _ctx_from_pred(self, s1_pred: torch.Tensor, r2_vec: Optional[torch.Tensor], charge: int, spin2S: int) -> torch.Tensor:
        """
        s1_pred: (N,) tensor (on device)
        r2_vec:  (N,) or None
        """
        N = s1_pred.shape[0]
        s1mean = torch.nanmean(s1_pred) if N>0 else torch.tensor(0.0, device=s1_pred.device)
        s1std  = torch.nanstd(s1_pred) if N>1 else torch.tensor(0.0, device=s1_pred.device)
        p75    = torch.quantile(s1_pred, 0.75) if N>0 else torch.tensor(0.0, device=s1_pred.device)
        p90    = torch.quantile(s1_pred, 0.90) if N>0 else torch.tensor(0.0, device=s1_pred.device)
        if (r2_vec is not None) and (r2_vec.numel() == N):
            r2mean = torch.nanmean(r2_vec)
        else:
            r2mean = torch.tensor(0.0, device=s1_pred.device)
        ctx = torch.stack([
            torch.tensor(float(N), device=s1_pred.device),
            s1mean, s1std, p75, p90,
            torch.tensor(float(charge), device=s1_pred.device),
            torch.tensor(float(spin2S), device=s1_pred.device),
            r2mean
        ], dim=0).view(1, -1)  # (1,8)
        return ctx

    def _train_tau_for_branch(self, branch: str):
        # Freeze predictor (we only train τ here)
        for p in self.model.parameters(): p.requires_grad_(False)
        pol = self.tau_policy[branch]
        dl = DataLoader(self.ds_train, batch_size=1, shuffle=True, collate_fn=collate)
        opt = torch.optim.Adam(pol.parameters(), lr=self.args.rl_lr)
        baseline = EMABaseline(beta=0.9)

        for ep in range(1, self.args.rl_epochs+1):
            used, fails = 0, 0
            for step, batch in enumerate(dl, start=1):
                X=batch['x'].to(self.device)
                mask=batch['mask'].to(self.device)
                metas=batch['meta']; E_dmrg=batch['E_dmrg'].to(self.device)
                r2=batch['r2'].to(self.device) if torch.any(batch['r2'].to(self.device) != 0) else None
                meta=metas[0]; scfref = self.scf_cache[meta['idx']]
                # predict s1 (detach so τ grads don't flow into predictor)
                with torch.no_grad():
                    Xn = self.scaler.transform(X)
                    t = self.model(Xn)[0, mask[0]]
                    s1_pred = self.ytf.inverse_torch(t)  # (N,)
                s1_pred_t = s1_pred  # (N,) tensor on device
                # context
                r2_vec = r2[0, mask[0]] if r2 is not None else None
                ctx = self._ctx_from_pred(s1_pred_t, r2_vec, meta['charge'], meta['spin2S'])  # (1,8)

                # forward τ
                tau = pol(ctx, [self.args.basis])  # (1,)
                tau_clamped = tau  # already in [lo,hi]

                # selection (as tensors for size penalty)
                s_np = s1_pred_t.detach().cpu().numpy()
                act_idx = _active_from_tau(s_np, float(tau_clamped.item()))
                N = s_np.size
                max_allowed = _max_allowed_count(N, self.args.never_all_strict)
                floor = max(2, min(self.args.min_active_default, max_allowed))
                # force min/never-all/parity using numpy, then back to tensor count
                act_idx = _apply_min_neverall(s_np, act_idx, floor, self.args.never_all_strict, self.logger)
                if True:
                    desired_bit = preferred_parity_bit(scfref.spin2S)
                    act_idx = _choose_with_parity_preference(s_np, act_idx, floor, self.args.never_all_strict, desired_bit)
                k_sel = torch.tensor(float(len(act_idx)), device=self.device)

                # ASF energy (no refine during τ training to keep signal clean)
                try:
                    E_asf, act_used, tau_used = compute_asf_energy_with_threshold(
                        meta_item=meta, s1_vec_np=s_np, tau=float(tau_clamped.item()),
                        scf_cache=self.scf_cache, E_ccsd_full_cache=self.E_ccsd_full_cache,
                        E_ccsdt_full_cache=self.E_ccsdt_full_cache, branch=branch, basis=self.args.basis, logger=self.logger,
                        max_memory_mb=self.args.max_memory_mb, casci_cache=self.casci_cache, cascc_cache=self.cascc_cache,
                        asf_cache=self.asf_cache, min_active_default=self.args.min_active_default,
                        never_all=self.args.never_all_strict, prefer_parity=True,
                        tau_lower_hard=self.args.tau_lower_hard, tau_upper_hard=self.args.tau_upper_hard,
                        e_dmrg=float(E_dmrg.item()), do_refine=False, max_refine_steps=0,
                        progress=(step, len(dl)), log_summary=True
                    )
                except Exception as e:
                    fails += 1
                    continue

                # Total loss (tensors throughout)
                # energy loss in Ha
                E_dmrg_Ha = E_dmrg  # (1,)
                L_energy = (E_dmrg_Ha - torch.tensor(E_asf, device=self.device))**2
                # soft size nudge
                size_pen = self.args.size_penalty * (k_sel - float(self.args.k0_target))**2
                L_total = L_energy + size_pen

                opt.zero_grad()
                # ensure scalar
                loss_scalar = L_total if isinstance(L_total, torch.Tensor) else torch.tensor(L_total, device=self.device)
                loss_scalar.backward()
                nn.utils.clip_grad_norm_(pol.parameters(), 5.0)
                opt.step()

                used += 1
                self.logger.info(
                    "[RL] [%s] ep=%02d [%d/%d] idx=%d name=%s | tau=%.6f | k=%d | L=%.6e",
                    branch, ep, step, len(dl), meta['idx'], meta['name'],
                    float(tau_clamped.item()), int(k_sel.item()), float(loss_scalar.item())
                )

            self.logger.info(f"[τ-train:{branch}] ep={ep:02d} used={used} fails={fails}")
        # Save at the end of branch training
        if self.args.save_model:
            self.save_checkpoint(self.args.save_model)

    @torch.no_grad()
    def _select_branch_preview(self, s1_vec: np.ndarray, branch: str, scfref: ScfRef, meta) -> Tuple[List[int], float]:
        s = np.asarray(s1_vec, float)
        # Build context using predicted s1 stats
        s_t = torch.tensor(s, dtype=torch.float32, device=self.device)
        r2_vec = None
        ctx = self._ctx_from_pred(s_t, r2_vec, meta['charge'], meta['spin2S'])
        tau = self.tau_policy[branch](ctx, [self.args.basis]).item()
        act = _active_from_tau(s, tau)
        max_allowed = _max_allowed_count(s.size, self.args.never_all_strict)
        floor = max(2, min(self.args.min_active_default, max_allowed))
        act = _apply_min_neverall(s, act, floor, self.args.never_all_strict, self.logger)
        desired_bit = preferred_parity_bit(scfref.spin2S)
        act = _choose_with_parity_preference(s, act, floor, self.args.never_all_strict, desired_bit)
        return act, tau

    @torch.no_grad()
    def evaluate(self, branches: List[str], tag="TEST",
                 out_csv: str = "test_summary_ctx.csv",
                 out_json: str = "chosen_active_sets_test.json"):
        self._compute_s1_metrics(self.ds_test, "test", epoch=-1)
        dl = DataLoader(self.ds_test, batch_size=1, shuffle=False, collate_fn=collate)
        rows = []; chosen_json = []
        total = len(self.ds_test.records)
        for k_global, batch in enumerate(dl, start=1):
            X=batch['x'].to(self.device); mask=batch['mask'].to(self.device)
            metas=batch['meta']; E_dmrg=batch['E_dmrg'].to(self.device)
            meta=metas[0]; e_dmrg = float(E_dmrg.item()); scfref = self.scf_cache[meta['idx']]
            self.logger.info(f"[EVAL] [{k_global}/{total}] idx={meta['idx']} name={meta['name']}")
            # ŝ1
            s1_vec = self._predict_s1_np(X, mask)
            per_branch = {}
            for b in branches:
                try:
                    act0, tau_in = self._select_branch_preview(s1_vec, b, scfref, meta)
                    E_asf, act_used, tau_used = compute_asf_energy_with_threshold(
                        meta_item=meta, s1_vec_np=s1_vec, tau=tau_in, scf_cache=self.scf_cache,
                        E_ccsd_full_cache=self.E_ccsd_full_cache, E_ccsdt_full_cache=self.E_ccsdt_full_cache,
                        branch=b, basis=self.args.basis, logger=self.logger, max_memory_mb=self.args.max_memory_mb,
                        casci_cache=self.casci_cache, cascc_cache=self.cascc_cache, asf_cache=self.asf_cache,
                        min_active_default=self.args.min_active_default, never_all=self.args.never_all_strict, prefer_parity=True,
                        tau_lower_hard=self.args.tau_lower_hard, tau_upper_hard=self.args.tau_upper_hard,
                        e_dmrg=e_dmrg, do_refine=False, max_refine_steps=0,
                        progress=(k_global, total), log_summary=True
                    )
                    _dump_pre_asf(self.args.pre_asf_dump_csv, meta['idx'], meta['name'], b, scfref.nmo, tau_in, tau_used, s1_vec, act_used)
                    delta_ev = e_dmrg*HARTREE2EV - E_asf*HARTREE2EV
                    per_branch[b] = {
                        "tau_in": float(tau_in), "tau_used": float(tau_used),
                        "nact": len(act_used), "active_idx": act_used,
                        "E_ASF_eV": float(E_asf*HARTREE2EV), "Delta_eV": float(delta_ev)
                    }
                except Exception as e:
                    self.logger.warning(f"[{tag} {b}] idx={meta['idx']} ASF fail: {repr(e)}")
                    per_branch[b] = {"tau_in": None, "tau_used": None, "active_idx": [], "E_ASF_eV": None}
            row = {"idx": meta['idx'], "name": meta['name'],
                   "charge": meta['charge'], "spin2S": meta['spin2S'],
                   "E_DMRG_eV": e_dmrg*HARTREE2EV, "y_transform": self.args.y_transform}
            for b in branches:
                v = per_branch.get(b, {})
                row.update({
                    f"tau_in_{b}": v.get("tau_in"), f"tau_used_{b}": v.get("tau_used"),
                    f"nact_{b}": v.get("nact", 0), f"active_idx_{b}": ";".join(map(str, v.get("active_idx", []))),
                    f"E_ASF_eV_{b}": v.get("E_ASF_eV"), f"Delta_eV_{b}": v.get("Delta_eV")
                })
            rows.append(row)
            chosen_json.append({
                "idx": meta['idx'], "name": meta['name'], "charge": meta['charge'], "spin2S": meta['spin2S'],
                "nmo": int(self.scf_cache[meta['idx']].nmo), "branches": per_branch, "used_predicted_s1": True,
                "y_transform": self.args.y_transform
            })

        if out_csv and rows:
            os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
            base = ["idx","name","charge","spin2S","E_DMRG_eV","y_transform"]
            cols=[]
            for b in branches:
                cols += [f"tau_in_{b}", f"tau_used_{b}", f"nact_{b}", f"active_idx_{b}", f"E_ASF_eV_{b}", f"Delta_eV_{b}"]
            with open(out_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=base+cols); w.writeheader()
                for r in rows: w.writerow(r)
            self.logger.info(f"[{tag}] Saved CSV to {out_csv}")

        if out_json and chosen_json:
            os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(_to_jsonable(chosen_json), f, indent=2)
            self.logger.info(f"[{tag}] Saved JSON to {out_json}")

    @torch.no_grad()
    def deploy(self, branches: List[str], out_json: str = "active_sets_deploy.json"):
        results = []
        total = len(self.ds_full.records)
        for k_global, rec in enumerate(self.ds_full.records, start=1):
            self.logger.info(f"[DEPLOY] [{k_global}/{total}] idx={rec.idx} name={rec.name}")
            x = torch.from_numpy(rec.x_feat).unsqueeze(0).to(self.device)
            mask = torch.ones(1, rec.x_feat.shape[0], dtype=torch.bool, device=self.device)
            s1_vec = self._predict_s1_np(x, mask)
            scfref = self.scf_cache[rec.idx]
            per_branch = {}
            for b in branches:
                try:
                    act0, tau_in = self._select_branch_preview(s1_vec, b, scfref, {'charge':rec.charge,'spin2S':rec.spin2S})
                    E_asf_Ha, act_idx_used, tau_used = compute_asf_energy_with_threshold(
                        meta_item={'idx': rec.idx, 'name': rec.name, 'charge': rec.charge, 'spin2S': rec.spin2S, 'basis': rec.basis},
                        s1_vec_np=s1_vec, tau=tau_in, scf_cache=self.scf_cache,
                        E_ccsd_full_cache=self.E_ccsd_full_cache, E_ccsdt_full_cache=self.E_ccsdt_full_cache,
                        branch=b, basis=self.args.basis, logger=self.logger, max_memory_mb=self.args.max_memory_mb,
                        casci_cache=self.casci_cache, cascc_cache=self.cascc_cache, asf_cache=self.asf_cache,
                        min_active_default=self.args.min_active_default, never_all=self.args.never_all_strict, prefer_parity=True,
                        tau_lower_hard=self.args.tau_lower_hard, tau_upper_hard=self.args.tau_upper_hard,
                        e_dmrg=None, do_refine=False, max_refine_steps=0,
                        progress=(k_global, total), log_summary=True
                    )
                    E_ASF_eV = float(E_asf_Ha * HARTREE2EV)
                    per_branch[b] = {
                        "tau_in": float(tau_in), "tau_used": float(tau_used),
                        "active_idx": act_idx_used, "E_ASF_eV": E_ASF_eV
                    }
                except Exception as e:
                    self.logger.warning(f"[DEPLOY {b}] ASF failed for idx={rec.idx} ({rec.name}); reason={repr(e)}")
                    per_branch[b] = {"tau_in": float(tau_in) if 'tau_in' in locals() else None, "tau_used": None, "active_idx": [], "E_ASF_eV": None}
            results.append({
                "idx": rec.idx, "name": rec.name, "charge": rec.charge, "spin2S": rec.spin2S,
                "nmo": int(self.scf_cache[rec.idx].nmo), "branches": per_branch,
                "used_predicted_s1": True, "y_transform": self.args.y_transform
            })

        if out_json and results:
            os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(_to_jsonable(results), f, indent=2)
            self.logger.info(f"[DEPLOY] Saved active sets (with ASF) to {out_json}")

# -------------------------- Main --------------------------
def _parse_branch_list(text: str, default_list: List[str]) -> List[str]:
    if not text: return default_list
    t = text.strip().lower()
    if t in ("both","all","*"): return ["ccsd","ccsdt"]
    parts = [p.strip().lower() for p in t.split(",") if p.strip()]
    out = [p for p in parts if p in ("ccsd","ccsdt")]
    return sorted(set(out)) or default_list

def main():
    ap = argparse.ArgumentParser(description="SAS RL — Contextual τ + Improved ŝ1 Predictor")
    ap.add_argument("--mode", choices=["train","evaluate","deploy"], default="train")

    # Data
    ap.add_argument("--xyz", required=True)
    ap.add_argument("--desc_txt", required=True)
    ap.add_argument("--s1_txt", default="", help="s1 entropies file (REQUIRED for train/evaluate; optional for deploy)")
    ap.add_argument("--dmrg", default="", help="Required for train/evaluate; optional for deploy")
    ap.add_argument("--basis", default="sto-3g")
    ap.add_argument("--dmrg_unit", choices=["ev","hartree"], default="ev")
    ap.add_argument("--charge_json", required=True)
    ap.add_argument("--spin_json",   required=True)

    # Predictor
    ap.add_argument("--hidden", type=int, default=384)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--early_stop_patience", type=int, default=240)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--torch_compile", action="store_true")

    # Target transform
    ap.add_argument("--y_transform", choices=["logit01","zscore","log1p","sqrt","none"], default="logit01")
    ap.add_argument("--y_eps", type=float, default=1e-4)

    # Selection / τ policy (contextual)
    ap.add_argument("--min_active_default", type=int, default=6, help="Minimum active orbitals before parity logic")
    ap.add_argument("--never_all_strict", action="store_true", default=True, help="Disallow selecting all N orbitals (enforce ≤ N-1)")
    ap.add_argument("--tau_lower_hard", type=float, default=0.02)
    ap.add_argument("--tau_upper_hard", type=float, default=0.30)

    # RL for τ
    ap.add_argument("--rl_epochs", type=int, default=2)
    ap.add_argument("--rl_lr", type=float, default=2.5e-3)
    ap.add_argument("--k0_target", type=float, default=10.0)
    ap.add_argument("--size_penalty", type=float, default=0.03)
    ap.add_argument("--tau_ctx_hidden", type=int, default=64)

    # Caches / convergence / splits
    ap.add_argument("--persist_cc_cache", default="", help="Path to persist/restore CCSD and CCSD(T) energies")
    ap.add_argument("--skip_triples", action="store_true", help="Skip CCSD(T) triples to save time")
    ap.add_argument("--keep_nonconverged", action="store_true", help="Keep molecules whose SCF/CCSD did not converge")
    ap.add_argument("--split_seed", type=int, default=2024)
    ap.add_argument("--test_fraction", type=float, default=0.2)

    # Outputs / checkpoints
    ap.add_argument("--save_model", default="s1_and_tau_ctx.pt")
    ap.add_argument("--load_model", default="", help="Optional: warm start from checkpoint")
    ap.add_argument("--s1_metrics_csv", default="s1_metrics.csv")
    ap.add_argument("--pre_asf_dump_csv", default="")
    ap.add_argument("--out_csv", default="test_summary_ctx.csv", help="Evaluation CSV path")
    ap.add_argument("--out_json", default="chosen_active_sets_test.json", help="Evaluation JSON path")
    ap.add_argument("--deploy_out_json", default="active_sets_deploy.json")

    # Memory / logging / misc
    ap.add_argument("--max_memory_mb", type=int, default=224096)
    ap.add_argument("--log_file", default="run_ctx.log")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--seed", type=int, default=1337)

    args = ap.parse_args()

    # Logger
    logger = setup_logger(args.log_file, args.debug)
    logger.info("SAS RL — Contextual τ + Improved ŝ1 Predictor")

    # Seeds
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    # Trainer + data/cache/model
    trainer = Trainer(args, logger)
    trainer.build_data_and_cache()
    trainer.build_model()

    # Branch parsing
    branches = _parse_branch_list("both", default_list=["ccsd", "ccsdt"])
    logger.info(f"Active branches: {branches}")

    # Mode
    if args.mode == "train":
        logger.info("[MODE] TRAIN")
        trainer.train_predictor()
        for b in branches:
            trainer._train_tau_for_branch(b)
        trainer.evaluate(branches=branches, tag="TEST",
                         out_csv=args.out_csv, out_json=args.out_json)

    elif args.mode == "evaluate":
        logger.info("[MODE] EVALUATE")
        if args.load_model and os.path.isfile(args.load_model):
            trainer.load_checkpoint_if_any(args.load_model)
        trainer.evaluate(branches=branches, tag="TEST",
                         out_csv=args.out_csv, out_json=args.out_json)

    else:  # deploy
        logger.info("[MODE] DEPLOY")
        trainer.deploy(branches=branches, out_json=args.deploy_out_json)

if __name__ == "__main__":
    main()
