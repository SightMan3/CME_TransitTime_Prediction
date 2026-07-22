"""
sabrina_cascade.py
==================
Leak-free reproduction of the Guastavino et al. (2023, ApJ 954:151) cascade,
Configuration C5 (mixed loss lambda=0.5, drag parameter C used as an N2 input).

The whole point of this script is the ORDERING of operations inside
`run_one_split`:

    for each random train/val/test split:
        1. train a FRESH N1 on the TRAIN rows only        (loss = mean((r(C)-1)^2))
        2. use that N1 to predict C for ALL rows          -> test rows get an
                                                             OUT-OF-SAMPLE C
        3. train N2 on the train rows (input includes the predicted C)
        4. score N2 on the TEST rows, whose C came from an N1 blind to them

Because C is regenerated *inside* every split, no test event is ever scored
with a C that was fit to its own observed transit time. That is the difference
between this and a pipeline that computes one global C_pred.npy on the whole
archive and reuses it across splits (which leaks the target into every test
fold, and inverts the DBM back to t_obs to ~0.007%).

Expected outcome: test MAE around ~9-10 h (matching the paper's 9.64 h),
N1's out-of-sample r(C)-in-[0.95,1.05] around ~70-80% (NOT ~100%), and the
heavy C tail relaxes on its own because an out-of-sample N1 cannot fit each
event's idiosyncratic C exactly.

Implementation notes carried over from the debugging we did:
  * N2 target is normalized SCALE-ONLY (divide by train std, no mean shift) so
    the Softplus output head can represent it; full standardization + Softplus
    floors every below-mean event.
  * The N2 network sees log10(Mass) and log10(C): both span orders of magnitude,
    and linear standardization turns the largest event into a ~+10 sigma spike
    the net extrapolates to garbage. The PHYSICS term still uses raw units.
  * The DBM distance uses the log1p form  (1/g)*log(1+z) = v_diff*t*log1p(z)/z,
    which is finite and accurate as the drag term -> 0.
  * N1 uses a log-C head (C = exp(z)) with Kaiming init and the output bias
    seeded at log(median required C of the TRAIN rows) so it starts at the right
    scale. Seeding uses train rows only -> no leakage.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import copy
from sklearn.preprocessing import StandardScaler
from scipy.optimize import brentq

# ----------------------------------------------------------------------
# constants
# ----------------------------------------------------------------------
R_SUN_KM = 695_700.0
AU_KM    = 1.495_978_7e8
R0_KM    = 20.0 * R_SUN_KM            # eruption height, 20 R_sun

# ----------------------------------------------------------------------
# data + speed-condition filter (paper Eq. 10)
# ----------------------------------------------------------------------
df = pd.read_csv('./sabrina_set.csv')

tt_s  = df['Transit_time'].to_numpy() * 3600.0
v_bar = (AU_KM - R0_KM) / tt_s
v0    = df['v_r'].to_numpy()
w     = df['w_km_s'].to_numpy()
lo, hi = np.minimum(v0, w), np.maximum(v0, w)
df = df[(v_bar >= lo) & (v_bar <= hi)].reset_index(drop=True)

# five physical input features (order matters; index 1 = Mass)
X5    = df[['v_r', 'Mass', 'A', 'rho_g_km3', 'w_km_s']].to_numpy().astype(np.float64)
t_obs = df['Transit_time'].to_numpy() * 3600.0       # seconds
N     = len(X5)
print(f"events after speed filter: N = {N}")

# ======================================================================
# drag-based model: r(C) in au   (paper Eq. 6, differentiable sign,
# numerically stable log1p form)
# ======================================================================
def drag_distance_au(C, v0, w, A, rho, m, t, delta=1e-6):
    dv    = v0 - w
    sigma = dv / torch.sqrt(dv * dv + delta)
    gamma = C * A * rho / m
    g_dv  = gamma * sigma                       # = term in the N2 loss
    z     = g_dv * dv * t                        # >= 0 for the valid sign branch
    # (1/g_dv)*log(1+z) = dv*t * log1p(z)/z, stable as z -> 0
    ratio = torch.where(torch.abs(z) < 1e-6, 1.0 - z / 2.0, torch.log1p(z) / z)
    r_km  = dv * t * ratio + w * t + R0_KM
    return r_km / AU_KM

# ======================================================================
# N1 : drag-parameter estimator (log-C head, Kaiming init)
# ======================================================================
class N1(nn.Module):
    def __init__(self, in_dim=5):
        super().__init__()
        sizes = [in_dim, 200, 100, 50, 30, 25, 10]
        layers = []
        for a, b in zip(sizes[:-1], sizes[1:]):
            layers += [nn.Linear(a, b), nn.ReLU()]
        layers += [nn.Linear(10, 1)]            # linear head -> log C
        self.net = nn.Sequential(*layers)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            nn.init.zeros_(m.bias)

    def seed_bias(self, c_scale):
        with torch.no_grad():
            self.net[-1].bias.fill_(float(np.log(max(c_scale, 1e-6))))

    def forward(self, x_norm):
        z = self.net(x_norm).squeeze(-1)
        return torch.exp(torch.clamp(z, min=-12.0, max=16.0))   # C = exp(z) > 0


def required_C(v0, w, A, rho, m, tt_hours, delta=1e-6):
    """C that puts r at 1 au at the observed TT. Used ONLY to seed the N1 output
    bias at the right scale (computed on train rows only). Robust to events the
    DBM can't bracket: falls back to the log-grid C minimizing |r-1|."""
    t   = tt_hours * 3600.0
    dv  = v0 - w
    sig = dv / np.sqrt(dv * dv + delta)
    def f(C):
        g = C * A * rho / m
        return ((sig / g) * np.log(1 + g * sig * dv * t) + w * t + R0_KM) / AU_KM - 1.0
    grid = np.logspace(-3, 8, 60)
    vals = np.array([f(C) for C in grid])
    sign_change = np.where(np.sign(vals[:-1]) != np.sign(vals[1:]))[0]
    if len(sign_change):
        i = sign_change[0]
        try:
            return brentq(f, grid[i], grid[i + 1])
        except ValueError:
            pass
    return float(grid[np.argmin(np.abs(vals))])      # always finite


def L_c(C, phys):
    r = drag_distance_au(C, phys["v0"], phys["w"], phys["A"],
                         phys["rho"], phys["m"], phys["t"])
    return torch.mean((r - 1.0) ** 2)


def phys_dict(rows):
    return {
        "v0":  torch.tensor(X5[rows, 0], dtype=torch.float32),
        "m":   torch.tensor(X5[rows, 1], dtype=torch.float32),
        "A":   torch.tensor(X5[rows, 2], dtype=torch.float32),
        "rho": torch.tensor(X5[rows, 3], dtype=torch.float32),
        "w":   torch.tensor(X5[rows, 4], dtype=torch.float32),
        "t":   torch.tensor(t_obs[rows], dtype=torch.float32),
    }


def fit_N1(model, Xn_tr, phys_tr, Xn_va, phys_va,
           max_epochs=3000, patience=400, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)   # no weight decay (approach-3 style fit)
    best, best_state, wait = float("inf"), None, 0
    for epoch in range(max_epochs):
        model.train(); opt.zero_grad()
        loss = L_c(model(Xn_tr), phys_tr)
        if not torch.isfinite(loss):
            break
        loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            val = L_c(model(Xn_va), phys_va).item()
        if val < best - 1e-12:
            best, wait = val, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model

# ======================================================================
# N2 : travel-time predictor
# ======================================================================
class N2Model(nn.Module):
    def __init__(self, input_size=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 200), nn.ReLU(),
            nn.Linear(200, 100), nn.ReLU(),
            nn.Linear(100, 50), nn.ReLU(),
            nn.Linear(50, 30), nn.ReLU(),
            nn.Linear(30, 25), nn.ReLU(),
            nn.Linear(25, 10), nn.ReLU(),
            nn.Linear(10, 1), nn.Softplus(),
        )

    def forward(self, x):
        return self.net(x)


def init_weights_n2(m):
    if isinstance(m, nn.Linear):
        nn.init.uniform_(m.weight, 0.0, 0.01)
        nn.init.zeros_(m.bias)


class L_t(nn.Module):
    """N2 loss: lambda*data + (1-lambda)*physics. Data term in scale-only y units;
    physics term in au, evaluated at the predicted TT with the per-split C."""
    def __init__(self, y_scale, lambda_val=0.5, delta=1e-6):
        super().__init__()
        self.y_scale = y_scale
        self.lambda_val = lambda_val
        self.delta = delta

    def forward(self, y_pred_scaled, y_true_scaled, x_unscaled):
        tt = y_pred_scaled * self.y_scale                       # seconds
        v0  = x_unscaled[:, 0:1]; m = x_unscaled[:, 1:2]; A = x_unscaled[:, 2:3]
        rho = x_unscaled[:, 3:4]; w = x_unscaled[:, 4:5]; C = x_unscaled[:, 5:6]
        r_au = drag_distance_au(C, v0, w, A, rho, m, tt, self.delta)
        loss_data = torch.mean((y_pred_scaled - y_true_scaled) ** 2)
        loss_phys = torch.mean((r_au - 1.0) ** 2)
        return self.lambda_val * loss_data + (1.0 - self.lambda_val) * loss_phys


def fit_N2(model, X_tr, y_tr, Xu_tr, X_va, y_va, Xu_va, crit,
           max_epochs=2000, patience=300, lr=1e-3, weight_decay=1e-4):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best, best_state, wait = float("inf"), copy.deepcopy(model.state_dict()), 0
    for epoch in range(max_epochs):
        model.train(); opt.zero_grad()
        crit(model(X_tr), y_tr, Xu_tr).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            val = crit(model(X_va), y_va, Xu_va).item()
        if val < best:
            best, wait = val, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    return model

# ======================================================================
# one honest split: N1 (train rows) -> out-of-sample C -> N2 -> test
# ======================================================================
def run_one_split(seed):
    rng  = np.random.default_rng(seed)
    perm = rng.permutation(N)
    n_tr, n_va = int(0.70 * N), int(0.15 * N)
    tr, va, te = perm[:n_tr], perm[n_tr:n_tr + n_va], perm[n_tr + n_va:]

    torch.manual_seed(seed)

    # ---- 1. N1 on TRAIN rows only -------------------------------------
    n1_scaler = StandardScaler().fit(X5[tr])
    Xn_all = torch.tensor(n1_scaler.transform(X5), dtype=torch.float32)   # for prediction
    Xn_tr  = torch.tensor(n1_scaler.transform(X5[tr]), dtype=torch.float32)
    Xn_va  = torch.tensor(n1_scaler.transform(X5[va]), dtype=torch.float32)

    med_train_C = np.nanmedian([required_C(*X5[i], t_obs[i] / 3600.0) for i in tr])
    if not np.isfinite(med_train_C):
        med_train_C = 100.0

    n1 = N1(in_dim=5)
    n1.seed_bias(med_train_C)
    n1 = fit_N1(n1, Xn_tr, phys_dict(tr), Xn_va, phys_dict(va))

    # ---- 2. predict C for ALL rows (test rows are out-of-sample) ------
    n1.eval()
    with torch.no_grad():
        C_pred = n1(Xn_all).numpy()                                    # shape (N,)

    # N1 diagnostic on the TEST rows: how well does out-of-sample C put r at 1 au?
    with torch.no_grad():
        r_te = drag_distance_au(torch.tensor(C_pred[te], dtype=torch.float32),
                                **phys_dict(te)).numpy()
    n1_tol = float(np.mean((r_te >= 0.95) & (r_te <= 1.05)))           # ~0.7-0.8 if honest

    # ---- 3. assemble N2 features --------------------------------------
    # raw 6-col matrix for the PHYSICS term (real units)
    X6 = np.column_stack([X5, C_pred])                                 # [v0,m,A,rho,w,C]
    # network input: log10(Mass) and log10(C); everything else linear
    Xnet = X6.copy()
    Xnet[:, 1] = np.log10(Xnet[:, 1])
    Xnet[:, 5] = np.log10(np.clip(Xnet[:, 5], 1e-6, None))

    n2_scaler = StandardScaler().fit(Xnet[tr])
    X2_tr = torch.tensor(n2_scaler.transform(Xnet[tr]), dtype=torch.float32)
    X2_va = torch.tensor(n2_scaler.transform(Xnet[va]), dtype=torch.float32)
    X2_te = torch.tensor(n2_scaler.transform(Xnet[te]), dtype=torch.float32)
    Xu_tr = torch.tensor(X6[tr], dtype=torch.float32)                  # raw -> physics
    Xu_va = torch.tensor(X6[va], dtype=torch.float32)

    y_scale = t_obs[tr].std()
    y2_tr = torch.tensor((t_obs[tr] / y_scale).reshape(-1, 1), dtype=torch.float32)
    y2_va = torch.tensor((t_obs[va] / y_scale).reshape(-1, 1), dtype=torch.float32)

    # ---- 4. N2 train + score on TEST ----------------------------------
    n2 = N2Model(input_size=6)
    n2.apply(init_weights_n2)
    crit = L_t(y_scale, lambda_val=0.5)
    n2 = fit_N2(n2, X2_tr, y2_tr, Xu_tr, X2_va, y2_va, Xu_va, crit)

    n2.eval()
    with torch.no_grad():
        pred_h = n2(X2_te).numpy().flatten() * y_scale / 3600.0
    true_h = t_obs[te] / 3600.0
    abs_err = np.abs(pred_h - true_h)
    return te, abs_err, n1_tol

# ======================================================================
# aggregate over many random splits
# ======================================================================
if __name__ == "__main__":
    N_SPLITS = 40
    results, n1_tols = [], []
    for s in range(N_SPLITS):
        te, err, tol = run_one_split(s)
        results.append((te, err)); n1_tols.append(tol)
        print(f"split {s:2d}  test MAE {err.mean():5.2f} h   N1 out-of-sample r-tol {tol*100:4.0f}%")

    pooled = np.concatenate([e for _, e in results])
    smae   = np.array([e.mean() for _, e in results])
    sums   = np.zeros(N); cnt = np.zeros(N)
    for idx, err in results:
        np.add.at(sums, idx, err); np.add.at(cnt, idx, 1)
    per_event = np.divide(sums, cnt, out=np.full(N, np.nan), where=cnt > 0)

    print("\n================  HONEST C5 (out-of-sample C)  ================")
    print(f"splits: {N_SPLITS}")
    print(f"per-split test MAE : median {np.median(smae):.2f} h   "
          f"IQR [{np.percentile(smae,25):.2f}, {np.percentile(smae,75):.2f}] h")
    print(f"pooled test errors : median {np.median(pooled):.2f} h   "
          f"mean {pooled.mean():.2f} h   STD {pooled.std():.2f} h")
    print(f"N1 out-of-sample r(C) in [0.95,1.05]: median {np.median(n1_tols)*100:.0f}% "
          f"(leaky whole-archive N1 would be ~100%)")
    worst = np.argsort(per_event)[::-1][:6]
    print("worst events (row, mean err h, n):",
          ", ".join(f"{i}:{per_event[i]:.1f}(n{int(cnt[i])})" for i in worst))
    print("paper C5 (test): median 9.46 h, mean 9.64 h, pooled STD ~7.46 h")