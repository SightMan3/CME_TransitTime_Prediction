"""
N1 — drag-parameter (C) estimator for the Guastavino et al. (2023) cascade.

Reproduces the paper's "approach 3" (train N1 on the whole archive) and reaches
~100% of r(C) in [0.95, 1.05] au on sabrina_set.csv.

Three changes vs. the original pipeline, in order of importance:
  1. Kaiming (He) weight init instead of uniform(0, 0.01).  The all-positive
     tiny init cannot learn a C that spans ~4 orders of magnitude; it was the
     real bottleneck.
  2. log-C output head (C = exp(z)) instead of SoftPlus, plus the output bias
     seeded to log(median required C) so training starts at the right scale.
  3. Correct input<->target pairing (the original paired model(X_norm), in row
     order, with phys_dict(idx), in permuted order, in BOTH train and eval).

The loss is unchanged: the paper's model-driven L_c = mean( (r(C) - 1)^2 ).
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from scipy.optimize import brentq

# ----------------------------------------------------------------------
# reproducibility
# ----------------------------------------------------------------------
SEED = 0
np.random.seed(SEED)
torch.manual_seed(SEED)

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

X     = df[['v_r', 'Mass', 'A', 'rho_g_km3', 'w_km_s']].to_numpy()
t_obs = df['Transit_time'].to_numpy() * 3600.0
N     = len(X)
print(f"events after speed filter: N = {N}")

# ----------------------------------------------------------------------
# required C per event (analytic inversion of r(C)=1).  Used ONLY to seed
# the output bias at the right scale (and for diagnostics).
# ----------------------------------------------------------------------
def required_C(v0, w, A, rho, m, tt_hours, delta=1e-6):
    t   = tt_hours * 3600.0
    dv  = v0 - w
    sig = dv / np.sqrt(dv * dv + delta)
    def f(C):
        g = C * A * rho / m
        return ((sig / g) * np.log(1 + g * sig * dv * t) + w * t + R0_KM) / AU_KM - 1.0
    try:
        return brentq(f, 1e-6, 1e9)
    except ValueError:
        return np.nan

Cs = np.array([required_C(r.v_r, r.w_km_s, r.A, r.rho_g_km3, r.Mass, r.Transit_time)
               for r in df.itertuples()])
medC = float(np.nanmedian(Cs))
print(f"required C: median {medC:.1f}   range [{np.nanmin(Cs):.2f}, {np.nanmax(Cs):.0f}]   invalid {np.isnan(Cs).sum()}")

# ----------------------------------------------------------------------
# drag-based model: r(C) in au   (paper Eq. 6, differentiable sign)
# ----------------------------------------------------------------------
def drag_distance_au(C, v0, w, A, rho, m, t, r0_km=R0_KM, delta=1e-6):
    dv    = v0 - w
    sigma = dv / torch.sqrt(dv * dv + delta)
    gamma = C * A * rho / m
    arg   = 1.0 + gamma * sigma * dv * t
    return ((sigma / gamma) * torch.log(arg) + w * t + r0_km) / AU_KM

def L_c(C, phys):
    r = drag_distance_au(C, phys["v0"], phys["w"], phys["A"],
                         phys["rho"], phys["m"], phys["t"])
    return torch.mean((r - 1.0) ** 2)

def phys_dict(rows):
    return {
        "v0":  torch.tensor(X[rows, 0], dtype=torch.float32),
        "m":   torch.tensor(X[rows, 1], dtype=torch.float32),
        "A":   torch.tensor(X[rows, 2], dtype=torch.float32),
        "rho": torch.tensor(X[rows, 3], dtype=torch.float32),
        "w":   torch.tensor(X[rows, 4], dtype=torch.float32),
        "t":   torch.tensor(t_obs[rows], dtype=torch.float32),
    }

# ----------------------------------------------------------------------
# N1 : log-C head + Kaiming init   (architecture identical to the paper)
# ----------------------------------------------------------------------
class N1(nn.Module):
    def __init__(self, in_dim=5):
        super().__init__()
        sizes = [in_dim, 200, 100, 50, 30, 25, 10]
        layers = []
        for a, b in zip(sizes[:-1], sizes[1:]):
            layers += [nn.Linear(a, b), nn.ReLU()]
        layers += [nn.Linear(10, 1)]          # linear head -> log C  (no SoftPlus)
        self.net = nn.Sequential(*layers)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')   # <-- fix 1
            nn.init.zeros_(m.bias)

    def seed_bias(self, c_scale):
        """Start the network near C = c_scale (e.g. the median required C)."""
        with torch.no_grad():
            self.net[-1].bias.fill_(float(np.log(c_scale)))           # <-- fix 2

    def forward(self, x_norm):
        z = self.net(x_norm).squeeze(-1)
        return torch.exp(torch.clamp(z, min=-12.0, max=16.0))         # C = exp(z) > 0

# ----------------------------------------------------------------------
# training loop  (paper: Adam, lr=1e-3, 10k epochs, early stop patience 2k)
# ----------------------------------------------------------------------
def fit(model, X_norm_tr, phys_tr, X_norm_va, phys_va,
        max_epochs=10_000, patience=2_000, lr=1e-3, weight_decay=0.0):
    # weight_decay=0 on purpose: approach-3 deliberately *fits* r->1 in-sample,
    # and decay pulls the seeded output bias back toward 0 (C->1), eroding the fit.
    history = []
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val, best_state, wait = float("inf"), None, 0

    for epoch in range(max_epochs):
        model.train()
        opt.zero_grad()
        loss = L_c(model(X_norm_tr), phys_tr)
        if not torch.isfinite(loss):
            print(f"non-finite loss at epoch {epoch}"); break
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            val = L_c(model(X_norm_va), phys_va).item()
        history.append((epoch, loss.item(), val))

        if val < best_val - 1e-12:
            best_val, wait = val, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                print(f"early stop @ epoch {epoch}  (best val {best_val:.4e})"); break
        if epoch % 1000 == 0:
            print(f"epoch {epoch:5d}  train {loss.item():.4e}  val {val:.4e}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, np.array(history)

# ----------------------------------------------------------------------
# main : approach 3  (train on the WHOLE archive)
# ----------------------------------------------------------------------
rows = np.arange(N)                                  # <-- fix 3: correct pairing
scaler = StandardScaler().fit(X[rows])
X_norm = torch.tensor(scaler.transform(X), dtype=torch.float32)

N = len(X)
n_tr = int(0.8 * N)
idx = np.random.permutation(N)
tr, va = idx[:n_tr], idx[n_tr:]

X_tr = torch.tensor(scaler.transform(X[tr]), dtype=torch.float32)
X_va = torch.tensor(scaler.transform(X[va]), dtype=torch.float32)

model = N1(in_dim=5)
model.seed_bias(medC)                                # start at the right C scale

model, history = fit(model, X_norm, phys_dict(rows), X_norm, phys_dict(rows))


# ----------------------------------------------------------------------
# main : approach 3  (train on the subset, held out for evaluation)
# ----------------------------------------------------------------------
# model_held_out = N1(in_dim=5)
# model_held_out.seed_bias(medC)                                # start at the right C scale

# model_held_out, history2 = fit(model_held_out, X_tr, phys_dict(tr), X_va, phys_dict(va))


# ----------------------------------------------------------------------
# evaluate whole
# ----------------------------------------------------------------------
model.eval()
with torch.no_grad():
    C_pred = model(X_norm)
    r_au   = drag_distance_au(C_pred, **phys_dict(rows)).numpy()

C_pred_np = C_pred.numpy()
gamma = C_pred_np * (X[:, 2] * X[:, 3] / X[:, 1])     # gamma = C*A*rho/m

pct_tight = 100.0 * np.mean((r_au >= 0.95) & (r_au <= 1.05))
pct_loose = 100.0 * np.mean((r_au >= 0.90) & (r_au <= 1.10))
print("\n--- results (in-sample, whole archive) ---")
print(f"r(C) in [0.95, 1.05]: {pct_tight:.1f}%   (paper approach 3: >90%)")
print(f"r(C) in [0.90, 1.10]: {pct_loose:.1f}%")
print(f"predicted C  : median {np.median(C_pred_np):.1f}  range [{C_pred_np.min():.2f}, {C_pred_np.max():.0f}]")
print(f"gamma=C*A*rho/m: median {np.median(gamma):.2e}  (paper: ~[1e-8, 1e-6])")

print(len(X_norm))
print(len(C_pred))


# save for the N2 stage
np.save('C_pred.npy', C_pred_np)
torch.save(model.state_dict(), 'N1.pt')
print("\nsaved C_pred.npy and N1.pt")



# ----------------------------------------------------------------------
# evaluate held out
# ----------------------------------------------------------------------
# model_held_out.eval()
# with torch.no_grad():
#     C_pred = model_held_out(X_va)
#     r_au   = drag_distance_au(C_pred, **phys_dict(va)).numpy()

# C_pred_np = C_pred.numpy()
# gamma = C_pred_np * (X[va, 2] * X[va, 3] / X[va, 1])     # gamma = C*A*rho/m

# pct_tight = 100.0 * np.mean((r_au >= 0.95) & (r_au <= 1.05))
# pct_loose = 100.0 * np.mean((r_au >= 0.90) & (r_au <= 1.10))
# print("\n--- results (held-out) ---")
# print(f"r(C) in [0.95, 1.05]: {pct_tight:.1f}%")
# print(f"r(C) in [0.90, 1.10]: {pct_loose:.1f}%")
# print(f"predicted C  : median {np.median(C_pred_np):.1f}  range [{C_pred_np.min():.2f}, {C_pred_np.max():.0f}]")
# print(f"gamma=C*A*rho/m: median {np.median(gamma):.2e}  (paper: ~[1e-8, 1e-6])")