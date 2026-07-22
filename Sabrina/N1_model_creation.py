import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import torch
    import torch.nn as nn
    from sklearn.preprocessing import StandardScaler
    import marimo as mo

    return StandardScaler, mo, nn, np, pd, plt, torch


@app.cell
def _():
    R_SUN_KM = 695_700.0
    AU_KM    = 1.495_978_7e8
    R0_KM    = 20.0 * R_SUN_KM
    return AU_KM, R0_KM


@app.cell
def _(AU_KM, R0_KM, np, pd):
    df = pd.read_csv('./sabrina_set.csv')

    tt_s  = df['Transit_time'].to_numpy() * 3600.0
    v_bar = (AU_KM - R0_KM) / tt_s
    v0    = df['v_r'].to_numpy()
    w     = df['w_km_s'].to_numpy()

    lo, hi = np.minimum(v0, w), np.maximum(v0, w)
    df = df[(v_bar >= lo) & (v_bar <= hi)]
    df
    return (df,)


@app.cell
def _(StandardScaler, df, np, torch):
    X = df[['v_r', 'Mass', 'A', 'rho_g_km3', 'w_km_s']].to_numpy()
    t_obs = df['Transit_time'].to_numpy() * 3600

    N = len(X)
    n_tr = int(0.8 * N)
    idx = np.random.permutation(N)
    tr, va = idx[:n_tr], idx[n_tr:]

    scaler = StandardScaler().fit(X[tr])
    X_tr = torch.tensor(scaler.transform(X[tr]), dtype=torch.float32)
    X_va = torch.tensor(scaler.transform(X[va]), dtype=torch.float32)
    X_norm = torch.tensor(scaler.transform(X), dtype=torch.float32)
    return N, X, X_norm, t_obs


@app.cell
def _(X, t_obs, torch):
    def phys_dict(rows):
        return {
            "v0":  torch.tensor(X[rows, 0], dtype=torch.float32),
            "m":   torch.tensor(X[rows, 1], dtype=torch.float32),
            "A":   torch.tensor(X[rows, 2], dtype=torch.float32),
            "rho": torch.tensor(X[rows, 3], dtype=torch.float32),
            "w":   torch.tensor(X[rows, 4], dtype=torch.float32),
            "t":   torch.tensor(t_obs[rows],     dtype=torch.float32),
        }

    return (phys_dict,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # N1 Model Architecture
    1. 5 input neurons
    3. Hidden 1: 200
    4. Hidden 2: 100
    5. Hidden 3: 50
    6. Hidden 4: 30
    7. Hidden 5: 25
    8. Hidden 6: 10
    """)
    return


@app.cell
def _(mo):
    mo.image("public/N1_diagram.png")
    return


@app.cell
def _(nn):
    class N1(nn.Module):
        def __init__(self, in_dim):
            super().__init__()
            sizes = [in_dim, 200, 100, 50, 30, 25, 10]

            layers = []
            for a, b in zip(sizes[:-1], sizes[1:]):
                layers += [nn.Linear(a, b), nn.ReLU()]
            layers += [nn.Linear(10, 1), nn.Softplus()]
            self.net = nn.Sequential(*layers)
            self.apply(self._init)

        @staticmethod
        def _init(m):
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, 0.0, 0.01)
                nn.init.zeros_(m.bias)

        def forward(self, x_norm):
            return self.net(x_norm).squeeze(-1)


    return


@app.cell
def _(nn, torch):
    class N1_logC(nn.Module):
        def __init__(self, in_dim):
            super().__init__()
            sizes = [in_dim, 200, 100, 50, 30, 25, 10]
            layers = []
            for a, b in zip(sizes[:-1], sizes[1:]):
                layers += [nn.Linear(a, b), nn.ReLU()]
            layers += [nn.Linear(10, 1)]          # linear head -> outputs log C (no Softplus)
            self.net = nn.Sequential(*layers)
            self.apply(self._init)

        @staticmethod
        def _init(m):
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, 0.0, 0.01)
                nn.init.zeros_(m.bias)

        def forward(self, x_norm):
            logC = self.net(x_norm).squeeze(-1)
            return torch.exp(torch.clamp(logC, min=-10.0, max=12.0))

    return (N1_logC,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## N1 Loss function
    $$
    L_c(t, N1(x)) = \left(\frac{m\sigma}{A\rho N1(x)}\log\left(1 + \sigma\frac{A\rho N1(x)}{m}(v_0 - w)t\right) + wt + r_0 - 1\right)^2
    $$

    kde
    $$
    \sigma = \frac{v_0 - w}{\sqrt{(v_0 - w)^2 + \delta}}
    $$
    kde $\delta$ je mala kladna hodnota, v nasom pripade $\delta = 1\times10^{-6}$.

    ### Drag_distance_au function
    vypocitava
    $$
    r(t, N1(x)) = \frac{m\sigma}{A\rho N1(x)}\log\left(1 + \sigma\frac{A\rho N1(x)}{m}(v_0 - w)t\right) + wt + r_0
    $$

    ### L_c function
    hlavna loss funkcia, vypocitava
    $$
    L_c(t, N1(x)) = (r(t, N1(x)) - 1)^2
    $$

    teda porovnavame $r(t, N1(x))$ oproti jednotke, teda cely vysledok drag-based modelu musi byt vynormovana vzdialenostou 1AU, kedze vsetko co pocitame v rovnici je v nejakej forme kilometrov:

    - $\omega$ - $km/s$
    - $\rho$   - $g/km^3$
    - $A$ - $km^2$
    - $v_0$ - $km/s$
    """)
    return


@app.cell
def _(AU_KM, R0_KM, torch):
    def drag_distance_au(C, v0, w, A, rho, m, t, r0_km=R0_KM, delta=1e-6):
        dv    = v0 - w
        sigma = dv / torch.sqrt(dv * dv + delta)   # smooth sign  (paper's σ, differentiable)
        gamma = C * A * rho / m                     # [km^-1]; depends on C
        # arg = 1 + σ·γ·(v0-w)·t = 1 + γ·|v0-w|·t  >= ~1  -> log always defined
        arg   = 1.0 + gamma * sigma * dv * t
        r_km  = (sigma / gamma) * torch.log(arg) + w * t + r0_km
        # Small-gamma limit (if you ever drop SoftPlus): (σ/γ)·log(arg) -> (v0-w)·t
        return r_km / AU_KM

    return (drag_distance_au,)


@app.cell
def _(drag_distance_au, torch):
    def L_c(C, phys):
        """phys: dict of physical tensors {v0,w,A,rho,m,t}. Target r = 1 au."""
        r = drag_distance_au(C, phys["v0"], phys["w"], phys["A"],
                             phys["rho"], phys["m"], phys["t"])
        return torch.mean((r - 1.0) ** 2)

    return (L_c,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## N1 Training loop
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Checking units
    """)
    return


@app.cell
def _(df, np):
    d = df
    for c in ['rel_wid','Mass','v_r','w_km_s','rho_g_km3','A','Transit_time']:
        print(f"{c:12s}", d[c].min(), d[c].max())

    # master unit check: gamma at C=1 should be ~1e-7 km^-1 if A, rho, m units agree
    gamma_C1 = d['A'] * d['rho_g_km3'] / d['Mass']
    print("median gamma (C=1):", np.median(gamma_C1), "km^-1   target ~0.7e-7")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Checking required C for events
    """)
    return


@app.cell
def _(AU_KM, R0_KM, df, np):
    from scipy.optimize import brentq


    def required_C(v0, w, A, rho, m, tt_hours, delta=1e-6):
        t  = tt_hours*3600.0
        dv = v0 - w
        sig = dv/np.sqrt(dv*dv + delta)
        def f(C):
            g = C*A*rho/m
            return ((sig/g)*np.log(1 + g*sig*dv*t) + w*t + R0_KM)/AU_KM - 1.0
        try:    return brentq(f, 1e-4, 1e7)
        except ValueError: return np.nan

    Cs = np.array([required_C(r.v_r, r.w_km_s, r.A, r.rho_g_km3, r.Mass, r.Transit_time)
                   for r in df.itertuples()])
    print("required C  median", np.nanmedian(Cs), " range", np.nanmin(Cs), np.nanmax(Cs))
    print("events with NO valid C:", np.isnan(Cs).sum())
    return (Cs,)


@app.cell
def _(L_c, torch):
    def fit(model, X_norm_tr, phys_tr, X_norm_va, phys_va,
            max_epochs=10_000, patience=2_000, lr=1e-3, weight_decay=1e-4):
        history = []                              # local, so reruns don't accumulate
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        best_val, best_state, wait = float("inf"), None, 0

        for epoch in range(max_epochs):
            model.train()
            opt.zero_grad()
            loss = L_c(model(X_norm_tr), phys_tr)

            if not torch.isfinite(loss): 
                print(f"Non-finite loss at epoch {epoch} — check the log-space bias seed.")
                break

            loss.backward()
            opt.step()

            model.eval()
            with torch.no_grad():
                val = L_c(model(X_norm_va), phys_va).item()

            history.append((epoch, loss.item(), val))   # log every epoch, both losses

            if val < best_val - 1e-12:
                best_val, wait = val, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                wait += 1
                if wait >= patience:
                    print(f"Early stop @ epoch {epoch} (best val {best_val:.4e})")
                    break

            if epoch % 500 == 0:
                print(f"epoch {epoch:5d}  train {loss.item():.4e}  val {val:.4e}")

        if best_state is not None:
            model.load_state_dict(best_state)
        return model, history

    return (fit,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## N1 Main loop
    """)
    return


@app.cell
def _(Cs, N, N1_logC, X_norm, fit, np, phys_dict, plt, torch):
    N1_model = N1_logC(in_dim=5)
    with torch.no_grad():
        N1_model.net[-1].bias.fill_(float(np.log(np.nanmedian(Cs))))
    N1_model_trained, history = fit(N1_model, X_norm, phys_dict(np.arange(N)), X_norm, phys_dict(np.arange(N)))

    h = np.array(history)                      # columns: epoch, train, val
    plt.plot(h[:,0], h[:,1], label='train')
    plt.plot(h[:,0], h[:,2], label='val')
    plt.yscale('log'); plt.legend()
    return (N1_model_trained,)


@app.cell
def _(N, N1_model_trained, X_norm, drag_distance_au, np, phys_dict, torch):
    N1_model_trained.eval()
    with torch.no_grad():
        C_pred = N1_model_trained(X_norm)
    print("predicted C (sample):", C_pred[:5].tolist())

    with torch.no_grad():
        r_au = drag_distance_au(C_pred, **phys_dict(np.arange(N)))
    print("r(C) in au (should sit near 1):", r_au[:5].tolist())
    r_au = np.array(r_au)
    return (r_au,)


@app.cell
def _(np, r_au):
    np.mean((r_au >= 0.95) & (r_au <= 1.05)) * 100
    return


@app.cell
def _(plt, r_au):
    plt.hist(r_au, bins=20)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
