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
def _(df, np):
    C_pred = np.load('./C_pred.npy')

    df['C_pred'] = C_pred
    df
    return


@app.cell
def _(StandardScaler, df, np, torch):
    X = df[['v_r', 'Mass', 'A', 'rho_g_km3', 'w_km_s', 'C_pred']].to_numpy()
    y = df['Transit_time'].to_numpy() * 3600

    N = len(X)
    idx = np.arange(N)
    n_tr = int(0.8 * N)
    tr, va = idx[:n_tr], idx[n_tr:]

    scaler = StandardScaler().fit(X[tr])
    X_tr = torch.tensor(scaler.transform(X[tr]), dtype=torch.float32)
    X_va = torch.tensor(scaler.transform(X[va]), dtype=torch.float32)

    y_scale = y[tr].std()
    y_tr = torch.tensor((y[tr] / y_scale).reshape(-1, 1), dtype=torch.float32)
    y_va = torch.tensor((y[va] / y_scale).reshape(-1, 1), dtype=torch.float32)
    return X, X_tr, X_va, tr, va, y, y_scale, y_tr, y_va


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # N2 Loss function
    $$
    L_t(t, N2(\bar{\mathbf{x}})) = \lambda(t - N2(\bar{\mathbf{x}}))^2 + (1 - \lambda)(r(N2(\bar{\mathbf{x}}), N1(\mathbf{x})) - 1)^2
    $$

    $$
    = \lambda(t - N2(\bar{\mathbf{x}}))^2 + (1 - \lambda)\left(\frac{m\sigma}{A\rho N1(x)}\cdot log\left(1 + \sigma\frac{A\rho N1(x)}{m}\right)(v_0 - w)N2(\bar{x}) + v_0N2(\bar{x}) + r_0 - 1\right)^2
    $$
    """)
    return


@app.cell
def _(nn, torch):
    class L_t(nn.Module):
        def __init__(self, y_scaler, AU_KM, R0_KM, lambda_val=0.5, delta=1e-6):
            super().__init__()
            # Save scaler parameters to unscale target variables inside the loss function
            self.y_scale = y_scaler

            # Configuration parameters (C5 uses lambda = 0.5)
            self.lambda_val = lambda_val
            self.delta = delta
            self.AU_KM = AU_KM
            self.R0_KM = R0_KM

        def forward(self, y_pred_scaled, y_true_scaled, x_unscaled):
            device = y_pred_scaled.device
            scale = self.y_scale

            # 1. Unscale TT back to seconds for physical equations
            tt_sec = y_pred_scaled * scale

            # Extract unscaled features
            v0 = x_unscaled[:, 0:1]
            m = x_unscaled[:, 1:2]
            A = x_unscaled[:, 2:3]
            rho = x_unscaled[:, 3:4]
            w = x_unscaled[:, 4:5]
            C = x_unscaled[:, 5:6]

            # 2. Physics Model: sigma calculation (Eq 6)
            # Add a tiny epsilon to v0 - w to avoid exact zero division
            v_diff = v0 - w
            v_diff = torch.where(torch.abs(v_diff) < 1e-12, torch.sign(v_diff) * 1e-12 + 1e-12, v_diff)
            sigma = v_diff / torch.sqrt(v_diff**2 + self.delta)

            # Physics Model: kinematic calculation (Eq 7)
            coeff = (A * rho * C) / m
            term = coeff * sigma
            term_safe = torch.where(torch.abs(term) < 1e-12, torch.sign(term) * 1e-12 + 1e-12, term)

            log_arg = 1.0 + term_safe * v_diff * tt_sec
            log_arg = torch.clamp(log_arg, min=1e-7) # Prevent NaNs in log

            # Calculate r(t, C) in km, then convert to AU
            r_km = (1.0 / term_safe) * torch.log(log_arg) + w * tt_sec + self.R0_KM
            r_AU = r_km / self.AU_KM

            # 3. Combine Data Loss and Physics Loss
            loss_data = torch.mean((y_pred_scaled - y_true_scaled)**2)
            loss_phys = torch.mean((r_AU - 1.0)**2) # Target position at transit time is 1 AU

            return self.lambda_val * loss_data + (1.0 - self.lambda_val) * loss_phys

    return (L_t,)


@app.cell
def _(nn):
    class N2Model(nn.Module):
        def __init__(self, input_size=6):
            super().__init__()

            self.net = nn.Sequential(
                nn.Linear(input_size, 200),
                nn.ReLU(),
                nn.Linear(200, 100),
                nn.ReLU(),
                nn.Linear(100, 50),
                nn.ReLU(),
                nn.Linear(50, 30),
                nn.ReLU(),
                nn.Linear(30, 25),
                nn.ReLU(),
                nn.Linear(25, 10),
                nn.ReLU(),
                nn.Linear(10, 1),
                nn.Softplus() # SoftPlus ensures time predictions are always positive
            )

        def forward(self, x):
            return self.net(x)

    return (N2Model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Training loop
    """)
    return


@app.cell
def _(nn):
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.uniform_(m.weight, 0.0, 0.01)
            nn.init.zeros_(m.bias)

    return (init_weights,)


@app.cell
def _(
    AU_KM,
    L_t,
    N2Model,
    R0_KM,
    X,
    X_tr,
    X_va,
    init_weights,
    torch,
    tr,
    va,
    y_scale,
    y_tr,
    y_va,
):
    import copy

    # Unscaled inputs required for physics constraint
    X_unscaled_tr = torch.tensor(X[tr], dtype=torch.float32)
    X_unscaled_va = torch.tensor(X[va], dtype=torch.float32)

    model = N2Model(input_size=6)
    model.apply(init_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # Using lambda_val=0.5 corresponds to Configuration 5 (C5) in the paper
    criterion = L_t(y_scale, AU_KM, R0_KM, lambda_val=0.5)

    max_epochs = 10000
    patience = 2000
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_wts = copy.deepcopy(model.state_dict())

    train_losses = []
    val_losses = []

    for epoch in range(max_epochs):
        # --- Training Phase ---
        model.train()
        optimizer.zero_grad()

        y_pred_tr = model(X_tr)
        loss_tr = criterion(y_pred_tr, y_tr, X_unscaled_tr)

        loss_tr.backward()
        optimizer.step()

        # --- Validation Phase ---
        model.eval()
        with torch.no_grad():
            y_pred_va = model(X_va)
            loss_va = criterion(y_pred_va, y_va, X_unscaled_va)

        train_losses.append(loss_tr.item())
        val_losses.append(loss_va.item())

        # Early Stopping check
        if loss_va.item() < best_val_loss:
            best_val_loss = loss_va.item()
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve == patience:
            print(f"Early stopping triggered at epoch {epoch}. Restoring best model weights.")
            break

        if epoch % 500 == 0:
            print(f"Epoch {epoch} | Train Loss: {loss_tr.item():.4f} | Val Loss: {loss_va.item():.4f}")

    model.load_state_dict(best_model_wts)
    return copy, model


@app.cell
def _(X_va, model, np, torch, y_scale, y_va):
    model.eval()

    with torch.no_grad():
        y_va_pred_scaled = model(X_va).cpu().numpy()

    y_va_pred_seconds = y_scale*y_va_pred_scaled
    y_va_true_seconds = y_scale*y_va.cpu().numpy()

    y_va_pred_hours = y_va_pred_seconds / 3600.0
    y_va_true_hours = y_va_true_seconds / 3600.0

    absolute_errors_hours = np.abs(y_va_pred_hours - y_va_true_hours)
    relative_errors = absolute_errors_hours / y_va_true_hours

    metrics_summary = {
        "MAE_mean": np.mean(absolute_errors_hours),
        "MAE_median": np.median(absolute_errors_hours),
        "MAE_min": np.min(absolute_errors_hours),
        "MAE_max": np.max(absolute_errors_hours),
        "RAE_mean": np.mean(relative_errors),
        "RAE_median": np.median(relative_errors),
        "RAE_min": np.min(relative_errors),
        "RAE_max": np.max(relative_errors)
    }

    # Print the evaluation report
    print("=== N2 Model Evaluation Summary (Validation Set) ===")
    print(f"Mean Absolute Error (MAE): {metrics_summary['MAE_mean']:.2f} hours")
    print(f"Median Absolute Error:     {metrics_summary['MAE_median']:.2f} hours")
    print(f"Min / Max Absolute Error:  {metrics_summary['MAE_min']:.2f} / {metrics_summary['MAE_max']:.2f} hours")
    print("-" * 50)
    print(f"Mean Relative Absolute Error:   {metrics_summary['RAE_mean']:.4f}")
    print(f"Median Relative Absolute Error: {metrics_summary['RAE_median']:.4f}")
    return absolute_errors_hours, y_va_pred_hours, y_va_true_hours


@app.cell
def _(absolute_errors_hours, plt, y_va_pred_hours, y_va_true_hours):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Scatterplot of Actual vs Predicted Travel Times (Ref: Figure 5)
    ax1.scatter(y_va_true_hours, y_va_pred_hours, color='blue', alpha=0.7, edgecolors='k', label='CME Events')
    ideal_line = [min(y_va_true_hours), max(y_va_true_hours)]
    ax1.plot(ideal_line, ideal_line, color='black', linestyle='--', linewidth=2, label='Perfect Prediction')
    ax1.set_xlabel('Actual Transit Time (hours)')
    ax1.set_ylabel('Predicted Transit Time (hours)')
    ax1.set_title('Actual vs. Predicted CME Travel Times')
    ax1.legend()
    ax1.grid(True, linestyle=':', alpha=0.6)

    # 2. Histogram of Absolute Error Distributions (Ref: Figure 4)
    ax2.hist(absolute_errors_hours, bins=20, color='g', edgecolor='black', alpha=0.7)
    ax2.set_xlabel('Absolute Error (hours)')
    ax2.set_ylabel('Number of CME Events')
    ax2.set_title('Distribution of Prediction Absolute Errors')
    ax2.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _():
    return


@app.cell
def _(
    AU_KM,
    L_t,
    N2Model,
    R0_KM,
    StandardScaler,
    X,
    copy,
    init_weights,
    np,
    s,
    torch,
    y,
):
    def run_one_split(seed, max_epochs=10000, patience=1000):
        print(s)
    
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(X))
        n_tr = int(0.8 * len(X))
        tr, va = perm[:n_tr], perm[n_tr:]

        xs = StandardScaler().fit(X[tr])
        X_tr = torch.tensor(xs.transform(X[tr]), dtype=torch.float32)
        X_va = torch.tensor(xs.transform(X[va]), dtype=torch.float32)
        Xu_tr = torch.tensor(X[tr], dtype=torch.float32)
        Xu_va = torch.tensor(X[va], dtype=torch.float32)

        ys = y[tr].std()
        y_tr = torch.tensor((y[tr] / ys).reshape(-1, 1), dtype=torch.float32)
        y_va = torch.tensor((y[va] / ys).reshape(-1, 1), dtype=torch.float32)

        torch.manual_seed(seed)
        model = N2Model(input_size=6); model.apply(init_weights)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        crit = L_t(ys, AU_KM, R0_KM, lambda_val=0.5)

        best, best_wts, no_imp = float('inf'), copy.deepcopy(model.state_dict()), 0
        for ep in range(max_epochs):
            model.train(); opt.zero_grad()
            crit(model(X_tr), y_tr, Xu_tr).backward(); opt.step()
            model.eval()
            with torch.no_grad():
                v = crit(model(X_va), y_va, Xu_va).item()
            if v < best:
                best, best_wts, no_imp = v, copy.deepcopy(model.state_dict()), 0
            else:
                no_imp += 1
                if no_imp == patience: break
        model.load_state_dict(best_wts); model.eval()
        with torch.no_grad():
            pred = model(X_va).numpy().flatten() * ys / 3600.0
        return np.abs(pred - y[va] / 3600.0)

    errs = [run_one_split(s) for s in range(10)]
    split_mae = np.array([e.mean() for e in errs])
    pooled = np.concatenate(errs)

    print(f"Per-split mean MAE: median {np.median(split_mae):.2f} h, "
          f"IQR [{np.percentile(split_mae,25):.2f}, {np.percentile(split_mae,75):.2f}] h")
    print(f"Pooled errors: median {np.median(pooled):.2f} h, "
          f"STD {pooled.std():.2f} h  (paper C5 STD ~7.5 h)")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
