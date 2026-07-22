import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import pandas as pd
    import numpy as np
    import marimo as mo
    import matplotlib.pyplot as plt
    from hapiclient import hapi
    from datetime import datetime, timedelta
    import time

    return mo, np, pd


@app.cell
def _():
    r_0 = 1.3914*10**7
    AU_KM = 1.495978707e8
    return AU_KM, r_0


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # N1 features

    1. $v_0$ - initial speed, in dataset: $v_r$
    2. $m$ - mass, v datasete Mass
    3. $A$ - area, vypocitame z $rel_{wid}$ v datasete
    4. $\rho$ - density, ziskame z SOHO\CELIAS
    5. $\omega$ - solar wind speed, ziskame z SOHO\CELIAS

    vystup $N_1$ je $C$, bezdimenzionalny koeficient odporu.

    # N2 features
    1. $v_0$ - initial speed, in dataset: $v_r$
    2. $m$ - mass
    3. $A$ - area
    4. $\rho$ - density
    5. $\omega$ - solar wind speed
    6. $C$ z $N_1$

    vystup $N_2$ je $TT$, travel time.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Wind speed extraction
    Rychlost vetra sa berie ako priemer jednej hodiny po zacati eventu, netrapi nas kde je CME, iba rychlost vetra v L1 v tom case kedy ale CME ide, presnejsie ked je $20R\odot$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Vypocet Area z rel_wid

    $$
    A = \pi\left(r_0 \sin\left(\frac{rel_{wid}}{2}\right)\right)^2
    $$

    kde $r_0 = 20 R \odot = 1.3914\times10^7 \text{km}$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Loss function $L(t, N1(x))$ for N1
    $$
    L_c(t, N1(x)) = \left(\frac{m\sigma}{A\rho N1(x)}\log\left(1 + \sigma\frac{A\rho N1(x)}{m}(v_0 - w)t + wt\right) + r_0 - 1\right)^2
    $$

    kde
    $$
    \sigma = \frac{v_0 - w}{\sqrt{(v_0 - w)^2 + \delta}}
    $$
    kde $\delta$ je mala kladna hodnota
    """)
    return


@app.cell
def _(np, pd, r_0):
    napoletano_set = pd.read_csv('ICME_complete_dataset_v3.csv')

    napoletano_set = napoletano_set.iloc[1:].copy()

    cols_to_convert = ['Mass', 'v_r', 'rel_wid', 'Transit_time']
    napoletano_set[cols_to_convert] = napoletano_set[cols_to_convert].apply(pd.to_numeric, errors='coerce')

    sabrina_set = napoletano_set.loc[
        (napoletano_set['Mass'] != -9999.0), 
        ['Start_Date', 'Transit_time', 'v_r', 'Mass', 'rel_wid']
    ].copy()


    sabrina_set['A'] = np.pi * (r_0 * np.sin(sabrina_set['rel_wid'] / 2)) ** 2

    sabrina_set
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # SOHO\CELIAS $\rho$ a $\omega$ data
    """)
    return


@app.cell
def _(pd):
    sabrina_set_old = pd.read_csv('sabrina_set.csv')
    sabrina_set_old.describe()
    return


@app.cell
def _(AU_KM, np, r_0, sabrina_set_w_rho):
    df = sabrina_set_w_rho.copy()

    tt_s  = df['Transit_time'].to_numpy() * 3600.0
    v_bar = (AU_KM - r_0) / tt_s
    v0    = df['v_r'].to_numpy()
    w     = df['w_km_s'].to_numpy()

    lo, hi = np.minimum(v0, w), np.maximum(v0, w)
    final_set = df[(v_bar >= lo) & (v_bar <= hi)]
    len(final_set)
    return (final_set,)


@app.cell
def _(final_set):
    final_set[
        (final_set['rho_g_km3'] <= 200) 
    ].to_csv('./sabrina_set.csv')
    return


if __name__ == "__main__":
    app.run()
