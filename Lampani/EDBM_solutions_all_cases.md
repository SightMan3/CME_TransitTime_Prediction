# EDBM closed-form solutions — all cases

Extracted from Rossi et al. (2025), *A&A* **694**, A247. Equation numbers in brackets refer to that paper.

## The model

The extended drag-based model is the single equation of motion [Eq. 2]:

$$\ddot{r} = -\gamma\,|\dot{r}-w|\,(\dot{r}-w) + a$$

with initial conditions $r(0)=r_0$, $\dot r(0)=v_0$, constant solar wind speed $w$, drag parameter $\gamma>0$, and constant extra acceleration $a\neq 0$. Setting $a=0$ recovers the classical DBM.

**Equilibrium (asymptotic) speeds.** The added acceleration shifts the stable equilibrium away from the DBM value $v=w$:

- if $a>0$: stable equilibrium at $v = w + \sqrt{a/\gamma}$;
- if $a<0$: stable equilibrium at $v = w - \sqrt{-a/\gamma}$.

So for $a>0$ a CME can be pushed *above* $w$, and for $a<0$ it can be dragged *below* $w$ — both impossible in the DBM, whose only fixed point is $w$.

There are exactly **four** solution families, set by $\operatorname{sign}(a)$ and whether $v_0\le w$ or $v_0>w$. Two of them are piecewise (the speed crosses $w$ mid-flight); each piece corresponds to a different named regime. That is how four families produce the six regimes of Lampani's Table 1.

---

## Shared constants

$$\sigma_+ := \arctan\!\Big(\sqrt{\dfrac{\gamma}{a}}\,(w-v_0)\Big), \qquad
S_+ := \sqrt{\dfrac{a+\gamma(v_0-w)^2}{a}} \qquad (a>0)$$

$$\sigma_- := \arctan\!\Big(\sqrt{-\dfrac{\gamma}{a}}\,(v_0-w)\Big), \qquad
S_- := \sqrt{\dfrac{a-\gamma(v_0-w)^2}{a}} \qquad (a<0)$$

$$A_+ := \sqrt{\gamma}\,(v_0-w) + \sqrt{a}, \qquad B_+ := \sqrt{\gamma}\,(v_0-w) - \sqrt{a} \qquad (a>0,\ v_0>w)$$

$$A_- := \sqrt{\gamma}\,(v_0-w) + \sqrt{-a}, \qquad B_- := \sqrt{\gamma}\,(v_0-w) - \sqrt{-a} \qquad (a<0,\ v_0\le w)$$

The two "crossing times" (when $v$ reaches $w$) are

$$t^*_+ = \frac{\sigma_+}{\sqrt{a\gamma}} \quad (a>0), \qquad\qquad t^*_- = \frac{\sigma_-}{\sqrt{-a\gamma}} \quad (a<0).$$

---

## Family 1 — $a>0,\ v_0\le w$  [Eqs. 3–4]

Speed starts at/below $w$ and is pushed up; it reaches $w$ at $t^*_+$ and continues toward $w+\sqrt{a/\gamma}$.

**Speed:**

$$v(t)=\begin{cases}
\displaystyle w+\sqrt{\dfrac{a}{\gamma}}\,\tan\!\big(\sqrt{a\gamma}\,t-\sigma_+\big), & 0\le t\le t^*_+ \\
\displaystyle w+\sqrt{\dfrac{a}{\gamma}}\;\frac{e^{\,2(\sqrt{a\gamma}\,t-\sigma_+)}-1}{e^{\,2(\sqrt{a\gamma}\,t-\sigma_+)}+1}, & t> t^*_+
\end{cases}$$

**Position:**

$$r(t)=\begin{cases}
\displaystyle wt+r_0-\dfrac{1}{\gamma}\ln\!\big(S_+\cos(\sqrt{a\gamma}\,t-\sigma_+)\big), & 0\le t\le t^*_+ \\
\displaystyle \Big(w-\sqrt{\dfrac{a}{\gamma}}\Big)t+r_0+\dfrac{1}{\gamma}\!\left[\ln\!\Big(\frac{e^{\,2(\sqrt{a\gamma}\,t-\sigma_+)}+1}{2S_+}\Big)+\sigma_+\right], & t> t^*_+
\end{cases}$$

- **First branch ($t\le t^*_+$):** the CME stays $\le w$ → **Sub-wind (↗)**. Valid time domain $[0,\,t^*_+]$.
- **Second branch ($t> t^*_+$):** the CME has crossed above $w$ → **Cross-wind (↗)** ($v_0<w<v$). Valid time domain $(t^*_+,\,+\infty)$.

---

## Family 2 — $a>0,\ v_0> w$  [Eqs. 5–6]  →  **Super-wind (↗)**

Speed stays above $w$ for all $t\ge 0$, rising toward $w+\sqrt{a/\gamma}$. Time domain $[0,+\infty)$.

$$v(t)= w+\sqrt{\dfrac{a}{\gamma}}\;\frac{A_+\,e^{\,2\sqrt{a\gamma}\,t}+B_+}{A_+\,e^{\,2\sqrt{a\gamma}\,t}-B_+}, \qquad t\ge 0$$

$$r(t)= \Big(w-\sqrt{\dfrac{a}{\gamma}}\Big)t+r_0+\dfrac{1}{\gamma}\ln\!\Big(\frac{A_+\,e^{\,2\sqrt{a\gamma}\,t}-B_+}{2\sqrt{a}}\Big), \qquad t\ge 0$$

---

## Family 3 — $a<0,\ v_0\le w$  [Eqs. 7–8]  →  **Sub-wind (↘)**

Speed stays below $w$ for all $t\ge 0$, decaying toward $w-\sqrt{-a/\gamma}$. Time domain $[0,+\infty)$.

$$v(t)= w+\sqrt{-\dfrac{a}{\gamma}}\;\frac{A_-\,e^{\,-2\sqrt{-a\gamma}\,t}+B_-}{A_-\,e^{\,-2\sqrt{-a\gamma}\,t}-B_-}, \qquad t\ge 0$$

$$r(t)= \Big(w-\sqrt{-\dfrac{a}{\gamma}}\Big)t+r_0-\dfrac{1}{\gamma}\ln\!\Big(\frac{A_-\,e^{\,-2\sqrt{-a\gamma}\,t}-B_-}{2\sqrt{-a}}\Big), \qquad t\ge 0$$

---

## Family 4 — $a<0,\ v_0> w$  [Eqs. 9–10]

Speed starts above $w$ and is dragged down; it reaches $w$ at $t^*_-$ and continues toward $w-\sqrt{-a/\gamma}$.

**Speed:**

$$v(t)=\begin{cases}
\displaystyle w-\sqrt{-\dfrac{a}{\gamma}}\,\tan\!\big(\sqrt{-a\gamma}\,t-\sigma_-\big), & 0\le t\le t^*_- \\
\displaystyle w+\sqrt{-\dfrac{a}{\gamma}}\;\frac{e^{\,-2(\sqrt{-a\gamma}\,t-\sigma_-)}-1}{e^{\,-2(\sqrt{-a\gamma}\,t-\sigma_-)}+1}, & t> t^*_-
\end{cases}$$

**Position:**

$$r(t)=\begin{cases}
\displaystyle wt+r_0+\dfrac{1}{\gamma}\ln\!\big(S_-\cos(\sqrt{-a\gamma}\,t-\sigma_-)\big), & 0\le t\le t^*_- \\
\displaystyle \Big(w-\sqrt{-\dfrac{a}{\gamma}}\Big)t+r_0-\dfrac{1}{\gamma}\!\left[\ln\!\Big(\frac{e^{\,-2(\sqrt{-a\gamma}\,t-\sigma_-)}+1}{2S_-}\Big)-\sigma_-\right], & t> t^*_-
\end{cases}$$

- **First branch ($t\le t^*_-$):** the CME stays $\ge w$ → **Super-wind (↘)**. Valid time domain $[0,\,t^*_-]$.
- **Second branch ($t> t^*_-$):** the CME has crossed below $w$ → **Cross-wind (↘)** ($v_0>w>v$). Valid time domain $(t^*_-,\,+\infty)$. **This second branch is exactly Lampani's Eq. (3).**

---

## Regime map (Lampani Table 1 ↔ Rossi equations)

| Regime | Condition | $\operatorname{sign}(a)$ | Rossi eqs | Which branch | Time domain |
|---|---|---|---|---|---|
| Sub-wind (↗) | $v_0,v\le w,\ v_0\le v$ | $+$ (or $0$) | 3, 4 | 1st | $[0,\,t^*_+]$ |
| Cross-wind (↗) | $v_0<w<v$ | $+$ | 3, 4 | 2nd | $(t^*_+,\,+\infty)$ |
| Super-wind (↗) | $v_0,v\ge w,\ v_0\le v$ | $+$ | 5, 6 | single | $[0,\,+\infty)$ |
| Sub-wind (↘) | $v_0,v\le w,\ v_0> v$ | $-$ | 7, 8 | single | $[0,\,+\infty)$ |
| Super-wind (↘) | $v_0,v\ge w,\ v_0> v$ | $-$ (or $0$) | 9, 10 | 1st | $[0,\,t^*_-]$ |
| Cross-wind (↘) | $v_0>w>v$ | $-$ | 9, 10 | 2nd | $(t^*_-,\,+\infty)$ |

**Note.** Sub-wind (↗) and Super-wind (↘) are the two DBM-compatible regimes: they admit $a=0$, and as $a\to 0$ the formulas above continuously reduce to the DBM solution (Vršnak et al. 2013). The other four require $a\neq 0$.

---

## Implementation notes for an all-regime solver

1. **One $r(t)$ per regime in the loss.** To extend the Lampani pipeline beyond Cross-wind (↘), drop the correct branch above into the physics term of the loss (their Eq. 5), replacing $r_{\text{Cross-wind}(\searrow)}$ with the regime's $r_c(t)$.

2. **Stage-(i) acceleration estimate per regime.** Solve $r_c(t;a,x)=1\,\mathrm{AU}$ for $a$ by Newton–Raphson, exactly as Lampani does. Rossi solves the analogous root problems $f_v(a)=v(a;\dots)-v_{\text{target}}=0$ [Eq. 14] or $f_r(a)=r(a;\dots)-r_{\text{target}}=0$ [Eq. 15], initialising $a$ near the drag-acceleration magnitude and setting $a=0$ if the scheme fails to converge.

3. **Always validate the time domain after solving for $a$.** Both $t^*_+$ and $t^*_-$ depend on the unknown $a$. For the bounded regimes (Sub-wind ↗, Super-wind ↘) and for selecting the correct branch of the crossing regimes, recompute $t^*$ once $a$ is found and **reject the solution if the predicted arrival time violates the branch's time domain** (this is the same filtering Rossi and Lampani apply). For a crossing event you specifically need the arrival to fall in the second branch, i.e. predicted $t > t^*$.

4. **Numerical care near $a\to 0$.** The trig-form branches (Families 1 & 4 first branch) carry $\sqrt{a\gamma}$ / $\sqrt{-a\gamma}$ and $\arctan$ terms that are well-defined but stiff for very small $|a|$; for the two DBM-compatible regimes it is cleaner to fall back to the closed-form DBM solution when $|a|$ is below a tolerance, consistent with the $a\to 0$ limit.
