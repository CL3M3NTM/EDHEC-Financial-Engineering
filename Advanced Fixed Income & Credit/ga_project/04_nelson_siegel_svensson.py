"""
Question 3.1 -- Nelson-Siegel-Svensson fit to the bootstrapped zero curve.
Question 3.2 -- Factor loadings and contributions, and their economic drivers.

NSS yield function:
  y(m) = b0
       + b1 * [(1-exp(-m/tau1))/(m/tau1)]
       + b2 * [(1-exp(-m/tau1))/(m/tau1) - exp(-m/tau1)]
       + b3 * [(1-exp(-m/tau2))/(m/tau2) - exp(-m/tau2)]

  b0  -- LEVEL factor:      loading = 1 for all maturities (long-run yield level)
  b1  -- SLOPE factor:      loading decays monotonically from 1 (m->0) to 0 (m->inf)
  b2  -- CURVATURE factor:  loading is a hump, 0 at m->0 and m->inf, peak around tau1
  b3  -- 2nd CURVATURE (Svensson extension): a second hump around tau2

Methodology (documented, per the assignment's instructions):
  Jointly estimating (b0,b1,b2,b3,tau1,tau2) by unconstrained nonlinear least
  squares is numerically fragile here: because this curve is concave and
  saturates smoothly with NO pronounced second hump, beta0 (level) and beta2
  (curvature) become highly collinear whenever the optimizer pushes tau1 out
  past ~10 years (the curvature loading then barely decays back to zero
  within our 26y maturity range, so "level" and "curvature-still-active" are
  almost indistinguishable to the optimizer) -- different, equally-well-
  fitting local optima were found with wildly different, uninterpretable
  (beta0, beta2) splits across random restarts.

  We therefore use the standard **Diebold-Li two-step procedure**, which is
  both the textbook fix for this exact pathology and standard practice at
  central banks/commercial curve providers: for each CANDIDATE tau1 (and
  tau2) on a fine grid, the loadings become fixed basis functions of
  maturity, so the betas enter LINEARLY and can be solved by ordinary least
  squares (a well-posed, unique, convex problem for every candidate tau).
  We then simply pick the (tau1, tau2) pair on the grid that minimises the
  total residual sum of squares. This removes all the non-linear-optimizer
  fragility -- the only thing solved non-linearly is a 1-D (or 2-D) grid
  search over tau, which is robust and simple to inspect for sanity.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

curve = pd.read_csv('outputs/q1_bootstrapped_curve.csv')
m = curve['Years'].values
y_obs = curve['ZeroYield_BEY'].values

def nss_loadings(m, tau1, tau2):
    m = np.asarray(m, dtype=float)
    x1 = m / tau1
    load_level = np.ones_like(m)
    load_slope = (1 - np.exp(-x1)) / x1
    load_curv1 = load_slope - np.exp(-x1)
    if tau2 is not None:
        x2 = m / tau2
        load_curv2 = (1 - np.exp(-x2)) / x2 - np.exp(-x2)
    else:
        load_curv2 = None
    return load_level, load_slope, load_curv1, load_curv2

def ols_fit(y, L, S, C1, C2=None):
    cols = [L, S, C1] + ([C2] if C2 is not None else [])
    X = np.column_stack(cols)
    betas, *_ = np.linalg.lstsq(X, y, rcond=None)
    fitted = X @ betas
    rss = np.sum((fitted - y) ** 2)
    return betas, fitted, rss

# --- Step 1: Diebold-Li grid search for plain (3-factor) Nelson-Siegel ---
tau1_grid = np.linspace(0.3, 25, 500)
best_rss, best_tau1, best_betas_ns = np.inf, None, None
for tau1 in tau1_grid:
    L, S, C1, _ = nss_loadings(m, tau1, None)
    betas, fitted, rss = ols_fit(y_obs, L, S, C1)
    if rss < best_rss:
        best_rss, best_tau1, best_betas_ns = rss, tau1, betas

b0_ns, b1_ns, b2_ns = best_betas_ns
rmse_ns = np.sqrt(best_rss / len(y_obs)) * 10000
print("=== Plain Nelson-Siegel (3-factor), Diebold-Li grid search ===")
print(f"  beta0 (level)     = {b0_ns*100:.3f}%")
print(f"  beta1 (slope)     = {b1_ns*100:.3f}%")
print(f"  beta2 (curvature) = {b2_ns*100:.3f}%")
print(f"  tau1              = {best_tau1:.3f} years")
print(f"  RMSE              = {rmse_ns:.2f} bps")

# --- Step 2: 2-D grid search for the full Svensson (4-factor) model, to
# check whether the second curvature term earns its keep ---
tau1_grid2 = np.linspace(0.3, 15, 80)
tau2_grid2 = np.linspace(2, 30, 80)
best_rss_sv, best_t1_sv, best_t2_sv, best_betas_sv = np.inf, None, None, None
for tau1 in tau1_grid2:
    for tau2 in tau2_grid2:
        if abs(tau1 - tau2) < 0.5:
            continue  # avoid near-degenerate tau1~tau2
        L, S, C1, C2 = nss_loadings(m, tau1, tau2)
        betas, fitted, rss = ols_fit(y_obs, L, S, C1, C2)
        if rss < best_rss_sv:
            best_rss_sv, best_t1_sv, best_t2_sv, best_betas_sv = rss, tau1, tau2, betas

b0_sv, b1_sv, b2_sv, b3_sv = best_betas_sv
rmse_sv = np.sqrt(best_rss_sv / len(y_obs)) * 10000
print("\n=== Full Svensson (4-factor), 2-D Diebold-Li grid search ===")
print(f"  beta0={b0_sv*100:.3f}%  beta1={b1_sv*100:.3f}%  beta2={b2_sv*100:.3f}%  beta3={b3_sv*100:.3f}%")
print(f"  tau1={best_t1_sv:.3f}y  tau2={best_t2_sv:.3f}y")
print(f"  RMSE = {rmse_sv:.2f} bps  (improvement over plain NS: {rmse_ns-rmse_sv:.2f} bps)")

# --- Decision rule (documented assumption) ---
IMPROVEMENT_THRESHOLD_BPS = 1.5
if rmse_ns - rmse_sv < IMPROVEMENT_THRESHOLD_BPS:
    print(f"\n==> Decision: Svensson's 2nd curvature factor improves RMSE by only "
          f"{rmse_ns-rmse_sv:.2f}bps (< {IMPROVEMENT_THRESHOLD_BPS}bps threshold) -- our curve has a "
          f"single, mild hump with no evidence of a second one, so we ADOPT THE SIMPLER, better-")
    print("    identified 3-factor Nelson-Siegel as our final model.")
    b0, b1, b2, b3, tau1, tau2 = b0_ns, b1_ns, b2_ns, 0.0, best_tau1, best_tau1 * 3
    fitted_final = nss_yield_eval(m, b0, b1, b2, b3, tau1, tau2) if False else (
        b0_ns*nss_loadings(m, best_tau1, None)[0] + b1_ns*nss_loadings(m, best_tau1, None)[1]
        + b2_ns*nss_loadings(m, best_tau1, None)[2])
else:
    print(f"\n==> Decision: Svensson's 2nd curvature factor improves RMSE by "
          f"{rmse_ns-rmse_sv:.2f}bps -- we ADOPT THE FULL SVENSSON model.")
    b0, b1, b2, b3, tau1, tau2 = b0_sv, b1_sv, b2_sv, b3_sv, best_t1_sv, best_t2_sv
    L, S, C1, C2 = nss_loadings(m, tau1, tau2)
    fitted_final = b0*L + b1*S + b2*C1 + b3*C2

resid_bps = (fitted_final - y_obs) * 10000
print(f"\nFinal adopted model -- RMSE = {np.sqrt(np.mean(resid_bps**2)):.2f} bps, "
      f"max abs error = {np.max(np.abs(resid_bps)):.2f} bps")
print(f"  beta0 (level)      = {b0*100: .3f}%")
print(f"  beta1 (slope)      = {b1*100: .3f}%")
print(f"  beta2 (curvature1) = {b2*100: .3f}%")
print(f"  beta3 (curvature2) = {b3*100: .3f}%")
print(f"  tau1 = {tau1:.3f}y   tau2 = {tau2:.3f}y")

curve['NSS_Yield'] = fitted_final
curve.to_csv('outputs/q3_nss_fit.csv', index=False)
with open('outputs/q3_nss_params.csv', 'w') as f:
    f.write("param,value\n")
    for name, val in zip(['beta0','beta1','beta2','beta3','tau1','tau2'], [b0,b1,b2,b3,tau1,tau2]):
        f.write(f"{name},{val}\n")

# --- Plot 1: NSS/NS fit vs bootstrapped curve ---
m_fine = np.linspace(0.1, 26, 300)
L_f, S_f, C1_f, C2_f = nss_loadings(m_fine, tau1, tau2)
y_fine = b0*L_f + b1*S_f + b2*C1_f + b3*(C2_f if C2_f is not None else 0)

fig, ax = plt.subplots(figsize=(9, 5.5))
ax.plot(m, y_obs*100, 'o', ms=3, color='#4D4D4D', label='Bootstrapped zero yields (Q1.1)')
ax.plot(m_fine, y_fine*100, '-', lw=1.8, color='#9E1B32',
        label=f'Nelson-Siegel fit (RMSE={np.sqrt(np.mean(resid_bps**2)):.1f}bps)')
ax.set_xlabel('Maturity (years)')
ax.set_ylabel('Zero-coupon yield, BEY (%)')
ax.set_title('Q3.1 -- Nelson-Siegel(-Svensson) fit vs. bootstrapped curve')
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig('outputs/q3_nss_fit.png', dpi=150)

# --- Plot 2: factor loadings ---
fig2, ax2 = plt.subplots(figsize=(9, 5.5))
ax2.plot(m_fine, L_f, label='Level loading (=1, const.)', color='#4D4D4D')
ax2.plot(m_fine, S_f, label='Slope loading', color='#9E1B32')
ax2.plot(m_fine, C1_f, label=f'Curvature loading (peaks near tau1={tau1:.1f}y)', color='#1B6B9E')
if b3 != 0.0:
    ax2.plot(m_fine, C2_f, label=f'Curvature-2 loading (peaks near tau2={tau2:.1f}y)', color='#2E9E1B')
ax2.set_xlabel('Maturity (years)')
ax2.set_ylabel('Factor loading')
ax2.set_title('Q3.2 -- Nelson-Siegel factor loadings')
ax2.legend(frameon=False, fontsize=9)
fig2.tight_layout()
fig2.savefig('outputs/q3_nss_loadings.png', dpi=150)

# --- Plot 3: factor CONTRIBUTIONS (loading x beta) ---
fig3, ax3 = plt.subplots(figsize=(9, 5.5))
ax3.plot(m_fine, b0*L_f*100, label=f'Level contribution (b0={b0*100:.2f}%)', color='#4D4D4D')
ax3.plot(m_fine, b1*S_f*100, label=f'Slope contribution (b1={b1*100:.2f}%)', color='#9E1B32')
ax3.plot(m_fine, b2*C1_f*100, label=f'Curvature contribution (b2={b2*100:.2f}%)', color='#1B6B9E')
if b3 != 0.0:
    ax3.plot(m_fine, b3*C2_f*100, label=f'Curvature-2 contribution (b3={b3*100:.2f}%)', color='#2E9E1B')
ax3.plot(m_fine, y_fine*100, '--', lw=2, color='black', label='Total fitted yield')
ax3.axhline(0, color='grey', lw=0.5)
ax3.set_xlabel('Maturity (years)')
ax3.set_ylabel('Contribution to yield (%)')
ax3.set_title('Q3.2 -- Nelson-Siegel factor contributions (beta_i x loading_i)')
ax3.legend(frameon=False, fontsize=9)
fig3.tight_layout()
fig3.savefig('outputs/q3_nss_contributions.png', dpi=150)

print("\nSaved: q3_nss_fit.png, q3_nss_loadings.png, q3_nss_contributions.png")
