"""
Part Filippo
Question 4.1 -- Shock the 10-year key rate by 50bps; impact on the Q2.1 spline
                 zero-coupon curve. Plot the difference.
Question 4.2 -- KRD of a $1M, equally-weighted (across all 52 bonds) portfolio
                 to that 10Y shock.
Question 4.3 -- KRD to each of the 5 key rates (2,5,10,15,20y) individually.

Methodology & documented assumptions:
  - We work off the Q2.1 CUBIC SPLINE (knots at 2,5,10,15,20y), not the raw
    per-bond bootstrap or the NSS fit, because a "key-rate" shock is only a
    well-defined operation relative to a smooth curve MODEL with discrete
    nodes -- the spline's knots are exactly the assignment's key rates.
  - Each key-rate shock is the standard triangular ("tent") perturbation:
    +1x at its own knot, decaying linearly to 0 at the two adjacent knots,
    and held FLAT at its boundary value outside the outer knots (i.e. the
    2Y shock stays at full size for all maturities <= 2y; the 20Y shock
    stays at full size for all maturities >= 20y). This is the standard
    construction (Ho, 1992) which guarantees that the sum of all 5 key-rate
    shocks equals a uniform (parallel) shift of 1 at every maturity -- a
    useful internal consistency check we run below.
  - "Equally invested in all bonds" (Q4.2) is interpreted, as a documented
    assumption, as an equal DOLLAR MARKET VALUE in each of the 52 bonds:
    $1,000,000 / 52 ~= $19,230.77 of current market value in each bond,
    translated into a face-value holding of (that dollar amount / current
    market price) x 100 for each bond.
  - KRD is reported in DOLLAR terms (the P&L of the shock), consistent with
    how KRD was introduced and used in the course (a dollar risk measure,
    directly comparable to -- and summable with -- other dollar greeks);
    we also report it as a percentage of the $1M NAV for interpretability.
  - Pricing (base case AND shocked case) uses the SPLINE-implied yields (not
    the exact bootstrap), consistently, so that the KRD measures the
    portfolio's sensitivity to the curve MODEL, which is the entire point of
    a key-rate-duration exercise; the resulting "model NAV" therefore does
    not exactly equal $1,000,000 even before any shock (it differs by the
    Q2.2 spline fitting residual) -- we report and flag this base-case
    pricing error explicitly, so it is not confused with a hedging result.
"""
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

bonds = pd.read_csv('data/exhibit1_bonds.csv')
bonds = bonds[pd.to_numeric(bonds['Coupon'], errors='coerce').notna()].copy()
bonds['Maturity'] = pd.to_datetime(bonds['Maturity'])
bonds['Coupon'] = bonds['Coupon'].astype(float)
bonds['Price'] = bonds['Price'].astype(float)
bonds = bonds.sort_values('Maturity').reset_index(drop=True)
from datetime import date
SETTLEMENT = date(2003, 8, 15)
bonds['Periods'] = ((bonds['Maturity'].dt.year - SETTLEMENT.year) * 2
                     + (bonds['Maturity'].dt.month - SETTLEMENT.month) // 6)

curve = pd.read_csv('outputs/q2_spline_vs_bootstrap.csv')
all_periods = curve['Period'].values
all_years = curve['Years'].values
spline_yield_base = curve['SplineYield'].values  # from Q2.1

KEY_RATE_KNOTS = [2, 5, 10, 15, 20]

def triangular_shock(t, knots, j):
    k = knots[j]
    k_prev = knots[j-1] if j > 0 else None
    k_next = knots[j+1] if j < len(knots)-1 else None
    shock = np.zeros_like(t, dtype=float)
    if k_prev is None:
        shock[t <= k] = 1.0
    else:
        m = (t > k_prev) & (t <= k)
        shock[m] = (t[m] - k_prev) / (k - k_prev)
    if k_next is None:
        shock[t > k] = 1.0
    else:
        m = (t > k) & (t < k_next)
        shock[m] = (k_next - t[m]) / (k_next - k)
    return shock

# sanity check: the 5 key-rate shocks should sum to 1 everywhere (parallel shift)
total_shock_check = sum(triangular_shock(all_years, KEY_RATE_KNOTS, j) for j in range(5))
assert np.allclose(total_shock_check, 1.0), "Key-rate shocks do not sum to a parallel shift!"
print("Sanity check passed: the 5 triangular key-rate shocks sum to a parallel shift everywhere.\n")

def yield_to_discount(y_bey, t_years):
    """BEY (semi-annual compounding) yield -> discount factor."""
    return (1 + y_bey/2.0) ** (-2.0 * t_years)

def price_bond(coupon, periods, disc_factors_by_period):
    """disc_factors_by_period: dict period -> discount factor"""
    c = coupon/2.0
    pv = sum(c*disc_factors_by_period[p] for p in range(1, periods))
    pv += (c+100.0)*disc_factors_by_period[periods]
    return pv

def price_portfolio(disc_factors_by_period, weights_face):
    total = 0.0
    for _, row in bonds.iterrows():
        p = price_bond(row['Coupon'], int(row['Periods']), disc_factors_by_period)
        total += weights_face[row.name] * p / 100.0  # price is per 100 face
    return total

# --- Portfolio construction: $1M equally split in dollar market value across 52 bonds ---
NAV_TARGET = 1_000_000.0
n_bonds = len(bonds)
dollar_per_bond = NAV_TARGET / n_bonds
face_per_bond = dollar_per_bond / bonds['Price'] * 100.0  # face amount bought of each bond
weights_face = {i: face_per_bond.iloc[i] for i in range(n_bonds)}

# base discount factors from the (unshocked) Q2.1 spline yield, at every node
base_disc = {int(p): yield_to_discount(y, t) for p, t, y in zip(all_periods, all_years, spline_yield_base)}
base_NAV_model = price_portfolio(base_disc, weights_face)
print(f"Target NAV (sum of market values at actual market prices): ${NAV_TARGET:,.2f}")
print(f"Base-case 'model NAV' (repriced off the Q2.1 spline curve): ${base_NAV_model:,.2f}")
print(f"  -> difference: ${base_NAV_model-NAV_TARGET:,.2f} "
      f"({(base_NAV_model/NAV_TARGET-1)*100:.3f}%), reflecting the Q2.2 spline fitting residual.\n")

SHOCK_SIZE = 0.0050  # 50 bps

# ============== Q4.1: shock only the 10Y key rate, plot curve impact ==============
j10 = KEY_RATE_KNOTS.index(10)
shock_vec_10y = triangular_shock(all_years, KEY_RATE_KNOTS, j10)
shocked_yield_10y = spline_yield_base + SHOCK_SIZE * shock_vec_10y

fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
axes[0].plot(all_years, spline_yield_base*100, color='#4D4D4D', label='Base spline curve')
axes[0].plot(all_years, shocked_yield_10y*100, color='#9E1B32', label='Shocked curve (+50bps @ 10Y)')
axes[0].set_xlabel('Maturity (years)'); axes[0].set_ylabel('Zero yield, BEY (%)')
axes[0].set_title('Q4.1 -- Curve level: base vs. 10Y-shocked'); axes[0].legend(frameon=False)

axes[1].plot(all_years, (shocked_yield_10y-spline_yield_base)*10000, color='#9E1B32')
axes[1].axvline(10, color='grey', ls=':', lw=1)
axes[1].set_xlabel('Maturity (years)'); axes[1].set_ylabel('Yield change (bps)')
axes[1].set_title('Q4.1 -- Difference (shocked - base), the "tent" shape')
fig.tight_layout()
fig.savefig('outputs/q4_1_shock_10y.png', dpi=150)
print("Q4.1: the 50bps 10Y key-rate shock raises yields in a triangular 'tent' shape -- zero at the "
      "adjacent 5Y and 20Y knots, ramping linearly up to the full +50bps exactly at 10Y, then back down "
      "to zero at 20Y and beyond. No other part of the curve (<5y or >20y) is affected at all -- this is "
      "the defining, deliberately LOCAL feature of a key-rate shock, in contrast to a parallel shift.")
print("Saved outputs/q4_1_shock_10y.png\n")

# ============== Q4.2: portfolio KRD to the 10Y shock ==============
shocked_disc_10y = {int(p): yield_to_discount(y, t) for p, t, y in zip(all_periods, all_years, shocked_yield_10y)}
shocked_NAV_10y = price_portfolio(shocked_disc_10y, weights_face)
KRD_10y = shocked_NAV_10y - base_NAV_model
print(f"Q4.2: Base model NAV = ${base_NAV_model:,.2f}; shocked (10Y +50bps) NAV = ${shocked_NAV_10y:,.2f}")
print(f"      KRD_10Y = ${KRD_10y:,.2f}  ({KRD_10y/base_NAV_model*10000:.2f} bps of NAV per 50bps shock)\n")

# ============== Q4.3: KRD to each of the 5 key rates ==============
krd_results = []
for j, knot in enumerate(KEY_RATE_KNOTS):
    shock_vec = triangular_shock(all_years, KEY_RATE_KNOTS, j)
    shocked_yield = spline_yield_base + SHOCK_SIZE * shock_vec
    shocked_disc = {int(p): yield_to_discount(y, t) for p, t, y in zip(all_periods, all_years, shocked_yield)}
    shocked_NAV = price_portfolio(shocked_disc, weights_face)
    krd_results.append({'KeyRate': knot, 'Shocked_NAV': shocked_NAV, 'KRD_dollar': shocked_NAV - base_NAV_model})

krd_df = pd.DataFrame(krd_results)
krd_df['KRD_pct_of_NAV'] = krd_df['KRD_dollar'] / base_NAV_model * 100
krd_df.to_csv('outputs/q4_3_krd_by_keyrate.csv', index=False)
print("Q4.3: KRD by key rate (for a +50bps shock at each node):")
print(krd_df.to_string(index=False))

total_krd = krd_df['KRD_dollar'].sum()
# parallel-shift check: sum of individual KRDs should equal the KRD of a full +50bps parallel shift
parallel_shocked_yield = spline_yield_base + SHOCK_SIZE
parallel_disc = {int(p): yield_to_discount(y, t) for p, t, y in zip(all_periods, all_years, parallel_shocked_yield)}
parallel_NAV = price_portfolio(parallel_disc, weights_face)
parallel_KRD = parallel_NAV - base_NAV_model
print(f"\nSum of individual key-rate KRDs = ${total_krd:,.2f}")
print(f"KRD of a single +50bps PARALLEL shock = ${parallel_KRD:,.2f}")
print(f"  -> these match to within ${abs(total_krd-parallel_KRD):,.2f} (small residual from the convexity "
      f"of bond pricing in yield, since KRDs are summed linearly but repricing is nonlinear in yield).")

fig2, ax2 = plt.subplots(figsize=(8, 5))
ax2.bar([str(k)+'Y' for k in krd_df['KeyRate']], krd_df['KRD_dollar'], color='#9E1B32')
ax2.axhline(0, color='black', lw=0.8)
ax2.set_ylabel('KRD ($, P&L for +50bps shock)')
ax2.set_title('Q4.3 -- Key-rate duration profile of the $1M equally-weighted portfolio')
for i, v in enumerate(krd_df['KRD_dollar']):
    ax2.text(i, v, f'${v:,.0f}', ha='center', va='bottom' if v>=0 else 'top', fontsize=9)
fig2.tight_layout()
fig2.savefig('outputs/q4_3_krd_profile.png', dpi=150)
print("\nSaved outputs/q4_3_krd_profile.png")
