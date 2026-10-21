"""
Question 1.1 -- Bootstrapping the discount curve from Exhibit 1
================================================================
Settlement date: August 15, 2003. All 52 Treasury bonds in Exhibit 1 mature on
successive semi-annual dates (2/15 and 8/15 of each year, from 2004 to 2029),
with NO gaps -- i.e. there is exactly one bond maturing at each of the 52
semi-annual nodes t = 0.5, 1.0, 1.5, ..., 26.0 years from settlement.

This is the textbook "successive substitution" bootstrap: because the dates
line up on a single semi-annual grid, each bond's price equation introduces
exactly one new unknown discount factor (all earlier discount factors having
already been solved for by shorter-maturity bonds), so we can solve for them
one at a time, in order of increasing maturity.

Convention note (see the assignment's footnote on yield/compounding
convention): U.S. Treasury coupons are paid SEMI-ANNUALLY, and Exhibit 4's
model yields are explicitly given as Bond-Equivalent Yields (BEY), i.e.
(semi-annual yield) x 2. To compare like-for-like in Q1.2, we therefore derive
our own zero-coupon yields in the SAME semi-annual/BEY convention, not in a
continuously-compounded convention (which would silently bias every
comparison in Q1.2 towards one systematic direction).
"""
import pandas as pd
import numpy as np
from datetime import date

SETTLEMENT = date(2003, 8, 15)

def load_bonds():
    df = pd.read_csv('data/exhibit1_bonds.csv')
    df = df[pd.to_numeric(df['Coupon'], errors='coerce').notna()].copy()
    df['Maturity'] = pd.to_datetime(df['Maturity'])
    df['Coupon'] = df['Coupon'].astype(float)
    df['Price'] = df['Price'].astype(float)
    df = df.sort_values('Maturity').reset_index(drop=True)
    # number of semi-annual periods from settlement to maturity
    df['Periods'] = ((df['Maturity'].dt.year - SETTLEMENT.year) * 2
                      + (df['Maturity'].dt.month - SETTLEMENT.month) // 6)
    df['Years'] = df['Periods'] * 0.5
    return df

def bootstrap(df):
    """Sequential bootstrap of semi-annual discount factors D_k = P(0, k*0.5y)."""
    n = len(df)
    D = np.zeros(n + 1)  # D[0] = 1 (today), D[k] = discount factor to period k
    D[0] = 1.0
    for i, row in df.iterrows():
        k = int(row['Periods'])
        c = row['Coupon'] / 2.0  # semi-annual coupon cash flow per 100 face
        price = row['Price']
        # price = c * sum_{j=1}^{k-1} D[j]  +  (c + 100) * D[k]
        coupon_pv_so_far = c * D[1:k].sum()
        D[k] = (price - coupon_pv_so_far) / (c + 100.0)
    return D

def discount_to_bey(D, periods):
    """Convert a semi-annual discount factor D (for `periods` 6-month steps)
    into a Bond-Equivalent Yield: (1+BEY/2)^periods = 1/D  =>  BEY = 2*(D^(-1/periods)-1)."""
    return 2.0 * (D ** (-1.0 / periods) - 1.0)

if __name__ == '__main__':
    bonds = load_bonds()
    assert bonds['Periods'].is_monotonic_increasing and (bonds['Periods'].diff().dropna() == 1).all(), \
        "Expected an unbroken semi-annual maturity ladder -- check Exhibit 1 parsing."
    print(f"Loaded {len(bonds)} bonds, periods {bonds['Periods'].min()} to {bonds['Periods'].max()} "
          f"(i.e. {bonds['Years'].min()}y to {bonds['Years'].max()}y)")

    D = bootstrap(bonds)
    periods_arr = np.arange(1, len(D))
    years_arr = periods_arr * 0.5
    bey = discount_to_bey(D[1:], periods_arr)

    curve = pd.DataFrame({
        'Period': periods_arr,
        'Years': years_arr,
        'DiscountFactor': D[1:],
        'ZeroYield_BEY': bey,
    })
    curve.to_csv('outputs/q1_bootstrapped_curve.csv', index=False)

    print(curve.iloc[[0,1,3,9,19,29,39,49,51]].to_string(index=False))
    print("\nSaved full 52-point curve to outputs/q1_bootstrapped_curve.csv")
