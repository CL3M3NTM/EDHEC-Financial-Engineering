"""
Question 1.2 -- Compare market-implied (bootstrapped) yields to Deutsche
Bank's model yields (Exhibit 4), and identify buy/sell candidates.

Question 1.3 -- Discuss whether this is a risk-free arbitrage.

Relative-value logic (standard, and consistent with the case's own framing):
  - Yield and price move in OPPOSITE directions.
  - If the MARKET yield (our bootstrap) is HIGHER than the MODEL's "fair"
    yield  =>  the market price is LOWER than the model's fair price
             =>  the bond looks CHEAP relative to the model  =>  BUY it
               (expect its yield to fall / price to rise back toward "fair").
  - If the MARKET yield is LOWER than the MODEL yield  =>  the bond looks
    RICH (expensive) relative to the model  =>  SELL it (expect yield to
    rise / price to fall back toward "fair").
"""
import pandas as pd

curve = pd.read_csv('outputs/q1_bootstrapped_curve.csv')
db = pd.read_csv('data/exhibit4_dbmodel.csv')
db = db[db['Maturity_years'].astype(str).str.match(r'^\d+y$', na=False)].copy()
db['Years'] = db['Maturity_years'].str.replace('y', '', regex=False).astype(float)
db['DB_Model_BEY'] = db['DB_Model_BEY'].str.replace('%', '', regex=False).astype(float) / 100.0

merged = db.merge(curve[['Years', 'ZeroYield_BEY']], on='Years', how='left')
merged['Market_minus_Model_bps'] = (merged['ZeroYield_BEY'] - merged['DB_Model_BEY']) * 10000
merged['Signal'] = merged['Market_minus_Model_bps'].apply(
    lambda x: 'BUY (cheap: market yield > model)' if x > 0
    else 'SELL (rich: market yield < model)' if x < 0 else 'Fair'
)

merged_out = merged[['Years', 'DB_Model_BEY', 'ZeroYield_BEY', 'Market_minus_Model_bps', 'Signal']]
merged_out.columns = ['Maturity(y)', 'DB_Model_Yield', 'Market_Yield(Bootstrap)', 'Diff(bps)', 'Signal']
merged_out.to_csv('outputs/q1_2_comparison.csv', index=False)

pd.set_option('display.float_format', lambda x: f'{x:,.4f}')
print(merged_out.to_string(index=False))
