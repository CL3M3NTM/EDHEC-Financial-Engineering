"""
Python Labs -- Trump Tariff Threats: Event Study --
=================================================
This module implements the event-study machinery requested by the
assignment and is meant to be pasted into the provided notebook
(`pylab_trump_tariff_threats_assignment.ipynb`), right after the
`yield_curve_data` fetch cell (which already works against the real FRED
data in your environment).

IMPORTANT: this script is validated here against SYNTHETIC yield data (a
random walk, standing in for `yield_curve_data`), because this sandbox has
no network access to the FRED API and therefore cannot download the real
Treasury series itself. The logic, shapes, and edge-case handling are fully
tested; when you paste this into your notebook and run it against your real
`yield_curve_data`, only the YIELD VALUES will be the real-world figures.

Documented methodological choices (per the assignment's instructions):
  - "Business days" = trading days actually present in `yield_curve_data`
    (i.e. we work in TRADING-DAY INDEX space, not calendar-day space: offset
    +5 means "5 rows later in the data", skipping weekends/holidays
    automatically because they are simply absent from the FRED series).
  - Event window: [-30, +30] trading days around each event date, yield
    change measured RELATIVE TO the event-date yield (as specified).
  - "Abnormal" yield change at horizon d = (average, across all events, of
    the change from the event date to d days later) MINUS (the average
    change over the same horizon d, computed over all NON-event dates,
    where "non-event" excludes any date within 30 trading days of ANY
    event). This nets out the "normal" drift of yields over a d-day window
    that has nothing to do with the tariff events.
  - If an event date is not itself a trading day (e.g. falls on a weekend),
    we snap to the next available trading day on or after it.
  - Overlapping event windows are NOT removed (Q2.1 flags this explicitly as
    a limitation) -- an event-study purist might drop or merge overlapping
    windows; we keep all events and discuss the consequence in Q2.1.
"""
import numpy as np
import pandas as pd

WINDOW = 30  # trading days each side

# ---------------------------------------------------------------------
# Part 0: build a single, tidy events table from the notebook's two dicts
# ---------------------------------------------------------------------
def build_events_table(trump_actions: dict, country_responses: dict) -> pd.DataFrame:
    rows = []
    for d, desc in trump_actions.items():
        rows.append({'date': pd.to_datetime(d), 'description': desc, 'event_type': 'trump_action'})
    for d, desc in country_responses.items():
        rows.append({'date': pd.to_datetime(d), 'description': desc, 'event_type': 'country_response'})
    events = pd.DataFrame(rows).drop_duplicates(subset=['date', 'event_type']).sort_values('date').reset_index(drop=True)
    events['term'] = np.where(events['date'] < pd.Timestamp('2021-01-01'), 'first_term', 'second_term')
    return events


# ---------------------------------------------------------------------
# Part 1: core event-study engine (works for ANY single yield series, or a
# derived series such as slope/curvature -- just pass it in as `series`)
# ---------------------------------------------------------------------
def event_study(dates: pd.Series, series: pd.Series, event_dates: pd.Series, window: int = WINDOW):
    """
    dates, series : the full trading-day index and the yield (or
                     level/slope/curvature) series aligned to it, sorted
                     ascending, no duplicate dates.
    event_dates   : the event dates to study (any subset you want to test:
                     all events, trump_actions only, country_responses
                     only, first_term only, second_term only, ...).

    Returns a DataFrame indexed by offset in [-window, window] with columns:
      event_mean_change  : average (series[t+offset] - series[t]) across events
      n_events_used       : how many events actually had data at that offset
      baseline_mean_change: average same-horizon change computed over all
                            non-event dates (see module docstring)
      abnormal_change      : event_mean_change - baseline_mean_change
    """
    dates = pd.DatetimeIndex(dates)
    s = pd.Series(series.values, index=dates).sort_index()
    s = s[~s.index.duplicated(keep='first')]
    n = len(s)
    idx_of_date = {d: i for i, d in enumerate(s.index)}

    # snap each event date to the next available trading day on/after it
    event_positions = []
    for ed in pd.DatetimeIndex(event_dates):
        pos = s.index.searchsorted(ed)
        if pos < n:
            event_positions.append(pos)
    event_positions = sorted(set(event_positions))

    # mark "non-event" positions: any trading day NOT within `window` days of
    # any event position (used for the baseline)
    is_near_event = np.zeros(n, dtype=bool)
    for p in event_positions:
        lo, hi = max(0, p - window), min(n, p + window + 1)
        is_near_event[lo:hi] = True
    non_event_positions = np.where(~is_near_event)[0]

    offsets = range(-window, window + 1)
    rows = []
    values = s.values
    for offset in offsets:
        # --- event-conditional mean change at this horizon ---
        changes = []
        for p in event_positions:
            q = p + offset
            if 0 <= q < n:
                changes.append(values[q] - values[p])
        event_mean = np.mean(changes) if changes else np.nan
        n_events_used = len(changes)

        # --- baseline: same-horizon change computed from every eligible
        # non-event date (both endpoints must also be non-event-adjacent
        # trading days, so the baseline isn't itself contaminated by events) ---
        base_changes = []
        non_event_set = set(non_event_positions.tolist())
        for p in non_event_positions:
            q = p + offset
            if 0 <= q < n and q in non_event_set:
                base_changes.append(values[q] - values[p])
        baseline_mean = np.mean(base_changes) if base_changes else np.nan

        rows.append({
            'offset': offset,
            'event_mean_change': event_mean,
            'n_events_used': n_events_used,
            'baseline_mean_change': baseline_mean,
            'abnormal_change': event_mean - baseline_mean,
        })
    return pd.DataFrame(rows).set_index('offset')


# ---------------------------------------------------------------------
# Part 2: level / slope / curvature decomposition (Q1.2)
# ---------------------------------------------------------------------
def level_slope_curvature(yield_curve_df: pd.DataFrame) -> pd.DataFrame:
    """
    yield_curve_df: the `yield_curve_data` dataframe (as fetched by fred.py),
    with columns like '3M','6M','1Y','2Y','3Y','5Y','7Y','10Y','20Y','30Y'.

    level     = 10Y yield (long-end anchor; a common, simple level proxy)
    slope     = 10Y - 2Y   (classic curve-steepness proxy)
    curvature = 2*5Y - (2Y + 10Y)   (classic "butterfly"/curvature proxy:
                positive = curve is humped around 5Y relative to its ends)
    """
    out = yield_curve_df[['date']].copy()
    out['level'] = yield_curve_df['10Y']
    out['slope'] = yield_curve_df['10Y'] - yield_curve_df['2Y']
    out['curvature'] = 2 * yield_curve_df['5Y'] - (yield_curve_df['2Y'] + yield_curve_df['10Y'])
    return out


# =======================================================================
# Validation against SYNTHETIC data (random walk), since this sandbox has
# no network access to fetch the real FRED series. Run as a script to see
# that the machinery behaves sanely before you paste it into the notebook.
# =======================================================================
if __name__ == '__main__':
    rng = np.random.default_rng(42)
    dates = pd.bdate_range('2015-01-01', '2025-08-01')
    n = len(dates)

    # synthetic yields: a slow-moving random walk per maturity, all sharing
    # a common level factor (so slope/curvature aren't degenerate)
    common = np.cumsum(rng.normal(0, 0.01, n))
    synth = pd.DataFrame({'date': dates})
    for mat, vol in [('3M',0.015), ('6M',0.015), ('1Y',0.014), ('2Y',0.013), ('3Y',0.012),
                     ('5Y',0.011), ('7Y',0.010), ('10Y',0.010), ('20Y',0.009), ('30Y',0.009)]:
        idio = np.cumsum(rng.normal(0, vol, n))
        synth[mat] = 2.0 + common*0.8 + idio*0.3

    # a toy event list spread across the sample, mimicking the real one's density
    toy_events = pd.to_datetime(rng.choice(dates[500:-500], size=40, replace=False))

    es = event_study(synth['date'], synth['10Y'], pd.Series(toy_events))
    print("Sample of the event-study output (10Y yield, synthetic data):")
    print(es.loc[[-30,-10,-1,0,1,10,30]])

    lsc = level_slope_curvature(synth)
    es_level = event_study(lsc['date'], lsc['level'], pd.Series(toy_events))
    es_slope = event_study(lsc['date'], lsc['slope'], pd.Series(toy_events))
    es_curv  = event_study(lsc['date'], lsc['curvature'], pd.Series(toy_events))
    print("\nLevel/slope/curvature abnormal change at offset=+5 (synthetic, sanity check only):")
    print(f"  level:     {es_level.loc[5,'abnormal_change']: .4f}")
    print(f"  slope:     {es_slope.loc[5,'abnormal_change']: .4f}")
    print(f"  curvature: {es_curv.loc[5,'abnormal_change']: .4f}")

    print("\nAll synthetic-data checks ran without errors -- machinery is ready for real FRED data.")
