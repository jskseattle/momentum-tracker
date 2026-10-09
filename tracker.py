#!/usr/bin/env python3
"""
Momentum & Narrative Tracker
Identifies sustained sector/thematic momentum and rotation signals.
Filters out noise by requiring multi-timeframe confirmation and volume conviction.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import json
import os
import sys
import warnings
warnings.filterwarnings("ignore")

# ─── ETF Universe ────────────────────────────────────────────────────────
# Grouped by narrative theme. SPY is the benchmark everything is measured against.

ETFS = {
    # Benchmarks
    "SPY": {"name": "S&P 500", "group": "Benchmark"},
    "QQQ": {"name": "Nasdaq 100", "group": "Benchmark"},
    "IWM": {"name": "Small Caps", "group": "Benchmark"},
    "DIA": {"name": "Dow 30", "group": "Benchmark"},

    # Sectors
    "XLK": {"name": "Technology", "group": "Sector"},
    "XLF": {"name": "Financials", "group": "Sector"},
    "XLE": {"name": "Energy", "group": "Sector"},
    "XLV": {"name": "Healthcare", "group": "Sector"},
    "XLI": {"name": "Industrials", "group": "Sector"},
    "XLU": {"name": "Utilities", "group": "Sector"},
    "XLRE": {"name": "Real Estate", "group": "Sector"},
    "XLP": {"name": "Cons. Staples", "group": "Sector"},
    "XLY": {"name": "Cons. Discret.", "group": "Sector"},
    "XLC": {"name": "Communication", "group": "Sector"},
    "XLB": {"name": "Materials", "group": "Sector"},

    # Narrative/Thematic plays
    "SMH": {"name": "Semiconductors", "group": "Thematic"},
    "IGV": {"name": "Software", "group": "Thematic"},
    "XBI": {"name": "Biotech", "group": "Thematic"},
    "ARKK": {"name": "Disruption", "group": "Thematic"},
    "HACK": {"name": "Cybersecurity", "group": "Thematic"},
    "TAN": {"name": "Solar", "group": "Thematic"},
    "ROBO": {"name": "Robotics & AI", "group": "Thematic"},
    "GDX": {"name": "Gold Miners", "group": "Thematic"},
    "KWEB": {"name": "China Internet", "group": "Thematic"},
    "XHB": {"name": "Homebuilders", "group": "Thematic"},
    "JETS": {"name": "Airlines", "group": "Thematic"},
    "BITQ": {"name": "Crypto/Blockchain", "group": "Thematic"},
}

BENCHMARK = "SPY"


def fetch_data(tickers, period="6mo"):
    """Fetch historical OHLCV data for all tickers."""
    ticker_list = list(tickers.keys())
    data = yf.download(ticker_list, period=period, progress=False, group_by="ticker")
    return data


def safe_get_series(data, ticker, field):
    """Extract a price/volume series handling yfinance multi-level columns."""
    try:
        if isinstance(data.columns, pd.MultiIndex):
            return data[(ticker, field)].dropna()
        else:
            return data[field].dropna()
    except (KeyError, TypeError):
        return pd.Series(dtype=float)


def calculate_momentum(data, tickers):
    """Calculate multi-timeframe momentum with volume confirmation and relative strength."""
    results = []

    spy_prices = safe_get_series(data, BENCHMARK, "Close")
    if spy_prices.empty:
        print("  ERROR: Could not fetch SPY data.")
        return pd.DataFrame()

    for ticker, meta in tickers.items():
        prices = safe_get_series(data, ticker, "Close")
        volume = safe_get_series(data, ticker, "Volume")

        if len(prices) < 30 or len(volume) < 25:
            continue

        current_price = prices.iloc[-1]

        # ── Price Rate of Change (multi-timeframe) ──
        roc_5d = (prices.iloc[-1] / prices.iloc[-6] - 1) * 100
        roc_10d = (prices.iloc[-1] / prices.iloc[-11] - 1) * 100
        roc_20d = (prices.iloc[-1] / prices.iloc[-21] - 1) * 100

        # ── Relative Strength vs SPY ──
        spy_roc_5d = (spy_prices.iloc[-1] / spy_prices.iloc[-6] - 1) * 100
        spy_roc_10d = (spy_prices.iloc[-1] / spy_prices.iloc[-11] - 1) * 100
        spy_roc_20d = (spy_prices.iloc[-1] / spy_prices.iloc[-21] - 1) * 100

        rs_5d = roc_5d - spy_roc_5d
        rs_10d = roc_10d - spy_roc_10d
        rs_20d = roc_20d - spy_roc_20d
        rs_composite = rs_5d * 0.45 + rs_10d * 0.35 + rs_20d * 0.20

        # ── Volume Analysis ──
        vol_5d = volume.iloc[-5:].mean()
        vol_20d = volume.iloc[-20:].mean()
        vol_ratio = vol_5d / vol_20d if vol_20d > 0 else 1.0

        # Volume trend: is volume increasing or decreasing over last 10 days?
        vol_first_5 = volume.iloc[-10:-5].mean()
        vol_last_5 = volume.iloc[-5:].mean()
        vol_trend = (vol_last_5 / vol_first_5 - 1) * 100 if vol_first_5 > 0 else 0

        # ── Momentum Persistence (streak) ──
        daily_rets = prices.pct_change().dropna().iloc[-20:]
        streak = 0
        for ret in reversed(daily_rets.values):
            if np.isnan(ret):
                break
            if streak == 0:
                streak = 1 if ret > 0 else -1
            elif (streak > 0 and ret > 0) or (streak < 0 and ret < 0):
                streak += (1 if streak > 0 else -1)
            else:
                break

        # ── Timeframe Alignment ──
        signs = [np.sign(roc_5d), np.sign(roc_10d), np.sign(roc_20d)]
        aligned_up = all(s > 0 for s in signs)
        aligned_down = all(s < 0 for s in signs)
        aligned = aligned_up or aligned_down

        # ── 5d momentum acceleration ──
        # Compare current 5d ROC to prior 5d ROC (shifted by 5 days)
        prev_roc_5d = (prices.iloc[-6] / prices.iloc[-11] - 1) * 100 if len(prices) >= 11 else 0
        momentum_accel = roc_5d - prev_roc_5d

        # ── 52-week high proximity ──
        high_52w = prices.iloc[-252:].max() if len(prices) >= 252 else prices.max()
        pct_from_high = (current_price / high_52w - 1) * 100

        # ── Composite Momentum Score ──
        price_score = roc_5d * 0.40 + roc_10d * 0.35 + roc_20d * 0.25
        vol_factor = min(max(vol_ratio, 0.5), 2.0)
        composite = price_score * vol_factor

        # ── Status Classification ──
        # Requires multi-day confirmation — no single-day moves qualify
        if aligned_up and price_score > 2 and vol_ratio > 1.15:
            status = "STRONG"
            status_icon = "▲▲"
        elif roc_5d > 1.5 and roc_10d > 0 and momentum_accel > 0:
            status = "BUILDING"
            status_icon = "▲ "
        elif roc_5d > 0 and roc_10d > 0 and vol_ratio < 0.8:
            status = "FADING"
            status_icon = "△ "
        elif aligned_down and price_score < -2 and vol_ratio > 1.15:
            status = "WEAK"
            status_icon = "▼▼"
        elif roc_5d < -1.5 and roc_10d < 0:
            status = "SELLING"
            status_icon = "▼ "
        else:
            status = "NEUTRAL"
            status_icon = "── "

        results.append({
            "ticker": ticker,
            "name": meta["name"],
            "group": meta["group"],
            "price": round(current_price, 2),
            "roc_5d": round(roc_5d, 2),
            "roc_10d": round(roc_10d, 2),
            "roc_20d": round(roc_20d, 2),
            "rs_5d": round(rs_5d, 2),
            "rs_10d": round(rs_10d, 2),
            "rs_20d": round(rs_20d, 2),
            "rs_composite": round(rs_composite, 2),
            "vol_ratio": round(vol_ratio, 2),
            "vol_trend": round(vol_trend, 1),
            "streak": int(streak),
            "aligned": aligned,
            "aligned_up": aligned_up,
            "aligned_down": aligned_down,
            "momentum_accel": round(momentum_accel, 2),
            "pct_from_high": round(pct_from_high, 1),
            "composite": round(composite, 2),
            "status": status,
            "status_icon": status_icon,
        })

    return pd.DataFrame(results)


def detect_rotations(df):
    """Identify active sector rotations — money flowing in or out of a theme."""
    rotations = {"into": [], "out_of": []}

    non_bench = df[df["group"] != "Benchmark"]

    for _, row in non_bench.iterrows():
        # Rotation INTO: 5d RS clearly positive and accelerating vs longer timeframes
        if row["rs_5d"] > 1.5 and row["rs_5d"] > row["rs_10d"] and row["vol_ratio"] > 1.0:
            rotations["into"].append({
                "ticker": row["ticker"],
                "name": row["name"],
                "rs_5d": row["rs_5d"],
                "rs_10d": row["rs_10d"],
                "vol_ratio": row["vol_ratio"],
            })

        # Rotation OUT OF: 5d RS clearly negative and worsening
        elif row["rs_5d"] < -1.5 and row["rs_5d"] < row["rs_10d"] and row["vol_ratio"] > 1.0:
            rotations["out_of"].append({
                "ticker": row["ticker"],
                "name": row["name"],
                "rs_5d": row["rs_5d"],
                "rs_10d": row["rs_10d"],
                "vol_ratio": row["vol_ratio"],
            })

    rotations["into"].sort(key=lambda x: x["rs_5d"], reverse=True)
    rotations["out_of"].sort(key=lambda x: x["rs_5d"])

    return rotations


def print_report(df, rotations):
    """Print formatted terminal report."""
    now = datetime.now()

    W = 85
    print(f"\n{'=' * W}")
    print(f"  MOMENTUM & NARRATIVE TRACKER")
    print(f"  {now.strftime('%A, %B %d %Y %I:%M %p')}")
    print(f"{'=' * W}")

    # ── Market Pulse ──
    bench = df[df["group"] == "Benchmark"].sort_values("composite", ascending=False)
    print(f"\n  MARKET PULSE")
    print(f"  {'─' * (W - 4)}")
    for _, r in bench.iterrows():
        print(f"  {r['status_icon']} {r['name']:<14} "
              f"5d:{r['roc_5d']:>+6.1f}%  10d:{r['roc_10d']:>+6.1f}%  "
              f"20d:{r['roc_20d']:>+6.1f}%  "
              f"Vol:{r['vol_ratio']:>4.1f}x  52wH:{r['pct_from_high']:>+5.1f}%")

    # ── Rotation Signals ──
    if rotations["into"] or rotations["out_of"]:
        print(f"\n  ROTATION SIGNALS")
        print(f"  {'─' * (W - 4)}")
        if rotations["into"]:
            for r in rotations["into"]:
                print(f"  >>> Money rotating INTO {r['name']} ({r['ticker']}): "
                      f"RS 5d +{r['rs_5d']:.1f}% vs SPY, Vol {r['vol_ratio']:.1f}x")
        if rotations["out_of"]:
            for r in rotations["out_of"]:
                print(f"  <<< Money rotating OUT OF {r['name']} ({r['ticker']}): "
                      f"RS 5d {r['rs_5d']:.1f}% vs SPY, Vol {r['vol_ratio']:.1f}x")

    # ── Sector Rankings ──
    sectors = df[df["group"] == "Sector"].sort_values("composite", ascending=False)
    print(f"\n  SECTOR MOMENTUM (ranked by composite score)")
    print(f"  {'─' * (W - 4)}")
    print(f"  {'':2} {'Ticker':<5} {'Name':<16} {'Status':<10} "
          f"{'5d':>6} {'10d':>6} {'20d':>6} {'RS':>6} {'Vol':>5} {'Accel':>6} {'Strk':>5}")
    print(f"  {'─' * (W - 4)}")
    for _, r in sectors.iterrows():
        print(f"  {r['status_icon']} {r['ticker']:<5} {r['name']:<16} {r['status']:<10} "
              f"{r['roc_5d']:>+5.1f}% {r['roc_10d']:>+5.1f}% {r['roc_20d']:>+5.1f}% "
              f"{r['rs_composite']:>+5.1f} {r['vol_ratio']:>4.1f}x "
              f"{r['momentum_accel']:>+5.1f} {r['streak']:>+4d}")

    # ── Thematic Rankings ──
    thematic = df[df["group"] == "Thematic"].sort_values("composite", ascending=False)
    print(f"\n  THEMATIC / NARRATIVE MOMENTUM (ranked by composite score)")
    print(f"  {'─' * (W - 4)}")
    print(f"  {'':2} {'Ticker':<5} {'Name':<16} {'Status':<10} "
          f"{'5d':>6} {'10d':>6} {'20d':>6} {'RS':>6} {'Vol':>5} {'Accel':>6} {'Strk':>5}")
    print(f"  {'─' * (W - 4)}")
    for _, r in thematic.iterrows():
        print(f"  {r['status_icon']} {r['ticker']:<5} {r['name']:<16} {r['status']:<10} "
              f"{r['roc_5d']:>+5.1f}% {r['roc_10d']:>+5.1f}% {r['roc_20d']:>+5.1f}% "
              f"{r['rs_composite']:>+5.1f} {r['vol_ratio']:>4.1f}x "
              f"{r['momentum_accel']:>+5.1f} {r['streak']:>+4d}")

    # ── Summary Signals ──
    non_bench = df[df["group"] != "Benchmark"]
    print(f"\n  SIGNAL SUMMARY")
    print(f"  {'─' * (W - 4)}")

    strong = non_bench[non_bench["status"] == "STRONG"]
    if len(strong) > 0:
        print(f"  ▲▲ Strong momentum:  {', '.join(strong['name'].values)}")

    building = non_bench[non_bench["status"] == "BUILDING"]
    if len(building) > 0:
        print(f"  ▲  Building:         {', '.join(building['name'].values)}")

    fading = non_bench[non_bench["status"] == "FADING"]
    if len(fading) > 0:
        print(f"  △  Fading (low vol): {', '.join(fading['name'].values)}")

    selling = non_bench[non_bench["status"] == "SELLING"]
    if len(selling) > 0:
        print(f"  ▼  Selling:          {', '.join(selling['name'].values)}")

    weak = non_bench[non_bench["status"] == "WEAK"]
    if len(weak) > 0:
        print(f"  ▼▼ Weak:             {', '.join(weak['name'].values)}")

    neutral_count = len(non_bench[non_bench["status"] == "NEUTRAL"])
    if neutral_count > 0:
        print(f"  ── Neutral:          {neutral_count} sectors/themes with no clear signal")

    # ── Near 52-week highs ──
    near_highs = non_bench[non_bench["pct_from_high"] > -3].sort_values("pct_from_high", ascending=False)
    if len(near_highs) > 0:
        print(f"\n  NEAR 52-WEEK HIGHS (within 3%)")
        print(f"  {'─' * (W - 4)}")
        for _, r in near_highs.iterrows():
            print(f"  {r['name']:<20} {r['pct_from_high']:>+5.1f}% from high")

    print(f"\n  {'─' * (W - 4)}")
    print(f"  5d/10d/20d = rate of change | RS = relative strength vs SPY (composite)")
    print(f"  Vol = 5d avg volume / 20d avg | Accel = momentum acceleration (5d vs prior 5d)")
    print(f"  Strk = consecutive up/down days | Requires multi-day confirmation, not 1-day moves")
    print(f"{'=' * W}\n")


def generate_html_report(df, rotations):
    """Generate a self-contained HTML dashboard with embedded data."""
    now = datetime.now()
    data_json = df.to_json(orient="records")
    rotations_json = json.dumps(rotations)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Momentum Tracker</title>
<style>
:root {{
  --bg: #0d1117;
  --surface: #161b22;
  --border: #30363d;
  --text: #e6edf3;
  --text-dim: #8b949e;
  --green: #3fb950;
  --green-bg: rgba(63,185,80,0.1);
  --blue: #58a6ff;
  --blue-bg: rgba(88,166,255,0.1);
  --yellow: #d29922;
  --yellow-bg: rgba(210,153,34,0.1);
  --red: #f85149;
  --red-bg: rgba(248,81,73,0.1);
  --purple: #bc8cff;
}}
@media (prefers-color-scheme: light) {{
  :root:not([data-theme="dark"]) {{
    --bg: #f6f8fa;
    --surface: #ffffff;
    --border: #d0d7de;
    --text: #1f2328;
    --text-dim: #656d76;
    --green: #1a7f37;
    --green-bg: rgba(26,127,55,0.08);
    --blue: #0969da;
    --blue-bg: rgba(9,105,218,0.08);
    --yellow: #9a6700;
    --yellow-bg: rgba(154,103,0,0.08);
    --red: #cf222e;
    --red-bg: rgba(207,34,46,0.08);
    --purple: #8250df;
  }}
}}
:root[data-theme="dark"] {{
  --bg: #0d1117;
  --surface: #161b22;
  --border: #30363d;
  --text: #e6edf3;
  --text-dim: #8b949e;
  --green: #3fb950;
  --green-bg: rgba(63,185,80,0.1);
  --blue: #58a6ff;
  --blue-bg: rgba(88,166,255,0.1);
  --yellow: #d29922;
  --yellow-bg: rgba(210,153,34,0.1);
  --red: #f85149;
  --red-bg: rgba(248,81,73,0.1);
  --purple: #bc8cff;
}}
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
  background: var(--bg);
  color: var(--text);
  padding: 16px;
  line-height: 1.5;
}}
.header {{
  text-align: center;
  padding: 24px 0 16px;
}}
.header h1 {{
  font-size: 1.5rem;
  font-weight: 600;
  letter-spacing: 0.03em;
}}
.header .date {{
  color: var(--text-dim);
  font-size: 0.875rem;
  margin-top: 4px;
}}
.section {{
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 8px;
  margin: 16px auto;
  max-width: 1200px;
  overflow: hidden;
}}
.section-title {{
  padding: 12px 16px;
  font-size: 0.75rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--text-dim);
  border-bottom: 1px solid var(--border);
}}
table {{
  width: 100%;
  border-collapse: collapse;
  font-size: 0.8125rem;
}}
th {{
  text-align: left;
  padding: 8px 12px;
  font-size: 0.6875rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--text-dim);
  border-bottom: 1px solid var(--border);
  white-space: nowrap;
}}
th.num {{ text-align: right; }}
td {{
  padding: 8px 12px;
  border-bottom: 1px solid var(--border);
  white-space: nowrap;
}}
td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
tr:last-child td {{ border-bottom: none; }}
.pos {{ color: var(--green); }}
.neg {{ color: var(--red); }}
.dim {{ color: var(--text-dim); }}
.badge {{
  display: inline-block;
  padding: 2px 8px;
  border-radius: 12px;
  font-size: 0.6875rem;
  font-weight: 600;
  letter-spacing: 0.03em;
}}
.badge-strong {{ background: var(--green-bg); color: var(--green); }}
.badge-building {{ background: var(--blue-bg); color: var(--blue); }}
.badge-fading {{ background: var(--yellow-bg); color: var(--yellow); }}
.badge-selling {{ background: var(--red-bg); color: var(--red); }}
.badge-weak {{ background: var(--red-bg); color: var(--red); }}
.badge-neutral {{ background: var(--border); color: var(--text-dim); }}
.rotation-card {{
  padding: 12px 16px;
  border-bottom: 1px solid var(--border);
  display: flex;
  align-items: center;
  gap: 12px;
}}
.rotation-card:last-child {{ border-bottom: none; }}
.rotation-arrow {{
  font-size: 1.25rem;
  flex-shrink: 0;
}}
.rotation-into .rotation-arrow {{ color: var(--green); }}
.rotation-out .rotation-arrow {{ color: var(--red); }}
.rotation-detail {{
  font-size: 0.8125rem;
}}
.rotation-detail strong {{ font-weight: 600; }}
.rotation-detail span {{ color: var(--text-dim); margin-left: 8px; }}
.signal-grid {{
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
  gap: 1px;
  background: var(--border);
}}
.signal-cell {{
  background: var(--surface);
  padding: 12px 16px;
}}
.signal-cell .label {{
  font-size: 0.6875rem;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--text-dim);
  margin-bottom: 4px;
}}
.signal-cell .value {{
  font-size: 0.875rem;
}}
.empty {{ padding: 16px; color: var(--text-dim); font-size: 0.8125rem; text-align: center; }}
.bar-container {{
  width: 60px;
  height: 6px;
  background: var(--border);
  border-radius: 3px;
  display: inline-block;
  vertical-align: middle;
  margin-left: 6px;
  overflow: hidden;
}}
.bar {{
  height: 100%;
  border-radius: 3px;
}}
.bar.positive {{ background: var(--green); }}
.bar.negative {{ background: var(--red); float: right; }}
.footer {{
  text-align: center;
  padding: 20px;
  color: var(--text-dim);
  font-size: 0.75rem;
}}
</style>
</head>
<body>
<div class="header">
  <h1>Momentum &amp; Narrative Tracker</h1>
  <div class="date">{now.strftime('%A, %B %d %Y &mdash; %I:%M %p')}</div>
</div>

<div id="app"></div>

<div class="footer">
  Multi-timeframe confirmation required. No single-day moves qualify.<br>
  5d/10d/20d = rate of change &middot; RS = relative strength vs SPY &middot;
  Vol = 5d avg / 20d avg &middot; Accel = 5d momentum vs prior 5d
</div>

<script>
const DATA = {data_json};
const ROTATIONS = {rotations_json};

function fmt(v, suffix='%') {{
  if (v == null || isNaN(v)) return '—';
  const sign = v > 0 ? '+' : '';
  return sign + v.toFixed(1) + suffix;
}}

function colorClass(v) {{
  if (v > 0.3) return 'pos';
  if (v < -0.3) return 'neg';
  return 'dim';
}}

function badgeClass(status) {{
  return 'badge badge-' + status.toLowerCase();
}}

function volBar(ratio) {{
  const pct = Math.min(Math.abs(ratio - 1) * 100, 100);
  const cls = ratio >= 1 ? 'positive' : 'negative';
  return `<span class="dim">${{ratio.toFixed(1)}}x</span>
    <span class="bar-container"><span class="bar ${{cls}}" style="width:${{pct}}%"></span></span>`;
}}

function buildTable(items, id) {{
  if (!items.length) return '<div class="empty">No data</div>';
  let html = '<table><thead><tr>';
  html += '<th>Ticker</th><th>Name</th><th>Status</th>';
  html += '<th class="num">5d</th><th class="num">10d</th><th class="num">20d</th>';
  html += '<th class="num">RS</th><th class="num">Vol</th>';
  html += '<th class="num">Accel</th><th class="num">Streak</th><th class="num">52w Hi</th>';
  html += '</tr></thead><tbody>';
  items.forEach(r => {{
    html += '<tr>';
    html += `<td><strong>${{r.ticker}}</strong></td>`;
    html += `<td>${{r.name}}</td>`;
    html += `<td><span class="${{badgeClass(r.status)}}">${{r.status}}</span></td>`;
    html += `<td class="num ${{colorClass(r.roc_5d)}}">${{fmt(r.roc_5d)}}</td>`;
    html += `<td class="num ${{colorClass(r.roc_10d)}}">${{fmt(r.roc_10d)}}</td>`;
    html += `<td class="num ${{colorClass(r.roc_20d)}}">${{fmt(r.roc_20d)}}</td>`;
    html += `<td class="num ${{colorClass(r.rs_composite)}}">${{fmt(r.rs_composite, '')}}</td>`;
    html += `<td class="num">${{volBar(r.vol_ratio)}}</td>`;
    html += `<td class="num ${{colorClass(r.momentum_accel)}}">${{fmt(r.momentum_accel)}}</td>`;
    html += `<td class="num ${{colorClass(r.streak)}}">${{r.streak > 0 ? '+' : ''}}${{r.streak}}d</td>`;
    html += `<td class="num ${{colorClass(r.pct_from_high + 3)}}">${{fmt(r.pct_from_high)}}</td>`;
    html += '</tr>';
  }});
  html += '</tbody></table>';
  return html;
}}

function buildRotations() {{
  const into = ROTATIONS.into || [];
  const out = ROTATIONS.out_of || [];
  if (!into.length && !out.length) return '';

  let html = '<div class="section"><div class="section-title">Rotation Signals</div>';
  into.forEach(r => {{
    html += `<div class="rotation-card rotation-into">
      <div class="rotation-arrow">▶</div>
      <div class="rotation-detail">
        <strong>INTO ${{r.name}} (${{r.ticker}})</strong>
        <span>RS +${{r.rs_5d.toFixed(1)}}% vs SPY · Vol ${{r.vol_ratio.toFixed(1)}}x</span>
      </div>
    </div>`;
  }});
  out.forEach(r => {{
    html += `<div class="rotation-card rotation-out">
      <div class="rotation-arrow">◀</div>
      <div class="rotation-detail">
        <strong>OUT OF ${{r.name}} (${{r.ticker}})</strong>
        <span>RS ${{r.rs_5d.toFixed(1)}}% vs SPY · Vol ${{r.vol_ratio.toFixed(1)}}x</span>
      </div>
    </div>`;
  }});
  html += '</div>';
  return html;
}}

function buildSignalSummary() {{
  const groups = {{}};
  DATA.filter(d => d.group !== 'Benchmark').forEach(d => {{
    if (!groups[d.status]) groups[d.status] = [];
    groups[d.status].push(d.name);
  }});

  const labels = {{
    STRONG: ['Strong Momentum', 'green'],
    BUILDING: ['Building', 'blue'],
    FADING: ['Fading (low volume)', 'yellow'],
    SELLING: ['Selling', 'red'],
    WEAK: ['Weak', 'red'],
    NEUTRAL: ['Neutral', 'dim'],
  }};

  let cells = '';
  for (const [status, [label, color]] of Object.entries(labels)) {{
    if (groups[status] && groups[status].length) {{
      cells += `<div class="signal-cell">
        <div class="label">${{label}}</div>
        <div class="value" style="color:var(--${{color}})">${{groups[status].join(', ')}}</div>
      </div>`;
    }}
  }}

  if (!cells) return '';
  return `<div class="section"><div class="section-title">Signal Summary</div>
    <div class="signal-grid">${{cells}}</div></div>`;
}}

function render() {{
  const app = document.getElementById('app');
  let html = '';

  const benchmarks = DATA.filter(d => d.group === 'Benchmark')
    .sort((a, b) => b.composite - a.composite);
  const sectors = DATA.filter(d => d.group === 'Sector')
    .sort((a, b) => b.composite - a.composite);
  const thematic = DATA.filter(d => d.group === 'Thematic')
    .sort((a, b) => b.composite - a.composite);

  html += `<div class="section"><div class="section-title">Market Pulse</div>${{buildTable(benchmarks)}}</div>`;
  html += buildRotations();
  html += buildSignalSummary();
  html += `<div class="section"><div class="section-title">Sector Momentum</div>${{buildTable(sectors)}}</div>`;
  html += `<div class="section"><div class="section-title">Thematic / Narrative Momentum</div>${{buildTable(thematic)}}</div>`;

  app.innerHTML = html;
}}

render();
</script>
</body>
</html>"""
    return html


def save_outputs(df, rotations, output_dir):
    """Save CSV data and HTML report."""
    date_str = datetime.now().strftime("%Y-%m-%d")

    csv_path = os.path.join(output_dir, f"momentum_{date_str}.csv")
    df.to_csv(csv_path, index=False)

    html_content = generate_html_report(df, rotations)
    html_path = os.path.join(output_dir, f"momentum_{date_str}.html")
    with open(html_path, "w") as f:
        f.write(html_content)

    latest_path = os.path.join(output_dir, "momentum_latest.html")
    with open(latest_path, "w") as f:
        f.write(html_content)

    json_path = os.path.join(output_dir, "momentum_latest.json")
    output = {
        "generated": datetime.now().isoformat(),
        "data": json.loads(df.to_json(orient="records")),
        "rotations": rotations,
    }
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)

    return csv_path, html_path


def main():
    print("\n  Fetching market data (6 months)...")
    data = fetch_data(ETFS)

    print("  Calculating momentum signals...")
    df = calculate_momentum(data, ETFS)

    if df.empty:
        print("  ERROR: No data returned. Check your internet connection.")
        sys.exit(1)

    rotations = detect_rotations(df)

    print_report(df, rotations)

    output_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path, html_path = save_outputs(df, rotations, output_dir)

    print(f"  Saved: {os.path.basename(csv_path)}")
    print(f"  Saved: {os.path.basename(html_path)}")
    print(f"  Open momentum_latest.html in your browser for the dashboard.\n")


if __name__ == "__main__":
    main()
