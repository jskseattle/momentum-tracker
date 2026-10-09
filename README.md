# Momentum & Narrative Tracker

Track sector and thematic ETF momentum to identify market rotations and narrative shifts before they're obvious.

Built on the thesis that markets are increasingly narrative-driven. Chip runs, software recoveries, defensive rotations — these moves play out over days and weeks, not hours. This tracker filters out daily noise and surfaces only sustained, volume-confirmed momentum.

## What it does

Analyzes 27 sector and thematic ETFs across multiple timeframes to answer three questions:

1. **Where is momentum right now?** Multi-timeframe rate of change (5d, 10d, 20d) ranked by composite score
2. **Is the momentum real?** Volume confirmation separates conviction from drift
3. **What's rotating?** Relative strength vs SPY detects money flowing into or out of themes

## How momentum is classified

No single-day moves qualify. Every status requires multi-day confirmation:

| Status | Meaning | Criteria |
|--------|---------|----------|
| **STRONG** | Sustained uptrend with conviction | All timeframes (5d/10d/20d) aligned up + above-average volume |
| **BUILDING** | Momentum accelerating | 5d and 10d positive, acceleration increasing |
| **FADING** | Price up but conviction gone | Positive returns on declining volume |
| **SELLING** | Active selling pressure | 5d and 10d negative |
| **WEAK** | Sustained downtrend with conviction | All timeframes aligned down + above-average volume |
| **NEUTRAL** | No clear signal | Mixed signals across timeframes |

## Metrics

| Metric | What it measures |
|--------|-----------------|
| **5d / 10d / 20d** | Rate of change over each period |
| **RS** | Relative strength vs SPY — how much the ETF outperforms or underperforms the broad market |
| **Vol** | 5-day average volume / 20-day average — above 1.2x signals conviction |
| **Accel** | Momentum acceleration — current 5d rate vs prior 5d rate |
| **Streak** | Consecutive up or down days |
| **52w Hi** | Distance from 52-week high — near-high themes tend to maintain momentum |

## ETF universe

**Benchmarks:** SPY, QQQ, IWM, DIA

**Sectors:** XLK (Technology), XLF (Financials), XLE (Energy), XLV (Healthcare), XLI (Industrials), XLU (Utilities), XLRE (Real Estate), XLP (Consumer Staples), XLY (Consumer Discretionary), XLC (Communication), XLB (Materials)

**Thematic / Narrative:** SMH (Semiconductors), IGV (Software), XBI (Biotech), ARKK (Disruption), HACK (Cybersecurity), TAN (Solar), ROBO (Robotics & AI), GDX (Gold Miners), KWEB (China Internet), XHB (Homebuilders), JETS (Airlines), BITQ (Crypto/Blockchain)

## Quick start

```bash
pip install -r requirements.txt
python3 tracker.py
```

Outputs:
- **Terminal report** with ranked momentum, rotation signals, and signal summary
- **`momentum_latest.html`** — self-contained dashboard you can open in any browser
- **`momentum_latest.json`** — structured data for further analysis
- **`momentum_YYYY-MM-DD.csv`** — daily snapshot

## Composite score

The composite momentum score weights shorter timeframes more heavily (they catch emerging momentum) and adjusts for volume:

```
price_score = 5d_roc * 0.40 + 10d_roc * 0.35 + 20d_roc * 0.25
volume_factor = clamp(vol_ratio, 0.5, 2.0)
composite = price_score * volume_factor
```

Above-average volume amplifies the score. Below-average volume dampens it.

## Rotation detection

A rotation signal fires when:
- **INTO:** 5-day relative strength vs SPY > +1.5%, accelerating vs 10-day RS, on above-average volume
- **OUT OF:** 5-day relative strength vs SPY < -1.5%, decelerating vs 10-day RS, on above-average volume

This catches money moving between themes before the narrative catches up.

## Customization

Edit the `ETFS` dictionary in `tracker.py` to add or remove tickers. Each entry needs a `name` and a `group` (`Benchmark`, `Sector`, or `Thematic`):

```python
"DFEN": {"name": "Defense", "group": "Thematic"},
"XME": {"name": "Metals & Mining", "group": "Thematic"},
"IBB": {"name": "Biotech (large)", "group": "Thematic"},
```

## Requirements

- Python 3.9+
- `yfinance`, `pandas`, `numpy` (see `requirements.txt`)
- Internet connection for Yahoo Finance data

## License

MIT
