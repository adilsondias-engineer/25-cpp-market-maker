# ITCH Training Data Format Specification

**Version:** 1.0  
**Date:** December 2025  
**Purpose:** Define the data format for fine-tuning Llama 4 Scout on NASDAQ ITCH order flow data

---

## 1. Overview

This document specifies the training data format for fine-tuning Llama 4 Scout to predict
BBO (Best Bid/Offer) price movements from NASDAQ ITCH 5.0 order book data.

### 1.1 Task Definition

**Input:** Sequence of 100 order book snapshots (10 seconds @ 100ms intervals)  
**Output:** Price direction prediction over next 500ms  
**Labels:** UP (2), NEUTRAL (1), DOWN (0)

### 1.2 Data Sources

- **Primary:** NASDAQ ITCH 5.0 binary files (e.g., `12302019.NASDAQ_ITCH50`)
- **Secondary:** MySQL database with imported ITCH messages
- **Symbols:** AAPL, TSLA, SPY, QQQ, GOOGL, MSFT, AMZN, NVDA (8 symbols)

---

## 2. File Format

### 2.1 JSONL Structure

Each line is a JSON object representing one training sample:

```json
{
  "symbol": "AAPL",
  "timestamp": "2019-12-30T09:30:05.123456",
  "sequence_id": 12345,
  "features": [
    [0.123, 0.456, ...],
    [0.124, 0.457, ...],
    ...
  ],
  "label": 2,
  "label_horizon_ms": 500,
  "metadata": {
    "current_price": 150.25,
    "future_price": 150.35,
    "return_bps": 6.65,
    "num_orders": 1234,
    "num_trades": 56
  }
}
```

### 2.2 Field Descriptions

| Field | Type | Description |
|-------|------|-------------|
| `symbol` | string | Stock ticker (e.g., "AAPL") |
| `timestamp` | string | ISO 8601 timestamp of last snapshot in sequence |
| `sequence_id` | int | Unique identifier for this sequence |
| `features` | float[][] | Shape (100, 200) - 100 snapshots × 200 features |
| `label` | int | Prediction target: 0=DOWN, 1=NEUTRAL, 2=UP |
| `label_horizon_ms` | int | Prediction horizon in milliseconds |
| `metadata` | object | Additional context for analysis |

### 2.3 File Splits

```
data/
├── itch_train.jsonl    # 80% of samples (~64,000)
├── itch_eval.jsonl     # 10% of samples (~8,000)
├── itch_test.jsonl     # 10% of samples (~8,000)
└── stats.json          # Dataset statistics
```

---

## 3. Feature Specification

### 3.1 Feature Vector (200 dimensions)

Each snapshot contains 200 features organized into 8 categories:

| Category | Features | Indices | Description |
|----------|----------|---------|-------------|
| Basic BBO | 4 | 0-3 | Bid/ask price and size |
| Spread Metrics | 4 | 4-7 | Spread, mid price, microprice |
| Order Book Imbalance | 8 | 8-15 | Imbalance at 4 price levels |
| Price Momentum | 12 | 16-27 | Returns at 6 lookback windows |
| Volume Metrics | 8 | 28-35 | Volume, VWAP, depth |
| Volatility | 8 | 36-43 | Realized and Parkinson vol |
| Order Flow | 16 | 44-59 | Aggressor ratio, toxicity |
| Cross-Symbol | 140 | 60-199 | Correlations, relative strength |

### 3.2 Detailed Feature Definitions

#### 3.2.1 Basic BBO (Indices 0-3)

```
Index  Name        Description                          Units
-----  ----------  -----------------------------------  --------
0      bid_price   Best bid price                       USD
1      bid_size    Shares at best bid                   shares
2      ask_price   Best ask price                       USD
3      ask_size    Shares at best ask                   shares
```

#### 3.2.2 Spread Metrics (Indices 4-7)

```
Index  Name        Description                          Units
-----  ----------  -----------------------------------  --------
4      spread_abs  Absolute spread (ask - bid)          USD
5      spread_bps  Spread in basis points               bps
6      mid_price   (bid + ask) / 2                      USD
7      micro_price Size-weighted mid price              USD
```

**Microprice Formula:**
```
microprice = (bid_price × ask_size + ask_price × bid_size) / (bid_size + ask_size)
```

#### 3.2.3 Order Book Imbalance (Indices 8-15)

For each of 4 price levels:
```
Index    Name               Description
-----    ---------------    ------------------------------------
8, 10,   bid_imbalance_Ln   (bid_size - ask_size) / (bid + ask)
12, 14                      at price level n (n=1,2,3,4)

9, 11,   total_size_Ln      bid_size + ask_size at level n
13, 15
```

**Interpretation:**
- Positive imbalance → More bids than asks → Buying pressure
- Negative imbalance → More asks than bids → Selling pressure

#### 3.2.4 Price Momentum (Indices 16-27)

For each lookback window [1, 5, 10, 20, 50, 100] ticks:
```
Index   Name            Description
-----   --------------  ------------------------------------
16, 18  return_Nt       (price_now - price_N_ago) / price_N_ago
20, 22                  where N = 1, 5, 10, 20, 50, 100
24, 26

17, 19  return_vol_Nt   Standard deviation of returns over
21, 23                  last N ticks
25, 27
```

#### 3.2.5 Volume Metrics (Indices 28-35)

```
Index  Name             Description
-----  ---------------  ------------------------------------
28     bid_vol_total    Total shares on bid side (all levels)
29     ask_vol_total    Total shares on ask side (all levels)
30     vwap_bid         Volume-weighted avg price (bid top 5)
31     vwap_ask         Volume-weighted avg price (ask top 5)
32     vol_imbalance    (bid_vol - ask_vol) / (bid_vol + ask_vol)
33     trade_vol_1m     Total trade volume in last 1 minute
34     bid_depth_1pct   Shares within 1% of mid (bid side)
35     ask_depth_1pct   Shares within 1% of mid (ask side)
```

#### 3.2.6 Volatility (Indices 36-43)

For each window [10, 30, 60, 120] seconds:
```
Index    Name                 Description
-----    ------------------   ------------------------------------
36, 38   realized_vol_Ns      Annualized realized volatility
40, 42                        sqrt(252 × 6.5 × 3600 / N) × std(returns)

37, 39   parkinson_vol_Ns     Parkinson (high-low) volatility
41, 43                        sqrt(log(high/low)² / (4 × ln(2)))
```

#### 3.2.7 Order Flow (Indices 44-59)

```
Index  Name                Description
-----  ------------------  ------------------------------------
44     aggressor_ratio     Buy-initiated volume / total volume (1m)
45     trade_imbalance_1m  (buy_vol - sell_vol) / total_vol (1m)
46     trade_imbalance_5m  Same as above, 5 minute window
47     order_arrival_rate  Orders per second (since open)
48     cancel_rate         Cancels per second
49     modify_rate         Modifies per second
50     large_trade_ind     1 if last trade > 2× average, else 0
51     sweep_indicator     1 if trade crossed multiple levels
52     bid_pressure        bid_size / (bid_size + ask_size)
53     ask_pressure        ask_size / (bid_size + ask_size)
54     flow_toxicity       |trade_imbalance| (simplified VPIN)
55     kyle_lambda_est     Price impact estimate
56     queue_pos_bid       Estimated queue position (bid)
57     queue_pos_ask       Estimated queue position (ask)
58     time_at_best_bid    Time since last BBO change (bid)
59     time_at_best_ask    Time since last BBO change (ask)
```

**Key Signals:**
- `aggressor_ratio > 0.5` → Net buying pressure
- `flow_toxicity > 0.3` → Informed trading likely
- `large_trade_ind = 1` → Institutional activity

#### 3.2.8 Cross-Symbol Features (Indices 60-199)

For each symbol pair (current symbol vs other 7 symbols):
```
For symbol S in [AAPL, TSLA, QQQ, GOOGL, MSFT, AMZN, NVDA] (excluding SPY):

Index         Name              Description
-----         ---------------   ------------------------------------
60+4n         corr_spy_S        Rolling correlation with SPY
61+4n         rel_strength_S    Current price / S price
62+4n         beta_S            Beta vs S (simplified)
63+4n         spread_ratio_S    Current spread / S spread

where n = 0..6 for each of 7 symbols
```

**Total:** 7 symbols × 4 features = 28 features per symbol × 5 = 140 features

---

## 4. Label Specification

### 4.1 Label Definition

Labels are determined by price movement over the prediction horizon:

```python
def generate_label(current_mid: float, future_mid: float, threshold_bps: float = 5.0) -> int:
    return_bps = (future_mid - current_mid) / current_mid * 10000
    
    if return_bps > threshold_bps:
        return 2  # UP
    elif return_bps < -threshold_bps:
        return 0  # DOWN
    else:
        return 1  # NEUTRAL
```

### 4.2 Label Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `horizon_ms` | 500 | Prediction horizon in milliseconds |
| `threshold_bps` | 5.0 | Movement threshold in basis points |

### 4.3 Label Distribution (Expected)

Based on typical market conditions:

| Label | Description | Expected % | Actual (Sample) |
|-------|-------------|------------|-----------------|
| 0 | DOWN | 30-35% | 32% |
| 1 | NEUTRAL | 30-40% | 36% |
| 2 | UP | 30-35% | 32% |

**Note:** Distribution may vary by:
- Market conditions (volatile vs calm)
- Time of day (open/close vs midday)
- Symbol (high-beta stocks more volatile)

---

## 5. Text Formatting for LLM

### 5.1 Prompt Structure

The numerical features are converted to natural language for LLM consumption:

```
### Market Data Analysis for AAPL
Timestamp: 2019-12-30T09:30:05.123456
Sequence Length: 100 snapshots

#### Current Order Book State (Latest Snapshot):
- Bid: 150.2500 x 500
- Ask: 150.2700 x 300
- Spread: 0.80 bps
- Mid Price: 150.2600

#### Order Book Imbalance:
- Level 1 Imbalance: 0.2500 (positive = more bids)

#### Price Momentum:
- 1-tick return: 0.0002
- 10-tick return: 0.0015
- 100-tick return: 0.0045

#### Volatility:
- 10s realized vol: 0.1234
- 60s realized vol: 0.1456

#### Order Flow Metrics:
- Aggressor Ratio: 0.5500 (>0.5 = buy pressure)
- Trade Imbalance: 0.1200
- Flow Toxicity: 0.1200

#### Historical Sequence Statistics:
- Mean Mid Price: 150.2400
- Price Trend: 0.000023
- Volume Trend: 12.5000
- Volatility Trend: 0.000012

#### Full Feature Vector (Latest):
- bid_price: 150.250000
- bid_size: 500.000000
- ask_price: 150.270000
- ask_size: 300.000000
- spread_bps: 0.800000
- mid_price: 150.260000
- bid_imbalance_L1: 0.250000
- return_1t: 0.000200
- return_10t: 0.001500
- return_100t: 0.004500
- realized_vol_10s: 0.123400
- realized_vol_60s: 0.145600
- aggressor_ratio: 0.550000
- trade_imbalance_1m: 0.120000
- flow_toxicity: 0.120000
```

### 5.2 Response Format

The model is trained to respond with:

```
Based on my analysis:

1. Key observations:
- Order book shows positive imbalance (more bids than asks)
- Recent price momentum is positive
- Order flow indicates buying pressure (aggressor ratio > 0.5)
- Volatility is moderate, supporting directional move

2. Prediction: **UP**

3. Confidence: MEDIUM
```

### 5.3 Llama 4 Chat Template

```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an expert quantitative trader analyzing real-time market data from NASDAQ ITCH 5.0 feed...<|eot_id|><|start_header_id|>user<|end_header_id|>

Analyze the following market data and predict the price direction for the next 500ms:

[MARKET DATA TEXT]

Provide your prediction...<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Based on my analysis:
...<|eot_id|>
```

---

## 6. Data Quality Requirements

### 6.1 Feature Validation

All features must pass these checks:

```python
def validate_features(features: np.ndarray) -> bool:
    # Shape check
    assert features.shape == (100, 200), "Invalid shape"
    
    # No NaN/Inf values
    assert not np.isnan(features).any(), "Contains NaN"
    assert not np.isinf(features).any(), "Contains Inf"
    
    # Price sanity (assuming normalized)
    assert features[:, 0].min() >= -10, "Bid price too low"
    assert features[:, 2].min() >= -10, "Ask price too low"
    
    # Spread should be non-negative
    assert (features[:, 4] >= 0).all(), "Negative spread"
    
    return True
```

### 6.2 Label Validation

```python
def validate_label(label: int, metadata: dict) -> bool:
    # Label in valid range
    assert label in [0, 1, 2], "Invalid label"
    
    # Metadata consistency
    return_bps = metadata['return_bps']
    
    if label == 2:  # UP
        assert return_bps > 5.0, "Label/return mismatch"
    elif label == 0:  # DOWN
        assert return_bps < -5.0, "Label/return mismatch"
    else:  # NEUTRAL
        assert -5.0 <= return_bps <= 5.0, "Label/return mismatch"
    
    return True
```

### 6.3 Sequence Continuity

```python
def validate_sequence(sample: dict) -> bool:
    features = np.array(sample['features'])
    
    # No large price jumps (> 1% between consecutive snapshots)
    mid_prices = (features[:, 0] + features[:, 2]) / 2
    returns = np.diff(mid_prices) / mid_prices[:-1]
    assert np.abs(returns).max() < 0.01, "Price jump detected"
    
    return True
```

---

## 7. Dataset Statistics

### 7.1 Expected Dataset Size

| Split | Samples | % | Notes |
|-------|---------|---|-------|
| Train | 64,000 | 80% | Main training set |
| Eval | 8,000 | 10% | Validation during training |
| Test | 8,000 | 10% | Final evaluation |
| **Total** | **80,000** | 100% | 10,000 per symbol |

### 7.2 Storage Requirements

| Item | Size | Notes |
|------|------|-------|
| Single sample (JSON) | ~160 KB | 100 × 200 floats + metadata |
| Train file | ~10 GB | 64,000 samples |
| Eval file | ~1.3 GB | 8,000 samples |
| Test file | ~1.3 GB | 8,000 samples |
| **Total** | ~13 GB | Uncompressed JSONL |

### 7.3 Compression (Optional)

```bash
# Compress with gzip for storage
gzip -k data/itch_train.jsonl  # ~2.5 GB compressed

# Read compressed in Python
import gzip
import json

with gzip.open('data/itch_train.jsonl.gz', 'rt') as f:
    for line in f:
        sample = json.loads(line)
```

---

## 8. Usage Examples

### 8.1 Loading Data

```python
import json
import numpy as np

def load_jsonl(filepath: str):
    """Load JSONL file and yield samples."""
    with open(filepath, 'r') as f:
        for line in f:
            yield json.loads(line)

# Load all training samples
train_samples = list(load_jsonl('data/itch_train.jsonl'))
print(f"Loaded {len(train_samples)} training samples")

# Extract features and labels
X = np.array([s['features'] for s in train_samples])  # (N, 100, 200)
y = np.array([s['label'] for s in train_samples])      # (N,)
```

### 8.2 Converting to HuggingFace Dataset

```python
from datasets import Dataset

def sample_to_text(sample: dict) -> dict:
    """Convert sample to text format for LLM."""
    # ... text conversion logic ...
    return {
        'text': formatted_text,
        'label': sample['label'],
        'symbol': sample['symbol'],
    }

# Create HuggingFace dataset
hf_dataset = Dataset.from_generator(
    lambda: (sample_to_text(s) for s in load_jsonl('data/itch_train.jsonl'))
)
```

### 8.3 Batching for Training

```python
from torch.utils.data import DataLoader

def collate_fn(batch):
    """Custom collate function for variable-length sequences."""
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
    }

dataloader = DataLoader(
    train_dataset,
    batch_size=2,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=4,
)
```

---

## 9. Appendix

### 9.1 ITCH Message Types Reference

| Type | Name | Fields Used |
|------|------|-------------|
| A | Add Order | order_ref, buy_sell, shares, symbol, price |
| E | Order Executed | order_ref, executed_shares, match_number |
| X | Order Cancel | order_ref, cancelled_shares |
| D | Order Delete | order_ref |
| U | Order Replace | original_ref, new_ref, shares, price |
| P | Trade | order_ref, buy_sell, shares, symbol, price, match_number |

### 9.2 Symbol Reference

| Symbol | Name | Sector | Avg Daily Volume |
|--------|------|--------|------------------|
| AAPL | Apple Inc | Technology | 50M+ |
| TSLA | Tesla Inc | Consumer | 30M+ |
| SPY | S&P 500 ETF | Index | 80M+ |
| QQQ | Nasdaq 100 ETF | Index | 50M+ |
| GOOGL | Alphabet Inc | Technology | 20M+ |
| MSFT | Microsoft Corp | Technology | 30M+ |
| AMZN | Amazon.com Inc | Consumer | 5M+ |
| NVDA | NVIDIA Corp | Technology | 40M+ |

### 9.3 Feature Normalization

All features are Z-score normalized per sequence:

```python
def normalize(features: np.ndarray) -> np.ndarray:
    """Z-score normalize features across sequence dimension."""
    mean = np.mean(features, axis=0, keepdims=True)
    std = np.std(features, axis=0, keepdims=True) + 1e-8
    return (features - mean) / std
```

**Why per-sequence normalization:**
- Handles different price scales across symbols
- Adapts to changing market conditions
- Preserves relative patterns within sequence

---

## 10. Changelog

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | Dec 2025 | Initial specification |

---

**Document Owner:** Adilson Dias  
**Last Updated:** December 2025
