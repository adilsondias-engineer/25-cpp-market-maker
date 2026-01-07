# Project 25: Market Maker FSM (XGBoost + Strategy)

## Part of FPGA Trading Systems Portfolio

This project is part of a complete end-to-end trading system:
- **Main Repository:** [fpga-trading-systems](https://github.com/adilsondias-engineer/fpga-trading-systems)
- **Project Number:** 25 of 30
- **Category:** C++ Application
- **Dependencies:** Project 24 (Order Gateway - Disruptor output), Project 26 (Order Execution)

---

**Platform:** Linux
**Technology:** C++20, LMAX Disruptor, XGBoost (CUDA 13.0), spdlog, nlohmann/json
**Status:** Completed

---

## Overview

The Market Maker FSM is an automated trading strategy that consumes raw BBO (Best Bid/Offer) data from Project 24 via **LMAX Disruptor lock-free IPC**, performs **XGBoost GPU inference** locally, and generates two-sided quotes with position management and risk controls.

**Design Decision:** XGBoost inference runs in P25 (not P24) for **pipeline parallelism**. This allows P24 to process the next BBO while P25 runs GPU inference on the previous BBO, reducing effective latency.

**Data Flow:**
```
Project 24 (Order Gateway)
    ↓ Disruptor IPC (/dev/shm/bbo_ring_gateway)
    ↓ Raw BBO Data
Project 25 (Market Maker FSM)  ← YOU ARE HERE
    ├─ Disruptor Consumer (raw BBO)
    ├─ XGBoost GPU Predictor (84% accuracy, ~10-100μs)
    ├─ Fair Value Calculation
    ├─ Quote Generation (with prediction-based edge)
    ├─ Risk Management
    └─ Order Producer → P26
    ↓ Disruptor IPC (/dev/shm/order_ring_mm)
Project 26 (Order Execution)
    ↓ Simulated Fills
    ↓ Disruptor IPC (/dev/shm/fill_ring_oe)
Project 25 (receives fills)
    └─ Position & PnL Updates
```

**Key Features:**
- **XGBoost GPU Inference:** Local GPU prediction (84% accuracy, ~10-100 μs)
- **Disruptor-Only IPC:** No TCP/network dependencies, pure shared memory
- **Prediction-Aware Trading:** Uses XGBoost confidence to adjust edge
- **Position Skew:** Inventory management to reduce risk
- **Risk Controls:** Maximum position and notional limits

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│              Project 24 (Order Gateway - PCIe Passthrough)       │
│  PCIe Listener → BBO Validator → Disruptor Producer (raw BBO)   │
└────────────────────────┬────────────────────────────────────────┘
                         │
        POSIX Shared Memory (/dev/shm/bbo_ring_gateway)
        Ring Buffer: 1024 entries × 128 bytes = 131 KB
        Lock-Free IPC: Atomic sequence numbers (~0.50 μs)
        Contains: Raw BBO data (no prediction)
                         │
┌────────────────────────┴────────────────────────────────────────┐
│                    Market Maker FSM (Project 25)                 │
│                                                                  │
│  ┌────────────────┐     ┌──────────────────────────┐            │
│  │  Disruptor     │────→│     BBO Parser           │            │
│  │  Consumer      │     │  (Fixed-size structs)    │            │
│  │  (Lock-Free)   │     │  Raw BBO data            │            │
│  └────────────────┘     └──────────┬───────────────┘            │
│                                    │                             │
│                                    ↓                             │
│                         ┌──────────────────┐                     │
│                         │  XGBoost GPU     │                     │
│                         │  Predictor       │                     │
│                         │  (CUDA 13.0)     │                     │
│                         │  84% accuracy    │                     │
│                         │  ~10-100 μs      │                     │
│                         └──────────┬───────┘                     │
│                                    │                             │
│                                    ↓                             │
│                         ┌──────────────────┐                     │
│                         │  Market Maker    │                     │
│                         │      FSM         │                     │
│                         └─────────┬────────┘                     │
│                                   │                              │
│          ┌────────────────────────┼────────────────┐             │
│          ↓                        ↓                ↓             │
│  ┌──────────────┐      ┌──────────────┐  ┌──────────────┐       │
│  │ Fair Value   │      │ Quote        │  │ Risk         │       │
│  │ Calculation  │      │ Generation   │  │ Management   │       │
│  │              │      │ (w/ XGBoost) │  │              │       │
│  └──────────────┘      └──────────────┘  └──────────────┘       │
│                                   │                              │
│                                   ↓                              │
│                         ┌──────────────────┐                     │
│                         │  Position        │                     │
│                         │  Tracker         │                     │
│                         └──────────────────┘                     │
│                                   │                              │
│                                   ↓                              │
│                         ┌──────────────────┐                     │
│                         │  Order Producer  │                     │
│                         │  (Disruptor)     │                     │
│                         └──────────┬───────┘                     │
│                                    │                             │
└────────────────────────────────────┼─────────────────────────────┘
                                     │
        POSIX Shared Memory (/dev/shm/order_ring_mm)
        Orders to Project 26 (Order Execution)
                                     │
┌────────────────────────────────────┼─────────────────────────────┐
│                                    ↓                             │
│                    Project 26 (Order Execution)                  │
│                    Simulated Fills → Fill Producer               │
└────────────────────────────────────┬─────────────────────────────┘
                                     │
        POSIX Shared Memory (/dev/shm/fill_ring_oe)
        Fills back to Project 25
                                     │
                                     ↓
                    Project 25: Position & PnL Updates
```

---

## FSM States

The market maker operates as a finite state machine:

1. **IDLE** - Waiting for BBO updates from Disruptor
2. **PREDICT** - Running XGBoost GPU inference on BBO data
3. **CALCULATE** - Computing fair value from BBO + prediction
4. **QUOTE** - Generating bid/ask quotes with position skew
5. **RISK_CHECK** - Validating position and notional limits
6. **ORDER_GEN** - Sending orders via Disruptor to P26
7. **WAIT_FILL** - Waiting for fills from P26

**State Transitions:**
```
IDLE → PREDICT → CALCULATE → QUOTE → RISK_CHECK → ORDER_GEN → WAIT_FILL → PREDICT
                    ↑                      ↓
                    └──────────────────────┘
                         (Risk Failed)
```

---

## Features

### 1. XGBoost GPU Predictor

GPU-accelerated market prediction using XGBoost C API:

- **Model:** `model/itch_predictor.ubj` (36 MB, UBJson format)
- **Accuracy:** 84% (trained on historical ITCH data)
- **Backend:** CUDA 13.0 GPU acceleration (RTX 5090)
- **Latency:** ~10-100 μs per prediction
- **Output:** Prediction probability (up/down/neutral) with confidence

**XGBoost Configuration:**
```json
{
  "xgboost": {
    "model_path": "model/itch_predictor.ubj",
    "use_gpu": true,
    "gpu_id": 0,
    "confidence_threshold": 0.6
  }
}
```

**Pipeline Parallelism:** By running XGBoost in P25 (not P24), P24 can process the next BBO from PCIe while P25 runs GPU inference on the previous BBO. This overlaps PCIe I/O with GPU computation for lower effective latency.

### 2. Prediction-Aware Trading

Uses prediction confidence to adjust trading behavior:
- **High confidence (>0.8):** Wider edge, more aggressive quotes
- **Medium confidence (0.6-0.8):** Standard edge
- **Low confidence (<0.6):** Narrower edge, more conservative

### 3. Fair Value Calculation

```
mid_price = (bid_price + ask_price) / 2
weighted_price = (bid_price * bid_shares + ask_price * ask_shares) / (bid_shares + ask_shares)
fair_value = (mid_price + weighted_price) / 2
```

### 4. Quote Generation with Position Skew

```
edge = fair_value * (edge_bps / 10000) * confidence_factor
inventory_ratio = current_position / max_position
skew = fair_value * (position_skew_bps / 10000) * inventory_ratio

bid_price = fair_value - edge + skew
ask_price = fair_value + edge + skew
```

**Position Skew Logic:**
- Long position: Skew quotes DOWN to encourage selling
- Short position: Skew quotes UP to encourage buying
- Larger positions = larger skew adjustment

### 5. Position Management

- Real-time position tracking per symbol
- Realized PnL calculation on fills
- Unrealized PnL marking to market
- Weighted average entry price

### 6. Risk Controls

- Maximum position limits per symbol
- Maximum notional exposure limits
- Pre-trade risk checks
- Automatic quote rejection when limits exceeded

---

## Configuration

**config.json:**
```json
{
  "log_level": "info",
  "min_spread_bps": 5.0,
  "edge_bps": 2.0,
  "max_position": 500,
  "position_skew_bps": 1.0,
  "quote_size": 100,
  "max_notional": 1000000.0,
  "xgboost": {
    "model_path": "model/itch_predictor.ubj",
    "use_gpu": true,
    "gpu_id": 0,
    "confidence_threshold": 0.6
  },
  "disruptor": {
    "shm_name": "gateway"
  },
  "enable_order_execution": true,
  "order_ring_path": "order_ring_mm",
  "fill_ring_path": "fill_ring_oe",
  "performance": {
    "enable_rt": false,
    "cpu_cores": [4, 5]
  }
}
```

### Configuration Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `log_level` | string | Log level: trace, debug, info, warn, error | info |
| `min_spread_bps` | double | Minimum spread in basis points | 5.0 |
| `edge_bps` | double | Edge added to fair value (bps) | 2.0 |
| `max_position` | int | Maximum position per symbol | 500 |
| `position_skew_bps` | double | Inventory skew adjustment (bps) | 1.0 |
| `quote_size` | int | Quote size per side | 100 |
| `max_notional` | double | Maximum notional exposure | 1000000.0 |
| `xgboost.model_path` | string | Path to XGBoost model file | model/itch_predictor.ubj |
| `xgboost.use_gpu` | bool | Enable GPU acceleration | true |
| `xgboost.gpu_id` | int | CUDA GPU device ID | 0 |
| `xgboost.confidence_threshold` | double | Minimum prediction confidence | 0.6 |
| `disruptor.shm_name` | string | Shared memory name for BBO ring | gateway |
| `enable_order_execution` | bool | Enable order execution to P26 | true |
| `order_ring_path` | string | Shared memory for orders to P26 | order_ring_mm |
| `fill_ring_path` | string | Shared memory for fills from P26 | fill_ring_oe |
| `performance.enable_rt` | bool | Enable RT scheduling | false |
| `performance.cpu_cores` | array | CPU cores for affinity | [4, 5] |

---

## Build Instructions

### Prerequisites

- Linux (Ubuntu 22.04+ recommended)
- GCC 11+ or Clang 14+ (C++20 support)
- CMake 3.20+
- CUDA Toolkit 13.0 (for XGBoost GPU)

### Dependencies

```bash
# Install vcpkg dependencies
vcpkg install spdlog nlohmann-json

# XGBoost is built from source in model/xgboost/
```

### Build

```bash
cd 25-market-maker
mkdir build
cd build
cmake ..
make -j$(nproc)
```

---

## Usage

```bash
# Run with default config.json
./market_maker

# Run with custom config file
./market_maker config_prod.json

# With RT optimizations (requires CAP_SYS_NICE)
sudo setcap cap_sys_nice=eip ./market_maker
./market_maker
```

---

## Performance

### Expected Latency

| Component | Latency | Notes |
|-----------|---------|-------|
| Disruptor Consumer (from P24) | ~0.5 μs | Lock-free poll |
| BBO Parse | ~0.1 μs | Fixed-size struct |
| XGBoost GPU Inference | ~10-100 μs | CUDA prediction |
| Fair Value Calculation | < 0.1 μs | Simple arithmetic |
| Quote Generation | < 0.1 μs | Price calculation + skew |
| Risk Check | < 0.05 μs | Position limit checks |
| Order Producer (to P26) | ~0.5 μs | Lock-free publish |
| **Total FSM Processing** | **~12-102 μs** | Including GPU inference |

### End-to-End Performance

| Path | Latency | Notes |
|------|---------|-------|
| P24 → P25 (Raw BBO) | ~0.5 μs | Disruptor IPC |
| P25 XGBoost Inference | ~10-100 μs | GPU prediction |
| P25 FSM Processing | ~1-2 μs | Trading logic |
| P25 → P26 (Orders) | ~0.5 μs | Disruptor IPC |
| P26 → P25 (Fills) | ~0.5 μs | Disruptor IPC |
| **Total Round-Trip** | **~13-105 μs** | Quote to fill |

**Pipeline Parallelism Benefit:** While P25 runs inference, P24 can read the next BBO from PCIe. This overlap reduces effective latency in steady-state operation.

---

## Code Structure

```
25-market-maker/
├── config.json               # Configuration file
├── CMakeLists.txt            # Build configuration
├── src/
│   ├── main.cpp              # Entry point, config loading
│   ├── market_maker_fsm.cpp  # FSM implementation
│   ├── xgboost_predictor.cpp # XGBoost GPU inference
│   ├── feature_extractor.cpp # BBO to feature vector
│   ├── position_tracker.cpp  # Position and PnL tracking
│   └── order_producer.cpp    # Disruptor producer to P26
├── include/
│   ├── market_maker_fsm.h    # FSM class definition
│   ├── xgboost_predictor.h   # XGBoost predictor interface
│   ├── feature_extractor.h   # Feature extractor interface
│   ├── position_tracker.h    # Position tracker
│   ├── order_types.h         # BBO, Quote, Order, Fill structs
│   ├── order_producer.h      # Order producer interface
│   └── disruptor_client.h    # Disruptor consumer interface
├── model/
│   ├── itch_predictor.ubj    # XGBoost model (36 MB)
│   └── xgboost/              # XGBoost source (for building)
└── vcpkg.json                # Dependency manifest
```

---

## Example Output

```
[info] Loaded config from config.json
[info] XGBoost model loaded: model/itch_predictor.ubj (36 MB)
[info] XGBoost GPU enabled: device 0 (RTX 5090)
[info] MarketMakerFSM initialized: spread=5 bps, edge=2 bps, max_pos=500
[info] Connected to BBO ring: /dev/shm/bbo_ring_gateway
[info] Connected to order ring: /dev/shm/order_ring_mm
[info] Connected to fill ring: /dev/shm/fill_ring_oe
[info] Market Maker FSM running (Disruptor mode)

[debug] BBO: AAPL bid=150.00x100 ask=150.05x100
[debug] PREDICT: inference=45μs, prediction=UP, conf=0.82
[debug] CALCULATE: fair_value=150.0250, spread=0.0500
[debug] QUOTE: bid=149.9950x100, ask=150.0550x100
[debug] RISK_CHECK: Passed (pos=0, max=500)
[info] ORDER: BID 100@149.9950 / ASK 100@150.0550

[info] FILL: BUY 100 AAPL @ 149.9950
[info] POSITION: AAPL +100 shares, entry=149.9950, unrealized=0.00

^C
[info] Shutdown signal received
[info] Final PnL: realized=0.00, unrealized=0.00
[info] Shutdown complete
```

---

## Related Projects

- **[23-order-book/](../23-order-book/)** - FPGA order book (PCIe source)
- **[24-order-gateway/](../24-order-gateway/)** - Order Gateway (PCIe passthrough, raw BBO producer)
- **[26-order-execution/](../26-order-execution/)** - Order Execution (simulated fills)
- **[28-complete-system/](../28-complete-system/)** - System Orchestrator

---

## Data Flow (Full System)

```
FPGA (Project 23)
    ↓ PCIe Gen2 x4 (/dev/xdma0_c2h_0)
Project 24: Order Gateway
    ├─ PCIeListener (48-byte BBO packets)
    ├─ BBOValidator (filter invalid data)
    └─ Disruptor Producer (raw BBO)
    ↓ Shared Memory (/dev/shm/bbo_ring_gateway)
Project 25: Market Maker  ← YOU ARE HERE
    ├─ Disruptor Consumer (raw BBO)
    ├─ XGBoostPredictor (84% accuracy, ~10-100μs)
    ├─ MarketMakerFSM (strategy logic)
    ├─ PositionTracker (PnL management)
    └─ OrderProducer (orders to P26)
    ↓ Shared Memory (/dev/shm/order_ring_mm)
Project 26: Order Execution
    ├─ Order Consumer
    ├─ Simulated Fill (~50 μs latency)
    └─ Fill Producer (fills to P25)
    ↓ Shared Memory (/dev/shm/fill_ring_oe)
Project 25: (receives fills)
    └─ Updates position, PnL
```

---

**Build Time:** ~60 seconds (including XGBoost)
**Hardware Status:** Tested with P24 Disruptor producer

---

## Known Issues

### CUDA 13.1 / CCCL 3.1.2 Incompatibility (Resolved with CUDA 13.0)

**Current Status:** GPU inference is **ENABLED** using CUDA 13.0 (CCCL 3.0.1).

**Background:** CUDA 13.1 ships with CCCL 3.1.2 which has breaking changes to the CUB `DeviceRunLengthEncode` API. XGBoost's `algorithm.cuh` uses the older API signature, causing compilation errors. CUDA 13.0 (CCCL 3.0.1) is compatible.

**glibc 2.42 rsqrt Conflict:** glibc 2.42+ (Ubuntu 25.04+) introduces `rsqrt()` with `noexcept(true)` (C23 IEC 60559) that conflicts with CUDA's rsqrt device function. This requires a one-time patch to CUDA headers.

**Current Configuration:**
- CUDA 13.0.48 with CCCL 3.0.1
- XGBoost 3.2.0 with GPU support
- glibc 2.42 rsqrt patch applied
- GPU inference: ~10-50 μs per prediction

**If Rebuilding on a New System:**

1. **Install CUDA 13.0** (not 13.1) from NVIDIA:
```bash
# CUDA 13.0 is required for XGBoost GPU compatibility
# CUDA 13.1 (CCCL 3.1.2) has breaking API changes
```

2. **Apply glibc 2.42 rsqrt patch** (required for Ubuntu 25.04+):
```bash
sudo sed -i.bak \
  -e 's/extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ double                 rsqrt(double x);/extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ double                 rsqrt(double x) noexcept(true);/' \
  -e 's/extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  rsqrtf(float x);/extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  rsqrtf(float x) noexcept(true);/' \
  /usr/local/cuda-13.0/targets/x86_64-linux/include/crt/math_functions.h
```

3. **Build:**
```bash
rm -rf build && mkdir build && cd build && cmake .. && make -j$(nproc)
```

**To Revert the rsqrt Patch:**
```bash
sudo mv /usr/local/cuda-13.0/targets/x86_64-linux/include/crt/math_functions.h.bak \
        /usr/local/cuda-13.0/targets/x86_64-linux/include/crt/math_functions.h
```

**Future:** When XGBoost releases a CCCL 3.1.2-compatible version, upgrade to CUDA 13.1+ for latest features.
