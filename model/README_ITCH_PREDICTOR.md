# ITCH Predictor C++ Implementation

This directory contains a C++ implementation of the ITCH order book predictor using XGBoost.

## Files

- `itch_predictor.hpp` - Header file with `ITCHPredictor` class and `BboUpdate` struct
- `itch_predictor.cpp` - Implementation file
- `test_itch_predictor.cpp` - Test program
- `convert_model_to_xgb.py` - Script to convert joblib model to XGBoost native format
- `full_optmization2.py` - Python training script (reference)

## Building

### Prerequisites

1. **XGBoost C API**: Install XGBoost with C API support
   ```bash
   # Option 1: Build from source
   git clone https://github.com/dmlc/xgboost.git
   cd xgboost
   mkdir build && cd build
   cmake .. -DUSE_CUDA=OFF
   cmake --build . --config Release
   
   # Option 2: Use pre-built libraries
   # Download from: https://xgboost.readthedocs.io/en/latest/build.html
   ```

2. **C++ Compiler**: GCC 7+, Clang 5+, or MSVC 2017+

### Compilation

```bash
# Linux/Mac
g++ -std=c++17 -O2 -I/path/to/xgboost/include \
    -L/path/to/xgboost/lib \
    -o test_itch_predictor \
    itch_predictor.cpp test_itch_predictor.cpp \
    -lxgboost -pthread

# Windows (MSVC)
# Option 1: Normal linking (DLL must be in PATH or same directory as .exe)
cl /EHsc /std:c++17 /O2 /I"C:\path\to\xgboost\include" \
   itch_predictor.cpp test_itch_predictor.cpp \
   /link /LIBPATH:"C:\path\to\xgboost\lib" xgboost.lib

# Option 2: Delay loading (DLL loaded only when needed - recommended)
# Note: delayimp.lib is required for delay loading
cl /EHsc /std:c++17 /O2 /I"C:\path\to\xgboost\include" \
   itch_predictor.cpp test_itch_predictor.cpp \
   /link /LIBPATH:"C:\path\to\xgboost\lib" xgboost.lib delayimp.lib /DELAYLOAD:xgboost.dll
```

## Converting the Model

XGBoost C API requires models in native format (`.ubj` or `.json`), not joblib format.

```bash
# Convert joblib model to XGBoost native format
python convert_model_to_xgb.py \
    --input itch_predictor_xgb_81pct.joblib \
    --output itch_predictor.ubj \
    --format ubj
```

## Usage

### Basic Usage

```cpp
#include "itch_predictor.hpp"

// Load model
ITCHPredictor predictor("itch_predictor.ubj");

// Update with BBO data
BboUpdate bbo(
    150.25f,  // bid_price
    100.0f,   // bid_size
    150.30f,  // ask_price
    100.0f,   // ask_size
    150.275f, // mid_price
    0.05f,    // spread
    3.33f,    // spread_bps
    0.0f,     // imbalance
    500.0f,   // bid_depth
    500.0f,   // ask_depth
    0.5f,     // aggressor_ratio
    1000.0f,  // trade_volume
    123456789000000000LL // timestamp_ns
);

predictor.update(bbo);

// Need 20 snapshots before prediction
// ... add 19 more updates ...

// Predict
int prediction = predictor.predict(); // 0=DOWN, 1=NEUTRAL, 2=UP
float confidence = predictor.confidence(); // 0.0 to 1.0
```

### Running the Test

```bash
./test_itch_predictor itch_predictor.ubj
```

## Feature Extraction

The implementation matches the Python feature extraction in `full_optmization2.py`:

1. **Flattened snapshots**: 20 snapshots × 20 features = 400 features
2. **Derived features**: ~29 additional features including:
   - Price changes and velocities
   - Spread statistics
   - Order flow metrics
   - Imbalance metrics
   - Volatility measures

Total feature count: ~429 features

## Model Requirements

- **Sequence length**: 20 snapshots (2 seconds @ 100ms intervals)
- **Features per snapshot**: 20 features
- **Output classes**: 3 (DOWN=0, NEUTRAL=1, UP=2)
- **Model format**: XGBoost native (.ubj or .json)

## Notes

- The predictor maintains a sliding window of the last 20 snapshots
- Predictions require at least 20 snapshots
- The model expects features in the exact order as defined in `BboUpdate::to_features()`
- Feature extraction matches the Python training code exactly

## Troubleshooting

### "Failed to load XGBoost model"
- Ensure the model file is in XGBoost native format (`.ubj` or `.json`), not joblib
- Use `convert_model_to_xgb.py` to convert the model

### "Failed to create DMatrix"
- Check that feature vector size matches model expectations (~429 features)
- Verify all features are valid floats (not NaN or Inf)

### Link errors
- Ensure XGBoost library is properly linked
- Check that XGBoost C API headers are in include path
- On Windows, may need to link additional dependencies (pthread, etc.)

