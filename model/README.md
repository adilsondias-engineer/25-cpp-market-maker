# Llama 4 Scout Fine-Tuning for NASDAQ ITCH Order Flow Prediction

Fine-tune Meta's Llama 4 Scout (17B active / 109B total MoE) on NASDAQ ITCH 5.0 market data for real-time BBO price movement prediction.

## Overview

This project enables training a state-of-the-art LLM to predict short-term price movements from order book data, suitable for integration with FPGA-based trading systems.

### Key Features

- **Model:** Llama 4 Scout (17B active parameters, 109B total, MoE architecture)
- **Data:** NASDAQ ITCH 5.0 order book sequences (200 features × 100 snapshots)
- **Task:** 3-class classification (UP/NEUTRAL/DOWN) at 500ms horizon
- **Training:** LoRA fine-tuning on RTX 5090 (32GB VRAM)
- **Inference:** 5-15ms per prediction with TensorRT optimization

## Quick Start

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install Flash Attention (RTX 5090)
pip install flash-attn --no-build-isolation
```

### 2. Prepare Training Data

**Option A: From ITCH Binary File**
```bash
python prepare_itch_data.py \
    --source file \
    --input /work/projects/12302019.NASDAQ_ITCH50 \
    --output data/ \
    --sequence-length 100 \
    --horizon-ms 500 \
    --threshold-bps 5.0
```

**Option B: From MySQL Database**
```bash
python prepare_itch_data.py \
    --source mysql \
    --mysql-host venus \
    --mysql-db itch_data \
    --mysql-user fpga \
    --mysql-password password \
    --mysql-table itch_messages \
    --output data/ \
    --limit 1000000
```

### 3. Train Model

```bash
# Basic training (BF16)
python train_llama4_scout.py \
    --train-file data/itch_train.jsonl \
    --eval-file data/itch_eval.jsonl \
    --output-dir ./llama4-scout-itch \
    --epochs 5 \
    --batch-size 2 \
    --learning-rate 2e-5

# With 4-bit quantization (larger batches)
python train_llama4_scout.py \
    --train-file data/itch_train.jsonl \
    --eval-file data/itch_eval.jsonl \
    --output-dir ./llama4-scout-itch-4bit \
    --use-4bit \
    --epochs 5 \
    --batch-size 4 \
    --learning-rate 2e-5
```

### 4. Export for Production

```bash
python train_llama4_scout.py \
    --train-file data/itch_train.jsonl \
    --eval-file data/itch_eval.jsonl \
    --output-dir ./llama4-scout-itch \
    --export-path ./llama4-scout-itch-merged
```

## Project Structure

```
llama4_itch_training/
├── train_llama4_scout.py     # Main training script
├── prepare_itch_data.py      # Data preparation pipeline
├── DATA_FORMAT.md            # Training data specification
├── requirements.txt          # Python dependencies
├── README.md                 # This file
└── data/                     # Generated data directory
    ├── itch_train.jsonl      # Training samples (80%)
    ├── itch_eval.jsonl       # Validation samples (10%)
    ├── itch_test.jsonl       # Test samples (10%)
    └── stats.json            # Dataset statistics
```

## Hardware Requirements

### Minimum (INT4 Quantization)
- **GPU:** NVIDIA RTX 4090 (24GB VRAM)
- **RAM:** 64GB system memory
- **Storage:** 500GB SSD (model checkpoints)

### Recommended (BF16)
- **GPU:** NVIDIA RTX 5090 (32GB VRAM)
- **RAM:** 128GB system memory
- **Storage:** 1TB NVMe SSD

### Training Time Estimates (RTX 5090)

| Configuration | Batch Size | Time/Epoch | Total (5 epochs) |
|--------------|------------|------------|------------------|
| BF16 | 2 | 45 min | ~4 hours |
| INT4 | 4 | 30 min | ~2.5 hours |
| INT4 + FA2 | 8 | 20 min | ~1.5 hours |

## Training Data Format

See [DATA_FORMAT.md](DATA_FORMAT.md) for complete specification.

### Summary

**Input Features (200 per snapshot):**
- Basic BBO (4): bid_price, bid_size, ask_price, ask_size
- Spread metrics (4): spread, mid_price, microprice
- Order book imbalance (8): imbalance at 4 price levels
- Price momentum (12): returns at 6 lookback windows
- Volume metrics (8): volume, VWAP, depth
- Volatility (8): realized and Parkinson vol
- Order flow (16): aggressor ratio, toxicity, pressure
- Cross-symbol (140): correlations, relative strength

**Labels:**
- 0 = DOWN (price drops > 5 bps in 500ms)
- 1 = NEUTRAL (price stays within ±5 bps)
- 2 = UP (price rises > 5 bps in 500ms)

## Model Architecture

### Llama 4 Scout Specifications

| Parameter | Value |
|-----------|-------|
| Total Parameters | 109B |
| Active Parameters | 17B (MoE) |
| Experts | 16 |
| Context Window | 10M tokens |
| Training Data | 40T tokens |
| Knowledge Cutoff | August 2024 |

### LoRA Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Rank (r) | 32 | Balance of capacity/efficiency |
| Alpha | 64 | 2× rank (standard) |
| Dropout | 0.05 | Regularization |
| Target Modules | q,k,v,o,gate,up,down | Attention + MoE experts |

### Training Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Learning Rate | 2e-5 | Conservative for fine-tuning |
| Weight Decay | 0.01 | Regularization |
| Warmup | 10% | Gradual LR increase |
| Scheduler | Cosine | Smooth decay |
| Gradient Checkpointing | Yes | Save VRAM |
| Mixed Precision | BF16 | RTX 5090 native |

## Performance Expectations

### Training Metrics (Target)

| Metric | Target | Notes |
|--------|--------|-------|
| Train Loss | < 0.8 | Cross-entropy |
| Eval Accuracy | > 45% | 3-class (33% baseline) |
| Eval F1 | > 0.40 | Macro-averaged |

### Inference Performance (RTX 5090)

| Quantization | Latency (batch=1) | Throughput |
|--------------|-------------------|------------|
| BF16 | 15-25ms | 40-65 samples/sec |
| INT4 | 8-15ms | 65-125 samples/sec |
| INT4 + TensorRT | 5-10ms | 100-200 samples/sec |

## Integration with Trading System

### C++ Inference (ONNX Runtime)

```cpp
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>

class OrderFlowPredictor {
public:
    OrderFlowPredictor(const std::string& model_path) {
        env_ = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "OrderFlowPredictor");
        
        Ort::SessionOptions options;
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        options.AppendExecutionProvider_CUDA(OrtCUDAProviderOptions{});
        
        session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), options);
    }
    
    int predict(const std::vector<float>& features) {
        // features: 100 snapshots × 200 features = 20,000 floats
        // ... inference logic ...
        return predicted_label;  // 0=DOWN, 1=NEUTRAL, 2=UP
    }
    
private:
    Ort::Env env_;
    std::unique_ptr<Ort::Session> session_;
};
```

### Integration with FPGA Pipeline

**Note:** The production system uses XGBoost (84% accuracy, ~100μs inference) instead of LLaMA (70% accuracy, 5-15ms inference). This documentation remains for reference on the LLaMA training workflow.

**Current Production Architecture (XGBoost):**
```
FPGA Order Book (Project 23)
    ↓ BBO packets (PCIe C2H DMA)
Project 24 (Order Gateway)
    ↓ Feature extraction + XGBoost GPU inference (~100μs)
Project 25 (Market Maker)
    ↓ Quote adjustment based on prediction
Project 26 (Order Execution)
```

**Alternative LLaMA Architecture (Higher latency):**
```
FPGA Order Book (Project 23)
    ↓ BBO packets (PCIe C2H DMA)
Project 24 (Order Gateway)
    ↓ Feature extraction + LLaMA inference (5-15ms)
Project 25 (Market Maker)
    ↓ Quote adjustment based on prediction
Project 26 (Order Execution)
```

## Troubleshooting

### CUDA Out of Memory

```bash
# Solution 1: Enable 4-bit quantization
python train_llama4_scout.py --use-4bit --batch-size 2

# Solution 2: Reduce sequence length
# Edit train_llama4_scout.py: max_seq_length = 2048

# Solution 3: Gradient checkpointing (already enabled by default)
```

### Slow Training

```bash
# Ensure Flash Attention is installed
pip install flash-attn --no-build-isolation

# Enable TF32 (already enabled by default)
# Check CUDA version: nvcc --version (need 12.4+)
```

### Model Not Fitting on GPU

The Llama 4 Scout model (109B total params) requires:
- BF16: ~25GB VRAM (fits on RTX 5090)
- INT4: ~15GB VRAM (fits on RTX 4090)

If you're seeing OOM errors, try:
1. Use INT4 quantization: `--use-4bit`
2. Reduce batch size: `--batch-size 1`
3. Reduce sequence length in config

## License

This project is for educational and research purposes. 

- **Llama 4:** Subject to [Llama 4 Community License](https://llama.meta.com/llama4/license/)
- **NASDAQ ITCH:** Subject to NASDAQ data licensing agreements

## Acknowledgments

- Meta AI for Llama 4 Scout model
- Hugging Face for transformers library
- NASDAQ for ITCH 5.0 specification

## Contact

**Author:** Adilson Dias  
**Repository:** [fpga-trading-systems](https://github.com/adilsondias-engineer/fpga-trading-systems)  
**LinkedIn:** [adilsondias](https://www.linkedin.com/in/adilsondias/)
