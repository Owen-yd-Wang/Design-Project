# Tiny CNN for CIFAR-10 — Basys 3 (XC7A35T) Target

A tiny ~48K-parameter CNN that classifies CIFAR-10 images and **fits on the Basys 3 FPGA** (225 KB BRAM, 90 DSPs).

## What's in here

| File | What it does |
|------|-------------|
| `tiny_cnn_cifar10.py` | Trains the CNN, evaluates it, exports to ONNX (float32 + int8), prints resource estimation |
| `basys3_sim.py` | Simulates running the trained model on the Basys 3 FPGA with int8 integer-only arithmetic |

## Quick Start

```bash
# Install dependencies
pip3 install torch torchvision onnxruntime onnxscript scikit-learn

# Step 1: Train the model and export to ONNX
python3 tiny_cnn_cifar10.py

# Step 2: Run the FPGA behavioral simulation
python3 basys3_sim.py
```

Step 1 takes ~10 minutes (training 30 epochs). Step 2 takes ~1 minute.

## Results Summary

### Model Architecture
```
Conv2d(3→16, 3x3)  → ReLU → MaxPool2x2     16x16x16
Conv2d(16→32, 3x3) → ReLU → MaxPool2x2     32x8x8
Conv2d(32→32, 3x3) → ReLU → MaxPool2x2     32x4x4
FC(512→64)          → ReLU
FC(64→10)

Total parameters: 47,818
Int8 model size:   ~47 KB
```

### Accuracy
| Model | Macro F1 | Accuracy |
|-------|----------|----------|
| Float32 (PyTorch) | 0.7400 | 74.2% |
| Int8 (FPGA sim) | 0.7395 | 74.1% |

### FPGA Resource Usage (Basys 3 / XC7A35T)
| Resource | Used | Available | Utilization |
|----------|------|-----------|-------------|
| BRAM | 66.1 KB | 225 KB | 29.4% |
| DSP48E1 | 90 (time-multiplexed) | 90 | Reused per layer |
| Latency | 0.25 ms per image | — | — |
| Throughput | 4,008 images/sec | — | @ 100 MHz |

**Verdict: PASS** — Model fits on Basys 3 with 70% BRAM headroom.

### Bottleneck
Conv2 (16→32, 3x3) takes 52.5% of total inference time due to having the most multiply-accumulate operations (1.18M MACs).

## What the FPGA simulation does

`basys3_sim.py` doesn't just estimate — it actually **runs every test image through integer-only math**, the same way the FPGA's DSP48E1 slices would compute it:

- All weights quantized to int8 (-128 to 127)
- All activations quantized to int8 between layers
- Multiply-accumulate uses int32 accumulators (same as DSP48E1)
- ReLU = clamp negatives to zero (free on FPGA)
- MaxPool = integer comparisons (no DSP needed)
- Clock cycles counted per layer based on MAC count / DSP allocation

## Generated Files (not committed)

After running the scripts, these files are created locally:
- `tiny_cnn_cifar10.pth` — PyTorch model checkpoint
- `tiny_cnn_cifar10.onnx` — Float32 ONNX model
- `tiny_cnn_cifar10_int8.onnx` — Int8 quantized ONNX model
- `data/` — CIFAR-10 dataset (auto-downloaded)
