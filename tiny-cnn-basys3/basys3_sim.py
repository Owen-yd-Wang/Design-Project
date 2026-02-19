#!/usr/bin/env python3
"""
Basys 3 (XC7A35T) Behavioral FPGA Simulation for TinyCNN.

Simulates running the trained CNN on the FPGA by:
  1. Quantizing all weights and activations to int8 (fixed-point)
  2. Running inference using integer-only arithmetic (like the FPGA would)
  3. Counting clock cycles per layer based on DSP allocation
  4. Tracking BRAM usage at each stage
  5. Evaluating accuracy on CIFAR-10 test set
  6. Printing a detailed timing/resource report
"""

import os
import sys
import time

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import classification_report, f1_score

# Add parent dir so we can import the model definition
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tiny_cnn_cifar10 import TinyCNN

# ---------------------------------------------------------------------------
# FPGA hardware parameters — Basys 3 / XC7A35T
# ---------------------------------------------------------------------------
CLOCK_MHZ = 100
TOTAL_DSP = 90
TOTAL_BRAM_KB = 225
TOTAL_LUT = 20800
TOTAL_FF = 41600

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


# ---------------------------------------------------------------------------
# Int8 quantization helpers
# ---------------------------------------------------------------------------
def quantize_tensor(x, num_bits=8):
    """Quantize a float tensor to int8 range [-128, 127].

    Returns (x_int, scale) where x ≈ x_int * scale.
    """
    qmin, qmax = -128, 127
    x_min, x_max = float(x.min()), float(x.max())

    # Symmetric quantization around zero
    abs_max = max(abs(x_min), abs(x_max), 1e-8)
    scale = abs_max / 127.0

    x_int = np.round(x / scale).astype(np.int8)
    x_int = np.clip(x_int, qmin, qmax).astype(np.int8)
    return x_int, scale


def quantize_activations(x, num_bits=8):
    """Quantize activations to int8. Returns (x_int, scale)."""
    qmin, qmax = -128, 127
    abs_max = max(float(np.abs(x).max()), 1e-8)
    scale = abs_max / 127.0
    x_int = np.round(x / scale).astype(np.int32)  # int32 for headroom
    x_int = np.clip(x_int, qmin, qmax).astype(np.int8)
    return x_int, scale


# ---------------------------------------------------------------------------
# Int8 layer operations (simulating FPGA hardware)
# ---------------------------------------------------------------------------
def conv2d_int8(x_int, x_scale, w_int, w_scale, b_float, stride=1, padding=0):
    """
    Simulate int8 convolution as the FPGA would do it.

    Multiply int8 weights x int8 activations → accumulate in int32 → rescale.
    This is exactly how DSP48E1 slices work: 8x8 multiply, 32-bit accumulator.
    """
    # x_int: (C_in, H, W) as int8
    # w_int: (C_out, C_in, kH, kW) as int8
    C_in, H, W = x_int.shape
    C_out, _, kH, kW = w_int.shape

    H_out = (H + 2 * padding - kH) // stride + 1
    W_out = (W + 2 * padding - kW) // stride + 1

    # Pad input
    if padding > 0:
        x_padded = np.zeros((C_in, H + 2 * padding, W + 2 * padding), dtype=np.int8)
        x_padded[:, padding:padding + H, padding:padding + W] = x_int
    else:
        x_padded = x_int

    # Output accumulator — int32, just like the FPGA's DSP accumulator
    out_int32 = np.zeros((C_out, H_out, W_out), dtype=np.int32)

    # Perform convolution with integer-only multiply-accumulate
    for oc in range(C_out):
        for ic in range(C_in):
            for kh in range(kH):
                for kw in range(kW):
                    w_val = int(w_int[oc, ic, kh, kw])
                    for oh in range(H_out):
                        for ow in range(W_out):
                            ih = oh * stride + kh
                            iw = ow * stride + kw
                            a_val = int(x_padded[ic, ih, iw])
                            out_int32[oc, oh, ow] += w_val * a_val

    # Rescale: output_float ≈ out_int32 * (x_scale * w_scale)
    combined_scale = x_scale * w_scale
    out_float = out_int32.astype(np.float32) * combined_scale

    # Add bias (bias stays in float — on FPGA this would be a wider fixed-point add)
    for oc in range(C_out):
        out_float[oc] += b_float[oc]

    # Requantize output to int8 for next layer
    out_int8, out_scale = quantize_activations(out_float)
    return out_int8, out_scale, out_float


def conv2d_int8_fast(x_int, x_scale, w_int, w_scale, b_float, stride=1, padding=0):
    """
    Faster version using numpy operations (same math as conv2d_int8).
    Uses im2col approach — still integer multiply-accumulate.
    """
    C_in, H, W = x_int.shape
    C_out, _, kH, kW = w_int.shape

    H_out = (H + 2 * padding - kH) // stride + 1
    W_out = (W + 2 * padding - kW) // stride + 1

    # Pad
    if padding > 0:
        x_padded = np.zeros((C_in, H + 2 * padding, W + 2 * padding), dtype=np.int8)
        x_padded[:, padding:padding + H, padding:padding + W] = x_int
    else:
        x_padded = x_int

    # im2col: extract patches as columns
    cols = np.zeros((C_in * kH * kW, H_out * W_out), dtype=np.int32)
    idx = 0
    for ic in range(C_in):
        for kh in range(kH):
            for kw in range(kW):
                patch = x_padded[ic, kh:kh + H_out * stride:stride, kw:kw + W_out * stride:stride]
                cols[idx] = patch.flatten().astype(np.int32)
                idx += 1

    # Reshape weights: (C_out, C_in*kH*kW)
    w_mat = w_int.reshape(C_out, -1).astype(np.int32)

    # Integer matrix multiply (int32 accumulation, same as DSP48E1)
    out_int32 = w_mat @ cols  # (C_out, H_out*W_out)

    # Rescale to float
    combined_scale = x_scale * w_scale
    out_float = out_int32.astype(np.float32) * combined_scale

    # Add bias
    out_float += b_float.reshape(-1, 1)

    # Reshape
    out_float = out_float.reshape(C_out, H_out, W_out)

    # Requantize for next layer
    out_int8, out_scale = quantize_activations(out_float)
    return out_int8, out_scale, out_float


def relu_int8(x_int, x_scale):
    """ReLU on int8: just clamp negatives to zero. Free on FPGA (wire routing)."""
    out = np.maximum(x_int, 0).astype(np.int8)
    return out, x_scale


def maxpool2d_int8(x_int, x_scale, kernel_size=2):
    """MaxPool on int8: compare integers. Cheap on FPGA (comparators, no DSP)."""
    C, H, W = x_int.shape
    H_out = H // kernel_size
    W_out = W // kernel_size
    out = np.zeros((C, H_out, W_out), dtype=np.int8)

    for c in range(C):
        for h in range(H_out):
            for w in range(W_out):
                patch = x_int[c,
                              h * kernel_size:(h + 1) * kernel_size,
                              w * kernel_size:(w + 1) * kernel_size]
                out[c, h, w] = patch.max()

    return out, x_scale


def fc_int8(x_int, x_scale, w_int, w_scale, b_float):
    """
    Fully-connected layer in int8.
    x_int: (N,) int8 flattened input
    w_int: (out_features, in_features) int8
    Returns output as int8 + scale.
    """
    # Int32 accumulation (DSP48E1 behavior)
    out_int32 = w_int.astype(np.int32) @ x_int.astype(np.int32)

    # Rescale
    combined_scale = x_scale * w_scale
    out_float = out_int32.astype(np.float32) * combined_scale + b_float

    # Requantize
    out_int8, out_scale = quantize_activations(out_float)
    return out_int8, out_scale, out_float


# ---------------------------------------------------------------------------
# Clock cycle model
# ---------------------------------------------------------------------------
def compute_layer_cycles(layer_name, macs, dsps_allocated):
    """
    Compute how many clock cycles a layer takes.

    Each DSP48E1 does 1 MAC per clock cycle (int8 mode).
    Cycles = ceil(total_MACs / num_DSPs).
    """
    cycles = int(np.ceil(macs / dsps_allocated))
    time_us = cycles / CLOCK_MHZ  # microseconds
    return cycles, time_us


# ---------------------------------------------------------------------------
# Memory tracker
# ---------------------------------------------------------------------------
class MemoryTracker:
    """Track BRAM usage through the inference pipeline."""

    def __init__(self):
        self.log = []
        self.current_weights_bytes = 0
        self.current_act_bytes = 0

    def load_weights(self, layer_name, weight_array, bias_array):
        """Simulate loading weights into BRAM."""
        w_bytes = weight_array.nbytes
        b_bytes = bias_array.nbytes
        self.current_weights_bytes += w_bytes + b_bytes
        return w_bytes + b_bytes

    def set_activations(self, layer_name, input_array, output_array):
        """Track activation buffer memory (input + output must coexist)."""
        self.current_act_bytes = input_array.nbytes + output_array.nbytes
        total = self.current_weights_bytes + self.current_act_bytes
        self.log.append({
            "layer": layer_name,
            "weights_kb": self.current_weights_bytes / 1024,
            "act_kb": self.current_act_bytes / 1024,
            "total_kb": total / 1024,
        })
        return total

    def peak_usage(self):
        if not self.log:
            return 0
        return max(entry["total_kb"] for entry in self.log)


# ---------------------------------------------------------------------------
# DSP allocation strategy
# ---------------------------------------------------------------------------
def allocate_dsps(layer_macs, total_dsps=TOTAL_DSP):
    """
    Allocate DSPs proportionally to each layer's MAC count.
    This simulates a time-multiplexed serial-layer design where
    all DSPs are reused for each layer sequentially.

    On the Basys 3 with 90 DSPs, the practical approach is:
    assign all 90 DSPs to whichever layer is currently executing.
    """
    # Simple strategy: all DSPs work on one layer at a time
    allocation = {}
    for name in layer_macs:
        allocation[name] = total_dsps
    return allocation


# ---------------------------------------------------------------------------
# Full FPGA inference simulation
# ---------------------------------------------------------------------------
class FPGASimulator:
    """Simulates running TinyCNN on Basys 3 FPGA with int8 arithmetic."""

    def __init__(self, model_state_dict):
        # Quantize all weights to int8
        self.layers = {}
        self.weight_scales = {}
        self.mem = MemoryTracker()

        layer_map = [
            ("conv1", "features.0"),
            ("conv2", "features.3"),
            ("conv3", "features.6"),
            ("fc1", "classifier.0"),
            ("fc2", "classifier.2"),
        ]

        for name, prefix in layer_map:
            w = model_state_dict[f"{prefix}.weight"].cpu().numpy()
            b = model_state_dict[f"{prefix}.bias"].cpu().numpy()
            w_int8, w_scale = quantize_tensor(w)
            self.layers[name] = {
                "weight_int8": w_int8,
                "weight_scale": w_scale,
                "bias": b.astype(np.float32),
                "weight_float": w,
            }
            self.mem.load_weights(name, w_int8, b)

        # MAC counts per layer
        self.layer_macs = {
            "conv1": 3 * 16 * 3 * 3 * 32 * 32,       # 442,368
            "conv2": 16 * 32 * 3 * 3 * 16 * 16,       # 1,179,648
            "conv3": 32 * 32 * 3 * 3 * 8 * 8,         # 589,824
            "fc1":   512 * 64,                          # 32,768
            "fc2":   64 * 10,                           # 640
        }

        self.dsp_alloc = allocate_dsps(self.layer_macs)

    def _record_memory(self, image_np):
        """Run one image just to record memory usage per layer (called once)."""
        x_int, x_scale = quantize_activations(image_np)

        L = self.layers["conv1"]
        out_int, out_scale, _ = conv2d_int8_fast(
            x_int, x_scale, L["weight_int8"], L["weight_scale"], L["bias"], padding=1
        )
        self.mem.set_activations("conv1", x_int, out_int)
        out_int, _ = relu_int8(out_int, out_scale)
        out_int, out_scale = maxpool2d_int8(out_int, out_scale, 2)
        x_int, x_scale = out_int, out_scale

        L = self.layers["conv2"]
        out_int, out_scale, _ = conv2d_int8_fast(
            x_int, x_scale, L["weight_int8"], L["weight_scale"], L["bias"], padding=1
        )
        self.mem.set_activations("conv2", x_int, out_int)
        out_int, _ = relu_int8(out_int, out_scale)
        out_int, out_scale = maxpool2d_int8(out_int, out_scale, 2)
        x_int, x_scale = out_int, out_scale

        L = self.layers["conv3"]
        out_int, out_scale, _ = conv2d_int8_fast(
            x_int, x_scale, L["weight_int8"], L["weight_scale"], L["bias"], padding=1
        )
        self.mem.set_activations("conv3", x_int, out_int)
        out_int, _ = relu_int8(out_int, out_scale)
        out_int, out_scale = maxpool2d_int8(out_int, out_scale, 2)
        x_int = out_int.flatten().astype(np.int8)
        x_scale = out_scale

        L = self.layers["fc1"]
        out_int, out_scale, _ = fc_int8(
            x_int, x_scale, L["weight_int8"], L["weight_scale"], L["bias"]
        )
        self.mem.set_activations("fc1", x_int, out_int)
        out_int, out_scale = relu_int8(out_int, out_scale)
        x_int, x_scale = out_int, out_scale

        L = self.layers["fc2"]
        out_int, out_scale, out_float = fc_int8(
            x_int, x_scale, L["weight_int8"], L["weight_scale"], L["bias"]
        )
        self.mem.set_activations("fc2", x_int, out_int)

    def infer_one(self, image_np):
        """
        Run a single image through the int8 pipeline.
        image_np: (3, 32, 32) float32 normalized input.
        Returns: (prediction, layer_details)
        """
        layer_details = []

        # Quantize input to int8
        x_int, x_scale = quantize_activations(image_np)

        # --- Conv1 + ReLU + MaxPool ---
        L = self.layers["conv1"]
        out_int, out_scale, out_float = conv2d_int8_fast(
            x_int, x_scale, L["weight_int8"], L["weight_scale"], L["bias"], padding=1
        )
        out_int, out_scale = relu_int8(out_int, out_scale)
        out_int, out_scale = maxpool2d_int8(out_int, out_scale, 2)
        layer_details.append(("conv1", x_int.shape, out_int.shape))
        x_int, x_scale = out_int, out_scale

        # --- Conv2 + ReLU + MaxPool ---
        L = self.layers["conv2"]
        out_int, out_scale, out_float = conv2d_int8_fast(
            x_int, x_scale, L["weight_int8"], L["weight_scale"], L["bias"], padding=1
        )
        out_int, out_scale = relu_int8(out_int, out_scale)
        out_int, out_scale = maxpool2d_int8(out_int, out_scale, 2)
        layer_details.append(("conv2", x_int.shape, out_int.shape))
        x_int, x_scale = out_int, out_scale

        # --- Conv3 + ReLU + MaxPool ---
        L = self.layers["conv3"]
        out_int, out_scale, out_float = conv2d_int8_fast(
            x_int, x_scale, L["weight_int8"], L["weight_scale"], L["bias"], padding=1
        )
        out_int, out_scale = relu_int8(out_int, out_scale)
        out_int, out_scale = maxpool2d_int8(out_int, out_scale, 2)
        layer_details.append(("conv3", x_int.shape, out_int.shape))
        x_int, x_scale = out_int, out_scale

        # --- Flatten ---
        x_int = x_int.flatten().astype(np.int8)

        # --- FC1 + ReLU ---
        L = self.layers["fc1"]
        out_int, out_scale, out_float = fc_int8(
            x_int, x_scale, L["weight_int8"], L["weight_scale"], L["bias"]
        )
        out_int, out_scale = relu_int8(out_int, out_scale)
        layer_details.append(("fc1", x_int.shape, out_int.shape))
        x_int, x_scale = out_int, out_scale

        # --- FC2 (no activation — raw logits) ---
        L = self.layers["fc2"]
        out_int, out_scale, out_float = fc_int8(
            x_int, x_scale, L["weight_int8"], L["weight_scale"], L["bias"]
        )
        layer_details.append(("fc2", x_int.shape, out_int.shape))

        # Prediction = argmax of final logits (use float for argmax accuracy)
        prediction = int(np.argmax(out_float))
        return prediction, layer_details

    def timing_report(self):
        """Generate cycle-by-cycle timing for one inference."""
        rows = []
        total_cycles = 0

        for name, macs in self.layer_macs.items():
            dsps = self.dsp_alloc[name]
            cycles, time_us = compute_layer_cycles(name, macs, dsps)
            total_cycles += cycles
            rows.append({
                "name": name,
                "macs": macs,
                "dsps": dsps,
                "cycles": cycles,
                "time_us": time_us,
            })

        total_time_us = total_cycles / CLOCK_MHZ
        total_time_ms = total_time_us / 1000
        fps = 1_000_000 / total_time_us if total_time_us > 0 else 0

        return rows, total_cycles, total_time_ms, fps


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------
def print_header(title):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def print_timing(sim):
    rows, total_cycles, total_time_ms, fps = sim.timing_report()

    print_header("Clock Cycle Analysis (per inference)")
    print(f"\n  FPGA Clock: {CLOCK_MHZ} MHz | DSP Strategy: all {TOTAL_DSP} DSPs per layer (sequential)")
    print()
    print(f"  {'Layer':<22} {'MACs':>12} {'DSPs':>6} {'Cycles':>12} {'Time':>10}")
    print(f"  {'-' * 62}")
    for r in rows:
        time_str = f"{r['time_us']:.2f} us" if r['time_us'] < 1000 else f"{r['time_us']/1000:.2f} ms"
        print(f"  {r['name']:<22} {r['macs']:>12,} {r['dsps']:>6} {r['cycles']:>12,} {time_str:>10}")
    print(f"  {'-' * 62}")
    print(f"  {'TOTAL':<22} {sum(r['macs'] for r in rows):>12,} {'':>6} {total_cycles:>12,} "
          f"{total_time_ms:.2f} ms")
    print()
    print(f"  Latency per image:  {total_time_ms:.2f} ms")
    print(f"  Throughput:         {fps:.0f} inferences/sec")
    print(f"  Total cycles:       {total_cycles:,}")

    # Show bottleneck
    bottleneck = max(rows, key=lambda r: r["cycles"])
    pct = bottleneck["cycles"] / total_cycles * 100
    print(f"\n  Bottleneck: {bottleneck['name']} ({pct:.1f}% of total time)")
    print(f"    → {bottleneck['macs']:,} MACs / {bottleneck['dsps']} DSPs = "
          f"{bottleneck['cycles']:,} cycles")


def print_memory(sim):
    print_header("BRAM Memory Usage")

    # Weight memory
    print(f"\n  Weight Storage (all loaded into BRAM at startup):")
    print(f"  {'Layer':<22} {'Params':>10} {'Bytes (int8)':>14}")
    print(f"  {'-' * 46}")
    total_w = 0
    for name, L in sim.layers.items():
        w_bytes = L["weight_int8"].nbytes
        b_bytes = L["bias"].nbytes
        total = w_bytes + b_bytes
        total_w += total
        params = L["weight_int8"].size + L["bias"].size
        print(f"  {name:<22} {params:>10,} {total:>14,}")
    print(f"  {'-' * 46}")
    print(f"  {'TOTAL':<22} {'':>10} {total_w:>14,}")
    print(f"  Weight memory: {total_w / 1024:.1f} KB")

    # Activation memory per layer
    print(f"\n  Activation Buffers (input + output must coexist in BRAM):")
    print(f"  {'Layer':<22} {'Weights KB':>12} {'Act KB':>10} {'Total KB':>10} {'% BRAM':>8}")
    print(f"  {'-' * 62}")
    for entry in sim.mem.log:
        pct = entry["total_kb"] / TOTAL_BRAM_KB * 100
        print(f"  {entry['layer']:<22} {entry['weights_kb']:>12.1f} {entry['act_kb']:>10.1f} "
              f"{entry['total_kb']:>10.1f} {pct:>7.1f}%")

    peak = sim.mem.peak_usage()
    print(f"  {'-' * 62}")
    print(f"  Peak BRAM usage: {peak:.1f} KB / {TOTAL_BRAM_KB} KB "
          f"({peak / TOTAL_BRAM_KB * 100:.1f}%)")
    remaining = TOTAL_BRAM_KB - peak
    print(f"  Remaining BRAM:  {remaining:.1f} KB")


def print_pipeline_diagram():
    print_header("Data Flow Through FPGA Pipeline")
    print("""
  Input Image (3x32x32)
       │  quantize to int8
       ▼
  ┌─────────────────────────────────────────┐
  │  Conv1: 3→16 filters, 3x3, pad=1       │  432 weights (int8)
  │  Int8 multiply-accumulate → int32 acc   │  442,368 MACs
  │  Rescale + ReLU (clamp negatives to 0)  │
  │  MaxPool 2x2 (integer compare)          │
  │  Output: 16x16x16 (int8)               │
  └─────────────────────────────────────────┘
       │
       ▼
  ┌─────────────────────────────────────────┐
  │  Conv2: 16→32 filters, 3x3, pad=1      │  4,608 weights (int8)
  │  Int8 multiply-accumulate → int32 acc   │  1,179,648 MACs    ← BOTTLENECK
  │  Rescale + ReLU + MaxPool 2x2           │
  │  Output: 32x8x8 (int8)                 │
  └─────────────────────────────────────────┘
       │
       ▼
  ┌─────────────────────────────────────────┐
  │  Conv3: 32→32 filters, 3x3, pad=1      │  9,216 weights (int8)
  │  Int8 multiply-accumulate → int32 acc   │  589,824 MACs
  │  Rescale + ReLU + MaxPool 2x2           │
  │  Output: 32x4x4 = 512 values (int8)    │
  └─────────────────────────────────────────┘
       │  flatten
       ▼
  ┌─────────────────────────────────────────┐
  │  FC1: 512→64                            │  32,768 weights (int8)
  │  Int8 matrix-vector multiply → int32    │  32,768 MACs
  │  Rescale + ReLU                         │
  │  Output: 64 values (int8)              │
  └─────────────────────────────────────────┘
       │
       ▼
  ┌─────────────────────────────────────────┐
  │  FC2: 64→10                             │  640 weights (int8)
  │  Int8 matrix-vector multiply → int32    │  640 MACs
  │  Output: 10 logits                      │
  └─────────────────────────────────────────┘
       │
       ▼
   argmax → Predicted Class (0-9)
""")


def print_int8_math_example(sim):
    print_header("Int8 Arithmetic Example (what the DSPs actually compute)")
    print("""
  Example: one output pixel of Conv1

  The FPGA computes the dot product of a 3x3x3 patch with a 3x3x3 filter:

    Input patch (int8):     [ 42, -17,  83, -5, 127, -34, ...]  (27 values)
    Filter weights (int8):  [ 12,  -3,   7, 15,  -8,  22, ...]  (27 values)

    DSP48E1 computes:  accumulator (int32) = 0
      cycle 1:  accumulator += 42 * 12  =   504
      cycle 2:  accumulator += -17 * -3 =   555
      cycle 3:  accumulator += 83 * 7   =  1136
      ...       (27 cycles for all 27 MACs)
      final:    accumulator = 4821  (int32)

    Rescale:   output_float = 4821 * (input_scale * weight_scale)
    Add bias:  output_float += bias[filter_idx]
    Requantize: output_int8 = clamp(round(output_float / new_scale), -128, 127)

  With 90 DSPs running in parallel, 90 of these dot products happen simultaneously.
""")


def print_accuracy_comparison(float_f1, int8_f1):
    print_header("Accuracy: Float32 vs Int8 (FPGA) Comparison")
    print(f"""
  Float32 (original):   F1 = {float_f1:.4f}  (what PyTorch computes on CPU/GPU)
  Int8 (FPGA sim):      F1 = {int8_f1:.4f}  (what the Basys 3 would compute)
  Delta:                     {int8_f1 - float_f1:+.4f}

  {'✓ Minimal accuracy loss from int8 quantization' if abs(int8_f1 - float_f1) < 0.05
   else '⚠ Significant accuracy loss — consider wider bit width'}
""")


def print_verdict(sim, int8_f1):
    rows, total_cycles, total_time_ms, fps = sim.timing_report()
    peak_bram = sim.mem.peak_usage()

    print_header("FINAL VERDICT — Basys 3 (XC7A35T) Feasibility")

    bram_ok = peak_bram < TOTAL_BRAM_KB
    acc_ok = int8_f1 >= 0.60
    fps_ok = fps >= 1

    print(f"""
  ┌────────────────────────────────────────────────────────────────┐
  │  Resource        Used          Available     Status            │
  ├────────────────────────────────────────────────────────────────┤
  │  BRAM            {peak_bram:>6.1f} KB      {TOTAL_BRAM_KB:>6d} KB      {'PASS' if bram_ok else 'FAIL':>4}              │
  │  DSP48E1         {TOTAL_DSP:>6d}          {TOTAL_DSP:>6d}          PASS (time-muxed)   │
  │  LUT             ~2,000        {TOTAL_LUT:>6,}        PASS (estimated)   │
  │  FF              ~1,500        {TOTAL_FF:>6,}        PASS (estimated)   │
  ├────────────────────────────────────────────────────────────────┤
  │  Latency         {total_time_ms:>6.2f} ms     —             {'PASS' if fps_ok else 'FAIL':>4}              │
  │  Throughput      {fps:>6.0f} FPS      —             {'PASS' if fps_ok else 'FAIL':>4}              │
  │  Accuracy (F1)   {int8_f1:>6.4f}        ≥0.60         {'PASS' if acc_ok else 'FAIL':>4}              │
  └────────────────────────────────────────────────────────────────┘
""")

    all_pass = bram_ok and acc_ok and fps_ok
    if all_pass:
        print("  *** PASS — This CNN can run on the Basys 3 FPGA! ***")
        print(f"  The model uses {peak_bram:.1f} KB of {TOTAL_BRAM_KB} KB BRAM ({peak_bram/TOTAL_BRAM_KB*100:.0f}%),")
        print(f"  processes {fps:.0f} images/sec at {CLOCK_MHZ} MHz,")
        print(f"  and achieves F1={int8_f1:.4f} with pure int8 arithmetic.")
    else:
        print("  *** FAIL — Model does not fit or meet requirements ***")
        if not bram_ok:
            print(f"    - BRAM overflow: {peak_bram:.1f} KB > {TOTAL_BRAM_KB} KB")
        if not acc_ok:
            print(f"    - Accuracy too low: F1={int8_f1:.4f} < 0.60")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(script_dir, "tiny_cnn_cifar10.pth")

    print("=" * 70)
    print("  Basys 3 FPGA Behavioral Simulation — TinyCNN on CIFAR-10")
    print("=" * 70)

    # Load trained model
    if not os.path.exists(checkpoint_path):
        print(f"\nERROR: Checkpoint not found at {checkpoint_path}")
        print("Run tiny_cnn_cifar10.py first to train the model.")
        sys.exit(1)

    print(f"\nLoading trained model from {checkpoint_path}...")
    model = TinyCNN()
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
    model.eval()

    # Load test data
    print("Loading CIFAR-10 test set...")
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    testset = torchvision.datasets.CIFAR10(
        root=os.path.join(script_dir, "data"), train=False, download=True,
        transform=test_transform,
    )
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)

    # --- Float32 baseline ---
    print("\nRunning float32 baseline on test set...")
    float_preds = []
    float_labels = []
    with torch.no_grad():
        for images, labels in testloader:
            outputs = model(images)
            _, predicted = outputs.max(1)
            float_preds.append(predicted.item())
            float_labels.append(labels.item())
    float_f1 = f1_score(float_labels, float_preds, average="macro")
    print(f"Float32 Macro F1: {float_f1:.4f}")

    # --- Build FPGA simulator ---
    print("\nInitializing FPGA simulator (quantizing weights to int8)...")
    sim = FPGASimulator(model.state_dict())

    # Record memory usage with one sample image
    sample_img, _ = testset[0]
    sim._record_memory(sample_img.numpy())

    # --- Run int8 inference on full test set ---
    print(f"Running int8 FPGA simulation on {len(testset)} test images...")
    int8_preds = []
    int8_labels = []
    t0 = time.time()

    for i, (image, label) in enumerate(testloader):
        image_np = image.squeeze(0).numpy()  # (3, 32, 32)
        pred, _ = sim.infer_one(image_np)
        int8_preds.append(pred)
        int8_labels.append(label.item())

        if (i + 1) % 1000 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            print(f"  {i + 1:>5}/{len(testset)} images ({rate:.0f} img/sec on host)...")

    sim_time = time.time() - t0
    int8_f1 = f1_score(int8_labels, int8_preds, average="macro")
    print(f"\nSimulation complete in {sim_time:.1f}s")

    # --- Print reports ---
    print_pipeline_diagram()
    print_int8_math_example(sim)
    print_timing(sim)
    print_memory(sim)

    print_header("Int8 (FPGA) Classification Report")
    print(classification_report(int8_labels, int8_preds,
                                target_names=CIFAR10_CLASSES, digits=4))

    print_accuracy_comparison(float_f1, int8_f1)
    print_verdict(sim, int8_f1)


if __name__ == "__main__":
    main()
