#!/usr/bin/env python3
"""
Generate real test vectors for RTL vs Python MAC unit comparison.

Loads actual int8 quantized weights from tiny_mems_int8/ .mem files
(exported from tiny_cnn_cifar10_int8.onnx) and a real CIFAR-10 test image,
then creates interleaved data/weight pair .mem files for Verilog $readmemh.

The MAC unit (mac_uint8_int32.sv) operates on uint8 inputs with uint32
accumulation, so all values are treated as unsigned bytes.

Output: real_test_vectors/ directory with .mem files + expected_results.csv
"""

import os
import struct
import csv
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
MEMS_DIR = os.path.join(PROJECT_DIR, "tiny_mems_int8")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "real_test_vectors")
CIFAR_DIR = os.path.join(PROJECT_DIR, "cifar-10-batches-py")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_mem_file(path):
    """Load a .mem hex file into a list of uint8 values."""
    values = []
    with open(path) as f:
        for line in f:
            token = line.strip()
            if token and not token.startswith("//"):
                values.append(int(token, 16) & 0xFF)
    return np.array(values, dtype=np.uint8)


def load_cifar10_image(index=0):
    """Load a single CIFAR-10 test image as uint8 [3, 32, 32]."""
    import pickle
    test_batch = os.path.join(CIFAR_DIR, "test_batch")
    if not os.path.exists(test_batch):
        # Try torchvision download as fallback
        try:
            import torchvision
            dataset = torchvision.datasets.CIFAR10(
                root=os.path.join(PROJECT_DIR, "data"), train=False, download=True
            )
            img = np.array(dataset[index][0])  # HWC uint8
            return img.transpose(2, 0, 1)      # CHW uint8
        except Exception:
            raise FileNotFoundError(
                f"CIFAR-10 test batch not found at {test_batch}"
            )
    with open(test_batch, "rb") as f:
        batch = pickle.load(f, encoding="bytes")
    data = batch[b"data"]  # (10000, 3072)
    img = data[index].reshape(3, 32, 32).astype(np.uint8)
    return img


def write_mem_file(path, values):
    """Write uint8 values as hex .mem file (one byte per line)."""
    with open(path, "w") as f:
        for v in values:
            f.write(f"{int(v) & 0xFF:02x}\n")


def unsigned_dot(data, weights):
    """Compute uint8 x uint8 dot product with uint32 accumulation."""
    acc = 0
    for d, w in zip(data, weights):
        acc += int(d & 0xFF) * int(w & 0xFF)
    return acc & 0xFFFFFFFF


def extract_conv_patch(image, row, col, in_channels, kernel_size=3, pad=1):
    """Extract a padded convolution patch as flattened uint8 array."""
    _, H, W = image.shape
    patch = []
    for c in range(in_channels):
        for kr in range(kernel_size):
            for kc in range(kernel_size):
                r = row - pad + kr
                c_idx = col - pad + kc
                if 0 <= r < H and 0 <= c_idx < W:
                    patch.append(int(image[c, r, c_idx]))
                else:
                    patch.append(0)  # zero-padding
    return np.array(patch, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load real CIFAR-10 test image (index 0 = cat)
    print("Loading CIFAR-10 test image (index 0)...")
    image = load_cifar10_image(0)
    print(f"  Image shape: {image.shape}, dtype: {image.dtype}")
    print(f"  Label: cat (class 3)")

    # Load Conv1 weights: shape [16, 3, 3, 3] = 432 values
    conv1_w_path = os.path.join(MEMS_DIR, "conv1.weight_quantized.mem")
    conv1_weights_flat = load_mem_file(conv1_w_path)
    print(f"  Conv1 weights: {len(conv1_weights_flat)} values")

    # Conv1 filter 0: first 27 values (3*3*3)
    conv1_f0 = conv1_weights_flat[:27]

    # Load Conv2 weights: shape [32, 16, 3, 3] = 4608 values
    conv2_w_path = os.path.join(MEMS_DIR, "conv2.weight_quantized.mem")
    conv2_weights_flat = load_mem_file(conv2_w_path)
    print(f"  Conv2 weights: {len(conv2_weights_flat)} values")

    # Conv2 filter 0: first 144 values (16*3*3)
    conv2_f0 = conv2_weights_flat[:144]

    # Load FC2 weights: shape [10, 64] = 640 values
    fc2_w_path = os.path.join(MEMS_DIR, "fc2.weight_quantized.mem")
    fc2_weights_flat = load_mem_file(fc2_w_path)
    print(f"  FC2 weights: {len(fc2_weights_flat)} values")

    # FC2 neuron 0: first 64 values
    fc2_n0 = fc2_weights_flat[:64]

    # -- Generate a synthetic Conv2 input activation (post-Conv1+ReLU+Pool) --
    # For bit-exact testing, we need deterministic activations.
    # Use the actual image patch values run through conv1 filter 0 as a proxy,
    # or just use part of the image as representative uint8 activations.
    # Here we use a portion of the original image as stand-in activations
    # for conv2 (16 channels x 3x3 patch = 144 values)
    conv2_act = image.flatten()[:144].astype(np.uint8)
    if len(conv2_act) < 144:
        conv2_act = np.pad(conv2_act, (0, 144 - len(conv2_act)))

    # FC2 input: 64 representative activation values
    fc2_input = image.flatten()[:64].astype(np.uint8)

    # -----------------------------------------------------------------------
    # Test 1: Conv1 filter 0 at pixel (5,5) — interior, 27 MACs
    # -----------------------------------------------------------------------
    conv1_act_5x5 = extract_conv_patch(image, row=5, col=5, in_channels=3)
    expected_1 = unsigned_dot(conv1_act_5x5, conv1_f0)
    print(f"\nTest 1: Conv1 f0 @ (5,5) — 27 MACs")
    print(f"  Expected (unsigned): {expected_1}")

    # -----------------------------------------------------------------------
    # Test 2: Conv1 filter 0 at pixel (0,0) — corner with zero-padding
    # -----------------------------------------------------------------------
    conv1_act_0x0 = extract_conv_patch(image, row=0, col=0, in_channels=3)
    expected_2 = unsigned_dot(conv1_act_0x0, conv1_f0)
    print(f"\nTest 2: Conv1 f0 @ (0,0) — 27 MACs (corner)")
    print(f"  Expected (unsigned): {expected_2}")

    # -----------------------------------------------------------------------
    # Test 3: Conv2 filter 0 at pixel (4,4) — 144 MACs
    # -----------------------------------------------------------------------
    expected_3 = unsigned_dot(conv2_act, conv2_f0)
    print(f"\nTest 3: Conv2 f0 @ (4,4) — 144 MACs")
    print(f"  Expected (unsigned): {expected_3}")

    # -----------------------------------------------------------------------
    # Test 4: FC2 neuron 0 — 64 MACs
    # -----------------------------------------------------------------------
    expected_4 = unsigned_dot(fc2_input, fc2_n0)
    print(f"\nTest 4: FC2 neuron 0 — 64 MACs")
    print(f"  Expected (unsigned): {expected_4}")

    # -----------------------------------------------------------------------
    # Write .mem files (individual + interleaved pairs)
    # -----------------------------------------------------------------------
    print("\nWriting .mem files...")

    # Individual activation/weight files
    write_mem_file(os.path.join(OUTPUT_DIR, "conv1_act_5x5.mem"), conv1_act_5x5)
    write_mem_file(os.path.join(OUTPUT_DIR, "conv1_weight_f0.mem"), conv1_f0)
    write_mem_file(os.path.join(OUTPUT_DIR, "conv1_act_0x0.mem"), conv1_act_0x0)
    write_mem_file(os.path.join(OUTPUT_DIR, "conv2_act_4x4.mem"), conv2_act)
    write_mem_file(os.path.join(OUTPUT_DIR, "conv2_weight_f0.mem"), conv2_f0)
    write_mem_file(os.path.join(OUTPUT_DIR, "fc2_input.mem"), fc2_input)
    write_mem_file(os.path.join(OUTPUT_DIR, "fc2_weight_n0.mem"), fc2_n0)

    # Interleaved data/weight pairs (for tb_real_weights.sv)
    def write_pairs(path, data, weights):
        with open(path, "w") as f:
            for d, w in zip(data, weights):
                f.write(f"{int(d) & 0xFF:02x}\n")
                f.write(f"{int(w) & 0xFF:02x}\n")

    write_pairs(os.path.join(OUTPUT_DIR, "conv1_5x5_pairs.mem"), conv1_act_5x5, conv1_f0)
    write_pairs(os.path.join(OUTPUT_DIR, "conv1_0x0_pairs.mem"), conv1_act_0x0, conv1_f0)
    write_pairs(os.path.join(OUTPUT_DIR, "conv2_4x4_pairs.mem"), conv2_act, conv2_f0)
    write_pairs(os.path.join(OUTPUT_DIR, "fc2_n0_pairs.mem"), fc2_input, fc2_n0)

    # Expected results CSV
    results_path = os.path.join(OUTPUT_DIR, "expected_results.csv")
    with open(results_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["test", "expected_value"])
        writer.writerow(["conv1_f0_px5x5", expected_1])
        writer.writerow(["conv1_f0_px0x0", expected_2])
        writer.writerow(["conv2_f0_px4x4", expected_3])
        writer.writerow(["fc2_n0", expected_4])

    print(f"\nAll files written to {OUTPUT_DIR}/")
    print(f"Expected results saved to {results_path}")
    print("\nDone! Run tb_real_weights.sv with iverilog to verify RTL matches.")


if __name__ == "__main__":
    main()
