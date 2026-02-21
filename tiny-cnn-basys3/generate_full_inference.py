#!/usr/bin/env python3
"""
Generate test data for full end-to-end int8 CNN inference through RTL.

Extracts all quantization parameters from the ONNX int8 model,
computes the reference inference step-by-step, and generates .mem files
for the SystemVerilog testbench (tb_full_inference.sv).

Output: full_inference_data/ directory with all .mem files + reference results.
"""

import os
import numpy as np
import onnx
import onnxruntime as ort
import pickle

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "full_inference_data")
ONNX_PATH = os.path.join(SCRIPT_DIR, "tiny_cnn_cifar10_int8.onnx")
CIFAR_PATH = os.path.join(PROJECT_DIR, "cifar-10-batches-py", "test_batch")


def write_mem_hex8(path, data_flat):
    """Write int8/uint8 values as 2-digit hex, one per line."""
    with open(path, "w") as f:
        for v in data_flat:
            f.write(f"{int(v) & 0xFF:02x}\n")


def write_mem_hex32(path, data_flat):
    """Write int32 values as 8-digit hex, one per line."""
    with open(path, "w") as f:
        for v in data_flat:
            f.write(f"{int(v) & 0xFFFFFFFF:08x}\n")


def write_mem_float(path, data_flat):
    """Write float32 values as text, one per line."""
    with open(path, "w") as f:
        for v in data_flat:
            f.write(f"{float(v):.10e}\n")


def qlinear_conv2d(x_int8, x_scale, x_zp, w_int8, w_scale, w_zp,
                   bias_int32, y_scale, y_zp, pad=1, stride=1):
    """
    Integer-only QLinearConv matching ONNX runtime.
    x_int8: [C_in, H, W] int8
    w_int8: [C_out, C_in, kH, kW] int8
    Returns: [C_out, H_out, W_out] int8
    """
    C_out, C_in, kH, kW = w_int8.shape
    _, H, W = x_int8.shape
    H_out = (H + 2 * pad - kH) // stride + 1
    W_out = (W + 2 * pad - kW) // stride + 1

    # Pad input
    x_padded = np.full((C_in, H + 2*pad, W + 2*pad), x_zp, dtype=np.int8)
    x_padded[:, pad:pad+H, pad:pad+W] = x_int8

    # Real-value scale factor
    M = float(x_scale) * float(w_scale) / float(y_scale)

    output = np.zeros((C_out, H_out, W_out), dtype=np.int8)

    for f in range(C_out):
        for oh in range(H_out):
            for ow in range(W_out):
                # Signed integer accumulation
                acc = int(bias_int32[f])
                for c in range(C_in):
                    for kh in range(kH):
                        for kw in range(kW):
                            x_val = int(x_padded[c, oh*stride+kh, ow*stride+kw])
                            w_val = int(w_int8[f, c, kh, kw])
                            acc += (x_val - int(x_zp)) * (w_val - int(w_zp))
                # Requantize
                result = round(acc * M) + int(y_zp)
                result = max(-128, min(127, result))
                output[f, oh, ow] = np.int8(result)

    return output


def maxpool2d(x_int8, kernel=2, stride=2):
    """MaxPool2d on int8 data."""
    C, H, W = x_int8.shape
    H_out = H // stride
    W_out = W // stride
    output = np.zeros((C, H_out, W_out), dtype=np.int8)
    for c in range(C):
        for oh in range(H_out):
            for ow in range(W_out):
                patch = x_int8[c, oh*stride:oh*stride+kernel, ow*stride:ow*stride+kernel]
                output[c, oh, ow] = np.max(patch)
    return output


def qlinear_gemm(x_int8, x_scale, x_zp, w_int8, w_scale, w_zp,
                 bias_int32, y_scale, y_zp):
    """
    Integer-only QGemm: x @ w^T + bias
    x_int8: [N] int8 (flattened input)
    w_int8: [Out, In] int8
    Returns: [Out] int8
    """
    Out, In = w_int8.shape
    M = float(x_scale) * float(w_scale) / float(y_scale)

    output = np.zeros(Out, dtype=np.int8)
    for o in range(Out):
        acc = int(bias_int32[o])
        for i in range(In):
            x_val = int(x_int8[i])
            w_val = int(w_int8[o, i])
            acc += (x_val - int(x_zp)) * (w_val - int(w_zp))
        result = round(acc * M) + int(y_zp)
        result = max(-128, min(127, result))
        output[o] = np.int8(result)

    return output


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # -----------------------------------------------------------------------
    # Load ONNX model and extract all parameters
    # -----------------------------------------------------------------------
    model = onnx.load(ONNX_PATH)
    params = {}
    for init in model.graph.initializer:
        params[init.name] = onnx.numpy_helper.to_array(init)

    print("=== Quantization Parameters ===")

    # Layer parameters
    layers = {
        "conv1": {
            "x_scale": float(params["input_scale"]),
            "x_zp": int(params["input_zero_point"]),
            "w": params["features.0.weight_quantized"],     # [16,3,3,3]
            "w_scale": float(params["features.0.weight_scale"]),
            "w_zp": int(params["features.0.weight_zero_point"]),
            "bias": params["features.0.bias_quantized"],     # [16] int32
            "y_scale": float(params["conv2d_scale"]),
            "y_zp": int(params["conv2d_zero_point"]),
        },
        "conv2": {
            "x_scale": float(params["conv2d_scale"]),
            "x_zp": int(params["conv2d_zero_point"]),
            "w": params["features.3.weight_quantized"],     # [32,16,3,3]
            "w_scale": float(params["features.3.weight_scale"]),
            "w_zp": int(params["features.3.weight_zero_point"]),
            "bias": params["features.3.bias_quantized"],     # [32] int32
            "y_scale": float(params["conv2d_1_scale"]),
            "y_zp": int(params["conv2d_1_zero_point"]),
        },
        "conv3": {
            "x_scale": float(params["conv2d_1_scale"]),
            "x_zp": int(params["conv2d_1_zero_point"]),
            "w": params["features.6.weight_quantized"],     # [32,32,3,3]
            "w_scale": float(params["features.6.weight_scale"]),
            "w_zp": int(params["features.6.weight_zero_point"]),
            "bias": params["features.6.bias_quantized"],     # [32] int32
            "y_scale": float(params["conv2d_2_scale"]),
            "y_zp": int(params["conv2d_2_zero_point"]),
        },
        "fc1": {
            "x_scale": float(params["conv2d_2_scale"]),
            "x_zp": int(params["conv2d_2_zero_point"]),
            "w": params["classifier.0.weight_quantized"],   # [64,512]
            "w_scale": float(params["classifier.0.weight_scale"]),
            "w_zp": int(params["classifier.0.weight_zero_point"]),
            "bias": params["classifier.0.bias_quantized"],   # [64] int32
            "y_scale": float(params["linear_scale"]),
            "y_zp": int(params["linear_zero_point"]),
        },
        "fc2": {
            "x_scale": float(params["linear_scale"]),
            "x_zp": int(params["linear_zero_point"]),
            "w": params["classifier.2.weight_quantized"],   # [10,64]
            "w_scale": float(params["classifier.2.weight_scale"]),
            "w_zp": int(params["classifier.2.weight_zero_point"]),
            "bias": params["classifier.2.bias_quantized"],   # [10] int32
            "y_scale": float(params["output_scale"]),
            "y_zp": int(params["output_zero_point"]),
        },
    }

    for name, L in layers.items():
        print(f"\n  {name}:")
        print(f"    x_scale={L['x_scale']:.8f}, x_zp={L['x_zp']}")
        print(f"    w_scale={L['w_scale']:.8f}, w_zp={L['w_zp']}")
        print(f"    y_scale={L['y_scale']:.8f}, y_zp={L['y_zp']}")
        print(f"    w shape={L['w'].shape}, bias shape={L['bias'].shape}")
        M = L['x_scale'] * L['w_scale'] / L['y_scale']
        print(f"    requant M={M:.10f}")

    # -----------------------------------------------------------------------
    # Load and quantize input image
    # -----------------------------------------------------------------------
    with open(CIFAR_PATH, "rb") as f:
        batch = pickle.load(f, encoding="bytes")
    img_float = batch[b"data"][0].reshape(3, 32, 32).astype(np.float32) / 255.0
    label = batch[b"labels"][0]
    print(f"\nInput image: CIFAR-10 index 0, label={label} (3=cat)")

    # Quantize input: x_q = round(x_float / input_scale) + input_zp
    inp_scale = float(params["input_scale"])
    inp_zp = int(params["input_zero_point"])
    img_quantized = np.clip(np.round(img_float / inp_scale) + inp_zp, -128, 127).astype(np.int8)
    print(f"Input quantized: shape={img_quantized.shape}, range=[{img_quantized.min()}, {img_quantized.max()}]")

    # -----------------------------------------------------------------------
    # Run full inference
    # -----------------------------------------------------------------------
    print("\n=== Running Full Int8 Inference ===")

    # Conv1: [3,32,32] -> [16,32,32]
    L = layers["conv1"]
    print("\nConv1: [3,32,32] -> [16,32,32]...")
    conv1_out = qlinear_conv2d(img_quantized, L["x_scale"], L["x_zp"],
                                L["w"], L["w_scale"], L["w_zp"],
                                L["bias"], L["y_scale"], L["y_zp"])
    print(f"  Conv1 output range: [{conv1_out.min()}, {conv1_out.max()}]")

    # MaxPool1: [16,32,32] -> [16,16,16]
    pool1_out = maxpool2d(conv1_out)
    print(f"  Pool1 output: {pool1_out.shape}, range: [{pool1_out.min()}, {pool1_out.max()}]")

    # Conv2: [16,16,16] -> [32,16,16]
    L = layers["conv2"]
    print("\nConv2: [16,16,16] -> [32,16,16]...")
    conv2_out = qlinear_conv2d(pool1_out, L["x_scale"], L["x_zp"],
                                L["w"], L["w_scale"], L["w_zp"],
                                L["bias"], L["y_scale"], L["y_zp"])
    print(f"  Conv2 output range: [{conv2_out.min()}, {conv2_out.max()}]")

    # MaxPool2: [32,16,16] -> [32,8,8]
    pool2_out = maxpool2d(conv2_out)
    print(f"  Pool2 output: {pool2_out.shape}, range: [{pool2_out.min()}, {pool2_out.max()}]")

    # Conv3: [32,8,8] -> [32,8,8]
    L = layers["conv3"]
    print("\nConv3: [32,8,8] -> [32,8,8]...")
    conv3_out = qlinear_conv2d(pool2_out, L["x_scale"], L["x_zp"],
                                L["w"], L["w_scale"], L["w_zp"],
                                L["bias"], L["y_scale"], L["y_zp"])
    print(f"  Conv3 output range: [{conv3_out.min()}, {conv3_out.max()}]")

    # MaxPool3: [32,8,8] -> [32,4,4]
    pool3_out = maxpool2d(conv3_out)
    print(f"  Pool3 output: {pool3_out.shape}, range: [{pool3_out.min()}, {pool3_out.max()}]")

    # Flatten: [32,4,4] -> [512]
    flat = pool3_out.flatten()
    print(f"\nFlatten: {flat.shape}")

    # FC1: [512] -> [64]
    L = layers["fc1"]
    print("\nFC1: [512] -> [64]...")
    fc1_out = qlinear_gemm(flat, L["x_scale"], L["x_zp"],
                            L["w"], L["w_scale"], L["w_zp"],
                            L["bias"], L["y_scale"], L["y_zp"])
    print(f"  FC1 output range: [{fc1_out.min()}, {fc1_out.max()}]")

    # FC2: [64] -> [10]
    L = layers["fc2"]
    print("\nFC2: [64] -> [10]...")
    fc2_out = qlinear_gemm(fc1_out, L["x_scale"], L["x_zp"],
                            L["w"], L["w_scale"], L["w_zp"],
                            L["bias"], L["y_scale"], L["y_zp"])
    print(f"  FC2 output (int8): {fc2_out}")

    # Dequantize final output
    out_scale = float(params["output_scale"])
    out_zp = int(params["output_zero_point"])
    fc2_float = (fc2_out.astype(np.float32) - out_zp) * out_scale
    print(f"  FC2 output (float): {fc2_float}")
    print(f"  Prediction: class {np.argmax(fc2_float)} (3=cat)")

    # Verify against ONNX runtime
    print("\n=== Verification vs ONNX Runtime ===")
    sess = ort.InferenceSession(ONNX_PATH)
    ort_out = sess.run(None, {"input": img_float[np.newaxis]})[0][0]
    print(f"  ONNX Runtime output: {ort_out}")
    print(f"  Our output:          {fc2_float}")
    print(f"  Max difference:      {np.max(np.abs(ort_out - fc2_float)):.6f}")

    # -----------------------------------------------------------------------
    # Write .mem files for testbench
    # -----------------------------------------------------------------------
    print("\n=== Writing .mem files ===")

    # Input image (int8 stored as hex bytes)
    write_mem_hex8(os.path.join(OUTPUT_DIR, "input_image.mem"), img_quantized.flatten())
    print(f"  input_image.mem: {img_quantized.size} bytes")

    # Weights and biases for each layer
    for name, L in layers.items():
        write_mem_hex8(os.path.join(OUTPUT_DIR, f"{name}_weights.mem"), L["w"].flatten())
        write_mem_hex32(os.path.join(OUTPUT_DIR, f"{name}_bias.mem"), L["bias"].flatten())
        print(f"  {name}_weights.mem: {L['w'].size} bytes")
        print(f"  {name}_bias.mem: {L['bias'].size} words")

    # Intermediate activations (for verification)
    write_mem_hex8(os.path.join(OUTPUT_DIR, "conv1_out.mem"), conv1_out.flatten())
    write_mem_hex8(os.path.join(OUTPUT_DIR, "pool1_out.mem"), pool1_out.flatten())
    write_mem_hex8(os.path.join(OUTPUT_DIR, "conv2_out.mem"), conv2_out.flatten())
    write_mem_hex8(os.path.join(OUTPUT_DIR, "pool2_out.mem"), pool2_out.flatten())
    write_mem_hex8(os.path.join(OUTPUT_DIR, "conv3_out.mem"), conv3_out.flatten())
    write_mem_hex8(os.path.join(OUTPUT_DIR, "pool3_out.mem"), pool3_out.flatten())
    write_mem_hex8(os.path.join(OUTPUT_DIR, "fc1_out.mem"), fc1_out.flatten())
    write_mem_hex8(os.path.join(OUTPUT_DIR, "fc2_out.mem"), fc2_out.flatten())

    # Quantization parameters file (for testbench to read)
    with open(os.path.join(OUTPUT_DIR, "quant_params.txt"), "w") as f:
        for name, L in layers.items():
            M = L['x_scale'] * L['w_scale'] / L['y_scale']
            f.write(f"{name} x_zp={L['x_zp']} w_zp={L['w_zp']} y_zp={L['y_zp']} M={M:.10f}\n")
        f.write(f"output scale={out_scale:.10f} zp={out_zp}\n")

    # Final output reference
    with open(os.path.join(OUTPUT_DIR, "final_output.txt"), "w") as f:
        f.write("# Final 1x10 output vector (int8 quantized)\n")
        f.write("# class, int8_value, float_value\n")
        classes = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
        for i in range(10):
            marker = " <-- PREDICTED" if i == np.argmax(fc2_float) else ""
            f.write(f"{i},{classes[i]},{int(fc2_out[i])},{fc2_float[i]:.6f}{marker}\n")

    print(f"\n=== FINAL 1x10 OUTPUT VECTOR ===")
    classes = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
    print(f"{'Class':<12} {'Int8':>6} {'Float':>10}")
    print("-" * 32)
    for i in range(10):
        marker = " <--" if i == np.argmax(fc2_float) else ""
        print(f"{classes[i]:<12} {int(fc2_out[i]):>6} {fc2_float[i]:>10.4f}{marker}")

    print(f"\nPrediction: class {np.argmax(fc2_float)} = {classes[np.argmax(fc2_float)]}")
    print(f"\nAll files written to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
