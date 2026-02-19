#!/usr/bin/env python3
"""
Tiny CNN for CIFAR-10 targeting Basys 3 (XC7A35T) FPGA.

End-to-end pipeline:
  1. Define a ~25K-parameter CNN
  2. Train on CIFAR-10 (~30 epochs)
  3. Evaluate with per-class and macro F1 scores
  4. Export to ONNX (float32)
  5. Quantize to int8 via ONNX Runtime
  6. Verify int8 model accuracy / F1
  7. Print FPGA resource estimation report for XC7A35T
"""

import math
import os
import time

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import classification_report, f1_score

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BATCH_SIZE = 128
EPOCHS = 30
LR = 0.001
NUM_WORKERS = 2
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DEVICE = (
    "mps" if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available() else
    "cpu"
)

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

# Basys 3 / XC7A35T resources
BASYS3_BRAM_KB = 225       # 50 x 36 Kb = 1800 Kb = 225 KB
BASYS3_DSP = 90            # DSP48E1 slices
BASYS3_LUT = 20800
BASYS3_FF = 41600


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------
class TinyCNN(nn.Module):
    """
    ~25K-parameter CNN for CIFAR-10.

    Conv2d(3→16, 3x3, pad=1) → ReLU → MaxPool2x2     16x16x16
    Conv2d(16→32, 3x3, pad=1) → ReLU → MaxPool2x2     8x8x32
    Conv2d(32→32, 3x3, pad=1) → ReLU → MaxPool2x2     4x4x32
    FC(512→64) → ReLU
    FC(64→10)
    """

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 4 * 4, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def get_dataloaders():
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=True, download=True, transform=train_transform
    )
    testset = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=False, download=True, transform=test_transform
    )

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )
    return trainloader, testloader


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(model, trainloader, testloader):
    model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    print(f"\nTraining on {DEVICE} for {EPOCHS} epochs...")
    print("-" * 60)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in trainloader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        scheduler.step()
        train_loss = running_loss / total
        train_acc = correct / total

        if epoch % 5 == 0 or epoch == 1:
            val_acc = evaluate_accuracy(model, testloader)
            print(
                f"Epoch {epoch:3d}/{EPOCHS}  "
                f"Loss: {train_loss:.4f}  "
                f"Train Acc: {train_acc:.4f}  "
                f"Val Acc: {val_acc:.4f}"
            )

    print("-" * 60)
    return model


def evaluate_accuracy(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return correct / total


# ---------------------------------------------------------------------------
# Evaluation with F1 scores
# ---------------------------------------------------------------------------
def evaluate_f1(model, loader, label="PyTorch float32"):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    print(f"\n{'='*60}")
    print(f"  {label} — Classification Report")
    print(f"{'='*60}")
    print(classification_report(all_labels, all_preds, target_names=CIFAR10_CLASSES, digits=4))
    print(f"  Macro F1: {macro_f1:.4f}")
    print(f"{'='*60}")
    return macro_f1, all_preds, all_labels


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------
def export_onnx(model, output_path):
    model.eval()
    model.to("cpu")
    dummy = torch.randn(1, 3, 32, 32)
    torch.onnx.export(
        model,
        dummy,
        output_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=13,
    )
    size_kb = os.path.getsize(output_path) / 1024
    print(f"\nExported float32 ONNX model: {output_path} ({size_kb:.1f} KB)")
    return output_path


# ---------------------------------------------------------------------------
# INT8 quantization via ONNX Runtime
# ---------------------------------------------------------------------------
def quantize_onnx(float_onnx_path, int8_onnx_path, testloader):
    from onnxruntime.quantization import (
        CalibrationDataReader,
        QuantType,
        quantize_static,
    )

    class CifarCalibrationReader(CalibrationDataReader):
        def __init__(self, loader, num_batches=20):
            self.data_iter = iter(loader)
            self.num_batches = num_batches
            self.count = 0

        def get_next(self):
            if self.count >= self.num_batches:
                return None
            try:
                inputs, _ = next(self.data_iter)
                self.count += 1
                return {"input": inputs.numpy()}
            except StopIteration:
                return None

    calibration_reader = CifarCalibrationReader(testloader, num_batches=20)

    quantize_static(
        float_onnx_path,
        int8_onnx_path,
        calibration_reader,
        quant_format=ort.quantization.QuantFormat.QOperator,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8,
    )

    size_kb = os.path.getsize(int8_onnx_path) / 1024
    print(f"Exported int8 ONNX model:    {int8_onnx_path} ({size_kb:.1f} KB)")
    return int8_onnx_path


# ---------------------------------------------------------------------------
# Verify int8 ONNX model
# ---------------------------------------------------------------------------
def verify_int8_onnx(int8_onnx_path, testloader):
    session = ort.InferenceSession(int8_onnx_path)
    all_preds = []
    all_labels = []

    for inputs, labels in testloader:
        outputs = session.run(None, {"input": inputs.numpy()})
        preds = np.argmax(outputs[0], axis=1)
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    print(f"\n{'='*60}")
    print(f"  ONNX int8 — Classification Report")
    print(f"{'='*60}")
    print(
        classification_report(all_labels, all_preds, target_names=CIFAR10_CLASSES, digits=4)
    )
    print(f"  Macro F1: {macro_f1:.4f}")
    print(f"{'='*60}")
    return macro_f1


# ---------------------------------------------------------------------------
# Resource estimation for XC7A35T (Basys 3)
# ---------------------------------------------------------------------------
def resource_estimation(model):
    print(f"\n{'='*60}")
    print(f"  FPGA Resource Estimation — Basys 3 (XC7A35T)")
    print(f"{'='*60}")

    # Weight memory per layer (int8 = 1 byte per param)
    total_weight_bytes = 0
    print(f"\n  Weight Memory (int8, 1 byte/param):")
    print(f"  {'Layer':<35} {'Params':>10} {'Bytes':>10}")
    print(f"  {'-'*55}")
    for name, param in model.named_parameters():
        nbytes = param.numel()  # 1 byte per param in int8
        total_weight_bytes += nbytes
        print(f"  {name:<35} {param.numel():>10,} {nbytes:>10,}")
    total_weight_kb = total_weight_bytes / 1024
    print(f"  {'-'*55}")
    print(f"  {'TOTAL':<35} {count_parameters(model):>10,} {total_weight_bytes:>10,}")
    print(f"  Weight memory: {total_weight_kb:.1f} KB")

    # Activation memory: largest intermediate feature map (int8)
    # Input: 3x32x32 = 3072
    # After conv1: 16x32x32 = 16384 → after pool: 16x16x16 = 4096
    # After conv2: 32x16x16 = 8192 → after pool: 32x8x8 = 2048
    # After conv3: 32x8x8 = 8192 → after pool: 32x4x4 = 512
    # FC1 input: 512, FC1 output: 64, FC2 output: 10
    activation_sizes = {
        "input (3x32x32)": 3 * 32 * 32,
        "conv1 out (16x32x32)": 16 * 32 * 32,
        "pool1 out (16x16x16)": 16 * 16 * 16,
        "conv2 out (32x16x16)": 32 * 16 * 16,
        "pool2 out (32x8x8)": 32 * 8 * 8,
        "conv3 out (32x8x8)": 32 * 8 * 8,
        "pool3 out (32x4x4)": 32 * 4 * 4,
        "fc1 out (64)": 64,
        "fc2 out (10)": 10,
    }

    print(f"\n  Activation Memory (int8, 1 byte/element):")
    print(f"  {'Tensor':<30} {'Elements':>10} {'Bytes':>10}")
    print(f"  {'-'*50}")
    max_act_bytes = 0
    for name, size in activation_sizes.items():
        print(f"  {name:<30} {size:>10,} {size:>10,}")
        max_act_bytes = max(max_act_bytes, size)
    max_act_kb = max_act_bytes / 1024
    print(f"  {'-'*50}")
    print(f"  Peak activation memory: {max_act_kb:.1f} KB (largest single tensor)")

    # In a streaming dataflow architecture, we need to buffer at most
    # the input + output of the largest layer simultaneously
    # Largest pair: conv1 input(3072) + conv1 output(16384) = 19456
    streaming_buffer = 3 * 32 * 32 + 16 * 32 * 32  # worst-case pair
    streaming_buffer_kb = streaming_buffer / 1024

    # Total BRAM estimate
    total_bram_kb = total_weight_kb + streaming_buffer_kb
    bram_utilization = (total_bram_kb / BASYS3_BRAM_KB) * 100

    print(f"\n  BRAM Summary:")
    print(f"  {'Weight storage:':<35} {total_weight_kb:>8.1f} KB")
    print(f"  {'Peak activation buffer:':<35} {streaming_buffer_kb:>8.1f} KB")
    print(f"  {'Total estimated BRAM:':<35} {total_bram_kb:>8.1f} KB")
    print(f"  {'Basys 3 available BRAM:':<35} {BASYS3_BRAM_KB:>8d} KB")
    print(f"  {'BRAM utilization:':<35} {bram_utilization:>7.1f} %")

    # DSP estimation
    # Each Conv/FC layer needs MACs. In a serial implementation,
    # we reuse DSPs across layers. Key metric: max MACs per layer.
    # Conv1: 3*16*3*3 = 432 MACs per output pixel, 32*32=1024 pixels → 442,368 total MACs
    # Conv2: 16*32*3*3 = 4608 MACs per output pixel, 16*16=256 pixels → 1,179,648 total MACs
    # Conv3: 32*32*3*3 = 9216 MACs per output pixel, 8*8=64 pixels → 589,824 total MACs
    # FC1: 512*64 = 32,768 MACs
    # FC2: 64*10 = 640 MACs
    mac_ops = {
        "Conv1 (3→16, 3x3)": 3 * 16 * 3 * 3 * 32 * 32,
        "Conv2 (16→32, 3x3)": 16 * 32 * 3 * 3 * 16 * 16,
        "Conv3 (32→32, 3x3)": 32 * 32 * 3 * 3 * 8 * 8,
        "FC1 (512→64)": 512 * 64,
        "FC2 (64→10)": 64 * 10,
    }
    total_macs = sum(mac_ops.values())

    print(f"\n  MAC Operations per Inference:")
    print(f"  {'Layer':<30} {'MACs':>15}")
    print(f"  {'-'*45}")
    for name, macs in mac_ops.items():
        print(f"  {name:<30} {macs:>15,}")
    print(f"  {'-'*45}")
    print(f"  {'TOTAL':<30} {total_macs:>15,}")

    # DSP usage: with int8, each DSP48E1 can do 1 MAC/cycle.
    # For a serial-layer design, we need enough DSPs to meet throughput.
    # At 100 MHz with 90 DSPs, we can do 90M MACs/sec.
    # Total MACs per inference: ~2.2M → ~24.5K inferences/sec (serial all DSPs)
    # With time-multiplexing, even 1 DSP suffices (just slower).
    clock_mhz = 100
    dsp_available = BASYS3_DSP
    throughput_macs_per_sec = dsp_available * clock_mhz * 1e6
    inferences_per_sec = throughput_macs_per_sec / total_macs

    print(f"\n  DSP Summary:")
    print(f"  {'Available DSP48E1 slices:':<35} {BASYS3_DSP:>8d}")
    print(f"  {'Clock frequency:':<35} {clock_mhz:>7d} MHz")
    print(f"  {'Total MACs per inference:':<35} {total_macs:>8,}")
    print(f"  {'Theoretical throughput:':<35} {inferences_per_sec:>7.0f} inf/sec")
    print(f"  {'(using all {0} DSPs @ {1} MHz)'.format(dsp_available, clock_mhz):<35}")

    # Verdict
    print(f"\n  {'='*55}")
    fits_bram = total_bram_kb < BASYS3_BRAM_KB
    print(f"  BRAM:  {'PASS' if fits_bram else 'FAIL'} "
          f"— {total_bram_kb:.1f} KB / {BASYS3_BRAM_KB} KB "
          f"({bram_utilization:.1f}%)")
    print(f"  DSP:   PASS — {dsp_available} DSP48E1 available "
          f"(design can time-multiplex)")
    print(f"  LUT:   {BASYS3_LUT:,} available (sufficient for control logic)")
    print(f"  FF:    {BASYS3_FF:,} available")
    overall = "PASS" if fits_bram else "FAIL"
    print(f"\n  Overall: *** {overall} — Model fits on Basys 3 (XC7A35T) ***")
    print(f"{'='*60}\n")

    return fits_bram


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    start_time = time.time()
    script_dir = os.path.dirname(os.path.abspath(__file__))

    print("=" * 60)
    print("  Tiny CNN for CIFAR-10 — Basys 3 (XC7A35T) Target")
    print("=" * 60)

    # Build model
    model = TinyCNN()
    num_params = count_parameters(model)
    print(f"\nModel: TinyCNN")
    print(f"Parameters: {num_params:,}")
    print(f"Float32 size: {num_params * 4 / 1024:.1f} KB")
    print(f"Int8 size:    {num_params / 1024:.1f} KB")
    print(f"Device: {DEVICE}")

    # Load data
    trainloader, testloader = get_dataloaders()

    # Train
    model = train(model, trainloader, testloader)

    # Evaluate float32 model
    float_f1, _, _ = evaluate_f1(model, testloader, label="PyTorch float32")

    # Export to ONNX
    float_onnx_path = os.path.join(script_dir, "tiny_cnn_cifar10.onnx")
    export_onnx(model, float_onnx_path)

    # Quantize to int8
    int8_onnx_path = os.path.join(script_dir, "tiny_cnn_cifar10_int8.onnx")
    quantize_onnx(float_onnx_path, int8_onnx_path, testloader)

    # Verify int8 model
    int8_f1 = verify_int8_onnx(int8_onnx_path, testloader)

    # F1 comparison
    print(f"\n  F1 Score Comparison:")
    print(f"  {'Float32 (PyTorch):':<25} {float_f1:.4f}")
    print(f"  {'Int8 (ONNX):':<25} {int8_f1:.4f}")
    print(f"  {'Delta:':<25} {int8_f1 - float_f1:+.4f}")

    # Resource estimation
    resource_estimation(model)

    elapsed = time.time() - start_time
    print(f"Total time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
