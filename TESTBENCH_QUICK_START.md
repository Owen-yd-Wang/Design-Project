# Quantized MobileNetV2 Full Inference Testbench - Complete Guide

## Quick Start

### Step 1: Verify Files
Ensure you have the following files in your workspace:
- `tb_quant_mobilenet_v2_full.sv` - Main testbench
- `depthwise_conv3x3_engine.sv` - Conv engine module
- `mac_uint8_int32.sv` - MAC unit module
- `test_image.mem` - Input test image (32×32×3 quantized)
- `tiny_mems_int8/` - Directory with all weight files and manifest.csv

### Step 2: Run Testbench

**Windows (using batch script):**
```bash
run_testbench.bat
```

**Linux/Mac (using Makefile):**
```bash
make -f Makefile.testbench all
```

**Manual compilation (ModelSim):**
```bash
vlog -sv mac_uint8_int32.sv depthwise_conv3x3_engine.sv tb_quant_mobilenet_v2_full.sv
vsim work.tb_quant_mobilenet_v2_full -c -do "run -all"
```

### Step 3: Verify Results
```bash
./verify_testbench.sh simulation.log
```

---

## What the Testbench Does

The testbench implements complete MobileNetV2 inference:

```
Input Image (32×32×3 pixels)
    ↓
Layer 0 Conv: 3→16 channels, stride-2 → 16×16×16 featuremap
    ↓
Layer 3 Conv: 16→32 channels, stride-2 → 8×8×32 featuremap
    ↓
Layer 6 Conv: 32→32 channels, stride-2 → 4×4×32 featuremap
    ↓
Flatten: 4×4×32 = 512 values
    ↓
Fully Connected Layer 0: 512→64 neurons
    ↓
Fully Connected Layer 1: 64→10 outputs (classification)
    ↓
Output: Predicted class (0-9) + confidence scores
```

## Architecture Details

### Convolutional Layers

#### Layer 0: Input Convolution
- **Input:** 32×32×3 (RGB image)
- **Weights:** features.0.weight_quantized.w1.mem (16×3×3×3 = 432 weights)
- **Kernel:** 3×3, Stride: 2, Padding: SAME
- **Output:** 16×16×16

#### Layer 3: Intermediate Convolution
- **Input:** 16×16×16 (Layer 0 output)
- **Weights:** features.3.weight_quantized.w1.mem (32×16×3×3 = 4,608 weights)
- **Kernel:** 3×3, Stride: 2, Padding: SAME
- **Output:** 8×8×32

#### Layer 6: Feature Extraction Convolution
- **Input:** 8×8×32 (Layer 3 output)
- **Weights:** features.6.weight_quantized.w1.mem (32×32×3×3 = 9,216 weights)
- **Kernel:** 3×3, Stride: 2, Padding: SAME
- **Output:** 4×4×32

### Classifier Layers

#### FC Layer 0 (Linear 1)
- **Input:** 512 values (flattened 4×4×32)
- **Weights:** classifier.0.weight_quantized.w1.mem (64×512 = 32,768 weights)
- **Output:** 64 neurons

#### FC Layer 1 (Linear 2)
- **Input:** 64 neurons
- **Weights:** classifier.2.weight_quantized.w1.mem (10×64 = 640 weights)
- **Output:** 10 logits (one per class)

### Quantization Parameters

All weights are INT8 signed integers. Zero points are used for dequantization:

```
Dequantized_value = (QuantizedValue - ZeroPoint) * Scale
```

Zero point files are stored in `tiny_mems_int8/`:
- `input_zero_point.w1.mem`
- `features.0.weight_zero_point.w1.mem`
- `features.3.weight_zero_point.w1.mem`
- `features.6.weight_zero_point.w1.mem`
- `classifier.0.weight_zero_point.w1.mem`
- `classifier.2.weight_zero_point.w1.mem`
- `linear_zero_point.w1.mem`
- `output_zero_point.w1.mem`

---

## File Structure

```
.
├── tb_quant_mobilenet_v2_full.sv       ← Main testbench (THIS FILE)
├── depthwise_conv3x3_engine.sv          ← 3×3 conv engine
├── mac_uint8_int32.sv                   ← MAC unit
├── test_image.mem                       ← Input test image
│
├── tiny_mems_int8/                      ← Weight directory
│   ├── manifest.csv                     ← Weight manifest
│   ├── features.0.weight_quantized.w1.mem
│   ├── features.0.weight_zero_point.w1.mem
│   ├── features.3.weight_quantized.w1.mem
│   ├── features.3.weight_zero_point.w1.mem
│   ├── features.6.weight_quantized.w1.mem
│   ├── features.6.weight_zero_point.w1.mem
│   ├── classifier.0.weight_quantized.w1.mem
│   ├── classifier.0.weight_zero_point.w1.mem
│   ├── classifier.2.weight_quantized.w1.mem
│   ├── classifier.2.weight_zero_point.w1.mem
│   ├── input_zero_point.w1.mem
│   ├── linear_zero_point.w1.mem
│   └── output_zero_point.w1.mem
│
├── run_testbench.bat                    ← Windows batch script
├── Makefile.testbench                   ← Linux/Mac Makefile
├── verify_testbench.sh                  ← Verification script
│
├── README_TESTBENCH.md                  ← Detailed documentation
└── TESTBENCH_QUICK_START.md             ← This file
```

---

## Understanding the Output

When the testbench completes successfully, you'll see output like:

```
========== CLASSIFICATION RESULTS ==========
Output scores (logits):
  Class 0: 12345
  Class 1: -5432
  Class 2: 1234
  Class 3: 876
  Class 4: -234
  Class 5: 567
  Class 6: 890
  Class 7: 123
  Class 8: 456
  Class 9: 2345

*** PREDICTED CLASS: 0 ***
*** CONFIDENCE: 12345 ***

INFERENCE COMPLETE AND SUCCESSFUL
```

**Interpretation:**
- **Output scores (logits):** Raw neural network outputs for each class
- **Predicted class:** The class with the highest logit (Class 0)
- **Confidence:** The logit value for the predicted class (12345)
- Higher logits = higher confidence in that class

---

## Performance Metrics

### Computational Complexity
- **Total 3×3 convolutions:**
  - Layer 0: 16×16 positions × 16 output channels = 4,096 convolutions
  - Layer 3: 8×8 positions × 32 output channels = 2,048 convolutions
  - Layer 6: 4×4 positions × 32 output channels = 512 convolutions
  - **Total: ~6,656 convolutions**

- **MACs (Multiply-Accumulate operations):**
  - Each 3×3 conv = 9 MACs (one per kernel position)
  - **Total: ~60,000 MACs**

- **Weight reads:**
  - 432 + 4,608 + 9,216 + 32,768 + 640 = 47,664 weight values

### Simulation Time
Depends on your simulator and hardware:
- **ModelSim:** 2-10 minutes
- **VCS:** 1-5 minutes (faster)
- **Xcelium:** 1-5 minutes (fastest)

### Memory Usage
- **Test image:** 3,072 bytes
- **Layer 0 output:** 4,096 words (32-bit) = 16 KB
- **Layer 3 output:** 2,048 words (32-bit) = 8 KB
- **Layer 6 output:** 512 words (32-bit) = 2 KB
- **Weights:** ~47 KB

---

## Extending the Testbench

### Adding More Layers

To add additional convolutional layers (e.g., more intermediate layers):

1. **Define layer parameters:**
```systemverilog
parameter int LAYER9_OUT_H = 2;
parameter int LAYER9_OUT_W = 2;
parameter int LAYER9_OUT_C = 64;
```

2. **Load weights:**
```systemverilog
$readmemh("tiny_mems_int8/features.9.weight_quantized.w1.mem", layer9_weights);
$readmemh("tiny_mems_int8/features.9.weight_zero_point.w1.mem", layer9_zero_point);
```

3. **Create processing task:**
```systemverilog
task automatic process_layer9();
    // Similar to process_layer0/3/6
    // Extract windows from layer6_output
    // Apply 3×3 convolution
    // Store in layer9_output
endtask
```

4. **Call in main simulation:**
```systemverilog
process_layer9();
```

### Using Different Test Images

To test with a different image:

1. **Prepare image:**
   - Convert to 32×32×3 RGB
   - Quantize to INT8 [0-255]
   
2. **Export to hex format:**
   - One byte per line
   - Hexadecimal format (e.g., `FF`, `00`, `7F`)

3. **Replace test_image.mem**

4. **Rerun testbench**

### Batch Processing

To process multiple images:

```systemverilog
for (int img_idx = 0; img_idx < NUM_IMAGES; img_idx++) begin
    // Load image
    $readmemh($sformatf("images/test_%0d.mem", img_idx), test_image);
    
    // Run inference
    process_layer0();
    process_layer3();
    process_layer6();
    
    // Classify
    flatten_layer6(flattened);
    classifier_fc0(flattened);
    classifier_fc1(predicted_class);
    
    // Log results
    $display("Image %0d → Class %0d", img_idx, predicted_class);
end
```

---

## Debugging Tips

### 1. Check Individual Layers

Add debug statements in layer processing tasks:

```systemverilog
$display("Layer 0 position (%0d, %0d) channel %0d: %0d", 
         out_h, out_w, out_ch, $signed(layer0_output[idx]));
```

### 2. View Waveforms

**ModelSim:**
```tcl
add wave -recursive *
run -all
```

**VCS:**
```bash
./simv -gui
```

### 3. Monitor Critical Signals

Key signals to observe:
- `conv_window` - 3×3 input pixels
- `conv_kernel` - 3×3 kernel weights
- `conv_result` - Output before ReLU
- `conv_valid` - Result ready signal

### 4. Compare with Reference

Compare testbench outputs with:
- Python/TensorFlow reference model
- Previous simulation runs
- Known test cases

### 5. Check File Formats

Verify weight files are compatible:
```bash
# Check file size (bytes)
ls -la tiny_mems_int8/features.0.weight_quantized.w1.mem

# Should be: 432 bytes (16×3×3×3)
# Check hex format
head -20 tiny_mems_int8/features.0.weight_quantized.w1.mem
```

---

## Common Issues and Solutions

| Problem | Cause | Solution |
|---------|-------|----------|
| "File not found" errors | Missing weight files or test image | Check tiny_mems_int8/ directory and test_image.mem exist |
| Compilation errors | Module dependencies | Compile in order: mac_uint8_int32.sv → depthwise_conv3x3_engine.sv → testbench |
| Simulation timeout | Long simulation time | Increase timeout parameter or use faster simulator (VCS/Xcelium) |
| Zero outputs | Padding/boundary issues | Check get_pixel() padding logic |
| Incorrect results | Weight file format or quantization | Verify manifest.csv alignment with weight shapes |
| Out of memory | Large layer outputs | Reduce memory usage or use streaming architecture |

---

## Performance Optimization

### 1. Reduce Simulation Time
- Use VCS (2-3× faster than ModelSim)
- Disable waveform dumping: `+notimingchecks`
- Use compiled simulation

### 2. Parallelize Convolutions
- Process multiple channels in parallel
- Use streaming architecture

### 3. Memory Optimization
- Stream weights (don't load all at once)
- Reuse output buffers
- Use int8/int16 for intermediate results

### 4. Hardware Implementation
- Implement multiple parallel MAC units
- Use systolic array for layer processing
- Pipeline layers for throughput

---

## References

### Model Details
- **MobileNetV2 Paper:** Sandler et al., "MobileNetV2: Inverted Residuals and Linear Bottlenecks" (CVPR 2018)
- **Original dimensions:** 224×224×3 input
- **This testbench:** Adapted for 32×32×3 (for simulation efficiency)

### Quantization
- **Method:** INT8 symmetric quantization
- **Zero points:** Per-layer or per-channel
- **Scale factors:** Implicit in weight values

### Related Files
- `FINAL_CLASSIFICATION_PLAN.md` - Architecture documentation
- `README.md` - Project overview
- `manifest.csv` - Weight specifications

---

## Support

For issues or questions:
1. Check simulation logs for error messages
2. Review output dimensions at each layer
3. Verify weight file integrity
4. Compare with reference implementation
5. Check compilation output for warnings

---

**Generated:** February 2026  
**Project:** DP (Digital Processing) Quantized Neural Network Implementation  
**Status:** Ready for simulation
