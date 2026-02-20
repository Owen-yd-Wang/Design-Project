//==============================================================================
// Testbench: tb_real_weights
// Description: RTL vs Python comparison using REAL trained model weights
//              and a REAL CIFAR-10 test image (cat, index 0)
//
// Loads actual int8 quantized weights from .mem files exported from the
// trained ONNX model (tiny_cnn_cifar10_int8.onnx).
//
// The MAC unit operates on uint8 inputs with uint32 accumulation.
// Python reference computes the same unsigned arithmetic for bit-exact match.
//
// Tests:
//   1. Conv1 filter 0 at pixel (5,5) — interior pixel, 27 MACs
//   2. Conv1 filter 0 at pixel (0,0) — corner with zero-padding, 27 MACs
//   3. Conv2 filter 0 at pixel (4,4) — 144 MACs
//   4. FC2 neuron 0 — 64 MACs
//==============================================================================

module tb_real_weights;

    logic        clock;
    logic        reset;
    logic [7:0]  data_in;
    logic [7:0]  weight_in;
    logic        enable;
    logic        clear_acc;
    logic [31:0] acc_out;
    logic        valid;

    int test_passed = 0;
    int test_failed = 0;
    int log_file;

    // Clock: 100 MHz
    initial begin
        clock = 0;
        forever #5 clock = ~clock;
    end

    mac_uint8_int32 dut (
        .clock(clock), .reset(reset),
        .data_in(data_in), .weight_in(weight_in),
        .enable(enable), .clear_acc(clear_acc),
        .acc_out(acc_out), .valid(valid)
    );

    // Memory arrays for real test vectors (interleaved data/weight pairs)
    logic [7:0] conv1_5x5_pairs [0:53];    // 27 MACs x 2 = 54
    logic [7:0] conv1_0x0_pairs [0:53];    // 27 MACs x 2 = 54
    logic [7:0] conv2_4x4_pairs [0:287];   // 144 MACs x 2 = 288
    logic [7:0] fc2_n0_pairs    [0:127];   // 64 MACs x 2 = 128

    // Helper task: run a MAC dot product and check result
    task automatic run_dot_product(
        input string test_name,
        input string description,
        input int num_macs,
        input logic [31:0] expected_unsigned
    );
        $display("\n--- %s ---", test_name);
        $display("    %s", description);

        clear_acc = 1;
        @(posedge clock);
        clear_acc = 0;
        enable = 1;

        // The data/weight values are set by the caller before each cycle
        for (int i = 0; i < num_macs; i++) begin
            @(posedge clock);
        end
        enable = 0;
        @(posedge clock);

        if (acc_out == expected_unsigned) begin
            $display("PASS | %s | RTL: %0d | Python: %0d", test_name, acc_out, expected_unsigned);
            $fdisplay(log_file, "PASS,%s,%0d,%0d", test_name, acc_out, expected_unsigned);
            test_passed++;
        end else begin
            $display("FAIL | %s | RTL: %0d | Python: %0d", test_name, acc_out, expected_unsigned);
            $fdisplay(log_file, "FAIL,%s,%0d,%0d", test_name, acc_out, expected_unsigned);
            test_failed++;
        end
    endtask

    initial begin
        // Load real model weights and CIFAR-10 image patches
        $readmemh("tiny-cnn-basys3/real_test_vectors/conv1_5x5_pairs.mem", conv1_5x5_pairs);
        $readmemh("tiny-cnn-basys3/real_test_vectors/conv1_0x0_pairs.mem", conv1_0x0_pairs);
        $readmemh("tiny-cnn-basys3/real_test_vectors/conv2_4x4_pairs.mem", conv2_4x4_pairs);
        $readmemh("tiny-cnn-basys3/real_test_vectors/fc2_n0_pairs.mem",    fc2_n0_pairs);

        log_file = $fopen("rtl_real_weights_results.csv", "w");
        $fdisplay(log_file, "status,test,rtl_value,python_value");

        // Initialize
        reset = 1; data_in = 0; weight_in = 0;
        enable = 0; clear_acc = 0;
        repeat(3) @(posedge clock);
        reset = 0;
        @(posedge clock);

        $display("");
        $display("============================================================");
        $display("  RTL vs Python — Real Trained Weights, Real CIFAR-10 Image");
        $display("  Test image: cat (CIFAR-10 index 0)");
        $display("  Weights: from tiny_cnn_cifar10_int8.onnx");
        $display("============================================================");

        // ============================================================
        // Test 1: Conv1 filter 0, pixel (5,5) — interior, 27 MACs
        // Python expected (unsigned): 466870
        // ============================================================
        $display("\n--- Test 1: Conv1 filter0 @ pixel(5,5) — 27 MACs ---");
        $display("    Real cat image patch x Real trained Conv1 filter 0");

        clear_acc = 1; @(posedge clock); clear_acc = 0;
        enable = 1;
        for (int i = 0; i < 27; i++) begin
            data_in   = conv1_5x5_pairs[i*2];
            weight_in = conv1_5x5_pairs[i*2 + 1];
            @(posedge clock);
        end
        enable = 0; @(posedge clock);

        if (acc_out == 32'd466870) begin
            $display("PASS | Conv1 f0 @ (5,5) | RTL: %0d | Python: 466870", acc_out);
            $fdisplay(log_file, "PASS,conv1_f0_px5x5,%0d,466870", acc_out);
            test_passed++;
        end else begin
            $display("FAIL | Conv1 f0 @ (5,5) | RTL: %0d | Python: 466870", acc_out);
            $fdisplay(log_file, "FAIL,conv1_f0_px5x5,%0d,466870", acc_out);
            test_failed++;
        end

        // ============================================================
        // Test 2: Conv1 filter 0, pixel (0,0) — corner, 27 MACs
        // Python expected (unsigned): 118476
        // ============================================================
        $display("\n--- Test 2: Conv1 filter0 @ pixel(0,0) — corner, 27 MACs ---");
        $display("    Corner pixel with zero-padding");

        clear_acc = 1; @(posedge clock); clear_acc = 0;
        enable = 1;
        for (int i = 0; i < 27; i++) begin
            data_in   = conv1_0x0_pairs[i*2];
            weight_in = conv1_0x0_pairs[i*2 + 1];
            @(posedge clock);
        end
        enable = 0; @(posedge clock);

        if (acc_out == 32'd118476) begin
            $display("PASS | Conv1 f0 @ (0,0) | RTL: %0d | Python: 118476", acc_out);
            $fdisplay(log_file, "PASS,conv1_f0_px0x0,%0d,118476", acc_out);
            test_passed++;
        end else begin
            $display("FAIL | Conv1 f0 @ (0,0) | RTL: %0d | Python: 118476", acc_out);
            $fdisplay(log_file, "FAIL,conv1_f0_px0x0,%0d,118476", acc_out);
            test_failed++;
        end

        // ============================================================
        // Test 3: Conv2 filter 0, pixel (4,4) — 144 MACs
        // Python expected (unsigned): 2576776
        // ============================================================
        $display("\n--- Test 3: Conv2 filter0 @ pixel(4,4) — 144 MACs ---");
        $display("    16-channel input x Real trained Conv2 filter 0");

        clear_acc = 1; @(posedge clock); clear_acc = 0;
        enable = 1;
        for (int i = 0; i < 144; i++) begin
            data_in   = conv2_4x4_pairs[i*2];
            weight_in = conv2_4x4_pairs[i*2 + 1];
            @(posedge clock);
        end
        enable = 0; @(posedge clock);

        if (acc_out == 32'd2576776) begin
            $display("PASS | Conv2 f0 @ (4,4) | RTL: %0d | Python: 2576776", acc_out);
            $fdisplay(log_file, "PASS,conv2_f0_px4x4,%0d,2576776", acc_out);
            test_passed++;
        end else begin
            $display("FAIL | Conv2 f0 @ (4,4) | RTL: %0d | Python: 2576776", acc_out);
            $fdisplay(log_file, "FAIL,conv2_f0_px4x4,%0d,2576776", acc_out);
            test_failed++;
        end

        // ============================================================
        // Test 4: FC2 neuron 0 — 64 MACs
        // Python expected (unsigned): 1255461
        // ============================================================
        $display("\n--- Test 4: FC2 neuron 0 — 64 MACs ---");
        $display("    64-element input x Real trained FC2 weights row 0");

        clear_acc = 1; @(posedge clock); clear_acc = 0;
        enable = 1;
        for (int i = 0; i < 64; i++) begin
            data_in   = fc2_n0_pairs[i*2];
            weight_in = fc2_n0_pairs[i*2 + 1];
            @(posedge clock);
        end
        enable = 0; @(posedge clock);

        if (acc_out == 32'd1255461) begin
            $display("PASS | FC2 neuron 0 | RTL: %0d | Python: 1255461", acc_out);
            $fdisplay(log_file, "PASS,fc2_n0,%0d,1255461", acc_out);
            test_passed++;
        end else begin
            $display("FAIL | FC2 neuron 0 | RTL: %0d | Python: 1255461", acc_out);
            $fdisplay(log_file, "FAIL,fc2_n0,%0d,1255461", acc_out);
            test_failed++;
        end

        // ============================================================
        // Summary
        // ============================================================
        $display("");
        $display("============================================================");
        $display("  RESULTS: %0d / %0d tests passed", test_passed, test_passed + test_failed);
        if (test_failed == 0)
            $display("  ALL TESTS PASSED — RTL matches Python exactly");
        else
            $display("  %0d TESTS FAILED", test_failed);
        $display("============================================================");
        $display("");

        $fclose(log_file);
        $finish;
    end

    // Waveform dump
    initial begin
        $dumpfile("rtl_real_weights.vcd");
        $dumpvars(0, tb_real_weights);
    end

endmodule
