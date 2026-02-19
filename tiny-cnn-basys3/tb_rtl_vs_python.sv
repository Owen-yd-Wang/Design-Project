//==============================================================================
// Testbench: tb_rtl_vs_python
// Description: RTL vs Python comparison testbench
//              Runs same CNN-relevant computations as Python, logs results
//              for side-by-side verification
//
// Tests:
//   1. Single MAC: 42 * 12 = 504
//   2. 3x3x3 Conv1 kernel dot product (27 MACs)
//   3. 3x3x16 Conv2 kernel dot product (144 MACs)
//   4. FC dot product (64 elements)
//==============================================================================

module tb_rtl_vs_python;

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

    // Data arrays for realistic CNN test vectors
    // Conv1 kernel: 3x3x3 = 27 values (activation, weight pairs)
    logic [7:0] conv1_data [0:26];
    logic [7:0] conv1_weight [0:26];

    // Conv2 kernel: 3x3x16 = 144 values
    logic [7:0] conv2_data [0:143];
    logic [7:0] conv2_weight [0:143];

    // FC: 64 values
    logic [7:0] fc_data [0:63];
    logic [7:0] fc_weight [0:63];

    task check_result(input string test_name, input int expected);
        if (acc_out == expected) begin
            $display("PASS | %s | RTL: %0d | Expected: %0d", test_name, acc_out, expected);
            $fdisplay(log_file, "PASS,%s,%0d,%0d", test_name, acc_out, expected);
            test_passed++;
        end else begin
            $display("FAIL | %s | RTL: %0d | Expected: %0d", test_name, acc_out, expected);
            $fdisplay(log_file, "FAIL,%s,%0d,%0d", test_name, acc_out, expected);
            test_failed++;
        end
    endtask

    initial begin
        log_file = $fopen("rtl_results.csv", "w");
        $fdisplay(log_file, "status,test,rtl_value,expected_value");

        // Seed test data with deterministic values
        // Conv1: 27 data/weight pairs
        for (int i = 0; i < 27; i++) begin
            conv1_data[i]   = (i * 7 + 13) % 256;  // pseudo-random 0-255
            conv1_weight[i] = (i * 11 + 3) % 256;
        end
        // Conv2: 144 data/weight pairs
        for (int i = 0; i < 144; i++) begin
            conv2_data[i]   = (i * 5 + 17) % 256;
            conv2_weight[i] = (i * 13 + 7) % 256;
        end
        // FC: 64 data/weight pairs
        for (int i = 0; i < 64; i++) begin
            fc_data[i]   = (i * 3 + 29) % 256;
            fc_weight[i] = (i * 9 + 41) % 256;
        end

        // Initialize
        reset = 1; data_in = 0; weight_in = 0;
        enable = 0; clear_acc = 0;
        repeat(3) @(posedge clock);
        reset = 0;
        @(posedge clock);

        // ============================================================
        // Test 1: Single MAC (42 * 12 = 504)
        // ============================================================
        $display("\n=== Test 1: Single MAC ===");
        clear_acc = 1; @(posedge clock); clear_acc = 0;
        data_in = 8'd42; weight_in = 8'd12;
        enable = 1; @(posedge clock);
        enable = 0; @(posedge clock);
        check_result("Single MAC (42*12)", 504);

        // ============================================================
        // Test 2: Conv1 kernel dot product (3x3x3 = 27 MACs)
        // ============================================================
        $display("\n=== Test 2: Conv1 Dot Product (27 MACs) ===");
        clear_acc = 1; @(posedge clock); clear_acc = 0;
        enable = 1;
        for (int i = 0; i < 27; i++) begin
            data_in = conv1_data[i];
            weight_in = conv1_weight[i];
            @(posedge clock);
        end
        enable = 0; @(posedge clock);

        // Compute expected: sum of (data[i] * weight[i]) for i=0..26
        begin
            int exp;
            exp = 0;
            for (int i = 0; i < 27; i++)
                exp = exp + conv1_data[i] * conv1_weight[i];
            check_result("Conv1 dot product (27 MACs)", exp);
        end

        // ============================================================
        // Test 3: Conv2 kernel dot product (3x3x16 = 144 MACs)
        // ============================================================
        $display("\n=== Test 3: Conv2 Dot Product (144 MACs) ===");
        clear_acc = 1; @(posedge clock); clear_acc = 0;
        enable = 1;
        for (int i = 0; i < 144; i++) begin
            data_in = conv2_data[i];
            weight_in = conv2_weight[i];
            @(posedge clock);
        end
        enable = 0; @(posedge clock);

        begin
            int exp;
            exp = 0;
            for (int i = 0; i < 144; i++)
                exp = exp + conv2_data[i] * conv2_weight[i];
            check_result("Conv2 dot product (144 MACs)", exp);
        end

        // ============================================================
        // Test 4: FC dot product (64 elements)
        // ============================================================
        $display("\n=== Test 4: FC Dot Product (64 MACs) ===");
        clear_acc = 1; @(posedge clock); clear_acc = 0;
        enable = 1;
        for (int i = 0; i < 64; i++) begin
            data_in = fc_data[i];
            weight_in = fc_weight[i];
            @(posedge clock);
        end
        enable = 0; @(posedge clock);

        begin
            int exp;
            exp = 0;
            for (int i = 0; i < 64; i++)
                exp = exp + fc_data[i] * fc_weight[i];
            check_result("FC dot product (64 MACs)", exp);
        end

        // ============================================================
        // Test 5: Back-to-back layers (Conv1 then Conv2 without reset)
        // Simulates layer sequencing on the FPGA
        // ============================================================
        $display("\n=== Test 5: Layer Sequencing (Conv1 -> clear -> Conv2) ===");
        // Conv1
        clear_acc = 1; @(posedge clock); clear_acc = 0;
        enable = 1;
        for (int i = 0; i < 27; i++) begin
            data_in = conv1_data[i];
            weight_in = conv1_weight[i];
            @(posedge clock);
        end
        enable = 0; @(posedge clock);
        begin
            int exp1;
            exp1 = 0;
            for (int i = 0; i < 27; i++)
                exp1 = exp1 + conv1_data[i] * conv1_weight[i];
            check_result("Layer seq: Conv1 result", exp1);
        end

        // Clear and immediately start Conv2
        clear_acc = 1; @(posedge clock); clear_acc = 0;
        enable = 1;
        for (int i = 0; i < 144; i++) begin
            data_in = conv2_data[i];
            weight_in = conv2_weight[i];
            @(posedge clock);
        end
        enable = 0; @(posedge clock);
        begin
            int exp2;
            exp2 = 0;
            for (int i = 0; i < 144; i++)
                exp2 = exp2 + conv2_data[i] * conv2_weight[i];
            check_result("Layer seq: Conv2 result", exp2);
        end

        // ============================================================
        // Summary
        // ============================================================
        $display("\n========================================");
        $display("  RTL vs Python Comparison Summary");
        $display("========================================");
        $display("  Tests Passed: %0d / %0d", test_passed, test_passed + test_failed);
        if (test_failed == 0)
            $display("  ALL TESTS PASSED");
        else
            $display("  %0d TESTS FAILED", test_failed);
        $display("========================================\n");

        $fclose(log_file);
        $finish;
    end

    // Waveform dump
    initial begin
        $dumpfile("rtl_vs_python.vcd");
        $dumpvars(0, tb_rtl_vs_python);
    end

endmodule
