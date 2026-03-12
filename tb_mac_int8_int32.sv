//==============================================================================
// Testbench: tb_mac_int8_int32
// Tests signed int8 × int8 multiply-accumulate, including negative weights.
//==============================================================================

module tb_mac_int8_int32;

    logic        clock, reset;
    logic signed [7:0]  data_in, weight_in;
    logic        enable, clear_acc;
    logic signed [31:0] acc_out;
    logic        valid;

    int passed = 0;
    int failed = 0;

    initial clock = 0;
    always #5 clock = ~clock;   // 100 MHz

    mac_int8_int32 dut (.*);

    // Helper task: apply one MAC cycle
    task mac_cycle(input logic signed [7:0] d, w);
        data_in = d; weight_in = w;
        enable = 1; clear_acc = 0;
        @(posedge clock); #1;
        enable = 0;
    endtask

    // Helper task: check result
    task check(input string name, input logic signed [31:0] expected);
        if (acc_out === expected) begin
            $display("PASS [%s] acc_out = %0d", name, acc_out);
            passed++;
        end else begin
            $display("FAIL [%s] expected %0d, got %0d", name, expected, acc_out);
            failed++;
        end
    endtask

    initial begin
        // Init
        reset = 1; enable = 0; clear_acc = 0;
        data_in = 0; weight_in = 0;
        @(posedge clock); @(posedge clock); #1;
        reset = 0;

        //----------------------------------------------------------------------
        // Test 1: Positive × Positive  (same as before)
        //----------------------------------------------------------------------
        clear_acc = 1; @(posedge clock); #1; clear_acc = 0;
        mac_cycle(8'sd10, 8'sd5);   // 10 × 5 = 50
        check("Pos×Pos", 32'sd50);

        //----------------------------------------------------------------------
        // Test 2: Negative weight × Positive activation
        //----------------------------------------------------------------------
        clear_acc = 1; @(posedge clock); #1; clear_acc = 0;
        mac_cycle(8'sd4, -8'sd3);   // 4 × (-3) = -12
        check("Pos×Neg", -32'sd12);

        //----------------------------------------------------------------------
        // Test 3: Negative × Negative = Positive
        //----------------------------------------------------------------------
        clear_acc = 1; @(posedge clock); #1; clear_acc = 0;
        mac_cycle(-8'sd7, -8'sd6);  // (-7) × (-6) = 42
        check("Neg×Neg", 32'sd42);

        //----------------------------------------------------------------------
        // Test 4: Accumulate across multiple MACs (like a real dot product)
        //   Dot product of [3, -2, 5] · [1, -4, 2] = 3 + 8 + 10 = 21
        //----------------------------------------------------------------------
        clear_acc = 1; @(posedge clock); #1; clear_acc = 0;
        mac_cycle(8'sd3,  8'sd1);   // +3
        mac_cycle(-8'sd2, -8'sd4);  // +8
        mac_cycle(8'sd5,  8'sd2);   // +10
        check("DotProduct", 32'sd21);

        //----------------------------------------------------------------------
        // Test 5: Maximum positive: 127 × 127 = 16129
        //----------------------------------------------------------------------
        clear_acc = 1; @(posedge clock); #1; clear_acc = 0;
        mac_cycle(8'sd127, 8'sd127);
        check("MaxPos", 32'sd16129);

        //----------------------------------------------------------------------
        // Test 6: Most negative: -128 × 127 = -16256
        //----------------------------------------------------------------------
        clear_acc = 1; @(posedge clock); #1; clear_acc = 0;
        mac_cycle(-8'sd128, 8'sd127);
        check("MaxNeg", -32'sd16256);

        //----------------------------------------------------------------------
        // Test 7: Clear mid-accumulation
        //----------------------------------------------------------------------
        clear_acc = 1; @(posedge clock); #1; clear_acc = 0;
        mac_cycle(8'sd10, 8'sd10);  // +100
        // Now clear
        clear_acc = 1; @(posedge clock); #1; clear_acc = 0;
        check("ClearMid", 32'sd0);

        //----------------------------------------------------------------------
        // Test 8: Synchronous reset
        //----------------------------------------------------------------------
        clear_acc = 1; @(posedge clock); #1; clear_acc = 0;
        mac_cycle(8'sd20, 8'sd20);  // +400
        reset = 1; @(posedge clock); #1; reset = 0;
        check("SyncReset", 32'sd0);

        //----------------------------------------------------------------------
        // Summary
        //----------------------------------------------------------------------
        $display("\n========================================");
        $display("  PASSED: %0d / FAILED: %0d", passed, failed);
        $display("========================================");
        if (failed == 0)
            $display("  ALL TESTS PASSED");
        else
            $display("  SOME TESTS FAILED");

        $finish;
    end

endmodule
