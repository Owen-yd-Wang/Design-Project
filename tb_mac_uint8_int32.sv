//==============================================================================
// Testbench: tb_mac_uint8_int32
// Description: Comprehensive testbench for MAC unit verification
//              Tests dot product computation, clear, enable, and edge cases
//
// Author: Cognichip Co-Design Team
//==============================================================================

module tb_mac_uint8_int32;

    //==========================================================================
    // Testbench Signals
    //==========================================================================
    
    logic        clock;
    logic        reset;
    logic [7:0]  data_in;
    logic [7:0]  weight_in;
    logic        enable;
    logic        clear_acc;
    logic [31:0] acc_out;
    logic        valid;
    
    // Testbench variables
    int          test_passed;
    int          test_failed;
    logic [31:0] expected_result;
    
    //==========================================================================
    // Clock Generation (10ns period = 100MHz)
    //==========================================================================
    
    initial begin
        clock = 0;
        forever #5 clock = ~clock;
    end
    
    //==========================================================================
    // DUT Instantiation
    //==========================================================================
    
    mac_uint8_int32 dut (
        .clock      (clock),
        .reset      (reset),
        .data_in    (data_in),
        .weight_in  (weight_in),
        .enable     (enable),
        .clear_acc  (clear_acc),
        .acc_out    (acc_out),
        .valid      (valid)
    );
    
    //==========================================================================
    // Test Stimulus and Checking
    //==========================================================================
    
    initial begin
        $display("TEST START");
        
        // Initialize counters
        test_passed = 0;
        test_failed = 0;
        
        // Initialize signals
        reset = 1;
        data_in = 8'b0;
        weight_in = 8'b0;
        enable = 0;
        clear_acc = 0;
        expected_result = 32'b0;
        
        // Wait for a few cycles
        repeat(3) @(posedge clock);
        reset = 0;
        @(posedge clock);
        
        $display("\n=== Test 1: Basic MAC Operation (Simple Dot Product) ===");
        // Compute: 10*5 = 50
        clear_acc = 1;
        @(posedge clock);
        clear_acc = 0;
        
        data_in = 8'd10;
        weight_in = 8'd5;
        enable = 1;
        @(posedge clock);
        enable = 0;
        @(posedge clock);
        
        expected_result = 32'd50;
        if (acc_out == expected_result) begin
            $display("LOG: %0t : INFO : tb_mac_uint8_int32 : dut.acc_out : expected_value: %0d actual_value: %0d", $time, expected_result, acc_out);
            test_passed++;
        end else begin
            $display("LOG: %0t : ERROR : tb_mac_uint8_int32 : dut.acc_out : expected_value: %0d actual_value: %0d", $time, expected_result, acc_out);
            test_failed++;
        end
        
        $display("\n=== Test 2: Accumulation (Multiple MACs) ===");
        // Clear and compute: 3*2 + 4*5 + 6*3 = 6 + 20 + 18 = 44
        clear_acc = 1;
        @(posedge clock);
        clear_acc = 0;
        
        // First MAC: 3*2 = 6
        data_in = 8'd3;
        weight_in = 8'd2;
        enable = 1;
        @(posedge clock);
        
        // Second MAC: 4*5 = 20 (acc = 6 + 20 = 26)
        data_in = 8'd4;
        weight_in = 8'd5;
        enable = 1;
        @(posedge clock);
        
        // Third MAC: 6*3 = 18 (acc = 26 + 18 = 44)
        data_in = 8'd6;
        weight_in = 8'd3;
        enable = 1;
        @(posedge clock);
        
        enable = 0;
        @(posedge clock);
        
        expected_result = 32'd44;
        if (acc_out == expected_result) begin
            $display("LOG: %0t : INFO : tb_mac_uint8_int32 : dut.acc_out : expected_value: %0d actual_value: %0d", $time, expected_result, acc_out);
            test_passed++;
        end else begin
            $display("LOG: %0t : ERROR : tb_mac_uint8_int32 : dut.acc_out : expected_value: %0d actual_value: %0d", $time, expected_result, acc_out);
            test_failed++;
        end
        
        $display("\n=== Test 3: 3x3 Convolution Dot Product (Realistic CNN Example) ===");
        // Simulate a 3x3 kernel convolution at one spatial position
        // Image patch: [1,2,3; 4,5,6; 7,8,9]
        // Kernel:      [1,0,1; 0,1,0; 1,0,1]
        // Expected: 1*1 + 2*0 + 3*1 + 4*0 + 5*1 + 6*0 + 7*1 + 8*0 + 9*1 = 1+3+5+7+9 = 25
        
        clear_acc = 1;
        @(posedge clock);
        clear_acc = 0;
        
        // Position (0,0): data=1, weight=1
        data_in = 8'd1; weight_in = 8'd1; enable = 1; @(posedge clock);
        // Position (0,1): data=2, weight=0
        data_in = 8'd2; weight_in = 8'd0; enable = 1; @(posedge clock);
        // Position (0,2): data=3, weight=1
        data_in = 8'd3; weight_in = 8'd1; enable = 1; @(posedge clock);
        // Position (1,0): data=4, weight=0
        data_in = 8'd4; weight_in = 8'd0; enable = 1; @(posedge clock);
        // Position (1,1): data=5, weight=1
        data_in = 8'd5; weight_in = 8'd1; enable = 1; @(posedge clock);
        // Position (1,2): data=6, weight=0
        data_in = 8'd6; weight_in = 8'd0; enable = 1; @(posedge clock);
        // Position (2,0): data=7, weight=1
        data_in = 8'd7; weight_in = 8'd1; enable = 1; @(posedge clock);
        // Position (2,1): data=8, weight=0
        data_in = 8'd8; weight_in = 8'd0; enable = 1; @(posedge clock);
        // Position (2,2): data=9, weight=1
        data_in = 8'd9; weight_in = 8'd1; enable = 1; @(posedge clock);
        
        enable = 0;
        @(posedge clock);
        
        expected_result = 32'd25;
        if (acc_out == expected_result) begin
            $display("LOG: %0t : INFO : tb_mac_uint8_int32 : dut.acc_out : expected_value: %0d actual_value: %0d", $time, expected_result, acc_out);
            test_passed++;
        end else begin
            $display("LOG: %0t : ERROR : tb_mac_uint8_int32 : dut.acc_out : expected_value: %0d actual_value: %0d", $time, expected_result, acc_out);
            test_failed++;
        end
        
        $display("\n=== Test 4: Clear During Accumulation ===");
        // Start accumulating, then clear mid-way
        clear_acc = 1;
        @(posedge clock);
        clear_acc = 0;
        
        data_in = 8'd10;
        weight_in = 8'd10;
        enable = 1;
        @(posedge clock);  // acc = 100
        
        data_in = 8'd20;
        weight_in = 8'd5;
        enable = 1;
        @(posedge clock);  // acc = 100 + 100 = 200
        
        // Now clear
        clear_acc = 1;
        enable = 0;
        @(posedge clock);
        clear_acc = 0;
        @(posedge clock);
        
        expected_result = 32'd0;
        if (acc_out == expected_result) begin
            $display("LOG: %0t : INFO : tb_mac_uint8_int32 : dut.acc_out : expected_value: %0d actual_value: %0d", $time, expected_result, acc_out);
            test_passed++;
        end else begin
            $display("LOG: %0t : ERROR : tb_mac_uint8_int32 : dut.acc_out : expected_value: %0d actual_value: %0d", $time, expected_result, acc_out);
            test_failed++;
        end
        
        $display("\n=== Test 5: Enable Control (Hold) ===");
        // Start accumulating, disable enable, value should hold
        clear_acc = 1;
        @(posedge clock);
        clear_acc = 0;
        
        data_in = 8'd15;
        weight_in = 8'd4;
        enable = 1;
        @(posedge clock);  // acc = 60
        
        enable = 0;  // Disable, should hold
        data_in = 8'd99;  // Change data (shouldn't affect)
        weight_in = 8'd99;
        @(posedge clock);
        @(posedge clock);
        
        expected_result = 32'd60;
        if (acc_out == expected_result) begin
            $display("LOG: %0t : INFO : tb_mac_uint8_int32 : dut.acc_out : expected_value: %0d actual_value: %0d", $time, expected_result, acc_out);
            test_passed++;
        end else begin
            $display("LOG: %0t : ERROR : tb_mac_uint8_int32 : dut.acc_out : expected_value: %0d actual_value: %0d", $time, expected_result, acc_out);
            test_failed++;
        end
        
        $display("\n=== Test 6: Maximum Value Test ===");
        // Test with maximum uint8 values: 255 * 255 = 65,025
        clear_acc = 1;
        @(posedge clock);
        clear_acc = 0;
        
        data_in = 8'd255;
        weight_in = 8'd255;
        enable = 1;
        @(posedge clock);
        enable = 0;
        @(posedge clock);
        
        expected_result = 32'd65025;
        if (acc_out == expected_result) begin
            $display("LOG: %0t : INFO : tb_mac_uint8_int32 : dut.acc_out : expected_value: %0d actual_value: %0d", $time, expected_result, acc_out);
            test_passed++;
        end else begin
            $display("LOG: %0t : ERROR : tb_mac_uint8_int32 : dut.acc_out : expected_value: %0d actual_value: %0d", $time, expected_result, acc_out);
            test_failed++;
        end
        
        $display("\n=== Test 7: Large Accumulation (320 MACs - typical pointwise conv) ===");
        // Simulate pointwise conv with 320 input channels
        // Each MAC: 100 * 50 = 5,000
        // Total: 320 * 5,000 = 1,600,000
        clear_acc = 1;
        @(posedge clock);
        clear_acc = 0;
        
        data_in = 8'd100;
        weight_in = 8'd50;
        enable = 1;
        
        repeat(320) @(posedge clock);
        
        enable = 0;
        @(posedge clock);
        
        expected_result = 32'd1600000;
        if (acc_out == expected_result) begin
            $display("LOG: %0t : INFO : tb_mac_uint8_int32 : dut.acc_out : expected_value: %0d actual_value: %0d", $time, expected_result, acc_out);
            test_passed++;
        end else begin
            $display("LOG: %0t : ERROR : tb_mac_uint8_int32 : dut.acc_out : expected_value: %0d actual_value: %0d", $time, expected_result, acc_out);
            test_failed++;
        end
        
        $display("\n=== Test 8: Reset Functionality ===");
        // Accumulate some value, then reset
        clear_acc = 0;
        data_in = 8'd50;
        weight_in = 8'd50;
        enable = 1;
        @(posedge clock);
        
        // Apply reset
        reset = 1;
        @(posedge clock);
        reset = 0;
        @(posedge clock);
        
        expected_result = 32'd0;
        if (acc_out == expected_result && valid == 1'b0) begin
            $display("LOG: %0t : INFO : tb_mac_uint8_int32 : dut.acc_out : expected_value: %0d actual_value: %0d", $time, expected_result, acc_out);
            test_passed++;
        end else begin
            $display("LOG: %0t : ERROR : tb_mac_uint8_int32 : dut.acc_out : expected_value: %0d actual_value: %0d", $time, expected_result, acc_out);
            test_failed++;
        end
        
        //======================================================================
        // Final Report
        //======================================================================
        
        $display("\n========================================");
        $display("        TEST SUMMARY");
        $display("========================================");
        $display("Tests Passed: %0d", test_passed);
        $display("Tests Failed: %0d", test_failed);
        $display("========================================\n");
        
        if (test_failed == 0) begin
            $display("TEST PASSED");
        end else begin
            $display("ERROR");
            $error("TEST FAILED - %0d test(s) failed", test_failed);
        end
        
        $finish;
    end
    
    //==========================================================================
    // Waveform Dump
    //==========================================================================
    
    initial begin
        $dumpfile("dumpfile.fst");
        $dumpvars(0);
    end

endmodule
