//==============================================================================
// Testbench: tb_depthwise_conv3x3_engine
// Description: Comprehensive testbench for Depthwise 3×3 Convolution Engine
//              Tests various convolution patterns, edge cases, and validates
//              the parallel MAC operation and summation tree
//
// Author: Cognichip Co-Design Team
//==============================================================================

module tb_depthwise_conv3x3_engine;

    //==========================================================================
    // Testbench Signals
    //==========================================================================
    
    logic        clock;
    logic        reset;
    logic [7:0]  window_in [8:0];
    logic [7:0]  kernel_weights [8:0];
    logic        start_conv;
    logic        clear;
    logic [31:0] conv_result;
    logic        result_valid;
    
    // Testbench variables
    int          test_passed;
    int          test_failed;
    logic [31:0] expected_result;
    int          golden_sum;
    
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
    
    depthwise_conv3x3_engine dut (
        .clock          (clock),
        .reset          (reset),
        .window_in      (window_in),
        .kernel_weights (kernel_weights),
        .start_conv     (start_conv),
        .clear          (clear),
        .conv_result    (conv_result),
        .result_valid   (result_valid)
    );
    
    //==========================================================================
    // Golden Reference Model
    //==========================================================================
    
    function automatic int compute_golden_conv(
        input logic [7:0] window [8:0],
        input logic [7:0] kernel [8:0]
    );
        int sum;
        sum = 0;
        for (int i = 0; i < 9; i++) begin
            sum += window[i] * kernel[i];
        end
        return sum;
    endfunction
    
    //==========================================================================
    // Helper Task: Load Window and Kernel
    //==========================================================================
    
    task automatic load_test_data(
        input logic [7:0] test_window [8:0],
        input logic [7:0] test_kernel [8:0]
    );
        for (int i = 0; i < 9; i++) begin
            window_in[i] = test_window[i];
            kernel_weights[i] = test_kernel[i];
        end
    endtask
    
    //==========================================================================
    // Helper Task: Run Convolution and Check Result
    //==========================================================================
    
    task automatic run_conv_and_check(
        input string test_name,
        input int expected
    );
        // Clear accumulators before each test
        clear = 1;
        @(posedge clock);
        clear = 0;
        @(posedge clock);
        
        // Trigger convolution
        start_conv = 1;
        @(posedge clock);
        start_conv = 0;
        
        // Wait for result to be valid
        wait(result_valid == 1'b1);
        @(posedge clock);
        
        // Check result
        if (conv_result == expected) begin
            $display("LOG: %0t : INFO : tb_depthwise_conv3x3_engine : dut.conv_result : expected_value: %0d actual_value: %0d", 
                     $time, expected, conv_result);
            $display("[PASS] %s", test_name);
            test_passed++;
        end else begin
            $display("LOG: %0t : ERROR : tb_depthwise_conv3x3_engine : dut.conv_result : expected_value: %0d actual_value: %0d", 
                     $time, expected, conv_result);
            $display("[FAIL] %s - Expected: %0d, Got: %0d", test_name, expected, conv_result);
            test_failed++;
        end
        
        // Wait a cycle
        @(posedge clock);
    endtask
    
    //==========================================================================
    // Test Stimulus and Checking
    //==========================================================================
    
    initial begin
        $display("TEST START");
        $display("========================================");
        $display("  Depthwise Conv 3x3 Engine Testbench");
        $display("========================================\n");
        
        // Initialize counters
        test_passed = 0;
        test_failed = 0;
        
        // Initialize signals
        reset = 1;
        start_conv = 0;
        clear = 0;
        
        // Initialize arrays
        for (int i = 0; i < 9; i++) begin
            window_in[i] = 8'b0;
            kernel_weights[i] = 8'b0;
        end
        
        // Wait for reset
        repeat(3) @(posedge clock);
        reset = 0;
        @(posedge clock);
        
        //======================================================================
        // Test 1: Simple Identity Kernel
        //======================================================================
        $display("\n=== Test 1: Identity Kernel (center weight = 1) ===");
        // Window: all 5s, Kernel: center=1, rest=0
        // Expected: 5 * 1 = 5
        begin
            static logic [7:0] test_win [9] = '{5, 5, 5, 5, 5, 5, 5, 5, 5};
            static logic [7:0] test_ker [9] = '{0, 0, 0, 0, 1, 0, 0, 0, 0};
            load_test_data(test_win, test_ker);
            golden_sum = compute_golden_conv(test_win, test_ker);
            run_conv_and_check("Identity Kernel", golden_sum);
        end
        
        //======================================================================
        // Test 2: Box Blur (All 1s)
        //======================================================================
        $display("\n=== Test 2: Box Blur Kernel (all 1s) ===");
        // Window: [10,20,30; 40,50,60; 70,80,90]
        // Kernel: all 1s
        // Expected: 10+20+30+40+50+60+70+80+90 = 450
        begin
            static logic [7:0] test_win [9] = '{10, 20, 30, 40, 50, 60, 70, 80, 90};
            static logic [7:0] test_ker [9] = '{1, 1, 1, 1, 1, 1, 1, 1, 1};
            load_test_data(test_win, test_ker);
            golden_sum = compute_golden_conv(test_win, test_ker);
            run_conv_and_check("Box Blur", golden_sum);
        end
        
        //======================================================================
        // Test 3: Edge Detection Kernel
        //======================================================================
        $display("\n=== Test 3: Edge Detection (Sobel-like) ===");
        // Window: [100, 100, 100; 100, 100, 100; 0, 0, 0]
        // Kernel: [-1, 0, 1; -2, 0, 2; -1, 0, 1] (Sobel X)
        // Using unsigned: map negatives to high values or use 0
        // Simplified: [0, 0, 1; 0, 0, 2; 0, 0, 1]
        // Expected: 100*1 + 100*2 + 0*1 = 300
        begin
            static logic [7:0] test_win [9] = '{100, 100, 100, 100, 100, 100, 0, 0, 0};
            static logic [7:0] test_ker [9] = '{0, 0, 1, 0, 0, 2, 0, 0, 1};
            load_test_data(test_win, test_ker);
            golden_sum = compute_golden_conv(test_win, test_ker);
            run_conv_and_check("Edge Detection", golden_sum);
        end
        
        //======================================================================
        // Test 4: MobileNet-Style Kernel
        //======================================================================
        $display("\n=== Test 4: MobileNet-Style Depthwise Kernel ===");
        // Realistic MobileNet pattern
        // Window: [10, 20, 30; 40, 50, 60; 70, 80, 90]
        // Kernel: [1, 0, 1; 0, 2, 0; 1, 0, 1]
        // Expected: 10*1 + 30*1 + 50*2 + 70*1 + 90*1 = 10+30+100+70+90 = 300
        begin
            static logic [7:0] test_win [9] = '{10, 20, 30, 40, 50, 60, 70, 80, 90};
            static logic [7:0] test_ker [9] = '{1, 0, 1, 0, 2, 0, 1, 0, 1};
            load_test_data(test_win, test_ker);
            golden_sum = compute_golden_conv(test_win, test_ker);
            run_conv_and_check("MobileNet-Style", golden_sum);
        end
        
        //======================================================================
        // Test 5: All Zeros (No Operation)
        //======================================================================
        $display("\n=== Test 5: Zero Kernel (should output 0) ===");
        begin
            static logic [7:0] test_win [9] = '{50, 60, 70, 80, 90, 100, 110, 120, 130};
            static logic [7:0] test_ker [9] = '{0, 0, 0, 0, 0, 0, 0, 0, 0};
            load_test_data(test_win, test_ker);
            golden_sum = compute_golden_conv(test_win, test_ker);
            run_conv_and_check("Zero Kernel", golden_sum);
        end
        
        //======================================================================
        // Test 6: Maximum Values
        //======================================================================
        $display("\n=== Test 6: Maximum uint8 Values ===");
        // Window: all 255, Kernel: all 1s
        // Expected: 9 * 255 * 1 = 2295
        begin
            static logic [7:0] test_win [9] = '{255, 255, 255, 255, 255, 255, 255, 255, 255};
            static logic [7:0] test_ker [9] = '{1, 1, 1, 1, 1, 1, 1, 1, 1};
            load_test_data(test_win, test_ker);
            golden_sum = compute_golden_conv(test_win, test_ker);
            run_conv_and_check("Maximum Values", golden_sum);
        end
        
        //======================================================================
        // Test 7: Large Accumulation
        //======================================================================
        $display("\n=== Test 7: Large Accumulation Test ===");
        // Window: all 200, Kernel: all 100
        // Expected: 9 * 200 * 100 = 180,000
        begin
            static logic [7:0] test_win [9] = '{200, 200, 200, 200, 200, 200, 200, 200, 200};
            static logic [7:0] test_ker [9] = '{100, 100, 100, 100, 100, 100, 100, 100, 100};
            load_test_data(test_win, test_ker);
            golden_sum = compute_golden_conv(test_win, test_ker);
            run_conv_and_check("Large Accumulation", golden_sum);
        end
        
        //======================================================================
        // Test 8: Alternating Pattern
        //======================================================================
        $display("\n=== Test 8: Checkerboard Pattern ===");
        // Window: checkerboard, Kernel: inverse checkerboard
        // Expected: sum of products
        begin
            static logic [7:0] test_win [9] = '{100, 0, 100, 0, 100, 0, 100, 0, 100};
            static logic [7:0] test_ker [9] = '{0, 50, 0, 50, 0, 50, 0, 50, 0};
            load_test_data(test_win, test_ker);
            golden_sum = compute_golden_conv(test_win, test_ker);
            run_conv_and_check("Checkerboard", golden_sum);
        end
        
        //======================================================================
        // Test 9: Sequential Convolutions (No Clear Between)
        //======================================================================
        $display("\n=== Test 9: Sequential Convolutions ===");
        
        // First convolution
        begin
            static logic [7:0] test_win [9] = '{1, 2, 3, 4, 5, 6, 7, 8, 9};
            static logic [7:0] test_ker [9] = '{1, 1, 1, 1, 1, 1, 1, 1, 1};
            load_test_data(test_win, test_ker);
            golden_sum = compute_golden_conv(test_win, test_ker);
            run_conv_and_check("Sequential Conv #1", golden_sum);
        end
        
        // Second convolution immediately after
        begin
            static logic [7:0] test_win [9] = '{10, 10, 10, 10, 10, 10, 10, 10, 10};
            static logic [7:0] test_ker [9] = '{2, 2, 2, 2, 2, 2, 2, 2, 2};
            load_test_data(test_win, test_ker);
            golden_sum = compute_golden_conv(test_win, test_ker);
            run_conv_and_check("Sequential Conv #2", golden_sum);
        end
        
        //======================================================================
        // Test 10: Clear Functionality
        //======================================================================
        $display("\n=== Test 10: Clear Signal Test ===");
        
        // Load data but clear immediately
        begin
            static logic [7:0] test_win [9] = '{50, 50, 50, 50, 50, 50, 50, 50, 50};
            static logic [7:0] test_ker [9] = '{10, 10, 10, 10, 10, 10, 10, 10, 10};
            load_test_data(test_win, test_ker);
        end
        
        clear = 1;
        @(posedge clock);
        clear = 0;
        @(posedge clock);
        @(posedge clock);
        
        // Result should be 0 after clear (no start_conv was asserted)
        if (conv_result == 32'b0) begin
            $display("LOG: %0t : INFO : tb_depthwise_conv3x3_engine : dut.conv_result : expected_value: 0 actual_value: %0d", 
                     $time, conv_result);
            $display("[PASS] Clear Functionality");
            test_passed++;
        end else begin
            $display("LOG: %0t : ERROR : tb_depthwise_conv3x3_engine : dut.conv_result : expected_value: 0 actual_value: %0d", 
                     $time, conv_result);
            $display("[FAIL] Clear Functionality");
            test_failed++;
        end
        
        //======================================================================
        // Test 11: Reset During Operation
        //======================================================================
        $display("\n=== Test 11: Reset During Convolution ===");
        
        // Start a convolution
        begin
            static logic [7:0] test_win [9] = '{100, 100, 100, 100, 100, 100, 100, 100, 100};
            static logic [7:0] test_ker [9] = '{5, 5, 5, 5, 5, 5, 5, 5, 5};
            load_test_data(test_win, test_ker);
        end
        
        start_conv = 1;
        @(posedge clock);
        start_conv = 0;
        
        // Apply reset before result is valid
        reset = 1;
        @(posedge clock);
        reset = 0;
        @(posedge clock);
        @(posedge clock);
        
        // Result should be 0 after reset
        if (conv_result == 32'b0 && result_valid == 1'b0) begin
            $display("LOG: %0t : INFO : tb_depthwise_conv3x3_engine : dut.conv_result : expected_value: 0 actual_value: %0d", 
                     $time, conv_result);
            $display("[PASS] Reset During Operation");
            test_passed++;
        end else begin
            $display("LOG: %0t : ERROR : tb_depthwise_conv3x3_engine : dut.conv_result : expected_value: 0 actual_value: %0d", 
                     $time, conv_result);
            $display("[FAIL] Reset During Operation");
            test_failed++;
        end
        
        //======================================================================
        // Test 12: Real MobileNetV2-Style Computation
        //======================================================================
        $display("\n=== Test 12: Realistic MobileNetV2 Layer Simulation ===");
        // Simulate actual quantized values from a trained model
        begin
            static logic [7:0] test_win [9] = '{128, 135, 142, 130, 140, 145, 125, 138, 150};
            static logic [7:0] test_ker [9] = '{1, 2, 1, 2, 4, 2, 1, 2, 1};  // Gaussian-like
            load_test_data(test_win, test_ker);
            golden_sum = compute_golden_conv(test_win, test_ker);
            run_conv_and_check("MobileNetV2 Realistic", golden_sum);
        end
        
        //======================================================================
        // Final Report
        //======================================================================
        
        repeat(5) @(posedge clock);
        
        $display("\n========================================");
        $display("        TEST SUMMARY");
        $display("========================================");
        $display("Tests Passed: %0d", test_passed);
        $display("Tests Failed: %0d", test_failed);
        $display("Total Tests:  %0d", test_passed + test_failed);
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
    // Timeout Watchdog
    //==========================================================================
    
    initial begin
        #100000;  // 100us timeout
        $display("\n========================================");
        $display("ERROR: Simulation timeout!");
        $display("========================================\n");
        $display("ERROR");
        $error("TEST FAILED - Simulation timeout");
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
