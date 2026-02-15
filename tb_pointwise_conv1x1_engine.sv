//==============================================================================
// Testbench: tb_pointwise_conv1x1_engine
// Description: Comprehensive testbench for Pointwise 1×1 Convolution Engine
//              Tests various channel configurations, multi-cycle accumulation,
//              and validates the streaming data interface
//
// Author: Cognichip Co-Design Team
//==============================================================================

module tb_pointwise_conv1x1_engine;

    //==========================================================================
    // Testbench Parameters
    //==========================================================================
    
    parameter int NUM_MACS = 16;  // Match DUT configuration
    
    //==========================================================================
    // Testbench Signals
    //==========================================================================
    
    logic        clock;
    logic        reset;
    logic [9:0]  num_input_channels;
    logic [9:0]  num_output_channels;
    logic [7:0]  activations [NUM_MACS-1:0];
    logic [7:0]  weights [NUM_MACS-1:0];
    logic        start_conv;
    logic        clear;
    logic        load_data;
    logic [31:0] conv_result;
    logic        result_valid;
    logic        busy;
    
    // Testbench variables
    int          test_passed;
    int          test_failed;
    logic [31:0] expected_result;
    int          golden_sum;
    
    // Test data storage
    logic [7:0]  test_activations [319:0];  // Max 320 channels
    logic [7:0]  test_weights [319:0];      // Max 320 channels
    
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
    
    pointwise_conv1x1_engine #(
        .NUM_MACS(NUM_MACS)
    ) dut (
        .clock              (clock),
        .reset              (reset),
        .num_input_channels (num_input_channels),
        .num_output_channels(num_output_channels),
        .activations        (activations),
        .weights            (weights),
        .start_conv         (start_conv),
        .clear              (clear),
        .load_data          (load_data),
        .conv_result        (conv_result),
        .result_valid       (result_valid),
        .busy               (busy)
    );
    
    //==========================================================================
    // Golden Reference Model
    //==========================================================================
    
    function automatic int compute_golden_pointwise(
        input int num_channels,
        input logic [7:0] act [319:0],
        input logic [7:0] wt [319:0]
    );
        int sum;
        sum = 0;
        for (int i = 0; i < num_channels; i++) begin
            sum += act[i] * wt[i];
        end
        return sum;
    endfunction
    
    //==========================================================================
    // Helper Task: Load Test Data Arrays
    //==========================================================================
    
    task automatic load_test_arrays(
        input int num_ch,
        input logic [7:0] act_data [319:0],
        input logic [7:0] wt_data [319:0]
    );
        for (int i = 0; i < 320; i++) begin
            test_activations[i] = (i < num_ch) ? act_data[i] : 8'b0;
            test_weights[i] = (i < num_ch) ? wt_data[i] : 8'b0;
        end
    endtask
    
    //==========================================================================
    // Helper Task: Run Pointwise Convolution with Streaming
    //==========================================================================
    
    task automatic run_pointwise_conv(
        input string test_name,
        input int num_in_ch,
        input int expected
    );
        int num_batches;
        int batch;
        int ch_idx;
        
        // Calculate number of batches needed
        num_batches = (num_in_ch + NUM_MACS - 1) / NUM_MACS;
        
        // Configure
        num_input_channels = num_in_ch;
        num_output_channels = 1;  // Testing one output channel at a time
        
        @(posedge clock);
        
        // Start convolution
        start_conv = 1;
        @(posedge clock);
        start_conv = 0;
        
        // Stream data in batches
        for (batch = 0; batch < num_batches; batch++) begin
            // Load batch of activations and weights
            for (int i = 0; i < NUM_MACS; i++) begin
                ch_idx = batch * NUM_MACS + i;
                if (ch_idx < num_in_ch) begin
                    activations[i] = test_activations[ch_idx];
                    weights[i] = test_weights[ch_idx];
                end else begin
                    activations[i] = 8'b0;
                    weights[i] = 8'b0;
                end
            end
            
            // Assert load_data for one cycle
            load_data = 1;
            @(posedge clock);
            load_data = 0;
            
            // Wait a cycle between batches
            @(posedge clock);
        end
        
        // Wait for result
        wait(result_valid == 1'b1);
        @(posedge clock);
        
        // Check result
        if (conv_result == expected) begin
            $display("LOG: %0t : INFO : tb_pointwise_conv1x1_engine : dut.conv_result : expected_value: %0d actual_value: %0d", 
                     $time, expected, conv_result);
            $display("[PASS] %s (channels=%0d, batches=%0d)", test_name, num_in_ch, num_batches);
            test_passed++;
        end else begin
            $display("LOG: %0t : ERROR : tb_pointwise_conv1x1_engine : dut.conv_result : expected_value: %0d actual_value: %0d", 
                     $time, expected, conv_result);
            $display("[FAIL] %s - Expected: %0d, Got: %0d", test_name, expected, conv_result);
            test_failed++;
        end
        
        @(posedge clock);
    endtask
    
    //==========================================================================
    // Test Stimulus and Checking
    //==========================================================================
    
    initial begin
        $display("TEST START");
        $display("========================================");
        $display("  Pointwise Conv 1x1 Engine Testbench");
        $display("========================================\n");
        
        // Initialize counters
        test_passed = 0;
        test_failed = 0;
        
        // Initialize signals
        reset = 1;
        start_conv = 0;
        clear = 0;
        load_data = 0;
        num_input_channels = 0;
        num_output_channels = 0;
        
        // Initialize arrays
        for (int i = 0; i < NUM_MACS; i++) begin
            activations[i] = 8'b0;
            weights[i] = 8'b0;
        end
        
        for (int i = 0; i < 320; i++) begin
            test_activations[i] = 8'b0;
            test_weights[i] = 8'b0;
        end
        
        // Wait for reset
        repeat(3) @(posedge clock);
        reset = 0;
        @(posedge clock);
        
        //======================================================================
        // Test 1: Simple Single-Batch (4 channels)
        //======================================================================
        $display("\n=== Test 1: Single Batch - 4 Channels ===");
        // Inputs: [10, 20, 30, 40]
        // Weights: [1, 2, 1, 2]
        // Expected: 10*1 + 20*2 + 30*1 + 40*2 = 10+40+30+80 = 160
        begin
            static logic [7:0] act [320];
            static logic [7:0] wt [320];
            act[0] = 10; act[1] = 20; act[2] = 30; act[3] = 40;
            wt[0] = 1; wt[1] = 2; wt[2] = 1; wt[3] = 2;
            for (int i = 4; i < 320; i++) begin
                act[i] = 8'b0;
                wt[i] = 8'b0;
            end
            load_test_arrays(4, act, wt);
            golden_sum = compute_golden_pointwise(4, test_activations, test_weights);
            run_pointwise_conv("Single Batch 4ch", 4, golden_sum);
        end
        
        //======================================================================
        // Test 2: Exactly One Batch (16 channels)
        //======================================================================
        $display("\n=== Test 2: Exactly One Batch - 16 Channels ===");
        // All inputs = 5, all weights = 10
        // Expected: 16 * 5 * 10 = 800
        begin
            static logic [7:0] act [320];
            static logic [7:0] wt [320];
            for (int i = 0; i < 16; i++) begin
                act[i] = 8'd5;
                wt[i] = 8'd10;
            end
            for (int i = 16; i < 320; i++) begin
                act[i] = 8'b0;
                wt[i] = 8'b0;
            end
            load_test_arrays(16, act, wt);
            golden_sum = compute_golden_pointwise(16, test_activations, test_weights);
            run_pointwise_conv("Exactly One Batch 16ch", 16, golden_sum);
        end
        
        //======================================================================
        // Test 3: Two Batches (32 channels - typical MobileNetV2)
        //======================================================================
        $display("\n=== Test 3: Two Batches - 32 Channels ===");
        // Ascending pattern: [1, 2, 3, ..., 32]
        // Weights: all 1s
        // Expected: 1+2+3+...+32 = 32*33/2 = 528
        begin
            static logic [7:0] act [320];
            static logic [7:0] wt [320];
            for (int i = 0; i < 32; i++) begin
                act[i] = i + 1;
                wt[i] = 8'd1;
            end
            for (int i = 32; i < 320; i++) begin
                act[i] = 8'b0;
                wt[i] = 8'b0;
            end
            load_test_arrays(32, act, wt);
            golden_sum = compute_golden_pointwise(32, test_activations, test_weights);
            run_pointwise_conv("Two Batches 32ch", 32, golden_sum);
        end
        
        //======================================================================
        // Test 4: Multiple Batches (64 channels)
        //======================================================================
        $display("\n=== Test 4: Four Batches - 64 Channels ===");
        // All activations = 100, all weights = 2
        // Expected: 64 * 100 * 2 = 12,800
        begin
            static logic [7:0] act [320];
            static logic [7:0] wt [320];
            for (int i = 0; i < 64; i++) begin
                act[i] = 8'd100;
                wt[i] = 8'd2;
            end
            for (int i = 64; i < 320; i++) begin
                act[i] = 8'b0;
                wt[i] = 8'b0;
            end
            load_test_arrays(64, act, wt);
            golden_sum = compute_golden_pointwise(64, test_activations, test_weights);
            run_pointwise_conv("Four Batches 64ch", 64, golden_sum);
        end
        
        //======================================================================
        // Test 5: MobileNetV2 Expansion (32→192 channels)
        //======================================================================
        $display("\n=== Test 5: MobileNetV2 Expansion - 32 Channels ===");
        // Realistic expansion layer
        // Mix of values
        begin
            static logic [7:0] act [320];
            static logic [7:0] wt [320];
            for (int i = 0; i < 32; i++) begin
                act[i] = 8'd128 + (i % 10);  // Around 128
                wt[i] = 8'd1 + (i % 3);      // 1, 2, or 3
            end
            for (int i = 32; i < 320; i++) begin
                act[i] = 8'b0;
                wt[i] = 8'b0;
            end
            load_test_arrays(32, act, wt);
            golden_sum = compute_golden_pointwise(32, test_activations, test_weights);
            run_pointwise_conv("MobileNetV2 Expansion 32ch", 32, golden_sum);
        end
        
        //======================================================================
        // Test 6: Large Channel Count (128 channels)
        //======================================================================
        $display("\n=== Test 6: Many Batches - 128 Channels ===");
        // All activations = 50, all weights = 3
        // Expected: 128 * 50 * 3 = 19,200
        begin
            static logic [7:0] act [320];
            static logic [7:0] wt [320];
            for (int i = 0; i < 128; i++) begin
                act[i] = 8'd50;
                wt[i] = 8'd3;
            end
            for (int i = 128; i < 320; i++) begin
                act[i] = 8'b0;
                wt[i] = 8'b0;
            end
            load_test_arrays(128, act, wt);
            golden_sum = compute_golden_pointwise(128, test_activations, test_weights);
            run_pointwise_conv("Many Batches 128ch", 128, golden_sum);
        end
        
        //======================================================================
        // Test 7: Maximum MobileNetV2 Channels (320 channels)
        //======================================================================
        $display("\n=== Test 7: Maximum Channels - 320 ===");
        // All activations = 10, all weights = 5
        // Expected: 320 * 10 * 5 = 16,000
        begin
            static logic [7:0] act [320];
            static logic [7:0] wt [320];
            for (int i = 0; i < 320; i++) begin
                act[i] = 8'd10;
                wt[i] = 8'd5;
            end
            load_test_arrays(320, act, wt);
            golden_sum = compute_golden_pointwise(320, test_activations, test_weights);
            run_pointwise_conv("Maximum 320ch", 320, golden_sum);
        end
        
        //======================================================================
        // Test 8: Zero Weights
        //======================================================================
        $display("\n=== Test 8: Zero Weights ===");
        // Any activations, zero weights
        // Expected: 0
        begin
            static logic [7:0] act [320];
            static logic [7:0] wt [320];
            for (int i = 0; i < 32; i++) begin
                act[i] = 8'd255;  // Max values
                wt[i] = 8'd0;     // Zero weights
            end
            for (int i = 32; i < 320; i++) begin
                act[i] = 8'b0;
                wt[i] = 8'b0;
            end
            load_test_arrays(32, act, wt);
            golden_sum = compute_golden_pointwise(32, test_activations, test_weights);
            run_pointwise_conv("Zero Weights", 32, golden_sum);
        end
        
        //======================================================================
        // Test 9: Maximum Values
        //======================================================================
        $display("\n=== Test 9: Maximum uint8 Values ===");
        // All 255×255 for 16 channels
        // Expected: 16 * 255 * 255 = 1,040,400
        begin
            static logic [7:0] act [320];
            static logic [7:0] wt [320];
            for (int i = 0; i < 16; i++) begin
                act[i] = 8'd255;
                wt[i] = 8'd255;
            end
            for (int i = 16; i < 320; i++) begin
                act[i] = 8'b0;
                wt[i] = 8'b0;
            end
            load_test_arrays(16, act, wt);
            golden_sum = compute_golden_pointwise(16, test_activations, test_weights);
            run_pointwise_conv("Maximum Values", 16, golden_sum);
        end
        
        //======================================================================
        // Test 10: Odd Number of Channels (17)
        //======================================================================
        $display("\n=== Test 10: Odd Channel Count - 17 ===");
        // Tests partial batch handling
        begin
            static logic [7:0] act [320];
            static logic [7:0] wt [320];
            for (int i = 0; i < 17; i++) begin
                act[i] = 8'd10;
                wt[i] = 8'd10;
            end
            for (int i = 17; i < 320; i++) begin
                act[i] = 8'b0;
                wt[i] = 8'b0;
            end
            load_test_arrays(17, act, wt);
            golden_sum = compute_golden_pointwise(17, test_activations, test_weights);
            run_pointwise_conv("Odd Channels 17", 17, golden_sum);
        end
        
        //======================================================================
        // Test 11: Clear Signal Test
        //======================================================================
        $display("\n=== Test 11: Clear Signal Test ===");
        
        clear = 1;
        @(posedge clock);
        clear = 0;
        @(posedge clock);
        @(posedge clock);
        
        if (conv_result == 32'b0 && !busy) begin
            $display("LOG: %0t : INFO : tb_pointwise_conv1x1_engine : dut.conv_result : expected_value: 0 actual_value: %0d", 
                     $time, conv_result);
            $display("[PASS] Clear Functionality");
            test_passed++;
        end else begin
            $display("LOG: %0t : ERROR : tb_pointwise_conv1x1_engine : dut.conv_result : expected_value: 0 actual_value: %0d", 
                     $time, conv_result);
            $display("[FAIL] Clear Functionality");
            test_failed++;
        end
        
        //======================================================================
        // Test 12: Reset Test
        //======================================================================
        $display("\n=== Test 12: Reset During Operation ===");
        
        // Start a computation
        num_input_channels = 32;
        start_conv = 1;
        @(posedge clock);
        start_conv = 0;
        
        // Apply reset
        reset = 1;
        @(posedge clock);
        reset = 0;
        @(posedge clock);
        
        if (conv_result == 32'b0 && !busy && !result_valid) begin
            $display("LOG: %0t : INFO : tb_pointwise_conv1x1_engine : dut.conv_result : expected_value: 0 actual_value: %0d", 
                     $time, conv_result);
            $display("[PASS] Reset Functionality");
            test_passed++;
        end else begin
            $display("LOG: %0t : ERROR : tb_pointwise_conv1x1_engine : dut.conv_result : expected_value: 0 actual_value: %0d", 
                     $time, conv_result);
            $display("[FAIL] Reset Functionality");
            test_failed++;
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
        #500000;  // 500us timeout (longer for multi-batch tests)
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
