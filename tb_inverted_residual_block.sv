//==============================================================================
// Testbench: tb_inverted_residual_block
// Description: Improved testbench with accurate data feeding sequences
//              Properly streams data to pointwise engines in batches
//
// Author: Cognichip Co-Design Team
//==============================================================================

module tb_inverted_residual_block;

    //==========================================================================
    // Parameters
    //==========================================================================
    
    parameter int NUM_MACS = 16;
    
    //==========================================================================
    // Testbench Signals
    //==========================================================================
    
    logic        clock;
    logic        reset;
    logic [9:0]  input_channels;
    logic [9:0]  expand_channels;
    logic [9:0]  output_channels;
    logic        use_residual;
    logic [7:0]  input_activation [NUM_MACS-1:0];
    logic        input_valid;
    logic [7:0]  dw_window [8:0];
    logic        dw_window_valid;
    logic [7:0]  expand_weights [NUM_MACS-1:0];
    logic [7:0]  dw_kernel_weights [8:0];
    logic [7:0]  project_weights [NUM_MACS-1:0];
    logic        start_block;
    logic        load_expand_data;
    logic        load_project_data;
    logic [7:0]  output_activation;
    logic        output_valid;
    logic        busy;
    
    // Test tracking
    int test_passed;
    int test_failed;
    
    //==========================================================================
    // Clock Generation
    //==========================================================================
    
    initial begin
        clock = 0;
        forever #5 clock = ~clock;
    end
    
    //==========================================================================
    // DUT Instantiation
    //==========================================================================
    
    inverted_residual_block #(
        .NUM_MACS(NUM_MACS)
    ) dut (
        .clock              (clock),
        .reset              (reset),
        .input_channels     (input_channels),
        .expand_channels    (expand_channels),
        .output_channels    (output_channels),
        .use_residual       (use_residual),
        .input_activation   (input_activation),
        .input_valid        (input_valid),
        .dw_window          (dw_window),
        .dw_window_valid    (dw_window_valid),
        .expand_weights     (expand_weights),
        .dw_kernel_weights  (dw_kernel_weights),
        .project_weights    (project_weights),
        .start_block        (start_block),
        .load_expand_data   (load_expand_data),
        .load_project_data  (load_project_data),
        .output_activation  (output_activation),
        .output_valid       (output_valid),
        .busy               (busy)
    );
    
    //==========================================================================
    // Helper Task: Feed Expansion Data
    //==========================================================================
    
    task automatic feed_expansion_data(input int num_input_ch);
        int num_batches;
        int batch;
        
        num_batches = (num_input_ch + NUM_MACS - 1) / NUM_MACS;
        $display("[TB] %0d batches for %0d channels", 
                 num_batches, num_input_ch);
        
        for (batch = 0; batch < num_batches; batch++) begin
            // Load activations and weights for this batch
            for (int i = 0; i < NUM_MACS; i++) begin
                input_activation[i] = 8'd10 + (batch * NUM_MACS + i);
                expand_weights[i] = 8'd1;
            end
            
            // Pulse load_expand_data
            @(posedge clock);
            load_expand_data = 1;
            @(posedge clock);
            load_expand_data = 0;
            
            // Wait one cycle between batches
            @(posedge clock);
        end
    endtask
    
    //==========================================================================
    // Helper Task: Feed Projection Data
    //==========================================================================
    
    task automatic feed_projection_data(input int num_expand_ch);
        int num_batches;
        int batch;
        
        num_batches = (num_expand_ch + NUM_MACS - 1) / NUM_MACS;
        $display("[TB]ches for %0d channels", 
                 num_batches, num_expand_ch);
        
        for (batch = 0; batch < num_batches; batch++) begin
            // Load activations (would be from depthwise in real system)
            for (int i = 0; i < NUM_MACS; i++) begin
                input_activation[i] = 8'd5 + (batch * NUM_MACS + i);
                project_weights[i] = 8'd1;
            end
            
            // Pulse load_project_data
            @(posedge clock);
            load_project_data = 1;
            @(posedge clock);
            load_project_data = 0;
            
            // Wait one cycle between batches
            @(posedge clock);
        end
    endtask
    
    //==========================================================================
    // Background Task: Provide Depthwise Window
    //==========================================================================
    
    initial begin
        dw_window_valid = 0;
        for (int i = 0; i < 9; i++) begin
            dw_window[i] = 8'd5;
            dw_kernel_weights[i] = 8'd1;
        end
        
        // Wait for reset to complete
        wait(reset == 0);
        
        // Keep window valid throughout test
        forever begin
            @(posedge clock);
            dw_window_valid = (busy == 1);  // Valid when block is busy
        end
    end
    
    //==========================================================================
    // Main Test Stimulus
    //==========================================================================
    
    initial begin
        $display("TEST START");
        $display("========================================");
        $display("  Inverted Residual Block Testbench");
        $display("========================================\n");
        
        // Initialize
        test_passed = 0;
        test_failed = 0;
        
        reset = 1;
        start_block = 0;
        input_valid = 0;
        dw_window_valid = 0;
        load_expand_data = 0;
        load_project_data = 0;
        use_residual = 0;
        
        input_channels = 10'd16;
        expand_channels = 10'd32;
        output_channels = 10'd16;
        
        // Initialize arrays
        for (int i = 0; i < NUM_MACS; i++) begin
            input_activation[i] = 8'd10;
            expand_weights[i] = 8'd1;
            project_weights[i] = 8'd1;
        end
        
        for (int i = 0; i < 9; i++) begin
            dw_window[i] = 8'd5;
            dw_kernel_weights[i] = 8'd1;
        end
        
        repeat(5) @(posedge clock);
        reset = 0;
        @(posedge clock);
        
        //======================================================================
        // Test 1: Complete Block Execution with Data Feeding
        //======================================================================
        $display("\n=== Test 1: Complete Block Execution (16→32→16) ===");
        
        // Start the block
        start_block = 1;
        input_valid = 1;
        @(posedge clock);
        start_block = 0;
        
        // Wait for busy
        wait(busy == 1);
        $display("LOG: %0t : INFO : tb_inverted_residual_block : Block started", $time);
        test_passed++;
        
        // Monitor and feed data based on internal state
        fork
            begin
                // Feed expansion data
                repeat(3) @(posedge clock);  // Wait for expansion to start
                feed_expansion_data(input_channels);
            end
            
            begin
                // Wait for depthwise to complete (indicated by internal signals)
                repeat(50) @(posedge clock);
                
                // Feed projection data
                feed_projection_data(expand_channels);
            end
            
            begin
                // Timeout
                repeat(500) @(posedge clock);
                if (output_valid != 1'b1) begin
                    $display("LOG: %0t : ERROR : Test 1 timeout", $time);
                    test_failed++;
                    $finish;
                end
            end
            
            begin
                // Wait for completion
                wait(output_valid == 1'b1);
                $display("LOG: %0t : INFO : tb_inverted_residual_block : Block completed successfully", $time);
                $display("Output activation: %0d", output_activation);
                test_passed++;
            end
        join_any
        disable fork;
        
        repeat(10) @(posedge clock);
        
        //======================================================================
        // Test 2: Smaller Configuration (8→16→8)
        //======================================================================
        $display("\n=== Test 2: Smaller Block (8→16→8) ===");
        
        input_channels = 10'd8;
        expand_channels = 10'd16;
        output_channels = 10'd8;
        
        start_block = 1;
        input_valid = 1;
        @(posedge clock);
        start_block = 0;
        
        wait(busy == 1);
        $display("LOG: %0t : INFO : Test 2 started", $time);
        
        fork
            begin
                repeat(3) @(posedge clock);
                feed_expansion_data(input_channels);
            end
            
            begin
                repeat(40) @(posedge clock);
                feed_projection_data(expand_channels);
            end
            
            begin
                repeat(400) @(posedge clock);
                if (output_valid != 1'b1) begin
                    $display("LOG: %0t : ERROR : Test 2 timeout", $time);
                    test_failed++;
                    $finish;
                end
            end
            
            begin
                wait(output_valid == 1'b1);
                $display("LOG: %0t : INFO : Test 2 completed", $time);
                $display("Output: %0d", output_activation);
                test_passed++;
            end
        join_any
        disable fork;
        
        repeat(10) @(posedge clock);
        
        //======================================================================
        // Final Report
        //======================================================================
        
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
        #2000000;  // 2ms timeout
        $display("\n========================================");
        $display("ERROR: Global simulation timeout!");
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
