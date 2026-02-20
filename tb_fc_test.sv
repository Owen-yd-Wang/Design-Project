// Minimal test for FC layer (512→1) using pointwise engine
module tb_fc_test;

    parameter int NUM_MACS = 16;
    logic clock, reset;
    logic [9:0] pw_in_ch, pw_out_ch;
    logic [7:0] pw_act [0:15], pw_wt [0:15];
    logic pw_start, pw_clear, pw_load;
    logic [31:0] pw_result;
    logic pw_valid, pw_busy;
    
    pointwise_conv1x1_engine #(.NUM_MACS(NUM_MACS)) pw (
        .clock(clock), .reset(reset),
        .num_input_channels(pw_in_ch), .num_output_channels(pw_out_ch),
        .activations(pw_act), .weights(pw_wt),
        .start_conv(pw_start), .clear(pw_clear), .load_data(pw_load),
        .conv_result(pw_result), .result_valid(pw_valid), .busy(pw_busy)
    );
    
    initial begin clock = 0; forever #5 clock = ~clock; end
    
    initial begin
        int batch, i, ch_idx;
        
        reset=1; pw_start=0; pw_clear=0; pw_load=0;
        pw_in_ch = 512; pw_out_ch = 1;
        
        repeat(5) @(posedge clock);
        reset=0; @(posedge clock);
        
        $display("Testing FC: 512 inputs -> 1 output");
        $display("32 batches of 16 MACs each\n");
        
        // Initialize test data
        for (i=0; i<16; i++) begin
            pw_act[i] = i+1;
            pw_wt[i] = 1;
        end
        
        // Start computation
        pw_clear=1; @(posedge clock); pw_clear=0;
        $display("t=%0t: Cleared", $time);
        
        pw_start=1; @(posedge clock); pw_start=0;
        $display("t=%0t: Started", $time);
        
        // Load 32 batches
        for (batch=0; batch<32; batch++) begin
            $display("t=%0t: Loading batch %0d", $time, batch);
            pw_load=1; @(posedge clock); pw_load=0;
            repeat(2) @(posedge clock);
        end
        
        $display("t=%0t: All batches loaded, waiting for result...", $time);
        
        // Wait for result with timeout
        fork
            begin
                wait(pw_valid);
                $display("t=%0t: Got result = %0d", $time, $signed(pw_result));
                $display("\n✅ TEST PASSED!");
            end
            begin
                #10000;
                $display("t=%0t: TIMEOUT - result_valid never pulsed!", $time);
                $display("❌ TEST FAILED - FC layer stuck!");
            end
        join_any
        
        repeat(5) @(posedge clock);
        $finish;
    end

endmodule
