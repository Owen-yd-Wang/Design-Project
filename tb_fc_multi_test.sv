// Test FC with MULTIPLE outputs (512→64) like complete inference
module tb_fc_multi_test;

    parameter int NUM_MACS = 16;
    logic clock, reset;
    logic [9:0] pw_in_ch, pw_out_ch;
    logic [7:0] pw_act [0:15], pw_wt [0:15];
    logic pw_start, pw_clear, pw_load;
    logic [31:0] pw_result;
    logic pw_valid, pw_busy;
    logic [7:0] fm_fc1 [0:63];
    
    pointwise_conv1x1_engine #(.NUM_MACS(NUM_MACS)) pw (
        .clock(clock), .reset(reset),
        .num_input_channels(pw_in_ch), .num_output_channels(pw_out_ch),
        .activations(pw_act), .weights(pw_wt),
        .start_conv(pw_start), .clear(pw_clear), .load_data(pw_load),
        .conv_result(pw_result), .result_valid(pw_valid), .busy(pw_busy)
    );
    
    initial begin clock = 0; forever #5 clock = ~clock; end
    
    function automatic logic [7:0] relu(int val);
        if (val < 0) return 8'd0;
        if (val > 255) return 8'd255;
        return val[7:0];
    endfunction
    
    initial begin
        int batch, i, ch_idx, och;
        
        reset=1; pw_start=0; pw_clear=0; pw_load=0;
        repeat(5) @(posedge clock);
        reset=0; @(posedge clock);
        
        $display("Testing FC: 512 inputs -> 64 outputs");
        $display("Same pattern as tb_complete_inference FC1\n");
        
        pw_in_ch = 512;
        pw_out_ch = 1;
        
        // Process 64 outputs (EXACTLY like tb_complete_inference)
        for (och=0; och<64; och++) begin
            $display("t=%0t: Starting output channel %0d", $time, och);
            
            pw_clear=1; @(posedge clock); pw_clear=0;
            pw_start=1; @(posedge clock); pw_start=0;
            
            // Process in batches of 16
            for (batch=0; batch<32; batch++) begin
                for (i=0; i<16; i++) begin
                    ch_idx = batch*16 + i;
                    if (ch_idx < 512) begin
                        pw_act[i] = (i+1);
                        pw_wt[i] = 1;
                    end else begin
                        pw_act[i] = 8'h00;
                        pw_wt[i] = 8'h00;
                    end
                end
                pw_load=1; @(posedge clock); pw_load=0;
                repeat(2) @(posedge clock);
            end
            
            $display("t=%0t: Output %0d - waiting for result...", $time, och);
            wait(pw_valid); @(posedge clock);
            fm_fc1[och] = relu($signed(pw_result));
            $display("t=%0t: Output %0d complete, result=%0d\n", $time, och, fm_fc1[och]);

            reset = 1; @(posedge clock); reset=0;
        end
        
        $display("✅ TEST PASSED - All 64 outputs computed!");
        $display("Sample outputs: %0d, %0d, %0d, %0d", 
                 fm_fc1[0], fm_fc1[1], fm_fc1[2], fm_fc1[3]);
        
        repeat(5) @(posedge clock);
        $finish;
    end
    
    initial begin
        #100000;
        $display("TIMEOUT at t=%0t", $time);
        $finish;
    end

endmodule
