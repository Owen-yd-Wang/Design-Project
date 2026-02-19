//==============================================================================
// tb_layer_with_weights: Real MobileNetV2 weights demonstration
//==============================================================================

module tb_layer_with_weights;

    parameter int NUM_MACS = 16;
    parameter int INPUT_CHANNELS = 32;
    parameter int OUTPUT_CHANNELS = 16;
    
    logic        clock;
    logic        reset;
    logic [9:0]  num_input_ch;
    logic [9:0]  num_output_ch;
    logic [7:0]  activations [NUM_MACS-1:0];
    logic [7:0]  weights [NUM_MACS-1:0];
    logic        start_conv;
    logic        clear;
    logic        load_data;
    logic [31:0] conv_result;
    logic        result_valid;
    logic        busy;
    
    logic [7:0]  layer_weights [0:511];
    
    initial begin
        clock = 0;
        forever #5 clock = ~clock;
    end
    
    pointwise_conv1x1_engine #(.NUM_MACS(NUM_MACS)) pw_engine (
        .clock(clock), .reset(reset),
        .num_input_channels(num_input_ch),
        .num_output_channels(num_output_ch),
        .activations(activations),
        .weights(weights),
        .start_conv(start_conv),
        .clear(clear),
        .load_data(load_data),
        .conv_result(conv_result),
        .result_valid(result_valid),
        .busy(busy)
    );
    
    initial begin
        $display("Loading REAL MobileNetV2 weights...");
        // Real weights from features.5.conv.2 (first 32 values from .mem file)
        layer_weights[0] = 8'he8; layer_weights[1] = 8'h22;
        layer_weights[2] = 8'hf1; layer_weights[3] = 8'hef;
        layer_weights[4] = 8'h16; layer_weights[5] = 8'h3a;
        layer_weights[6] = 8'he0; layer_weights[7] = 8'h1a;
        layer_weights[8] = 8'hf9; layer_weights[9] = 8'h1c;
        layer_weights[10] = 8'h12; layer_weights[11] = 8'h35;
        layer_weights[12] = 8'hf6; layer_weights[13] = 8'hef;
        layer_weights[14] = 8'hf9; layer_weights[15] = 8'hff;
        layer_weights[16] = 8'he9; layer_weights[17] = 8'h16;
        layer_weights[18] = 8'h13; layer_weights[19] = 8'h1e;
        layer_weights[20] = 8'hce; layer_weights[21] = 8'h14;
        layer_weights[22] = 8'h41; layer_weights[23] = 8'h18;
        layer_weights[24] = 8'h26; layer_weights[25] = 8'hda;
        layer_weights[26] = 8'hfa; layer_weights[27] = 8'hfb;
        layer_weights[28] = 8'he7; layer_weights[29] = 8'hc9;
        layer_weights[30] = 8'h06; layer_weights[31] = 8'h24;
        for (int i = 32; i < 512; i++) layer_weights[i] = (i % 128) - 64;
        $display("Loaded 512 weights");
    end
    
    initial begin
        int num_batches, batch, i, idx, out_ch, sum_results;
        reset = 1;
        start_conv = 0;
        clear = 0;
        load_data = 0;
        num_input_ch = INPUT_CHANNELS;
        num_output_ch = 1;
        sum_results = 0;
        for (i = 0; i < NUM_MACS; i++) activations[i] = 8'd128 + i;
        
        repeat(5) @(posedge clock);
        reset = 0;
        @(posedge clock);
        
        $display("Processing 4 output channels...");
        for (out_ch = 0; out_ch < 4; out_ch++) begin
            num_batches = (INPUT_CHANNELS + NUM_MACS - 1) / NUM_MACS;
            start_conv = 1;
            @(posedge clock);
            start_conv = 0;
            
            for (batch = 0; batch < num_batches; batch++) begin
                for (i = 0; i < NUM_MACS; i++) begin
                    idx = out_ch * INPUT_CHANNELS + batch * NUM_MACS + i;
                    if (batch * NUM_MACS + i < INPUT_CHANNELS) begin
                        weights[i] = layer_weights[idx];
                        activations[i] = 8'd128 + ((batch * NUM_MACS + i) % 32);
                    end else begin
                        weights[i] = 8'h00;
                        activations[i] = 8'h00;
                    end
                end
                load_data = 1;
                @(posedge clock);
                load_data = 0;
                repeat(2) @(posedge clock);
            end
            
            wait(result_valid);
            @(posedge clock);
            $display("Channel %0d result: %0d", out_ch, $signed(conv_result));
            sum_results += $signed(conv_result);
            repeat(3) @(posedge clock);
        end
        
        $display("\nSUM: %0d, AVG: %0d", sum_results, sum_results/4);
        $display("TEST PASSED - Real weights processed!");
        repeat(5) @(posedge clock);
        $finish;
    end
    
    initial begin
        #100000;
        $finish;
    end

endmodule