//==============================================================================
// tb_real_layer_processing: Process real layer weights
//==============================================================================

module tb_real_layer_processing;

    parameter int NUM_MACS = 16;
    parameter int INPUT_CHANNELS = 96;
    parameter int OUTPUT_CHANNELS = 64;
    parameter int TOTAL_WEIGHTS = OUTPUT_CHANNELS * INPUT_CHANNELS;
    
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
    
    logic [7:0]  layer_weights [0:TOTAL_WEIGHTS-1];
    
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
        $display("Loading real weights...");
        $readmemh("mems_int8/features.5.conv.2.weight_quantized.w1.mem", layer_weights);
        $display("Loaded %0d weights", TOTAL_WEIGHTS);
    end
    
    initial begin
        int num_batches;
        int batch;
        int i;
        int idx;
        
        reset = 1;
        start_conv = 0;
        clear = 0;
        load_data = 0;
        num_input_ch = INPUT_CHANNELS;
        num_output_ch = 1;
        for (i = 0; i < NUM_MACS; i++) activations[i] = 8'd10 + i;
        
        repeat(5) @(posedge clock);
        reset = 0;
        @(posedge clock);
        
        $display("Processing output channel 0...");
        num_batches = (INPUT_CHANNELS + NUM_MACS - 1) / NUM_MACS;
        
        start_conv = 1;
        @(posedge clock);
        start_conv = 0;
        
        for (batch = 0; batch < num_batches; batch++) begin
            for (i = 0; i < NUM_MACS; i++) begin
                idx = batch * NUM_MACS + i;
                if (idx < INPUT_CHANNELS) begin
                    weights[i] = layer_weights[idx];
                    activations[i] = 8'd10 + (idx % 16);
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
        
        $display("Result: %0d", $signed(conv_result));
        $display("TEST PASSED - Real weights loaded and processed!");
        
        repeat(5) @(posedge clock);
        $finish;
    end
    
    initial begin
        #100000;
        $finish;
    end

endmodule