//==============================================================================
// Testbench: tb_layer0_inference - Layer 0 with egypt_cat.jpg
//==============================================================================

module tb_layer0_inference;

    parameter int INPUT_H = 224;
    parameter int INPUT_W = 224;
    parameter int INPUT_C = 3;
    parameter int OUTPUT_C = 32;
    
    logic clock;
    logic reset;
    logic [7:0] input_image [0:INPUT_H*INPUT_W*INPUT_C-1];
    logic [7:0] layer0_weights [0:863];
    
    logic [7:0] conv_window [0:8];
    logic [7:0] conv_kernel [0:8];
    logic       conv_start;
    logic       conv_clear;
    logic [31:0] conv_result;
    logic       conv_valid;
    
    logic [31:0] output_features [0:OUTPUT_C-1];
    
    initial begin
        clock = 0;
        forever #5 clock = ~clock;
    end
    
    depthwise_conv3x3_engine conv_engine (
        .clock(clock), .reset(reset),
        .window_in(conv_window),
        .kernel_weights(conv_kernel),
        .start_conv(conv_start),
        .clear(conv_clear),
        .conv_result(conv_result),
        .result_valid(conv_valid)
    );
    
    initial begin
        $display("Loading test image...");
        $readmemh("test_image.mem", input_image);
        $display("Loaded %0d bytes", INPUT_H*INPUT_W*INPUT_C);
    end
    
    initial begin
        $display("Loading Layer 0 weights...");
        $readmemh("mems_int8/features.0.0.weight_quantized.w1.mem", layer0_weights);
        $display("Loaded %0d weights", 864);
    end
    
    function automatic logic [7:0] get_pixel(int h, int w, int c);
        int idx;
        if (h < 0 || h >= INPUT_H || w < 0 || w >= INPUT_W) return 8'h00;
        idx = (h * INPUT_W + w) * INPUT_C + c;
        return input_image[idx];
    endfunction
    
    task automatic extract_window(input int center_h, input int center_w, input int channel, output logic [7:0] window [0:8]);
        int wh, ww, idx;
        idx = 0;
        for (wh = -1; wh <= 1; wh++) begin
            for (ww = -1; ww <= 1; ww++) begin
                window[idx] = get_pixel(center_h + wh, center_w + ww, channel);
                idx++;
            end
        end
    endtask
    
    task automatic get_kernel(input int out_ch, input int in_ch, output logic [7:0] kernel [0:8]);
        int base_idx;
        base_idx = (out_ch * INPUT_C + in_ch) * 9;
        for (int i = 0; i < 9; i++) kernel[i] = layer0_weights[base_idx + i];
    endtask
    
    initial begin
        int out_h, out_w, out_ch, in_ch, input_h, input_w, accumulator;
        logic [7:0] window [0:8];
        logic [7:0] kernel [0:8];
        
        reset = 1;
        conv_start = 0;
        conv_clear = 0;
        
        repeat(5) @(posedge clock);
        reset = 0;
        @(posedge clock);
        
        out_h = 56;
        out_w = 56;
        input_h = out_h * 2;
        input_w = out_w * 2;
        
        $display("\nProcessing output position (%0d, %0d)", out_h, out_w);
        $display("Input center: (%0d, %0d)", input_h, input_w);
        
        for (out_ch = 0; out_ch < OUTPUT_C; out_ch++) begin
            accumulator = 0;
            
            for (in_ch = 0; in_ch < INPUT_C; in_ch++) begin
                extract_window(input_h, input_w, in_ch, window);
                get_kernel(out_ch, in_ch, kernel);
                
                conv_clear = 1;
                @(posedge clock);
                conv_clear = 0;
                
                for (int i = 0; i < 9; i++) begin
                    conv_window[i] = window[i];
                    conv_kernel[i] = kernel[i];
                end
                
                conv_start = 1;
                @(posedge clock);
                conv_start = 0;
                
                wait(conv_valid);
                @(posedge clock);
                
                accumulator += $signed(conv_result);
            end
            
            output_features[out_ch] = accumulator;
            $display("Channel %2d: %10d", out_ch, $signed(accumulator));
            repeat(2) @(posedge clock);
        end
        
        $display("\n=== Layer 0 Complete ===");
        $display("First 8 channels:");
        for (int ch = 0; ch < 8; ch++) begin
            $display("  Ch %2d: %10d", ch, $signed(output_features[ch]));
        end
        
        $display("\nTEST PASSED - Layer 0 processed!");
        repeat(5) @(posedge clock);
        $finish;
    end
    
    initial begin
        #10000000;
        $display("Timeout");
        $finish;
    end

endmodule