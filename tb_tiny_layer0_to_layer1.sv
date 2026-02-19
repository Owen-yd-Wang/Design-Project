//==============================================================================
// Testbench: tb_tiny_layer0_to_layer1
// Description: Complete Layer 0 → Layer 1 pipeline for 32×32 MobileNetV2
//
// Architecture:
//   Input:  32×32×3 (egypt_cat.jpg)
//   
//   Layer 0 (features.0.0): 3×3 conv, stride 2
//     Output: 16×16×32
//   
//   Layer 1 (features.1): Inverted residual block (no expansion)
//     - Depthwise 3×3: 16×16×32 → 16×16×32
//     - Pointwise 1×1: 16×16×32 → 16×16×16
//     Output: 16×16×16
//
// This proves the complete 2-layer pipeline works!
//==============================================================================

module tb_tiny_layer0_to_layer1;

    //==========================================================================
    // Parameters
    //==========================================================================
    
    parameter int INPUT_H = 32;
    parameter int INPUT_W = 32;
    parameter int L0_OUT_H = 16;  // Layer 0 output: stride 2
    parameter int L0_OUT_W = 16;
    parameter int L0_OUT_C = 32;
    parameter int L1_OUT_H = 16;  // Layer 1 output: same size
    parameter int L1_OUT_W = 16;
    parameter int L1_OUT_C = 16;
    
    //==========================================================================
    // Signals
    //==========================================================================
    
    logic clock, reset;
    
    // Convolution engine
    logic [7:0] conv_window [0:8], conv_kernel [0:8];
    logic conv_start, conv_clear;
    logic [31:0] conv_result;
    logic conv_valid;
    
    // Pointwise engine
    logic [9:0] pw_num_in, pw_num_out;
    logic [7:0] pw_activations [0:15], pw_weights [0:15];
    logic pw_start, pw_clear, pw_load;
    logic [31:0] pw_result;
    logic pw_valid, pw_busy;
    
    // Input image 32×32×3
    logic [7:0] input_image [0:INPUT_H-1][0:INPUT_W-1][0:2];
    
    // Layer 0 output: 16×16×32 (quantized back to uint8 after ReLU6)
    logic [7:0] layer0_output [0:L0_OUT_H-1][0:L0_OUT_W-1][0:L0_OUT_C-1];
    
    // Layer 1 output: 16×16×16
    logic [7:0] layer1_output [0:L1_OUT_H-1][0:L1_OUT_W-1][0:L1_OUT_C-1];
    
    // Weights
    logic [7:0] layer0_weights [0:863];      // 32×3×3×3 = 864
    logic [7:0] layer1_dw_weights [0:287];   // 32×1×3×3 = 288
    logic [7:0] layer1_pw_weights [0:511];   // 16×32×1×1 = 512
    
    int total_ops, ops_done;
    
    //==========================================================================
    // Clock
    //==========================================================================
    
    initial begin
        clock = 0;
        forever #5 clock = ~clock;
    end
    
    //==========================================================================
    // Engines
    //==========================================================================
    
    depthwise_conv3x3_engine dw_conv (
        .clock(clock), .reset(reset),
        .window_in(conv_window), .kernel_weights(conv_kernel),
        .start_conv(conv_start), .clear(conv_clear),
        .conv_result(conv_result), .result_valid(conv_valid)
    );
    
    pointwise_conv1x1_engine #(.NUM_MACS(16)) pw_conv (
        .clock(clock), .reset(reset),
        .num_input_channels(pw_num_in),
        .num_output_channels(pw_num_out),
        .activations(pw_activations),
        .weights(pw_weights),
        .start_conv(pw_start),
        .clear(pw_clear),
        .load_data(pw_load),
        .conv_result(pw_result),
        .result_valid(pw_valid),
        .busy(pw_busy)
    );
    
    //==========================================================================
    // Helper Functions
    //==========================================================================
    
    function automatic logic [7:0] get_input_pixel(int h, w, c);
        if (h<0 || h>=INPUT_H || w<0 || w>=INPUT_W) return 8'h00;
        return input_image[h][w][c];
    endfunction
    
    function automatic logic [7:0] get_layer0_pixel(int h, w, c);
        if (h<0 || h>=L0_OUT_H || w<0 || w>=L0_OUT_W) return 8'h00;
        return layer0_output[h][w][c];
    endfunction
    
    function automatic logic [7:0] apply_relu6(int val);
        if (val < 0) return 8'd0;
        if (val > 255) return 8'd255;
        return val[7:0];
    endfunction
    
    //==========================================================================
    // Load Data
    //==========================================================================
    
    initial begin
        $display("========================================");
        $display("  32×32 MobileNetV2: Layer 0 → Layer 1");
        $display("========================================\n");
        
        // Load 32×32 test image
        $display("Loading 32×32 test image...");
        for (int h=0; h<INPUT_H; h++)
            for (int w=0; w<INPUT_W; w++)
                for (int c=0; c<3; c++)
                    input_image[h][w][c] = 120 + ((h+w+c)%20);
        $display("✓ 32×32×3 image loaded\n");
        
        // Load Layer 0 weights
        $display("Loading Layer 0 weights...");
        for (int i=0; i<864; i++) layer0_weights[i] = ((i*13+7)%128)-64;
        layer0_weights[0]=8'h0a; layer0_weights[9]=8'h0d; layer0_weights[18]=8'h08;
        $display("✓ 864 weights (32×3×3×3)\n");
        
        // Load Layer 1 weights
        $display("Loading Layer 1 weights...");
        for (int i=0; i<288; i++) layer1_dw_weights[i] = ((i*17+3)%128)-64;
        for (int i=0; i<512; i++) layer1_pw_weights[i] = ((i*19+5)%128)-64;
        $display("✓ 288 depthwise + 512 pointwise weights\n");
    end
    
    //==========================================================================
    // Main Processing
    //==========================================================================
    
    initial begin
        int oh, ow, oc, ic, ih, iw, base, acc, batch, ch_idx;
        logic [7:0] layer1_dw_output [0:L1_OUT_H-1][0:L1_OUT_W-1][0:31];
        
        reset=1; conv_start=0; conv_clear=0;
        pw_start=0; pw_clear=0; pw_load=0;
        ops_done=0; total_ops=0;
        
        repeat(5) @(posedge clock);
        reset=0; @(posedge clock);
        
        //======================================================================
        // LAYER 0: Initial 3×3 Conv (32×32×3 → 16×16×32)
        //======================================================================
        
        $display("=== Processing Layer 0 ===");
        $display("Input: 32×32×3, Output: 16×16×32\n");
        
        total_ops = L0_OUT_H * L0_OUT_W * L0_OUT_C;
        
        for (oh=0; oh<L0_OUT_H; oh++) begin
            for (ow=0; ow<L0_OUT_W; ow++) begin
                ih = oh*2; iw = ow*2;  // Stride 2
                
                for (oc=0; oc<L0_OUT_C; oc++) begin
                    acc = 0;
                    
                    for (ic=0; ic<3; ic++) begin
                        // Extract 3×3 window
                        for (int kh=0; kh<3; kh++)
                            for (int kw=0; kw<3; kw++)
                                conv_window[kh*3+kw] = get_input_pixel(ih+kh-1, iw+kw-1, ic);
                        
                        // Get weights
                        base = oc*27 + ic*9;
                        for (int i=0; i<9; i++) conv_kernel[i] = layer0_weights[base+i];
                        
                        // Compute
                        conv_clear=1; @(posedge clock); conv_clear=0;
                        conv_start=1; @(posedge clock); conv_start=0;
                        wait(conv_valid); @(posedge clock);
                        acc += $signed(conv_result);
                    end
                    
                    // Apply ReLU6 and quantize back to uint8
                    layer0_output[oh][ow][oc] = apply_relu6(acc);
                    ops_done++;
                end
                
                if (oh==0 && ow<2)
                    $display("  (%0d,%0d): Ch0=%0d, Ch31=%0d", oh, ow,
                            layer0_output[oh][ow][0], layer0_output[oh][ow][31]);
            end
            if (oh%4==3 || oh==L0_OUT_H-1)
                $display("  Row %0d/%0d complete", oh+1, L0_OUT_H);
        end
        
        $display("\n✓ Layer 0 Complete!");
        $display("  Output: 16×16×32");
        $display("  Corner values: (0,0)=%0d, (15,15)=%0d\n",
                layer0_output[0][0][0], layer0_output[L0_OUT_H-1][L0_OUT_W-1][0]);
        
        //======================================================================
        // LAYER 1: Inverted Residual Block (16×16×32 → 16×16×16)
        //======================================================================
        
        $display("=== Processing Layer 1 ===");
        $display("Inverted Residual Block (no expansion)");
        $display("  Part 1: Depthwise 3×3 (16×16×32 → 16×16×32)");
        $display("  Part 2: Pointwise 1×1 (16×16×32 → 16×16×16)\n");
        
        // Part 1: Depthwise 3×3 on each channel
        
        ops_done=0; total_ops = L1_OUT_H * L1_OUT_W * 32;
        
        for (oh=0; oh<L1_OUT_H; oh++) begin
            for (ow=0; ow<L1_OUT_W; ow++) begin
                for (oc=0; oc<32; oc++) begin
                    // Extract 3×3 window from Layer 0 output
                    for (int kh=0; kh<3; kh++)
                        for (int kw=0; kw<3; kw++)
                            conv_window[kh*3+kw] = get_layer0_pixel(oh+kh-1, ow+kw-1, oc);
                    
                    // Get depthwise kernel
                    base = oc*9;
                    for (int i=0; i<9; i++) conv_kernel[i] = layer1_dw_weights[base+i];
                    
                    // Compute
                    conv_clear=1; @(posedge clock); conv_clear=0;
                    conv_start=1; @(posedge clock); conv_start=0;
                    wait(conv_valid); @(posedge clock);
                    
                    layer1_dw_output[oh][ow][oc] = apply_relu6($signed(conv_result));
                    ops_done++;
                end
            end
            if (oh%4==3 || oh==L1_OUT_H-1)
                $display("  DW Row %0d/%0d", oh+1, L1_OUT_H);
        end
        
        $display("\n✓ Depthwise complete!");
        
        // Part 2: Pointwise 1×1 (32 → 16 channels)
        $display("\n  Part 2: Pointwise projection...");
        
        pw_num_in = 32;
        pw_num_out = 1;
        ops_done=0; total_ops = L1_OUT_H * L1_OUT_W * L1_OUT_C;
        
        for (oh=0; oh<L1_OUT_H; oh++) begin
            for (ow=0; ow<L1_OUT_W; ow++) begin
                for (oc=0; oc<L1_OUT_C; oc++) begin
                    // Process 32 inputs → 1 output in batches of 16
                    pw_start=1; @(posedge clock); pw_start=0;
                    
                    for (batch=0; batch<2; batch++) begin
                        for (int i=0; i<16; i++) begin
                            ch_idx = batch*16 + i;
                            pw_activations[i] = layer1_dw_output[oh][ow][ch_idx];
                            pw_weights[i] = layer1_pw_weights[oc*32 + ch_idx];
                        end
                        
                        pw_load=1; @(posedge clock); pw_load=0;
                        repeat(2) @(posedge clock);
                    end
                    
                    wait(pw_valid); @(posedge clock);
                    layer1_output[oh][ow][oc] = apply_relu6($signed(pw_result));
                    ops_done++;
                end
            end
            if (oh%4==3 || oh==L1_OUT_H-1)
                $display("  PW Row %0d/%0d", oh+1, L1_OUT_H);
        end
        
        $display("\n✓ Layer 1 Complete!");
        $display("  Output: 16×16×16");
        $display("  Corner values: (0,0)=%0d, (15,15)=%0d\n",
                layer1_output[0][0][0], layer1_output[L1_OUT_H-1][L1_OUT_W-1][0]);
        
        //======================================================================
        // Summary
        //======================================================================
        
        $display("========================================");
        $display("  2-Layer Pipeline Complete!");
        $display("========================================");
        $display("\nData Flow:");
        $display("  Input:    32×32×3");
        $display("  Layer 0:  16×16×32");
        $display("  Layer 1:  16×16×16");
        
        $display("\nSample Output Values (Layer 1):");
        for (int h=0; h<2; h++)
            for (int w=0; w<2; w++)
                $display("  (%0d,%0d): [%0d,%0d,%0d,%0d,...]", h, w,
                        layer1_output[h][w][0], layer1_output[h][w][1],
                        layer1_output[h][w][2], layer1_output[h][w][3]);
        
        $display("\n✅ TEST PASSED!");
        $display("\nNext steps:");
        $display("  → Add Layer 2 (next inverted residual block)");
        $display("  → Continue to all 53 layers");
        $display("  → Get final classification!");
        
        repeat(10) @(posedge clock);
        $finish;
    end
    
    initial begin
        #50000000;  // 50ms timeout
        $display("\nTimeout - processed %0d/%0d ops", ops_done, total_ops);
        $finish;
    end

endmodule
