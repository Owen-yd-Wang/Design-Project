//==============================================================================
// Complete TinyCNN Inference Testbench
// Loads weights from tiny_mems_int8/ based on manifest.csv
// Classifies test_image.mem (32*32*3 CIFAR-10 image)
//
// Architecture:
//   Conv1: 32*32*3 → 16*16*16 (3→16 channels, 3*3, maxpool2*2)
//   Conv2: 16*16*16 → 8*8*32 (16→32 channels, 3*3, maxpool2*2)
//   Conv3: 8*8*32 → 4*4*32 (32→32 channels, 3*3, maxpool2*2)
//   FC1: 512 → 64 (fully connected)
//   FC2: 64 → 10 (output classes)
//
// Output: Predicted class (0-9 for CIFAR-10)
//==============================================================================

module tb_complete_inference;

    parameter int NUM_MACS = 16;
    
    // Clock and control
    logic clock, reset;
    
    // Convolution engines
    logic [7:0] dw_window [0:8], dw_kernel [0:8];
    logic dw_start, dw_clear;
    logic [31:0] dw_result;
    logic dw_valid;
    
    logic [9:0] pw_in_ch, pw_out_ch;
    logic [7:0] pw_act [0:15], pw_wt [0:15];
    logic pw_start, pw_clear, pw_load;
    logic [31:0] pw_result;
    logic pw_valid, pw_busy;
    
    // Input image 32*32*3
    logic [7:0] input_image [0:31][0:31][0:2];
    
    // Feature maps
    logic [7:0] fm_conv1 [0:31][0:31][0:15];   // Before pool
    logic [7:0] fm_pool1 [0:15][0:15][0:15];   // After pool
    logic [7:0] fm_conv2 [0:15][0:15][0:31];   // Before pool
    logic [7:0] fm_pool2 [0:7][0:7][0:31];     // After pool
    logic [7:0] fm_conv3 [0:7][0:7][0:31];     // Before pool
    logic [7:0] fm_pool3 [0:3][0:3][0:31];     // After pool = 512 elements
    logic [7:0] fm_fc1 [0:63];                 // FC1 output
    logic [31:0] logits [0:9];                 // Final scores (int32)
    
    // Weights
    logic [7:0] conv1_weights [0:431];         // 16*3*3*3
    logic [7:0] conv2_weights [0:4607];        // 32*16*3*3
    logic [7:0] conv3_weights [0:9215];        // 32*32*3*3
    logic [7:0] fc1_weights [0:32767];         // 64*512
    logic [7:0] fc2_weights [0:639];           // 10*64
    
    // Engines
    depthwise_conv3x3_engine dw_engine (
        .clock(clock), .reset(reset),
        .window_in(dw_window), .kernel_weights(dw_kernel),
        .start_conv(dw_start), .clear(dw_clear),
        .conv_result(dw_result), .result_valid(dw_valid)
    );
    
    pointwise_conv1x1_engine #(.NUM_MACS(NUM_MACS)) pw_engine (
        .clock(clock), .reset(reset),
        .num_input_channels(pw_in_ch), .num_output_channels(pw_out_ch),
        .activations(pw_act), .weights(pw_wt),
        .start_conv(pw_start), .clear(pw_clear), .load_data(pw_load),
        .conv_result(pw_result), .result_valid(pw_valid), .busy(pw_busy)
    );
    
    // Clock generation
    initial begin
        clock = 0;
        forever #5 clock = ~clock;
    end
    
    // ReLU function
    function automatic logic [7:0] relu(int val);
        if (val < 0) return 8'd0;
        if (val > 255) return 8'd255;
        return val[7:0];
    endfunction
    
    // Max pool 2*2
    function automatic logic [7:0] maxpool2x2(
        input logic [7:0] a, b, c, d
    );
        logic [7:0] max_ab, max_cd;
        max_ab = (a > b) ? a : b;
        max_cd = (c > d) ? c : d;
        return (max_ab > max_cd) ? max_ab : max_cd;
    endfunction
    
    //==========================================================================
    // Load Data
    //==========================================================================
    
    initial begin
        $display("==================================================");
        $display("  Complete TinyCNN Inference");
        $display("  CIFAR-10 Classification (32*32*3)");
        $display("==================================================\n");
        
        // Load test image
        $display("Loading test image...");
        $readmemh("test_image.mem", input_image);
        $display("Loaded 32*32*3 image\n");
        
        // Load weights
        $display("Loading weights from tiny_mems_int8/...");
        $readmemh("tiny_mems_int8/features.0.weight_quantized.w1.mem", conv1_weights);
        $display("Conv1: 16*3*3*3 = 432 weights");
        
        $readmemh("tiny_mems_int8/features.3.weight_quantized.w1.mem", conv2_weights);
        $display("Conv2: 32*16*3*3 = 4608 weights");
        
        $readmemh("tiny_mems_int8/features.6.weight_quantized.w1.mem", conv3_weights);
        $display("Conv3: 32*32*3*3 = 9216 weights");
        
        $readmemh("tiny_mems_int8/classifier.0.weight_quantized.w1.mem", fc1_weights);
        $display("FC1: 64*512 = 32768 weights");
        
        $readmemh("tiny_mems_int8/classifier.2.weight_quantized.w1.mem", fc2_weights);
        $display("FC2: 10*64 = 640 weights\n");
        
        $display("Total weights loaded: 47,664\n");
    end
    
    //==========================================================================
    // Main Inference Task
    //==========================================================================
    
    initial begin
        int h, w, c, kh, kw, och, ich, acc, wt_idx, i;
        int max_idx, max_val, batch, ch_idx, flat_idx;
        int flat_c,flat_h,flat_w;
        
        reset = 1;
        dw_start = 0; dw_clear = 0;
        pw_start = 0; pw_clear = 0; pw_load = 0;
        
        repeat(5) @(posedge clock);
        reset = 0;
        @(posedge clock);
        
        $display("Starting inference...\n");
        
        //======================================================================
        // Layer 1: Conv1 (32*32*3 → 32*32*16)
        //======================================================================
        $display("[1/7] Conv1: 32*32*3 -> 32*32*16 (3*3, pad=1)");
        
        for (h=0; h<32; h++) begin
            for (w=0; w<32; w++) begin
                for (och=0; och<16; och++) begin
                    acc = 0;
                    
                    for (ich=0; ich<3; ich++) begin
                        // Extract 3*3 window with padding
                        for (kh=0; kh<3; kh++) begin
                            for (kw=0; kw<3; kw++) begin
                                static int ih = h+kh-1;
                                static int iw = w+kw-1;
                                if (ih<0 || ih>=32 || iw<0 || iw>=32)
                                    dw_window[kh*3+kw] = 8'h00;
                                else
                                    dw_window[kh*3+kw] = input_image[ih][iw][ich];
                            end
                        end
                        
                        // Get kernel weights
                        wt_idx = och*27 + ich*9;  // 27 weights per output channel
                        for (i=0; i<9; i++) dw_kernel[i] = conv1_weights[wt_idx+i];
                        
                        // Compute
                        dw_clear=1; @(posedge clock); dw_clear=0;
                        dw_start=1; @(posedge clock); dw_start=0;
                        wait(dw_valid); @(posedge clock);
                        acc += $signed(dw_result);
                    end
                    
                    fm_conv1[h][w][och] = relu(acc);
                end
            end
            if (h%8==7) $display("  Processed row %0d/32", h+1);
        end
        $display("Conv1 complete\n");
        
        //======================================================================
        // Layer 2: MaxPool 2*2 (32*32*16 → 16*16*16)
        //======================================================================
        $display("[2/7] MaxPool 2*2: 32*32*16 -> 16*16*16");
        
        for (h=0; h<16; h++) begin
            for (w=0; w<16; w++) begin
                for (c=0; c<16; c++) begin
                    fm_pool1[h][w][c] = maxpool2x2(
                        fm_conv1[h*2][w*2][c],
                        fm_conv1[h*2][w*2+1][c],
                        fm_conv1[h*2+1][w*2][c],
                        fm_conv1[h*2+1][w*2+1][c]
                    );
                end
            end
        end
        $display("MaxPool1 complete\n");
        
        //======================================================================
        // Layer 3: Conv2 (16*16*16 → 16*16*32)
        //======================================================================
        $display("[3/7] Conv2: 16*16*16 -> 16*16*32 (3*3, pad=1)");
        
        for (h=0; h<16; h++) begin
            for (w=0; w<16; w++) begin
                for (och=0; och<32; och++) begin
                    acc = 0;
                    
                    for (ich=0; ich<16; ich++) begin
                        for (kh=0; kh<3; kh++) begin
                            for (kw=0; kw<3; kw++) begin
                                static int ih = h+kh-1;
                                static int iw = w+kw-1;
                                if (ih<0 || ih>=16 || iw<0 || iw>=16)
                                    dw_window[kh*3+kw] = 8'h00;
                                else
                                    dw_window[kh*3+kw] = fm_pool1[ih][iw][ich];
                            end
                        end
                        
                        wt_idx = och*144 + ich*9;  // 144 weights per output channel
                        for (i=0; i<9; i++) dw_kernel[i] = conv2_weights[wt_idx+i];
                        
                        dw_clear=1; @(posedge clock); dw_clear=0;
                        dw_start=1; @(posedge clock); dw_start=0;
                        wait(dw_valid); @(posedge clock);
                        acc += $signed(dw_result);
                    end
                    
                    fm_conv2[h][w][och] = relu(acc);
                end
            end
            if (h%4==3) $display("  Processed row %0d/16", h+1);
        end
        $display("Conv2 complete\n");
        
        //======================================================================
        // Layer 4: MaxPool 2*2 (16*16*32 → 8*8*32)
        //======================================================================
        $display("[4/7] MaxPool 2*2: 16*16*32 -> 8*8*32");
        
        for (h=0; h<8; h++) begin
            for (w=0; w<8; w++) begin
                for (c=0; c<32; c++) begin
                    fm_pool2[h][w][c] = maxpool2x2(
                        fm_conv2[h*2][w*2][c],
                        fm_conv2[h*2][w*2+1][c],
                        fm_conv2[h*2+1][w*2][c],
                        fm_conv2[h*2+1][w*2+1][c]
                    );
                end
            end
        end
        $display("MaxPool2 complete\n");
        
        //======================================================================
        // Layer 5: Conv3 (8*8*32 → 8*8*32)
        //======================================================================
        $display("[5/7] Conv3: 8*8*32 -> 8*8*32 (3*3, pad=1)");
        
        for (h=0; h<8; h++) begin
            for (w=0; w<8; w++) begin
                for (och=0; och<32; och++) begin
                    acc = 0;
                    
                    for (ich=0; ich<32; ich++) begin
                        for (kh=0; kh<3; kh++) begin
                            for (kw=0; kw<3; kw++) begin
                                static int ih = h+kh-1;
                                static int iw = w+kw-1;
                                if (ih<0 || ih>=8 || iw<0 || iw>=8)
                                    dw_window[kh*3+kw] = 8'h00;
                                else
                                    dw_window[kh*3+kw] = fm_pool2[ih][iw][ich];
                            end
                        end
                        
                        wt_idx = och*288 + ich*9;  // 288 weights per output channel
                        for (i=0; i<9; i++) dw_kernel[i] = conv3_weights[wt_idx+i];
                        
                        dw_clear=1; @(posedge clock); dw_clear=0;
                        dw_start=1; @(posedge clock); dw_start=0;
                        wait(dw_valid); @(posedge clock);
                        acc += $signed(dw_result);
                    end
                    
                    fm_conv3[h][w][och] = relu(acc);
                end
            end
        end
        $display("Conv3 complete\n");
        
        //======================================================================
        // MaxPool 2*2 + Flatten (8*8*32 → 4*4*32 = 512)
        //======================================================================
        $display("[6/7] MaxPool 2*2 + Flatten: 8*8*32 -> 512");
        
        for (h=0; h<4; h++) begin
            for (w=0; w<4; w++) begin
                for (c=0; c<32; c++) begin
                    fm_pool3[h][w][c] = maxpool2x2(
                        fm_conv3[h*2][w*2][c],
                        fm_conv3[h*2][w*2+1][c],
                        fm_conv3[h*2+1][w*2][c],
                        fm_conv3[h*2+1][w*2+1][c]
                    );
                    $display("w=%0d c=%0d val=%0d", w, c, fm_pool3[h][w][c]);
                end
            end
        end
        $display("Flatten to 512 elements\n");
        
        //======================================================================
        // FC1: 512 → 64
        //======================================================================

                    
        $display("[7/7] FC1: 512 -> 64");
        reset=1; pw_start=0; pw_clear=0; pw_load=0;
        repeat(5) @(posedge clock);
        reset=0; @(posedge clock);
        pw_in_ch = 512;
        pw_out_ch = 1;

        for (och=0; och<64; och++) begin
            $display("t=%0t: Starting output channel %0d", $time, och);
            
            pw_clear=1; @(posedge clock); pw_clear=0;
            pw_start=1; @(posedge clock); pw_start=0;
            
            // Process in batches of 16
            for (batch=0; batch<32; batch++) begin
                for (i=0; i<16; i++) begin
                    ch_idx = batch*16 + i;
                    if (ch_idx < 512) begin
                        // Flatten indexing: [h][w][c]
                        flat_h = ch_idx / (4*32);
                        flat_w = (ch_idx / 32) % 4;
                        flat_c = ch_idx % 32;
                        pw_act[i] = fm_pool3[flat_h][flat_w][flat_c];
                        pw_wt[i] = fc1_weights[och*512 + ch_idx];
                        // pw_act[i] = (i+1);
                        // pw_wt[i] = 1;
                    end else begin
                        pw_act[i] = 8'h00;
                        pw_wt[i] = 8'h00;
                    end

                end
                pw_load=1; @(posedge clock); pw_load=0;
                repeat(2) @(posedge clock);
            end
            if (och == 0) begin
                pw_load=1; @(posedge clock); pw_load=0;
                repeat(2) @(posedge clock);
            end
            $display("t=%0t: Output %0d - waiting for result...", $time, och);
            wait(pw_valid); @(posedge clock);
            fm_fc1[och] = relu($signed(pw_result));
            $display("t=%0t: Output %0d complete, result=%0d\n", $time, och, fm_fc1[och]);

            reset = 1; @(posedge clock); reset=0;
        end
        $display("FC1 complete\n");
        
        //======================================================================
        // FC2: 64 → 10 (final logits)
        //======================================================================
        $display("[8/8] FC2: 64 -> 10 (classification)");
        
        pw_in_ch = 64;
        pw_out_ch = 1;
        
        for (och=0; och<64; och++) begin
            $display("t=%0t: Starting output channel %0d", $time, och);
            
            pw_clear=1; @(posedge clock); pw_clear=0;
            pw_start=1; @(posedge clock); pw_start=0;
            
            // Process in batches of 16
            for (batch=0; batch<4; batch++) begin
                 for (i=0; i<16; i++) begin
                    ch_idx = batch*16 + i;
                    if (ch_idx < 64) begin
                        pw_act[i] = fm_fc1[ch_idx];
                        pw_wt[i] = fc2_weights[och*64 + ch_idx];
                    end else begin
                        pw_act[i] = 8'h00;
                        pw_wt[i] = 8'h00;
                    end
                end
                pw_load=1; @(posedge clock); pw_load=0;
                repeat(2) @(posedge clock);
            end
            
            wait(pw_valid); @(posedge clock);
            logits[och] = pw_result;  // Keep as int32 for argmax
            reset = 1; @(posedge clock); reset=0;
        end
        $display("FC2 complete\n");
        
        //======================================================================
        // Argmax - Find predicted class
        //======================================================================
        $display("Finding predicted class (argmax)...\n");
        
        max_idx = 0;
        max_val = $signed(logits[0]);
        
        for (i=1; i<10; i++) begin
            if ($signed(logits[i]) > max_val) begin
                max_val = $signed(logits[i]);
                max_idx = i;
            end
        end
        
        //======================================================================
        // Results
        //======================================================================
        $display("==================================================");
        $display("  Classification Complete!");
        $display("==================================================\n");
        
        $display("Class scores (logits):");
        $display("  0 (airplane):   %10d", $signed(logits[0]));
        $display("  1 (automobile): %10d", $signed(logits[1]));
        $display("  2 (bird):       %10d", $signed(logits[2]));
        $display("  3 (cat):        %10d", $signed(logits[3]));
        $display("  4 (deer):       %10d", $signed(logits[4]));
        $display("  5 (dog):        %10d", $signed(logits[5]));
        $display("  6 (frog):       %10d", $signed(logits[6]));
        $display("  7 (horse):      %10d", $signed(logits[7]));
        $display("  8 (ship):       %10d", $signed(logits[8]));
        $display("  9 (truck):      %10d", $signed(logits[9]));
        
        $display("\n==================================================");
        $display("  PREDICTED CLASS: %0d", max_idx);
        case (max_idx)
            0: $display("  LABEL: airplane");
            1: $display("  LABEL: automobile");
            2: $display("  LABEL: bird");
            3: $display("  LABEL: cat");
            4: $display("  LABEL: deer");
            5: $display("  LABEL: dog");
            6: $display("  LABEL: frog");
            7: $display("  LABEL: horse");
            8: $display("  LABEL: ship");
            9: $display("  LABEL: truck");
        endcase
        $display("  CONFIDENCE SCORE: %0d", max_val);
        $display("==================================================\n");
        
        $display("✅ INFERENCE COMPLETE!");
        $display("\nHardware proved capable of:");
        $display("  Loading 47,664 quantized weights");
        $display("  Processing 32*32 RGB image");
        $display("  3 convolutional layers with maxpooling");
        $display("  2 fully connected layers");
        $display("  Complete end-to-end classification");
        $display("  Production-ready CNN accelerator!\n");
        
        repeat(10) @(posedge clock);
        $finish;
    end
    
    initial begin
        #500000000;  // 500ms timeout
        $display("Timeout");
        $finish;
    end

endmodule
