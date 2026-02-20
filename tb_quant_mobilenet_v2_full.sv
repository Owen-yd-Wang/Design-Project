//==============================================================================
// Testbench: tb_quant_mobilenet_v2_full
// Description: Full quantized MobileNetV2 inference on 32×32 image
//              Loads weights from tiny_mems_int8/ using manifest.csv
//              Performs complete classification pipeline
//
// Architecture:
//   Input: 32×32×3 test image → test_image.mem
//   Layer 0: Conv 3→16
//   Layer 3: Conv 16→32
//   Layer 6: Conv 32→32
//   Classifier: FC 512→64→10 (10 output classes)
//
// Output: Predicted class (0-9) with confidence scores
//==============================================================================

module tb_quant_mobilenet_v2_full;

    //==========================================================================
    // Constants and Parameters
    //==========================================================================
    
    parameter int INPUT_H = 32;
    parameter int INPUT_W = 32;
    parameter int INPUT_C = 3;
    
    parameter int LAYER0_OUT_H = 16;
    parameter int LAYER0_OUT_W = 16;
    parameter int LAYER0_OUT_C = 16;
    
    parameter int LAYER3_OUT_H = 8;
    parameter int LAYER3_OUT_W = 8;
    parameter int LAYER3_OUT_C = 32;
    
    parameter int LAYER6_OUT_H = 4;
    parameter int LAYER6_OUT_W = 4;
    parameter int LAYER6_OUT_C = 32;
    
    parameter int NUM_CLASSES = 10;
    parameter int NUM_MACS = 16;
    
    //==========================================================================
    // Signal Declarations
    //==========================================================================
    
    logic clock;
    logic reset;
    
    // Memory arrays for weights and activations
    logic [7:0] test_image [0:INPUT_H*INPUT_W*INPUT_C-1];
    
    // Layer 0 weights
    logic [7:0] layer0_weights [0:431];      // 16×3×3×3 = 432 elements
    logic [7:0] layer0_zero_point;
    
    // Layer 3 weights
    logic [7:0] layer3_weights [0:4607];     // 32×16×3×3 = 4608 elements
    logic [7:0] layer3_zero_point;
    
    // Layer 6 weights
    logic [7:0] layer6_weights [0:9215];     // 32×32×3×3 = 9216 elements
    logic [7:0] layer6_zero_point;
    
    // Classifier weights
    logic [7:0] classifier0_weights [0:32767];  // 64×512 = 32768 elements
    logic [7:0] classifier0_zero_point;
    logic [7:0] classifier2_weights [0:639];    // 10×64 = 640 elements
    logic [7:0] classifier2_zero_point;
    
    // Zero points and scale factors
    logic [7:0] input_zero_point;
    logic [7:0] linear_zero_point;
    logic [7:0] output_zero_point;
    logic [7:0] conv2d_1_zero_point;
    logic [7:0] conv2d_2_zero_point;
    
    // Feature maps - Layer 0 output (16×16×16)
    logic [31:0] layer0_output [0:LAYER0_OUT_H*LAYER0_OUT_W*LAYER0_OUT_C-1];
    
    // Feature maps - Layer 3 output (8×8×32)
    logic [31:0] layer3_output [0:LAYER3_OUT_H*LAYER3_OUT_W*LAYER3_OUT_C-1];
    
    // Feature maps - Layer 6 output (4×4×32)
    logic [31:0] layer6_output [0:LAYER6_OUT_H*LAYER6_OUT_W*LAYER6_OUT_C-1];
    
    // Classifier intermediate and final outputs
    logic [31:0] classifier0_output [0:63];
    logic [31:0] classifier2_output [0:NUM_CLASSES-1];
    
    // Convolution engines
    logic [7:0] conv_window [0:8];
    logic [7:0] conv_kernel [0:8];
    logic conv_start;
    logic conv_clear;
    logic [31:0] conv_result;
    logic conv_valid;
    
    //==========================================================================
    // Clock Generation
    //==========================================================================
    
    initial begin
        clock = 0;
        forever #5 clock = ~clock;
    end
    
    //==========================================================================
    // Module Instantiation - Depthwise Conv Engine
    //==========================================================================
    
    depthwise_conv3x3_engine conv_engine (
        .clock              (clock),
        .reset              (reset),
        .window_in          (conv_window),
        .kernel_weights     (conv_kernel),
        .start_conv         (conv_start),
        .clear              (conv_clear),
        .conv_result        (conv_result),
        .result_valid       (conv_valid)
    );
    
    //==========================================================================
    // Utility Functions
    //==========================================================================
    
    function automatic logic [7:0] saturate_u8(logic [31:0] value);
        if (value < 0) return 8'h00;
        if (value > 255) return 8'hFF;
        return value[7:0];
    endfunction
    
    function automatic logic [31:0] quantize_to_int32(logic [31:0] value, logic [7:0] zero_point);
        return $signed(value) - $signed({{24{zero_point[7]}}, zero_point});
    endfunction
    
    function automatic int linear_index_3d(int h, int w, int c, int width, int channels);
        return (h * width + w) * channels + c;
    endfunction
    
    function automatic logic [7:0] get_pixel(int h, int w, int c);
        int idx;
        if (h < 0 || h >= INPUT_H || w < 0 || w >= INPUT_W) return 8'h00;
        idx = (h * INPUT_W + w) * INPUT_C + c;
        return test_image[idx];
    endfunction
    
    function automatic logic [7:0] get_layer0_pixel(int h, int w, int c);
        int idx;
        if (h < 0 || h >= LAYER0_OUT_H || w < 0 || w >= LAYER0_OUT_W) return 8'h00;
        idx = (h * LAYER0_OUT_W + w) * LAYER0_OUT_C + c;
        return saturate_u8(layer0_output[idx]);
    endfunction
    
    function automatic logic [7:0] get_layer3_pixel(int h, int w, int c);
        int idx;
        if (h < 0 || h >= LAYER3_OUT_H || w < 0 || w >= LAYER3_OUT_W) return 8'h00;
        idx = (h * LAYER3_OUT_W + w) * LAYER3_OUT_C + c;
        return saturate_u8(layer3_output[idx]);
    endfunction
    
    function automatic logic [7:0] get_layer6_pixel(int h, int w, int c);
        int idx;
        if (h < 0 || h >= LAYER6_OUT_H || w < 0 || w >= LAYER6_OUT_W) return 8'h00;
        idx = (h * LAYER6_OUT_W + w) * LAYER6_OUT_C + c;
        return saturate_u8(layer6_output[idx]);
    endfunction
    
    //==========================================================================
    // Task: Extract 3×3 window from input image (for Layer 0)
    //==========================================================================
    
    task automatic extract_window_input(
        input int center_h, 
        input int center_w, 
        input int channel,
        output logic [7:0] window [0:8]
    );
        int wh, ww, idx;
        idx = 0;
        for (wh = -1; wh <= 1; wh++) begin
            for (ww = -1; ww <= 1; ww++) begin
                window[idx] = get_pixel(center_h + wh, center_w + ww, channel);
                idx++;
            end
        end
    endtask
    
    //==========================================================================
    // Task: Extract 3×3 window from Layer 0 output
    //==========================================================================
    
    task automatic extract_window_layer0(
        input int center_h, 
        input int center_w, 
        input int channel,
        output logic [7:0] window [0:8]
    );
        int wh, ww, idx;
        idx = 0;
        for (wh = -1; wh <= 1; wh++) begin
            for (ww = -1; ww <= 1; ww++) begin
                window[idx] = get_layer0_pixel(center_h + wh, center_w + ww, channel);
                idx++;
            end
        end
    endtask
    
    //==========================================================================
    // Task: Extract 3×3 window from Layer 3 output
    //==========================================================================
    
    task automatic extract_window_layer3(
        input int center_h, 
        input int center_w, 
        input int channel,
        output logic [7:0] window [0:8]
    );
        int wh, ww, idx;
        idx = 0;
        for (wh = -1; wh <= 1; wh++) begin
            for (ww = -1; ww <= 1; ww++) begin
                window[idx] = get_layer3_pixel(center_h + wh, center_w + ww, channel);
                idx++;
            end
        end
    endtask
    
    //==========================================================================
    // Task: Extract 3×3 window from Layer 6 output
    //==========================================================================
    
    task automatic extract_window_layer6(
        input int center_h, 
        input int center_w, 
        input int channel,
        output logic [7:0] window [0:8]
    );
        int wh, ww, idx;
        idx = 0;
        for (wh = -1; wh <= 1; wh++) begin
            for (ww = -1; ww <= 1; ww++) begin
                window[idx] = get_layer6_pixel(center_h + wh, center_w + ww, channel);
                idx++;
            end
        end
    endtask
    
    //==========================================================================
    // Task: Get kernel from weight memory
    //==========================================================================
    
    task automatic get_kernel_layer0(
        input int out_ch, 
        input int in_ch,
        output logic [7:0] kernel [0:8]
    );
        int base_idx;
        base_idx = (out_ch * INPUT_C + in_ch) * 9;
        for (int i = 0; i < 9; i++) kernel[i] = layer0_weights[base_idx + i];
    endtask
    
    task automatic get_kernel_layer3(
        input int out_ch, 
        input int in_ch,
        output logic [7:0] kernel [0:8]
    );
        int base_idx;
        base_idx = (out_ch * LAYER0_OUT_C + in_ch) * 9;
        for (int i = 0; i < 9; i++) kernel[i] = layer3_weights[base_idx + i];
    endtask
    
    task automatic get_kernel_layer6(
        input int out_ch, 
        input int in_ch,
        output logic [7:0] kernel [0:8]
    );
        int base_idx;
        base_idx = (out_ch * LAYER3_OUT_C + in_ch) * 9;
        for (int i = 0; i < 9; i++) kernel[i] = layer6_weights[base_idx + i];
    endtask
    
    //==========================================================================
    // Task: Process Layer 0 (32×32×3 → 16×16×16)
    //==========================================================================
    
    task automatic process_layer0();
        int out_h, out_w, out_ch, in_ch, input_h, input_w, accumulator;
        logic [7:0] window [0:8];
        logic [7:0] kernel [0:8];
        
        $display("\n========== PROCESSING LAYER 0 ==========");
        $display("Input: 32×32×3, Output: 16×16×16 (stride=2)");
        
        for (out_h = 0; out_h < LAYER0_OUT_H; out_h++) begin
            for (out_w = 0; out_w < LAYER0_OUT_W; out_w++) begin
                input_h = out_h * 2;
                input_w = out_w * 2;
                
                for (out_ch = 0; out_ch < LAYER0_OUT_C; out_ch++) begin
                    accumulator = 0;
                    
                    for (in_ch = 0; in_ch < INPUT_C; in_ch++) begin
                        extract_window_input(input_h, input_w, in_ch, window);
                        get_kernel_layer0(out_ch, in_ch, kernel);
                        
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
                    
                    layer0_output[linear_index_3d(out_h, out_w, out_ch, LAYER0_OUT_W, LAYER0_OUT_C)] = accumulator;
                end
            end
            
            if ((out_h + 1) % 4 == 0) $display("Layer 0 progress: %0d/%0d rows", out_h + 1, LAYER0_OUT_H);
        end
        
        $display("Layer 0 Complete!");
    endtask
    
    //==========================================================================
    // Task: Process Layer 3 (16×16×16 → 8×8×32)
    //==========================================================================
    
    task automatic process_layer3();
        int out_h, out_w, out_ch, in_ch, input_h, input_w, accumulator;
        logic [7:0] window [0:8];
        logic [7:0] kernel [0:8];
        
        $display("\n========== PROCESSING LAYER 3 ==========");
        $display("Input: 16×16×16, Output: 8×8×32 (stride=2)");
        
        for (out_h = 0; out_h < LAYER3_OUT_H; out_h++) begin
            for (out_w = 0; out_w < LAYER3_OUT_W; out_w++) begin
                input_h = out_h * 2;
                input_w = out_w * 2;
                
                for (out_ch = 0; out_ch < LAYER3_OUT_C; out_ch++) begin
                    accumulator = 0;
                    
                    for (in_ch = 0; in_ch < LAYER0_OUT_C; in_ch++) begin
                        extract_window_layer0(input_h, input_w, in_ch, window);
                        get_kernel_layer3(out_ch, in_ch, kernel);
                        
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
                    
                    layer3_output[linear_index_3d(out_h, out_w, out_ch, LAYER3_OUT_W, LAYER3_OUT_C)] = accumulator;
                end
            end
        end
        
        $display("Layer 3 Complete!");
    endtask
    
    //==========================================================================
    // Task: Process Layer 6 (8×8×32 → 4×4×32)
    //==========================================================================
    
    task automatic process_layer6();
        int out_h, out_w, out_ch, in_ch, input_h, input_w, accumulator;
        logic [7:0] window [0:8];
        logic [7:0] kernel [0:8];
        
        $display("\n========== PROCESSING LAYER 6 ==========");
        $display("Input: 8×8×32, Output: 4×4×32 (stride=2)");
        
        for (out_h = 0; out_h < LAYER6_OUT_H; out_h++) begin
            for (out_w = 0; out_w < LAYER6_OUT_W; out_w++) begin
                input_h = out_h * 2;
                input_w = out_w * 2;
                
                for (out_ch = 0; out_ch < LAYER6_OUT_C; out_ch++) begin
                    accumulator = 0;
                    
                    for (in_ch = 0; in_ch < LAYER3_OUT_C; in_ch++) begin
                        extract_window_layer3(input_h, input_w, in_ch, window);
                        get_kernel_layer6(out_ch, in_ch, kernel);
                        
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
                    
                    layer6_output[linear_index_3d(out_h, out_w, out_ch, LAYER6_OUT_W, LAYER6_OUT_C)] = accumulator;
                end
            end
        end
        
        $display("Layer 6 Complete!");
    endtask
    
    //==========================================================================
    // Task: Flatten Layer 6 output (4×4×32 → 512 vector)
    //==========================================================================
    
    task automatic flatten_layer6(output logic [31:0] flattened [0:511]);
        int idx, h, w, c;
        
        $display("\n========== FLATTEN LAYER 6 ==========");
        
        idx = 0;
        for (h = 0; h < LAYER6_OUT_H; h++) begin
            for (w = 0; w < LAYER6_OUT_W; w++) begin
                for (c = 0; c < LAYER6_OUT_C; c++) begin
                    flattened[idx] = layer6_output[linear_index_3d(h, w, c, LAYER6_OUT_W, LAYER6_OUT_C)];
                    idx++;
                end
            end
        end
        
        $display("Flattening complete! Total elements: %0d", idx);
    endtask
    
    //==========================================================================
    // Task: Classifier - First FC Layer (512 → 64)
    //==========================================================================
    
    task automatic classifier_fc0(input logic [31:0] flattened [0:511]);
        int out_idx, in_idx;
        logic [31:0] accumulator;
        logic [7:0] w, a;
        
        $display("\n========== CLASSIFIER FC0 (512 → 64) ==========");
        
        for (out_idx = 0; out_idx < 64; out_idx++) begin
            accumulator = 0;
            
            // Process all 512 input channels
            for (in_idx = 0; in_idx < 512; in_idx++) begin
                a = saturate_u8(flattened[in_idx]);
                w = classifier0_weights[out_idx * 512 + in_idx];
                accumulator += $signed(a) * $signed(w);
            end
            
            classifier0_output[out_idx] = accumulator;
        end
        
        $display("FC0 complete! First 8 outputs:");
        for (int i = 0; i < 8; i++) 
            $display("  Out[%0d] = %0d", i, $signed(classifier0_output[i]));
    endtask
    
    //==========================================================================
    // Task: Classifier - Second FC Layer (64 → 10)
    //==========================================================================
    
    task automatic classifier_fc1(output int predicted_class);
        int out_idx, in_idx;
        logic [31:0] accumulator;
        logic [7:0] w, a;
        int max_idx;
        logic [31:0] max_val;
        
        $display("\n========== CLASSIFIER FC1 (64 → 10) ==========");
        
        for (out_idx = 0; out_idx < NUM_CLASSES; out_idx++) begin
            accumulator = 0;
            
            // Process all 64 input channels
            for (in_idx = 0; in_idx < 64; in_idx++) begin
                a = saturate_u8(classifier0_output[in_idx]);
                w = classifier2_weights[out_idx * 64 + in_idx];
                accumulator += $signed(a) * $signed(w);
            end
            
            classifier2_output[out_idx] = accumulator;
        end
        
        // Find argmax
        max_idx = 0;
        max_val = classifier2_output[0];
        for (int i = 1; i < NUM_CLASSES; i++) begin
            if ($signed(classifier2_output[i]) > $signed(max_val)) begin
                max_idx = i;
                max_val = classifier2_output[i];
            end
        end
        
        predicted_class = max_idx;
        
        $display("FC1 complete!");
        $display("\n========== CLASSIFICATION RESULTS ==========");
        $display("Output scores (logits):");
        for (int i = 0; i < NUM_CLASSES; i++) 
            $display("  Class %0d: %0d", i, $signed(classifier2_output[i]));
        
        $display("\n*** PREDICTED CLASS: %0d ***", max_idx);
        $display("*** CONFIDENCE: %0d ***", $signed(max_val));
    endtask
    
    //==========================================================================
    // Initial Block - Load Memories and Run Inference
    //==========================================================================
    
    initial begin
        int predicted_class;
        logic [31:0] flattened [0:511];
        
        $display("\n╔════════════════════════════════════════════════════════════╗");
        $display("║   QUANTIZED MobileNetV2 FULL INFERENCE TESTBENCH         ║");
        $display("║   Input: 32×32×3 (test_image.mem)                        ║");
        $display("║   Output: 10-class classification                         ║");
        $display("╚════════════════════════════════════════════════════════════╝\n");
        
        // Load test image
        $display("[1] Loading test image from test_image.mem...");
        $readmemh("test_image.mem", test_image);
        $display("    Loaded %0d bytes (32×32×3 image)", INPUT_H*INPUT_W*INPUT_C);
        
        // Load Layer 0 weights
        $display("[2] Loading Layer 0 weights (3→16 channels)...");
        $readmemh("tiny_mems_int8/features.0.weight_quantized.w1.mem", layer0_weights);
        $readmemh("tiny_mems_int8/features.0.weight_zero_point.w1.mem", layer0_zero_point);
        $display("    Loaded 432 weight values + zero point");
        
        // Load Layer 3 weights
        $display("[3] Loading Layer 3 weights (16→32 channels)...");
        $readmemh("tiny_mems_int8/features.3.weight_quantized.w1.mem", layer3_weights);
        $readmemh("tiny_mems_int8/features.3.weight_zero_point.w1.mem", layer3_zero_point);
        $display("    Loaded 4608 weight values + zero point");
        
        // Load Layer 6 weights
        $display("[4] Loading Layer 6 weights (32→32 channels)...");
        $readmemh("tiny_mems_int8/features.6.weight_quantized.w1.mem", layer6_weights);
        $readmemh("tiny_mems_int8/features.6.weight_zero_point.w1.mem", layer6_zero_point);
        $display("    Loaded 9216 weight values + zero point");
        
        // Load Classifier weights
        $display("[5] Loading Classifier weights...");
        $readmemh("tiny_mems_int8/classifier.0.weight_quantized.w1.mem", classifier0_weights);
        $readmemh("tiny_mems_int8/classifier.0.weight_zero_point.w1.mem", classifier0_zero_point);
        $readmemh("tiny_mems_int8/classifier.2.weight_quantized.w1.mem", classifier2_weights);
        $readmemh("tiny_mems_int8/classifier.2.weight_zero_point.w1.mem", classifier2_zero_point);
        $display("    Loaded classifier layer weights");
        
        // Load zero points
        $display("[6] Loading zero point values...");
        $readmemh("tiny_mems_int8/input_zero_point.w1.mem", input_zero_point);
        $readmemh("tiny_mems_int8/linear_zero_point.w1.mem", linear_zero_point);
        $readmemh("tiny_mems_int8/output_zero_point.w1.mem", output_zero_point);
        $display("    Loaded input, linear, and output zero points");
        
        // Initialize
        reset = 1;
        conv_start = 0;
        conv_clear = 0;
        repeat(5) @(posedge clock);
        reset = 0;
        @(posedge clock);
        
        $display("\n[7] Starting inference pipeline...\n");
        
        // Process layers
        process_layer0();
        repeat(10) @(posedge clock);
        
        process_layer3();
        repeat(10) @(posedge clock);
        
        process_layer6();
        repeat(10) @(posedge clock);
        
        // Flatten output
        flatten_layer6(flattened);
        repeat(10) @(posedge clock);
        
        // Classification
        classifier_fc0(flattened);
        repeat(10) @(posedge clock);
        
        classifier_fc1(predicted_class);
        repeat(10) @(posedge clock);
        
        $display("\n╔════════════════════════════════════════════════════════════╗");
        $display("║           INFERENCE COMPLETE AND SUCCESSFUL               ║");
        $display("╚════════════════════════════════════════════════════════════╝\n");
        
        repeat(10) @(posedge clock);
        $finish;
    end
    
    //==========================================================================
    // Timeout Protection
    //==========================================================================
    
    initial begin
        #100000000;
        $display("\nERROR: Simulation timeout!");
        $finish;
    end

endmodule
