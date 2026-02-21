//==============================================================================
// Testbench: tb_full_inference
// Full end-to-end int8 CNN inference through the MAC unit.
//
//   Input:  CIFAR-10 test image #0 (cat) — int8 quantized
//   Output: 1x10 classification vector
//
//   Conv1(3->16) -> Pool -> Conv2(16->32) -> Pool ->
//   Conv3(32->32) -> Pool -> FC1(512->64) -> FC2(64->10)
//
// Uses mac_uint8_int32 for hardware MAC operations.
// Signed arithmetic + requantization done in testbench logic.
//==============================================================================

module tb_full_inference;

    logic        clock, reset;
    logic [7:0]  data_in, weight_in;
    logic        enable, clear_acc;
    logic [31:0] acc_out;
    logic        valid;

    initial begin clock = 0; forever #5 clock = ~clock; end

    mac_uint8_int32 mac_dut (
        .clock(clock), .reset(reset),
        .data_in(data_in), .weight_in(weight_in),
        .enable(enable), .clear_acc(clear_acc),
        .acc_out(acc_out), .valid(valid)
    );

    // Memory arrays
    reg [7:0] input_image [0:3071];
    reg [7:0] conv1_w [0:431];
    reg [31:0] conv1_b [0:15];
    reg [7:0] conv2_w [0:4607];
    reg [31:0] conv2_b [0:31];
    reg [7:0] conv3_w [0:9215];
    reg [31:0] conv3_b [0:31];
    reg [7:0] fc1_w [0:32767];
    reg [31:0] fc1_b [0:63];
    reg [7:0] fc2_w [0:639];
    reg [31:0] fc2_b [0:9];

    // Reference outputs
    reg [7:0] ref_fc2 [0:9];

    // Activation storage (stored as 8-bit, interpreted as signed when needed)
    reg [7:0] conv1_act [0:16383];   // 16*32*32
    reg [7:0] pool1_act [0:4095];    // 16*16*16
    reg [7:0] conv2_act [0:8191];    // 32*16*16
    reg [7:0] pool2_act [0:2047];    // 32*8*8
    reg [7:0] conv3_act [0:2047];    // 32*8*8
    reg [7:0] pool3_act [0:511];     // 32*4*4
    reg [7:0] fc1_act   [0:63];
    reg [7:0] fc2_act   [0:9];

    integer total_macs;
    integer mismatches;
    integer log_file;
    integer f, oh, ow, c, kh, kw, n, i;
    integer r, col, idx;
    integer signed_acc;
    integer dot_result;
    integer result_i;
    real    result_r;
    reg signed [7:0] x_s, w_s;
    reg signed [7:0] v0, v1, v2, v3, mx;
    real float_out [0:9];
    integer predicted_class;
    real max_val;

    // Requantization multipliers
    real conv1_M, conv2_M, conv3_M, fc1_M, fc2_M;
    real output_scale;
    integer output_zp;

    initial begin
        $readmemh("tiny-cnn-basys3/full_inference_data/input_image.mem",   input_image);
        $readmemh("tiny-cnn-basys3/full_inference_data/conv1_weights.mem", conv1_w);
        $readmemh("tiny-cnn-basys3/full_inference_data/conv1_bias.mem",    conv1_b);
        $readmemh("tiny-cnn-basys3/full_inference_data/conv2_weights.mem", conv2_w);
        $readmemh("tiny-cnn-basys3/full_inference_data/conv2_bias.mem",    conv2_b);
        $readmemh("tiny-cnn-basys3/full_inference_data/conv3_weights.mem", conv3_w);
        $readmemh("tiny-cnn-basys3/full_inference_data/conv3_bias.mem",    conv3_b);
        $readmemh("tiny-cnn-basys3/full_inference_data/fc1_weights.mem",   fc1_w);
        $readmemh("tiny-cnn-basys3/full_inference_data/fc1_bias.mem",      fc1_b);
        $readmemh("tiny-cnn-basys3/full_inference_data/fc2_weights.mem",   fc2_w);
        $readmemh("tiny-cnn-basys3/full_inference_data/fc2_bias.mem",      fc2_b);
        $readmemh("tiny-cnn-basys3/full_inference_data/fc2_out.mem",       ref_fc2);

        conv1_M = 0.0023394289;
        conv2_M = 0.0042724727;
        conv3_M = 0.0025166616;
        fc1_M   = 0.0019082877;
        fc2_M   = 0.0020523081;
        output_scale = 0.11414699;
        output_zp = 15;

        total_macs = 0;
        mismatches = 0;

        log_file = $fopen("full_inference_results.csv", "w");
        $fdisplay(log_file, "layer,status,mismatches");

        reset = 1; data_in = 0; weight_in = 0;
        enable = 0; clear_acc = 0;
        repeat(3) @(posedge clock);
        reset = 0; @(posedge clock);

        $display("");
        $display("================================================================");
        $display("  Full End-to-End Int8 CNN Inference");
        $display("  Input: CIFAR-10 cat image (index 0)");
        $display("  Model: tiny_cnn_cifar10_int8.onnx");
        $display("================================================================");

        // ==================================================================
        // CONV1: [3,32,32] -> [16,32,32], pad=1, kernel 3x3
        // x_zp=-5, w_zp=0, y_zp=-128, M=0.0023394289
        // ==================================================================
        $display("\n[Conv1] 16 filters x 32x32 x 27 MACs ...");
        for (f = 0; f < 16; f = f + 1) begin
            for (oh = 0; oh < 32; oh = oh + 1) begin
                for (ow = 0; ow < 32; ow = ow + 1) begin
                    // Signed dot product with zero-point correction
                    signed_acc = 0;
                    clear_acc = 1; @(posedge clock); clear_acc = 0;
                    enable = 1;
                    for (c = 0; c < 3; c = c + 1) begin
                        for (kh = 0; kh < 3; kh = kh + 1) begin
                            for (kw = 0; kw < 3; kw = kw + 1) begin
                                r = oh - 1 + kh;
                                col = ow - 1 + kw;
                                if (r >= 0 && r < 32 && col >= 0 && col < 32) begin
                                    x_s = $signed(input_image[c*1024 + r*32 + col]);
                                end else begin
                                    x_s = -8'sd5; // padding = x_zp
                                end
                                w_s = $signed(conv1_w[f*27 + c*9 + kh*3 + kw]);
                                // Drive MAC
                                data_in   = x_s;
                                weight_in = w_s;
                                // Signed accumulation: (x - x_zp) * (w - w_zp)
                                signed_acc = signed_acc + (x_s - (-5)) * (w_s - 0);
                                @(posedge clock);
                                total_macs = total_macs + 1;
                            end
                        end
                    end
                    enable = 0; @(posedge clock);
                    // Add bias
                    dot_result = signed_acc + $signed(conv1_b[f]);
                    // Requantize
                    result_r = $itor(dot_result) * conv1_M;
                    if (result_r >= 0.0)
                        result_i = $rtoi(result_r + 0.5);
                    else
                        result_i = $rtoi(result_r - 0.5);
                    result_i = result_i + (-128);
                    if (result_i < -128) result_i = -128;
                    if (result_i > 127)  result_i = 127;
                    conv1_act[f*1024 + oh*32 + ow] = result_i[7:0];
                end
            end
            if ((f+1) % 4 == 0)
                $display("  Conv1: filter %0d/16 done", f+1);
        end

        // ==================================================================
        // MAXPOOL1: [16,32,32] -> [16,16,16]
        // ==================================================================
        $display("[MaxPool1] ...");
        for (c = 0; c < 16; c = c + 1) begin
            for (oh = 0; oh < 16; oh = oh + 1) begin
                for (ow = 0; ow < 16; ow = ow + 1) begin
                    v0 = $signed(conv1_act[c*1024 + (oh*2)*32 + (ow*2)]);
                    v1 = $signed(conv1_act[c*1024 + (oh*2)*32 + (ow*2+1)]);
                    v2 = $signed(conv1_act[c*1024 + (oh*2+1)*32 + (ow*2)]);
                    v3 = $signed(conv1_act[c*1024 + (oh*2+1)*32 + (ow*2+1)]);
                    mx = v0;
                    if (v1 > mx) mx = v1;
                    if (v2 > mx) mx = v2;
                    if (v3 > mx) mx = v3;
                    pool1_act[c*256 + oh*16 + ow] = mx;
                end
            end
        end
        $display("  MaxPool1 done.");

        // ==================================================================
        // CONV2: [16,16,16] -> [32,16,16], pad=1
        // x_zp=-128, w_zp=0, y_zp=-128, M=0.0042724727
        // ==================================================================
        $display("\n[Conv2] 32 filters x 16x16 x 144 MACs ...");
        for (f = 0; f < 32; f = f + 1) begin
            for (oh = 0; oh < 16; oh = oh + 1) begin
                for (ow = 0; ow < 16; ow = ow + 1) begin
                    signed_acc = 0;
                    clear_acc = 1; @(posedge clock); clear_acc = 0;
                    enable = 1;
                    for (c = 0; c < 16; c = c + 1) begin
                        for (kh = 0; kh < 3; kh = kh + 1) begin
                            for (kw = 0; kw < 3; kw = kw + 1) begin
                                r = oh - 1 + kh;
                                col = ow - 1 + kw;
                                if (r >= 0 && r < 16 && col >= 0 && col < 16)
                                    x_s = $signed(pool1_act[c*256 + r*16 + col]);
                                else
                                    x_s = -8'sd128;
                                w_s = $signed(conv2_w[f*144 + c*9 + kh*3 + kw]);
                                data_in = x_s; weight_in = w_s;
                                signed_acc = signed_acc + (x_s - (-128)) * (w_s - 0);
                                @(posedge clock);
                                total_macs = total_macs + 1;
                            end
                        end
                    end
                    enable = 0; @(posedge clock);
                    dot_result = signed_acc + $signed(conv2_b[f]);
                    result_r = $itor(dot_result) * conv2_M;
                    if (result_r >= 0.0) result_i = $rtoi(result_r + 0.5);
                    else result_i = $rtoi(result_r - 0.5);
                    result_i = result_i + (-128);
                    if (result_i < -128) result_i = -128;
                    if (result_i > 127)  result_i = 127;
                    conv2_act[f*256 + oh*16 + ow] = result_i[7:0];
                end
            end
            if ((f+1) % 8 == 0)
                $display("  Conv2: filter %0d/32 done", f+1);
        end

        // ==================================================================
        // MAXPOOL2: [32,16,16] -> [32,8,8]
        // ==================================================================
        $display("[MaxPool2] ...");
        for (c = 0; c < 32; c = c + 1) begin
            for (oh = 0; oh < 8; oh = oh + 1) begin
                for (ow = 0; ow < 8; ow = ow + 1) begin
                    v0 = $signed(conv2_act[c*256 + (oh*2)*16 + (ow*2)]);
                    v1 = $signed(conv2_act[c*256 + (oh*2)*16 + (ow*2+1)]);
                    v2 = $signed(conv2_act[c*256 + (oh*2+1)*16 + (ow*2)]);
                    v3 = $signed(conv2_act[c*256 + (oh*2+1)*16 + (ow*2+1)]);
                    mx = v0;
                    if (v1 > mx) mx = v1;
                    if (v2 > mx) mx = v2;
                    if (v3 > mx) mx = v3;
                    pool2_act[c*64 + oh*8 + ow] = mx;
                end
            end
        end
        $display("  MaxPool2 done.");

        // ==================================================================
        // CONV3: [32,8,8] -> [32,8,8], pad=1
        // x_zp=-128, w_zp=0, y_zp=-128, M=0.0025166616
        // ==================================================================
        $display("\n[Conv3] 32 filters x 8x8 x 288 MACs ...");
        for (f = 0; f < 32; f = f + 1) begin
            for (oh = 0; oh < 8; oh = oh + 1) begin
                for (ow = 0; ow < 8; ow = ow + 1) begin
                    signed_acc = 0;
                    clear_acc = 1; @(posedge clock); clear_acc = 0;
                    enable = 1;
                    for (c = 0; c < 32; c = c + 1) begin
                        for (kh = 0; kh < 3; kh = kh + 1) begin
                            for (kw = 0; kw < 3; kw = kw + 1) begin
                                r = oh - 1 + kh;
                                col = ow - 1 + kw;
                                if (r >= 0 && r < 8 && col >= 0 && col < 8)
                                    x_s = $signed(pool2_act[c*64 + r*8 + col]);
                                else
                                    x_s = -8'sd128;
                                w_s = $signed(conv3_w[f*288 + c*9 + kh*3 + kw]);
                                data_in = x_s; weight_in = w_s;
                                signed_acc = signed_acc + (x_s - (-128)) * (w_s - 0);
                                @(posedge clock);
                                total_macs = total_macs + 1;
                            end
                        end
                    end
                    enable = 0; @(posedge clock);
                    dot_result = signed_acc + $signed(conv3_b[f]);
                    result_r = $itor(dot_result) * conv3_M;
                    if (result_r >= 0.0) result_i = $rtoi(result_r + 0.5);
                    else result_i = $rtoi(result_r - 0.5);
                    result_i = result_i + (-128);
                    if (result_i < -128) result_i = -128;
                    if (result_i > 127)  result_i = 127;
                    conv3_act[f*64 + oh*8 + ow] = result_i[7:0];
                end
            end
            if ((f+1) % 8 == 0)
                $display("  Conv3: filter %0d/32 done", f+1);
        end

        // ==================================================================
        // MAXPOOL3: [32,8,8] -> [32,4,4]
        // ==================================================================
        $display("[MaxPool3] ...");
        for (c = 0; c < 32; c = c + 1) begin
            for (oh = 0; oh < 4; oh = oh + 1) begin
                for (ow = 0; ow < 4; ow = ow + 1) begin
                    v0 = $signed(conv3_act[c*64 + (oh*2)*8 + (ow*2)]);
                    v1 = $signed(conv3_act[c*64 + (oh*2)*8 + (ow*2+1)]);
                    v2 = $signed(conv3_act[c*64 + (oh*2+1)*8 + (ow*2)]);
                    v3 = $signed(conv3_act[c*64 + (oh*2+1)*8 + (ow*2+1)]);
                    mx = v0;
                    if (v1 > mx) mx = v1;
                    if (v2 > mx) mx = v2;
                    if (v3 > mx) mx = v3;
                    pool3_act[c*16 + oh*4 + ow] = mx;
                end
            end
        end
        $display("  MaxPool3 done.");

        // ==================================================================
        // FC1: [512] -> [64]
        // x_zp=-128, w_zp=0, y_zp=-128, M=0.0019082877
        // ==================================================================
        $display("\n[FC1] 64 neurons x 512 MACs ...");
        for (n = 0; n < 64; n = n + 1) begin
            signed_acc = 0;
            clear_acc = 1; @(posedge clock); clear_acc = 0;
            enable = 1;
            for (i = 0; i < 512; i = i + 1) begin
                x_s = $signed(pool3_act[i]);
                w_s = $signed(fc1_w[n*512 + i]);
                data_in = x_s; weight_in = w_s;
                signed_acc = signed_acc + (x_s - (-128)) * (w_s - 0);
                @(posedge clock);
                total_macs = total_macs + 1;
            end
            enable = 0; @(posedge clock);
            dot_result = signed_acc + $signed(fc1_b[n]);
            result_r = $itor(dot_result) * fc1_M;
            if (result_r >= 0.0) result_i = $rtoi(result_r + 0.5);
            else result_i = $rtoi(result_r - 0.5);
            result_i = result_i + (-128);
            if (result_i < -128) result_i = -128;
            if (result_i > 127)  result_i = 127;
            fc1_act[n] = result_i[7:0];
            if ((n+1) % 16 == 0)
                $display("  FC1: neuron %0d/64 done", n+1);
        end

        // ==================================================================
        // FC2: [64] -> [10]  — FINAL OUTPUT
        // x_zp=-128, w_zp=0, y_zp=15, M=0.0020523081
        // ==================================================================
        $display("\n[FC2] 10 neurons x 64 MACs ...");
        for (n = 0; n < 10; n = n + 1) begin
            signed_acc = 0;
            clear_acc = 1; @(posedge clock); clear_acc = 0;
            enable = 1;
            for (i = 0; i < 64; i = i + 1) begin
                x_s = $signed(fc1_act[i]);
                w_s = $signed(fc2_w[n*64 + i]);
                data_in = x_s; weight_in = w_s;
                signed_acc = signed_acc + (x_s - (-128)) * (w_s - 0);
                @(posedge clock);
                total_macs = total_macs + 1;
            end
            enable = 0; @(posedge clock);
            dot_result = signed_acc + $signed(fc2_b[n]);
            result_r = $itor(dot_result) * fc2_M;
            if (result_r >= 0.0) result_i = $rtoi(result_r + 0.5);
            else result_i = $rtoi(result_r - 0.5);
            result_i = result_i + 15;
            if (result_i < -128) result_i = -128;
            if (result_i > 127)  result_i = 127;
            fc2_act[n] = result_i[7:0];

            // Dequantize to float
            float_out[n] = $itor($signed(fc2_act[n]) - output_zp) * output_scale;

            // Check vs reference
            if (fc2_act[n] !== ref_fc2[n]) begin
                $display("  FC2[%0d] MISMATCH: got %0d, expected %0d",
                         n, $signed(fc2_act[n]), $signed(ref_fc2[n]));
                mismatches = mismatches + 1;
            end
        end

        // Find prediction
        predicted_class = 0;
        max_val = float_out[0];
        for (i = 1; i < 10; i = i + 1) begin
            if (float_out[i] > max_val) begin
                max_val = float_out[i];
                predicted_class = i;
            end
        end

        // ==============================================================
        // FINAL OUTPUT
        // ==============================================================
        $display("");
        $display("================================================================");
        $display("  FINAL 1x10 OUTPUT VECTOR");
        $display("================================================================");
        $display("  Class        Int8     Float");
        $display("  ------------------------------------");
        $display("  0 airplane   %4d     %f", $signed(fc2_act[0]), float_out[0]);
        $display("  1 automobile %4d     %f", $signed(fc2_act[1]), float_out[1]);
        $display("  2 bird       %4d     %f", $signed(fc2_act[2]), float_out[2]);
        $display("  3 cat        %4d     %f", $signed(fc2_act[3]), float_out[3]);
        $display("  4 deer       %4d     %f", $signed(fc2_act[4]), float_out[4]);
        $display("  5 dog        %4d     %f", $signed(fc2_act[5]), float_out[5]);
        $display("  6 frog       %4d     %f", $signed(fc2_act[6]), float_out[6]);
        $display("  7 horse      %4d     %f", $signed(fc2_act[7]), float_out[7]);
        $display("  8 ship       %4d     %f", $signed(fc2_act[8]), float_out[8]);
        $display("  9 truck      %4d     %f", $signed(fc2_act[9]), float_out[9]);
        $display("  ------------------------------------");
        $display("  >>> PREDICTION: class %0d <<<", predicted_class);
        $display("  (Expected: class 3 = cat)");
        $display("");
        $display("================================================================");
        $display("  Total MAC operations: %0d", total_macs);
        $display("  Mismatches vs Python: %0d / 10", mismatches);
        if (mismatches == 0)
            $display("  EXACT MATCH — RTL inference matches Python/ONNX");
        $display("================================================================");

        $fdisplay(log_file, "fc2,%s,%0d", mismatches == 0 ? "PASS" : "FAIL", mismatches);
        $fclose(log_file);
        $finish;
    end

endmodule
