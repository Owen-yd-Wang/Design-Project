//==============================================================================
// Module: depthwise_conv3x3_engine
// Description: Depthwise 3×3 Convolution Engine for MobileNetV2
//              Uses 9 parallel MAC units to compute one output pixel
//
// Operation:
//   - Receives a 3×3 window of input pixels (one channel)
//   - Receives a 3×3 kernel of weights
//   - Computes element-wise multiplication and sum using 9 MACs
//   - Outputs the convolution result
//
// Parallelism:
//   - 9 MAC units operate in parallel (one per kernel position)
//   - Single cycle multiply, followed by tree-based summation
//
// Usage in MobileNetV2:
//   - Processes one channel at a time
//   - Must be called separately for each input channel
//   - Typically followed by pointwise convolution
//
// Author: Cognichip Co-Design Team
//==============================================================================

module depthwise_conv3x3_engine (
    input  logic        clock,
    input  logic        reset,         // Active high reset
    
    // 3×3 Input window (flattened: [0]=top-left, [8]=bottom-right)
    // Layout: [0][1][2]
    //         [3][4][5]
    //         [6][7][8]
    input  logic [7:0]  window_in [8:0],    // 9 pixels from input feature map
    
    // 3×3 Kernel weights (flattened: same layout as window)
    input  logic [7:0]  kernel_weights [8:0], // 9 kernel weights
    
    // Control signals
    input  logic        start_conv,    // Start convolution computation
    input  logic        clear,         // Clear all accumulators
    
    // Output
    output logic [31:0] conv_result,   // Final convolution result
    output logic        result_valid   // Result is valid
);

    //==========================================================================
    // Internal Signals
    //==========================================================================
    
    // MAC unit outputs (one per kernel position)
    logic [31:0] mac_outputs [8:0];
    logic        mac_valids  [8:0];
    
    // MAC control signals
    logic        mac_enable;
    logic        mac_clear;
    
    // Summation tree signals
    logic [31:0] sum_stage1 [4:0];  // First stage: 9→5 partial sums
    logic [31:0] sum_stage2 [2:0];  // Second stage: 5→3 partial sums
    logic [31:0] sum_stage3 [1:0];  // Third stage: 3→2 partial sums
    logic [31:0] final_sum;         // Final stage: 2→1 result
    
    // Pipeline registers for valid signal
    logic        valid_stage1;
    logic        valid_stage2;
    logic        valid_stage3;
    logic        valid_final;
    
    //==========================================================================
    // 9 Parallel MAC Units (One per 3×3 kernel position)
    //==========================================================================
    
    generate
        for (genvar i = 0; i < 9; i++) begin : mac_array
            mac_uint8_int32 mac_inst (
                .clock      (clock),
                .reset      (reset),
                .data_in    (window_in[i]),      // Input pixel
                .weight_in  (kernel_weights[i]),  // Kernel weight
                .enable     (mac_enable),
                .clear_acc  (mac_clear),
                .acc_out    (mac_outputs[i]),    // Product output
                .valid      (mac_valids[i])
            );
        end
    endgenerate
    
    //==========================================================================
    // Control Logic
    //==========================================================================
    
    // MAC enable: trigger on start_conv
    assign mac_enable = start_conv;
    assign mac_clear  = clear;
    
    //==========================================================================
    // Summation Tree (Combines 9 MAC outputs into single result)
    //==========================================================================
    // 
    // Tree structure for efficient summation:
    //
    //   MAC0  MAC1  MAC2  MAC3  MAC4  MAC5  MAC6  MAC7  MAC8
    //     │    │     │     │     │     │     │     │     │
    //     └────┴─────┘     └─────┴─────┘     └─────┴─────┘
    //         │                 │                 │          MAC8
    //      sum[0]            sum[1]            sum[2]        │
    //         │                 │                 │          │
    //         └─────────────────┴─────────────────┴──────────┘
    //                           │
    //                      final_sum
    //
    //==========================================================================
    
    // Stage 1: Reduce 9 MACs to 5 partial sums
    always_comb begin
        sum_stage1[0] = mac_outputs[0] + mac_outputs[1];  // Top-left + Top-center
        sum_stage1[1] = mac_outputs[2] + mac_outputs[3];  // Top-right + Mid-left
        sum_stage1[2] = mac_outputs[4] + mac_outputs[5];  // Center + Mid-right
        sum_stage1[3] = mac_outputs[6] + mac_outputs[7];  // Bottom-left + Bottom-center
        sum_stage1[4] = mac_outputs[8];                    // Bottom-right (no pair)
    end
    
    // Stage 2: Reduce 5 to 3 partial sums
    always_comb begin
        sum_stage2[0] = sum_stage1[0] + sum_stage1[1];
        sum_stage2[1] = sum_stage1[2] + sum_stage1[3];
        sum_stage2[2] = sum_stage1[4];
    end
    
    // Stage 3: Reduce 3 to 2 partial sums
    always_comb begin
        sum_stage3[0] = sum_stage2[0] + sum_stage2[1];
        sum_stage3[1] = sum_stage2[2];
    end
    
    // Final stage: Reduce 2 to 1 final result
    always_comb begin
        final_sum = sum_stage3[0] + sum_stage3[1];
    end
    
    //==========================================================================
    // Output Pipeline
    //==========================================================================
    // Pipeline the valid signal through the summation tree stages
    // (In this design, summation is combinational, so valid propagates immediately)
    
    always_ff @(posedge clock) begin
        if (reset) begin
            valid_stage1 <= 1'b0;
            valid_stage2 <= 1'b0;
            valid_stage3 <= 1'b0;
            valid_final  <= 1'b0;
            conv_result  <= 32'b0;
        end else begin
            // Pipeline the valid signal
            valid_stage1 <= mac_valids[0];  // All MACs produce valid at same time
            valid_stage2 <= valid_stage1;
            valid_stage3 <= valid_stage2;
            valid_final  <= valid_stage3;
            
            // Register the final result
            conv_result <= final_sum;
        end
    end
    
    assign result_valid = valid_final;
    
    //==========================================================================
    // Debug Assertions (Synthesis will ignore these)
    //==========================================================================
    
    // synthesis translate_off
    
    // Monitor for potential overflow in summation
    always_ff @(posedge clock) begin
        if (!reset && result_valid) begin
            if (conv_result > 32'h7FFF_FFFF || conv_result < 32'h8000_0000) begin
                $display("INFO: Large convolution result detected: %0d", $signed(conv_result));
            end
        end
    end
    
    // Display computation when enabled
    always_ff @(posedge clock) begin
        if (!reset && start_conv) begin
            $display("\n=== Depthwise Conv 3×3 Computation ===");
            $display("Window:   [%3d %3d %3d]", window_in[0], window_in[1], window_in[2]);
            $display("          [%3d %3d %3d]", window_in[3], window_in[4], window_in[5]);
            $display("          [%3d %3d %3d]", window_in[6], window_in[7], window_in[8]);
            $display("Kernel:   [%3d %3d %3d]", kernel_weights[0], kernel_weights[1], kernel_weights[2]);
            $display("          [%3d %3d %3d]", kernel_weights[3], kernel_weights[4], kernel_weights[5]);
            $display("          [%3d %3d %3d]", kernel_weights[6], kernel_weights[7], kernel_weights[8]);
        end
        
        if (!reset && result_valid) begin
            $display("Result: %0d", $signed(conv_result));
            $display("======================================\n");
        end
    end
    
    // synthesis translate_on

endmodule
