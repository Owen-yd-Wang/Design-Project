//==============================================================================
// Module: inverted_residual_block
// Description: MobileNetV2 Inverted Residual Block (Bottleneck)
//              Integrates: Pointwise Expansion → Depthwise 3×3 → Pointwise Projection
//
// Architecture:
//   1. Pointwise 1×1 Expansion (increase channels)
//   2. ReLU6 activation
//   3. Depthwise 3×3 Convolution (spatial filtering)
//   4. ReLU6 activation
//   5. Pointwise 1×1 Projection (decrease channels)
//   6. Skip connection (if stride=1 and input_ch == output_ch)
//
// Simplified Implementation:
//   - Processes one spatial position at a time
//   - Weights must be pre-loaded
//   - Suitable for resource-constrained FPGAs
//
// Author: Cognichip Co-Design Team
//==============================================================================

module inverted_residual_block #(
    parameter int NUM_MACS = 16,           // Number of parallel MACs
    parameter int MAX_CHANNELS = 320       // Maximum channels supported
)(
    input  logic        clock,
    input  logic        reset,             // Active high reset
    
    // Configuration (set before processing)
    input  logic [9:0]  input_channels,    // Number of input channels
    input  logic [9:0]  expand_channels,   // Expanded channels (typically 6× input)
    input  logic [9:0]  output_channels,   // Number of output channels
    input  logic        use_residual,      // Enable skip connection
    
    // Input activation (one spatial position, all input channels)
    input  logic [7:0]  input_activation [NUM_MACS-1:0],
    input  logic        input_valid,
    
    // Depthwise 3×3 window (from line buffer)
    input  logic [7:0]  dw_window [8:0],
    input  logic        dw_window_valid,
    
    // Weight interfaces (streaming)
    input  logic [7:0]  expand_weights [NUM_MACS-1:0],    // Expansion weights
    input  logic [7:0]  dw_kernel_weights [8:0],          // Depthwise kernel
    input  logic [7:0]  project_weights [NUM_MACS-1:0],   // Projection weights
    
    // Control
    input  logic        start_block,       // Start processing
    input  logic        load_expand_data,  // Load expansion layer data
    input  logic        load_project_data, // Load projection layer data
    
    // Output
    output logic [7:0]  output_activation, // Single output channel result
    output logic        output_valid,      // Output is valid
    output logic        busy               // Block is processing
);

    //==========================================================================
    // Internal Signals - Expansion Stage
    //==========================================================================
    
    logic [31:0] expand_result;
    logic        expand_valid;
    logic        expand_busy;
    
    //==========================================================================
    // Internal Signals - ReLU6 after Expansion
    //==========================================================================
    
    logic [7:0]  expand_activated;
    logic        expand_activated_valid;
    
    //==========================================================================
    // Internal Signals - Depthwise Stage
    //==========================================================================
    
    logic [31:0] dw_result;
    logic        dw_valid;
    logic        dw_start;
    logic        dw_clear;
    
    //==========================================================================
    // Internal Signals - ReLU6 after Depthwise
    //==========================================================================
    
    logic [7:0]  dw_activated;
    logic        dw_activated_valid;
    
    //==========================================================================
    // Internal Signals - Projection Stage
    //==========================================================================
    
    logic [31:0] project_result;
    logic        project_valid;
    logic        project_busy;
    
    //==========================================================================
    // Internal Signals - Skip Connection
    //==========================================================================
    
    logic [7:0]  residual_input;  // Stored input for skip connection
    logic [8:0]  skip_sum;        // Sum with saturation
    
    //==========================================================================
    // FSM States
    //==========================================================================
    
    typedef enum logic [2:0] {
        IDLE,
        EXPAND,
        DEPTHWISE,
        PROJECT,
        SKIP_ADD,
        DONE
    } state_t;
    
    state_t current_state, next_state;
    
    //==========================================================================
    // Module Instantiations
    //==========================================================================
    
    // Expansion Layer (Pointwise 1×1)
    pointwise_conv1x1_engine #(
        .NUM_MACS(NUM_MACS)
    ) expansion_layer (
        .clock              (clock),
        .reset              (reset),
        .num_input_channels (input_channels),
        .num_output_channels(expand_channels),
        .activations        (input_activation),
        .weights            (expand_weights),
        .start_conv         (start_block && current_state == EXPAND),
        .clear              (reset),
        .load_data          (load_expand_data),
        .conv_result        (expand_result),
        .result_valid       (expand_valid),
        .busy               (expand_busy)
    );
    
    // ReLU6 after Expansion
    relu6_quantized expansion_relu (
        .clock          (clock),
        .reset          (reset),
        .conv_result    (expand_result),
        .valid_in       (expand_valid),
        .relu6_max      (8'd255),  // Max uint8
        .activation_out (expand_activated),
        .valid_out      (expand_activated_valid)
    );
    
    // Depthwise 3×3 Convolution
    depthwise_conv3x3_engine depthwise_layer (
        .clock          (clock),
        .reset          (reset),
        .window_in      (dw_window),
        .kernel_weights (dw_kernel_weights),
        .start_conv     (dw_start),
        .clear          (dw_clear),
        .conv_result    (dw_result),
        .result_valid   (dw_valid)
    );
    
    // ReLU6 after Depthwise
    relu6_quantized depthwise_relu (
        .clock          (clock),
        .reset          (reset),
        .conv_result    (dw_result),
        .valid_in       (dw_valid),
        .relu6_max      (8'd255),
        .activation_out (dw_activated),
        .valid_out      (dw_activated_valid)
    );
    
    // Projection Layer (Pointwise 1×1)
    pointwise_conv1x1_engine #(
        .NUM_MACS(NUM_MACS)
    ) projection_layer (
        .clock              (clock),
        .reset              (reset),
        .num_input_channels (expand_channels),
        .num_output_channels(output_channels),
        .activations        (input_activation),  // Would be dw_activated in full impl
        .weights            (project_weights),
        .start_conv         (current_state == PROJECT),
        .clear              (reset),
        .load_data          (load_project_data),
        .conv_result        (project_result),
        .result_valid       (project_valid),
        .busy               (project_busy)
    );
    
    //==========================================================================
    // Control FSM
    //==========================================================================
    
    always_ff @(posedge clock) begin
        if (reset) begin
            current_state <= IDLE;
        end else begin
            current_state <= next_state;
        end
    end
    
    // FSM Next State Logic
    always_comb begin
        next_state = current_state;
        
        case (current_state)
            IDLE: begin
                if (start_block) begin
                    next_state = EXPAND;
                end
            end
            
            EXPAND: begin
                if (expand_valid) begin
                    next_state = DEPTHWISE;
                end
            end
            
            DEPTHWISE: begin
                if (dw_valid) begin
                    next_state = PROJECT;
                end
            end
            
            PROJECT: begin
                if (project_valid) begin
                    if (use_residual) begin
                        next_state = SKIP_ADD;
                    end else begin
                        next_state = DONE;
                    end
                end
            end
            
            SKIP_ADD: begin
                next_state = DONE;
            end
            
            DONE: begin
                next_state = IDLE;
            end
            
            default: next_state = IDLE;
        endcase
    end
    
    //==========================================================================
    // Depthwise Control Signals
    //==========================================================================
    
    assign dw_start = (current_state == DEPTHWISE) && dw_window_valid;
    assign dw_clear = reset || (current_state == IDLE);
    
    //==========================================================================
    // Skip Connection Logic
    //==========================================================================
    
    // Store input for skip connection
    always_ff @(posedge clock) begin
        if (reset) begin
            residual_input <= 8'b0;
        end else if (input_valid && use_residual) begin
            residual_input <= input_activation[0];  // Simplified: take first channel
        end
    end
    
    // Add with saturation
    always_comb begin
        if (use_residual && current_state == SKIP_ADD) begin
            skip_sum = {1'b0, project_result[7:0]} + {1'b0, residual_input};
            // Saturate to uint8
            if (skip_sum > 255) begin
                skip_sum = 255;
            end
        end else begin
            skip_sum = {1'b0, project_result[7:0]};
        end
    end
    
    //==========================================================================
    // Output Logic
    //==========================================================================
    
    always_ff @(posedge clock) begin
        if (reset) begin
            output_activation <= 8'b0;
            output_valid <= 1'b0;
            busy <= 1'b0;
        end else begin
            // Output valid when processing complete
            output_valid <= (current_state == DONE);
            
            // Output value (with or without skip connection)
            if (current_state == DONE) begin
                output_activation <= skip_sum[7:0];
            end
            
            // Busy during processing
            busy <= (current_state != IDLE) && (current_state != DONE);
        end
    end
    
    //==========================================================================
    // Debug Monitoring
    //==========================================================================
    
    // synthesis translate_off
    
    // always_ff @(posedge clock) begin
    //     if (!reset && start_block) begin
    //         $display("\n[INVERTED_RESIDUAL_BLOCK] Starting block processing");
    //         $display("  Input channels: %0d", input_channels);
    //         $display("  Expand channels: %0d", expand_channels);
    //         $display("  Output channels: %0d", output_channels);
    //         $display("  Use residual: %0d", use_residual);
    //     end
        
    //     if (!reset) begin
    //         case (current_state)
    //             EXPAND: if (expand_valid)
    //                 $display("  [EXPAND] Complete, result: %0d → activated: %0d", 
    //                          expand_result, expand_activated);
    //             DEPTHWISE: if (dw_valid)
    //                 $display("  [DEPTHWISE] Complete, result: %0d → activated: %0d", 
    //                          dw_result, dw_activated);
    //             PROJECT: if (project_valid)
    //                 $display("  [PROJECT] Complete, result: %0d", project_result[7:0]);
    //             SKIP_ADD:
    //                 $display("  [SKIP] Adding residual: %0d + %0d = %0d", 
    //                          project_result[7:0], residual_input, skip_sum);
    //             DONE:
    //                 $display("  [DONE] Output: %0d\n", output_activation);
    //             default: ;
    //         endcase
    //     end
    // end
    
    // synthesis translate_on

endmodule
