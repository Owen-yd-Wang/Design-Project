//==============================================================================
// Module: relu6_quantized
// Description: ReLU6 Activation with Requantization for MobileNetV2
//              Converts int32 convolution output to uint8 with ReLU6 activation
//
// Operation:
//   1. Take int32 convolution result
//   2. Apply requantization (scale factor)
//   3. Apply ReLU6 clipping: clamp(x, 0, 6) in quantized domain
//   4. Output uint8 value
//
// Quantization:
//   - Input: int32 (accumulated convolution result)
//   - Output: uint8 (quantized activation)
//   - ReLU6 bounds in uint8: [zero_point, zero_point + 6*scale]
//
// For simplicity, this implementation assumes:
//   - Input scale = output scale (simplified quantization)
//   - Zero point = 0 (symmetric quantization)
//   - ReLU6 bounds: [0, 255] in uint8 domain
//
// Author: Cognichip Co-Design Team
//==============================================================================

module relu6_quantized (
    input  logic        clock,
    input  logic        reset,         // Active high reset
    
    // Input from convolution (int32)
    input  logic [31:0] conv_result,
    input  logic        valid_in,
    
    // Quantization parameters (configurable per layer)
    input  logic [7:0]  relu6_max,     // Maximum value for ReLU6 in uint8 (typically 255 for scale=1)
    
    // Output (uint8)
    output logic [7:0]  activation_out,
    output logic        valid_out
);

    //==========================================================================
    // Internal Signals
    //==========================================================================
    
    logic signed [31:0] signed_input;
    logic        [31:0] clamped_value;
    logic        [7:0]  output_reg;
    logic               valid_reg;
    
    //==========================================================================
    // ReLU6 Logic
    //==========================================================================
    
    // Convert to signed for comparison
    assign signed_input = $signed(conv_result);
    
    // Apply ReLU6 clipping
    always_comb begin
        if (signed_input < 0) begin
            // ReLU: clip negative values to 0
            clamped_value = 32'b0;
        end else if (signed_input > relu6_max) begin
            // ReLU6: clip values above max to max
            clamped_value = {24'b0, relu6_max};
        end else begin
            // Pass through values in [0, relu6_max]
            clamped_value = conv_result;
        end
    end
    
    //==========================================================================
    // Output Register (for timing)
    //==========================================================================
    
    always_ff @(posedge clock) begin
        if (reset) begin
            output_reg <= 8'b0;
            valid_reg  <= 1'b0;
        end else begin
            // Register the clamped output
            output_reg <= clamped_value[7:0];  // Take lower 8 bits as uint8
            valid_reg  <= valid_in;
        end
    end
    
    assign activation_out = output_reg;
    assign valid_out = valid_reg;
    
    //==========================================================================
    // Debug Assertions
    //==========================================================================
    
    // synthesis translate_off
    
    // Monitor activations
    always_ff @(posedge clock) begin
        if (!reset && valid_in) begin
            if (signed_input < 0) begin
                $display("ReLU6: Input %0d clipped to 0", signed_input);
            end else if (signed_input > relu6_max) begin
                $display("ReLU6: Input %0d clipped to %0d", signed_input, relu6_max);
            end
        end
    end
    
    // synthesis translate_on

endmodule
