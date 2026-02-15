//==============================================================================
// Module: mac_uint8_int32
// Description: Multiply-Accumulate unit for uint8 precision with int32 accumulator
//              Designed for quantized neural network inference (MobileNetV2)
//
// Features:
//   - uint8 x uint8 multiplication
//   - 32-bit signed accumulation
//   - Synchronous clear
//   - Enable control
//   - Registered outputs for timing
//
// Author: Cognichip Co-Design Team
//==============================================================================

module mac_uint8_int32 (
    input  logic        clock,
    input  logic        reset,         // Active high reset
    
    // Data inputs
    input  logic [7:0]  data_in,       // uint8 activation/input data
    input  logic [7:0]  weight_in,     // uint8 weight
    
    // Control signals
    input  logic        enable,        // Enable MAC operation
    input  logic        clear_acc,     // Clear accumulator (active high)
    
    // Outputs
    output logic [31:0] acc_out,       // int32 accumulated result
    output logic        valid          // Output valid indicator
);

    //==========================================================================
    // Internal Signals
    //==========================================================================
    
    logic [15:0] product;              // uint8 x uint8 = uint16
    logic [31:0] accumulator;          // 32-bit accumulator register
    logic [31:0] next_accumulator;     // Next accumulator value
    
    //==========================================================================
    // Multiplication Stage
    //==========================================================================
    
    // Combinational multiplication
    // uint8 × uint8 → uint16 (max: 255 × 255 = 65,025)
    always_comb begin
        product = data_in * weight_in;
    end
    
    //==========================================================================
    // Accumulation Logic
    //==========================================================================
    
    // Accumulator next-state logic
    always_comb begin
        if (clear_acc) begin
            // Clear accumulator
            next_accumulator = 32'b0;
        end else if (enable) begin
            // Accumulate: add product to current accumulator
            // Extend uint16 product to int32 (zero-extend since unsigned)
            next_accumulator = accumulator + {16'b0, product};
        end else begin
            // Hold current value
            next_accumulator = accumulator;
        end
    end
    
    //==========================================================================
    // Accumulator Register
    //==========================================================================
    
    always_ff @(posedge clock) begin
        if (reset) begin
            accumulator <= 32'b0;
            valid       <= 1'b0;
        end else begin
            accumulator <= next_accumulator;
            // Valid goes high after first accumulation
            valid       <= enable | clear_acc;
        end
    end
    
    //==========================================================================
    // Output Assignment
    //==========================================================================
    
    assign acc_out = accumulator;
    
    //==========================================================================
    // Assertions for Simulation (optional, can be disabled for synthesis)
    //==========================================================================
    
    // synthesis translate_off
    
    // Check for potential overflow (informational warning)
    always_ff @(posedge clock) begin
        if (!reset && enable && !clear_acc) begin
            // Check if we're approaching overflow
            if (accumulator > 32'h7FFF_0000) begin
                $warning("MAC accumulator approaching positive overflow: %0d", accumulator);
            end
        end
    end
    
    // synthesis translate_on

endmodule
