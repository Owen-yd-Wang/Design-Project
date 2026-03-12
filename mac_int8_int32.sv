//==============================================================================
// Module: mac_int8_int32
// Description: Multiply-Accumulate unit for signed int8 inputs with int32 accumulator.
//              Replaces mac_uint8_int32 for correct CNN inference with negative weights.
//
// Changes from mac_uint8_int32:
//   - data_in and weight_in are now SIGNED int8 (range -128 to 127)
//   - product is sign-extended to int32 before accumulation
//   - accumulator handles negative values correctly
//
// Author: Cognichip Co-Design Team
//==============================================================================

module mac_int8_int32 (
    input  logic        clock,
    input  logic        reset,         // Active high synchronous reset

    // Data inputs (signed int8)
    input  logic signed [7:0]  data_in,    // int8 activation
    input  logic signed [7:0]  weight_in,  // int8 weight (can be negative)

    // Control signals
    input  logic        enable,        // Enable MAC operation
    input  logic        clear_acc,     // Synchronously clear accumulator

    // Outputs
    output logic signed [31:0] acc_out,   // int32 accumulated result
    output logic               valid      // High the cycle after enable or clear
);

    //--------------------------------------------------------------------------
    // Internal signals
    //--------------------------------------------------------------------------
    logic signed [15:0] product;           // int8 × int8 = int16 (fits in 16 bits)
    logic signed [31:0] accumulator;
    logic signed [31:0] next_accumulator;

    //--------------------------------------------------------------------------
    // Signed multiply (combinational)
    // int8 × int8 → int16: max magnitude = 127×127 = 16129, min = -128×127 = -16256
    //--------------------------------------------------------------------------
    always_comb begin
        product = data_in * weight_in;
    end

    //--------------------------------------------------------------------------
    // Accumulator next-state logic
    //--------------------------------------------------------------------------
    always_comb begin
        if (clear_acc)
            next_accumulator = 32'sd0;
        else if (enable)
            // Sign-extend int16 product to int32 before adding
            next_accumulator = accumulator + {{16{product[15]}}, product};
        else
            next_accumulator = accumulator;
    end

    //--------------------------------------------------------------------------
    // Sequential register
    //--------------------------------------------------------------------------
    always_ff @(posedge clock) begin
        if (reset) begin
            accumulator <= 32'sd0;
            valid       <= 1'b0;
        end else begin
            accumulator <= next_accumulator;
            valid       <= enable | clear_acc;
        end
    end

    //--------------------------------------------------------------------------
    // Output
    //--------------------------------------------------------------------------
    assign acc_out = accumulator;

    //--------------------------------------------------------------------------
    // Simulation-only overflow check
    //--------------------------------------------------------------------------
    // synthesis translate_off
    always_ff @(posedge clock) begin
        if (!reset && enable && !clear_acc) begin
            if (accumulator > 32'sh7FFF_0000)
                $warning("mac_int8_int32: accumulator near positive overflow: %0d", accumulator);
            if (accumulator < -32'sh7FFF_0000)
                $warning("mac_int8_int32: accumulator near negative overflow: %0d", accumulator);
        end
    end
    // synthesis translate_on

endmodule
