//==============================================================================
// Module: pointwise_conv1x1_engine
// Description: Parameterized Pointwise 1×1 Convolution Engine for MobileNetV2
//              Performs channel mixing using parallel MAC units
//
// Operation:
//   - Computes 1×1 convolution (dot product across input channels)
//   - For each output channel: out[c] = sum(in[i] * weight[c][i]) for all i
//   - Processes multiple output channels in parallel (limited by NUM_MACS)
//
// Parameterization:
//   - NUM_MACS: Number of parallel MAC units (trade-off: speed vs resources)
//   - MAX_INPUT_CHANNELS: Maximum number of input channels supported
//   - MAX_OUTPUT_CHANNELS: Maximum number of output channels supported
//
// Usage in MobileNetV2:
//   - Expansion: 1×1 conv to increase channels (e.g., 32→192)
//   - Projection: 1×1 conv to reduce channels (e.g., 192→32)
//   - Processes one spatial position at a time
//
// Author: Cognichip Co-Design Team
//==============================================================================

module pointwise_conv1x1_engine #(
    parameter int NUM_MACS = 16,              // Number of parallel MAC units
    parameter int MAX_INPUT_CHANNELS = 320,   // Max input channels (MobileNetV2: up to 320)
    parameter int MAX_OUTPUT_CHANNELS = 320   // Max output channels
)(
    input  logic        clock,
    input  logic        reset,         // Active high reset
    
    // Configuration (set before start_conv)
    input  logic [9:0]  num_input_channels,   // Actual number of input channels (1-320)
    input  logic [9:0]  num_output_channels,  // Actual number of output channels (1-320)
    
    // Input activations (one spatial position, all channels)
    input  logic [7:0]  activations [NUM_MACS-1:0],  // Streamed input channels
    
    // Weights (one output channel's weights at a time)
    input  logic [7:0]  weights [NUM_MACS-1:0],      // Weight vector for current output
    
    // Control signals
    input  logic        start_conv,    // Start convolution computation
    input  logic        clear,         // Clear all accumulators
    input  logic        load_data,     // Load new activation/weight data
    
    // Outputs
    output logic [31:0] conv_result,   // Current output channel result
    output logic        result_valid,  // Result is valid
    output logic        busy           // Engine is computing
);

    //==========================================================================
    // Internal Signals
    //==========================================================================
    
    // MAC unit connections
    logic [31:0] mac_outputs [NUM_MACS-1:0];
    logic        mac_valids  [NUM_MACS-1:0];
    logic        mac_enable;
    logic        mac_clear;
    
    // Summation tree for parallel MAC outputs
    logic [31:0] partial_sum;          // Sum of all MAC outputs
    logic [31:0] accumulated_result;   // Accumulated over multiple cycles
    
    // Control FSM
    typedef enum logic [2:0] {
        IDLE,
        COMPUTE,
        ACCUMULATE,
        DONE
    } state_t;
    
    state_t current_state, next_state;
    
    // Counters
    logic [9:0]  mac_count;            // Number of MACs processed
    logic [9:0]  output_ch_count;      // Current output channel being processed
    logic        compute_done;
    
    //==========================================================================
    // Parallel MAC Array
    //==========================================================================
    
    generate
        for (genvar i = 0; i < NUM_MACS; i++) begin : mac_array
            mac_uint8_int32 mac_inst (
                .clock      (clock),
                .reset      (reset),
                .data_in    (activations[i]),
                .weight_in  (weights[i]),
                .enable     (mac_enable),
                .clear_acc  (mac_clear),
                .acc_out    (mac_outputs[i]),
                .valid      (mac_valids[i])
            );
        end
    endgenerate
    
    //==========================================================================
    // Summation Tree (Parallel Reduction)
    //==========================================================================
    // Sums all NUM_MACS outputs into a single partial_sum
    
    always_comb begin
        partial_sum = 32'b0;
        for (int i = 0; i < NUM_MACS; i++) begin
            partial_sum = partial_sum + mac_outputs[i];
        end
    end
    
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
                if (start_conv) begin
                    next_state = COMPUTE;
                end
            end
            
            COMPUTE: begin
                if (compute_done) begin
                    next_state = DONE;
                end
            end
            
            DONE: begin
                next_state = IDLE;
            end
            
            default: next_state = IDLE;
        endcase
    end
    
    //==========================================================================
    // Datapath Control
    //==========================================================================
    
    // MAC control signals
    always_comb begin
        mac_enable = (current_state == COMPUTE) && load_data;
        mac_clear  = clear || (current_state == IDLE && start_conv);
    end
    
    // Compute completion detection
    always_comb begin
        // Done when we've processed all input channels
        compute_done = (mac_count >= num_input_channels);
    end
    
    // MAC counter (tracks how many input channels processed)
    always_ff @(posedge clock) begin
        if (reset || clear) begin
            mac_count <= 10'b0;
        end else if (current_state == COMPUTE && load_data) begin
            if (mac_count + NUM_MACS >= num_input_channels) begin
                mac_count <= num_input_channels;
            end else begin
                mac_count <= mac_count + NUM_MACS;
            end
        end
    end
    
    //==========================================================================
    // Result Accumulation
    //==========================================================================
    // For cases where num_input_channels > NUM_MACS, we need to accumulate
    // partial sums across multiple cycles
    
    always_ff @(posedge clock) begin
        if (reset || clear || (current_state == IDLE && start_conv)) begin
            accumulated_result <= 32'b0;
        end else if (current_state == COMPUTE) begin
            accumulated_result <= accumulated_result + partial_sum;
        end
    end
    
    //==========================================================================
    // Output Logic
    //==========================================================================
    
    always_ff @(posedge clock) begin
        if (reset) begin
            conv_result  <= 32'b0;
            result_valid <= 1'b0;
            busy         <= 1'b0;
        end else begin
            // Update result when computation completes
            if (current_state == DONE) begin
                conv_result  <= accumulated_result;
                result_valid <= 1'b1;
            end else begin
                result_valid <= 1'b0;
            end
            
            // Busy signal
            busy <= (current_state == COMPUTE);
        end
    end

    // always @(posedge clock) if (load_data || result_valid) begin 
    //     $display("t=%0t load=%0b busy=%0b valid=%0b start=%0b state=%0b reset=%0b mac_count=%0d in_ch=%0d partial_sum=%0d acc=%0d", 
    //              $time, load_data, busy, result_valid, start_conv, current_state, reset, mac_count, num_input_channels, partial_sum, accumulated_result);
    // end
    //==========================================================================
    // Debug Assertions (Synthesis will ignore these)
    //==========================================================================
    
    // synthesis translate_off
    
    // Monitor configuration
    // always_ff @(posedge clock) begin
    //     if (!reset && start_conv) begin
    //         $display("\n=== Pointwise Conv 1×1 Computation ===");
    //         $display("Input Channels:  %0d", num_input_channels);
    //         $display("Output Channels: %0d", num_output_channels);
    //         $display("Parallel MACs:   %0d", NUM_MACS);
    //         if (num_input_channels > NUM_MACS) begin
    //             $display("Note: Will require %0d cycles per output channel", 
    //                      (num_input_channels + NUM_MACS - 1) / NUM_MACS);
    //         end
    //     end
        
    //     if (!reset && result_valid) begin
    //         $display("Output Channel Result: %0d", $signed(conv_result));
    //         $display("======================================\n");
    //     end
    // end
    
    // // Check for overflow
    // always_ff @(posedge clock) begin
    //     if (!reset && result_valid) begin
    //         if (conv_result > 32'h7FFF_FFFF) begin
    //             $display("INFO: Large pointwise result detected: %0d", $signed(conv_result));
    //         end
    //     end
    // end
    
    // synthesis translate_on

endmodule
