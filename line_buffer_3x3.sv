//==============================================================================
// Module: line_buffer_3x3
// Description: Line Buffer for 3×3 Window Extraction from Streaming Image
//              Stores 3 rows of image data and outputs sliding 3×3 windows
//
// Operation:
//   - Receives streaming pixel input (one pixel per cycle)
//   - Maintains 3 line buffers (current row + 2 previous rows)
//   - Outputs 3×3 windows for convolution processing
//   - Generates valid signal when window is ready
//
// Memory Architecture:
//   - 3 FIFOs of IMAGE_WIDTH depth each
//   - Total memory: 3 × IMAGE_WIDTH × 8 bits
//   - For 224×224 image: 3 × 224 × 8 = 5,376 bits (0.66 KB)
//
// Window Layout (output):
//   [0][1][2]    Row 0 (oldest)
//   [3][4][5]    Row 1
//   [6][7][8]    Row 2 (newest)
//
// Usage in MobileNetV2:
//   - Feeds depthwise_conv3x3_engine with 3×3 windows
//   - Processes one channel at a time
//   - Repeat for all input channels
//
// Author: Cognichip Co-Design Team
//==============================================================================

module line_buffer_3x3 #(
    parameter int IMAGE_WIDTH = 224,   // Width of input image
    parameter int IMAGE_HEIGHT = 224   // Height of input image
)(
    input  logic       clock,
    input  logic       reset,          // Active high reset
    
    // Streaming pixel input
    input  logic [7:0] pixel_in,       // Input pixel value
    input  logic       pixel_valid,    // Input pixel is valid
    
    // Control
    input  logic       start_frame,    // Start of new frame
    
    // 3×3 Window output
    output logic [7:0] window_out [8:0], // 3×3 window (flattened)
    output logic       window_valid    // Output window is valid
);

    //==========================================================================
    // Internal Signals
    //==========================================================================
    
    // Three line buffers (shift registers)
    logic [7:0] line_buffer_0 [IMAGE_WIDTH-1:0];  // Oldest row
    logic [7:0] line_buffer_1 [IMAGE_WIDTH-1:0];  // Middle row
    logic [7:0] line_buffer_2 [IMAGE_WIDTH-1:0];  // Newest row (current)
    
    // Current position counters
    logic [15:0] col_counter;      // Column position (0 to IMAGE_WIDTH-1)
    logic [15:0] row_counter;      // Row position (0 to IMAGE_HEIGHT-1)
    
    // 3×3 window extraction registers
    logic [7:0] window_row0 [2:0]; // Top row of window
    logic [7:0] window_row1 [2:0]; // Middle row of window
    logic [7:0] window_row2 [2:0]; // Bottom row of window
    
    // Valid signal generation
    logic       window_valid_internal;
    
    //==========================================================================
    // Position Counter Logic
    //==========================================================================
    
    always_ff @(posedge clock) begin
        if (reset || start_frame) begin
            col_counter <= 16'b0;
            row_counter <= 16'b0;
        end else if (pixel_valid) begin
            if (col_counter == IMAGE_WIDTH - 1) begin
                // End of row, move to next row
                col_counter <= 16'b0;
                if (row_counter < IMAGE_HEIGHT - 1) begin
                    row_counter <= row_counter + 1;
                end else begin
                    row_counter <= 16'b0;  // Wrap to start of frame
                end
            end else begin
                col_counter <= col_counter + 1;
            end
        end
    end
    
    //==========================================================================
    // Line Buffer Shift Logic
    //==========================================================================
    
    always_ff @(posedge clock) begin
        if (reset || start_frame) begin
            // Clear all line buffers
            for (int i = 0; i < IMAGE_WIDTH; i++) begin
                line_buffer_0[i] <= 8'b0;
                line_buffer_1[i] <= 8'b0;
                line_buffer_2[i] <= 8'b0;
            end
        end else if (pixel_valid) begin
            // When we reach end of row, shift lines
            if (col_counter == IMAGE_WIDTH - 1) begin
                // Shift line buffers (row complete)
                for (int i = 0; i < IMAGE_WIDTH; i++) begin
                    line_buffer_0[i] <= line_buffer_1[i];  // Oldest ← Middle
                    line_buffer_1[i] <= line_buffer_2[i];  // Middle ← Newest
                end
            end
            
            // Always write new pixel to current line buffer
            line_buffer_2[col_counter] <= pixel_in;
        end
    end
    
    //==========================================================================
    // 3×3 Window Extraction Logic
    //==========================================================================
    
    always_comb begin
        // Extract 3 pixels from each line buffer
        // Window needs current column and 2 previous columns
        
        if (col_counter >= 2) begin
            // Enough history to form a window
            // Extract 3×3 window centered at current position
            
            // Row 0 (oldest line)
            window_row0[0] = line_buffer_0[col_counter - 2];
            window_row0[1] = line_buffer_0[col_counter - 1];
            window_row0[2] = line_buffer_0[col_counter];
            
            // Row 1 (middle line)
            window_row1[0] = line_buffer_1[col_counter - 2];
            window_row1[1] = line_buffer_1[col_counter - 1];
            window_row1[2] = line_buffer_1[col_counter];
            
            // Row 2 (newest line)
            window_row2[0] = line_buffer_2[col_counter - 2];
            window_row2[1] = line_buffer_2[col_counter - 1];
            window_row2[2] = line_buffer_2[col_counter];
        end else begin
            // Not enough history, output zeros (padding)
            for (int i = 0; i < 3; i++) begin
                window_row0[i] = 8'b0;
                window_row1[i] = 8'b0;
                window_row2[i] = 8'b0;
            end
        end
    end
    
    //==========================================================================
    // Window Valid Generation
    //==========================================================================
    
    always_comb begin
        // Window is valid when:
        // 1. We have at least 3 rows of data (row_counter >= 2)
        // 2. We have at least 3 columns of data (col_counter >= 2)
        // 3. Input pixel is valid
        window_valid_internal = pixel_valid && 
                               (row_counter >= 2) && 
                               (col_counter >= 2);
    end
    
    //==========================================================================
    // Output Assignment
    //==========================================================================
    
    always_ff @(posedge clock) begin
        if (reset) begin
            for (int i = 0; i < 9; i++) begin
                window_out[i] <= 8'b0;
            end
            window_valid <= 1'b0;
        end else begin
            // Flatten 3×3 window to 9-element array
            window_out[0] <= window_row0[0];
            window_out[1] <= window_row0[1];
            window_out[2] <= window_row0[2];
            window_out[3] <= window_row1[0];
            window_out[4] <= window_row1[1];
            window_out[5] <= window_row1[2];
            window_out[6] <= window_row2[0];
            window_out[7] <= window_row2[1];
            window_out[8] <= window_row2[2];
            
            window_valid <= window_valid_internal;
        end
    end
    
    //==========================================================================
    // Debug Assertions
    //==========================================================================
    
    // synthesis translate_off
    
    // Monitor frame processing
    always_ff @(posedge clock) begin
        if (!reset && start_frame) begin
            $display("[LINE_BUFFER] Start of new frame");
        end
        
        if (!reset && pixel_valid && col_counter == IMAGE_WIDTH - 1) begin
            $display("[LINE_BUFFER] End of row %0d", row_counter);
        end
        
        if (!reset && window_valid) begin
            $display("[LINE_BUFFER] Valid window at (row=%0d, col=%0d)", row_counter, col_counter);
        end
    end
    
    // synthesis translate_on

endmodule
