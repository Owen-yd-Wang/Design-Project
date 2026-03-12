//==============================================================================
// Module: top_basys3
// Top-level wrapper for TinyCNN on Basys3 (XC7A35T).
//
// Workflow:
//   1. PC sends 3072 bytes (CIFAR-10 image, int8) over UART at 115200 baud.
//   2. FPGA receives all bytes, then automatically starts CNN inference.
//   3. When inference is done:
//      - 7-segment display shows predicted class (0-9)
//      - LEDs[3:0] also show the class in binary
//      - UART sends back "Class: N\r\n" string
//
// Basys3 I/O used:
//   CLK100MHZ  : 100 MHz onboard clock
//   CPU_RESETN : active-low reset button (BTNC on some boards)
//   RsRx       : USB-UART receive  (Basys3 built-in FTDI chip)
//   RsTx       : USB-UART transmit (for sending result back)
//   LED[3:0]   : shows class in binary
//   SEG[6:0]   : 7-segment cathodes
//   AN[3:0]    : 7-segment anodes (we use rightmost digit only)
//==============================================================================

module top_basys3 (
    input  logic        CLK100MHZ,
    input  logic        CPU_RESETN,  // active-low

    // UART
    input  logic        RsRx,
    output logic        RsTx,

    // Display
    output logic [15:0] LED,
    output logic [6:0]  SEG,         // a-g segments (active low on Basys3)
    output logic [3:0]  AN           // anode select (active low on Basys3)
);

// =============================================================================
// Clock and reset
// =============================================================================
logic clk, rst;
assign clk = CLK100MHZ;
assign rst = CPU_RESETN;    // BTNC (U18) is active-high: press=reset, release=run

// =============================================================================
// UART receiver
// =============================================================================
logic [7:0] uart_byte;
logic       uart_valid;

uart_rx #(.CLK_HZ(100_000_000), .BAUD(115_200)) u_rx (
    .clk      (clk),
    .rst      (rst),
    .rx       (RsRx),
    .data_out (uart_byte),
    .valid    (uart_valid)
);

// =============================================================================
// Image buffer controller
// Receives 3072 bytes from UART, then triggers CNN inference.
// =============================================================================
logic [11:0] rx_count;     // how many bytes received so far (0..3071)
logic        img_we;
logic [11:0] img_addr;
logic signed [7:0] img_din;
logic        cnn_start;
logic        image_ready;

always_ff @(posedge clk) begin
    if (rst) begin
        rx_count    <= 0;
        img_we      <= 0;
        img_addr    <= 0;
        img_din     <= 0;
        cnn_start   <= 0;
        image_ready <= 0;
    end else begin
        img_we    <= 0;
        cnn_start <= 0;

        if (uart_valid && !image_ready) begin
            img_we   <= 1;
            img_addr <= rx_count;
            img_din  <= $signed(uart_byte);   // treat byte as signed int8

            if (rx_count == 3071) begin
                rx_count    <= 0;
                image_ready <= 1;
                cnn_start   <= 1;             // pulse start to CNN engine
            end else begin
                rx_count <= rx_count + 1;
            end
        end

        // After CNN is done (done pulse), allow new image
        if (cnn_done) begin
            image_ready <= 0;
        end
    end
end

// =============================================================================
// CNN inference engine
// =============================================================================
logic       cnn_busy;
logic       cnn_done;
logic [3:0] cnn_result;

tiny_cnn_engine u_cnn (
    .clk      (clk),
    .rst      (rst),
    .img_we   (img_we),
    .img_addr (img_addr),
    .img_din  (img_din),
    .start    (cnn_start),
    .busy     (cnn_busy),
    .done     (cnn_done),
    .result   (cnn_result)
);

// =============================================================================
// Result register (latch result when done)
// =============================================================================
logic [3:0] result_reg;
always_ff @(posedge clk) begin
    if (rst)        result_reg <= 0;
    else if (cnn_done) result_reg <= cnn_result;
end

// =============================================================================
// LED output
//   LED[3:0]  : class in binary
//   LED[15]   : on when CNN is busy (running)
//   LED[14]   : on when waiting for image bytes
// =============================================================================
assign LED[3:0]  = result_reg;
assign LED[15]   = cnn_busy;
assign LED[14]   = !image_ready && !cnn_busy;  // waiting for image
assign LED[13:4] = 0;

// =============================================================================
// 7-segment display (rightmost digit shows class 0-9)
// Basys3: SEG is active-low cathodes, AN is active-low anode select
// =============================================================================
// 7-segment encoding for digits 0-9 (segments: gfedcba, active-low)
function automatic logic [6:0] digit_to_seg(input logic [3:0] d);
    case (d)
        4'd0: return 7'b100_0000;   // 0
        4'd1: return 7'b111_1001;   // 1
        4'd2: return 7'b010_0100;   // 2
        4'd3: return 7'b011_0000;   // 3
        4'd4: return 7'b001_1001;   // 4
        4'd5: return 7'b001_0010;   // 5
        4'd6: return 7'b000_0010;   // 6
        4'd7: return 7'b111_1000;   // 7
        4'd8: return 7'b000_0000;   // 8
        4'd9: return 7'b001_0000;   // 9
        default: return 7'b111_1111; // blank
    endcase
endfunction

assign SEG = digit_to_seg(result_reg);  // rightmost digit
assign AN  = 4'b1110;                   // enable only rightmost digit

// =============================================================================
// UART transmitter — sends "Class: N\r\n" when result is ready
// =============================================================================
// Simple byte-by-byte TX state machine
// Message: "Class: 0\r\n" = 10 bytes
// "C","l","a","s","s",":"," ","0","\r","\n"

localparam integer TX_CYCLES = 100_000_000 / 115_200;  // 868

logic [7:0]  tx_byte;
logic        tx_start;
logic        tx_busy;

uart_tx #(.CLK_HZ(100_000_000), .BAUD(115_200)) u_tx (
    .clk   (clk),
    .rst   (rst),
    .din   (tx_byte),
    .start (tx_start),
    .tx    (RsTx),
    .busy  (tx_busy)
);

// TX sequencer
logic [3:0] tx_idx;
logic       tx_active;

// Message: "Class: X\r\n"
// Indices: 0='C', 1='l', 2='a', 3='s', 4='s', 5=':', 6=' ', 7=digit, 8='\r', 9='\n'
function automatic logic [7:0] tx_char(input logic [3:0] i, input logic [3:0] cls);
    case (i)
        4'd0: return 8'h43;  // 'C'
        4'd1: return 8'h6C;  // 'l'
        4'd2: return 8'h61;  // 'a'
        4'd3: return 8'h73;  // 's'
        4'd4: return 8'h73;  // 's'
        4'd5: return 8'h3A;  // ':'
        4'd6: return 8'h20;  // ' '
        4'd7: return 8'h30 + {4'b0, cls};  // '0' + class
        4'd8: return 8'h0D;  // '\r'
        4'd9: return 8'h0A;  // '\n'
        default: return 8'h00;
    endcase
endfunction

always_ff @(posedge clk) begin
    if (rst) begin
        tx_idx    <= 0;
        tx_active <= 0;
        tx_start  <= 0;
        tx_byte   <= 0;
    end else begin
        tx_start <= 0;

        if (cnn_done && !tx_active) begin
            // Start transmitting result
            tx_active <= 1;
            tx_idx    <= 0;
        end

        if (tx_active && !tx_busy && !tx_start) begin
            if (tx_idx < 10) begin
                tx_byte  <= tx_char(tx_idx, cnn_result);
                tx_start <= 1;
                tx_idx   <= tx_idx + 1;
            end else begin
                tx_active <= 0;
                tx_idx    <= 0;
            end
        end
    end
end

endmodule
