//==============================================================================
// Module: uart_rx
// Standard UART receiver, 8N1 (8 data bits, no parity, 1 stop bit).
//
// Parameters:
//   CLK_HZ   : system clock frequency (default 100 MHz for Basys3)
//   BAUD     : baud rate (default 115200)
//
// Interface:
//   rx       : serial input (connect to USB-UART RXD pin on Basys3)
//   data_out : received byte
//   valid    : pulses high for 1 cycle when data_out is ready
//==============================================================================

module uart_rx #(
    parameter CLK_HZ = 100_000_000,
    parameter BAUD   = 115_200
)(
    input  logic       clk,
    input  logic       rst,
    input  logic       rx,        // serial in (from USB-UART bridge)
    output logic [7:0] data_out,  // received byte
    output logic       valid      // 1-cycle pulse per received byte
);

// Number of clock cycles per bit
localparam integer CYCLES_PER_BIT = CLK_HZ / BAUD;   // = 868 at 100 MHz / 115200

// =============================================================================
// Double-flop rx to synchronise and debounce
// =============================================================================
logic rx_sync0, rx_sync1;
always_ff @(posedge clk) begin
    rx_sync0 <= rx;
    rx_sync1 <= rx_sync0;
end

// =============================================================================
// State machine
// =============================================================================
typedef enum logic [1:0] {
    S_IDLE,       // waiting for start bit (rx goes low)
    S_START,      // confirm start bit at mid-bit
    S_DATA,       // receive 8 data bits
    S_STOP        // verify stop bit
} state_t;

state_t             state;
logic [9:0]         baud_cnt;   // counts up to CYCLES_PER_BIT
logic [2:0]         bit_idx;    // which data bit we're receiving (0..7)
logic [7:0]         shift_reg;  // shift register for incoming bits

// Half-bit period for sampling in the middle of each bit
localparam integer HALF_BIT = CYCLES_PER_BIT / 2;

always_ff @(posedge clk) begin
    if (rst) begin
        state     <= S_IDLE;
        baud_cnt  <= 0;
        bit_idx   <= 0;
        shift_reg <= 0;
        data_out  <= 0;
        valid     <= 0;
    end else begin
        valid <= 0;   // default: not valid

        case (state)

        // ---------------------------------------------------------------
        // Wait for falling edge (start bit)
        // ---------------------------------------------------------------
        S_IDLE: begin
            if (!rx_sync1) begin          // rx went low → start bit detected
                baud_cnt <= 0;
                state    <= S_START;
            end
        end

        // ---------------------------------------------------------------
        // Wait half a bit period, then sample to confirm start bit
        // ---------------------------------------------------------------
        S_START: begin
            if (baud_cnt < HALF_BIT - 1) begin
                baud_cnt <= baud_cnt + 1;
            end else begin
                baud_cnt <= 0;
                if (!rx_sync1) begin      // still low → valid start bit
                    bit_idx <= 0;
                    state   <= S_DATA;
                end else begin            // glitch, not a real start bit
                    state   <= S_IDLE;
                end
            end
        end

        // ---------------------------------------------------------------
        // Sample 8 data bits, LSB first, at the middle of each bit period
        // ---------------------------------------------------------------
        S_DATA: begin
            if (baud_cnt < CYCLES_PER_BIT - 1) begin
                baud_cnt <= baud_cnt + 1;
            end else begin
                baud_cnt           <= 0;
                shift_reg[bit_idx] <= rx_sync1;   // sample data bit

                if (bit_idx < 7) begin
                    bit_idx <= bit_idx + 1;
                end else begin
                    bit_idx <= 0;
                    state   <= S_STOP;
                end
            end
        end

        // ---------------------------------------------------------------
        // Stop bit: wait one bit period, output byte
        // ---------------------------------------------------------------
        S_STOP: begin
            if (baud_cnt < CYCLES_PER_BIT - 1) begin
                baud_cnt <= baud_cnt + 1;
            end else begin
                baud_cnt <= 0;
                if (rx_sync1) begin       // valid stop bit (high)
                    data_out <= shift_reg;
                    valid    <= 1;
                end
                // Whether stop bit is valid or not, go back to IDLE
                state <= S_IDLE;
            end
        end

        default: state <= S_IDLE;
        endcase
    end
end

endmodule
