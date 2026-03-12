//==============================================================================
// Module: uart_tx
// Standard UART transmitter, 8N1.
//==============================================================================

module uart_tx #(
    parameter CLK_HZ = 100_000_000,
    parameter BAUD   = 115_200
)(
    input  logic       clk,
    input  logic       rst,
    input  logic [7:0] din,    // byte to send
    input  logic       start,  // pulse to begin transmission
    output logic       tx,     // serial output
    output logic       busy    // high while transmitting
);

localparam integer CYCLES_PER_BIT = CLK_HZ / BAUD;

typedef enum logic [1:0] { S_IDLE, S_START, S_DATA, S_STOP } state_t;
state_t     state;
logic [9:0] baud_cnt;
logic [2:0] bit_idx;
logic [7:0] shift_reg;

always_ff @(posedge clk) begin
    if (rst) begin
        state    <= S_IDLE;
        tx       <= 1;       // idle high
        busy     <= 0;
        baud_cnt <= 0;
        bit_idx  <= 0;
        shift_reg <= 0;
    end else begin
        case (state)

        S_IDLE: begin
            tx   <= 1;
            busy <= 0;
            if (start) begin
                shift_reg <= din;
                baud_cnt  <= 0;
                busy      <= 1;
                tx        <= 0;   // start bit (low)
                state     <= S_START;
            end
        end

        S_START: begin
            if (baud_cnt < CYCLES_PER_BIT - 1) baud_cnt <= baud_cnt + 1;
            else begin
                baud_cnt <= 0;
                tx       <= shift_reg[0];
                bit_idx  <= 0;
                state    <= S_DATA;
            end
        end

        S_DATA: begin
            if (baud_cnt < CYCLES_PER_BIT - 1) baud_cnt <= baud_cnt + 1;
            else begin
                baud_cnt <= 0;
                if (bit_idx < 7) begin
                    bit_idx   <= bit_idx + 1;
                    tx        <= shift_reg[bit_idx + 1];
                end else begin
                    tx    <= 1;   // stop bit (high)
                    state <= S_STOP;
                end
            end
        end

        S_STOP: begin
            if (baud_cnt < CYCLES_PER_BIT - 1) baud_cnt <= baud_cnt + 1;
            else begin
                baud_cnt <= 0;
                busy     <= 0;
                state    <= S_IDLE;
            end
        end

        default: state <= S_IDLE;
        endcase
    end
end

endmodule
