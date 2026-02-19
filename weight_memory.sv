//==============================================================================
// Module: weight_memory
//==============================================================================

module weight_memory #(
    parameter int WEIGHT_DEPTH = 65536,
    parameter int ADDR_WIDTH = 16
)(
    input  logic                    clock,
    input  logic                    reset,
    input  logic [ADDR_WIDTH-1:0]  read_addr,
    input  logic                    read_en,
    output logic [7:0]              read_data,
    output logic                    read_valid
);

    logic [7:0] weight_mem [0:WEIGHT_DEPTH-1];
    
    initial begin
        for (int i = 0; i < WEIGHT_DEPTH; i++) begin
            weight_mem[i] = (i % 256);
        end
    end
    
    always_ff @(posedge clock) begin
        if (reset) begin
            read_data  <= 8'h00;
            read_valid <= 1'b0;
        end else begin
            if (read_en) begin
                read_data  <= weight_mem[read_addr];
                read_valid <= 1'b1;
            end else begin
                read_valid <= 1'b0;
            end
        end
    end

endmodule