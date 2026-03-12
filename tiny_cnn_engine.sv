//==============================================================================
// Module: tiny_cnn_engine
// TinyCNN inference engine for Basys3 (XC7A35T).
//
// Architecture:
//   Conv1(3->16, 3x3, pad=1) -> MaxPool2 ->
//   Conv2(16->32,3x3, pad=1) -> MaxPool2 ->
//   Conv3(32->32,3x3, pad=1) -> MaxPool2 ->
//   FC1(512->64) -> FC2(64->10) -> argmax -> result (0-9)
//
// Quantisation follows ONNX QLinearConv:
//   acc = bias + sum((x_q - x_zp) * (w_q - w_zp))    [w_zp=0 for all layers]
//   y_q = clamp(round(acc * M) + y_zp, -128, 127)
//   where M = x_scale * w_scale / y_scale
//
// Image input: write 3072 bytes via img_we/img_addr/img_din before asserting start.
// Result valid when done pulses high.
//==============================================================================

module tiny_cnn_engine (
    input  logic        clk,
    input  logic        rst,           // synchronous active-high reset

    // Image load (write 3×32×32 = 3072 int8 bytes before start)
    input  logic        img_we,
    input  logic [11:0] img_addr,      // 0..3071
    input  logic signed [7:0] img_din,

    // Control
    input  logic        start,
    output logic        busy,
    output logic        done,          // 1-cycle pulse when result ready
    output logic [3:0]  result         // predicted class 0-9
);

// =============================================================================
// Requantisation constants   M = round(x_scale * w_scale / y_scale * 2^20)
// =============================================================================
// From quant_params.txt:
//   conv1: M=0.0023394289 → 0.0023394289*1048576 = 2453.8 → 2454
//   conv2: M=0.0042724727 → 4479.5 → 4480
//   conv3: M=0.0025166616 → 2638.5 → 2639
//   fc1:   M=0.0019082877 → 2001.5 → 2002
//   fc2:   M=0.0020523081 → 2152.7 → 2153
localparam [20:0] M_CONV1 = 21'd2454;
localparam [20:0] M_CONV2 = 21'd4480;
localparam [20:0] M_CONV3 = 21'd2639;
localparam [20:0] M_FC1   = 21'd2002;
localparam [20:0] M_FC2   = 21'd2153;

// Input zero-points  (x_zp)
localparam signed [7:0] XZP_CONV1 = -8'sd5;
localparam signed [7:0] XZP_CONV2 = -8'sd128;
localparam signed [7:0] XZP_CONV3 = -8'sd128;
localparam signed [7:0] XZP_FC1   = -8'sd128;
localparam signed [7:0] XZP_FC2   = -8'sd128;

// Output zero-points (y_zp)
localparam signed [7:0] YZP_CONV1 = -8'sd128;
localparam signed [7:0] YZP_CONV2 = -8'sd128;
localparam signed [7:0] YZP_CONV3 = -8'sd128;
localparam signed [7:0] YZP_FC1   = -8'sd128;
localparam signed [7:0] YZP_FC2   =  8'sd15;

// =============================================================================
// Weight memories  (synthesised as BRAM via $readmemh)
// =============================================================================
(* rom_style = "block" *) logic signed [7:0]  conv1_w [0:431];      // 16×3×3×3
logic signed [31:0] conv1_b [0:15];       // 16
(* rom_style = "block" *) logic signed [7:0]  conv2_w [0:4607];     // 32×16×3×3
logic signed [31:0] conv2_b [0:31];       // 32
(* rom_style = "block" *) logic signed [7:0]  conv3_w [0:9215];     // 32×32×3×3
logic signed [31:0] conv3_b [0:31];       // 32
(* rom_style = "block" *) logic signed [7:0]  fc1_w   [0:32767];    // 64×512
logic signed [31:0] fc1_b   [0:63];       // 64
logic signed [7:0]  fc2_w   [0:639];      // 10×64
logic signed [31:0] fc2_b   [0:9];        // 10

initial begin
    $readmemh("C:/Users/liche/Design-Project/tiny-cnn-basys3/full_inference_data/conv1_weights.mem", conv1_w);
    $readmemh("C:/Users/liche/Design-Project/tiny-cnn-basys3/full_inference_data/conv1_bias.mem",    conv1_b);
    $readmemh("C:/Users/liche/Design-Project/tiny-cnn-basys3/full_inference_data/conv2_weights.mem", conv2_w);
    $readmemh("C:/Users/liche/Design-Project/tiny-cnn-basys3/full_inference_data/conv2_bias.mem",    conv2_b);
    $readmemh("C:/Users/liche/Design-Project/tiny-cnn-basys3/full_inference_data/conv3_weights.mem", conv3_w);
    $readmemh("C:/Users/liche/Design-Project/tiny-cnn-basys3/full_inference_data/conv3_bias.mem",    conv3_b);
    $readmemh("C:/Users/liche/Design-Project/tiny-cnn-basys3/full_inference_data/fc1_weights.mem",   fc1_w);
    $readmemh("C:/Users/liche/Design-Project/tiny-cnn-basys3/full_inference_data/fc1_bias.mem",      fc1_b);
    $readmemh("C:/Users/liche/Design-Project/tiny-cnn-basys3/full_inference_data/fc2_weights.mem",   fc2_w);
    $readmemh("C:/Users/liche/Design-Project/tiny-cnn-basys3/full_inference_data/fc2_bias.mem",      fc2_b);
end

// =============================================================================
// Activation buffers — force BRAM inference to avoid LUT-RAM overflow
// =============================================================================
(* ram_style = "block" *) logic signed [7:0] img_buf  [0:3071];    // 3×32×32   input image
(* ram_style = "block" *) logic signed [7:0] pre_pool [0:16383];   // 16×32×32  conv1 pre-pool (16 KB)
(* ram_style = "block" *) logic signed [7:0] act_a    [0:4095];    // 16×16×16  after pool1
(* ram_style = "block" *) logic signed [7:0] act_b    [0:2047];    // 32×8×8    after pool2
(* ram_style = "block" *) logic signed [7:0] act_c    [0:511];     // 32×4×4    after pool3
logic signed [7:0] act_d    [0:63];      // 64        after FC1 (small, use registers)
logic signed [31:0] logits  [0:9];       // 10×int32  FC2 raw output (no requant needed for argmax)

// Image write port
always_ff @(posedge clk)
    if (img_we) img_buf[img_addr] <= img_din;

// =============================================================================
// BRAM read ports — one always_ff per array forces registered-output inference.
// Each address register is driven by the FSM; data is valid one cycle later.
// =============================================================================
logic [14:0] img_buf_raddr;   logic signed [7:0] img_buf_rdata;
logic [13:0] pre_pool_raddr;  logic signed [7:0] pre_pool_rdata;
logic [11:0] act_a_raddr;     logic signed [7:0] act_a_rdata;
logic [10:0] act_b_raddr;     logic signed [7:0] act_b_rdata;
logic [ 8:0] act_c_raddr;     logic signed [7:0] act_c_rdata;
logic [ 8:0] conv1_w_raddr;   logic signed [7:0] conv1_w_rdata;
logic [12:0] conv2_w_raddr;   logic signed [7:0] conv2_w_rdata;
logic [13:0] conv3_w_raddr;   logic signed [7:0] conv3_w_rdata;
logic [14:0] fc1_w_raddr;     logic signed [7:0] fc1_w_rdata;

always_ff @(posedge clk) img_buf_rdata  <= img_buf [img_buf_raddr];
always_ff @(posedge clk) pre_pool_rdata <= pre_pool[pre_pool_raddr];
always_ff @(posedge clk) act_a_rdata    <= act_a   [act_a_raddr];
always_ff @(posedge clk) act_b_rdata    <= act_b   [act_b_raddr];
always_ff @(posedge clk) act_c_rdata    <= act_c   [act_c_raddr];
always_ff @(posedge clk) conv1_w_rdata  <= conv1_w [conv1_w_raddr];
always_ff @(posedge clk) conv2_w_rdata  <= conv2_w [conv2_w_raddr];
always_ff @(posedge clk) conv3_w_rdata  <= conv3_w [conv3_w_raddr];
always_ff @(posedge clk) fc1_w_rdata    <= fc1_w   [fc1_w_raddr];

// =============================================================================
// State machine
// =============================================================================
typedef enum logic [5:0] {
    S_IDLE,
    // Conv1 + Pool1
    S_CONV1_INIT, S_CONV1_MAC, S_CONV1_STORE, S_CONV1_REQUAN,
    S_POOL1,
    // Conv2 + Pool2
    S_CONV2_INIT, S_CONV2_MAC, S_CONV2_STORE, S_CONV2_REQUAN,
    S_POOL2,
    // Conv3 + Pool3
    S_CONV3_INIT, S_CONV3_MAC, S_CONV3_STORE, S_CONV3_REQUAN,
    S_POOL3,
    // FC1
    S_FC1_INIT,   S_FC1_MAC,   S_FC1_STORE,   S_FC1_REQUAN,
    // FC2
    S_FC2_INIT,   S_FC2_MAC,   S_FC2_STORE,
    // Argmax + Done
    S_ARGMAX, S_DONE
} state_t;

state_t state;

// Loop counters
logic [5:0] cnt_oc;   // output channel  (max 63, FC1)
logic [4:0] cnt_oh;   // output row      (max 31, Conv1)
logic [4:0] cnt_ow;   // output col      (max 31, Conv1)
logic [8:0] cnt_ic;   // input channel   (max 511, FC1 flattened)
logic [1:0] cnt_kh;   // kernel row      (0..2)
logic [1:0] cnt_kw;   // kernel col      (0..2)

// Accumulator — use_dsp forces DSP48E1 inference, avoiding slow LUT multipliers
(* use_dsp = "yes" *) logic signed [31:0] acc;

// Pool helpers
logic [2:0]         pool_sub;          // sub-step 0..4 for 5-step BRAM-pipelined pool
logic signed [7:0]  pv0, pv1, pv2;    // 3 captured pool values; 4th read via pre_pool_rdata

// MAC pipeline helpers
logic [1:0]         mac_phase;         // 0=issue addr, 1=register x_adj/w, 2=accumulate
logic               mac_valid;         // in-bounds flag carried across phases
logic signed [8:0]  x_adj_reg;         // registered x_adj (BRAM_data - x_zp)
logic signed [7:0]  w_reg;             // registered weight data

// Requantisation pipeline register — breaks the critical path in STORE states.
// STORE: register acc*M (one cycle); REQUAN: shift+clamp+write (next cycle).
logic signed [63:0] req_tmp;

// Argmax helpers
logic [3:0]          arg_i;
logic signed [31:0]  arg_max_val;

// =============================================================================
// Clamp helper (function — synthesisable)
// =============================================================================
function automatic logic signed [7:0] clamp8(input logic signed [31:0] v);
    if      (v > 32'sd127)  return  8'sd127;
    else if (v < -32'sd128) return -8'sd128;
    else                    return  v[7:0];
endfunction

// =============================================================================
// Requantise: acc_int32  →  int8
//   y = clamp(round(acc * M_fp20 / 2^20) + y_zp, -128, 127)
//   Rounding: add 2^19 (=524288) before shifting
// =============================================================================
function automatic logic signed [7:0] req8(
    input logic signed [31:0] a,
    input logic [20:0]         M,
    input logic signed [7:0]   yzp
);
    logic signed [63:0] tmp;
    logic signed [31:0] shifted;
    begin
        tmp     = $signed(a) * $signed({1'b0, M});
        shifted = (tmp + 64'sd524288) >>> 20;   // arithmetic right shift
        return clamp8(shifted + $signed({{24{yzp[7]}}, yzp}));
    end
endfunction

// =============================================================================
// Main FSM
// =============================================================================
always_ff @(posedge clk) begin
    if (rst) begin
        state   <= S_IDLE;
        busy    <= 0;  done <= 0;  result <= 0;
        cnt_oc  <= 0;  cnt_oh <= 0;  cnt_ow <= 0;
        cnt_ic  <= 0;  cnt_kh <= 0;  cnt_kw <= 0;
        acc     <= 0;
        pool_sub <= 0;
        mac_phase <= 0;  mac_valid <= 0;
        x_adj_reg <= 0;  w_reg <= 0;
        req_tmp <= 0;
        img_buf_raddr  <= 0;  pre_pool_raddr <= 0;
        act_a_raddr    <= 0;  act_b_raddr    <= 0;  act_c_raddr   <= 0;
        conv1_w_raddr  <= 0;  conv2_w_raddr  <= 0;
        conv3_w_raddr  <= 0;  fc1_w_raddr    <= 0;
        arg_i    <= 0;
        arg_max_val <= 32'sh8000_0000;
    end else begin
        done <= 0;

        case (state)

        // =====================================================================
        S_IDLE: begin
            busy <= 0;
            if (start) begin
                busy   <= 1;
                state  <= S_CONV1_INIT;
                cnt_oc <= 0;  cnt_oh <= 0;  cnt_ow <= 0;
            end
        end

        // =====================================================================
        // CONV1 : 3×32×32 → 16×32×32 , pad=1 , x_zp=-5 , w_zp=0
        //   weight index : oc*27 + ic*9 + kh*3 + kw
        //   input  index : ic*1024 + ih*32 + iw
        // =====================================================================
        S_CONV1_INIT: begin
            acc    <= conv1_b[cnt_oc];    // start accumulator with bias
            cnt_ic <= 0;  cnt_kh <= 0;  cnt_kw <= 0;
            mac_phase <= 0;
            state  <= S_CONV1_MAC;
        end

        S_CONV1_MAC: begin
            begin
                // Padded input coordinates (signed for out-of-bounds detection)
                logic signed [6:0] ih, iw;
                ih = $signed({2'b0, cnt_oh}) + $signed({2'b0, cnt_kh}) - 7'sd1;
                iw = $signed({2'b0, cnt_ow}) + $signed({2'b0, cnt_kw}) - 7'sd1;

                if (mac_phase == 0) begin
                    // Phase 0: register BRAM read addresses; data valid next cycle
                    if (ih >= 0 && ih < 32 && iw >= 0 && iw < 32) begin
                        img_buf_raddr <= cnt_ic * 1024 + ih[4:0] * 32 + iw[4:0];
                        conv1_w_raddr <= cnt_oc * 27 + cnt_ic * 9 + cnt_kh * 3 + cnt_kw;
                        mac_valid <= 1;
                    end else begin
                        mac_valid <= 0;  // padding → contributes 0
                    end
                    mac_phase <= 1;
                end else if (mac_phase == 1) begin
                    // Phase 1: BRAM data valid — register x_adj and weight (breaks critical path)
                    if (mac_valid) begin
                        x_adj_reg <= $signed({1'b0, img_buf_rdata})
                                     - $signed({{1{XZP_CONV1[7]}}, XZP_CONV1});
                        w_reg     <= conv1_w_rdata;
                    end
                    mac_phase <= 2;
                end else begin
                    // Phase 2: registered values available — accumulate
                    if (mac_valid)
                        acc <= acc + $signed(x_adj_reg) * $signed(w_reg);
                    mac_phase <= 0;
                    // Advance inner loop kw → kh → ic
                    if (cnt_kw < 2)      cnt_kw <= cnt_kw + 1;
                    else begin  cnt_kw <= 0;
                        if (cnt_kh < 2)  cnt_kh <= cnt_kh + 1;
                        else begin  cnt_kh <= 0;
                            if (cnt_ic < 2)  cnt_ic <= cnt_ic + 1;  // 3 input channels
                            else begin  cnt_ic <= 0;  state <= S_CONV1_STORE; end
                        end
                    end
                end
            end
        end

        S_CONV1_STORE: begin
            // Cycle 1: register acc*M to break combinational multiply critical path
            req_tmp <= $signed(acc) * $signed({1'b0, M_CONV1});
            state   <= S_CONV1_REQUAN;
        end

        S_CONV1_REQUAN: begin
            // Cycle 2: shift + clamp + write; req_tmp is now stable
            pre_pool[cnt_oc * 1024 + cnt_oh * 32 + cnt_ow] <=
                clamp8(((req_tmp + 64'sd524288) >>> 20)
                       + $signed({{24{YZP_CONV1[7]}}, YZP_CONV1}));
            // Advance outer loop ow → oh → oc
            if (cnt_ow < 31)     begin cnt_ow <= cnt_ow + 1; state <= S_CONV1_INIT; end
            else begin  cnt_ow <= 0;
                if (cnt_oh < 31) begin cnt_oh <= cnt_oh + 1; state <= S_CONV1_INIT; end
                else begin  cnt_oh <= 0;
                    if (cnt_oc < 15) begin cnt_oc <= cnt_oc + 1; state <= S_CONV1_INIT; end
                    else begin  cnt_oc <= 0;  cnt_oh <= 0;  cnt_ow <= 0;
                        pool_sub <= 0;  state <= S_POOL1; end
                end
            end
        end

        // =====================================================================
        // POOL1 : 16×32×32 → 16×16×16 , MaxPool 2×2
        //   Output (oc,oh,ow) = max of 4 inputs at (oc, 2oh+{0,1}, 2ow+{0,1})
        //   Pool steps: 0→read v0, 1→read v1, 2→read v2, 3→read v3, 4→store
        // =====================================================================
        S_POOL1: begin
            // 5-step BRAM-pipelined max-pool 2x2.
            // pre_pool_rdata is valid ONE cycle after pre_pool_raddr is set.
            //   sub=0: issue addr for pv0
            //   sub=1: capture pv0, issue addr for pv1
            //   sub=2: capture pv1, issue addr for pv2
            //   sub=3: capture pv2, issue addr for pv3 (= 4th value)
            //   sub=4: pre_pool_rdata = pv3 value; compute max, store, advance
            case (pool_sub)
                3'd0: pre_pool_raddr <= cnt_oc * 1024 + cnt_oh * 64 + cnt_ow * 2;
                3'd1: begin pv0 <= pre_pool_rdata;
                            pre_pool_raddr <= cnt_oc * 1024 + cnt_oh * 64 + cnt_ow * 2 + 1; end
                3'd2: begin pv1 <= pre_pool_rdata;
                            pre_pool_raddr <= cnt_oc * 1024 + cnt_oh * 64 + 32 + cnt_ow * 2; end
                3'd3: begin pv2 <= pre_pool_rdata;
                            pre_pool_raddr <= cnt_oc * 1024 + cnt_oh * 64 + 32 + cnt_ow * 2 + 1; end
                3'd4: begin
                    begin
                        logic signed [7:0] mx;
                        mx = pv0;
                        if ($signed(pv1)             > $signed(mx)) mx = pv1;
                        if ($signed(pv2)             > $signed(mx)) mx = pv2;
                        if ($signed(pre_pool_rdata)  > $signed(mx)) mx = pre_pool_rdata;
                        act_a[cnt_oc * 256 + cnt_oh * 16 + cnt_ow] <= mx;
                    end
                    if (cnt_ow < 15)     begin cnt_ow <= cnt_ow + 1; end
                    else begin  cnt_ow <= 0;
                        if (cnt_oh < 15) begin cnt_oh <= cnt_oh + 1; end
                        else begin  cnt_oh <= 0;
                            if (cnt_oc < 15) begin cnt_oc <= cnt_oc + 1; end
                            else begin
                                cnt_oc <= 0;  cnt_oh <= 0;  cnt_ow <= 0;
                                state  <= S_CONV2_INIT;
                            end
                        end
                    end
                end
                default: ;
            endcase
            if (pool_sub < 4) pool_sub <= pool_sub + 1;
            else              pool_sub <= 0;
        end

        // =====================================================================
        // CONV2 : 16×16×16 → 32×16×16 , pad=1 , x_zp=-128
        //   weight index : oc*144 + ic*9 + kh*3 + kw
        //   input  index : ic*256 + ih*16 + iw   (from act_a)
        // =====================================================================
        S_CONV2_INIT: begin
            acc    <= conv2_b[cnt_oc];
            cnt_ic <= 0;  cnt_kh <= 0;  cnt_kw <= 0;
            mac_phase <= 0;
            state  <= S_CONV2_MAC;
        end

        S_CONV2_MAC: begin
            begin
                logic signed [6:0] ih, iw;
                ih = $signed({2'b0, cnt_oh}) + $signed({2'b0, cnt_kh}) - 7'sd1;
                iw = $signed({2'b0, cnt_ow}) + $signed({2'b0, cnt_kw}) - 7'sd1;

                if (mac_phase == 0) begin
                    // Phase 0: register BRAM read addresses
                    if (ih >= 0 && ih < 16 && iw >= 0 && iw < 16) begin
                        act_a_raddr   <= cnt_ic * 256 + ih[3:0] * 16 + iw[3:0];
                        conv2_w_raddr <= cnt_oc * 144 + cnt_ic * 9 + cnt_kh * 3 + cnt_kw;
                        mac_valid <= 1;
                    end else begin
                        mac_valid <= 0;
                    end
                    mac_phase <= 1;
                end else if (mac_phase == 1) begin
                    // Phase 1: BRAM data valid — register x_adj and weight (x_zp=-128)
                    if (mac_valid) begin
                        x_adj_reg <= $signed(act_a_rdata) + 9'sd128;
                        w_reg     <= conv2_w_rdata;
                    end
                    mac_phase <= 2;
                end else begin
                    // Phase 2: registered values available — accumulate
                    if (mac_valid)
                        acc <= acc + $signed(x_adj_reg) * $signed(w_reg);
                    mac_phase <= 0;
                    if (cnt_kw < 2)      cnt_kw <= cnt_kw + 1;
                    else begin  cnt_kw <= 0;
                        if (cnt_kh < 2)  cnt_kh <= cnt_kh + 1;
                        else begin  cnt_kh <= 0;
                            if (cnt_ic < 15) cnt_ic <= cnt_ic + 1;  // 16 input channels
                            else begin  cnt_ic <= 0;  state <= S_CONV2_STORE; end
                        end
                    end
                end
            end
        end

        S_CONV2_STORE: begin
            req_tmp <= $signed(acc) * $signed({1'b0, M_CONV2});
            state   <= S_CONV2_REQUAN;
        end

        S_CONV2_REQUAN: begin
            pre_pool[cnt_oc * 256 + cnt_oh * 16 + cnt_ow] <=
                clamp8(((req_tmp + 64'sd524288) >>> 20)
                       + $signed({{24{YZP_CONV2[7]}}, YZP_CONV2}));
            if (cnt_ow < 15)     begin cnt_ow <= cnt_ow + 1; state <= S_CONV2_INIT; end
            else begin  cnt_ow <= 0;
                if (cnt_oh < 15) begin cnt_oh <= cnt_oh + 1; state <= S_CONV2_INIT; end
                else begin  cnt_oh <= 0;
                    if (cnt_oc < 31) begin cnt_oc <= cnt_oc + 1; state <= S_CONV2_INIT; end
                    else begin  cnt_oc <= 0;  cnt_oh <= 0;  cnt_ow <= 0;
                        pool_sub <= 0;  state <= S_POOL2; end
                end
            end
        end

        // =====================================================================
        // POOL2 : 32×16×16 → 32×8×8 , MaxPool 2×2
        //   Source: pre_pool[oc*256 + ...], dest: act_b[oc*64 + oh*8 + ow]
        // =====================================================================
        S_POOL2: begin
            case (pool_sub)
                3'd0: pre_pool_raddr <= cnt_oc * 256 + cnt_oh * 32 + cnt_ow * 2;
                3'd1: begin pv0 <= pre_pool_rdata;
                            pre_pool_raddr <= cnt_oc * 256 + cnt_oh * 32 + cnt_ow * 2 + 1; end
                3'd2: begin pv1 <= pre_pool_rdata;
                            pre_pool_raddr <= cnt_oc * 256 + cnt_oh * 32 + 16 + cnt_ow * 2; end
                3'd3: begin pv2 <= pre_pool_rdata;
                            pre_pool_raddr <= cnt_oc * 256 + cnt_oh * 32 + 16 + cnt_ow * 2 + 1; end
                3'd4: begin
                    begin
                        logic signed [7:0] mx;
                        mx = pv0;
                        if ($signed(pv1)            > $signed(mx)) mx = pv1;
                        if ($signed(pv2)            > $signed(mx)) mx = pv2;
                        if ($signed(pre_pool_rdata) > $signed(mx)) mx = pre_pool_rdata;
                        act_b[cnt_oc * 64 + cnt_oh * 8 + cnt_ow] <= mx;
                    end
                    if (cnt_ow < 7)      begin cnt_ow <= cnt_ow + 1; end
                    else begin  cnt_ow <= 0;
                        if (cnt_oh < 7)  begin cnt_oh <= cnt_oh + 1; end
                        else begin  cnt_oh <= 0;
                            if (cnt_oc < 31) begin cnt_oc <= cnt_oc + 1; end
                            else begin
                                cnt_oc <= 0;  cnt_oh <= 0;  cnt_ow <= 0;
                                state  <= S_CONV3_INIT;
                            end
                        end
                    end
                end
                default: ;
            endcase
            if (pool_sub < 4) pool_sub <= pool_sub + 1;
            else              pool_sub <= 0;
        end

        // =====================================================================
        // CONV3 : 32×8×8 → 32×8×8 , pad=1 , x_zp=-128
        //   weight index : oc*288 + ic*9 + kh*3 + kw
        //   input  index : ic*64  + ih*8 + iw   (from act_b)
        // =====================================================================
        S_CONV3_INIT: begin
            acc    <= conv3_b[cnt_oc];
            cnt_ic <= 0;  cnt_kh <= 0;  cnt_kw <= 0;
            mac_phase <= 0;
            state  <= S_CONV3_MAC;
        end

        S_CONV3_MAC: begin
            begin
                logic signed [5:0] ih, iw;
                ih = $signed({1'b0, cnt_oh[2:0]}) + $signed({1'b0, cnt_kh}) - 6'sd1;
                iw = $signed({1'b0, cnt_ow[2:0]}) + $signed({1'b0, cnt_kw}) - 6'sd1;

                if (mac_phase == 0) begin
                    // Phase 0: register BRAM read addresses
                    if (ih >= 0 && ih < 8 && iw >= 0 && iw < 8) begin
                        act_b_raddr   <= cnt_ic[4:0] * 64 + ih[2:0] * 8 + iw[2:0];
                        conv3_w_raddr <= cnt_oc * 288 + cnt_ic[4:0] * 9 + cnt_kh * 3 + cnt_kw;
                        mac_valid <= 1;
                    end else begin
                        mac_valid <= 0;
                    end
                    mac_phase <= 1;
                end else if (mac_phase == 1) begin
                    // Phase 1: BRAM data valid — register x_adj and weight (x_zp=-128)
                    if (mac_valid) begin
                        x_adj_reg <= $signed(act_b_rdata) + 9'sd128;
                        w_reg     <= conv3_w_rdata;
                    end
                    mac_phase <= 2;
                end else begin
                    // Phase 2: registered values available — accumulate
                    if (mac_valid)
                        acc <= acc + $signed(x_adj_reg) * $signed(w_reg);
                    mac_phase <= 0;
                    if (cnt_kw < 2)      cnt_kw <= cnt_kw + 1;
                    else begin  cnt_kw <= 0;
                        if (cnt_kh < 2)  cnt_kh <= cnt_kh + 1;
                        else begin  cnt_kh <= 0;
                            if (cnt_ic < 31) cnt_ic <= cnt_ic + 1;  // 32 input channels
                            else begin  cnt_ic <= 0;  state <= S_CONV3_STORE; end
                        end
                    end
                end
            end
        end

        S_CONV3_STORE: begin
            req_tmp <= $signed(acc) * $signed({1'b0, M_CONV3});
            state   <= S_CONV3_REQUAN;
        end

        S_CONV3_REQUAN: begin
            pre_pool[cnt_oc * 64 + cnt_oh[2:0] * 8 + cnt_ow[2:0]] <=
                clamp8(((req_tmp + 64'sd524288) >>> 20)
                       + $signed({{24{YZP_CONV3[7]}}, YZP_CONV3}));
            if (cnt_ow < 7)      begin cnt_ow <= cnt_ow + 1; state <= S_CONV3_INIT; end
            else begin  cnt_ow <= 0;
                if (cnt_oh < 7)  begin cnt_oh <= cnt_oh + 1; state <= S_CONV3_INIT; end
                else begin  cnt_oh <= 0;
                    if (cnt_oc < 31) begin cnt_oc <= cnt_oc + 1; state <= S_CONV3_INIT; end
                    else begin  cnt_oc <= 0;  cnt_oh <= 0;  cnt_ow <= 0;
                        pool_sub <= 0;  state <= S_POOL3; end
                end
            end
        end

        // =====================================================================
        // POOL3 : 32×8×8 → 32×4×4 , MaxPool 2×2
        // =====================================================================
        S_POOL3: begin
            case (pool_sub)
                3'd0: pre_pool_raddr <= cnt_oc * 64 + cnt_oh * 16 + cnt_ow * 2;
                3'd1: begin pv0 <= pre_pool_rdata;
                            pre_pool_raddr <= cnt_oc * 64 + cnt_oh * 16 + cnt_ow * 2 + 1; end
                3'd2: begin pv1 <= pre_pool_rdata;
                            pre_pool_raddr <= cnt_oc * 64 + cnt_oh * 16 + 8 + cnt_ow * 2; end
                3'd3: begin pv2 <= pre_pool_rdata;
                            pre_pool_raddr <= cnt_oc * 64 + cnt_oh * 16 + 8 + cnt_ow * 2 + 1; end
                3'd4: begin
                    begin
                        logic signed [7:0] mx;
                        mx = pv0;
                        if ($signed(pv1)            > $signed(mx)) mx = pv1;
                        if ($signed(pv2)            > $signed(mx)) mx = pv2;
                        if ($signed(pre_pool_rdata) > $signed(mx)) mx = pre_pool_rdata;
                        act_c[cnt_oc[4:0] * 16 + cnt_oh[1:0] * 4 + cnt_ow[1:0]] <= mx;
                    end
                    if (cnt_ow < 3)      begin cnt_ow <= cnt_ow + 1; end
                    else begin  cnt_ow <= 0;
                        if (cnt_oh < 3)  begin cnt_oh <= cnt_oh + 1; end
                        else begin  cnt_oh <= 0;
                            if (cnt_oc < 31) begin cnt_oc <= cnt_oc + 1; end
                            else begin
                                cnt_oc <= 0;  cnt_ic <= 0;
                                state  <= S_FC1_INIT;
                            end
                        end
                    end
                end
                default: ;
            endcase
            if (pool_sub < 4) pool_sub <= pool_sub + 1;
            else              pool_sub <= 0;
        end

        // =====================================================================
        // FC1 : 512 → 64 ,  x_zp=-128
        //   Input = act_c flattened (32×4×4 = 512)
        //   weight index : oc*512 + ic
        // =====================================================================
        S_FC1_INIT: begin
            acc    <= fc1_b[cnt_oc];
            cnt_ic <= 0;
            mac_phase <= 0;
            state  <= S_FC1_MAC;
        end

        S_FC1_MAC: begin
            if (mac_phase == 0) begin
                // Phase 0: register BRAM read addresses
                act_c_raddr  <= cnt_ic[8:0];
                fc1_w_raddr  <= cnt_oc * 512 + cnt_ic;
                mac_phase    <= 1;
            end else if (mac_phase == 1) begin
                // Phase 1: BRAM data valid — register x_adj and weight (x_zp=-128)
                x_adj_reg <= $signed(act_c_rdata) + 9'sd128;
                w_reg     <= fc1_w_rdata;
                mac_phase <= 2;
            end else begin
                // Phase 2: registered values available — accumulate
                acc <= acc + $signed(x_adj_reg) * $signed(w_reg);
                mac_phase <= 0;
                if (cnt_ic < 511) cnt_ic <= cnt_ic + 1;
                else begin  cnt_ic <= 0;  state <= S_FC1_STORE; end
            end
        end

        S_FC1_STORE: begin
            req_tmp <= $signed(acc) * $signed({1'b0, M_FC1});
            state   <= S_FC1_REQUAN;
        end

        S_FC1_REQUAN: begin
            act_d[cnt_oc] <=
                clamp8(((req_tmp + 64'sd524288) >>> 20)
                       + $signed({{24{YZP_FC1[7]}}, YZP_FC1}));
            if (cnt_oc < 63) begin cnt_oc <= cnt_oc + 1; state <= S_FC1_INIT; end
            else begin  cnt_oc <= 0;  cnt_ic <= 0;  state <= S_FC2_INIT; end
        end

        // =====================================================================
        // FC2 : 64 → 10 ,  x_zp=-128
        //   weight index : oc*64 + ic
        //   Store raw int32 in logits[] (skip requant — argmax on int32 is correct)
        // =====================================================================
        S_FC2_INIT: begin
            acc    <= fc2_b[cnt_oc];
            cnt_ic <= 0;
            state  <= S_FC2_MAC;
        end

        S_FC2_MAC: begin
            begin
                logic signed [8:0] x_adj;
                x_adj = $signed(act_d[cnt_ic[5:0]]) + 9'sd128;
                acc   <= acc + $signed(x_adj) * $signed(fc2_w[cnt_oc * 64 + cnt_ic[5:0]]);
            end

            if (cnt_ic < 63) cnt_ic <= cnt_ic + 1;
            else begin  cnt_ic <= 0;  state <= S_FC2_STORE; end
        end

        S_FC2_STORE: begin
            logits[cnt_oc[3:0]] <= acc;   // store raw int32 (M_FC2 * acc is monotone → argmax safe)

            if (cnt_oc < 9) begin cnt_oc <= cnt_oc + 1; state <= S_FC2_INIT; end
            else begin
                cnt_oc      <= 0;
                arg_i       <= 0;
                result      <= 0;
                arg_max_val <= logits[0];
                state       <= S_ARGMAX;
            end
        end

        // =====================================================================
        // ARGMAX : scan logits[0..9] for maximum
        // =====================================================================
        S_ARGMAX: begin
            if (arg_i < 9) begin
                arg_i <= arg_i + 1;
                if (logits[arg_i + 1] > arg_max_val) begin
                    arg_max_val <= logits[arg_i + 1];
                    result      <= arg_i + 1;
                end
            end else begin
                state <= S_DONE;
            end
        end

        // =====================================================================
        S_DONE: begin
            done  <= 1;
            busy  <= 0;
            state <= S_IDLE;
        end

        default: state <= S_IDLE;
        endcase
    end
end

endmodule
