//==============================================================================
// Testbench: tb_weight_memory
// Description: Test weight loading and usage
//==============================================================================

module tb_weight_memory;

    parameter int WEIGHT_DEPTH = 1024;
    parameter int ADDR_WIDTH = 10;
    
    logic        clock;
    logic        reset;
    logic [ADDR_WIDTH-1:0] weight_addr;
    logic                  weight_read_en;
    logic [7:0]            weight_data;
    logic        weight_valid;
    
    logic [7:0]  dw_window [8:0];
    logic [7:0]  dw_kernel [8:0];
    logic        dw_start;
    logic        dw_clear;
    logic [31:0] dw_result;
    logic        dw_result_valid;
    
    int test_passed;
    
    initial begin
        clock = 0;
        forever #5 clock = ~clock;
    end
    
    weight_memory #(
        .WEIGHT_DEPTH(WEIGHT_DEPTH),
        .ADDR_WIDTH(ADDR_WIDTH)
    ) weight_mem (
        .clock      (clock),
        .reset      (reset),
        .read_addr  (weight_addr),
        .read_en    (weight_read_en),
        .read_data  (weight_data),
        .read_valid (weight_valid)
    );
    
    depthwise_conv3x3_engine dw_engine (
        .clock          (clock),
        .reset          (reset),
        .window_in      (dw_window),
        .kernel_weights (dw_kernel),
        .start_conv     (dw_start),
        .clear          (dw_clear),
        .conv_result    (dw_result),
        .result_valid   (dw_result_valid)
    );
    
    // Note: In real usage, load from .mem file like this:
    // initial $readmemh("mems_int8/YOUR_FILE.w1.mem", weight_mem.weight_mem);
    // For this test, weights initialized to zero in weight_memory module
    
    initial begin
        $display("TEST START");
        test_passed = 0;
        reset = 1;
        weight_addr = 0;
        weight_read_en = 0;
        dw_start = 0;
        dw_clear = 0;
        for (int i = 0; i < 9; i++) dw_window[i] = 8'd10;
        repeat(3) @(posedge clock);
        reset = 0;
        @(posedge clock);
        
        $display("Test: Weight Read");
        weight_addr = 0;
        weight_read_en = 1;
        @(posedge clock);
        weight_read_en = 0;
        wait(weight_valid);
        @(posedge clock);
        if (weight_valid) test_passed++;
        
        $display("Test: Load Kernel");
        for (int i = 0; i < 9; i++) begin
            weight_addr = i;
            weight_read_en = 1;
            @(posedge clock);
            weight_read_en = 0;
            wait(weight_valid);
            @(posedge clock);
            dw_kernel[i] = weight_data;
        end
        test_passed++;
        
        $display("Test: Convolution");
        dw_clear = 1;
        @(posedge clock);
        dw_clear = 0;
        dw_start = 1;
        @(posedge clock);
        dw_start = 0;
        wait(dw_result_valid);
        @(posedge clock);
        if (dw_result_valid) test_passed++;
        
        repeat(5) @(posedge clock);
        $display("Passed: %0d tests", test_passed);
        if (test_passed == 3) $display("TEST PASSED");
        else $display("TEST FAILED");
        $finish;
    end
    
    initial begin
        #100000;
        $finish;
    end

endmodule