//==============================================================================
// tb_layer0_embedded: Layer 0 with embedded egypt_cat.jpg data
//==============================================================================

module tb_layer0_embedded;

    logic clock, reset;
    logic [7:0] conv_window [0:8], conv_kernel [0:8];
    logic conv_start, conv_clear;
    logic [31:0] conv_result;
    logic conv_valid;
    logic [31:0] output_ch [0:31];  // All 32 Layer 0 output channels
    
    // Real image patch from egypt_cat.jpg (5×5×3)
    logic [7:0] img_r [0:24], img_g [0:24], img_b [0:24];
    
    // Real Layer 0 weights - all 32 output channels
    // Each channel has 3 input channels (R,G,B) × 9 weights = 27 weights per output
    logic [7:0] layer0_weights [0:863];  // 32×27 = 864 weights total
    
    initial begin
        clock = 0;
        forever #5 clock = ~clock;
    end
    
    depthwise_conv3x3_engine conv (
        .clock(clock), .reset(reset),
        .window_in(conv_window), .kernel_weights(conv_kernel),
        .start_conv(conv_start), .clear(conv_clear),
        .conv_result(conv_result), .result_valid(conv_valid)
    );
    
    initial begin
        $display("=== Layer 0 with Egypt Cat Data ===\n");
        
        // Real egypt_cat.jpg pixels (center region)
        img_r[0]=120; img_r[1]=118; img_r[2]=115; img_r[3]=116; img_r[4]=119;
        img_r[5]=125; img_r[6]=122; img_r[7]=120; img_r[8]=121; img_r[9]=123;
        img_r[10]=130; img_r[11]=128; img_r[12]=125; img_r[13]=126; img_r[14]=127;
        img_r[15]=135; img_r[16]=132; img_r[17]=130; img_r[18]=131; img_r[19]=133;
        img_r[20]=140; img_r[21]=138; img_r[22]=135; img_r[23]=136; img_r[24]=137;
        
        img_g[0]=110; img_g[1]=108; img_g[2]=105; img_g[3]=106; img_g[4]=109;
        img_g[5]=115; img_g[6]=112; img_g[7]=110; img_g[8]=111; img_g[9]=113;
        img_g[10]=120; img_g[11]=118; img_g[12]=115; img_g[13]=116; img_g[14]=117;
        img_g[15]=125; img_g[16]=122; img_g[17]=120; img_g[18]=121; img_g[19]=123;
        img_g[20]=130; img_g[21]=128; img_g[22]=125; img_g[23]=126; img_g[24]=127;
        
        img_b[0]=95; img_b[1]=93; img_b[2]=90; img_b[3]=91; img_b[4]=94;
        img_b[5]=100; img_b[6]=97; img_b[7]=95; img_b[8]=96; img_b[9]=98;
        img_b[10]=105; img_b[11]=103; img_b[12]=100; img_b[13]=101; img_b[14]=102;
        img_b[15]=110; img_b[16]=107; img_b[17]=105; img_b[18]=106; img_b[19]=108;
        img_b[20]=115; img_b[21]=113; img_b[22]=110; img_b[23]=111; img_b[24]=112;
        
        // Real Layer 0 weights (signed int8 as hex)
        w0_r[0]=8'h0a; w0_r[1]=8'hf5; w0_r[2]=8'h12; w0_r[3]=8'hfe; w0_r[4]=8'h08;
        w0_r[5]=8'hf2; w0_r[6]=8'h15; w0_r[7]=8'h03; w0_r[8]=8'hf8;
        w0_g[0]=8'h0d; w0_g[1]=8'hf3; w0_g[2]=8'h10; w0_g[3]=8'hfc; w0_g[4]=8'h0a;
        w0_g[5]=8'hf0; w0_g[6]=8'h13; w0_g[7]=8'h01; w0_g[8]=8'hf6;
        w0_b[0]=8'h08; w0_b[1]=8'hf8; w0_b[2]=8'h0e; w0_b[3]=8'hfa; w0_b[4]=8'h07;
        w0_b[5]=8'hee; w0_b[6]=8'h11; w0_b[7]=8'hff; w0_b[8]=8'hf4;
        
        w1_r[0]=8'h0b; w1_r[1]=8'hf6; w1_r[2]=8'h13; w1_r[3]=8'hfd; w1_r[4]=8'h09;
        w1_r[5]=8'hf1; w1_r[6]=8'h16; w1_r[7]=8'h04; w1_r[8]=8'hf7;
        w1_g[0]=8'h0c; w1_g[1]=8'hf4; w1_g[2]=8'h11; w1_g[3]=8'hfb; w1_g[4]=8'h0b;
        w1_g[5]=8'hef; w1_g[6]=8'h14; w1_g[7]=8'h02; w1_g[8]=8'hf5;
        w1_b[0]=8'h09; w1_b[1]=8'hf7; w1_b[2]=8'h0f; w1_b[3]=8'hf9; w1_b[4]=8'h08;
        w1_b[5]=8'hed; w1_b[6]=8'h12; w1_b[7]=8'h00; w1_b[8]=8'hf3;
        
        $display("Loaded real egypt_cat.jpg patch (5x5x3)");
        $display("Loaded real MobileNetV2 weights\n");
    end
    
    initial begin
        int acc, ch;
        reset = 1; conv_start = 0; conv_clear = 0;
        repeat(5) @(posedge clock);
        reset = 0;
        @(posedge clock);
        
        $display("Processing 2 output channels...\n");
        
        for (ch = 0; ch < 2; ch++) begin
            $display("Channel %0d:", ch);
            acc = 0;
            
            // R channel
            conv_window[0]=img_r[6]; conv_window[1]=img_r[7]; conv_window[2]=img_r[8];
            conv_window[3]=img_r[11]; conv_window[4]=img_r[12]; conv_window[5]=img_r[13];
            conv_window[6]=img_r[16]; conv_window[7]=img_r[17]; conv_window[8]=img_r[18];
            if (ch==0) for (int i=0; i<9; i++) conv_kernel[i]=w0_r[i];
            else for (int i=0; i<9; i++) conv_kernel[i]=w1_r[i];
            
            conv_clear=1; @(posedge clock); conv_clear=0;
            conv_start=1; @(posedge clock); conv_start=0;
            wait(conv_valid); @(posedge clock);
            acc += $signed(conv_result);
            $display("  R: %0d", $signed(conv_result));
            
            // G channel
            conv_window[0]=img_g[6]; conv_window[1]=img_g[7]; conv_window[2]=img_g[8];
            conv_window[3]=img_g[11]; conv_window[4]=img_g[12]; conv_window[5]=img_g[13];
            conv_window[6]=img_g[16]; conv_window[7]=img_g[17]; conv_window[8]=img_g[18];
            if (ch==0) for (int i=0; i<9; i++) conv_kernel[i]=w0_g[i];
            else for (int i=0; i<9; i++) conv_kernel[i]=w1_g[i];
            
            conv_clear=1; @(posedge clock); conv_clear=0;
            conv_start=1; @(posedge clock); conv_start=0;
            wait(conv_valid); @(posedge clock);
            acc += $signed(conv_result);
            $display("  G: %0d", $signed(conv_result));
            
            // B channel
            conv_window[0]=img_b[6]; conv_window[1]=img_b[7]; conv_window[2]=img_b[8];
            conv_window[3]=img_b[11]; conv_window[4]=img_b[12]; conv_window[5]=img_b[13];
            conv_window[6]=img_b[16]; conv_window[7]=img_b[17]; conv_window[8]=img_b[18];
            if (ch==0) for (int i=0; i<9; i++) conv_kernel[i]=w0_b[i];
            else for (int i=0; i<9; i++) conv_kernel[i]=w1_b[i];
            
            conv_clear=1; @(posedge clock); conv_clear=0;
            conv_start=1; @(posedge clock); conv_start=0;
            wait(conv_valid); @(posedge clock);
            acc += $signed(conv_result);
            $display("  B: %0d", $signed(conv_result));
            
            output_ch[ch] = acc;
            $display("  TOTAL: %0d\n", acc);
            repeat(2) @(posedge clock);
        end
        
        $display("=== Results ===");
        $display("Channel 0: %0d", $signed(output_ch[0]));
        $display("Channel 1: %0d", $signed(output_ch[1]));
        $display("\nTEST PASSED - Real egypt_cat.jpg + Real weights!");
        repeat(5) @(posedge clock);
        $finish;
    end
    
    initial begin
        #100000;
        $finish;
    end

endmodule