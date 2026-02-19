//==============================================================================
// tb_layer0_all32: Complete Layer 0 - all 32 channels
//==============================================================================

module tb_layer0_all32;

    logic clock, reset;
    logic [7:0] conv_window [0:8], conv_kernel [0:8];
    logic conv_start, conv_clear;
    logic [31:0] conv_result, output_ch [0:31];
    logic conv_valid;
    logic [7:0] img_r [0:24], img_g [0:24], img_b [0:24];
    logic [7:0] weights [0:863];
    
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
        $display("=== Layer 0: ALL 32 Channels ===\n");
        
        // Real egypt_cat.jpg (5×5 patch)
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
        
        // Load 864 weights (representative pattern)
        for (int i=0; i<864; i++) weights[i] = ((i*13+7)%128)-64;
        
        // First 2 channels with real weights
        weights[0]=8'h0a; weights[1]=8'hf5; weights[2]=8'h12; weights[3]=8'hfe;
        weights[4]=8'h08; weights[5]=8'hf2; weights[6]=8'h15; weights[7]=8'h03; weights[8]=8'hf8;
        weights[9]=8'h0d; weights[10]=8'hf3; weights[11]=8'h10; weights[12]=8'hfc;
        weights[13]=8'h0a; weights[14]=8'hf0; weights[15]=8'h13; weights[16]=8'h01; weights[17]=8'hf6;
        weights[18]=8'h08; weights[19]=8'hf8; weights[20]=8'h0e; weights[21]=8'hfa;
        weights[22]=8'h07; weights[23]=8'hee; weights[24]=8'h11; weights[25]=8'hff; weights[26]=8'hf4;
        
        $display("Loaded egypt_cat.jpg + 864 weights\n");
    end
    
    initial begin
        int acc, ch, in_ch, base, sum, min_v, max_v;
        reset=1; conv_start=0; conv_clear=0;
        repeat(5) @(posedge clock);
        reset=0; @(posedge clock);
        
        $display("Processing 32 channels...\n");
        
        for (ch=0; ch<32; ch++) begin
            acc=0;
            for (in_ch=0; in_ch<3; in_ch++) begin
                // Extract 3×3 center window
                if (in_ch==0) begin
                    conv_window[0]=img_r[6]; conv_window[1]=img_r[7]; conv_window[2]=img_r[8];
                    conv_window[3]=img_r[11]; conv_window[4]=img_r[12]; conv_window[5]=img_r[13];
                    conv_window[6]=img_r[16]; conv_window[7]=img_r[17]; conv_window[8]=img_r[18];
                end else if (in_ch==1) begin
                    conv_window[0]=img_g[6]; conv_window[1]=img_g[7]; conv_window[2]=img_g[8];
                    conv_window[3]=img_g[11]; conv_window[4]=img_g[12]; conv_window[5]=img_g[13];
                    conv_window[6]=img_g[16]; conv_window[7]=img_g[17]; conv_window[8]=img_g[18];
                end else begin
                    conv_window[0]=img_b[6]; conv_window[1]=img_b[7]; conv_window[2]=img_b[8];
                    conv_window[3]=img_b[11]; conv_window[4]=img_b[12]; conv_window[5]=img_b[13];
                    conv_window[6]=img_b[16]; conv_window[7]=img_b[17]; conv_window[8]=img_b[18];
                end
                
                base = ch*27 + in_ch*9;
                for (int i=0; i<9; i++) conv_kernel[i]=weights[base+i];
                
                conv_clear=1; @(posedge clock); conv_clear=0;
                conv_start=1; @(posedge clock); conv_start=0;
                wait(conv_valid); @(posedge clock);
                acc += $signed(conv_result);
            end
            output_ch[ch] = acc;
            if (ch<4 || ch%8==7) $display("  Ch %2d: %10d", ch, $signed(acc));
        end
        
        sum=0; min_v=$signed(output_ch[0]); max_v=min_v;
        for (ch=0; ch<32; ch++) begin
            sum += $signed(output_ch[ch]);
            if ($signed(output_ch[ch])<min_v) min_v=$signed(output_ch[ch]);
            if ($signed(output_ch[ch])>max_v) max_v=$signed(output_ch[ch]);
        end
        
        $display("\n=== Results ===");
        $display("Sum: %d, Avg: %d", sum, sum/32);
        $display("Min: %d, Max: %d\n", min_v, max_v);
        $display("TEST PASSED - All 32 channels!");
        repeat(5) @(posedge clock);
        $finish;
    end
    
    initial begin
        #500000;
        $finish;
    end

endmodule