## =============================================================================
## Basys3 (XC7A35T) Constraints for TinyCNN
## =============================================================================

## Clock 100 MHz
set_property PACKAGE_PIN W5  [get_ports CLK100MHZ]
set_property IOSTANDARD LVCMOS33 [get_ports CLK100MHZ]
create_clock -period 10.000 -name sys_clk [get_ports CLK100MHZ]

## Reset (BTNC - center button, active high after inversion in RTL)
set_property PACKAGE_PIN U18 [get_ports CPU_RESETN]
set_property IOSTANDARD LVCMOS33 [get_ports CPU_RESETN]

## USB-UART (FTDI FT2232HQ on Basys3)
set_property PACKAGE_PIN B18 [get_ports RsRx]
set_property IOSTANDARD LVCMOS33 [get_ports RsRx]
set_property PACKAGE_PIN A18 [get_ports RsTx]
set_property IOSTANDARD LVCMOS33 [get_ports RsTx]

## LEDs (LD0..LD15)
set_property PACKAGE_PIN U16 [get_ports {LED[0]}]
set_property PACKAGE_PIN E19 [get_ports {LED[1]}]
set_property PACKAGE_PIN U19 [get_ports {LED[2]}]
set_property PACKAGE_PIN V19 [get_ports {LED[3]}]
set_property PACKAGE_PIN W18 [get_ports {LED[4]}]
set_property PACKAGE_PIN U15 [get_ports {LED[5]}]
set_property PACKAGE_PIN U14 [get_ports {LED[6]}]
set_property PACKAGE_PIN V14 [get_ports {LED[7]}]
set_property PACKAGE_PIN V13 [get_ports {LED[8]}]
set_property PACKAGE_PIN V3  [get_ports {LED[9]}]
set_property PACKAGE_PIN W3  [get_ports {LED[10]}]
set_property PACKAGE_PIN U3  [get_ports {LED[11]}]
set_property PACKAGE_PIN P3  [get_ports {LED[12]}]
set_property PACKAGE_PIN N3  [get_ports {LED[13]}]
set_property PACKAGE_PIN P1  [get_ports {LED[14]}]
set_property PACKAGE_PIN L1  [get_ports {LED[15]}]
set_property IOSTANDARD LVCMOS33 [get_ports {LED[*]}]

## 7-Segment Display Cathodes (SEG a..g)
set_property PACKAGE_PIN W7  [get_ports {SEG[0]}]
set_property PACKAGE_PIN W6  [get_ports {SEG[1]}]
set_property PACKAGE_PIN U8  [get_ports {SEG[2]}]
set_property PACKAGE_PIN V8  [get_ports {SEG[3]}]
set_property PACKAGE_PIN U5  [get_ports {SEG[4]}]
set_property PACKAGE_PIN V5  [get_ports {SEG[5]}]
set_property PACKAGE_PIN U7  [get_ports {SEG[6]}]
set_property IOSTANDARD LVCMOS33 [get_ports {SEG[*]}]

## 7-Segment Anodes (AN0..AN3)
set_property PACKAGE_PIN U2  [get_ports {AN[0]}]
set_property PACKAGE_PIN U4  [get_ports {AN[1]}]
set_property PACKAGE_PIN V4  [get_ports {AN[2]}]
set_property PACKAGE_PIN W4  [get_ports {AN[3]}]
set_property IOSTANDARD LVCMOS33 [get_ports {AN[*]}]
