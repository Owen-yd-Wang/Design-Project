#!/usr/bin/env python3
"""
Rebuild Tiny_CNN_Basys3_Presentation.pptx from scratch using python-pptx.
Professional dark-blue title bar / white background theme.
"""

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
import os

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
DARK_BLUE = RGBColor(0x00, 0x2B, 0x5C)      # title bar background
MEDIUM_BLUE = RGBColor(0x00, 0x4E, 0x8C)     # accent / subtitle
LIGHT_BLUE = RGBColor(0x00, 0x7B, 0xC0)      # highlights
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
NEAR_WHITE = RGBColor(0xF5, 0xF5, 0xF5)
BLACK = RGBColor(0x1A, 0x1A, 0x1A)
GRAY = RGBColor(0x66, 0x66, 0x66)
LIGHT_GRAY = RGBColor(0xAA, 0xAA, 0xAA)

SLIDE_WIDTH = Inches(13.333)
SLIDE_HEIGHT = Inches(7.5)

# Image paths
IMG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tiny-cnn-basys3")
IMG_WAVEFORM = os.path.join(IMG_DIR, "waveform_rtl.png")
IMG_RTL_VS_PYTHON = os.path.join(IMG_DIR, "rtl_vs_python_comparison.png")
IMG_ACCUMULATOR = os.path.join(IMG_DIR, "accumulator_growth.png")

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def add_title_bar(slide, title_text):
    """Add a dark-blue title bar at the top of the slide with white title text."""
    bar = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE,
        left=Inches(0), top=Inches(0),
        width=SLIDE_WIDTH, height=Inches(1.1),
    )
    bar.fill.solid()
    bar.fill.fore_color.rgb = DARK_BLUE
    bar.line.fill.background()

    tf = bar.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(0.6)
    tf.margin_top = Inches(0.15)
    tf.margin_bottom = Inches(0.15)
    p = tf.paragraphs[0]
    p.text = title_text
    p.font.size = Pt(30)
    p.font.bold = True
    p.font.color.rgb = WHITE
    p.alignment = PP_ALIGN.LEFT
    return bar


def add_body_textbox(slide, left, top, width, height):
    """Add a textbox and return the text_frame."""
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = True
    return tf


def add_bullet(tf, text, level=0, font_size=20, bold=False, color=BLACK, first=False):
    """Add a bullet paragraph to an existing text_frame."""
    if first and tf.paragraphs[0].text == "":
        p = tf.paragraphs[0]
    else:
        p = tf.add_paragraph()
    p.text = text
    p.level = level
    p.space_after = Pt(6)
    p.space_before = Pt(4)
    p.font.size = Pt(font_size)
    p.font.bold = bold
    p.font.color.rgb = color
    return p


def add_footer(slide, text, color=LIGHT_GRAY, font_size=11):
    """Add a small footer line at the bottom of the slide."""
    txBox = slide.shapes.add_textbox(
        Inches(0.5), Inches(6.9), Inches(12.3), Inches(0.5)
    )
    tf = txBox.text_frame
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(font_size)
    p.font.color.rgb = color
    p.alignment = PP_ALIGN.CENTER


def add_thin_accent_line(slide, top):
    """Add a thin accent line across the slide."""
    line = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE,
        left=Inches(0.6), top=top,
        width=Inches(12.1), height=Pt(2),
    )
    line.fill.solid()
    line.fill.fore_color.rgb = LIGHT_BLUE
    line.line.fill.background()


# ---------------------------------------------------------------------------
# Build presentation
# ---------------------------------------------------------------------------

prs = Presentation()
prs.slide_width = SLIDE_WIDTH
prs.slide_height = SLIDE_HEIGHT

# Use the blank layout for all slides
blank_layout = prs.slide_layouts[6]  # blank


# ==============================
# Slide 1 — Title Slide
# ==============================
slide1 = prs.slides.add_slide(blank_layout)
# Full-slide dark blue background
bg_shape = slide1.shapes.add_shape(
    MSO_SHAPE.RECTANGLE,
    left=Inches(0), top=Inches(0),
    width=SLIDE_WIDTH, height=SLIDE_HEIGHT,
)
bg_shape.fill.solid()
bg_shape.fill.fore_color.rgb = DARK_BLUE
bg_shape.line.fill.background()

# Accent line
accent = slide1.shapes.add_shape(
    MSO_SHAPE.RECTANGLE,
    left=Inches(1.5), top=Inches(3.15),
    width=Inches(10.3), height=Pt(3),
)
accent.fill.solid()
accent.fill.fore_color.rgb = LIGHT_BLUE
accent.line.fill.background()

# Title
tf = add_body_textbox(slide1, Inches(1.5), Inches(1.5), Inches(10.3), Inches(1.5))
p = tf.paragraphs[0]
p.text = "Tiny CNN for CIFAR-10 on Basys 3 FPGA"
p.font.size = Pt(42)
p.font.bold = True
p.font.color.rgb = WHITE
p.alignment = PP_ALIGN.CENTER

# Subtitle
tf2 = add_body_textbox(slide1, Inches(1.5), Inches(3.4), Inches(10.3), Inches(1.0))
p2 = tf2.paragraphs[0]
p2.text = "CogniChip Co-Design Platform \u2014 Affordable Edge AI"
p2.font.size = Pt(24)
p2.font.color.rgb = NEAR_WHITE
p2.alignment = PP_ALIGN.CENTER

# Authors
tf3 = add_body_textbox(slide1, Inches(1.5), Inches(4.6), Inches(10.3), Inches(0.8))
p3 = tf3.paragraphs[0]
p3.text = "Che Li & Owen Wang  |  Advisors: Ramesh Karri, Weihua Xiao"
p3.font.size = Pt(18)
p3.font.color.rgb = RGBColor(0xBB, 0xCC, 0xDD)
p3.alignment = PP_ALIGN.CENTER

# GitHub
tf4 = add_body_textbox(slide1, Inches(1.5), Inches(5.5), Inches(10.3), Inches(0.6))
p4 = tf4.paragraphs[0]
p4.text = "GitHub: github.com/Owen-yd-Wang/Design-Project"
p4.font.size = Pt(14)
p4.font.color.rgb = LIGHT_BLUE
p4.alignment = PP_ALIGN.CENTER


# ==============================
# Slide 2 — Problem Statement
# ==============================
slide2 = prs.slides.add_slide(blank_layout)
add_title_bar(slide2, "Problem: GPU-Class Models Don\u2019t Fit on $35 FPGAs")

tf = add_body_textbox(slide2, Inches(0.8), Inches(1.5), Inches(11.7), Inches(5.0))
add_bullet(tf, "MobileNetV2: 3.5M params, 13.3 MB \u2192 needs ~15\u00d7 more memory than Basys 3",
           font_size=22, first=True)
add_bullet(tf, "Basys 3 (XC7A35T): 225 KB BRAM, 90 DSP48E1, no external DRAM", font_size=22)
add_bullet(tf, "Challenge: Design a CNN that fits AND maintains useful accuracy",
           font_size=22, bold=True, color=MEDIUM_BLUE)


# ==============================
# Slide 3 — Our Approach
# ==============================
slide3 = prs.slides.add_slide(blank_layout)
add_title_bar(slide3, "Methodology: Train \u2192 Quantize \u2192 Simulate \u2192 Verify")

tf = add_body_textbox(slide3, Inches(0.8), Inches(1.5), Inches(11.7), Inches(5.0))
add_bullet(tf, "Train tiny CNN on CIFAR-10 (10 classes, 32\u00d732 images)", font_size=22, first=True)
add_bullet(tf, "Int8 quantization: 4\u00d7 size reduction, <0.1% accuracy loss", font_size=22)
add_bullet(tf, "Behavioral FPGA simulation: model clock cycles, BRAM, DSP usage", font_size=22)
add_bullet(tf, "RTL verification: bit-exact match between hardware and Python", font_size=22)


# ==============================
# Slide 4 — CNN Architecture
# ==============================
slide4 = prs.slides.add_slide(blank_layout)
add_title_bar(slide4, "Architecture: 48K Parameters, 47 KB Int8")

# Network layers in a styled box
arch_box = slide4.shapes.add_shape(
    MSO_SHAPE.ROUNDED_RECTANGLE,
    left=Inches(0.8), top=Inches(1.5),
    width=Inches(11.7), height=Inches(4.5),
)
arch_box.fill.solid()
arch_box.fill.fore_color.rgb = RGBColor(0xF0, 0xF4, 0xF8)
arch_box.line.color.rgb = RGBColor(0xCC, 0xDD, 0xEE)
arch_box.line.width = Pt(1)

tf = arch_box.text_frame
tf.word_wrap = True
tf.margin_left = Inches(0.4)
tf.margin_top = Inches(0.3)

layers = [
    ("Conv1:", "3\u219216, 3\u00d73 \u2192 ReLU \u2192 MaxPool 2\u00d72   (16\u00d716\u00d716)"),
    ("Conv2:", "16\u219232, 3\u00d73 \u2192 ReLU \u2192 MaxPool 2\u00d72   (8\u00d78\u00d732)"),
    ("Conv3:", "32\u219232, 3\u00d73 \u2192 ReLU \u2192 MaxPool 2\u00d72   (4\u00d74\u00d732)"),
    ("FC1:", "512\u219264 \u2192 ReLU"),
    ("FC2:", "64\u219210 (output)"),
]

for i, (label, desc) in enumerate(layers):
    if i == 0:
        p = tf.paragraphs[0]
    else:
        p = tf.add_paragraph()
    run_label = p.add_run()
    run_label.text = label + "  "
    run_label.font.size = Pt(20)
    run_label.font.bold = True
    run_label.font.color.rgb = DARK_BLUE
    run_desc = p.add_run()
    run_desc.text = desc
    run_desc.font.size = Pt(20)
    run_desc.font.color.rgb = BLACK
    p.space_after = Pt(8)

# Summary line
p_sum = tf.add_paragraph()
p_sum.space_before = Pt(14)
run_s = p_sum.add_run()
run_s.text = "Total: 47,818 params  |  Int8: ~47 KB  |  BRAM: 29.4% of Basys 3"
run_s.font.size = Pt(20)
run_s.font.bold = True
run_s.font.color.rgb = MEDIUM_BLUE


# ==============================
# Slide 5 — RTL Design
# ==============================
slide5 = prs.slides.add_slide(blank_layout)
add_title_bar(slide5, "RTL Design: MAC Unit (mac_uint8_int32.sv)")

tf = add_body_textbox(slide5, Inches(0.8), Inches(1.5), Inches(11.7), Inches(5.0))
add_bullet(tf, "Core compute: uint8 \u00d7 uint8 \u2192 uint32 accumulator", font_size=22, first=True)
add_bullet(tf, "Maps to single DSP48E1 slice on Artix-7", font_size=22)
add_bullet(tf, "Synchronous clear, enable control, registered output", font_size=22)
add_bullet(tf, "Shared across all layers (conv + FC) via time-multiplexing", font_size=22)


# ==============================
# Slide 6 — Waveform (IMAGE)
# ==============================
slide6 = prs.slides.add_slide(blank_layout)
add_title_bar(slide6, "RTL Simulation: MAC Unit Waveform")

if os.path.exists(IMG_WAVEFORM):
    # Place image: nearly full width, below title bar
    slide6.shapes.add_picture(
        IMG_WAVEFORM,
        left=Inches(0.8), top=Inches(1.3),
        width=Inches(11.7), height=Inches(5.2),
    )

add_footer(slide6, "Real trained weights from tiny_cnn_cifar10_int8.onnx, CIFAR-10 cat image",
           color=GRAY, font_size=12)


# ==============================
# Slide 7 — RTL vs Python (IMAGE)
# ==============================
slide7 = prs.slides.add_slide(blank_layout)
add_title_bar(slide7, "RTL vs Python: Bit-Exact Match (Real Weights)")

if os.path.exists(IMG_RTL_VS_PYTHON):
    slide7.shapes.add_picture(
        IMG_RTL_VS_PYTHON,
        left=Inches(0.8), top=Inches(1.3),
        width=Inches(11.7), height=Inches(5.2),
    )

add_footer(slide7,
           "All 4 tests pass \u2014 RTL accumulator matches Python unsigned arithmetic exactly",
           color=GRAY, font_size=12)


# ==============================
# Slide 8 — Accumulator Growth (IMAGE)
# ==============================
slide8 = prs.slides.add_slide(blank_layout)
add_title_bar(slide8, "RTL vs Python: Conv2 Accumulator Trace")

if os.path.exists(IMG_ACCUMULATOR):
    slide8.shapes.add_picture(
        IMG_ACCUMULATOR,
        left=Inches(0.8), top=Inches(1.3),
        width=Inches(11.7), height=Inches(5.2),
    )

add_footer(slide8,
           "144 MAC operations using real Conv2 filter 0 weights, final value: 2,576,776",
           color=GRAY, font_size=12)


# ==============================
# Slide 9 — Behavioral Simulation Results
# ==============================
slide9 = prs.slides.add_slide(blank_layout)
add_title_bar(slide9, "Behavioral FPGA Simulation Results")

tf = add_body_textbox(slide9, Inches(0.8), Inches(1.5), Inches(11.7), Inches(5.0))
add_bullet(tf, "Float32 accuracy:  F1 = 0.7400", font_size=22, first=True)
add_bullet(tf, "Int8 accuracy:       F1 = 0.7395  (delta: \u22120.0005)", font_size=22)
add_bullet(tf, "Peak BRAM: 66.1 KB / 225 KB (29.4%)", font_size=22)
add_bullet(tf, "Estimated throughput: 4,008 images/sec @ 100 MHz", font_size=22)
add_bullet(tf, "Latency: 0.25 ms per image", font_size=22)


# ==============================
# Slide 10 — Performance Breakdown
# ==============================
slide10 = prs.slides.add_slide(blank_layout)
add_title_bar(slide10, "Performance Breakdown by Layer")

# Table: 7 rows x 5 cols
rows, cols = 7, 5
tbl_shape = slide10.shapes.add_table(rows, cols,
    left=Inches(0.8), top=Inches(1.4),
    width=Inches(11.7), height=Inches(4.5),
)
tbl = tbl_shape.table

# Column widths
col_widths = [Inches(1.6), Inches(2.5), Inches(2.0), Inches(3.0), Inches(2.6)]
for i, w in enumerate(col_widths):
    tbl.columns[i].width = w

headers = ["Layer", "Weights", "Size (KB)", "MACs", "Compute %"]
data = [
    ["Conv1",  "432",     "0.4",   "230K",   "12.0%"],
    ["Conv2",  "4,608",   "4.5",   "1.18M",  "52.5%  \u2190 bottleneck"],
    ["Conv3",  "9,216",   "9.0",   "590K",   "26.3%"],
    ["FC1",    "32,768",  "32.0",  "33K",    "1.5%"],
    ["FC2",    "640",     "0.6",   "640",    "<0.1%"],
]

# Style header row
for j, h in enumerate(headers):
    cell = tbl.cell(0, j)
    cell.text = h
    for paragraph in cell.text_frame.paragraphs:
        paragraph.font.size = Pt(18)
        paragraph.font.bold = True
        paragraph.font.color.rgb = WHITE
        paragraph.alignment = PP_ALIGN.CENTER
    cell.fill.solid()
    cell.fill.fore_color.rgb = DARK_BLUE

# Fill data
for i, row_data in enumerate(data):
    for j, val in enumerate(row_data):
        cell = tbl.cell(i + 1, j)
        cell.text = val
        for paragraph in cell.text_frame.paragraphs:
            paragraph.font.size = Pt(17)
            paragraph.font.color.rgb = BLACK
            paragraph.alignment = PP_ALIGN.CENTER
        # Alternate row shading
        cell.fill.solid()
        if i % 2 == 0:
            cell.fill.fore_color.rgb = RGBColor(0xE8, 0xEF, 0xF5)
        else:
            cell.fill.fore_color.rgb = WHITE

# Highlight Conv2 row (row index 2 in table)
for j in range(cols):
    cell = tbl.cell(2, j)
    cell.fill.solid()
    cell.fill.fore_color.rgb = RGBColor(0xFF, 0xF0, 0xD0)

# Summary row
summary_row = 6
tbl.cell(summary_row, 0).merge(tbl.cell(summary_row, 4))
cell = tbl.cell(summary_row, 0)
cell.text = "Bottleneck: Conv2 (52.5% of compute)"
for paragraph in cell.text_frame.paragraphs:
    paragraph.font.size = Pt(18)
    paragraph.font.bold = True
    paragraph.font.color.rgb = MEDIUM_BLUE
    paragraph.alignment = PP_ALIGN.CENTER
cell.fill.solid()
cell.fill.fore_color.rgb = RGBColor(0xF0, 0xF4, 0xF8)


# ==============================
# Slide 11 — Challenges & Lessons
# ==============================
slide11 = prs.slides.add_slide(blank_layout)
add_title_bar(slide11, "Challenges & Lessons Learned")

tf = add_body_textbox(slide11, Inches(0.8), Inches(1.5), Inches(11.7), Inches(5.0))
add_bullet(tf, "Signed vs unsigned arithmetic: MAC unit uses uint8, not int8", font_size=22, first=True)
add_bullet(tf, "Quantization: minimal accuracy loss with proper calibration", font_size=22)
add_bullet(tf, "Resource constraints drive architecture decisions", font_size=22)
add_bullet(tf, "Behavioral simulation catches issues before synthesis", font_size=22)


# ==============================
# Slide 12 — Future Work
# ==============================
slide12 = prs.slides.add_slide(blank_layout)
add_title_bar(slide12, "Next Steps")

tf = add_body_textbox(slide12, Inches(0.8), Inches(1.5), Inches(11.7), Inches(5.0))
add_bullet(tf, "Synthesize full CNN datapath on Basys 3", font_size=22, first=True)
add_bullet(tf, "UART interface for image input / classification output", font_size=22)
add_bullet(tf, "Multi-filter parallelism to improve throughput", font_size=22)
add_bullet(tf, "Compare with FINN-generated accelerator", font_size=22)

add_footer(slide12, "GitHub: github.com/Owen-yd-Wang/Design-Project", color=LIGHT_GRAY, font_size=13)


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "Tiny_CNN_Basys3_Presentation.pptx")
prs.save(output_path)
print(f"Saved presentation to {output_path}")
print(f"Total slides: {len(prs.slides)}")
