#!/usr/bin/env python3
"""
Generate waveform plots and RTL vs Python comparison charts.
Parses the VCD file from RTL simulation and plots key signals.
Also runs the same computations in Python for side-by-side comparison.
"""

import os
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

# -------------------------------------------------------------------------
# VCD Parser (minimal, for our signals)
# -------------------------------------------------------------------------
def parse_vcd(vcd_path):
    """Parse VCD file and extract signal traces."""
    signals = {}      # var_id -> {"name": ..., "width": ..., "values": [(time, val), ...]}
    id_to_name = {}
    current_time = 0

    with open(vcd_path) as f:
        in_defs = False
        scope_stack = []

        for line in f:
            line = line.strip()

            if line.startswith("$scope"):
                parts = line.split()
                if len(parts) >= 3:
                    scope_stack.append(parts[2])

            elif line.startswith("$upscope"):
                if scope_stack:
                    scope_stack.pop()

            elif line.startswith("$var"):
                parts = line.split()
                # $var wire 32 ! acc_out [31:0] $end
                if len(parts) >= 5:
                    width = int(parts[2])
                    var_id = parts[3]
                    name = parts[4]
                    full_name = ".".join(scope_stack + [name])
                    signals[var_id] = {"name": full_name, "width": width, "values": []}
                    id_to_name[var_id] = full_name

            elif line.startswith("#"):
                current_time = int(line[1:])

            elif line and line[0] in "01xzXZ":
                # Single-bit value change: 0! or 1!
                val = line[0]
                var_id = line[1:]
                if var_id in signals:
                    signals[var_id]["values"].append((current_time, val))

            elif line and line[0] == "b":
                # Multi-bit: bXXXX var_id
                parts = line.split()
                if len(parts) == 2:
                    val_str = parts[0][1:]  # remove 'b'
                    var_id = parts[1]
                    if var_id in signals:
                        try:
                            val = int(val_str, 2)
                        except ValueError:
                            val = 0
                        signals[var_id]["values"].append((current_time, val))

    return signals, id_to_name


def get_signal_trace(signals, name_substring):
    """Find a signal by name substring and return (times, values)."""
    for sid, info in signals.items():
        if name_substring in info["name"]:
            times = [v[0] for v in info["values"]]
            vals = [v[1] if isinstance(v[1], int) else (1 if v[1] == "1" else 0)
                    for v in info["values"]]
            return times, vals, info["name"]
    return [], [], ""


# -------------------------------------------------------------------------
# Python simulation (same test vectors as RTL)
# -------------------------------------------------------------------------
def python_mac_simulation():
    """Run the same test vectors as the RTL testbench in pure Python."""
    results = {}

    # Test 1: Single MAC
    results["Single MAC (42*12)"] = 42 * 12

    # Generate same pseudo-random test data as SystemVerilog
    conv1_data = [(i * 7 + 13) % 256 for i in range(27)]
    conv1_weight = [(i * 11 + 3) % 256 for i in range(27)]
    conv2_data = [(i * 5 + 17) % 256 for i in range(144)]
    conv2_weight = [(i * 13 + 7) % 256 for i in range(144)]
    fc_data = [(i * 3 + 29) % 256 for i in range(64)]
    fc_weight = [(i * 9 + 41) % 256 for i in range(64)]

    # Test 2: Conv1 dot product
    results["Conv1 dot product (27 MACs)"] = sum(d * w for d, w in zip(conv1_data, conv1_weight))

    # Test 3: Conv2 dot product
    results["Conv2 dot product (144 MACs)"] = sum(d * w for d, w in zip(conv2_data, conv2_weight))

    # Test 4: FC dot product
    results["FC dot product (64 MACs)"] = sum(d * w for d, w in zip(fc_data, fc_weight))

    # Test 5: Layer sequencing
    results["Layer seq: Conv1 result"] = results["Conv1 dot product (27 MACs)"]
    results["Layer seq: Conv2 result"] = results["Conv2 dot product (144 MACs)"]

    return results


# -------------------------------------------------------------------------
# Plot 1: Waveform diagram
# -------------------------------------------------------------------------
def plot_waveforms(signals):
    """Generate waveform plot from VCD data."""
    fig, axes = plt.subplots(6, 1, figsize=(16, 10), sharex=True)
    fig.patch.set_facecolor("#1a1a2e")

    signal_configs = [
        ("clock", "clock", "#00d2ff", False),
        ("reset", "reset", "#ff6b6b", False),
        ("enable", "enable", "#00e676", False),
        ("clear_acc", "clear_acc", "#ffa500", False),
        ("data_in", "data_in", "#bb86fc", True),
        ("acc_out", "acc_out", "#00e676", True),
    ]

    for ax_idx, (search_name, display_name, color, is_multi) in enumerate(signal_configs):
        ax = axes[ax_idx]
        ax.set_facecolor("#25253d")

        times, vals, full_name = get_signal_trace(signals, search_name)

        if times:
            # Convert to step plot
            plot_times = []
            plot_vals = []
            for i, (t, v) in enumerate(zip(times, vals)):
                if plot_times:
                    plot_times.append(t)
                    plot_vals.append(plot_vals[-1])
                plot_times.append(t)
                plot_vals.append(v)
            # Extend to end
            if plot_times:
                plot_times.append(max(plot_times) + 10)
                plot_vals.append(plot_vals[-1])

            if is_multi:
                ax.fill_between(plot_times, plot_vals, alpha=0.3, color=color, step="post")
                ax.step(plot_times, plot_vals, where="post", color=color, linewidth=1.5)
            else:
                ax.step(plot_times, plot_vals, where="post", color=color, linewidth=2)
                ax.fill_between(plot_times, plot_vals, alpha=0.2, color=color, step="post")
                ax.set_ylim(-0.2, 1.5)

        ax.set_ylabel(display_name, color="white", fontsize=11, fontweight="bold")
        ax.tick_params(colors="white", labelsize=9)
        ax.spines["bottom"].set_color("#555")
        ax.spines["left"].set_color("#555")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, alpha=0.15, color="white")

    axes[-1].set_xlabel("Simulation Time (ns)", color="white", fontsize=12)
    axes[0].set_title("MAC Unit RTL Waveform — Conv1 + Conv2 + FC Dot Products",
                       color="#00d2ff", fontsize=16, fontweight="bold", pad=15)

    # Annotate test regions
    # Test times (approximate from VCD): test1 ~40-70, test2 70-360, test3 360-1820, test4 1820-2480
    test_regions = [
        (35, 75, "Test 1\nSingle MAC"),
        (75, 380, "Test 2\nConv1 (27 MACs)"),
        (380, 1850, "Test 3\nConv2 (144 MACs)"),
        (1850, 2520, "Test 4\nFC (64 MACs)"),
        (2520, 4225, "Test 5\nLayer Seq"),
    ]
    for start, end, label in test_regions:
        axes[0].axvspan(start, end, alpha=0.08, color="white")
        mid = (start + end) / 2
        axes[0].text(mid, axes[0].get_ylim()[1] * 0.5, label,
                     ha="center", va="center", color="white", fontsize=7, alpha=0.7)

    plt.tight_layout()
    out_path = os.path.join(SCRIPT_DIR, "waveform_rtl.png")
    plt.savefig(out_path, dpi=150, facecolor="#1a1a2e", bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# -------------------------------------------------------------------------
# Plot 2: RTL vs Python comparison bar chart
# -------------------------------------------------------------------------
def plot_comparison(rtl_results_path, python_results):
    """Side-by-side bar chart comparing RTL and Python results."""

    # Parse RTL CSV
    rtl_results = {}
    with open(rtl_results_path) as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split(",")
            if len(parts) == 4:
                status, test, rtl_val, exp_val = parts
                rtl_results[test] = int(rtl_val)

    # Build comparison
    tests = list(python_results.keys())
    python_vals = [python_results[t] for t in tests]
    rtl_vals = [rtl_results.get(t, 0) for t in tests]
    matches = ["MATCH" if p == r else "MISMATCH" for p, r in zip(python_vals, rtl_vals)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7),
                                     gridspec_kw={"width_ratios": [3, 1]})
    fig.patch.set_facecolor("#1a1a2e")
    fig.suptitle("RTL vs Python Simulation Comparison",
                 color="#00d2ff", fontsize=20, fontweight="bold", y=0.98)

    # Bar chart
    ax1.set_facecolor("#25253d")
    x = np.arange(len(tests))
    width = 0.35
    bars1 = ax1.bar(x - width/2, python_vals, width, label="Python (int simulation)",
                     color="#00d2ff", alpha=0.8)
    bars2 = ax1.bar(x + width/2, rtl_vals, width, label="RTL (iverilog)",
                     color="#00e676", alpha=0.8)

    ax1.set_xlabel("Test Case", color="white", fontsize=12)
    ax1.set_ylabel("Accumulator Result (int32)", color="white", fontsize=12)
    ax1.set_xticks(x)
    short_labels = ["Single\nMAC", "Conv1\n27 MACs", "Conv2\n144 MACs",
                     "FC\n64 MACs", "Seq:\nConv1", "Seq:\nConv2"]
    ax1.set_xticklabels(short_labels, color="white", fontsize=10)
    ax1.tick_params(colors="white")
    ax1.legend(fontsize=11, loc="upper left")
    ax1.spines["bottom"].set_color("#555")
    ax1.spines["left"].set_color("#555")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.grid(True, alpha=0.15, color="white", axis="y")

    # Add value labels on bars
    for bar, val in zip(bars1, python_vals):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                 f"{val:,}", ha="center", va="bottom", color="#00d2ff", fontsize=8, fontweight="bold")
    for bar, val in zip(bars2, rtl_vals):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                 f"{val:,}", ha="center", va="bottom", color="#00e676", fontsize=8, fontweight="bold")

    # Match/mismatch summary table
    ax2.set_facecolor("#25253d")
    ax2.axis("off")

    table_data = [["Test", "Python", "RTL", "Status"]]
    for i, t in enumerate(tests):
        short = short_labels[i].replace("\n", " ")
        table_data.append([short, f"{python_vals[i]:,}", f"{rtl_vals[i]:,}", matches[i]])

    table = ax2.table(cellText=table_data, cellLoc="center", loc="center",
                       colWidths=[0.25, 0.25, 0.25, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(10)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#555")
        if row == 0:
            cell.set_facecolor("#00d2ff")
            cell.set_text_props(color="black", fontweight="bold")
        else:
            cell.set_facecolor("#25253d")
            if col == 3:
                if cell.get_text().get_text() == "MATCH":
                    cell.set_facecolor("#004020")
                    cell.set_text_props(color="#00e676", fontweight="bold")
                else:
                    cell.set_facecolor("#400020")
                    cell.set_text_props(color="#ff6b6b", fontweight="bold")
            else:
                cell.set_text_props(color="white")
        cell.set_height(0.1)

    plt.tight_layout()
    out_path = os.path.join(SCRIPT_DIR, "rtl_vs_python_comparison.png")
    plt.savefig(out_path, dpi=150, facecolor="#1a1a2e", bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# -------------------------------------------------------------------------
# Plot 3: Accumulator growth over time (zoomed into Conv2)
# -------------------------------------------------------------------------
def plot_accumulator_growth(signals):
    """Show accumulator growing cycle-by-cycle during Conv2 dot product."""
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#25253d")

    times, vals, _ = get_signal_trace(signals, "acc_out")

    if times:
        # Find Conv2 region (test 3): after ~380ns, 144 MACs
        # We look for the region where acc grows from 0 to ~2.2M
        conv2_times = []
        conv2_vals = []
        in_conv2 = False
        for t, v in zip(times, vals):
            if t >= 380 and t <= 1850:
                conv2_times.append(t)
                conv2_vals.append(v)

        if conv2_times:
            # Also compute Python reference
            conv2_data = [(i * 5 + 17) % 256 for i in range(144)]
            conv2_weight = [(i * 13 + 7) % 256 for i in range(144)]
            py_cumulative = []
            running = 0
            for d, w in zip(conv2_data, conv2_weight):
                running += d * w
                py_cumulative.append(running)

            # Plot RTL trace
            ax.step(conv2_times, conv2_vals, where="post", color="#00e676",
                    linewidth=2, label="RTL accumulator", alpha=0.9)
            ax.fill_between(conv2_times, conv2_vals, alpha=0.15, color="#00e676", step="post")

            # Plot Python reference (scaled to same time axis)
            if len(py_cumulative) > 1:
                py_times = np.linspace(conv2_times[0], conv2_times[-1], len(py_cumulative))
                ax.plot(py_times, py_cumulative, color="#00d2ff", linewidth=2,
                        linestyle="--", label="Python reference", alpha=0.9)

            ax.set_xlabel("Simulation Time (ns)", color="white", fontsize=12)
            ax.set_ylabel("Accumulator Value (int32)", color="white", fontsize=12)
            ax.set_title("Conv2 Dot Product: Accumulator Growth (144 MACs) — RTL vs Python",
                         color="#00d2ff", fontsize=14, fontweight="bold")
            ax.legend(fontsize=12)
            ax.tick_params(colors="white")
            ax.spines["bottom"].set_color("#555")
            ax.spines["left"].set_color("#555")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(True, alpha=0.15, color="white")

            # Annotate final value
            final_val = conv2_vals[-1]
            ax.annotate(f"Final: {final_val:,}",
                        xy=(conv2_times[-1], final_val),
                        xytext=(conv2_times[-1] - 300, final_val * 0.8),
                        color="#00e676", fontsize=12, fontweight="bold",
                        arrowprops=dict(arrowstyle="->", color="#00e676"))

    plt.tight_layout()
    out_path = os.path.join(SCRIPT_DIR, "accumulator_growth.png")
    plt.savefig(out_path, dpi=150, facecolor="#1a1a2e", bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("  RTL vs Python Comparison & Waveform Generation")
    print("=" * 60)

    # Parse VCD
    vcd_path = os.path.join(PROJECT_DIR, "rtl_vs_python.vcd")
    print(f"\nParsing VCD: {vcd_path}")
    signals, id_to_name = parse_vcd(vcd_path)
    print(f"  Found {len(signals)} signals")

    # Python simulation
    print("\nRunning Python simulation...")
    python_results = python_mac_simulation()
    for test, val in python_results.items():
        print(f"  {test}: {val:,}")

    # Generate plots
    print("\nGenerating waveform plot...")
    plot_waveforms(signals)

    print("Generating RTL vs Python comparison chart...")
    rtl_csv = os.path.join(PROJECT_DIR, "rtl_results.csv")
    plot_comparison(rtl_csv, python_results)

    print("Generating accumulator growth plot...")
    plot_accumulator_growth(signals)

    print("\nDone! Images saved to tiny-cnn-basys3/")


if __name__ == "__main__":
    main()
