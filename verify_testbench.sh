#!/bin/bash
#==============================================================================
# Verification Script for Quantized MobileNetV2 Full Inference Testbench
#==============================================================================

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║      MobileNetV2 Inference Testbench Verification         ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check simulation log
LOG_FILE="${1:-simulation.log}"

if [ ! -f "$LOG_FILE" ]; then
    echo -e "${RED}✗ Error: Log file $LOG_FILE not found${NC}"
    echo "  Please run: ./run_testbench.bat or make sim"
    exit 1
fi

echo "[1] Analyzing simulation log: $LOG_FILE"
echo ""

# Check for successful compilation
if grep -q "Compilation successful" "$LOG_FILE" 2>/dev/null; then
    echo -e "${GREEN}✓ Compilation successful${NC}"
else
    if ! grep -q "Elaborated" "$LOG_FILE" 2>/dev/null; then
        echo -e "${YELLOW}⚠ Compilation status unknown (typical for some simulators)${NC}"
    fi
fi

# Check for test start
if grep -q "QUANTIZED MobileNetV2 FULL INFERENCE TESTBENCH" "$LOG_FILE"; then
    echo -e "${GREEN}✓ Testbench started${NC}"
else
    echo -e "${RED}✗ Testbench did not start properly${NC}"
    exit 1
fi

# Check data loading
echo ""
echo "[2] File Loading Status:"
if grep -q "test_image.mem" "$LOG_FILE"; then
    SIZE=$(grep "Loaded.*bytes" "$LOG_FILE" | head -1 | grep -oE "[0-9]+")
    echo -e "${GREEN}✓ Test image loaded (${SIZE} bytes)${NC}"
else
    echo -e "${YELLOW}⚠ Test image loading information not found${NC}"
fi

if grep -q "features.0.weight" "$LOG_FILE"; then
    echo -e "${GREEN}✓ Layer 0 weights loaded${NC}"
else
    echo -e "${YELLOW}⚠ Layer 0 weights status not found${NC}"
fi

if grep -q "features.3.weight" "$LOG_FILE"; then
    echo -e "${GREEN}✓ Layer 3 weights loaded${NC}"
else
    echo -e "${YELLOW}⚠ Layer 3 weights status not found${NC}"
fi

if grep -q "features.6.weight" "$LOG_FILE"; then
    echo -e "${GREEN}✓ Layer 6 weights loaded${NC}"
else
    echo -e "${YELLOW}⚠ Layer 6 weights status not found${NC}"
fi

if grep -q "classifier layer" "$LOG_FILE"; then
    echo -e "${GREEN}✓ Classifier weights loaded${NC}"
else
    echo -e "${YELLOW}⚠ Classifier weights status not found${NC}"
fi

# Check layer processing
echo ""
echo "[3] Layer Processing Status:"
if grep -q "PROCESSING LAYER 0" "$LOG_FILE"; then
    echo -e "${GREEN}✓ Layer 0 processing started${NC}"
    if grep -q "Layer 0 Complete" "$LOG_FILE"; then
        echo -e "${GREEN}  └─ Layer 0 complete${NC}"
    else
        echo -e "${YELLOW}  └─ Layer 0 status unknown${NC}"
    fi
else
    echo -e "${YELLOW}⚠ Layer 0 processing information not found${NC}"
fi

if grep -q "PROCESSING LAYER 3" "$LOG_FILE"; then
    echo -e "${GREEN}✓ Layer 3 processing started${NC}"
    if grep -q "Layer 3 Complete" "$LOG_FILE"; then
        echo -e "${GREEN}  └─ Layer 3 complete${NC}"
    else
        echo -e "${YELLOW}  └─ Layer 3 status unknown${NC}"
    fi
else
    echo -e "${YELLOW}⚠ Layer 3 processing information not found${NC}"
fi

if grep -q "PROCESSING LAYER 6" "$LOG_FILE"; then
    echo -e "${GREEN}✓ Layer 6 processing started${NC}"
    if grep -q "Layer 6 Complete" "$LOG_FILE"; then
        echo -e "${GREEN}  └─ Layer 6 complete${NC}"
    else
        echo -e "${YELLOW}  └─ Layer 6 status unknown${NC}"
    fi
else
    echo -e "${YELLOW}⚠ Layer 6 processing information not found${NC}"
fi

# Check classification
echo ""
echo "[4] Classification Results:"
if grep -q "CLASSIFICATION RESULTS" "$LOG_FILE"; then
    echo -e "${GREEN}✓ Classification stage reached${NC}"
    echo ""
    
    # Extract predicted class
    PRED_CLASS=$(grep "PREDICTED CLASS:" "$LOG_FILE" | tail -1 | grep -oE "[0-9]+")
    if [ ! -z "$PRED_CLASS" ]; then
        echo -e "  ${BLUE}Predicted Class: ${PRED_CLASS}${NC}"
    fi
    
    # Extract confidence score
    CONFIDENCE=$(grep "CONFIDENCE:" "$LOG_FILE" | tail -1 | grep -oE "\-?[0-9]+$")
    if [ ! -z "$CONFIDENCE" ]; then
        echo -e "  ${BLUE}Confidence Score: ${CONFIDENCE}${NC}"
    fi
    
    # Show output scores
    echo ""
    echo "  Output Logits (first 5):"
    grep "Class [0-4]:" "$LOG_FILE" | tail -5 | while read line; do
        echo "    $line"
    done
else
    echo -e "${RED}✗ Classification results not found${NC}"
fi

# Check for successful completion
echo ""
echo "[5] Test Completion Status:"
if grep -q "INFERENCE COMPLETE AND SUCCESSFUL" "$LOG_FILE"; then
    echo -e "${GREEN}✓ Inference completed successfully!${NC}"
elif grep -q "finish" "$LOG_FILE"; then
    echo -e "${YELLOW}⚠ Test finished (check for errors above)${NC}"
else
    echo -e "${RED}✗ Test did not complete properly${NC}"
    if grep -q "ERROR\|error\|Error" "$LOG_FILE"; then
        echo ""
        echo "Errors found in log:"
        grep -i "error" "$LOG_FILE" | head -5
    fi
fi

# Simulation time if available
echo ""
echo "[6] Simulation Statistics:"
TIME=$(grep -i "simulation time\|elapsed" "$LOG_FILE" | tail -1)
if [ ! -z "$TIME" ]; then
    echo -e "  ${BLUE}${TIME}${NC}"
else
    echo "  (Simulation time not available)"
fi

# Summary
echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║                  Verification Complete                    ║"
echo "╚════════════════════════════════════════════════════════════╝"

# Check if we should provide recommendations
echo ""
if grep -q "ERROR\|Timeout\|timeout" "$LOG_FILE"; then
    echo -e "${RED}⚠ Issues detected in simulation. Recommendations:${NC}"
    echo "  1. Check that all weight files are present in tiny_mems_int8/"
    echo "  2. Verify test_image.mem is properly formatted"
    echo "  3. Increase timeout if simulation was interrupted"
    echo "  4. Check simulator logs for detailed error information"
    exit 1
elif grep -q "INFERENCE COMPLETE" "$LOG_FILE"; then
    echo -e "${GREEN}✓ All tests passed! Inference completed successfully.${NC}"
    exit 0
else
    echo -e "${YELLOW}⚠ Test status unclear. Review the log file manually:${NC}"
    echo "  tail -100 $LOG_FILE"
    exit 1
fi
