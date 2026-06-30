################################################################################
# Makefile — Depthwise Conv2D gcc CModel Test
#
# Self-contained: only gcc required.  No Xtensa toolchain dependency.
#
# Usage:
#   make              — build test_dwconv2d_cmodel
#   make run-all      — run all dw_*.in_out patterns, print tally
#   make run P=patterns/dw_p01_basic_cm1.in_out  — run single pattern
#   make clean        — remove build artefacts
#
# Pattern generation:
#   python3 gen_depthwise_patterns.py
################################################################################

CC        = gcc
CFLAGS    = -Wall -Wextra -O2 -g -std=c99
LDFLAGS   =

TARGET    = test_dwconv2d_cmodel
SRCS      = test_dwconv2d_cmodel.c conv2d_depthwise_cmodel.c
OBJS      = $(SRCS:.c=.o)

PATTERN_DIR ?= patterns

.PHONY: all run run-all clean help

all: $(TARGET)

$(TARGET): $(OBJS)
	@echo ">>> Linking $@ ..."
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	@echo ">>> Built: $@"

%.o: %.c
	@echo "  [gcc] Compiling $< ..."
	$(CC) $(CFLAGS) -c $< -o $@

# Convenience: generate patterns before building/running
patterns:
	python3 gen_depthwise_patterns.py

run: $(TARGET)
	@if [ -n "$(P)" ]; then fixture="$(P)"; \
	elif [ -f "$(PATTERN_DIR)/dw_p01_basic_cm1.in_out" ]; then \
		fixture="$(PATTERN_DIR)/dw_p01_basic_cm1.in_out"; \
	else echo "ERROR: specify P=<fixture>"; exit 1; fi; \
	echo ">>> Running: $$fixture"; \
	./$(TARGET) < "$$fixture"

run-all: $(TARGET)
	@echo ">>> Running all patterns under $(PATTERN_DIR)/ ..."
	@echo ""
	@pass=0; fail=0; \
	for f in $(PATTERN_DIR)/dw_*.in_out; do \
		[ -f "$$f" ] || { echo "No patterns found in $(PATTERN_DIR)/"; exit 1; }; \
		echo "--- $$(basename $$f) ---"; \
		result=$$(./$(TARGET) < "$$f" 2>&1); \
		echo "$$result" | grep -E "Pattern loaded|PASS|FAIL|Mismatch|WARNING|ERROR"; \
		if echo "$$result" | grep -q "^PASS$$"; then pass=$$((pass+1)); \
		else fail=$$((fail+1)); fi; \
		echo ""; \
	done; \
	echo "=================================================="; \
	echo "Results: PASS=$$pass  FAIL=$$fail"; \
	echo "=================================================="

clean:
	rm -f $(OBJS) $(TARGET)
	@echo ">>> Clean done"

help:
	@echo "Usage:"
	@echo "  make              — build $(TARGET)"
	@echo "  make run-all      — run all dw_*.in_out in $(PATTERN_DIR)/"
	@echo "  make run P=<f>    — run single fixture"
	@echo "  make clean        — remove build artefacts"
	@echo "  python3 gen_depthwise_patterns.py  — generate fixtures"
