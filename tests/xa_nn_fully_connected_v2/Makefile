# =============================================================================
# Makefile  —  FC CModel gcc test rig
#
# Targets
# -------
#   make build          compile both test binaries
#   make patterns       generate .in_out fixtures (runs gen_fc_patterns.py)
#   make run            build + run pattern harness (test_fc_patterns)
#   make run-tc         build + run host self-test  (test_fc_cmodel  TC01-15)
#   make run-all        patterns + run + run-tc
#   make clean          remove build/ and patterns/*.in_out
#   make help           print usage
#
# Variables (override on command line)
#   CC           = gcc
#   PYTHON       = python3
#   BUILD_DIR    = build
#   PATTERN_DIR  = patterns
# =============================================================================

CC          ?= gcc
PYTHON      ?= python3
BUILD_DIR   ?= build
PATTERN_DIR ?= patterns

CFLAGS  := -std=c11 -Wall -Wextra -O2 -g
LDFLAGS := -lm

SRC_CMODEL   := cmodel_fc_hifi5.c
SRC_PATTERNS := test_fc_patterns.c
SRC_TC       := test_fc_cmodel.c

BIN_PATTERNS := $(BUILD_DIR)/test_fc_patterns
BIN_TC       := $(BUILD_DIR)/test_fc_cmodel

GEN_SCRIPT   := $(PATTERN_DIR)/gen_fc_patterns.py

# ---------------------------------------------------------------------------
.PHONY: build patterns run run-tc run-all clean help

## Default goal
all: binaries

## Compile both binaries (alias 'build' also works)
binaries build: $(BIN_PATTERNS) $(BIN_TC)

$(BUILD_DIR)/.mkdir:
	mkdir -p $(BUILD_DIR)
	touch $(BUILD_DIR)/.mkdir

$(BIN_PATTERNS): $(SRC_PATTERNS) $(SRC_CMODEL) cmodel_fc_hifi5.h | $(BUILD_DIR)/.mkdir
	$(CC) $(CFLAGS) -o $@ $(SRC_PATTERNS) $(SRC_CMODEL) $(LDFLAGS)
	@echo "[build] $@ OK"

$(BIN_TC): $(SRC_TC) $(SRC_CMODEL) cmodel_fc_hifi5.h | $(BUILD_DIR)/.mkdir
	$(CC) $(CFLAGS) -o $@ $(SRC_TC) $(SRC_CMODEL) $(LDFLAGS)
	@echo "[build] $@ OK"

## Generate .in_out fixture files
patterns: $(GEN_SCRIPT)
	@echo "[gen] Generating fixtures in $(PATTERN_DIR)/ ..."
	cd $(PATTERN_DIR) && $(PYTHON) gen_fc_patterns.py
	@echo "[gen] Done."

## Run pattern harness
run: $(BIN_PATTERNS)
	@echo "[run] FC pattern test ..."
	$(BIN_PATTERNS)

## Run TC01-15 host self-test
run-tc: $(BIN_TC)
	@echo "[run-tc] FC cmodel self-test ..."
	$(BIN_TC)

## Generate patterns + run both tests
run-all: patterns build run run-tc

## Clean everything
clean:
	rm -rf $(BUILD_DIR)
	rm -f $(PATTERN_DIR)/*.in_out
	@echo "[clean] Done."

## Help
help:
	@echo ""
	@echo "FC CModel gcc test rig"
	@echo "----------------------"
	@echo "  make build       compile binaries ($(BIN_PATTERNS), $(BIN_TC))"
	@echo "  make patterns    generate .in_out fixtures via gen_fc_patterns.py"
	@echo "  make run         run pattern harness (golden + partial_sum compare)"
	@echo "  make run-tc      run TC01-15 host self-test"
	@echo "  make run-all     patterns + build + run + run-tc"
	@echo "  make clean       remove build/ and patterns/*.in_out"
	@echo ""
	@echo "Variables:"
	@echo "  CC=$(CC)  PYTHON=$(PYTHON)  BUILD_DIR=$(BUILD_DIR)"
	@echo ""
