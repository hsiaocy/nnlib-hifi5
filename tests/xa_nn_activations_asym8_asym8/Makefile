# =============================================================================
# Makefile — xa_nn_activations_asym8_asym8 gcc test harness
#
# Self-contained: no Xtensa toolchain, no external wrapper dependencies.
# All compilation uses the host 'gcc'.
#
# Targets
# -------
#   make             — build the test binary (default)
#   make build       — same as above
#   make run         FIXTURE=<path>  — build + run one fixture
#   make run-all     — build + run every pattern in ./patterns/
#   make gen         — (re)generate all .in_out fixtures via Python
#   make clean       — remove build artefacts
#
# Variables
# ---------
#   CC       compiler  (default: gcc)
#   CFLAGS   extra flags
#   BUILD    build output directory (default: build)
#   FIXTURE  .in_out file to use with 'make run' (default: first found)
#
# Pattern files live in ./patterns/  and match  p??_*.in_out
# =============================================================================

CC       ?= gcc
CFLAGS   ?= -O2 -Wall -Wextra -std=c99
BUILD    ?= build
BINARY   := $(BUILD)/test_activation_cmodel
# Avoid 'Circular build <- build/...' warning: mark BUILD dir as order-only dep
BUILD_DIR_STAMP := $(BUILD)/.stamp

SRCS     := activation_cmodel.c test_activation_cmodel.c
INCS     := activation_cmodel.h

PATTERN_DIR := patterns

# Default fixture for 'make run' (first file found alphabetically)
FIXTURE  ?= $(firstword $(wildcard $(PATTERN_DIR)/p01_*.in_out))

.PHONY: build run run-all gen clean

## build — compile the test binary
build: $(BINARY)

$(BINARY): $(SRCS) $(INCS) | $(BUILD_DIR_STAMP)
	$(CC) $(CFLAGS) -I. -o $@ $(SRCS)
	@echo "[gcc] Built $@"

$(BUILD_DIR_STAMP):
	@mkdir -p $(BUILD)
	@touch $@

## run — build then run a single fixture
run: $(BINARY)
	@if [ -z "$(FIXTURE)" ]; then \
	    echo "ERROR: no FIXTURE specified and no p01_*.in_out found in $(PATTERN_DIR)/"; \
	    exit 1; \
	fi
	@echo ""
	@echo "=== Running fixture: $(FIXTURE) ==="
	$(BINARY) < $(FIXTURE)

## run-all — run every pattern in ./patterns/
run-all: $(BINARY)
	@pass=0; fail=0; \
	for f in $(PATTERN_DIR)/p??.in_out $(PATTERN_DIR)/p??_*.in_out; do \
	    [ -f "$$f" ] || continue; \
	    echo ""; \
	    echo "=== $$f ==="; \
	    if $(BINARY) < "$$f"; then \
	        pass=$$((pass + 1)); \
	    else \
	        fail=$$((fail + 1)); \
	    fi; \
	done; \
	echo ""; \
	echo "##################################################"; \
	echo "## TOTAL: $$pass PASS  /  $$fail FAIL"; \
	echo "##################################################"; \
	[ "$$fail" -eq 0 ]

## gen — regenerate all .in_out fixtures
gen:
	@mkdir -p $(PATTERN_DIR)
	python3 gen_activation_patterns.py --out-dir $(PATTERN_DIR)

## clean — remove build directory
clean:
	@echo "[clean] Removing $(BUILD)/"
	rm -rf $(BUILD)
	@echo "Done."
