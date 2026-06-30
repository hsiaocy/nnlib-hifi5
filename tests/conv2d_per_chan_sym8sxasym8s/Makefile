# =============================================================================
# Makefile  —  Conv2D Std test dispatcher
#
# Selects target (hifi5 or cim) and pattern (p01..p12), then delegates to
# Makefile.hifi5 or Makefile.cim.
#
# Usage
# -----
#   make TARGET=hifi5  PATTERN=p03          # build only
#   make run TARGET=cim PATTERN=p11         # build + xt-run
#   make all TARGET=hifi5                   # run all patterns
#   make all                                # run all patterns on both targets
#   make clean TARGET=cim PATTERN=p05
#   make clean                              # clean all build artefacts
#
# Required environment / overrides
# ---------------------------------
#   NNLIB_BASE    path to xa_nnlib_hifi5 root
#                 default: ../../NW/TFLM/dsp/tflm/tflite-micro/tensorflow/lite/
#                          micro/tools/make/downloads/xa_nnlib_hifi5
#   XTENSA_CORE   Xtensa core name used to locate the pre-built library
#                 default: hifi5s
#   BUILD_DIR     output directory for compiled binaries
#                 default: build/
#
# TFLITE_SINGLE_ROUNDING is determined automatically per pattern:
#   p01  → SR=1   (single-rounding fixture)
#   all others → SR=0
# =============================================================================

# ---------------------------------------------------------------------------
# Project-level defaults
# ---------------------------------------------------------------------------
NNLIB_BASE  ?= ../../NW/TFLM/dsp/tflm/tflite-micro/tensorflow/lite/micro/tools/make/downloads/xa_nnlib_hifi5
XTENSA_CORE ?= hifi5s
BUILD_DIR   ?= build

CMODEL_SRC  := conv2d_std_cim_cmodel.c
CMODEL_INC  := .

# Directory holding the p*.in_out fixtures (override on command line).
PATTERN_DIR ?= ./patterns

# ---------------------------------------------------------------------------
# All known patterns
# ---------------------------------------------------------------------------
ALL_PATTERNS := p01 p02 p03 p04 p05 p06 p07 p08 p09 p10 p11 p12

# All valid targets
ALL_TARGETS  := hifi5 cim

# ---------------------------------------------------------------------------
# User inputs with defaults
# ---------------------------------------------------------------------------
TARGET  ?= hifi5
PATTERN ?= p01

# ---------------------------------------------------------------------------
# Validate TARGET
# ---------------------------------------------------------------------------
ifeq ($(filter $(TARGET),$(ALL_TARGETS)),)
  $(error Unknown TARGET='$(TARGET)'. Valid: $(ALL_TARGETS))
endif

# ---------------------------------------------------------------------------
# Resolve FIXTURE from PATTERN
# Glob the .in_out file matching the pattern prefix inside PATTERN_DIR
# (e.g. ./patterns/p03_stride2.in_out)
# ---------------------------------------------------------------------------
_FIXTURE_MATCH := $(wildcard $(PATTERN_DIR)/$(PATTERN)_*.in_out)
ifeq ($(_FIXTURE_MATCH),)
  # Only error if we are not in a clean-all or help context
  ifneq ($(MAKECMDGOALS),clean)
  ifneq ($(MAKECMDGOALS),help)
  ifneq ($(MAKECMDGOALS),all)
    $(error No .in_out file found for PATTERN='$(PATTERN)'. \
            Expected e.g. $(PATTERN_DIR)/$(PATTERN)_<desc>.in_out in current directory.)
  endif
  endif
  endif
  FIXTURE :=
else
  FIXTURE := $(firstword $(_FIXTURE_MATCH))
endif

# ---------------------------------------------------------------------------
# Per-pattern TFLITE_SINGLE_ROUNDING selection
#   p01 = single-rounding (SR=1); all others = double-rounding (SR=0)
# ---------------------------------------------------------------------------
ifeq ($(PATTERN),p01)
  TFLITE_SR := 1
else
  TFLITE_SR := 0
endif

# ---------------------------------------------------------------------------
# Sub-Makefile selection
# ---------------------------------------------------------------------------
ifeq ($(TARGET),hifi5)
  SUB_MK := Makefile.hifi5
else
  SUB_MK := Makefile.cim
endif

# ---------------------------------------------------------------------------
# Common variables forwarded to sub-Makefiles
# ---------------------------------------------------------------------------
# TOOLCHAIN is only meaningful for the cim sub-makefile (gcc default | xt).
# hifi5 ignores it (always xt-clang/xt-run).
TOOLCHAIN ?= gcc

SUBMAKE_VARS := \
    PATTERN=$(PATTERN) \
    FIXTURE=$(FIXTURE) \
    NNLIB_BASE=$(NNLIB_BASE) \
    XTENSA_CORE=$(XTENSA_CORE) \
    BUILD_DIR=$(BUILD_DIR) \
    TFLITE_SR=$(TFLITE_SR) \
    CMODEL_SRC=$(CMODEL_SRC) \
    CMODEL_INC=$(CMODEL_INC) \
    TOOLCHAIN=$(TOOLCHAIN)

# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------
.PHONY: build run all clean help _check_build_dir

## build — compile binary for TARGET+PATTERN (default goal)
build: _check_build_dir
	@echo "[$(TARGET)/$(PATTERN)] FIXTURE=$(FIXTURE)  SR=$(TFLITE_SR)"
	$(MAKE) -f $(SUB_MK) build $(SUBMAKE_VARS)

## run — compile + execute under xt-run
run: _check_build_dir
	@echo "[$(TARGET)/$(PATTERN)] FIXTURE=$(FIXTURE)  SR=$(TFLITE_SR)"
	$(MAKE) -f $(SUB_MK) run $(SUBMAKE_VARS)

## all [TARGET=hifi5|cim|<unset>] — run all patterns for one or both targets
all: _check_build_dir
ifeq ($(filter all,$(MAKECMDGOALS)),all)
  # Determine which targets to sweep
  ifdef TARGET_EXPLICIT
    _SWEEP_TARGETS := $(TARGET)
  else
    _SWEEP_TARGETS := $(ALL_TARGETS)
  endif
	@for tgt in $(ALL_TARGETS); do \
	    for pat in $(ALL_PATTERNS); do \
	        fix=$$(ls ${PATTERN_DIR}/$${pat}_*.in_out 2>/dev/null | head -1); \
	        if [ -z "$$fix" ]; then \
	            echo "[SKIP] No fixture for $$pat"; continue; \
	        fi; \
	        if [ "$$pat" = "p01" ]; then sr=1; else sr=0; fi; \
	        echo ""; \
	        echo "=== $$tgt / $$pat  FIXTURE=$$fix  SR=$$sr ==="; \
	        if [ "$$tgt" = "hifi5" ]; then submk=Makefile.hifi5; \
	        else submk=Makefile.cim; fi; \
	        $(MAKE) -f $$submk run \
	            PATTERN=$$pat FIXTURE=$$fix \
	            NNLIB_BASE=$(NNLIB_BASE) XTENSA_CORE=$(XTENSA_CORE) \
	            BUILD_DIR=$(BUILD_DIR) TFLITE_SR=$$sr \
	            CMODEL_SRC=$(CMODEL_SRC) CMODEL_INC=$(CMODEL_INC) \
	            TOOLCHAIN=$(TOOLCHAIN) \
	        || exit 1; \
	    done; \
	done
endif

## clean — remove build artefacts
##   make clean TARGET=cim PATTERN=p03   (single binary)
##   make clean                          (entire build/ directory)
clean:
ifdef PATTERN
	$(MAKE) -f $(SUB_MK) clean $(SUBMAKE_VARS)
else
	@echo "[clean] Removing $(BUILD_DIR)/"
	rm -rf $(BUILD_DIR)
endif

## help — print usage summary
help:
	@echo ""
	@echo "Conv2D Std test dispatcher"
	@echo "--------------------------"
	@echo "Targets:"
	@echo "  make [build] TARGET=<hifi5|cim> PATTERN=<p01..p12>"
	@echo "  make run     TARGET=<hifi5|cim> PATTERN=<p01..p12>"
	@echo "  make all     TARGET=<hifi5|cim>   # all patterns, one target"
	@echo "  make all                          # all patterns, both targets"
	@echo "  make clean   [TARGET=.. PATTERN=..]"
	@echo ""
	@echo "Variables (override on command line):"
	@echo "  TOOLCHAIN    = $(TOOLCHAIN)   (cim only: gcc default | xt)"
	@echo "  NNLIB_BASE   = $(NNLIB_BASE)"
	@echo "  XTENSA_CORE  = $(XTENSA_CORE)"
	@echo "  BUILD_DIR    = $(BUILD_DIR)"
	@echo ""
	@echo "Patterns:"
	@echo "  $(ALL_PATTERNS)"
	@echo ""
	@echo "TFLITE_SINGLE_ROUNDING:"
	@echo "  p01 -> SR=1   (single-rounding)"
	@echo "  p02..p12 -> SR=0 (double-rounding)"
	@echo ""

_check_build_dir:
	@mkdir -p $(BUILD_DIR)
