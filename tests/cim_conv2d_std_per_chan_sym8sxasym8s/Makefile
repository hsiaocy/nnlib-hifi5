# Makefile for Conv2D C Model Testing
# 
# This Makefile allows testing the C Model implementation instead of HiFi5 NNLib
#
# Usage:
#   make              - Build with C Model
#   make clean        - Clean build artifacts
#   make run          - Build and run test
#   make run-hifi5    - Build and run with HiFi5 NNLib (requires NNLib)

CC = gcc
CFLAGS = -Wall -Wextra -O2 -g -std=c99
LDFLAGS = -lm

# Source files
CMODEL_SRC = conv2d_std_cmodel.c
WRAPPER_SRC = conv2d_cmodel_wrapper.c

# Object files
CMODEL_OBJ = $(CMODEL_SRC:.c=.o)
WRAPPER_OBJ = $(WRAPPER_SRC:.c=.o)

# Output binary
TARGET = test_conv2d_cmodel

# Test data file
TEST_DATA = xa_nn_conv2d_std_v2_per_chan_sym8sxasym8s.in_out

# Default target: build with C Model
all: $(TARGET)

# Link all objects
#$(TARGET): $(TEST_OBJ) $(CMODEL_OBJ) $(WRAPPER_OBJ)
$(TARGET): $(CMODEL_OBJ) $(WRAPPER_OBJ)
	@echo "Linking $@..."
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "Build complete: $@"

# Compile test file
$(TEST_OBJ): $(TEST_SRC)
	@echo "Compiling $<..."
	$(CC) $(CFLAGS) -DUSE_CMODEL -c $< -o $@

# Compile C Model
$(CMODEL_OBJ): $(CMODEL_SRC)
	@echo "Compiling $<..."
	$(CC) $(CFLAGS) -c $< -o $@

# Compile wrapper
$(WRAPPER_OBJ): $(WRAPPER_SRC)
	@echo "Compiling $<..."
	$(CC) $(CFLAGS) -c $< -o $@

# Run test
run: $(TARGET)
	@echo "Running test..."
	@if [ -f "$(TEST_DATA)" ]; then \
		./$(TARGET); \
	else \
		echo "ERROR: Test data file not found: $(TEST_DATA)"; \
		echo "Please ensure $(TEST_DATA) exists in the current directory."; \
		exit 1; \
	fi

# Clean build artifacts
clean:
	@echo "Cleaning..."
	rm -f $(TARGET) $(TEST_OBJ) $(CMODEL_OBJ) $(WRAPPER_OBJ)
	rm -f SW_KWS_partialsum_data.txt
	@echo "Clean complete"

# Help
help:
	@echo "Conv2D C Model Test Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  make          - Build test with C Model"
	@echo "  make run      - Build and run test"
	@echo "  make clean    - Remove build artifacts"
	@echo "  make help     - Show this help message"
	@echo ""
	@echo "Files:"
	@echo "  Test:    $(TEST_SRC)"
	@echo "  C Model: $(CMODEL_SRC)"
	@echo "  Wrapper: $(WRAPPER_SRC)"
	@echo "  Output:  $(TARGET)"

.PHONY: all run clean help
