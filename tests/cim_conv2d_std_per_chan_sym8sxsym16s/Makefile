CC      = gcc
CFLAGS  = -Wall -Wextra -O2 -g -std=c99
LDFLAGS = -lm

TARGET = test_conv2d_simple
SRCS   = test_conv2d_simple.c conv2d_std_cmodel.c
OBJS   = $(SRCS:.c=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $(OBJS) $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

run: $(TARGET)
	@if [ -z "$(P)" ]; then \
		echo "Usage: make run P=patterns/p00_xxx.in_out"; \
		exit 1; \
	else \
		./$(TARGET) < $(P); \
	fi

run-all: $(TARGET)
	@for f in patterns/*.in_out; do \
		echo "Running $$f"; \
		./$(TARGET) < $$f || exit 1; \
		echo ""; \
	done

clean:
	rm -f $(OBJS) $(TARGET) SW_KW5_partialsum_data.txt SW_KWS_partialsum_data.txt

.PHONY: all run run-all clean
