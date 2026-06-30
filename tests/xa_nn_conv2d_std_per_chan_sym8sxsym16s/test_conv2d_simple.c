/*******************************************************************************
 * Simplified Test for xa_nn_conv2d_std_per_chan_sym8sxasym8s
 *
 * Read pattern from stdin (redirect *.in_out via shell):
 *   xt-run ./test_conv2d_simple < pattern.in_out
 *
 * Makefile usage:
 *   make run  P=patterns/p00_....in_out
 *   make run-all
 ******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "xa_nnlib_common.h"
#include "xa_nn_conv2d_std_state.h"

#include "xtensa/tie/xt_core.h"
#include "xtensa/xtruntime.h"

/* Maximum file size ~1 MB (covers all patterns including p06 many_channels) */
#define MAX_FILE_BYTES  (1024 * 1024)


/* ── Pattern data ─────────────────────────────────────────────────────────── */
typedef struct {
    WORD32 input_height, input_width, input_channels;
    WORD32 kernel_height, kernel_width, out_channels;
    WORD32 x_stride, y_stride, x_padding, y_padding;
    WORD32 out_height, out_width;
    WORD32 input_zero_bias, out_zero_bias, out_data_format;
    WORD32 out_activation_min, out_activation_max;

    WORD8  *input,  *kernel,  *golden_output;
    WORD32 *bias,   *out_multiplier, *out_shift;

    WORD32 input_size, kernel_size, bias_size, output_size;
} pattern_t;

/* ── Read all of stdin into a malloc'd buffer ─────────────────────────────── */
static char *read_stdin(WORD32 *out_len)
{
    char *buf = (char *)malloc(MAX_FILE_BYTES);
    if (!buf) {
        printf("ERROR: malloc(%d) failed\n", MAX_FILE_BYTES);
        return NULL;
    }

    WORD32 total = 0;
    int c;
    while ((c = getchar()) != EOF) {
        if (total >= MAX_FILE_BYTES - 1) {
            printf("ERROR: input exceeds %d bytes\n", MAX_FILE_BYTES);
            free(buf);
            return NULL;
        }
        buf[total++] = (char)c;
    }
    buf[total] = '\0';
    *out_len   = total;
    return buf;
}

/* ── Parser helpers ───────────────────────────────────────────────────────── */
static void parse_int_param(const char *hay, const char *name, WORD32 *dst)
{
    const char *p = hay;
    size_t name_len = strlen(name);
    while ((p = strstr(p, name)) != NULL) {
        const char *line_start = p;
        while (line_start > hay && *(line_start - 1) != '\n')
            line_start--;
        const char *q = line_start;
        while (*q == ' ' || *q == '\t') q++;
        if (strncmp(q, "int ", 4) == 0) {
            const char *eq = strchr(p + name_len, '=');
            if (eq) {
                *dst = (WORD32)strtol(eq + 1, NULL, 10);
                return;
            }
        }
        p++;
    }

}

static void parse_array_i8(const char *hay, const char *tag,
                            WORD8 *dst, WORD32 *out_size)
{
    const char *p = strstr(hay, tag);
    if (!p) return;
    p = strchr(p, '{');
    if (!p) return;
    p++;

    WORD32 idx = 0;
    char   *end;
    while (*p && *p != '}') {
        while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r' || *p == ',')
            p++;
        if (*p == '}' || *p == '\0') break;
        dst[idx++] = (WORD8)(WORD32)strtol(p, &end, 10);
        if (end == p) break;   /* strtol made no progress → stop */
        p = end;
    }
    if (out_size) *out_size = idx;
}

static void parse_array_i32(const char *hay, const char *tag,
                             WORD32 *dst, WORD32 *out_size)
{
    const char *p = strstr(hay, tag);
    if (!p) return;
    p = strchr(p, '{');
    if (!p) return;
    p++;

    WORD32 idx = 0;
    char   *end;
    while (*p && *p != '}') {
        while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r' || *p == ',')
            p++;
        if (*p == '}' || *p == '\0') break;
        dst[idx++] = (WORD32)strtol(p, &end, 10);
        if (end == p) break;
        p = end;
    }
    if (out_size) *out_size = idx;
}

/* ── Load pattern from text buffer ───────────────────────────────────────── */
static int load_pattern(const char *buf, pattern_t *pat)
{
    memset(pat, 0, sizeof(pattern_t));

    /* Scalar params */
    parse_int_param(buf, "input_height",     &pat->input_height);
    parse_int_param(buf, "input_width",      &pat->input_width);
    parse_int_param(buf, "input_channels",   &pat->input_channels);
    parse_int_param(buf, "kernel_height",    &pat->kernel_height);
    parse_int_param(buf, "kernel_width",     &pat->kernel_width);
    parse_int_param(buf, "out_channels",     &pat->out_channels);
    parse_int_param(buf, "x_stride",         &pat->x_stride);
    parse_int_param(buf, "y_stride",         &pat->y_stride);
    parse_int_param(buf, "x_padding",        &pat->x_padding);
    parse_int_param(buf, "y_padding",        &pat->y_padding);
    parse_int_param(buf, "out_height",       &pat->out_height);
    parse_int_param(buf, "out_width",        &pat->out_width);
    parse_int_param(buf, "input_zero_bias",  &pat->input_zero_bias);
    parse_int_param(buf, "out_zero_bias",    &pat->out_zero_bias);
    parse_int_param(buf, "out_data_format",  &pat->out_data_format);
    parse_int_param(buf, "out_activation_min", &pat->out_activation_min);
    parse_int_param(buf, "out_activation_max", &pat->out_activation_max);

    /* Allocate arrays */
    WORD32 max_io  = pat->input_height  * pat->input_width  * pat->input_channels  + 16;
    WORD32 max_ker = pat->kernel_height * pat->kernel_width * pat->input_channels * pat->out_channels + 16;
    WORD32 max_out = pat->out_height    * pat->out_width    * pat->out_channels   + 16;
    WORD32 max_ch  = pat->out_channels + 16;

    /* Guard: if params not parsed, fall back to safe maximums */
    if (max_io  < 16) max_io  = 32768;
    if (max_ker < 16) max_ker = 131072;
    if (max_out < 16) max_out = 131072;
    if (max_ch  < 16) max_ch  = 4096;

    pat->input         = (WORD8  *)malloc(max_io);
    pat->kernel        = (WORD8  *)malloc(max_ker);
    pat->golden_output = (WORD8  *)malloc(max_out);
    pat->bias          = (WORD32 *)malloc(max_ch * sizeof(WORD32));
    pat->out_multiplier= (WORD32 *)malloc(max_ch * sizeof(WORD32));
    pat->out_shift     = (WORD32 *)malloc(max_ch * sizeof(WORD32));

    if (!pat->input || !pat->kernel || !pat->golden_output ||
        !pat->bias  || !pat->out_multiplier || !pat->out_shift) {
        printf("ERROR: pattern buffer malloc failed\n");
        return 0;
    }

    /* Parse arrays */
    parse_array_i8 (buf, "int8_t input[",        pat->input,          &pat->input_size);
    parse_array_i8 (buf, "int8_t kernel[",        pat->kernel,         &pat->kernel_size);
    parse_array_i32(buf, "int32_t bias[",         pat->bias,           &pat->bias_size);
    parse_array_i32(buf, "int32_t out_multiplier[",pat->out_multiplier, NULL);
    parse_array_i32(buf, "int32_t out_shift[",    pat->out_shift,      NULL);
    parse_array_i8 (buf, "int8_t golden[",        pat->golden_output,  &pat->output_size);

    printf("Pattern loaded from stdin\n");
    printf("  input:  %dx%dx%d  (%d bytes)\n",
           pat->input_height,  pat->input_width,  pat->input_channels, pat->input_size);
    printf("  kernel: %dx%dx%d->%d  (%d bytes)\n",
           pat->kernel_height, pat->kernel_width, pat->input_channels, pat->out_channels, pat->kernel_size);
    printf("  output: %dx%dx%d  (%d bytes)\n",
           pat->out_height, pat->out_width, pat->out_channels, pat->output_size);

    return 1;
}

/* ── Compare output vs golden ────────────────────────────────────────────── */
static int compare_output(const WORD8 *output, const WORD8 *golden, WORD32 size)
{
    WORD32 n_miss = 0;
    for (WORD32 i = 0; i < size; i++) {
        if (output[i] != golden[i]) {
            if (n_miss < 10)
                printf("  Mismatch[%d]: got %d  expected %d\n",
                       i, (int)output[i], (int)golden[i]);
            n_miss++;
        }
    }
    if (n_miss)
        printf("  Total mismatches: %d / %d\n", n_miss, size);
    return (n_miss == 0);
}

/* ── main ────────────────────────────────────────────────────────────────── */
int main(void)
{
    #ifdef AE_MULAZB8Q8X8
      printf("RI6 path (AE_MULAZB8Q8X8)\n");
    #else
      printf("RI5 path (AE_MULA8Q8X16)\n");
    #endif

    printf("==================================================\n");
    printf("Conv2D Simplified Test (stdin-based pattern)\n");
    printf("==================================================\n\n");

    /* 1. Read stdin */
    WORD32 file_len = 0;
    char *buf = read_stdin(&file_len);
    if (!buf) return 1;
    printf("Read %d bytes from stdin\n", file_len);

    /* 2. Parse pattern */
    static pattern_t pat;   /* static → BSS, avoids large stack frame */
    if (!load_pattern(buf, &pat)) {
        free(buf);
        return 1;
    }
    free(buf);

    /* 3. Allocate output + scratch */
    WORD32 out_size = pat.out_height * pat.out_width * pat.out_channels;
    WORD8 *output = (WORD8 *)malloc(out_size);
    if (!output) { printf("ERROR: output malloc failed\n"); return 1; }
    memset(output, 0, out_size);

    WORD32 scratch_size = xa_nn_conv2d_std_getsize(
        pat.input_height, pat.input_width, pat.input_channels,
        pat.kernel_height, pat.kernel_width, pat.input_channels,
        pat.y_stride, pat.y_padding, pat.x_stride, pat.x_padding,
        pat.out_height, pat.out_width, pat.out_channels,
        PREC_ASYM8S, PREC_SYM8S, 1, 1, pat.out_data_format);

    printf("\nScratch size: %d bytes\n", scratch_size);
    VOID *scratch = (VOID *)malloc(scratch_size);
    if (!scratch) { printf("ERROR: scratch malloc failed\n"); free(output); return 1; }

    /* 4. Dump all parameters for debug */
    printf("--- Conv2D parameters ---\n");
    printf("  input_height    = %d\n", pat.input_height);
    printf("  input_width     = %d\n", pat.input_width);
    printf("  input_channels  = %d\n", pat.input_channels);
    printf("  kernel_height   = %d\n", pat.kernel_height);
    printf("  kernel_width    = %d\n", pat.kernel_width);
    printf("  out_channels    = %d\n", pat.out_channels);
    printf("  x_stride        = %d\n", pat.x_stride);
    printf("  y_stride        = %d\n", pat.y_stride);
    printf("  x_padding       = %d\n", pat.x_padding);
    printf("  y_padding       = %d\n", pat.y_padding);
    printf("  out_height      = %d\n", pat.out_height);
    printf("  out_width       = %d\n", pat.out_width);
    printf("  input_zero_bias = %d\n", pat.input_zero_bias);
    printf("  out_zero_bias   = %d\n", pat.out_zero_bias);
    printf("  out_data_format = %d\n", pat.out_data_format);
    printf("  act_min/max     = %d / %d\n", pat.out_activation_min, pat.out_activation_max);
    printf("  out_multiplier[0] = %d\n", pat.out_multiplier[0]);
    printf("  out_shift[0]      = %d\n", pat.out_shift[0]);
    printf("  input_size      = %d  (expected %d)\n",
           pat.input_size,
           pat.input_height * pat.input_width * pat.input_channels);
    printf("  kernel_size     = %d  (expected %d)\n",
           pat.kernel_size,
           pat.kernel_height * pat.kernel_width * pat.input_channels * pat.out_channels);
    printf("  output_size     = %d  (expected %d)\n",
           pat.output_size,
           pat.out_height * pat.out_width * pat.out_channels);
    printf("  scratch_size    = %d\n", scratch_size);
    printf("-------------------------\n\n");

    /* 4. Call HiFi5 Conv2D */
    printf("Calling xa_nn_conv2d_std_per_chan_sym8sxasym8s()...\n\n");

    // 20260421 added by Allenhsiao
    // HiFi5 ISS cycle counter
    unsigned int cyc_start, cyc_end;
    
    // read cycle counter xt-run
    cyc_start = XT_RSR_CCOUNT();

    WORD32 ret = xa_nn_conv2d_std_per_chan_sym8sxasym8s(
        output,
        pat.input, pat.kernel, pat.bias,
        pat.input_height, pat.input_width, pat.input_channels,
        pat.kernel_height, pat.kernel_width, pat.out_channels,
        pat.x_stride, pat.y_stride,
        pat.x_padding, pat.y_padding,
        pat.out_height, pat.out_width,
        pat.input_zero_bias,
        pat.out_multiplier, pat.out_shift,
        pat.out_zero_bias,
        pat.out_data_format, scratch);

    cyc_end = XT_RSR_CCOUNT();

    printf("Cycles: %u\n", cyc_end - cyc_start);

    free(scratch);

    if (ret != 0) {
        printf("ERROR: conv2d returned %d\n", ret);
        free(output);
        return 1;
    }

    /* 5. Compare */
    printf("Comparing with golden (%d bytes)...\n", out_size);
    int pass = compare_output(output, pat.golden_output, out_size);
    free(output);

    printf("\n==================================================\n");
    printf("%s\n", pass ? "PASS" : "FAIL");
    printf("==================================================\n");

    return pass ? 0 : 1;
}
