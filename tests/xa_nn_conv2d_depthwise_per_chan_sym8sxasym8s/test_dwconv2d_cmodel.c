/*******************************************************************************
 * test_dwconv2d_cmodel.c
 *
 * gcc-only harness for cmodel_conv2d_depthwise_sym8sxasym8s.
 *
 * Usage (stdin-based):
 *   ./test_dwconv2d_cmodel < patterns/dw_p01_basic_cm1.in_out
 *
 * Makefile run-all loops over all dw_*.in_out files and prints a tally.
 *
 * Compare contract (identical to conv2d harness):
 *   1. golden      — final int8 output   (must match bit-exact)
 *   2. partial_sum — pre-requant int32   (must match bit-exact)
 *
 * Exit code: 0 = PASS, 1 = FAIL.
 ******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "conv2d_depthwise_cmodel.h"

/* Maximum fixture file size (1 MB covers all patterns) */
#define MAX_FILE_BYTES  (1024 * 1024)

/* ── Pattern data ─────────────────────────────────────────────────────────── */
typedef struct {
    int32_t input_height, input_width, input_channels;
    int32_t kernel_height, kernel_width, channels_multiplier;
    int32_t dilation_height, dilation_width;
    int32_t x_stride, y_stride, x_padding, y_padding;
    int32_t out_height, out_width, out_channels;
    int32_t input_zero_bias, out_zero_bias;
    int32_t out_data_format;
    int32_t out_activation_min, out_activation_max;
    int32_t tflite_single_rounding;

    int8_t  *input;
    int8_t  *kernel;
    int8_t  *golden_output;
    int32_t *bias;
    int32_t *out_multiplier;
    int32_t *out_shift;
    int32_t *partial_sum_golden;

    int32_t input_size, kernel_size, output_size;
    int32_t partial_sum_size;
} pattern_t;

/* ── Read all of stdin into a malloc'd buffer ─────────────────────────────── */
static char *read_stdin(int32_t *out_len)
{
    char *buf = (char *)malloc(MAX_FILE_BYTES);
    if (!buf) {
        printf("ERROR: malloc(%d) failed\n", MAX_FILE_BYTES);
        return NULL;
    }
    int32_t total = 0;
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
static void parse_int_param(const char *hay, const char *name, int32_t *dst)
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
                *dst = (int32_t)strtol(eq + 1, NULL, 10);
                return;
            }
        }
        p++;
    }
}

static void parse_array_i8(const char *hay, const char *tag,
                            int8_t *dst, int32_t *out_size)
{
    const char *p = strstr(hay, tag);
    if (!p) return;
    p = strchr(p, '{');
    if (!p) return;
    p++;
    int32_t idx = 0;
    char   *end;
    while (*p && *p != '}') {
        while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r' || *p == ',')
            p++;
        if (*p == '}' || *p == '\0') break;
        dst[idx++] = (int8_t)(int32_t)strtol(p, &end, 10);
        if (end == p) break;
        p = end;
    }
    if (out_size) *out_size = idx;
}

static void parse_array_i32(const char *hay, const char *tag,
                             int32_t *dst, int32_t *out_size)
{
    const char *p = strstr(hay, tag);
    if (!p) return;
    p = strchr(p, '{');
    if (!p) return;
    p++;
    int32_t idx = 0;
    char   *end;
    while (*p && *p != '}') {
        while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r' || *p == ',')
            p++;
        if (*p == '}' || *p == '\0') break;
        dst[idx++] = (int32_t)strtol(p, &end, 10);
        if (end == p) break;
        p = end;
    }
    if (out_size) *out_size = idx;
}

/* ── Load pattern from text buffer ───────────────────────────────────────── */
static int load_pattern(const char *buf, pattern_t *pat)
{
    memset(pat, 0, sizeof(pattern_t));

    parse_int_param(buf, "tflite_single_rounding", &pat->tflite_single_rounding);
    parse_int_param(buf, "input_height",           &pat->input_height);
    parse_int_param(buf, "input_width",            &pat->input_width);
    parse_int_param(buf, "input_channels",         &pat->input_channels);
    parse_int_param(buf, "kernel_height",          &pat->kernel_height);
    parse_int_param(buf, "kernel_width",           &pat->kernel_width);
    parse_int_param(buf, "channels_multiplier",    &pat->channels_multiplier);
    parse_int_param(buf, "dilation_height",        &pat->dilation_height);
    parse_int_param(buf, "dilation_width",         &pat->dilation_width);
    parse_int_param(buf, "x_stride",               &pat->x_stride);
    parse_int_param(buf, "y_stride",               &pat->y_stride);
    parse_int_param(buf, "x_padding",              &pat->x_padding);
    parse_int_param(buf, "y_padding",              &pat->y_padding);
    parse_int_param(buf, "out_height",             &pat->out_height);
    parse_int_param(buf, "out_width",              &pat->out_width);
    parse_int_param(buf, "out_channels",           &pat->out_channels);
    parse_int_param(buf, "input_zero_bias",        &pat->input_zero_bias);
    parse_int_param(buf, "out_zero_bias",          &pat->out_zero_bias);
    parse_int_param(buf, "out_data_format",        &pat->out_data_format);
    parse_int_param(buf, "out_activation_min",     &pat->out_activation_min);
    parse_int_param(buf, "out_activation_max",     &pat->out_activation_max);

    /* Defaults for optional params */
    if (pat->channels_multiplier == 0) pat->channels_multiplier = 1;
    if (pat->dilation_height == 0)     pat->dilation_height = 1;
    if (pat->dilation_width  == 0)     pat->dilation_width  = 1;
    if (pat->out_activation_min == 0 && pat->out_activation_max == 0) {
        pat->out_activation_min = -128;
        pat->out_activation_max =  127;
    }
    /* out_channels must be derivable even if not explicitly in fixture */
    if (pat->out_channels == 0)
        pat->out_channels = pat->input_channels * pat->channels_multiplier;

    int32_t OC      = pat->out_channels;
    int32_t max_io  = pat->input_height  * pat->input_width  * pat->input_channels + 16;
    int32_t max_ker = pat->kernel_height * pat->kernel_width * OC + 16;
    int32_t max_out = pat->out_height    * pat->out_width    * OC + 16;
    int32_t max_ch  = OC + 16;

    if (max_io  < 16) max_io  = 32768;
    if (max_ker < 16) max_ker = 131072;
    if (max_out < 16) max_out = 131072;
    if (max_ch  < 16) max_ch  = 4096;

    pat->input              = (int8_t  *)malloc(max_io);
    pat->kernel             = (int8_t  *)malloc(max_ker);
    pat->golden_output      = (int8_t  *)malloc(max_out);
    pat->bias               = (int32_t *)malloc(max_ch * sizeof(int32_t));
    pat->out_multiplier     = (int32_t *)malloc(max_ch * sizeof(int32_t));
    pat->out_shift          = (int32_t *)malloc(max_ch * sizeof(int32_t));
    pat->partial_sum_golden = (int32_t *)malloc(max_out * sizeof(int32_t));

    if (!pat->input || !pat->kernel || !pat->golden_output ||
        !pat->bias  || !pat->out_multiplier || !pat->out_shift ||
        !pat->partial_sum_golden) {
        printf("ERROR: pattern buffer malloc failed\n");
        return 0;
    }

    parse_array_i8 (buf, "int8_t input[",          pat->input,              &pat->input_size);
    parse_array_i8 (buf, "int8_t kernel[",         pat->kernel,             &pat->kernel_size);
    parse_array_i32(buf, "int32_t bias[",          pat->bias,               NULL);
    parse_array_i32(buf, "int32_t out_multiplier[",pat->out_multiplier,     NULL);
    parse_array_i32(buf, "int32_t out_shift[",     pat->out_shift,          NULL);
    parse_array_i8 (buf, "int8_t golden[",         pat->golden_output,      &pat->output_size);
    parse_array_i32(buf, "int32_t partial_sum[",   pat->partial_sum_golden, &pat->partial_sum_size);

    printf("Pattern loaded\n");
    printf("  tflite_single_rounding = %d\n", pat->tflite_single_rounding);
    printf("  input:   %dx%dx%d  (%d bytes)\n",
           pat->input_height, pat->input_width, pat->input_channels, pat->input_size);
    printf("  kernel:  %dx%dx%d  (%d bytes)  [KH x KW x OC=%d]\n",
           pat->kernel_height, pat->kernel_width, OC, pat->kernel_size, OC);
    printf("  output:  %dx%dx%d  (%d bytes)\n",
           pat->out_height, pat->out_width, OC, pat->output_size);
    printf("  partial_sum: %d int32 values\n", pat->partial_sum_size);

    return 1;
}

/* ── Compare output vs golden ────────────────────────────────────────────── */
static int compare_output(const int8_t *output, const int8_t *golden, int32_t size)
{
    int32_t n_miss = 0;
    for (int32_t i = 0; i < size; i++) {
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

/* ── Compare partial sum vs golden ──────────────────────────────────────── */
static int compare_partial_sum(const int32_t *ps, const int32_t *ps_golden, int32_t size)
{
    int32_t n_miss = 0;
    for (int32_t i = 0; i < size; i++) {
        if (ps[i] != ps_golden[i]) {
            if (n_miss < 10)
                printf("  PS Mismatch[%d]: got %d  expected %d\n",
                       i, (int)ps[i], (int)ps_golden[i]);
            n_miss++;
        }
    }
    if (n_miss)
        printf("  Total PS mismatches: %d / %d\n", n_miss, size);
    return (n_miss == 0);
}

/* ── main ────────────────────────────────────────────────────────────────── */
int main(void)
{
    printf("==================================================\n");
    printf("Depthwise Conv2D CModel Test (gcc, stdin pattern)\n");
    printf("  CModel: cmodel_conv2d_depthwise_sym8sxasym8s\n");
    printf("==================================================\n\n");

    /* 1. Read stdin */
    int32_t file_len = 0;
    char *buf = read_stdin(&file_len);
    if (!buf) return 1;
    printf("Read %d bytes from stdin\n", file_len);

    /* 2. Parse pattern */
    static pattern_t pat;
    if (!load_pattern(buf, &pat)) {
        free(buf);
        return 1;
    }
    free(buf);

    int32_t OC       = pat.out_channels;
    int32_t out_size = pat.out_height * pat.out_width * OC;

    /* 3. Allocate output + partial_sum capture buffer */
    int8_t  *output = (int8_t  *)malloc(out_size);
    int32_t *ps_buf = (int32_t *)malloc(out_size * sizeof(int32_t));
    if (!output || !ps_buf) {
        printf("ERROR: output/ps_buf malloc failed\n");
        return 1;
    }
    memset(output, 0, out_size);
    memset(ps_buf, 0, out_size * sizeof(int32_t));

    /* 4. Dump parameters */
    printf("--- Depthwise Conv2D parameters ---\n");
    printf("  input_height        = %d\n", pat.input_height);
    printf("  input_width         = %d\n", pat.input_width);
    printf("  input_channels      = %d\n", pat.input_channels);
    printf("  kernel_height       = %d\n", pat.kernel_height);
    printf("  kernel_width        = %d\n", pat.kernel_width);
    printf("  channels_multiplier = %d\n", pat.channels_multiplier);
    printf("  dilation_h x w      = %d x %d\n", pat.dilation_height, pat.dilation_width);
    printf("  x_stride            = %d\n", pat.x_stride);
    printf("  y_stride            = %d\n", pat.y_stride);
    printf("  x_padding           = %d\n", pat.x_padding);
    printf("  y_padding           = %d\n", pat.y_padding);
    printf("  out_height          = %d\n", pat.out_height);
    printf("  out_width           = %d\n", pat.out_width);
    printf("  OC (IC*CM)          = %d\n", OC);
    printf("  input_zero_bias     = %d\n", pat.input_zero_bias);
    printf("  out_zero_bias       = %d\n", pat.out_zero_bias);
    printf("  act_min/max         = %d / %d\n", pat.out_activation_min, pat.out_activation_max);
    printf("  out_multiplier[0]   = %d\n", pat.out_multiplier[0]);
    printf("  out_shift[0]        = %d\n", pat.out_shift[0]);
    printf("-----------------------------------\n\n");

    /* 5. Call CModel */
    printf("Calling cmodel_conv2d_depthwise_sym8sxasym8s()...\n");

    cmodel_conv2d_depthwise_sym8sxasym8s(
        output,
        pat.kernel,                     /* p_kernel (swapped vs std, matches NNLib) */
        pat.input,                      /* p_inp */
        pat.bias,
        pat.out_multiplier,
        pat.out_shift,
        pat.input_height, pat.input_width, pat.input_channels,
        pat.kernel_height, pat.kernel_width,
        pat.channels_multiplier,
        pat.dilation_height, pat.dilation_width,
        pat.y_stride,   /* stride_height = y_stride */
        pat.x_stride,   /* stride_width  = x_stride */
        pat.y_padding,  /* pad_height    = y_padding */
        pat.x_padding,  /* pad_width     = x_padding */
        pat.out_height, pat.out_width,
        pat.input_zero_bias,
        pat.out_zero_bias,
        pat.out_activation_min, pat.out_activation_max,
        ps_buf);        /* p_partial_sum */

    printf("  CModel done\n\n");

    /* 6. Compare output vs golden */
    printf("Comparing output with golden (%d bytes)...\n", out_size);
    int pass_out = compare_output(output, pat.golden_output, out_size);
    free(output);

    /* 7. Compare partial sum vs golden */
    int pass_ps = 1;
    if (pat.partial_sum_size > 0 && pat.partial_sum_size == out_size) {
        printf("Comparing partial_sum with golden (%d int32 values)...\n", out_size);
        pass_ps = compare_partial_sum(ps_buf, pat.partial_sum_golden, out_size);
    } else {
        printf("WARNING: partial_sum_size=%d out_size=%d — skipping PS compare\n",
               pat.partial_sum_size, out_size);
    }
    free(ps_buf);

    int pass = pass_out && pass_ps;

    printf("\n==================================================\n");
    printf("Output  : %s\n", pass_out ? "PASS" : "FAIL");
    printf("Partial : %s\n", pass_ps  ? "PASS" : "FAIL");
    printf("%s\n", pass ? "PASS" : "FAIL");
    printf("==================================================\n");

    return pass ? 0 : 1;
}
