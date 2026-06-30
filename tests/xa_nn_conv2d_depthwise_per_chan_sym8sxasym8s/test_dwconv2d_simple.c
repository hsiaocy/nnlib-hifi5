/*******************************************************************************
 * Simplified Test for xa_nn_conv2d_depthwise_per_chan_sym8sxasym8s
 *
 * Three-way comparison:
 *   1. NNLib   xa_nn_conv2d_depthwise_per_chan_sym8sxasym8s  vs golden
 *   2. CModel  cmodel_conv2d_depthwise_sym8sxasym8s          vs golden
 *   3. NNLib   vs CModel  (should be bit-exact)
 *
 * Read pattern from stdin (redirect *.in_out via shell):
 *   xt-run ./test_dwconv2d_sr0 < patterns/p00_xxx.in_out
 *
 * Makefile usage:
 *   make -f Makefile.simple comp
 *   make -f Makefile.simple run  P=patterns/p00_xxx.in_out
 *   make -f Makefile.simple run-all
 ******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "xa_nnlib_common.h"

/* CModel header — sits next to this file */
#include "conv2d_depthwise_cmodel.h"

/* HiFi5 ISS cycle counter */
#include "xtensa/tie/xt_core.h"
#include "xtensa/xtruntime.h"

/* Maximum file size ~1 MB */
#define MAX_FILE_BYTES  (1024 * 1024)

/* ── Pattern data ─────────────────────────────────────────────────────────── */
typedef struct {
    WORD32 input_height, input_width, input_channels;
    WORD32 kernel_height, kernel_width, channels_multiplier;
    WORD32 dilation_height, dilation_width;
    WORD32 x_stride, y_stride, x_padding, y_padding;
    WORD32 out_height, out_width;
    WORD32 input_zero_bias, out_zero_bias;
    WORD32 inp_data_format, out_data_format;
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
        if (end == p) break;
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
    parse_int_param(buf, "input_height",        &pat->input_height);
    parse_int_param(buf, "input_width",         &pat->input_width);
    parse_int_param(buf, "input_channels",      &pat->input_channels);
    parse_int_param(buf, "kernel_height",       &pat->kernel_height);
    parse_int_param(buf, "kernel_width",        &pat->kernel_width);
    parse_int_param(buf, "channels_multiplier", &pat->channels_multiplier);
    parse_int_param(buf, "dilation_height",     &pat->dilation_height);
    parse_int_param(buf, "dilation_width",      &pat->dilation_width);
    parse_int_param(buf, "x_stride",            &pat->x_stride);
    parse_int_param(buf, "y_stride",            &pat->y_stride);
    parse_int_param(buf, "x_padding",           &pat->x_padding);
    parse_int_param(buf, "y_padding",           &pat->y_padding);
    parse_int_param(buf, "out_height",          &pat->out_height);
    parse_int_param(buf, "out_width",           &pat->out_width);
    parse_int_param(buf, "input_zero_bias",     &pat->input_zero_bias);
    parse_int_param(buf, "out_zero_bias",       &pat->out_zero_bias);
    parse_int_param(buf, "inp_data_format",     &pat->inp_data_format);
    parse_int_param(buf, "out_data_format",     &pat->out_data_format);
    parse_int_param(buf, "out_activation_min",  &pat->out_activation_min);
    parse_int_param(buf, "out_activation_max",  &pat->out_activation_max);

    /* Defaults for optional params */
    if (pat->channels_multiplier == 0) pat->channels_multiplier = 1;
    if (pat->dilation_height == 0)     pat->dilation_height = 1;
    if (pat->dilation_width == 0)      pat->dilation_width  = 1;
    if (pat->out_activation_min == 0 && pat->out_activation_max == 0) {
        pat->out_activation_min = -128;
        pat->out_activation_max =  127;
    }

    /* Derived sizes */
    WORD32 OC = pat->input_channels * pat->channels_multiplier;

    WORD32 max_io  = pat->input_height * pat->input_width * pat->input_channels + 16;
    WORD32 max_ker = pat->kernel_height * pat->kernel_width * OC + 16;
    WORD32 max_out = pat->out_height * pat->out_width * OC + 16;
    WORD32 max_ch  = OC + 16;

    if (max_io  < 16) max_io  = 32768;
    if (max_ker < 16) max_ker = 131072;
    if (max_out < 16) max_out = 131072;
    if (max_ch  < 16) max_ch  = 4096;

    pat->input          = (WORD8  *)malloc(max_io);
    pat->kernel         = (WORD8  *)malloc(max_ker);
    pat->golden_output  = (WORD8  *)malloc(max_out);
    pat->bias           = (WORD32 *)malloc(max_ch * sizeof(WORD32));
    pat->out_multiplier = (WORD32 *)malloc(max_ch * sizeof(WORD32));
    pat->out_shift      = (WORD32 *)malloc(max_ch * sizeof(WORD32));

    if (!pat->input || !pat->kernel || !pat->golden_output ||
        !pat->bias  || !pat->out_multiplier || !pat->out_shift) {
        printf("ERROR: pattern buffer malloc failed\n");
        return 0;
    }

    /* Parse arrays */
    parse_array_i8 (buf, "int8_t input[",         pat->input,          &pat->input_size);
    parse_array_i8 (buf, "int8_t kernel[",        pat->kernel,         &pat->kernel_size);
    parse_array_i32(buf, "int32_t bias[",         pat->bias,           &pat->bias_size);
    parse_array_i32(buf, "int32_t out_multiplier[",pat->out_multiplier, NULL);
    parse_array_i32(buf, "int32_t out_shift[",    pat->out_shift,      NULL);
    parse_array_i8 (buf, "int8_t golden[",        pat->golden_output,  &pat->output_size);

    printf("Pattern loaded from stdin\n");
    printf("  input:   %dx%dx%d  (%d bytes)\n",
           pat->input_height, pat->input_width, pat->input_channels, pat->input_size);
    printf("  kernel:  %dx%dx%d  (%d bytes)  [KH x KW x OC, OC=IC*CM=%d]\n",
           pat->kernel_height, pat->kernel_width, OC, pat->kernel_size, OC);
    printf("  output:  %dx%dx%d  (%d bytes)\n",
           pat->out_height, pat->out_width, OC, pat->output_size);

    return 1;
}

/* ── Compare output vs golden ────────────────────────────────────────────── */
static int compare_output(const char *label, const WORD8 *output,
                           const WORD8 *golden, WORD32 size)
{
    WORD32 n_miss = 0;
    for (WORD32 i = 0; i < size; i++) {
        if (output[i] != golden[i]) {
            if (n_miss < 10)
                printf("  [%s] Mismatch[%d]: got %d  expected %d\n",
                       label, i, (int)output[i], (int)golden[i]);
            n_miss++;
        }
    }
    if (n_miss)
        printf("  [%s] Total mismatches: %d / %d\n", label, n_miss, size);
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
    printf("Depthwise Conv2D Simplified Test (stdin pattern)\n");
    printf("  NNLib API : xa_nn_conv2d_depthwise_per_chan_sym8sxasym8s\n");
    printf("  CModel    : cmodel_conv2d_depthwise_sym8sxasym8s\n");
    printf("==================================================\n\n");

    /* 1. Read stdin */
    WORD32 file_len = 0;
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

    WORD32 OC       = pat.input_channels * pat.channels_multiplier;
    WORD32 out_size  = pat.out_height * pat.out_width * OC;

    /* 3. Allocate NNLib output + scratch */
    WORD8 *out_nnlib  = (WORD8 *)malloc(out_size);
    WORD8 *out_cmodel = (WORD8 *)malloc(out_size);
    if (!out_nnlib || !out_cmodel) {
        printf("ERROR: output malloc failed\n");
        return 1;
    }
    memset(out_nnlib,  0, out_size);
    memset(out_cmodel, 0, out_size);

    WORD32 scratch_size = xa_nn_conv2d_depthwise_getsize(
        pat.input_height, pat.input_width, pat.input_channels,
        pat.kernel_height, pat.kernel_width, pat.channels_multiplier,
        pat.x_stride, pat.y_stride,
        pat.x_padding, pat.y_padding,
        pat.out_height, pat.out_width,
        PREC_ASYM8S,         /* circ_buf_precision for asym8s input */
        pat.inp_data_format);

    printf("\nScratch size: %d bytes\n", scratch_size);
    if (scratch_size <= 0) {
        printf("ERROR: getsize returned %d\n", scratch_size);
        return 1;
    }
    VOID *scratch = (VOID *)malloc(scratch_size);
    if (!scratch) { printf("ERROR: scratch malloc failed\n"); return 1; }

    /* 4. Dump parameters */
    printf("--- Depthwise Conv2D parameters ---\n");
    printf("  input_height       = %d\n", pat.input_height);
    printf("  input_width        = %d\n", pat.input_width);
    printf("  input_channels     = %d\n", pat.input_channels);
    printf("  kernel_height      = %d\n", pat.kernel_height);
    printf("  kernel_width       = %d\n", pat.kernel_width);
    printf("  channels_multiplier= %d\n", pat.channels_multiplier);
    printf("  dilation_h x w     = %d x %d\n", pat.dilation_height, pat.dilation_width);
    printf("  x_stride           = %d\n", pat.x_stride);
    printf("  y_stride           = %d\n", pat.y_stride);
    printf("  x_padding          = %d\n", pat.x_padding);
    printf("  y_padding          = %d\n", pat.y_padding);
    printf("  out_height         = %d\n", pat.out_height);
    printf("  out_width          = %d\n", pat.out_width);
    printf("  input_zero_bias    = %d\n", pat.input_zero_bias);
    printf("  out_zero_bias      = %d\n", pat.out_zero_bias);
    printf("  inp_data_format    = %d\n", pat.inp_data_format);
    printf("  out_data_format    = %d\n", pat.out_data_format);
    printf("  act_min/max        = %d / %d\n", pat.out_activation_min, pat.out_activation_max);
    printf("  out_multiplier[0]  = %d\n", pat.out_multiplier[0]);
    printf("  out_shift[0]       = %d\n", pat.out_shift[0]);
    printf("  OC (IC*CM)         = %d\n", OC);
    printf("  input_size         = %d  (expected %d)\n",
           pat.input_size, pat.input_height * pat.input_width * pat.input_channels);
    printf("  kernel_size        = %d  (expected %d)\n",
           pat.kernel_size, pat.kernel_height * pat.kernel_width * OC);
    printf("  output_size        = %d  (expected %d)\n",
           pat.output_size, out_size);
    printf("  scratch_size       = %d\n", scratch_size);
    printf("-----------------------------------\n\n");

    /* ─────────────────────────────────────────────────────────────────────
     * 5a. Call NNLib depthwise conv2d
     *
     *   API:  xa_nn_conv2d_depthwise_per_chan_sym8sxasym8s(
     *             p_out, p_kernel, p_inp, p_bias,        ← ker/inp SWAPPED vs std
     *             IH, IW, IC, KH, KW, CM,                ← CM not OC
     *             x_stride, y_stride, x_padding, y_padding,
     *             OH, OW,
     *             input_zero_bias,
     *             p_out_multiplier, p_out_shift, out_zero_bias,
     *             inp_data_format, out_data_format,       ← extra: inp_data_format
     *             p_scratch);
     *
     *   NOTE:  Non-_v2 variant.  No dilation args.  NNLib internally
     *          treats dilation=1.  For dilated depthwise, call the
     *          dilated variant instead.
     * ─────────────────────────────────────────────────────────────────── */
    printf("Calling xa_nn_conv2d_depthwise_per_chan_sym8sxasym8s()...\n");

    unsigned int cyc_start, cyc_end;
    cyc_start = XT_RSR_CCOUNT();

    WORD32 ret = xa_nn_conv2d_depthwise_per_chan_sym8sxasym8s(
        out_nnlib,
        pat.kernel,                 /* p_kernel (2nd arg, NOT p_inp!) */
        pat.input,                  /* p_inp    (3rd arg)             */
        pat.bias,
        pat.input_height, pat.input_width, pat.input_channels,
        pat.kernel_height, pat.kernel_width,
        pat.channels_multiplier,
        pat.x_stride, pat.y_stride,
        pat.x_padding, pat.y_padding,
        pat.out_height, pat.out_width,
        pat.input_zero_bias,
        pat.out_multiplier, pat.out_shift,
        pat.out_zero_bias,
        pat.inp_data_format,
        pat.out_data_format,
        scratch);

    cyc_end = XT_RSR_CCOUNT();
    printf("  NNLib return = %d\n", ret);
    printf("  Cycles: %u\n\n", cyc_end - cyc_start);

    free(scratch);

    /* ─────────────────────────────────────────────────────────────────────
     * 5b. Call CModel
     * ─────────────────────────────────────────────────────────────────── */
    printf("Calling cmodel_conv2d_depthwise_sym8sxasym8s()...\n");

    cmodel_conv2d_depthwise_sym8sxasym8s(
        out_cmodel,
        pat.kernel,
        pat.input,
        pat.bias,
        pat.out_multiplier,
        pat.out_shift,
        pat.input_height, pat.input_width, pat.input_channels,
        pat.kernel_height, pat.kernel_width,
        pat.channels_multiplier,
        pat.dilation_height, pat.dilation_width,
        pat.y_stride, pat.x_stride,
        pat.y_padding, pat.x_padding,
        pat.out_height, pat.out_width,
        pat.input_zero_bias,
        pat.out_zero_bias,
        pat.out_activation_min, pat.out_activation_max);

    printf("  CModel done\n\n");

    /* 6. Three-way compare */
    printf("Comparing (%d bytes)...\n", out_size);

    int pass_nnlib  = 1;
    int pass_cmodel = 1;
    int pass_cross  = 1;

    if (ret == 0) {
        pass_nnlib = compare_output("NNLib vs golden", out_nnlib,
                                     pat.golden_output, out_size);
    } else {
        printf("  [NNLib] SKIPPED (returned error %d)\n", ret);
        pass_nnlib = 0;
    }

    pass_cmodel = compare_output("CModel vs golden", out_cmodel,
                                  pat.golden_output, out_size);

    if (ret == 0) {
        pass_cross = compare_output("NNLib vs CModel", out_nnlib,
                                     out_cmodel, out_size);
    }

    free(out_nnlib);
    free(out_cmodel);

    /* 7. Verdict */
    int all_pass = pass_nnlib && pass_cmodel && pass_cross;

    printf("\n==================================================\n");
    printf("  NNLib  vs golden : %s\n", pass_nnlib  ? "PASS" : "FAIL");
    printf("  CModel vs golden : %s\n", pass_cmodel ? "PASS" : "FAIL");
    printf("  NNLib  vs CModel : %s\n", pass_cross  ? "PASS" : "FAIL");
    printf("--------------------------------------------------\n");
    printf("  %s\n", all_pass ? "PASS" : "FAIL");
    printf("==================================================\n");

    return all_pass ? 0 : 1;
}
