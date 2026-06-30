/*******************************************************************************
 * test_activation_cmodel.c
 *
 * gcc test harness for the asym8->asym8 activation cmodel.
 *
 * Usage:
 *   ./test_activation_cmodel < patterns/p01_relu_zero_zp.in_out
 *
 * Or via Makefile:
 *   make run FIXTURE=patterns/p01_relu_zero_zp.in_out
 *   make run-all
 *
 * Reads a single .in_out fixture from stdin, parses the scalar params and
 * input/golden arrays, runs cmodel_activation_clamp_asym8(), compares output
 * against golden, and prints PASS or FAIL plus a mismatch count.
 *
 * NOTE: partial_sum is NOT present in activation .in_out fixtures.
 *       This is an elementwise op with no MAC accumulation.
 *       The harness does not attempt to parse or compare partial_sum.
 *
 * Multiple fixtures can be run sequentially via do_run_all (shell script).
 * The harness returns exit code 0 on PASS, 1 on FAIL.
 ******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "activation_cmodel.h"

/* ── Buffer size ─────────────────────────────────────────────────────────── */
#define MAX_FILE_BYTES  (256 * 1024)   /* 256 KB — enough for N=1024 fixture  */
#define MAX_ELEMENTS    (16 * 1024)    /* 16384 elements max per fixture       */

/* ── Pattern data ────────────────────────────────────────────────────────── */
typedef struct {
    int tflite_single_rounding;
    int num_elements;
    int input_zero_point;
    int output_zero_point;
    int activation_min;
    int activation_max;

    int8_t  *input;
    int8_t  *golden;
    int      input_size;
    int      golden_size;
} pattern_t;

/* ── Read all of stdin into a malloc'd buffer ─────────────────────────────── */
static char *read_stdin(int *out_len)
{
    char *buf = (char *)malloc(MAX_FILE_BYTES);
    if (!buf) {
        fprintf(stderr, "ERROR: malloc(%d) failed\n", MAX_FILE_BYTES);
        return NULL;
    }

    int total = 0;
    int c;
    while ((c = getchar()) != EOF) {
        if (total >= MAX_FILE_BYTES - 1) {
            fprintf(stderr, "ERROR: input exceeds %d bytes\n", MAX_FILE_BYTES);
            free(buf);
            return NULL;
        }
        buf[total++] = (char)c;
    }
    buf[total] = '\0';
    *out_len   = total;
    return buf;
}

/* ── Parser helpers — match the conv2d test_conv2d_simple.c contract ─────── */

static void parse_int_param(const char *hay, const char *name, int *dst)
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
                *dst = (int)strtol(eq + 1, NULL, 10);
                return;
            }
        }
        p++;
    }
}

static void parse_array_i8(const char *hay, const char *tag,
                            int8_t *dst, int *out_size)
{
    const char *p = strstr(hay, tag);
    if (!p) return;
    p = strchr(p, '{');
    if (!p) return;
    p++;

    int idx = 0;
    char *end;
    while (*p && *p != '}') {
        while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r' || *p == ',')
            p++;
        if (*p == '}' || *p == '\0') break;
        dst[idx++] = (int8_t)(int)strtol(p, &end, 10);
        if (end == p) break;
        p = end;
    }
    if (out_size) *out_size = idx;
}

/* ── Load pattern ────────────────────────────────────────────────────────── */
static int load_pattern(const char *buf, pattern_t *pat)
{
    memset(pat, 0, sizeof(pattern_t));

    /* Scalar params */
    parse_int_param(buf, "tflite_single_rounding", &pat->tflite_single_rounding);
    parse_int_param(buf, "num_elements",            &pat->num_elements);
    parse_int_param(buf, "input_zero_point",        &pat->input_zero_point);
    parse_int_param(buf, "output_zero_point",       &pat->output_zero_point);
    parse_int_param(buf, "activation_min",          &pat->activation_min);
    parse_int_param(buf, "activation_max",          &pat->activation_max);

    if (pat->num_elements <= 0 || pat->num_elements > MAX_ELEMENTS) {
        fprintf(stderr, "ERROR: num_elements=%d out of range [1,%d]\n",
                pat->num_elements, MAX_ELEMENTS);
        return 0;
    }

    pat->input  = (int8_t *)malloc(pat->num_elements);
    pat->golden = (int8_t *)malloc(pat->num_elements);
    if (!pat->input || !pat->golden) {
        fprintf(stderr, "ERROR: pattern buffer malloc failed\n");
        return 0;
    }

    parse_array_i8(buf, "int8_t input[",  pat->input,  &pat->input_size);
    parse_array_i8(buf, "int8_t golden[", pat->golden, &pat->golden_size);

    printf("Pattern loaded:\n");
    printf("  num_elements       = %d\n", pat->num_elements);
    printf("  input_zero_point   = %d\n", pat->input_zero_point);
    printf("  output_zero_point  = %d\n", pat->output_zero_point);
    printf("  activation_min     = %d\n", pat->activation_min);
    printf("  activation_max     = %d\n", pat->activation_max);
    printf("  tflite_single_rnd  = %d  (unused — clamp needs no requantize)\n",
           pat->tflite_single_rounding);
    printf("  input  parsed: %d elements\n", pat->input_size);
    printf("  golden parsed: %d elements\n", pat->golden_size);

    if (pat->input_size  != pat->num_elements) {
        fprintf(stderr, "WARNING: input_size=%d != num_elements=%d\n",
                pat->input_size, pat->num_elements);
    }
    if (pat->golden_size != pat->num_elements) {
        fprintf(stderr, "WARNING: golden_size=%d != num_elements=%d\n",
                pat->golden_size, pat->num_elements);
    }

    return 1;
}

/* ── Compare output vs golden ────────────────────────────────────────────── */
static int compare_output(const int8_t *output, const int8_t *golden, int size)
{
    int n_miss = 0;
    for (int i = 0; i < size; i++) {
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
    printf("==================================================\n");
    printf("Activation CModel Test (asym8->asym8, stdin fixture)\n");
    printf("==================================================\n\n");

    /* 1. Read stdin */
    int file_len = 0;
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

    /* 3. Allocate output buffer */
    int8_t *output = (int8_t *)malloc(pat.num_elements);
    if (!output) {
        fprintf(stderr, "ERROR: output malloc failed\n");
        return 1;
    }
    memset(output, 0, pat.num_elements);

    /* 4. Run cmodel */
    printf("\nRunning cmodel_activation_clamp_asym8(%d elements, min=%d, max=%d)...\n",
           pat.num_elements, pat.activation_min, pat.activation_max);

    cmodel_activation_clamp_asym8(
        output,
        pat.input,
        pat.num_elements,
        pat.activation_min,
        pat.activation_max);

    /* 5. Compare against golden */
    printf("\nComparing output with golden (%d bytes)...\n", pat.num_elements);
    int pass = compare_output(output, pat.golden, pat.num_elements);
    free(output);
    free(pat.input);
    free(pat.golden);

    /* 6. Print result */
    printf("\n==================================================\n");
    printf("Output  : %s\n", pass ? "PASS" : "FAIL");
    printf("%s\n", pass ? "PASS" : "FAIL");
    printf("==================================================\n");

    return pass ? 0 : 1;
}
