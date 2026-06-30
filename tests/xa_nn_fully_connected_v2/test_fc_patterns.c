/*
 * test_fc_patterns.c
 *
 * gcc test harness for cmodel_fc_hifi5 (sym8sxasym8s_asym8s variant).
 * Reads .in_out fixture files from patterns/, calls the cmodel FC function,
 * compares golden output and partial_sum, prints per-pattern PASS/FAIL,
 * and emits a final tally line.
 *
 * Usage:
 *   ./test_fc_patterns patterns/p01_baseline.in_out [patterns/p02_...in_out ...]
 *   (with no arguments: runs all *.in_out in ./patterns/)
 *
 * Build:
 *   gcc -std=c11 -Wall -O2 -o test_fc_patterns \
 *       test_fc_patterns.c cmodel_fc_hifi5.c -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <dirent.h>

#include "cmodel_fc_hifi5.h"

/* ------------------------------------------------------------------ */
/* Limits                                                               */
/* ------------------------------------------------------------------ */
#define MAX_FILE_BYTES   (1 << 20)   /* 1 MB per fixture             */
#define MAX_DEPTH        65536       /* max weight_depth or out_depth  */

/* ------------------------------------------------------------------ */
/* Pattern struct                                                       */
/* ------------------------------------------------------------------ */
typedef struct {
    int32_t weight_depth;
    int32_t out_depth;
    int32_t input_zero_bias;
    int32_t out_zero_bias;
    int32_t out_activation_min;
    int32_t out_activation_max;
    int32_t out_multiplier;
    int32_t out_shift;

    int8_t  *input;
    int8_t  *kernel;
    int32_t *bias;
    int8_t  *golden;
    int32_t *partial_sum_golden;

    int32_t  input_size;
    int32_t  kernel_size;
    int32_t  output_size;
    int32_t  partial_sum_size;
} pattern_t;

/* ------------------------------------------------------------------ */
/* Parser helpers (identical contract to test_conv2d_simple.c)         */
/* ------------------------------------------------------------------ */

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
    char *end;
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
    char *end;
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

/* ------------------------------------------------------------------ */
/* Read file into malloc'd buffer                                       */
/* ------------------------------------------------------------------ */
static char *read_file(const char *path, int32_t *out_len)
{
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        printf("ERROR: cannot open '%s'\n", path);
        return NULL;
    }
    char *buf = (char *)malloc(MAX_FILE_BYTES);
    if (!buf) {
        fclose(fp);
        printf("ERROR: malloc failed\n");
        return NULL;
    }
    int32_t total = 0;
    int c;
    while ((c = fgetc(fp)) != EOF) {
        if (total >= MAX_FILE_BYTES - 1) {
            printf("ERROR: file too large (>%d bytes)\n", MAX_FILE_BYTES);
            free(buf);
            fclose(fp);
            return NULL;
        }
        buf[total++] = (char)c;
    }
    buf[total] = '\0';
    *out_len   = total;
    fclose(fp);
    return buf;
}

/* ------------------------------------------------------------------ */
/* Load pattern from text buffer                                        */
/* ------------------------------------------------------------------ */
static int load_pattern(const char *buf, pattern_t *pat)
{
    memset(pat, 0, sizeof(pattern_t));

    /* Default activation range (standard int8) */
    pat->out_activation_min = -128;
    pat->out_activation_max =  127;

    parse_int_param(buf, "weight_depth",       &pat->weight_depth);
    parse_int_param(buf, "out_depth",           &pat->out_depth);
    parse_int_param(buf, "input_zero_bias",     &pat->input_zero_bias);
    parse_int_param(buf, "out_zero_bias",       &pat->out_zero_bias);
    parse_int_param(buf, "out_activation_min",  &pat->out_activation_min);
    parse_int_param(buf, "out_activation_max",  &pat->out_activation_max);
    parse_int_param(buf, "out_multiplier",      &pat->out_multiplier);
    parse_int_param(buf, "out_shift",           &pat->out_shift);

    if (pat->weight_depth < 1 || pat->out_depth < 1) {
        printf("ERROR: invalid weight_depth=%d out_depth=%d\n",
               pat->weight_depth, pat->out_depth);
        return 0;
    }

    /* Allocate buffers */
    int32_t max_in  = pat->weight_depth + 16;
    int32_t max_ker = (int32_t)pat->out_depth * pat->weight_depth + 16;
    int32_t max_out = pat->out_depth + 16;

    pat->input              = (int8_t  *)malloc(max_in);
    pat->kernel             = (int8_t  *)malloc(max_ker);
    pat->bias               = (int32_t *)malloc(max_out * sizeof(int32_t));
    pat->golden             = (int8_t  *)malloc(max_out);
    pat->partial_sum_golden = (int32_t *)malloc(max_out * sizeof(int32_t));

    if (!pat->input || !pat->kernel || !pat->bias ||
        !pat->golden || !pat->partial_sum_golden) {
        printf("ERROR: pattern buffer malloc failed\n");
        return 0;
    }

    /* Parse arrays */
    parse_array_i8 (buf, "int8_t input[",       pat->input,              &pat->input_size);
    parse_array_i8 (buf, "int8_t kernel[",       pat->kernel,             &pat->kernel_size);
    parse_array_i32(buf, "int32_t bias[",        pat->bias,               NULL);
    parse_array_i8 (buf, "int8_t golden[",       pat->golden,             &pat->output_size);
    parse_array_i32(buf, "int32_t partial_sum[", pat->partial_sum_golden, &pat->partial_sum_size);

    return 1;
}

static void free_pattern(pattern_t *pat)
{
    free(pat->input);
    free(pat->kernel);
    free(pat->bias);
    free(pat->golden);
    free(pat->partial_sum_golden);
    memset(pat, 0, sizeof(pattern_t));
}

/* ------------------------------------------------------------------ */
/* Partial sum capture: run FC and collect acc values before requant   */
/* This mirrors cmodel_fc_v2_sym8sxasym8s_asym8s but also captures    */
/* the pre-requant int32 accumulator for comparison.                   */
/* ------------------------------------------------------------------ */
static void fc_with_partial_sum(
    int8_t       *p_out,
    int32_t      *p_partial_sum,      /* [out_depth] pre-requant acc  */
    const int8_t *p_weight,
    const int8_t *p_inp,
    const int32_t *p_bias,
    int32_t       weight_depth,
    int32_t       out_depth,
    int32_t       input_zero_bias,
    int32_t       out_multiplier,
    int32_t       out_shift,
    int32_t       out_zero_bias,
    int32_t       act_min,
    int32_t       act_max)
{
    /*
     * We call the cmodel function for the final output (which is authoritative),
     * but also compute the partial_sum ourselves using the same arithmetic so the
     * comparison is meaningful.
     *
     * partial_sum[n] = clamp_int32( bias[n] + sum_m( W[n][m] * (inp[m] + izb) ) )
     *
     * This matches what gen_fc_patterns.py records.
     */
    for (int n = 0; n < out_depth; n++) {
        const int8_t *w_row = p_weight + (size_t)n * weight_depth;
        int64_t acc = p_bias ? (int64_t)p_bias[n] : 0;
        for (int m = 0; m < weight_depth; m++) {
            int32_t w = (int32_t)w_row[m];
            int32_t x = (int32_t)p_inp[m] + input_zero_bias;
            acc += (int64_t)w * x;
        }
        /* Clamp to int32 (matches Python clamp_int32) */
        if (acc >  (int64_t)INT32_MAX) acc =  (int64_t)INT32_MAX;
        if (acc < -(int64_t)2147483648LL) acc = -(int64_t)2147483648LL;
        p_partial_sum[n] = (int32_t)acc;
    }

    /* Final output via the authoritative cmodel (avoids duplicating requant) */
    cmodel_fc_v2_sym8sxasym8s_asym8s(
        p_out,
        p_weight, p_inp, p_bias,
        weight_depth, out_depth,
        input_zero_bias,
        out_multiplier, out_shift,
        out_zero_bias,
        act_min, act_max);
}

/* ------------------------------------------------------------------ */
/* Compare helpers                                                      */
/* ------------------------------------------------------------------ */
static int compare_output(const int8_t *got, const int8_t *golden, int32_t n)
{
    int32_t miss = 0;
    for (int32_t i = 0; i < n; i++) {
        if (got[i] != golden[i]) {
            if (miss < 10)
                printf("  Output Mismatch[%d]: got=%d  expected=%d\n",
                       i, (int)got[i], (int)golden[i]);
            miss++;
        }
    }
    if (miss) printf("  Total output mismatches: %d / %d\n", miss, n);
    return (miss == 0);
}

static int compare_partial_sum(const int32_t *got, const int32_t *golden, int32_t n)
{
    int32_t miss = 0;
    for (int32_t i = 0; i < n; i++) {
        if (got[i] != golden[i]) {
            if (miss < 10)
                printf("  PS Mismatch[%d]: got=%d  expected=%d\n",
                       i, (int)got[i], (int)golden[i]);
            miss++;
        }
    }
    if (miss) printf("  Total PS mismatches: %d / %d\n", miss, n);
    return (miss == 0);
}

/* ------------------------------------------------------------------ */
/* Run one fixture file                                                 */
/* ------------------------------------------------------------------ */
static int run_fixture(const char *path, int *pass_total, int *fail_total)
{
    printf("\n--- Fixture: %s ---\n", path);

    int32_t file_len = 0;
    char *buf = read_file(path, &file_len);
    if (!buf) { (*fail_total)++; return 0; }

    static pattern_t pat;
    if (!load_pattern(buf, &pat)) {
        free(buf);
        (*fail_total)++;
        return 0;
    }
    free(buf);

    printf("  weight_depth=%d  out_depth=%d\n",
           pat.weight_depth, pat.out_depth);
    printf("  izb=%d  ozb=%d  act=[%d,%d]\n",
           pat.input_zero_bias, pat.out_zero_bias,
           pat.out_activation_min, pat.out_activation_max);
    printf("  out_multiplier=%d  out_shift=%d\n",
           pat.out_multiplier, pat.out_shift);
    printf("  input_size=%d  kernel_size=%d  output_size=%d  ps_size=%d\n",
           pat.input_size, pat.kernel_size, pat.output_size, pat.partial_sum_size);

    /* Validate sizes */
    if (pat.input_size  != pat.weight_depth ||
        pat.kernel_size != (int32_t)pat.out_depth * pat.weight_depth ||
        pat.output_size != pat.out_depth) {
        printf("  ERROR: size mismatch — input_size=%d (expected %d), "
               "kernel_size=%d (expected %d), output_size=%d (expected %d)\n",
               pat.input_size, pat.weight_depth,
               pat.kernel_size, pat.out_depth * pat.weight_depth,
               pat.output_size, pat.out_depth);
        free_pattern(&pat);
        (*fail_total)++;
        return 0;
    }

    /* Output buffers */
    int8_t  *output      = (int8_t  *)malloc(pat.out_depth);
    int32_t *ps_computed = (int32_t *)malloc(pat.out_depth * sizeof(int32_t));
    if (!output || !ps_computed) {
        printf("  ERROR: output malloc failed\n");
        free(output); free(ps_computed);
        free_pattern(&pat);
        (*fail_total)++;
        return 0;
    }
    memset(output,      0, pat.out_depth);
    memset(ps_computed, 0, pat.out_depth * sizeof(int32_t));

    /* Run FC with partial_sum capture */
    fc_with_partial_sum(
        output, ps_computed,
        pat.kernel, pat.input, pat.bias,
        pat.weight_depth, pat.out_depth,
        pat.input_zero_bias,
        pat.out_multiplier, pat.out_shift,
        pat.out_zero_bias,
        pat.out_activation_min, pat.out_activation_max);

    /* Compare output */
    int pass_out = compare_output(output, pat.golden, pat.output_size);

    /* Compare partial_sum */
    int pass_ps = 1;
    if (pat.partial_sum_size == pat.out_depth) {
        pass_ps = compare_partial_sum(ps_computed, pat.partial_sum_golden,
                                      pat.partial_sum_size);
    } else {
        printf("  WARNING: partial_sum_size=%d != out_depth=%d, skipping PS compare\n",
               pat.partial_sum_size, pat.out_depth);
    }

    int pass = pass_out && pass_ps;
    printf("  Output  : %s\n", pass_out ? "PASS" : "FAIL");
    printf("  PartSum : %s\n", pass_ps  ? "PASS" : "FAIL");
    printf("  RESULT  : %s\n", pass     ? "PASS" : "FAIL");

    if (pass) (*pass_total)++; else (*fail_total)++;

    free(output);
    free(ps_computed);
    free_pattern(&pat);
    return pass;
}

/* ------------------------------------------------------------------ */
/* Collect *.in_out files from a directory                             */
/* ------------------------------------------------------------------ */
static int collect_fixtures(const char *dir,
                             char **paths, int max_paths)
{
    DIR *d = opendir(dir);
    if (!d) return 0;

    int count = 0;
    struct dirent *ent;
    while ((ent = readdir(d)) != NULL && count < max_paths) {
        const char *n = ent->d_name;
        size_t len = strlen(n);
        if (len > 7 && strcmp(n + len - 7, ".in_out") == 0) {
            size_t path_len = strlen(dir) + 1 + len + 1;
            paths[count] = (char *)malloc(path_len);
            if (paths[count]) {
                snprintf(paths[count], path_len, "%s/%s", dir, n);
                count++;
            }
        }
    }
    closedir(d);

    /* Sort alphabetically for deterministic order */
    for (int i = 0; i < count - 1; i++)
        for (int j = i + 1; j < count; j++)
            if (strcmp(paths[i], paths[j]) > 0) {
                char *tmp = paths[i]; paths[i] = paths[j]; paths[j] = tmp;
            }
    return count;
}

/* ------------------------------------------------------------------ */
/* main                                                                 */
/* ------------------------------------------------------------------ */
int main(int argc, char **argv)
{
    printf("==================================================\n");
    printf("  FC CModel Pattern Test (sym8sxasym8s_asym8s)\n");
    printf("==================================================\n");

    int pass_total = 0, fail_total = 0;

    if (argc > 1) {
        /* Explicit fixture files on command line */
        for (int i = 1; i < argc; i++)
            run_fixture(argv[i], &pass_total, &fail_total);
    } else {
        /* Auto-discover patterns/ directory */
        const char *dir = "patterns";
        char *paths[256];
        int n = collect_fixtures(dir, paths, 256);
        if (n == 0) {
            printf("No *.in_out fixtures found in '%s/'\n", dir);
            printf("Run: cd patterns && python3 gen_fc_patterns.py\n");
            return 1;
        }
        printf("Found %d fixture(s) in '%s/'\n", n, dir);
        for (int i = 0; i < n; i++) {
            run_fixture(paths[i], &pass_total, &fail_total);
            free(paths[i]);
        }
    }

    int total = pass_total + fail_total;
    printf("\n==================================================\n");
    printf("  TOTAL: %d / %d  PASS   %d FAIL\n",
           pass_total, total, fail_total);
    printf("==================================================\n");

    return (fail_total == 0) ? 0 : 1;
}
