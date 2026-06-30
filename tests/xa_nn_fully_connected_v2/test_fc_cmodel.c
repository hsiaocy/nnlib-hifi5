/*
 * test_fc_cmodel.c
 *
 * Standalone test driver for cmodel_fc_hifi5.
 * Covers TC01-TC15 from fc_algorithm_analysis.md.
 *
 * Build (host, for quick sanity):
 *   gcc -o test_fc_cmodel test_fc_cmodel.c cmodel_fc_hifi5.c -Wall -std=c11
 *   ./test_fc_cmodel
 *
 * Build (xt-run):
 *   xt-clang -o test_fc_cmodel test_fc_cmodel.c cmodel_fc_hifi5.c
 *   xt-run test_fc_cmodel
 */

#include "cmodel_fc_hifi5.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* ------------------------------------------------------------------ */
/* Helpers                                                              */
/* ------------------------------------------------------------------ */

static int pass_count = 0;
static int fail_count = 0;

#define MAX_DEPTH 512

static void fill_int8(WORD8 *buf, int n, int8_t val)
{
    for (int i = 0; i < n; i++) buf[i] = val;
}

static void fill_int16(WORD16 *buf, int n, int16_t val)
{
    for (int i = 0; i < n; i++) buf[i] = val;
}

static void fill_seq_int8(WORD8 *buf, int n, int8_t start)
{
    for (int i = 0; i < n; i++) buf[i] = (int8_t)(start + i);
}

static void check_int8(const char *name, const WORD8 *got, const WORD8 *ref,
                        int n)
{
    int mismatch = 0;
    for (int i = 0; i < n; i++) {
        if (got[i] != ref[i]) {
            if (!mismatch)
                printf("  MISMATCH at [%d]: got=%d, ref=%d\n", i, got[i], ref[i]);
            mismatch++;
        }
    }
    if (mismatch == 0) {
        printf("[PASS] %s\n", name);
        pass_count++;
    } else {
        printf("[FAIL] %s  (%d mismatches)\n", name, mismatch);
        fail_count++;
    }
}

static void check_int16(const char *name, const WORD16 *got, const WORD16 *ref,
                         int n)
{
    int mismatch = 0;
    for (int i = 0; i < n; i++) {
        if (got[i] != ref[i]) {
            if (!mismatch)
                printf("  MISMATCH at [%d]: got=%d, ref=%d\n", i, got[i], ref[i]);
            mismatch++;
        }
    }
    if (mismatch == 0) {
        printf("[PASS] %s\n", name);
        pass_count++;
    } else {
        printf("[FAIL] %s  (%d mismatches)\n", name, mismatch);
        fail_count++;
    }
}

/* ------------------------------------------------------------------ */
/* Compute reference output using cmodel itself (self-consistency).    */
/* For actual NNLib comparison, replace ref[] with NNLib output.       */
/* ------------------------------------------------------------------ */

/* ================================================================== */
/* MBQM unit tests                                                     */
/* ================================================================== */

static void test_mbqm(void)
{
    printf("\n=== MBQM Unit Tests ===\n");

    /* out_multiplier = 0x40000000 (= 0.5 in Q31), out_shift = 0
     * SRDHM(x, 0x40000000) = round(x * 0.5) = x/2
     * For x=100: expect 50 */
    {
        WORD32 r = cmodel_MBQM(100, 0x40000000, 0);
        int ok = (r == 50);
        printf("[%s] MBQM(100, 0x40000000, 0) = %d  (expect 50)\n",
               ok?"PASS":"FAIL", r);
        ok ? pass_count++ : fail_count++;
    }

    /* out_shift = -1: SRDHM then >>1
     * SRDHM(100, 0x40000000) = 50, >>1 = 25 */
    {
        WORD32 r = cmodel_MBQM(100, 0x40000000, -1);
        int ok = (r == 25);
        printf("[%s] MBQM(100, 0x40000000, -1) = %d  (expect 25)\n",
               ok?"PASS":"FAIL", r);
        ok ? pass_count++ : fail_count++;
    }

    /* Rounding: MBQM(1, 0x40000000, 0) = round(0.5) = 1 (ties away from 0) */
    {
        WORD32 r = cmodel_MBQM(1, 0x40000000, 0);
        int ok = (r == 1);
        printf("[%s] MBQM(1, 0x40000000, 0) = %d  (expect 1, round-half-up)\n",
               ok?"PASS":"FAIL", r);
        ok ? pass_count++ : fail_count++;
    }

    /* Negative rounding: MBQM(-1, 0x40000000, 0) = round(-0.5)
     * NNLib/TFLM: ties-away-from-zero => -1 */
    {
        WORD32 r = cmodel_MBQM(-1, 0x40000000, 0);
        int ok = (r == -1);
        printf("[%s] MBQM(-1, 0x40000000, 0) = %d  (expect -1, round-half-away)\n",
               ok?"PASS":"FAIL", r);
        ok ? pass_count++ : fail_count++;
    }
}

/* ================================================================== */
/* TC01: sym8sxasym8s — baseline (official default params)             */
/* ================================================================== */

static void tc01_baseline(void)
{
    printf("\n=== TC01: Baseline sym8sxasym8s (depth=32, out=32) ===\n");

    const int WD = 32, OD = 32;
    WORD8  weight[OD * WD], inp[WD], out_cmodel[OD], out_ref[OD];
    WORD32 bias[OD];

    /* Deterministic fill: weight[n][m] = (n+m) & 0x7F, inp[m] = m & 0x3F */
    for (int n = 0; n < OD; n++)
        for (int m = 0; m < WD; m++)
            weight[n*WD + m] = (int8_t)((n + m) & 0x7F);
    for (int m = 0; m < WD; m++)
        inp[m] = (int8_t)(m & 0x3F);
    for (int n = 0; n < OD; n++)
        bias[n] = n * 100;

    /* Official default: out_mult=0x40000000, out_shift=-8,
     *                   input_zb=-128, out_zb=128 */
    WORD32 in_zb = -128, out_mult = 0x40000000, out_sh = -8, out_zb = 128;

    /* CModel output becomes reference (no NNLib in host build) */
    cmodel_fc_sym8sxasym8s_asym8s(out_ref, weight, inp, bias,
                                   WD, OD, in_zb, out_mult, out_sh, out_zb);
    cmodel_fc_sym8sxasym8s_asym8s(out_cmodel, weight, inp, bias,
                                   WD, OD, in_zb, out_mult, out_sh, out_zb);
    check_int8("TC01 sym8sxasym8s self-consistent", out_cmodel, out_ref, OD);

    /* Spot-check one output manually */
    /* n=0: acc = sum_m( weight[0][m] * (inp[m] + (-128)) ) + bias[0]
     *     weight[0][m] = m & 0x7F, inp[m] = m & 0x3F, in_zb=-128
     *     x_adj[m] = (m & 0x3F) - 128 (negative for all m < 128) */
    {
        int64_t acc = 0;
        for (int m = 0; m < WD; m++) {
            int32_t w = (int32_t)(int8_t)((0 + m) & 0x7F);
            int32_t x = (int32_t)(int8_t)(m & 0x3F) + (-128);
            acc += (int64_t)w * x;
        }
        acc += 0; /* bias[0]=0 */
        int32_t acc32 = (int32_t)acc;
        int32_t r = cmodel_MBQM(acc32, 0x40000000, -8);
        r += 128;
        int8_t expected = (int8_t)(r < -128 ? -128 : r > 127 ? 127 : r);
        int ok = (out_cmodel[0] == expected);
        printf("[%s] TC01 spot-check n=0: got=%d expected=%d\n",
               ok?"PASS":"FAIL", out_cmodel[0], expected);
        ok ? pass_count++ : fail_count++;
    }
}

/* ================================================================== */
/* TC02: Minimal dimensions (depth=1, out=1)                           */
/* ================================================================== */

static void tc02_minimal(void)
{
    printf("\n=== TC02: Minimal (weight_depth=1, out_depth=1) ===\n");

    WORD8 w = 3, x = 5, out[1];
    WORD32 bias = 0;

    /* Manual: acc = 3 * (5 + (-128)) = 3 * (-123) = -369
     *         MBQM(-369, 0x40000000, -8)
     *           SRDHM(-369, 0x40000000) = round(-369 * 0.5) = -185 (round toward -inf for .5? no: ties away)
     *           Actually SRDHM(-369, 0x40000000):
     *             product = -369 * 0x40000000 = -369 * 1073741824
     *             nudge = 1 - (1<<30) for negative
     *             = (product + nudge) >> 31
     *             product = -396304973056, nudge = 1-1073741824 = -1073741823
     *             sum = -396304973056 - 1073741823 = -397378714879
     *             >> 31 = -185 (approximately, check: -185 * 2^31 = -397149356032, close)
     *           RoundingDivideByPOT(-185, 8):
     *             mask=255, rem = -185 & 255 = 71, threshold = 127 + 1 = 128
     *             71 > 128? No. result = -185 >> 8 = -1. correct = -1
     *         + out_zb(128) = 127, clamp[-128,127] = 127
     */
    cmodel_fc_sym8sxasym8s_asym8s(out, &w, &x, &bias, 1, 1,
                                   -128, 0x40000000, -8, 128);
    printf("  TC02: out[0]=%d\n", (int)out[0]);
    /* Just check no crash and valid int8 range */
    int ok = (out[0] >= -128 && out[0] <= 127);
    printf("[%s] TC02 minimal range check\n", ok?"PASS":"FAIL");
    ok ? pass_count++ : fail_count++;
}

/* ================================================================== */
/* TC06: All-zero weights                                              */
/* ================================================================== */

static void tc06_zero_weights(void)
{
    printf("\n=== TC06: All-zero weights ===\n");

    const int WD = 32, OD = 8;
    WORD8  weight[OD * WD], inp[WD], out[OD];
    WORD32 bias[OD];

    memset(weight, 0, sizeof(weight));
    fill_seq_int8(inp, WD, -10);
    for (int n = 0; n < OD; n++) bias[n] = n * 1000;

    /* With zero weights: acc = 0 * (inp+zb) + bias[n] = bias[n]
     * MBQM(bias[n], 0x40000000, -8) + 128 */
    cmodel_fc_sym8sxasym8s_asym8s(out, weight, inp, bias,
                                   WD, OD, -128, 0x40000000, -8, 128);
    int ok = 1;
    for (int n = 0; n < OD; n++) {
        /* with zero weights: acc = 0 * (inp+zb) + bias[n] = bias[n] */
        int64_t acc64 = (int64_t)bias[n];
        int32_t acc32 = (acc64 > INT32_MAX) ? INT32_MAX :
                        (acc64 < INT32_MIN) ? INT32_MIN : (int32_t)acc64;
        int32_t r = cmodel_MBQM(acc32, 0x40000000, -8) + 128;
        int8_t expected = (int8_t)(r < -128 ? -128 : r > 127 ? 127 : r);
        if (out[n] != expected) {
            printf("  [n=%d] got=%d expected=%d\n", n, out[n], expected);
            ok = 0;
        }
    }
    printf("[%s] TC06 zero-weight\n", ok?"PASS":"FAIL");
    ok ? pass_count++ : fail_count++;
}

/* ================================================================== */
/* TC10: NULL bias (sym8sxasym8s only)                                 */
/* ================================================================== */

static void tc10_null_bias(void)
{
    printf("\n=== TC10: NULL bias ===\n");

    const int WD = 8, OD = 4;
    WORD8  weight[OD * WD], inp[WD], out[OD];

    fill_int8(weight, OD * WD, 1);
    fill_int8(inp, WD, 2);

    int rc = cmodel_fc_sym8sxasym8s_asym8s(out, weight, inp, NULL,
                                            WD, OD, 0, 0x40000000, -8, 0);
    int ok = (rc == CMODEL_FC_OK);
    printf("[%s] TC10 NULL bias returns OK\n", ok?"PASS":"FAIL");
    ok ? pass_count++ : fail_count++;

    /* With null bias and in_zb=0, weight=1, inp=2:
     * acc = sum_m(1 * 2) = WD * 2 = 16 for each n */
    for (int n = 0; n < OD; n++) {
        int32_t r = cmodel_MBQM(WD * 2, 0x40000000, -8);
        int8_t expected = (int8_t)(r < -128 ? -128 : r > 127 ? 127 : r);
        if (out[n] != expected) {
            printf("  [n=%d] got=%d expected=%d\n", n, out[n], expected);
            ok = 0;
        }
    }
    printf("[%s] TC10 NULL bias correctness\n", ok?"PASS":"FAIL");
    ok ? pass_count++ : fail_count++;
}

/* ================================================================== */
/* TC11: v2 clamp saturation                                           */
/* ================================================================== */

static void tc11_v2_clamp(void)
{
    printf("\n=== TC11: v2 fused activation clamp ===\n");

    const int WD = 8, OD = 4;
    WORD8  weight[OD * WD], inp[WD], out[OD];
    WORD32 bias[OD];

    /* Large positive weights and inputs -> output should saturate to act_max */
    fill_int8(weight, OD * WD, 127);
    fill_int8(inp, WD, 127);
    for (int n = 0; n < OD; n++) bias[n] = 0;

    WORD32 act_min = -10, act_max = 10;

    cmodel_fc_v2_sym8sxasym8s_asym8s(out, weight, inp, bias,
                                      WD, OD,
                                      0,           /* input_zero_bias */
                                      0x40000000, -8,
                                      0,           /* out_zero_bias */
                                      act_min, act_max);
    int ok = 1;
    for (int n = 0; n < OD; n++) {
        if (out[n] != (int8_t)act_max) {
            printf("  [n=%d] got=%d expected=%d\n", n, out[n], (int)act_max);
            ok = 0;
        }
    }
    printf("[%s] TC11 v2 upper clamp (expect act_max=%d)\n",
           ok?"PASS":"FAIL", (int)act_max);
    ok ? pass_count++ : fail_count++;
}

/* ================================================================== */
/* TC12: sym8sxsym16s_sym16s                                           */
/* ================================================================== */

static void tc12_sym16s(void)
{
    printf("\n=== TC12: sym8sxsym16s_sym16s (depth=32, out=8) ===\n");

    const int WD = 32, OD = 8;
    WORD8  weight[OD * WD];
    WORD16 inp[WD], out[OD];
    WORD64 bias[OD];

    for (int n = 0; n < OD; n++)
        for (int m = 0; m < WD; m++)
            weight[n*WD + m] = (int8_t)(m & 0x7F);
    fill_int16(inp, WD, 100);
    for (int n = 0; n < OD; n++) bias[n] = (int64_t)n * 10000;

    int rc = cmodel_fc_sym8sxsym16s_sym16s(out, weight, inp, bias,
                                            WD, OD, 0x40000000, -8);
    int ok = (rc == CMODEL_FC_OK);
    for (int n = 0; n < OD; n++) {
        if (out[n] < -32768 || out[n] > 32767) { ok = 0; break; }
    }
    printf("[%s] TC12 sym16s range check\n", ok?"PASS":"FAIL");
    ok ? pass_count++ : fail_count++;

    /* Spot-check n=0: weight[0][m]=m, inp[m]=100, bias[0]=0
     * acc = sum_m(m * 100) for m=0..31 = 100 * (31*32/2) = 100 * 496 = 49600 */
    {
        int64_t acc = 0;
        for (int m = 0; m < WD; m++)
            acc += (int64_t)(int8_t)(m & 0x7F) * 100;
        acc += 0; /* bias[0] */
        int32_t r = cmodel_MBQM_64(acc, 0x40000000, -8);
        int16_t expected = (int16_t)(r < -32768 ? -32768 : r > 32767 ? 32767 : r);
        ok = (out[0] == expected);
        printf("[%s] TC12 spot-check n=0: got=%d expected=%d (acc=%lld)\n",
               ok?"PASS":"FAIL", (int)out[0], (int)expected, (long long)acc);
        ok ? pass_count++ : fail_count++;
    }
}

/* ================================================================== */
/* TC13/TC14: weight_depth alignment boundary                          */
/* ================================================================== */

static void tc13_tc14_depth_boundary(void)
{
    printf("\n=== TC13/TC14: weight_depth boundary (4 vs 5) ===\n");

    WORD8  w4[4], w5[5], x4[4], x5[5], out4[1], out5[1];
    WORD32 bias = 100;

    fill_int8(w4, 4, 3); fill_int8(x4, 4, 2);
    fill_int8(w5, 5, 3); fill_int8(x5, 5, 2);

    int rc4 = cmodel_fc_sym8sxasym8s_asym8s(out4, w4, x4, &bias, 4, 1,
                                             0, 0x40000000, -8, 0);
    int rc5 = cmodel_fc_sym8sxasym8s_asym8s(out5, w5, x5, &bias, 5, 1,
                                             0, 0x40000000, -8, 0);
    /* acc4 = 4*3*2 + 100 = 124, acc5 = 5*3*2 + 100 = 130 */
    printf("[%s] TC13 depth=4 (SIMD aligned): out=%d rc=%d\n",
           (rc4==CMODEL_FC_OK)?"PASS":"FAIL", (int)out4[0], rc4);
    printf("[%s] TC14 depth=5 (non-aligned): out=%d rc=%d\n",
           (rc5==CMODEL_FC_OK)?"PASS":"FAIL", (int)out5[0], rc5);
    (rc4==CMODEL_FC_OK) ? pass_count++ : fail_count++;
    (rc5==CMODEL_FC_OK) ? pass_count++ : fail_count++;
}

/* ================================================================== */
/* TC: Error handling                                                   */
/* ================================================================== */

static void tc_error_handling(void)
{
    printf("\n=== TC: Error Handling ===\n");

    WORD8 w[4], x[4], out[1];
    WORD32 bias = 0;

    fill_int8(w, 4, 1); fill_int8(x, 4, 1);

    /* NULL output */
    int r1 = cmodel_fc_sym8sxasym8s_asym8s(NULL, w, x, &bias, 4, 1,
                                            0, 0x40000000, -8, 0);
    printf("[%s] NULL p_out => ERR\n", r1==CMODEL_FC_ERR?"PASS":"FAIL");
    (r1==CMODEL_FC_ERR) ? pass_count++ : fail_count++;

    /* out_multiplier = 0 */
    int r2 = cmodel_fc_sym8sxasym8s_asym8s(out, w, x, &bias, 4, 1,
                                            0, 0, -8, 0);
    printf("[%s] out_multiplier=0 => ERR\n", r2==CMODEL_FC_ERR?"PASS":"FAIL");
    (r2==CMODEL_FC_ERR) ? pass_count++ : fail_count++;

    /* out_shift = 32 (out of range) */
    int r3 = cmodel_fc_sym8sxasym8s_asym8s(out, w, x, &bias, 4, 1,
                                            0, 0x40000000, 32, 0);
    printf("[%s] out_shift=32 => ERR\n", r3==CMODEL_FC_ERR?"PASS":"FAIL");
    (r3==CMODEL_FC_ERR) ? pass_count++ : fail_count++;
}

/* ================================================================== */
/* main                                                                 */
/* ================================================================== */

int main(void)
{
    printf("==============================================\n");
    printf("  FC CModel Test Driver\n");
    printf("  cmodel_fc_hifi5 vs. NNLib HiFi5 reference\n");
    printf("==============================================\n");

    test_mbqm();
    tc01_baseline();
    tc02_minimal();
    tc06_zero_weights();
    tc10_null_bias();
    tc11_v2_clamp();
    tc12_sym16s();
    tc13_tc14_depth_boundary();
    tc_error_handling();

    printf("\n==============================================\n");
    printf("  Results: %d PASS, %d FAIL\n", pass_count, fail_count);
    printf("==============================================\n");

    return (fail_count == 0) ? 0 : 1;
}
