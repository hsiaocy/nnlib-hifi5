/*
 * cmodel_fc_hifi5.c
 *
 * CModel reference implementation for HiFi5 NNLib Fully Connected kernels.
 * Pure scalar C — no SIMD, no AE registers.
 *
 * Bit-exact with xa_nnlib for all inputs within restriction bounds.
 *
 * Tested against:
 *   xa_nn_fully_connected_sym8sxasym8s_asym8s
 *   xa_nn_fully_connected_sym8sxsym16s_sym16s
 *   xa_nn_fully_connected_v2_sym8sxsym16s_sym16s
 */

#include "cmodel_fc_hifi5.h"
#include <stdint.h>
#include <stddef.h>

/* ================================================================== */
/* Internal helpers                                                     */
/* ================================================================== */

static inline WORD32 clamp32(WORD32 x, WORD32 lo, WORD32 hi)
{
    return (x < lo) ? lo : (x > hi) ? hi : x;
}

/* ================================================================== */
/* Core Math                                                            */
/* ================================================================== */

/*
 * SaturatingRoundingDoublingHighMul
 *
 * Computes: round(a * b / 2^31)  with saturation at INT32_MAX.
 *
 * Implements TFLM SaturatingRoundingDoublingHighMul.
 * The "nudge" implements round-half-away-from-zero (ties positive = up,
 * ties negative = down).
 */
WORD32 cmodel_SRDHM(WORD32 a, WORD32 b)
{
    /* Overflow case: INT32_MIN * INT32_MIN would overflow int64 */
    if (a == INT32_MIN && b == INT32_MIN)
        return INT32_MAX;

    int64_t product = (int64_t)a * (int64_t)b;

    /*
     * nudge = 2^30 if product >= 0, else 1 - 2^30
     * This gives round-half-away-from-zero semantics.
     */
    int32_t nudge = (product >= 0) ? (1 << 30) : (1 - (1 << 30));
    int32_t result = (int32_t)((product + nudge) >> 31);
    return result;
}

/*
 * RoundingDivideByPOT
 *
 * Computes: round(x / 2^exp), ties away from zero.
 * exp must be in [0, 31].
 */
WORD32 cmodel_RoundingDivideByPOT(WORD32 x, int exp)
{
    if (exp == 0)
        return x;

    int32_t mask      = (int32_t)((1u << exp) - 1u);
    int32_t remainder = x & mask;
    /* threshold for rounding: half of divisor, +1 if negative (so ties round away) */
    int32_t threshold = (mask >> 1) + (x < 0 ? 1 : 0);
    return (x >> exp) + (remainder > threshold ? 1 : 0);
}

/*
 * MultiplyByQuantizedMultiplier (32-bit accumulator)
 *
 * out_shift > 0 : left-shift accumulator before SRDHM
 *                 (scale > 1 range — uncommon but valid)
 * out_shift < 0 : SRDHM then right-shift result
 *                 (scale < 0.5 range — typical)
 * out_shift = 0 : SRDHM only
 *
 * Mathematical result:
 *   round(acc * out_multiplier * 2^out_shift)
 *   where out_multiplier is Q31 (represents value in [0,1))
 */
WORD32 cmodel_MBQM(WORD32 acc, WORD32 out_multiplier, WORD32 out_shift)
{
    WORD32 result;

    if (out_shift > 0) {
        /* Left-shift first to increase effective range, then multiply */
        /* NOTE: caller must ensure no overflow from left shift */
        acc    = acc << out_shift;
        result = cmodel_SRDHM(acc, out_multiplier);
    } else {
        result = cmodel_SRDHM(acc, out_multiplier);
        result = cmodel_RoundingDivideByPOT(result, -out_shift);
    }

    return result;
}

/*
 * MultiplyByQuantizedMultiplier (64-bit accumulator)
 *
 * For sym16s output path where accumulator is int64.
 * Reduces int64 -> int32 via the same SRDHM+shift mechanism.
 */
WORD32 cmodel_MBQM_64(WORD64 acc, WORD32 out_multiplier, WORD32 out_shift)
{
    /*
     * Strategy: reduce int64 acc to int32 range first.
     *
     * Step 1: compute high 32 bits of (acc * out_multiplier)
     *   This avoids int128; we treat out_multiplier as Q31.
     *   result_Q31 = round(acc * out_multiplier / 2^31)
     *
     * Step 2: apply out_shift as RoundingDivideByPOT or left shift.
     *
     * For int64 accumulator: split into high/low int32 portions.
     */
    int64_t a    = acc;
    int32_t mult = out_multiplier;

    /* Compute a * mult as int64, then >> 31 with rounding */
    /* Safe because |a| <= 2^55 in typical FC (int8 x int16 x 512 + int64 bias) */
    /* and |mult| < 2^31, so product < 2^87 which needs int128 strictly */
    /* However, for practical weight_depth <= 4096: max acc ~ 127*32767*4096 + bias */
    /* = ~17e9 < 2^34, so (int64)a * mult < 2^65, fits in int64 only if mult < 2^31 */
    /* Use __int128 for correctness:                                                 */

    __int128 product = (__int128)a * (__int128)mult;
    int32_t nudge    = (product >= 0) ? (1 << 30) : (1 - (1 << 30));
    int32_t result   = (int32_t)((product + nudge) >> 31);

    /* Apply out_shift */
    if (out_shift > 0) {
        result = result << out_shift;
    } else if (out_shift < 0) {
        result = cmodel_RoundingDivideByPOT(result, -out_shift);
    }

    return result;
}

/* ================================================================== */
/* FC: sym8sxasym8s_asym8s  (non-v2)                                  */
/* ================================================================== */

WORD32 cmodel_fc_sym8sxasym8s_asym8s(
    WORD8        *p_out,
    const WORD8  *p_weight,
    const WORD8  *p_inp,
    const WORD32 *p_bias,
    WORD32        weight_depth,
    WORD32        out_depth,
    WORD32        input_zero_bias,
    WORD32        out_multiplier,
    WORD32        out_shift,
    WORD32        out_zero_bias)
{
    /* Parameter validation */
    if (!p_out || !p_weight || !p_inp)
        return CMODEL_FC_ERR;
    if (out_depth < 1 || weight_depth < 1)
        return CMODEL_FC_ERR;
    if (out_multiplier <= 0)
        return CMODEL_FC_ERR;
    if (out_shift < -31 || out_shift > 31)
        return CMODEL_FC_ERR;
    /* Guide says [-127..128] for asym8s, but official testbench uses default=-128.
     * Allow [-128..128] to cover all practical TFLM-derived zero-point values.   */
    if (input_zero_bias < -128 || input_zero_bias > 128)
        return CMODEL_FC_ERR;
    if (out_zero_bias < -128 || out_zero_bias > 128)
        return CMODEL_FC_ERR;

    for (int n = 0; n < out_depth; n++) {
        const WORD8 *w_row = p_weight + (size_t)n * weight_depth;

        /*
         * Accumulate: acc += W[n][m] * (inp[m] + input_zero_bias)
         *
         * Use int64 to be safe against overflow when weight_depth is large.
         * NNLib HiFi5 SIMD version also operates in wider accumulator
         * internally (AE_MULA8QW8X16 etc.) before truncating.
         */
        int64_t acc = 0;
        for (int m = 0; m < weight_depth; m++) {
            int32_t w    = (int32_t)w_row[m];
            int32_t x    = (int32_t)p_inp[m] + input_zero_bias;
            acc         += (int64_t)w * x;
        }

        /* Add bias (int32 bias, same Q-space as acc) */
        if (p_bias)
            acc += (int64_t)p_bias[n];

        /*
         * Requantize: MBQM operates on int32.
         * acc should be within int32 range for typical models
         * (weight_depth < ~16M for int8 x int8 products).
         * Clamp to int32 before MBQM to be explicit.
         */
        int32_t acc32 = (acc > INT32_MAX) ? INT32_MAX :
                        (acc < INT32_MIN) ? INT32_MIN : (int32_t)acc;

        int32_t result = cmodel_MBQM(acc32, out_multiplier, out_shift);

        /* Add output zero point */
        result += out_zero_bias;

        /* Clamp to int8 range */
        p_out[n] = (WORD8)clamp32(result, -128, 127);
    }

    return CMODEL_FC_OK;
}

/* ================================================================== */
/* FC v2: sym8sxasym8s_asym8s  (with fused activation clamp)          */
/* ================================================================== */

WORD32 cmodel_fc_v2_sym8sxasym8s_asym8s(
    WORD8        *p_out,
    const WORD8  *p_weight,
    const WORD8  *p_inp,
    const WORD32 *p_bias,
    WORD32        weight_depth,
    WORD32        out_depth,
    WORD32        input_zero_bias,
    WORD32        out_multiplier,
    WORD32        out_shift,
    WORD32        out_zero_bias,
    WORD32        out_activation_min,
    WORD32        out_activation_max)
{
    /* Parameter validation */
    if (!p_out || !p_weight || !p_inp)
        return CMODEL_FC_ERR;
    if (out_depth < 1 || weight_depth < 1)
        return CMODEL_FC_ERR;
    if (out_multiplier <= 0)
        return CMODEL_FC_ERR;
    if (out_shift < -31 || out_shift > 31)
        return CMODEL_FC_ERR;
    if (out_activation_min > out_activation_max)
        return CMODEL_FC_ERR;
    if (input_zero_bias < -128 || input_zero_bias > 128)
        return CMODEL_FC_ERR;

    for (int n = 0; n < out_depth; n++) {
        const WORD8 *w_row = p_weight + (size_t)n * weight_depth;

        int64_t acc = 0;
        for (int m = 0; m < weight_depth; m++) {
            int32_t w  = (int32_t)w_row[m];
            int32_t x  = (int32_t)p_inp[m] + input_zero_bias;
            acc       += (int64_t)w * x;
        }

        if (p_bias)
            acc += (int64_t)p_bias[n];

        int32_t acc32 = (acc > INT32_MAX) ? INT32_MAX :
                        (acc < INT32_MIN) ? INT32_MIN : (int32_t)acc;

        int32_t result = cmodel_MBQM(acc32, out_multiplier, out_shift);
        result        += out_zero_bias;

        /* Fused activation clamp (v2: caller-specified range) */
        result = clamp32(result, out_activation_min, out_activation_max);

        p_out[n] = (WORD8)(int8_t)result;
    }

    return CMODEL_FC_OK;
}

/* ================================================================== */
/* FC: sym8sxsym16s_sym16s  (non-v2)                                  */
/* ================================================================== */

WORD32 cmodel_fc_sym8sxsym16s_sym16s(
    WORD16       *p_out,
    const WORD8  *p_weight,
    const WORD16 *p_inp,
    const WORD64 *p_bias,
    WORD32        weight_depth,
    WORD32        out_depth,
    WORD32        out_multiplier,
    WORD32        out_shift)
{
    if (!p_out || !p_weight || !p_inp || !p_bias)
        return CMODEL_FC_ERR;
    if (out_depth < 1 || weight_depth < 1)
        return CMODEL_FC_ERR;
    if (out_multiplier <= 0)
        return CMODEL_FC_ERR;
    if (out_shift < -31 || out_shift > 31)
        return CMODEL_FC_ERR;

    for (int n = 0; n < out_depth; n++) {
        const WORD8 *w_row = p_weight + (size_t)n * weight_depth;

        /*
         * int8 x int16 -> product is int24, sum of weight_depth terms.
         * Max magnitude: 127 * 32767 * 32768 ~ 1.36e11 < 2^37 -> safe in int64.
         */
        int64_t acc = 0;
        for (int m = 0; m < weight_depth; m++) {
            int64_t w  = (int64_t)w_row[m];
            int64_t x  = (int64_t)p_inp[m];
            acc       += w * x;
        }

        acc += p_bias[n];

        int32_t result = cmodel_MBQM_64(acc, out_multiplier, out_shift);

        /* Clamp to int16 (sym16s) */
        p_out[n] = (WORD16)clamp32(result, -32768, 32767);
    }

    return CMODEL_FC_OK;
}

/* ================================================================== */
/* FC v2: sym8sxsym16s_sym16s  (with fused activation clamp)          */
/* ================================================================== */

WORD32 cmodel_fc_v2_sym8sxsym16s_sym16s(
    WORD16       *p_out,
    const WORD8  *p_weight,
    const WORD16 *p_inp,
    const WORD64 *p_bias,
    WORD32        weight_depth,
    WORD32        out_depth,
    WORD32        out_multiplier,
    WORD32        out_shift,
    WORD32        out_activation_min,
    WORD32        out_activation_max)
{
    if (!p_out || !p_weight || !p_inp || !p_bias)
        return CMODEL_FC_ERR;
    if (out_depth < 1 || weight_depth < 1)
        return CMODEL_FC_ERR;
    if (out_multiplier <= 0)
        return CMODEL_FC_ERR;
    if (out_shift < -31 || out_shift > 31)
        return CMODEL_FC_ERR;
    if (out_activation_min > out_activation_max)
        return CMODEL_FC_ERR;

    for (int n = 0; n < out_depth; n++) {
        const WORD8 *w_row = p_weight + (size_t)n * weight_depth;

        int64_t acc = 0;
        for (int m = 0; m < weight_depth; m++) {
            int64_t w  = (int64_t)w_row[m];
            int64_t x  = (int64_t)p_inp[m];
            acc       += w * x;
        }

        acc += p_bias[n];

        int32_t result = cmodel_MBQM_64(acc, out_multiplier, out_shift);

        result = clamp32(result, out_activation_min, out_activation_max);

        p_out[n] = (WORD16)(int16_t)result;
    }

    return CMODEL_FC_OK;
}
