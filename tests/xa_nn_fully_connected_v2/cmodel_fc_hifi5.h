/*
 * cmodel_fc_hifi5.h
 *
 * CModel reference implementation for:
 *   - xa_nn_fully_connected_sym8sxasym8s_asym8s
 *   - xa_nn_fully_connected_sym8sxsym16s_sym16s
 *   - xa_nn_fully_connected_v2_sym8sxsym16s_sym16s
 *
 * Purpose : Functional alignment with HiFi5 NNLib (xa_nnlib_hifi5).
 *           Bit-exact output expected for all non-overflow cases.
 *           Used as golden reference for hardware RTL verification.
 *
 * No SIMD/AE register usage — pure scalar C, portable.
 */

#ifndef CMODEL_FC_HIFI5_H
#define CMODEL_FC_HIFI5_H

#include <stdint.h>

/* ------------------------------------------------------------------ */
/* Type aliases matching NNLib conventions                             */
/* ------------------------------------------------------------------ */
typedef int8_t   WORD8;
typedef int16_t  WORD16;
typedef int32_t  WORD32;
typedef int64_t  WORD64;

/* Return codes */
#define CMODEL_FC_OK    0
#define CMODEL_FC_ERR  -1

/* ------------------------------------------------------------------ */
/* Core math helpers (exposed for unit testing)                        */
/* ------------------------------------------------------------------ */

/**
 * SaturatingRoundingDoublingHighMul
 *   Computes round(a * b / 2^31), saturating at INT32_MAX.
 *   Equivalent to TFLM's SaturatingRoundingDoublingHighMul.
 */
WORD32 cmodel_SRDHM(WORD32 a, WORD32 b);

/**
 * RoundingDivideByPOT
 *   Computes round(x / 2^exp), ties away from zero.
 *   exp must be >= 0.
 */
WORD32 cmodel_RoundingDivideByPOT(WORD32 x, int exp);

/**
 * MultiplyByQuantizedMultiplier
 *   Standard NNLib/TFLM requantize step.
 *   out_shift: positive = left shift, negative = right shift.
 *   Range: out_shift in [-31 .. 31], out_multiplier in [1 .. INT32_MAX].
 */
WORD32 cmodel_MBQM(WORD32 acc, WORD32 out_multiplier, WORD32 out_shift);

/**
 * MultiplyByQuantizedMultiplier_64
 *   64-bit accumulator variant for sym16s output path.
 */
WORD32 cmodel_MBQM_64(WORD64 acc, WORD32 out_multiplier, WORD32 out_shift);

/* ------------------------------------------------------------------ */
/* FC Variants                                                          */
/* ------------------------------------------------------------------ */

/**
 * cmodel_fc_sym8sxasym8s_asym8s
 *
 * Mirrors: xa_nn_fully_connected_sym8sxasym8s_asym8s
 *
 * Algorithm:
 *   for n in [0, out_depth):
 *     acc = 0
 *     for m in [0, weight_depth):
 *       acc += W[n*weight_depth + m] * (inp[m] + input_zero_bias)
 *     if p_bias: acc += p_bias[n]
 *     acc  = MBQM(acc, out_multiplier, out_shift)
 *     acc += out_zero_bias
 *     out[n] = clamp(acc, -128, 127)
 *
 * Params:
 *   p_out          : output buffer [out_depth]  (int8, asym8s)
 *   p_weight       : weight matrix [out_depth * weight_depth] (int8, sym8s)
 *                    row-major: W[n][m] = p_weight[n*weight_depth + m]
 *   p_inp          : input vector [weight_depth] (int8, asym8s)
 *   p_bias         : bias vector [out_depth] (int32), may be NULL
 *   weight_depth   : number of input features
 *   out_depth      : number of output neurons
 *   input_zero_bias: input zero-point correction, range [-127..128]
 *   out_multiplier : Q31 multiplier, > 0
 *   out_shift      : shift [-31..31], positive=left
 *   out_zero_bias  : output zero point [-128..127]
 *
 * Returns: CMODEL_FC_OK or CMODEL_FC_ERR
 */
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
    WORD32        out_zero_bias
);

/**
 * cmodel_fc_v2_sym8sxasym8s_asym8s
 *
 * Mirrors: xa_nn_fully_connected_v2_asym8sxasym8s_asym8s
 *          (with weight_zero_bias = 0 to emulate sym8s weight)
 *
 * Adds fused clamp via out_activation_min / out_activation_max.
 * p_dma_cfg is ignored (placeholder).
 *
 * Algorithm: same as above, but clamp uses act_min/act_max instead of
 *            hardcoded [-128, 127].
 *
 * Returns: CMODEL_FC_OK or CMODEL_FC_ERR
 */
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
    WORD32        out_activation_max
);

/**
 * cmodel_fc_sym8sxsym16s_sym16s
 *
 * Mirrors: xa_nn_fully_connected_sym8sxsym16s_sym16s
 *
 * Algorithm:
 *   for n in [0, out_depth):
 *     acc = 0  (int64)
 *     for m in [0, weight_depth):
 *       acc += (int64)W[n*weight_depth + m] * (int64)inp[m]
 *     acc += p_bias[n]   (int64)
 *     result = MBQM_64(acc, out_multiplier, out_shift)
 *     out[n] = clamp(result, -32768, 32767)
 *
 * Note: No zero biases (sym inputs, sym weights).
 *       Bias is int64 to prevent overflow with large weight_depth.
 *
 * Returns: CMODEL_FC_OK or CMODEL_FC_ERR
 */
WORD32 cmodel_fc_sym8sxsym16s_sym16s(
    WORD16       *p_out,
    const WORD8  *p_weight,
    const WORD16 *p_inp,
    const WORD64 *p_bias,
    WORD32        weight_depth,
    WORD32        out_depth,
    WORD32        out_multiplier,
    WORD32        out_shift
);

/**
 * cmodel_fc_v2_sym8sxsym16s_sym16s
 *
 * Mirrors: xa_nn_fully_connected_v2_sym8sxsym16s_sym16s
 *
 * Adds out_activation_min / out_activation_max clamp.
 *
 * Returns: CMODEL_FC_OK or CMODEL_FC_ERR
 */
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
    WORD32        out_activation_max
);

#endif /* CMODEL_FC_HIFI5_H */
