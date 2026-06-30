/*
 * conv2d_depthwise_cmodel.h
 *
 * Header for the C reference model of xa_nn_conv2d_depthwise_sym8sxasym8s.
 *
 * ── Kernel layout ─────────────────────────────────────────────────────────
 *
 *   p_kernel shape: [KH, KW, OC]  (OC = IC * channels_multiplier)
 *   linear index  : kh * KW * OC + kw * OC + oc
 *   where oc      = ic * channels_multiplier + cm
 *
 * ── Parameter API notes ───────────────────────────────────────────────────
 *
 *   p_kernel and p_inp are in SWAPPED order (NNLib convention):
 *     NNLib: xa_nn_conv2d_depthwise_sym8sxasym8s(p_out, p_kernel, p_inp, ...)
 *
 *   input_zero_bias  = -input_zero_point   (pass NEGATED zero-point)
 *   kernel_zero_bias = 0                   (sym8s: always 0, not a parameter)
 *   out_zero_bias    = +output_zero_point  (added after requantization)
 *
 *   dilation_height / dilation_width = 1 for standard depthwise conv.
 *
 * ── partial_sum ───────────────────────────────────────────────────────────
 *
 *   p_partial_sum[oh*OW*OC + ow*OC + oc] =
 *       bias[oc] + sum_kh_kw( (inp+izp) * w )
 *   Captured BEFORE requantize.  Pass NULL to skip capture.
 */

#ifndef CONV2D_DEPTHWISE_CMODEL_H
#define CONV2D_DEPTHWISE_CMODEL_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void cmodel_conv2d_depthwise_sym8sxasym8s(
    int8_t        * __restrict__ p_out,
    const int8_t  * __restrict__ p_kernel,          /* [KH, KW, OC], sym8s   */
    const int8_t  * __restrict__ p_inp,             /* [IH, IW, IC], asym8s  */
    const int32_t * __restrict__ p_bias,            /* [OC], can be NULL      */
    const int32_t * __restrict__ p_out_multiplier,  /* [OC]                   */
    const int32_t * __restrict__ p_out_shift,       /* [OC]                   */
    int input_height,
    int input_width,
    int input_channels,
    int kernel_height,
    int kernel_width,
    int channels_multiplier,    /* depth multiplier, typically 1              */
    int dilation_height,        /* 1 = standard (no dilation)                 */
    int dilation_width,
    int stride_height,
    int stride_width,
    int pad_height,
    int pad_width,
    int output_height,
    int output_width,
    int input_zero_bias,        /* = -input_zero_point  (negated)             */
    int out_zero_bias,          /* = +output_zero_point                       */
    int activation_min,
    int activation_max,
    int32_t * __restrict__ p_partial_sum  /* [OH*OW*OC], can be NULL          */
);

#ifdef __cplusplus
}
#endif

#endif /* CONV2D_DEPTHWISE_CMODEL_H */
