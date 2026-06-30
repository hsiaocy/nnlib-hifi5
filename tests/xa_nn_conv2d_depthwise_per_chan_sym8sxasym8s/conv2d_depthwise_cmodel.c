/*
 * conv2d_depthwise_cmodel.c
 *
 * C reference model for xa_nn_conv2d_depthwise_sym8sxasym8s (HiFi5 NNLib).
 *
 * Requant core is verbatim from conv2d_std_cim_cmodel.c (quantize_change +
 * shift_change path).  Only the outer loop structure differs to implement
 * depthwise semantics.
 *
 * ── Depthwise convolution semantics ────────────────────────────────────────
 *
 *   Each input channel IC has its own set of (channels_multiplier) filters.
 *   Output channels: OC = IC * channels_multiplier
 *   Output channel index: oc = ic * channels_multiplier + cm
 *
 * ── Kernel layout [KH, KW, OC] (row-major) ────────────────────────────────
 *
 *   kernel[ kh * KW * OC + kw * OC + oc ]
 *   where oc = ic * CM + cm
 *
 * ── Zero-bias caller convention (identical to std conv2d) ─────────────────
 *
 *   input_zero_bias  = -input_zero_point   (caller passes NEGATED zero-point)
 *   kernel_zero_bias = 0                   (sym8s: always 0)
 *   out_zero_bias    = +output_zero_point  (added after requantization)
 *
 * ── partial_sum output ─────────────────────────────────────────────────────
 *
 *   p_partial_sum[oh * OW * OC + ow * OC + oc] =
 *       bias[oc] + sum_kh_kw( (inp+izp) * w )
 *   Captured BEFORE requantize.  Same semantics as conv2d cmodel.
 *   May be NULL (skipped when NULL).
 */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <limits.h>

#include "conv2d_depthwise_cmodel.h"

/* =========================================================================
 * Requant core — verbatim from conv2d_std_cim_cmodel.c
 * (quantize_change + shift_change path, always active)
 * ========================================================================= */

#define quantize_change
#define shift_change
#define LEFT_SHIFT(shift)  ((shift) > 0 ? (shift) : 0)
#define RIGHT_SHIFT(shift) ((shift) > 0 ? 0 : -(shift))

typedef union {
    int64_t long_long;
    struct {
        uint32_t low;
        int32_t  high;
    } word;
} nn_long_long;

static inline int64_t nn_divide_by_power_of_two(int32_t dividend,
                                                 int32_t exponent)
{
    int64_t result = 0;

    if (exponent <= 0) {                /* exponent < 0 → right shift */
        exponent = -exponent;
        if (exponent >= 32) {
            /* Large right-shift: any int32 value rounds to 0 (or ±1).
             * remainder_mask would span all 32 bits; dividend itself is
             * the remainder. Rounding: ties-away-from-zero. */
            int64_t remainder_mask = (1LL << exponent) - 1;
            int64_t remainder      = (int64_t)dividend & remainder_mask;
            result = (int64_t)dividend >> exponent;  /* 0 or sign-extension */
            int64_t threshold = remainder_mask >> 1;
            if (result < 0)
                threshold++;
            if (remainder > threshold)
                result++;
        } else {
            const int32_t remainder_mask = (1 << exponent) - 1;
            int32_t remainder = remainder_mask & dividend;

            result = dividend >> exponent;

            int32_t threshold = remainder_mask >> 1;
            if (result < 0)
                threshold++;

            if (remainder > threshold)
                result++;
        }
    } else {                            /* exponent > 0 → left shift */
        result = dividend * (1LL << exponent);
    }

    return result;
}

static inline int32_t nn_doubling_high_mult_no_sat(int32_t m1, int32_t m2)
{
    int32_t    result = 0;
    nn_long_long mult;

    mult.word.low  = 0;
    mult.word.high = 0;

    mult.long_long = mult.long_long + (int64_t)m1 * m2;

    result = (int32_t)(mult.long_long >> 31);

    return result;
}

static inline int64_t nn_requantize(int32_t val, int32_t multiplier,
                                     int32_t shift)
{
    return nn_divide_by_power_of_two(
               nn_doubling_high_mult_no_sat(val, multiplier),
               shift);
}

static inline int32_t clamp_i32(int64_t x, int32_t lo, int32_t hi)
{
    if (x < lo) return lo;
    if (x > hi) return hi;
    return (int32_t)x;
}

/* =========================================================================
 * cmodel_conv2d_depthwise_sym8sxasym8s
 *
 * NNLib-compatible API (p_kernel and p_inp SWAPPED vs std conv2d):
 *   p_out         — output  [OH, OW, OC]     int8,  NHWC
 *   p_kernel      — kernel  [KH, KW, OC]     int8,  sym8s
 *   p_inp         — input   [IH, IW, IC]     int8,  NHWC
 *   p_bias        — bias    [OC]             int32, can be NULL
 *   p_partial_sum — pre-requant acc [OH*OW*OC] int32, can be NULL
 * ========================================================================= */
void cmodel_conv2d_depthwise_sym8sxasym8s(
    int8_t        * __restrict__ p_out,
    const int8_t  * __restrict__ p_kernel,      /* [KH, KW, OC] */
    const int8_t  * __restrict__ p_inp,         /* [IH, IW, IC] */
    const int32_t * __restrict__ p_bias,
    const int32_t * __restrict__ p_out_multiplier,
    const int32_t * __restrict__ p_out_shift,
    int input_height,
    int input_width,
    int input_channels,
    int kernel_height,
    int kernel_width,
    int channels_multiplier,
    int dilation_height,
    int dilation_width,
    int stride_height,
    int stride_width,
    int pad_height,
    int pad_width,
    int output_height,
    int output_width,
    int input_zero_bias,        /* = -input_zero_point  (negated) */
    int out_zero_bias,          /* = +output_zero_point           */
    int activation_min,
    int activation_max,
    int32_t * __restrict__ p_partial_sum)  /* [OH*OW*OC], can be NULL */
{
    const int output_channels = input_channels * channels_multiplier;

    for (int oh = 0; oh < output_height; ++oh) {
        for (int ow = 0; ow < output_width; ++ow) {
            for (int oc = 0; oc < output_channels; ++oc) {

                const int ic = oc / channels_multiplier;

                /* Initialize accumulator with bias */
                int32_t acc = p_bias ? p_bias[oc] : 0;

                /* Convolve over kernel spatial extent */
                for (int kh = 0; kh < kernel_height; ++kh) {
                    const int ih = oh * stride_height
                                   + kh * dilation_height
                                   - pad_height;
                    if ((unsigned)ih >= (unsigned)input_height)
                        continue;

                    for (int kw = 0; kw < kernel_width; ++kw) {
                        const int iw = ow * stride_width
                                       + kw * dilation_width
                                       - pad_width;
                        if ((unsigned)iw >= (unsigned)input_width)
                            continue;

                        int32_t inp_v = (int32_t)p_inp[ih * input_width * input_channels
                                                        + iw * input_channels
                                                        + ic]
                                        + input_zero_bias;

                        int32_t w_v = (int32_t)p_kernel[kh * kernel_width * output_channels
                                                         + kw * output_channels
                                                         + oc];

                        acc += inp_v * w_v;
                    }
                }

                /* Capture partial sum BEFORE requantize */
                if (p_partial_sum) {
                    p_partial_sum[oh * output_width * output_channels
                                  + ow * output_channels
                                  + oc] = acc;
                }

                /* Requantize (verbatim nn_requantize path) */
                int64_t out_v = nn_requantize(acc,
                                              p_out_multiplier[oc],
                                              p_out_shift[oc]);

                /* Add output zero-bias and clamp */
                out_v += out_zero_bias;
                p_out[oh * output_width * output_channels
                      + ow * output_channels
                      + oc] = (int8_t)clamp_i32(out_v, activation_min, activation_max);
            }
        }
    }
}
