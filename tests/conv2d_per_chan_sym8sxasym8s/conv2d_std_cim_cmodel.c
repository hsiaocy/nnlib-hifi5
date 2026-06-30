/*
 * C Model of xa_nn_conv2d_std_per_chan_sym8sxasym8s
 *
 * This is a fully self-contained implementation with no dependency on HiFi5 NNLib.
 * It matches the functional behavior of the original Cadence NNLib function:
 *  - NHWC input/output tensor layout
 *  - Per-channel quantization (separate multiplier & shift for each output channel)
 *  - Zero-padding (when the convolution window goes out of bounds, treat as 0)
 *  - Optional bias addition per output channel
 *  - Input/kernel/output zero bias adjustments
 *  - Activation clamping to a min/max range
 *
 * Renamed to cmodel_conv2d_per_chan_sym8sxasym8s to avoid symbol collision
 * with the HiFi5 NNLib function when linked in the same test binary.
 *
 * Zero-bias caller convention (matches HiFi5 NNLib):
 *   input_zero_bias  = -input_zero_point   (caller passes NEGATED zero-point)
 *   kernel_zero_bias = 0                   (sym8s kernel: always symmetric)
 *   output_zero_bias = +output_zero_point  (added after requantization)
 *
 * Parameter naming follows the original NNLib API for easy hardware mapping.
 */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <limits.h>

#include "conv2d_std_cim_cmodel.h"

#define quantize_change
#define LEFT_SHIFT(shift)  ((shift) > 0 ? (shift) : 0)
#define RIGHT_SHIFT(shift) ((shift) > 0 ? 0 : -(shift))
#define shift_change
#define PARALLE_CAL

/*
 * requantize_rnd:
 *   Performs fixed-point requantization of the accumulator.
 *   The accumulator (int32_t) is multiplied by the given per-channel multiplier,
 *   then right-shifted by the given shift value with rounding.
 *   If shift < 0, the value is left-shifted (rare in quantized inference).
 */

/* =========================================================================
 * #ifdef quantize_change path (active — quantize_change is always defined)
 * ========================================================================= */
#ifdef quantize_change

typedef union {
    int64_t long_long;
    struct {
        uint32_t low;
        int32_t  high;
    } word;
} nn_long_long;

/* -------------------------------------------------------------------------
 * nn_divide_by_power_of_two
 * ------------------------------------------------------------------------- */
#ifdef shift_change
/* shift_change variant: exponent can be positive (left-shift) or
 * negative (right-shift).  Returns int64_t.                              */
static inline int64_t nn_divide_by_power_of_two(int32_t dividend,
                                                 int32_t exponent)
{
    int64_t result = 0;

    if (exponent <= 0) {                /* exponent < 0 → right shift */
        exponent = -exponent;
        const int32_t remainder_mask = (1 << exponent) - 1;
        int32_t remainder = remainder_mask & dividend;

        result = dividend >> exponent;

        int32_t threshold = remainder_mask >> 1;
        if (result < 0)
            threshold++;

        if (remainder > threshold)
            result++;
    } else {                            /* exponent > 0 → left shift */
        result = dividend * (1LL << exponent);
    }

    return result;
}

#else  /* !shift_change */

static inline int32_t nn_divide_by_power_of_two(int32_t dividend,
                                                 int32_t exponent)
{
    int32_t result          = 0;
    const int32_t remainder_mask = (1 << exponent) - 1;
    int32_t remainder       = remainder_mask & dividend;

    result = dividend >> exponent;
    int32_t threshold = remainder_mask >> 1;

    if (result < 0)
        threshold++;

    if (remainder > threshold)
        result++;

    return result;
}
#endif /* shift_change */

/* -------------------------------------------------------------------------
 * nn_doubling_high_mult_no_sat
 * Computes (int32_t)((m1 * m2) >> 31) using a 64-bit union.
 * ------------------------------------------------------------------------- */
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

/* -------------------------------------------------------------------------
 * nn_requantize
 * ------------------------------------------------------------------------- */
#ifdef shift_change
static inline int64_t nn_requantize(int32_t val, int32_t multiplier,
                                     int32_t shift)
{
    return nn_divide_by_power_of_two(
               nn_doubling_high_mult_no_sat(val, multiplier),
               shift);
}

#else  /* !shift_change */

static inline int32_t nn_requantize(int32_t val, int32_t multiplier,
                                     int32_t shift)
{
    return nn_divide_by_power_of_two(
               nn_doubling_high_mult_no_sat(
                   val * (1 << LEFT_SHIFT(shift)), multiplier),
               RIGHT_SHIFT(shift));
}
#endif /* shift_change */

#else  /* !quantize_change */
/* =========================================================================
 * Fallback path when quantize_change is NOT defined
 * ========================================================================= */
static inline int32_t requantize_rnd(int32_t acc, int32_t multiplier,
                                      int32_t shift)
{
    int64_t prod = (int64_t)acc * (int64_t)multiplier;
    if (shift < 0) {
        return (int32_t)(prod << (-shift));
    } else {
        int64_t rnd = (int64_t)1 << (shift - 1);   /* rounding offset */
        return (int32_t)((prod + rnd) >> shift);
    }
}
#endif /* quantize_change */

/* =========================================================================
 * clamp_i32
 *   Clamps a value into [lo, hi].
 *   shift_change variant accepts int64_t input to handle the wider
 *   intermediate result from nn_requantize.
 * ========================================================================= */
#ifdef shift_change
static inline int32_t clamp_i32(int64_t x, int32_t lo, int32_t hi)
{
    if (x < lo) return lo;
    if (x > hi) return hi;
    return (int32_t)x;
}
#else
static inline int32_t clamp_i32(int32_t x, int32_t lo, int32_t hi)
{
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}
#endif

/* =========================================================================
 * cmodel_conv2d_per_chan_sym8sxasym8s
 *
 * Main Conv2D loop.  Renamed from xa_nn_conv2d_std_per_chan_sym8sxasym8s
 * to avoid linker symbol collision with the HiFi5 NNLib implementation.
 *
 * Arguments: see conv2d_std_cmodel.h for full documentation.
 * ========================================================================= */
void cmodel_conv2d_per_chan_sym8sxasym8s(
    int8_t       * __restrict__ p_out,
    const int8_t * __restrict__ p_inp,
    const int8_t * __restrict__ p_kernel,
    const int32_t* __restrict__ p_bias,
    const int32_t* __restrict__ p_out_multiplier,
    const int32_t* __restrict__ p_out_shift,
    int input_height,
    int input_width,
    int input_channels,
    int kernel_height,
    int kernel_width,
    int stride_height,
    int stride_width,
    int pad_height,
    int pad_width,
    int output_height,
    int output_width,
    int output_channels,
    int input_zero_bias,
    int kernel_zero_bias,
    int output_zero_bias,
    int activation_min,
    int activation_max,
    int out_stride)
{
    for (int oh = 0; oh < output_height; ++oh) {
        for (int ow = 0; ow < output_width; ++ow) {
            for (int oc = 0; oc < output_channels; ++oc) {

                /* Initialize accumulator with bias */
                int32_t acc = p_bias ? p_bias[oc] : 0;

                /* Loop over kernel height */
                for (int kh = 0; kh < kernel_height; ++kh) {
                    int ih = oh * stride_height + kh - pad_height;
                    if ((unsigned)ih >= (unsigned)input_height)
                        continue;                          /* zero-padding row */

                    /* Loop over kernel width */
                    for (int kw = 0; kw < kernel_width; ++kw) {
                        int iw = ow * stride_width + kw - pad_width;
                        if ((unsigned)iw >= (unsigned)input_width)
                            continue;                      /* zero-padding col */

                        /* Base offsets for input and weight */
                        int in_base = (ih * input_width + iw) * input_channels;
                        int w_base  = ((oc * kernel_height + kh) * kernel_width
                                       + kw) * input_channels;

                        /* Accumulate over input channels */
                        for (int ic = 0; ic < input_channels; ++ic) {
                            int32_t inp = (int32_t)p_inp[in_base + ic]
                                          + input_zero_bias;
                            int32_t w   = (int32_t)p_kernel[w_base + ic]
                                          + kernel_zero_bias;
                            acc += inp * w;
                        }
                    }
                }

                /* Requantize accumulated value */
#ifdef quantize_change
  #ifdef shift_change
                int64_t out_v = nn_requantize(acc,
                                              p_out_multiplier[oc],
                                              p_out_shift[oc]);
  #else  /* !shift_change */
                int32_t out_v = nn_requantize(acc,
                                              p_out_multiplier[oc],
                                              p_out_shift[oc]);
  #endif /* shift_change */
#else  /* !quantize_change */
                int32_t out_v = requantize_rnd(acc,
                                               p_out_multiplier[oc],
                                               p_out_shift[oc]);
#endif /* quantize_change */

                /* Add output zero bias */
                out_v += output_zero_bias;

                /* Activation clamp */
                out_v = clamp_i32(out_v, activation_min, activation_max);

                /* Store as int8 — out_stride = OW * OC for packed NHWC */
                p_out[oh * out_stride + ow * output_channels + oc] =
                    (int8_t)out_v;
            }
        }
    }
}
