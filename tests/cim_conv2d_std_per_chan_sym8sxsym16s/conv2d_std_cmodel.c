/*
 * cim_conv2d_std_per_chan_sym8sxsym16s
 *
 * Self-contained C reference model for the HiFi5 NNLib function:
 *   xa_nn_conv2d_std_per_chan_sym8sxsym16s
 *
 * This model uses NHWC input/output layout and kernel layout
 * [output_channels, kernel_height, kernel_width, input_channels].
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

#include "conv2d_std_cmodel.h"

static inline int64_t div_by_power_of_two_round(int64_t dividend, int32_t exponent)
{
    if (exponent <= 0) {
        int32_t left_shift = -exponent;
        if (left_shift >= 31) {
            return (dividend >= 0) ? INT64_MAX : INT64_MIN;
        }
        return dividend * (1LL << left_shift);
    }

    if (exponent >= 63) {
        return 0;
    }

    int64_t mask = (1LL << exponent) - 1LL;
    int64_t remainder = dividend & mask;
    int64_t result = dividend >> exponent;
    int64_t threshold = mask >> 1;

    if (result < 0) {
        threshold++;
    }
    if (remainder > threshold) {
        result++;
    }
    return result;
}

static inline int64_t doubling_high_mult_64(int64_t m1, int32_t m2)
{
#if defined(__SIZEOF_INT128__)
    __int128 prod = (__int128)m1 * (__int128)m2;
    return (int64_t)(prod >> 31);
#else
    return (int64_t)(((long double)m1 * (long double)m2) / 2147483648.0L);
#endif
}

static inline int64_t requantize_64(int64_t val, int32_t multiplier, int32_t shift)
{
    return div_by_power_of_two_round(doubling_high_mult_64(val, multiplier), shift);
}

static inline int16_t clamp_i16(int64_t x, int32_t lo, int32_t hi)
{
    if (x < lo) return (int16_t)lo;
    if (x > hi) return (int16_t)hi;
    return (int16_t)x;
}

void cim_conv2d_std_per_chan_sym8sxsym16s(
    int16_t       * __restrict__ p_out,
    const int16_t * __restrict__ p_inp,
    const int8_t  * __restrict__ p_kernel,
    const int64_t * __restrict__ p_bias,
    const int32_t * __restrict__ p_out_multiplier,
    const int32_t * __restrict__ p_out_shift,
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
    if (!p_out || !p_inp || !p_kernel || !p_out_multiplier || !p_out_shift) {
        fprintf(stderr, "[cmodel] NULL pointer argument\n");
        return;
    }

    /* sym8sxsym16s is symmetric input/output. Keep these checks non-fatal so
     * test benches can still run, but report mismatched caller conventions. */
    if (input_zero_bias != 0 || kernel_zero_bias != 0 || output_zero_bias != 0) {
        fprintf(stderr,
                "[cmodel] warning: sym8sxsym16s expects zero biases 0/0/0, got %d/%d/%d\n",
                input_zero_bias, kernel_zero_bias, output_zero_bias);
    }

    if (out_stride <= 0) {
        out_stride = output_width * output_channels;
    }

    if (activation_min < INT16_MIN) activation_min = INT16_MIN;
    if (activation_max > INT16_MAX) activation_max = INT16_MAX;

    for (int oh = 0; oh < output_height; ++oh) {
        for (int ow = 0; ow < output_width; ++ow) {
            for (int oc = 0; oc < output_channels; ++oc) {
                int64_t acc = p_bias ? p_bias[oc] : 0;

                for (int kh = 0; kh < kernel_height; ++kh) {
                    int ih = oh * stride_height + kh - pad_height;
                    if ((unsigned)ih >= (unsigned)input_height) {
                        continue;
                    }

                    for (int kw = 0; kw < kernel_width; ++kw) {
                        int iw = ow * stride_width + kw - pad_width;
                        if ((unsigned)iw >= (unsigned)input_width) {
                            continue;
                        }

                        int in_base = (ih * input_width + iw) * input_channels;
                        int w_base = ((oc * kernel_height + kh) * kernel_width + kw) * input_channels;

                        for (int ic = 0; ic < input_channels; ++ic) {
                            int32_t inp = (int32_t)p_inp[in_base + ic] + input_zero_bias;
                            int32_t ker = (int32_t)p_kernel[w_base + ic] + kernel_zero_bias;
                            acc += (int64_t)inp * (int64_t)ker;
                        }
                    }
                }

                int64_t out_v = requantize_64(acc, p_out_multiplier[oc], p_out_shift[oc]);
                out_v += output_zero_bias;

                p_out[oh * out_stride + ow * output_channels + oc] =
                    clamp_i16(out_v, activation_min, activation_max);
            }
        }
    }
}
