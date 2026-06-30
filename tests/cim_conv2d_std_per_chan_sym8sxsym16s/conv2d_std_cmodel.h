/*******************************************************************************
 * conv2d_std_cmodel.h
 *
 * C reference model for cim_conv2d_std_per_chan_sym8sxsym16s.
 *
 * Functional target:
 *   xa_nn_conv2d_std_per_chan_sym8sxsym16s
 *
 * Tensor layout:
 *   input  : NHWC int16_t
 *   kernel : [OC, KH, KW, IC] int8_t
 *   output : NHWC int16_t
 *   bias   : per-output-channel int64_t
 *
 * sym8sxsym16s zero-bias convention:
 *   input_zero_bias  = 0
 *   kernel_zero_bias = 0
 *   output_zero_bias = 0
 ******************************************************************************/

#ifndef CONV2D_STD_CMODEL_H
#define CONV2D_STD_CMODEL_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

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
    int out_stride
);

#ifdef __cplusplus
}
#endif

#endif /* CONV2D_STD_CMODEL_H */
