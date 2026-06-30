/*******************************************************************************
 * conv2d_std_cmodel.h
 *
 * Header for the C reference model of xa_nn_conv2d_std_per_chan_sym8sxasym8s.
 *
 * The function is renamed to cmodel_conv2d_per_chan_sym8sxasym8s to avoid
 * linker symbol collision with the HiFi5 NNLib implementation that is linked
 * into the same test binary.
 *
 * Caller convention for zero-bias parameters (matches HiFi5 NNLib):
 *   input_zero_bias  = -input_zero_point   (pass the NEGATED zero-point)
 *   kernel_zero_bias = 0                   (sym8s kernel: always 0)
 *   output_zero_bias = +output_zero_point  (NOT negated — added after quant)
 ******************************************************************************/

#ifndef CONV2D_STD_CMODEL_H
#define CONV2D_STD_CMODEL_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * cmodel_conv2d_per_chan_sym8sxasym8s
 *
 *   C reference model implementing the same algorithm as the HiFi5 NNLib
 *   xa_nn_conv2d_std_per_chan_sym8sxasym8s function.
 *
 * Parameters
 * ----------
 *   p_out            Output tensor [OH, OW, OC] NHWC layout (int8)
 *   p_inp            Input  tensor [IH, IW, IC] NHWC layout (int8)
 *   p_kernel         Kernel weights [OC, KH, KW, IC] (int8, sym8s)
 *   p_bias           Per-channel bias [OC] (int32, can be NULL)
 *   p_out_multiplier Per-channel requant multiplier [OC] (Q31)
 *   p_out_shift      Per-channel requant shift [OC]
 *                      > 0 → left-shift exponent
 *                      < 0 → right-shift exponent (common case)
 *   input_height/width/channels  — input tensor shape
 *   kernel_height/width          — kernel spatial shape
 *   stride_height/width          — convolution stride (y/x)
 *   pad_height/width             — top/left zero-padding
 *   output_height/width/channels — output tensor shape
 *   input_zero_bias  -zero_point of input  (pass as NEGATIVE zero-point)
 *   kernel_zero_bias always 0 for sym8s
 *   output_zero_bias +zero_point of output (added after requantization)
 *   activation_min/max           — clamping range (int8 values)
 *   out_stride       Row stride of output in elements = out_width*out_channels
 *                    for packed NHWC
 */
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
    int out_stride
);

#ifdef __cplusplus
}
#endif

#endif /* CONV2D_STD_CMODEL_H */
