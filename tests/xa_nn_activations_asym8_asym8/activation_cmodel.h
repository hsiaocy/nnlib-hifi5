/*******************************************************************************
 * activation_cmodel.h
 *
 * C reference model for the asym8->asym8 vector activation family.
 *
 * Implemented (gcc-verifiable, bit-exact):
 *   - Generic clamp  (activation_min, activation_max)
 *   - ReLU           (clamp to [relu_min, 127], relu_min = zero_point)
 *   - ReLU6          (clamp to [relu_min, relu6_max], both derived from zp+scale)
 *
 * NOT implemented here (require NNLib LUT for bit-exactness, out of gcc scope):
 *   - Sigmoid        (STUB — see activation_cmodel.c)
 *   - Tanh           (STUB — see activation_cmodel.c)
 *
 * NOTE: This is a purely elementwise op.  There is NO MAC and NO partial_sum.
 *       The .in_out fixture format therefore omits kernel, bias, out_multiplier,
 *       out_shift, and partial_sum fields (see gen_activation_patterns.py).
 *
 * Caller convention (consistent with conv2d cmodel and NNLib):
 *   input_zero_point   — raw quantization zero-point of the input tensor
 *   output_zero_point  — raw quantization zero-point of the output tensor
 *   activation_min     — lower clamp bound (int8 quantized value, e.g. -128)
 *   activation_max     — upper clamp bound (int8 quantized value, e.g.  127)
 *
 * For ReLU  : caller pre-computes activation_min = quantized(0) = input_zero_point
 *             and activation_max = 127.
 * For ReLU6 : caller pre-computes both bounds from the float activation range.
 * For clamp : caller supplies arbitrary int8 bounds.
 ******************************************************************************/

#ifndef ACTIVATION_CMODEL_H
#define ACTIVATION_CMODEL_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * cmodel_activation_clamp_asym8
 *
 *   Elementwise clamp of an asym8 (int8) tensor to [activation_min, activation_max].
 *   This is the core primitive used by ReLU, ReLU6, and the generic clamp activation.
 *
 *   p_out          Output buffer (int8), length num_elements
 *   p_inp          Input  buffer (int8), length num_elements
 *   num_elements   Number of elements to process
 *   activation_min Lower clamp bound (int8 quantized value)
 *   activation_max Upper clamp bound (int8 quantized value)
 */
void cmodel_activation_clamp_asym8(
    int8_t       * __restrict__ p_out,
    const int8_t * __restrict__ p_inp,
    int num_elements,
    int activation_min,
    int activation_max
);

#ifdef __cplusplus
}
#endif

#endif /* ACTIVATION_CMODEL_H */
