#include <stdio.h>
#include <stdlib.h>
#include <string.h>


int main(void)
{
    int out_size = out_height * out_width * out_channels;
    int8_t *p_out = (int8_t *)malloc(out_size);
    if (!p_out) return -1;

    int scratch_size = xa_nn_conv2d_std_getsize(
        input_height, input_width, input_channels,
        kernel_height, kernel_width, input_channels,
        y_stride, y_padding, x_stride, x_padding,
        out_height, out_width, out_channels,
        PREC_ASYM8S, PREC_SYM8S, 1, 1, out_data_format);
    if (scratch_size < 0) return -2;

    void *p_scratch = malloc(scratch_size + 16);
    if (!p_scratch) return -3;
    p_scratch = (void *)(((unsigned long)p_scratch + 15) & ~15UL);

    int ret = cim_conv2d_std_per_chan_sym8sxasym8s(
        p_out, input, kernel, bias,
        input_height, input_width, input_channels,
        kernel_height, kernel_width, out_channels,
        x_stride, y_stride, x_padding, y_padding,
        out_height, out_width, input_zero_bias,
        out_multiplier, out_shift, out_zero_bias,
        out_data_format, p_scratch,
        out_activation_min, out_activation_max, 
        NULL);

    return ret;
}
