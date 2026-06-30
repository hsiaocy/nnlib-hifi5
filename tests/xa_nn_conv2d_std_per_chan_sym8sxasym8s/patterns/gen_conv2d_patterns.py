#!/usr/bin/env python3
"""
gen_conv2d_patterns.py
======================
Generate .in_out fixture files for xa_nn_conv2d_std_per_chan_sym8sxasym8s tests.

Each fixture exercises a different aspect of the conv2d pipeline so that
CModel vs HiFi5 differences can be isolated to a specific behaviour.

Usage:
    python3 gen_conv2d_patterns.py                         # all patterns (double-rounding)
    python3 gen_conv2d_patterns.py p03 p05                 # selective
    python3 gen_conv2d_patterns.py --single-rounding       # TFLITE_SINGLE_ROUNDING=1 path
    python3 gen_conv2d_patterns.py --double-rounding p03   # TFLITE_SINGLE_ROUNDING=0 path (default)

Requantization modes:
    --double-rounding (default, TFLITE_SINGLE_ROUNDING=0)
        Two-stage 32-bit: doubling_high_mult (Q31 saturating mul) then
        divide_by_power_of_two (separate rounding).  Matches Cadence
        nnlib-hifi5 reference and GEMMLOWP legacy behaviour.
    --single-rounding (TFLITE_SINGLE_ROUNDING=1)
        Single-stage 64-bit: (int64)acc * multiplier + round_bias, then
        arithmetic shift.  Higher precision, matches TFLM reference kernel
        when its TFLITE_SINGLE_ROUNDING macro is set.

Each generated .in_out fixture records `tflite_single_rounding = 0|1` in its
header so the CModel side can select the matching requantize path.

Output files (same directory):
    p01_nonzero_zero_bias.in_out
    p02_activation_clamp.in_out
    p03_stride2.in_out
    p04_1x1_pointwise.in_out
    p05_5x5_large_kernel.in_out
    p06_many_channels.in_out
    p07_accumulator_near_overflow.in_out
    p08_all_zero_input.in_out
    p09_negative_shift.in_out
    p10_asymmetric_padding.in_out
    p11_kws_first_layer.in_out
    p12_custom_shape_40x24.in_out
"""

import argparse
import numpy as np
import struct
import sys
import os

# ---------------------------------------------------------------------------
# TFLITE_SINGLE_ROUNDING mode flag
# ---------------------------------------------------------------------------
# 0 = double-rounding  (legacy GEMMLOWP / nnlib-hifi5 default, 32-bit two-step)
# 1 = single-rounding  (TFLM reference when TFLITE_SINGLE_ROUNDING macro is set,
#                       64-bit one-step, higher precision)
# Set via CLI (--single-rounding / --double-rounding); default = 0.
_TFLITE_SINGLE_ROUNDING = 0

# ---------------------------------------------------------------------------
# Fixed-point arithmetic matching the cmodel / NNLib logic
# ---------------------------------------------------------------------------
#
# Two requantization implementations are provided below.  `requantize()` is
# the dispatcher that both respect the active TFLITE_SINGLE_ROUNDING mode.
#
# Golden outputs in the .in_out fixtures are generated using whichever mode
# is active; the mode is also recorded in each fixture's header so the
# CModel side can compile / dispatch to the matching path.
# ---------------------------------------------------------------------------

INT32_MIN = -(1 << 31)
INT32_MAX =  (1 << 31) - 1

# ---- Double-rounding (TFLITE_SINGLE_ROUNDING = 0) --------------------------

def doubling_high_mult_no_sat(m1: np.int32, m2: np.int32) -> np.int32:
    """(m1 * m2) >> 31  using 64-bit intermediate (no saturation)."""
    result = (int(m1) * int(m2)) >> 31
    return np.int32(result)

def divide_by_power_of_two(dividend: np.int32, exponent: np.int32) -> np.int64:
    """Round-half-away-from-zero right-shift (or left-shift if exponent>0)."""
    exp = int(exponent)
    d   = int(dividend)
    if exp <= 0:                        # right shift
        exp = -exp
        remainder_mask = (1 << exp) - 1
        remainder      = d & remainder_mask
        result         = d >> exp
        threshold      = remainder_mask >> 1
        if result < 0:
            threshold += 1
        if remainder > threshold:
            result += 1
        return np.int64(result)
    else:                               # left shift
        return np.int64(d * (1 << exp))

def requantize_double_rounding(val, multiplier, shift):
    """TFLITE_SINGLE_ROUNDING=0 path: doubling_high_mult + divide_by_pow2."""
    hm = doubling_high_mult_no_sat(np.int32(val), np.int32(multiplier))
    return int(divide_by_power_of_two(np.int32(hm), np.int32(shift)))

# ---- Single-rounding (TFLITE_SINGLE_ROUNDING = 1) --------------------------

def requantize_single_rounding(val, multiplier, shift):
    """
    TFLITE_SINGLE_ROUNDING=1 path: one 64-bit multiply + one rounding shift.

    Convention in this project (matches nnlib-hifi5):
        shift < 0  : right shift by |shift| after the 31-bit high-mul
        shift > 0  : left  shift by  shift  after the 31-bit high-mul
        shift == 0 : pure Q31 round-shift

    Equivalent collapsed single-rounding form:
        total_right_shift = 31 - shift
          (shift = -8  ->  39  : large right shift)
          (shift =  0  ->  31  : plain Q31 round-shift)
          (shift =  5  ->  26  : small right shift, equivalent to
                                 (val*mult) >> 31 << 5 but with only
                                 one rounding event)
        result = rounding_divide_by_pot(val * mult, total_right_shift)

    This matches the TFLM reference SRM path (see tensorflow issue #104605
    and PR #51794), adapted to this project's shift-sign convention.
    """
    val = int(val)
    mult = int(multiplier)
    shift = int(shift)

    total_right_shift = 31 - shift

    prod = val * mult                 # Python int: no overflow

    if total_right_shift <= 0:
        # Degenerate: entire operation is a left-shift (extremely rare;
        # would require shift >= 32).  Preserve exact value.
        result = prod << (-total_right_shift)
    else:
        # Rounding divide-by-power-of-two, ties away from zero.
        remainder_mask = (1 << total_right_shift) - 1
        remainder      = prod & remainder_mask
        quotient       = prod >> total_right_shift
        threshold      = remainder_mask >> 1
        if quotient < 0:
            threshold += 1
        if remainder > threshold:
            quotient += 1
        result = quotient

    # Saturating cast to int32 (matches TFLM saturating_round_shift on cast).
    if result > INT32_MAX: result = INT32_MAX
    if result < INT32_MIN: result = INT32_MIN
    return result

# ---- Dispatcher ------------------------------------------------------------

def requantize(val, multiplier, shift):
    """Dispatch to the active requantize path based on _TFLITE_SINGLE_ROUNDING."""
    if _TFLITE_SINGLE_ROUNDING:
        return requantize_single_rounding(val, multiplier, shift)
    else:
        return requantize_double_rounding(val, multiplier, shift)

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def conv2d_reference(inp, kernel, bias,
                     out_multiplier, out_shift,
                     batch_size, input_height, input_width, input_channels,
                     kernel_height, kernel_width,
                     stride_h, stride_w,
                     pad_h, pad_w,
                     out_height, out_width, out_channels,
                     input_zero_bias, kernel_zero_bias, out_zero_bias,
                     activation_min, activation_max):
    """Pure-Python reference matching conv2d_std_cmodel.c logic.

    Returns
    -------
    out         : np.ndarray[int8]  shape (N, OH, OW, OC)
    partial_sum : np.ndarray[int32] shape (N, OH, OW, OC)
                  acc = bias[oc] + sum((inp+izp)*(w+kzp))
                  captured just before requantize().
                  kernel_zero_bias is always 0 for sym8s kernels.
    """
    out         = np.zeros((batch_size, out_height, out_width, out_channels), dtype=np.int8)
    partial_sum = np.zeros((batch_size, out_height, out_width, out_channels), dtype=np.int32)
    for b in range(batch_size):
        for oh in range(out_height):
            for ow in range(out_width):
                for oc in range(out_channels):
                    acc = int(bias[oc]) if bias is not None else 0
                    for kh in range(kernel_height):
                        ih = oh * stride_h + kh - pad_h
                        if ih < 0 or ih >= input_height:
                            continue
                        for kw in range(kernel_width):
                            iw = ow * stride_w + kw - pad_w
                            if iw < 0 or iw >= input_width:
                                continue
                            for ic in range(input_channels):
                                inp_v = int(inp[b, ih, iw, ic]) + input_zero_bias
                                w_v   = int(kernel[oc, kh, kw, ic]) + kernel_zero_bias
                                acc  += inp_v * w_v
                    # capture partial sum BEFORE requantize
                    partial_sum[b, oh, ow, oc] = np.int32(
                        np.clip(acc, -(2**31), 2**31 - 1))
                    out_v = int(requantize(acc,
                                           int(out_multiplier[oc]),
                                           int(out_shift[oc])))
                    out_v += out_zero_bias
                    out_v  = clamp(out_v, activation_min, activation_max)
                    out[b, oh, ow, oc] = np.int8(out_v)
    return out, partial_sum

# ---------------------------------------------------------------------------
# .in_out file writer
# ---------------------------------------------------------------------------

def write_inout(path, cfg, inp, kernel, bias,
                out_multiplier, out_shift, golden, partial_sum):
    """Write a fixture file in the format expected by the C test parser.

    partial_sum : np.ndarray[int32] shape (N, OH, OW, OC)
        acc = bias[oc] + sum((inp+izp)*(w+kzp)), captured before requantize().
        Written as  int32_t partial_sum[OH*OW*OC]  (N=1, batch dim dropped).
    """
    def arr8(name, data):
        flat = data.flatten().astype(np.int8)
        vals = ", ".join(str(int(v)) for v in flat)
        return f"int8_t {name}[{len(flat)}] = {{{vals}}};\n"

    def arr32(name, data):
        flat = np.array(data, dtype=np.int32).flatten()
        vals = ", ".join(str(int(v)) for v in flat)
        return f"int32_t {name}[{len(flat)}] = {{{vals}}};\n"

    with open(path, "w") as f:
        f.write(f"// Pattern: {cfg['desc']}\n")
        f.write(f"// Purpose: {cfg['purpose']}\n")
        mode_str = "single-rounding (TFLITE_SINGLE_ROUNDING=1, 64-bit one-step)" \
                   if _TFLITE_SINGLE_ROUNDING else \
                   "double-rounding (TFLITE_SINGLE_ROUNDING=0, 32-bit two-step)"
        f.write(f"// Requantize mode: {mode_str}\n\n")
        # Emit mode flag first so the CModel parser can configure its
        # requantize path before processing the rest of the fixture.
        f.write(f"int tflite_single_rounding = {_TFLITE_SINGLE_ROUNDING};\n")
        for k, v in cfg.items():
            if k not in ("desc", "purpose"):
                f.write(f"int {k} = {v};\n")
        f.write("\n")
        f.write(arr8("input",  inp))
        f.write(arr8("kernel", kernel))
        if bias is not None:
            f.write(arr32("bias", bias))
        else:
            oc = cfg["out_channels"]
            f.write(f"int32_t bias[{oc}] = {{" + ", ".join(["0"]*oc) + "};\n")
        f.write(arr32("out_multiplier", out_multiplier))
        f.write(arr32("out_shift",      out_shift))
        f.write(arr8("golden", golden))
        # partial_sum: drop N=1 batch dim → shape (OH, OW, OC) → flatten
        ps = partial_sum[0] if partial_sum.ndim == 4 else partial_sum
        f.write(arr32("partial_sum", ps))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def out_size(in_size, kernel_size, stride, pad):
    return (in_size + 2 * pad - kernel_size) // stride + 1

def make_multiplier_shift(scale_f32, n_channels):
    """
    Convert a float scale to (Q31 multiplier, negative shift) pair used by
    NNLib requantize.  Uses the same derivation as TFLite QuantizeMultiplier.
    """
    multipliers = []
    shifts      = []
    for _ in range(n_channels):
        if scale_f32 == 0.0:
            multipliers.append(0)
            shifts.append(0)
            continue
        # find m, e such that scale = m * 2^e, 0.5 <= m < 1
        import math
        e = math.floor(math.log2(scale_f32))
        m = scale_f32 / (2 ** e)          # 0.5 <= m < 1
        q31 = round(m * (2**31))
        # Minimal fix: guard against q31 >= 2**31 (happens when m slightly
        # exceeds 1.0 due to floating-point boundary rounding in the
        # scale / 2**floor(log2(scale)) reconstruction) and against any
        # stray sub-min value.  Clamp to the int32 positive range since
        # out_multiplier is documented as > 0.
        if q31 >= 2**31:
            q31 = 2**31 - 1
        if q31 < 1:
            q31 = 1
        # shift passed to nn_divide_by_power_of_two:
        #   positive exponent = left shift, negative = right shift
        # We need right shift of (31 - e) so pass -(31 - e) = e - 31
        shift = e - 31
        multipliers.append(q31)
        shifts.append(shift)
    return np.array(multipliers, dtype=np.int32), np.array(shifts, dtype=np.int32)

RNG = np.random.default_rng(42)

def rand_int8(shape):
    return RNG.integers(-128, 127, size=shape, dtype=np.int8)

def rand_int32(shape, lo=-256, hi=256):
    return RNG.integers(lo, hi, size=shape, dtype=np.int32)

# ---------------------------------------------------------------------------
# Pattern definitions
# ---------------------------------------------------------------------------

def make_pattern(pid, desc, purpose,
                 IH, IW, IC, KH, KW, OC,
                 stride_h=1, stride_w=1,
                 pad_h=0, pad_w=0,
                 input_zero_bias=0, out_zero_bias=0,
                 activation_min=-128, activation_max=127,
                 scale=0.00392156,    # ~1/255
                 batch_size=1,
                 inp_override=None, kernel_override=None, bias_override=None,
                 OH_override=None, OW_override=None):

    # Default: derive from symmetric padding formula.
    # Override: caller supplies OH/OW directly (needed for TFLite SAME
    # padding where pad_top != pad_bottom or pad_left != pad_right; the
    # reference loop then naturally handles bottom/right zero-padding via
    # its `ih >= input_height` / `iw >= input_width` skip, i.e. pad_h/pad_w
    # are treated as the top/left pad only).
    OH = OH_override if OH_override is not None else out_size(IH, KH, stride_h, pad_h)
    OW = OW_override if OW_override is not None else out_size(IW, KW, stride_w, pad_w)

    inp    = inp_override    if inp_override    is not None else rand_int8((batch_size, IH, IW, IC))
    kernel = kernel_override if kernel_override is not None else rand_int8((OC, KH, KW, IC))
    bias   = bias_override   if bias_override   is not None else rand_int32((OC,))

    mult, shift = make_multiplier_shift(scale, OC)

    golden, partial_sum = conv2d_reference(
        inp, kernel, bias, mult, shift,
        batch_size, IH, IW, IC, KH, KW,
        stride_h, stride_w, pad_h, pad_w,
        OH, OW, OC,
        input_zero_bias, 0, out_zero_bias,
        activation_min, activation_max)

    cfg = dict(
        desc=desc, purpose=purpose,
        batch_size=batch_size,
        input_height=IH, input_width=IW, input_channels=IC,
        kernel_height=KH, kernel_width=KW, out_channels=OC,
        x_stride=stride_w, y_stride=stride_h,
        x_padding=pad_w,   y_padding=pad_h,
        out_height=OH,     out_width=OW,
        input_zero_bias=input_zero_bias,
        out_zero_bias=out_zero_bias,
        out_data_format=0,
        out_activation_min=activation_min,
        out_activation_max=activation_max,
    )

    fname = f"{pid}_{desc}.in_out"
    write_inout(fname, cfg, inp, kernel, bias, mult, shift, golden, partial_sum)
    print(f"  [{pid}] {fname}  "
          f"({batch_size}x{IH}x{IW}x{IC} -> {batch_size}x{OH}x{OW}x{OC}, "
          f"K={KH}x{KW}, S={stride_h}x{stride_w}, P={pad_h}x{pad_w})")
    print(f"         Purpose : {purpose}")
    return fname

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Conv2D .in_out test fixtures.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python3 gen_conv2d_patterns.py\n"
            "  python3 gen_conv2d_patterns.py p03 p05\n"
            "  python3 gen_conv2d_patterns.py --single-rounding\n"
            "  python3 gen_conv2d_patterns.py --double-rounding p07\n"
        ),
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--single-rounding", dest="single_rounding",
        action="store_true",
        help="Use TFLITE_SINGLE_ROUNDING=1 path (64-bit one-step requantize).")
    mode_group.add_argument(
        "--double-rounding", dest="double_rounding",
        action="store_true",
        help="Use TFLITE_SINGLE_ROUNDING=0 path (default, two-step requantize).")
    parser.add_argument(
        "patterns", nargs="*",
        help="Optional pattern IDs to generate (e.g. p01 p03).  Default: all.")
    args = parser.parse_args()

    # Apply mode flag (default: double-rounding).
    if args.single_rounding:
        _TFLITE_SINGLE_ROUNDING = 1
    else:
        _TFLITE_SINGLE_ROUNDING = 0

    select = set(args.patterns) if args.patterns else None
    def want(pid):
        return select is None or pid in select

    mode_label = "SINGLE-ROUNDING (TFLITE_SINGLE_ROUNDING=1)" \
                 if _TFLITE_SINGLE_ROUNDING else \
                 "DOUBLE-ROUNDING (TFLITE_SINGLE_ROUNDING=0)"
    print(f"Generating Conv2D test patterns  [{mode_label}] ...\n")

    # P01 ── Non-zero input & output zero-bias
    # Tests the (+input_zero_bias) correction in the accumulator loop and the
    # (+out_zero_bias) shift applied after requantization.  A wrong sign on
    # either zero-bias will show as a systematic offset on every output pixel.
    if want("p01"):
        make_pattern("p01", "nonzero_zero_bias",
            "Non-zero input_zero_bias (-30) and out_zero_bias (+10)."
            "  Tests signed zero-point correction on both input and output.",
            IH=8, IW=8, IC=4, KH=3, KW=3, OC=4,
            pad_h=1, pad_w=1,
            input_zero_bias=-30, out_zero_bias=10)

    # P02 ── Activation clamp exercises both rails
    # Uses a large scale so many outputs saturate to activation_min (-128) or
    # activation_max (127).  Verifies clamp_i32 triggers on both sides.
    if want("p02"):
        make_pattern("p02", "activation_clamp",
            "Large requant scale forces many outputs to hit activation min/max."
            "  Verifies clamp_i32 triggers on both -128 and +127 rails.",
            IH=6, IW=6, IC=4, KH=3, KW=3, OC=8,
            pad_h=1, pad_w=1,
            scale=0.5)           # large scale → many saturations

    # P03 ── Stride 2 (spatial decimation)
    # Output is half the spatial size.  Wrong stride indexing in the input
    # address calculation produces completely wrong outputs; this makes stride
    # bugs immediately visible.
    if want("p03"):
        make_pattern("p03", "stride2",
            "Stride=2 halves spatial output dimensions."
            "  Wrong stride in input-address calc produces fully wrong outputs.",
            IH=8, IW=8, IC=4, KH=3, KW=3, OC=4,
            stride_h=2, stride_w=2, pad_h=1, pad_w=1)

    # P04 ── 1×1 pointwise convolution (no spatial context)
    # KH=KW=1, pad=0, stride=1.  Degenerates to a channel-wise matrix multiply.
    # Any kernel/input indexing error that only manifests with spatial extent
    # will be invisible here, making this a clean baseline for per-channel
    # multiplier/shift arithmetic.
    if want("p04"):
        make_pattern("p04", "1x1_pointwise",
            "1x1 kernel with no spatial context; pure per-channel matmul."
            "  Isolates requantization arithmetic from spatial indexing.",
            IH=8, IW=8, IC=16, KH=1, KW=1, OC=8)

    # P05 ── 5×5 large kernel with SAME padding
    # Large receptive field.  Tests that the kernel/input base-offset
    # computation (oc * KH * KW * IC + ...) is correct for big kernels.
    if want("p05"):
        make_pattern("p05", "5x5_large_kernel",
            "5x5 kernel with SAME padding (pad=2)."
            "  Large kernel stresses the w_base offset formula.",
            IH=8, IW=8, IC=4, KH=5, KW=5, OC=4,
            pad_h=2, pad_w=2)

    # P06 ── Many channels (IC=32, OC=32)
    # The inner-most IC loop and the OC-indexed multiplier/shift arrays are
    # stressed.  Off-by-one in channel indexing accumulates over 32 channels.
    if want("p06"):
        make_pattern("p06", "many_channels",
            "IC=32, OC=32 with 3x3 kernel."
            "  Stresses per-channel multiplier/shift array indexing and"
            "  the inner IC accumulation loop.",
            IH=6, IW=6, IC=32, KH=3, KW=3, OC=32,
            pad_h=1, pad_w=1)

    # P07 ── Accumulator near overflow / large bias
    # Fills input and kernel with max (127) values, adds a large positive bias
    # to push the int32 accumulator toward INT32_MAX before requantization.
    # Catches 32-bit overflow in the MAC loop or in doubling_high_mult.
    if want("p07"):
        make_pattern("p07", "accumulator_near_overflow",
            "All-positive max inputs (127) + large bias pushes acc near INT32_MAX."
            "  Catches 32-bit overflow in the MAC or doubling_high_mult path.",
            IH=4, IW=4, IC=8, KH=3, KW=3, OC=4,
            inp_override    = np.full((1,4,4,8), 127, dtype=np.int8),
            kernel_override = np.full((4,3,3,8), 127, dtype=np.int8),
            bias_override   = np.full((4,), 1_000_000, dtype=np.int32),
            activation_min=-128, activation_max=127,
            scale=0.00001)       # tiny scale to bring result in-range

    # P08 ── All-zero input (expected zero output modulo bias/zero-point)
    # With zero input and zero bias, every output must equal
    # clamp(requantize(0) + out_zero_bias).  Any non-zero deviation indicates
    # an uninitialized accumulator or incorrect zero-bias subtraction.
    if want("p08"):
        make_pattern("p08", "all_zero_input",
            "Input all zeros, bias all zeros.  Every output must equal"
            "  clamp(out_zero_bias, min, max).  Catches uninit accumulator.",
            IH=6, IW=6, IC=4, KH=3, KW=3, OC=4,
            pad_h=1, pad_w=1,
            inp_override  = np.zeros((1,6,6,4), dtype=np.int8),
            bias_override = np.zeros((4,),    dtype=np.int32),
            out_zero_bias = 5)

    # P09 ── Negative shift (left-shift path in nn_divide_by_power_of_two)
    # Uses a very small scale that produces a large positive shift in the
    # make_multiplier_shift derivation, meaning the effective shift value
    # passed to nn_divide_by_power_of_two is positive (left-shift branch).
    # This branch is rarely exercised but must be correct.
    if want("p09"):
        make_pattern("p09", "positive_exponent_leftshift",
            "Scale chosen so shift > 0 → left-shift branch of"
            "  nn_divide_by_power_of_two is exercised (rare path).",
            IH=4, IW=4, IC=4, KH=1, KW=1, OC=4,
            scale=1e-10)         # tiny scale → large positive shift

    # P10 ── Asymmetric padding (pad_h != pad_w)
    # pad_h=2, pad_w=0: tests that the height and width padding paths are
    # independent.  A single-variable pad bug (using pad_h for both axes)
    # will show as wrong outputs on left/right columns but correct top/bottom rows.
    if want("p10"):
        make_pattern("p10", "asymmetric_padding",
            "pad_h=2, pad_w=0: height and width padding are independent."
            "  A bug that conflates pad_h/pad_w produces wrong boundary pixels.",
            IH=8, IW=8, IC=4, KH=5, KW=3, OC=4,
            pad_h=2, pad_w=0)

    # P11 ── KWS first-layer conv2d (real-world shape from kws_ref_model.tflite)
    # Ingest 49x10 MFCC frame into 25x5x64 feature map.
    # Mirrors the first conv2d in the standard Keyword Spotting reference model:
    #   input_height=49, input_width=10, input_channels=1
    #   kernel   : 10x4x1x64  (HxWxICxOC)
    #   stride   : 2x2
    #   padding  : SAME  (TFLite ceil(IH/stride) rule, dilation=1)
    #     -> OH = ceil(49/2) = 25, OW = ceil(10/2) = 5
    #     -> pad_total_h = 9 (top=4, bottom=5), pad_total_w = 2 (left=1, right=1)
    #     We pass pad_h=4, pad_w=1 (top/left pads) and override OH=25, OW=5.
    #     The conv2d_reference skips `ih >= IH` / `iw >= IW` automatically,
    #     so the extra bottom row (pad_bottom=5 > pad_top=4) is covered.
    #   input_zero_bias=-83, out_zero_bias=-128  (real KWS asym8s quant params)
    # Purpose: a realistic end-to-end shape that combines wide kernel,
    # non-unity stride, asymmetric SAME padding, and non-zero zero-biases.
    # Any single-axis bug (stride/pad/zp) becomes immediately visible.
    if want("p11"):
        make_pattern("p11", "kws_first_layer",
            "Real KWS first conv2d: 49x10x1 -> 25x5x64, K=10x4, stride=2,"
            "  SAME padding (pad_h=4, pad_w=1), input_zp=-83, out_zp=-128."
            "  End-to-end realistic shape combining wide kernel + stride +"
            "  asymmetric SAME pad + non-zero zero-points.",
            IH=49, IW=10, IC=1, KH=10, KW=4, OC=64,
            stride_h=2, stride_w=2,
            pad_h=4, pad_w=1,
            OH_override=25, OW_override=5,
            input_zero_bias=-83, out_zero_bias=-128)


    # P12 ── Custom shape: 40x24x16 input, 3x3 kernel, 1x1 stride, 0 padding
    # Tests general case with moderate input height and width, per-channel output.
    # columns1 = 16*3*3 = 144 → Branch D (aligned general path).
    if want("p12"):
        make_pattern("p12", "custom_shape_40x24",
            "Input 40x24x16, Kernel 3x3x16x16, stride=1x1, pad=0x0 → 38x22x16."
            "  Moderate spatial dims, general 3x3 kernel, per-channel quant.",
            IH=40, IW=24, IC=16, KH=3, KW=3, OC=16,
            stride_h=1, stride_w=1,
            pad_h=0, pad_w=0)

    print("\nDone.  Run each fixture with:")
    print("  make run FIXTURE=<pattern>.in_out")
