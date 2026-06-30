#!/usr/bin/env python3
"""
gen_depthwise_patterns.py
=========================
Generate .in_out fixture files for xa_nn_conv2d_depthwise_sym8sxasym8s tests.

Each pattern targets a specific aspect of the depthwise conv pipeline.

Usage:
    python3 gen_depthwise_patterns.py          # all patterns (double-rounding)
    python3 gen_depthwise_patterns.py p1 p3    # selective
    python3 gen_depthwise_patterns.py --single-rounding
    python3 gen_depthwise_patterns.py --double-rounding p3

Output files (./patterns/ directory):
    dw_p01_basic_cm1.in_out
    dw_p02_nonzero_zero_bias.in_out
    dw_p03_channels_multiplier2.in_out
    dw_p04_dilation2.in_out
    dw_p05_stride2.in_out
    dw_p06_activation_clamp.in_out
    dw_p07_many_channels.in_out
    dw_p08_all_zero_input.in_out

.in_out contract (identical to conv2d fixtures):
    // Pattern: <desc>
    // Purpose: <purpose>
    // Requantize mode: ...

    int tflite_single_rounding = 0|1;
    int <scalar> = <v>;          (one per shape/quant param)
    int8_t  input[N]  = { ... };
    int8_t  kernel[N] = { ... };
    int32_t bias[OC]  = { ... };
    int32_t out_multiplier[OC] = { ... };
    int32_t out_shift[OC]      = { ... };
    int8_t  golden[N] = { ... };
    int32_t partial_sum[N] = { ... };   (pre-requant acc; MAC op)
"""

import argparse
import math
import os
import sys

import numpy as np

# ---------------------------------------------------------------------------
# TFLITE_SINGLE_ROUNDING mode flag
# ---------------------------------------------------------------------------
# 0 = double-rounding  (legacy GEMMLOWP / nnlib-hifi5 default)
# 1 = single-rounding  (TFLM reference when TFLITE_SINGLE_ROUNDING is set)
_TFLITE_SINGLE_ROUNDING = 0

# ---------------------------------------------------------------------------
# Fixed-point arithmetic — verbatim from gen_conv2d_patterns.py
# ---------------------------------------------------------------------------

INT32_MIN = -(1 << 31)
INT32_MAX =  (1 << 31) - 1

# ---- Double-rounding (TFLITE_SINGLE_ROUNDING = 0) --------------------------

def doubling_high_mult_no_sat(m1, m2):
    """(m1 * m2) >> 31  using 64-bit intermediate (no saturation)."""
    result = (int(m1) * int(m2)) >> 31
    return np.int32(result)

def divide_by_power_of_two(dividend, exponent):
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
    Matches nnlib-hifi5 single-rounding convention.
    """
    val   = int(val)
    mult  = int(multiplier)
    shift = int(shift)

    total_right_shift = 31 - shift

    prod = val * mult                   # Python int: no overflow

    if total_right_shift <= 0:
        result = prod << (-total_right_shift)
    else:
        remainder_mask = (1 << total_right_shift) - 1
        remainder      = prod & remainder_mask
        quotient       = prod >> total_right_shift
        threshold      = remainder_mask >> 1
        if quotient < 0:
            threshold += 1
        if remainder > threshold:
            quotient += 1
        result = quotient

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
    return max(lo, min(hi, int(x)))

# ---------------------------------------------------------------------------
# Depthwise reference (kernel layout [KH, KW, OC], OC = IC * CM)
# ---------------------------------------------------------------------------

def depthwise_reference(inp, kernel, bias,
                        out_multiplier, out_shift,
                        IH, IW, IC, KH, KW, CM,
                        dil_h, dil_w, str_h, str_w,
                        pad_h, pad_w, OH, OW,
                        input_zero_bias, out_zero_bias,
                        activation_min, activation_max):
    """
    Returns
    -------
    out         : np.ndarray[int8]  shape (OH, OW, OC)
    partial_sum : np.ndarray[int32] shape (OH, OW, OC)
                  acc = bias[oc] + sum((inp+izp)*w)
                  captured just before requantize().
    """
    OC = IC * CM
    out         = np.zeros((OH, OW, OC), dtype=np.int8)
    partial_sum = np.zeros((OH, OW, OC), dtype=np.int32)
    for oh in range(OH):
        for ow in range(OW):
            for oc in range(OC):
                ic  = oc // CM
                acc = int(bias[oc]) if bias is not None else 0
                for kh in range(KH):
                    ih = oh * str_h + kh * dil_h - pad_h
                    if ih < 0 or ih >= IH:
                        continue
                    for kw in range(KW):
                        iw = ow * str_w + kw * dil_w - pad_w
                        if iw < 0 or iw >= IW:
                            continue
                        inp_v = int(inp[ih, iw, ic]) + input_zero_bias
                        w_v   = int(kernel[kh, kw, oc])
                        acc  += inp_v * w_v
                # capture partial sum BEFORE requantize
                partial_sum[oh, ow, oc] = np.int32(
                    np.clip(acc, -(2**31), 2**31 - 1))
                out_v  = requantize(acc, int(out_multiplier[oc]), int(out_shift[oc]))
                out_v += out_zero_bias
                out[oh, ow, oc] = np.int8(clamp(out_v, activation_min, activation_max))
    return out, partial_sum

# ---------------------------------------------------------------------------
# Multiplier / shift helper
# ---------------------------------------------------------------------------

def make_mult_shift(scale, n):
    mults, shifts = [], []
    for _ in range(n):
        if scale == 0.0:
            mults.append(0); shifts.append(0); continue
        e   = math.floor(math.log2(scale))
        m   = scale / (2 ** e)
        q31 = round(m * (2**31))
        if q31 >= 2**31:
            q31 = 2**31 - 1
        if q31 < 1:
            q31 = 1
        mults.append(q31)
        shifts.append(e - 31)
    return np.array(mults, np.int32), np.array(shifts, np.int32)

def out_size(in_sz, k, s, p, d=1):
    return (in_sz + 2*p - d*(k-1) - 1) // s + 1

# ---------------------------------------------------------------------------
# .in_out writer  (depthwise variant, matching conv2d contract)
# ---------------------------------------------------------------------------

def write_inout(path, scalars, inp, kernel, bias, mult, shift, golden, partial_sum):
    """
    Write a fixture file in the format expected by the C test parser.

    partial_sum : np.ndarray[int32] shape (OH, OW, OC) — pre-requant acc.
    """
    def a8(name, data):
        f = data.flatten().astype(np.int8)
        return f"int8_t {name}[{len(f)}] = {{{', '.join(str(int(v)) for v in f)}}};\n"

    def a32(name, data):
        f = np.array(data, np.int32).flatten()
        return f"int32_t {name}[{len(f)}] = {{{', '.join(str(int(v)) for v in f)}}};\n"

    mode_str = (
        "single-rounding (TFLITE_SINGLE_ROUNDING=1, 64-bit one-step)"
        if _TFLITE_SINGLE_ROUNDING else
        "double-rounding (TFLITE_SINGLE_ROUNDING=0, 32-bit two-step)"
    )

    with open(path, "w") as f:
        f.write(f"// Pattern: {scalars['desc']}\n")
        f.write(f"// Purpose: {scalars['purpose']}\n")
        f.write(f"// Requantize mode: {mode_str}\n\n")
        f.write(f"int tflite_single_rounding = {_TFLITE_SINGLE_ROUNDING};\n")
        for k, v in scalars.items():
            if k not in ("desc", "purpose"):
                f.write(f"int {k} = {v};\n")
        f.write("\n")
        f.write(a8("input",  inp))
        f.write(a8("kernel", kernel))
        OC = len(mult)
        if bias is not None:
            f.write(a32("bias", bias))
        else:
            f.write(f"int32_t bias[{OC}] = {{" + ", ".join(["0"]*OC) + "};\n")
        f.write(a32("out_multiplier", mult))
        f.write(a32("out_shift",      shift))
        f.write(a8("golden", golden))
        f.write(a32("partial_sum", partial_sum))

# ---------------------------------------------------------------------------
# Pattern factory
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)

def make_dw_pattern(pid, desc, purpose,
                    IH, IW, IC, KH, KW, CM=1,
                    str_h=1, str_w=1, pad_h=0, pad_w=0,
                    dil_h=1, dil_w=1,
                    input_zero_bias=0, out_zero_bias=0,
                    activation_min=-128, activation_max=127,
                    scale=0.00392156,
                    inp=None, kernel=None, bias=None,
                    outdir="patterns"):

    OC = IC * CM
    OH = out_size(IH, KH, str_h, pad_h, dil_h)
    OW = out_size(IW, KW, str_w, pad_w, dil_w)

    if inp    is None: inp    = RNG.integers(-128, 127, (IH, IW, IC), dtype=np.int8)
    if kernel is None: kernel = RNG.integers(-128, 127, (KH, KW, OC), dtype=np.int8)
    if bias   is None: bias   = RNG.integers(-256, 256, (OC,),        dtype=np.int32)

    mult, shift = make_mult_shift(scale, OC)

    golden, partial_sum = depthwise_reference(
        inp, kernel, bias, mult, shift,
        IH, IW, IC, KH, KW, CM, dil_h, dil_w, str_h, str_w,
        pad_h, pad_w, OH, OW,
        input_zero_bias, out_zero_bias, activation_min, activation_max)

    scalars = dict(
        desc=desc, purpose=purpose,
        input_height=IH, input_width=IW, input_channels=IC,
        kernel_height=KH, kernel_width=KW,
        channels_multiplier=CM,
        dilation_height=dil_h, dilation_width=dil_w,
        x_stride=str_w, y_stride=str_h,
        x_padding=pad_w, y_padding=pad_h,
        out_height=OH, out_width=OW,
        out_channels=OC,
        input_zero_bias=input_zero_bias,
        out_zero_bias=out_zero_bias,
        out_data_format=0,
        out_activation_min=activation_min,
        out_activation_max=activation_max,
    )

    os.makedirs(outdir, exist_ok=True)
    fname = os.path.join(outdir, f"dw_{pid}_{desc}.in_out")
    write_inout(fname, scalars, inp, kernel, bias, mult, shift, golden, partial_sum)
    print(f"  [{pid}] {fname}")
    print(f"         Config : {IH}x{IW}x{IC} -> {OH}x{OW}x{OC}, "
          f"K={KH}x{KW}, CM={CM}, S={str_h}x{str_w}, P={pad_h}x{pad_w}, "
          f"D={dil_h}x{dil_w}")
    print(f"         Purpose: {purpose}")
    return fname

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Depthwise Conv2D .in_out test fixtures.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--single-rounding", dest="single_rounding",
        action="store_true",
        help="Use TFLITE_SINGLE_ROUNDING=1 path.")
    mode_group.add_argument(
        "--double-rounding", dest="double_rounding",
        action="store_true",
        help="Use TFLITE_SINGLE_ROUNDING=0 path (default).")
    parser.add_argument(
        "patterns", nargs="*",
        help="Optional pattern IDs to generate (e.g. p1 p3).  Default: all.")
    args = parser.parse_args()

    if args.single_rounding:
        _TFLITE_SINGLE_ROUNDING = 1
    else:
        _TFLITE_SINGLE_ROUNDING = 0

    select = set(args.patterns) if args.patterns else None
    def want(p): return select is None or p in select

    mode_label = ("SINGLE-ROUNDING (TFLITE_SINGLE_ROUNDING=1)"
                  if _TFLITE_SINGLE_ROUNDING else
                  "DOUBLE-ROUNDING (TFLITE_SINGLE_ROUNDING=0)")
    print(f"Generating Depthwise Conv2D test patterns  [{mode_label}] ...\n")

    # DW-P01  Basic CM=1 (standard depthwise, no frills)
    if want("p1"):
        make_dw_pattern("p01", "basic_cm1",
            "Standard depthwise CM=1.  Baseline for kernel/input indexing.",
            IH=6, IW=6, IC=8, KH=3, KW=3, CM=1, pad_h=1, pad_w=1)

    # DW-P02  Non-zero input & output zero-bias
    if want("p2"):
        make_dw_pattern("p02", "nonzero_zero_bias",
            "input_zero_bias=-20, out_zero_bias=8."
            "  Tests zero-point correction with per-channel depthwise indexing.",
            IH=6, IW=6, IC=4, KH=3, KW=3, CM=1, pad_h=1, pad_w=1,
            input_zero_bias=-20, out_zero_bias=8)

    # DW-P03  channels_multiplier = 2
    if want("p3"):
        make_dw_pattern("p03", "channels_multiplier2",
            "CM=2 doubles output channels.  ic=oc//2 sharing tests kernel"
            "  oc-indexing and per-channel quant param arrays.",
            IH=6, IW=6, IC=4, KH=3, KW=3, CM=2, pad_h=1, pad_w=1)

    # DW-P04  Dilation = 2
    if want("p4"):
        make_dw_pattern("p04", "dilation2",
            "dilation=2 with 3x3 kernel (effective RF=5x5)."
            "  Wrong dilation in ih/iw calc produces spatially shifted outputs.",
            IH=8, IW=8, IC=4, KH=3, KW=3, CM=1,
            pad_h=2, pad_w=2, dil_h=2, dil_w=2)

    # DW-P05  Stride 2
    if want("p5"):
        make_dw_pattern("p05", "stride2",
            "Stride=2 halves output spatial dims."
            "  Wrong stride in ih/iw produces fully wrong outputs.",
            IH=8, IW=8, IC=4, KH=3, KW=3, CM=1,
            str_h=2, str_w=2, pad_h=1, pad_w=1)

    # DW-P06  Activation clamp
    if want("p6"):
        make_dw_pattern("p06", "activation_clamp",
            "Large scale forces outputs to hit -128/+127 clamp rails.",
            IH=6, IW=6, IC=4, KH=3, KW=3, CM=1, pad_h=1, pad_w=1,
            scale=0.5)

    # DW-P07  Many channels
    if want("p7"):
        make_dw_pattern("p07", "many_channels",
            "IC=32, CM=1 (OC=32).  Stresses per-channel quant param indexing"
            "  and the ic=oc path across many channels.",
            IH=6, IW=6, IC=32, KH=3, KW=3, CM=1, pad_h=1, pad_w=1)

    # DW-P08  All-zero input
    if want("p8"):
        make_dw_pattern("p08", "all_zero_input",
            "Zero input + zero bias.  Every output = clamp(out_zero_bias)."
            "  Catches uninit accumulator or wrong ic indexing.",
            IH=6, IW=6, IC=4, KH=3, KW=3, CM=1, pad_h=1, pad_w=1,
            inp=np.zeros((6,6,4), dtype=np.int8),
            bias=np.zeros((4,),   dtype=np.int32),
            out_zero_bias=3)

    print("\nDone.  Build and run with:")
    print("  make")
    print("  make run-all")
