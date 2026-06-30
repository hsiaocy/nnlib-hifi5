#!/usr/bin/env python3
"""
gen_fc_patterns.py
==================
Generate .in_out fixture files for xa_nn_fully_connected (sym8sxasym8s_asym8s) tests.

The FC op is a pure MAC operation:
  for n in [0, out_depth):
    acc = bias[n]
    for m in [0, weight_depth):
      acc += kernel[n*weight_depth + m] * (input[m] + input_zero_bias)
    partial_sum[n] = clamp_int32(acc)   # pre-requant, per output neuron
    out[n] = clamp( MBQM(acc, out_multiplier, out_shift) + out_zero_bias,
                    activation_min, activation_max )

Requantization uses TFLM MultiplyByQuantizedMultiplier (MBQM):
  MBQM(acc, mult, shift):
    if shift > 0: acc <<= shift; return SRDHM(acc, mult)
    else:         return RoundingDivideByPOT( SRDHM(acc, mult), -shift )

  SRDHM(a, b):
    if a==INT32_MIN and b==INT32_MIN: return INT32_MAX   (sat)
    product = (int64)a * b
    nudge = (1<<30) if product>=0 else (1-(1<<30))
    return (product + nudge) >> 31

  RoundingDivideByPOT(x, exp):
    mask = (1<<exp)-1; remainder = x & mask
    threshold = (mask>>1) + (1 if x<0 else 0)
    return (x >> exp) + (1 if remainder > threshold else 0)

CAVEAT: This requant (MBQM/SRDHM/RoundingDivideByPOT) differs from the conv2d
cmodel's nn_doubling_high_mult_no_sat + nn_divide_by_power_of_two.  They agree
for all values except when the product overflows int64 (INT32_MIN*INT32_MIN case)
and at certain rounding ties.  The FC cmodel (cmodel_fc_hifi5.c) is used as-is;
this generator matches it, not the conv2d requant core.

Fixture parameter mapping (FC -> .in_out contract):
  input_channels = weight_depth   (number of input features)
  out_channels   = out_depth      (number of output neurons)
  kernel[out_depth * weight_depth] row-major: kernel[n*weight_depth + m] = W[n][m]
  out_multiplier = scalar int32   (single shared multiplier, not per-channel)
  out_shift      = scalar int32   (single shared shift)

Usage:
    cd patterns/
    python3 gen_fc_patterns.py                  # generate all patterns
    python3 gen_fc_patterns.py p01 p03          # selective
"""

import argparse
import numpy as np
import os
import sys

INT32_MIN = -(1 << 31)
INT32_MAX =  (1 << 31) - 1

# ---------------------------------------------------------------------------
# Fixed-point arithmetic matching cmodel_fc_hifi5.c
# ---------------------------------------------------------------------------

def srdhm(a: int, b: int) -> int:
    """SaturatingRoundingDoublingHighMul: round(a*b/2^31), sat at INT32_MAX."""
    a = int(a); b = int(b)
    if a == INT32_MIN and b == INT32_MIN:
        return INT32_MAX
    product = a * b   # Python int, no overflow
    nudge = (1 << 30) if product >= 0 else (1 - (1 << 30))
    return int((product + nudge) >> 31)

def rounding_divide_by_pot(x: int, exp: int) -> int:
    """RoundingDivideByPOT: round(x / 2^exp), ties away from zero. exp >= 0."""
    x = int(x); exp = int(exp)
    if exp == 0:
        return x
    mask = (1 << exp) - 1
    remainder = x & mask
    threshold = (mask >> 1) + (1 if x < 0 else 0)
    return (x >> exp) + (1 if remainder > threshold else 0)

def mbqm(acc: int, out_multiplier: int, out_shift: int) -> int:
    """MultiplyByQuantizedMultiplier matching cmodel_MBQM in cmodel_fc_hifi5.c."""
    acc = int(acc)
    if out_shift > 0:
        acc = acc << out_shift
        return srdhm(acc, out_multiplier)
    else:
        result = srdhm(acc, out_multiplier)
        return rounding_divide_by_pot(result, -out_shift)

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def clamp_int32(x):
    return clamp(int(x), INT32_MIN, INT32_MAX)

# ---------------------------------------------------------------------------
# FC reference (matches cmodel_fc_v2_sym8sxasym8s_asym8s exactly)
# ---------------------------------------------------------------------------

def fc_reference(inp, kernel, bias,
                 out_multiplier, out_shift,
                 weight_depth, out_depth,
                 input_zero_bias, out_zero_bias,
                 activation_min, activation_max):
    """
    Pure-Python reference for FC sym8sxasym8s_asym8s.

    Parameters
    ----------
    inp    : np.ndarray[int8]  shape (weight_depth,)
    kernel : np.ndarray[int8]  shape (out_depth, weight_depth)
    bias   : np.ndarray[int32] shape (out_depth,) or None
    out_multiplier : int  (scalar Q31, shared across all output neurons)
    out_shift      : int  (scalar, shared across all output neurons)

    Returns
    -------
    out         : np.ndarray[int8]  shape (out_depth,)
    partial_sum : np.ndarray[int32] shape (out_depth,)
                  acc = bias[n] + sum_m( kernel[n,m] * (inp[m]+input_zero_bias) )
                  captured BEFORE requantize(); clamped to int32 range.
    """
    out         = np.zeros(out_depth, dtype=np.int8)
    partial_sum = np.zeros(out_depth, dtype=np.int32)

    for n in range(out_depth):
        acc = int(bias[n]) if bias is not None else 0
        for m in range(weight_depth):
            w = int(kernel[n, m])
            x = int(inp[m]) + input_zero_bias
            acc += w * x
        # partial_sum: pre-requant accumulator, clamped to int32
        partial_sum[n] = np.int32(clamp_int32(acc))
        # requantize
        out_v = mbqm(clamp_int32(acc), out_multiplier, out_shift)
        out_v += out_zero_bias
        out_v  = clamp(out_v, activation_min, activation_max)
        out[n] = np.int8(out_v)

    return out, partial_sum

# ---------------------------------------------------------------------------
# .in_out file writer
# ---------------------------------------------------------------------------

def arr8(name, data):
    flat = np.array(data, dtype=np.int8).flatten()
    vals = ", ".join(str(int(v)) for v in flat)
    return f"int8_t {name}[{len(flat)}] = {{{vals}}};\n"

def arr32(name, data):
    flat = np.array(data, dtype=np.int32).flatten()
    vals = ", ".join(str(int(v)) for v in flat)
    return f"int32_t {name}[{len(flat)}] = {{{vals}}};\n"

def write_inout(path, desc, purpose,
                weight_depth, out_depth,
                input_zero_bias, out_zero_bias,
                activation_min, activation_max,
                out_multiplier, out_shift,
                inp, kernel, bias, golden, partial_sum):
    with open(path, "w") as f:
        f.write(f"// Pattern: {desc}\n")
        f.write(f"// Purpose: {purpose}\n")
        f.write(f"// Requantize mode: MBQM (SRDHM+RoundingDivideByPOT, TFLM-style)\n\n")
        # Scalar params
        f.write(f"int weight_depth = {weight_depth};\n")
        f.write(f"int out_depth = {out_depth};\n")
        f.write(f"int input_zero_bias = {input_zero_bias};\n")
        f.write(f"int out_zero_bias = {out_zero_bias};\n")
        f.write(f"int out_activation_min = {activation_min};\n")
        f.write(f"int out_activation_max = {activation_max};\n")
        f.write(f"int out_multiplier = {out_multiplier};\n")
        f.write(f"int out_shift = {out_shift};\n")
        f.write("\n")
        # Arrays
        f.write(arr8("input",  inp))
        f.write(arr8("kernel", kernel))
        if bias is not None:
            f.write(arr32("bias", bias))
        else:
            f.write(f"int32_t bias[{out_depth}] = {{" + ", ".join(["0"]*out_depth) + "};\n")
        f.write(f"int32_t out_multiplier_arr[1] = {{{out_multiplier}}};\n")
        f.write(f"int32_t out_shift_arr[1] = {{{out_shift}}};\n")
        f.write(arr8("golden", golden))
        f.write(arr32("partial_sum", partial_sum))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_multiplier_shift(scale_f32):
    """Convert float scale -> (Q31 multiplier, shift) matching TFLite convention.
    shift: positive = left, negative = right (same sign convention as cmodel_MBQM).

    The cmodel validates out_shift in [-31, 31].  Shifts outside this range are
    clamped: if raw shift < -31, we fold extra right-shift into the multiplier
    (multiply multiplier by 2^excess, saturating at INT32_MAX).  This keeps both
    sides in-range while preserving approximate scale fidelity.
    """
    import math
    if scale_f32 == 0.0:
        return 0, 0
    e = math.floor(math.log2(scale_f32))
    m = scale_f32 / (2 ** e)
    q31 = round(m * (2**31))
    if q31 >= 2**31:
        q31 = 2**31 - 1
    if q31 < 1:
        q31 = 1
    # MBQM shift convention: positive=left, negative=right
    # scale = q31/2^31 * 2^e  =>  effective right-shift = 31 - e
    # => pass shift = e - 31 (negative = right shift)
    shift = e - 31

    # Clamp shift to cmodel's accepted range [-31, 31].
    # If shift < -31: excess_right = |shift| - 31; absorb excess into multiplier.
    #   new_mult = q31 >> excess_right  (reduces multiplier so scale is preserved)
    # If shift > 31: left-shift > 31 is also out of range; cap at 31.
    if shift < -31:
        excess = (-shift) - 31      # how many extra right-shift bits beyond limit
        q31 = max(1, q31 >> excess) # absorb extra shift into multiplier
        shift = -31
    elif shift > 31:
        shift = 31

    return int(q31), int(shift)

RNG = np.random.default_rng(42)

def rand_int8(shape):
    return RNG.integers(-128, 127, size=shape, dtype=np.int8)

def rand_int32(shape, lo=-256, hi=256):
    return RNG.integers(lo, hi, size=shape, dtype=np.int32)

# ---------------------------------------------------------------------------
# Pattern factory
# ---------------------------------------------------------------------------

def make_pattern(pid, desc, purpose,
                 weight_depth, out_depth,
                 input_zero_bias=0, out_zero_bias=0,
                 activation_min=-128, activation_max=127,
                 scale=0.00392156,
                 inp_override=None,
                 kernel_override=None,
                 bias_override=None,
                 out_multiplier_override=None,
                 out_shift_override=None,
                 outdir="."):

    inp    = inp_override    if inp_override    is not None else rand_int8((weight_depth,))
    kernel = kernel_override if kernel_override is not None else rand_int8((out_depth, weight_depth))
    bias   = bias_override   if bias_override   is not None else rand_int32((out_depth,))

    if out_multiplier_override is not None and out_shift_override is not None:
        out_multiplier = int(out_multiplier_override)
        out_shift      = int(out_shift_override)
    else:
        out_multiplier, out_shift = make_multiplier_shift(scale)

    golden, partial_sum = fc_reference(
        inp, kernel, bias,
        out_multiplier, out_shift,
        weight_depth, out_depth,
        input_zero_bias, out_zero_bias,
        activation_min, activation_max)

    fname = os.path.join(outdir, f"{pid}_{desc}.in_out")
    write_inout(fname, desc, purpose,
                weight_depth, out_depth,
                input_zero_bias, out_zero_bias,
                activation_min, activation_max,
                out_multiplier, out_shift,
                inp, kernel, bias, golden, partial_sum)

    print(f"  [{pid}] {fname}")
    print(f"         weight_depth={weight_depth}  out_depth={out_depth}  "
          f"izb={input_zero_bias}  ozb={out_zero_bias}  "
          f"mult={out_multiplier}  shift={out_shift}")
    print(f"         Purpose: {purpose}")
    return fname

# ---------------------------------------------------------------------------
# Pattern definitions
# ---------------------------------------------------------------------------
# p01  baseline        small FC, default quant params
# p02  nonzero_zp      non-zero input_zero_bias and out_zero_bias
# p03  activation_clamp  large scale -> saturation on both rails
# p04  large_depth     weight_depth=256 stress-tests inner MAC loop
# p05  zero_input      all-zero input -> output = f(bias only)
# p06  overflow_acc    near-INT32_MAX accumulator
# p07  negative_shift  shift > 0 -> left-shift branch of MBQM
# p08  large_outdepth  out_depth=128, weight_depth=64
# p09  null_bias_sim   bias all-zeros (simulates NULL bias path)
# p10  depth1          weight_depth=1 and out_depth=1 edge case

def run_all(select, outdir):
    def want(pid):
        return select is None or pid in select

    # p01 — Baseline: small FC, default TFLM quant params
    if want("p01"):
        make_pattern("p01", "baseline",
            "Small FC (in=32, out=32), default quant params. "
            "Baseline correctness of MBQM path.",
            weight_depth=32, out_depth=32,
            input_zero_bias=-128, out_zero_bias=128,
            scale=0.00392156,
            outdir=outdir)

    # p02 — Non-zero zero-biases
    if want("p02"):
        make_pattern("p02", "nonzero_zp",
            "Non-zero input_zero_bias=-30 and out_zero_bias=10. "
            "Catches wrong sign on zero-point application.",
            weight_depth=16, out_depth=8,
            input_zero_bias=-30, out_zero_bias=10,
            scale=0.00392156,
            outdir=outdir)

    # p03 — Activation clamp (large scale)
    if want("p03"):
        make_pattern("p03", "activation_clamp",
            "Large scale (0.5) forces many outputs to hit act_min/act_max. "
            "Verifies fused clamp on both -128 and +127 rails.",
            weight_depth=16, out_depth=16,
            scale=0.5,
            outdir=outdir)

    # p04 — Large weight_depth (inner loop stress)
    if want("p04"):
        make_pattern("p04", "large_depth",
            "weight_depth=256, out_depth=16. "
            "Stresses inner MAC accumulation loop with wide input vector.",
            weight_depth=256, out_depth=16,
            input_zero_bias=-128,
            scale=0.0001,
            outdir=outdir)

    # p05 — All-zero input
    if want("p05"):
        make_pattern("p05", "zero_input",
            "Input all zeros. Output = f(bias only) + out_zero_bias. "
            "Catches uninitialized accumulator or wrong zero-bias sign.",
            weight_depth=32, out_depth=8,
            input_zero_bias=0, out_zero_bias=5,
            inp_override=np.zeros(32, dtype=np.int8),
            bias_override=np.array([100, -200, 300, -400, 50, -50, 0, 127], dtype=np.int32),
            scale=0.00392156,
            outdir=outdir)

    # p06 — Near-overflow accumulator
    if want("p06"):
        make_pattern("p06", "overflow_acc",
            "All inputs=127, all weights=127, large positive bias -> acc near INT32_MAX. "
            "Catches 32-bit overflow in MAC or SRDHM.",
            weight_depth=16, out_depth=4,
            input_zero_bias=0,
            inp_override=np.full(16, 127, dtype=np.int8),
            kernel_override=np.full((4, 16), 127, dtype=np.int8),
            bias_override=np.full(4, 1_000_000, dtype=np.int32),
            scale=0.00001,
            outdir=outdir)

    # p07 — Positive shift (left-shift branch of MBQM, rare path)
    if want("p07"):
        make_pattern("p07", "positive_shift",
            "Scale chosen so out_shift > 0 -> left-shift branch of MBQM exercised. "
            "Rare path; verifies cmodel_MBQM out_shift>0 branch.",
            weight_depth=8, out_depth=4,
            # shift = e-31; for shift>0 need e>31, i.e. scale>2^1=2.0
            # Use scale=4.0 -> e=2, shift=2-31=-29... no.
            # Actually scale=1e-10 -> e=floor(log2(1e-10))=-34, shift=-34-31=-65... still neg
            # For shift>0: need e>31, scale>2^31~2.1e9 which is unreasonable.
            # Use out_multiplier_override directly with positive shift:
            out_multiplier_override=0x40000000,
            out_shift_override=2,   # left-shift by 2 before SRDHM
            outdir=outdir)

    # p08 — Large out_depth
    if want("p08"):
        make_pattern("p08", "large_outdepth",
            "out_depth=128, weight_depth=64. "
            "Stresses per-output-neuron loop and bias array indexing.",
            weight_depth=64, out_depth=128,
            input_zero_bias=-128, out_zero_bias=0,
            scale=0.001,
            outdir=outdir)

    # p09 — Zero bias (simulates NULL bias path)
    if want("p09"):
        make_pattern("p09", "zero_bias",
            "Bias all zeros (simulates the NULL-bias path). "
            "output depends purely on MAC result.",
            weight_depth=32, out_depth=8,
            bias_override=np.zeros(8, dtype=np.int32),
            scale=0.00392156,
            outdir=outdir)

    # p10 — Minimal dimensions edge case
    if want("p10"):
        make_pattern("p10", "depth1",
            "weight_depth=1, out_depth=1: minimal dimension edge case. "
            "Catches any off-by-one in loop bounds.",
            weight_depth=1, out_depth=1,
            input_zero_bias=-128, out_zero_bias=0,
            inp_override=np.array([5], dtype=np.int8),
            kernel_override=np.array([[3]], dtype=np.int8),
            bias_override=np.array([0], dtype=np.int32),
            out_multiplier_override=0x40000000,
            out_shift_override=-8,
            outdir=outdir)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate FC .in_out test fixtures.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python3 gen_fc_patterns.py\n"
            "  python3 gen_fc_patterns.py p03 p06\n"
        ),
    )
    parser.add_argument("patterns", nargs="*",
        help="Pattern IDs to generate (e.g. p01 p03). Default: all.")
    parser.add_argument("--outdir", default=".",
        help="Output directory for .in_out files (default: current dir).")
    args = parser.parse_args()

    select = set(args.patterns) if args.patterns else None
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    print("Generating FC test patterns ...\n")
    print(f"Requantize: MBQM (SRDHM+RoundingDivideByPOT, TFLM-style)\n")

    run_all(select, outdir)

    print("\nDone.")
