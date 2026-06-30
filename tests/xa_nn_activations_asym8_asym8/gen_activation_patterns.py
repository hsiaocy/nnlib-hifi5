#!/usr/bin/env python3
"""
gen_activation_patterns.py
==========================
Generate .in_out fixture files for xa_nn_activations_asym8_asym8 tests.

Op family: asym8->asym8 elementwise activation (ReLU / ReLU6 / generic clamp).

This op is purely elementwise — there is NO MAC and NO partial_sum.
The .in_out fixture format therefore omits:
  kernel, bias, out_multiplier, out_shift, partial_sum
Only the fields that apply to this op are written:
  tflite_single_rounding  (always 0; clamp needs no requantize path)
  num_elements
  input_zero_point
  output_zero_point
  activation_min
  activation_max
  int8_t input[N]
  int8_t golden[N]

Field naming follows the conv2d .in_out convention where fields overlap
(tflite_single_rounding, activation_min, activation_max).

Usage:
    python3 gen_activation_patterns.py            # all patterns
    python3 gen_activation_patterns.py p01 p03    # selective

Patterns generated:
    p01_relu_zero_zp          — ReLU with zero_point=0
    p02_relu_nonzero_zp       — ReLU with non-zero zero_point
    p03_relu6_zero_zp         — ReLU6 with zero_point=0 (scale chosen so 6 maps to 42)
    p04_relu6_nonzero_zp      — ReLU6 with non-zero zero_point
    p05_clamp_full_range      — Generic clamp, full int8 range [-128, 127] (identity)
    p06_clamp_narrow          — Generic clamp, narrow window [-10, 10]
    p07_all_negative_input    — All-negative inputs, ReLU should saturate all to min
    p08_all_positive_input    — All-positive inputs below relu6, all pass unchanged
    p09_boundary_values       — Inputs exactly at activation_min and activation_max
    p10_large_vector          — Large vector (N=1024) exercises loop throughput

Output directory: ./patterns/
"""

import argparse
import numpy as np
import os
import sys

# ---------------------------------------------------------------------------
# NOTE: tflite_single_rounding is always 0 for this op.
# The clamp activation performs no requantization; the flag is kept in the
# fixture header only for tooling uniformity with the conv2d .in_out format.
# ---------------------------------------------------------------------------
_TFLITE_SINGLE_ROUNDING = 0

RNG = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# Reference implementation
# ---------------------------------------------------------------------------

def clamp(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, x))

def activation_clamp_reference(inp, activation_min, activation_max):
    """Elementwise clamp.  inp is np.ndarray[int8].  Returns np.ndarray[int8]."""
    out = np.empty_like(inp)
    for i in range(len(inp)):
        out[i] = np.int8(clamp(int(inp[i]), activation_min, activation_max))
    return out

# ---------------------------------------------------------------------------
# .in_out writer
# ---------------------------------------------------------------------------

def arr8(name, data):
    flat = np.asarray(data, dtype=np.int8).flatten()
    vals = ", ".join(str(int(v)) for v in flat)
    return f"int8_t {name}[{len(flat)}] = {{{vals}}};\n"

def write_inout(path, cfg, inp, golden):
    """
    Write a fixture file.

    cfg fields written as 'int <name> = <value>;':
      tflite_single_rounding, num_elements, input_zero_point,
      output_zero_point, activation_min, activation_max

    Arrays written:
      int8_t input[N]
      int8_t golden[N]

    Fields omitted for this op (no MAC, no requantize, no partial_sum):
      kernel, bias, out_multiplier, out_shift, partial_sum
    """
    with open(path, "w") as f:
        f.write(f"// Pattern: {cfg['desc']}\n")
        f.write(f"// Purpose: {cfg['purpose']}\n")
        f.write(f"// Requantize mode: N/A — clamp activation has no requantize step\n")
        f.write(f"// NOTE: partial_sum omitted — this is an elementwise op with no MAC\n\n")
        # Mode flag first (uniformity with conv2d tooling)
        f.write(f"int tflite_single_rounding = {_TFLITE_SINGLE_ROUNDING};\n")
        # Scalar params
        for k in ("num_elements", "input_zero_point", "output_zero_point",
                  "activation_min", "activation_max"):
            f.write(f"int {k} = {cfg[k]};\n")
        f.write("\n")
        f.write(arr8("input",  inp))
        f.write(arr8("golden", golden))

# ---------------------------------------------------------------------------
# Pattern builder
# ---------------------------------------------------------------------------

def make_pattern(pid, desc, purpose, inp, activation_min, activation_max,
                 input_zero_point=0, output_zero_point=0, out_dir="patterns"):
    inp = np.asarray(inp, dtype=np.int8)
    golden = activation_clamp_reference(inp, activation_min, activation_max)
    cfg = dict(
        desc=desc,
        purpose=purpose,
        num_elements=len(inp),
        input_zero_point=input_zero_point,
        output_zero_point=output_zero_point,
        activation_min=activation_min,
        activation_max=activation_max,
    )
    os.makedirs(out_dir, exist_ok=True)
    fname = os.path.join(out_dir, f"{pid}_{desc}.in_out")
    write_inout(fname, cfg, inp, golden)
    n_clamped = int(np.sum((inp < activation_min) | (inp > activation_max)))
    print(f"  [{pid}] {fname}  N={len(inp)}, "
          f"clamp=[{activation_min},{activation_max}], "
          f"clamped={n_clamped}/{len(inp)}")
    print(f"         Purpose : {purpose}")
    return fname

# ---------------------------------------------------------------------------
# Pattern definitions
# ---------------------------------------------------------------------------

def define_patterns(out_dir, select):
    def want(pid):
        return select is None or pid in select

    # ── P01 ── ReLU with zero_point=0
    # activation_min = 0 (quantized(0) when zp=0), activation_max = 127
    # Mix of negative and positive inputs.  Negative values must clamp to 0.
    if want("p01"):
        inp = RNG.integers(-128, 128, size=64, dtype=np.int8)
        make_pattern(
            "p01", "relu_zero_zp",
            "ReLU: zero_point=0, clamp=[0,127]. Negative inputs must map to 0.",
            inp, activation_min=0, activation_max=127,
            input_zero_point=0, output_zero_point=0, out_dir=out_dir)

    # ── P02 ── ReLU with non-zero zero_point
    # zero_point = -30 (common in practice; quantized(0.0) = -30).
    # activation_min = -30 (= quantized(0)), activation_max = 127.
    # Tests that the harness correctly uses quantized bounds, not float 0.
    if want("p02"):
        inp = RNG.integers(-128, 128, size=64, dtype=np.int8)
        zp = -30
        make_pattern(
            "p02", "relu_nonzero_zp",
            "ReLU: zero_point=-30, activation_min=-30. "
            "Verifies clamp uses quantized(0)=zp, not literal 0.",
            inp, activation_min=zp, activation_max=127,
            input_zero_point=zp, output_zero_point=zp, out_dir=out_dir)

    # ── P03 ── ReLU6 with zero_point=0
    # scale = 6/170 so quantized(6) ≈ 42.  activation_min=0, activation_max=42.
    # Tests the upper clamp rail (many outputs above 42 must saturate).
    if want("p03"):
        inp = RNG.integers(-128, 128, size=64, dtype=np.int8)
        # With zp=0, scale=6/170: quantized(6) = round(6 / (6/170)) = round(170) = 127 ...
        # use a simpler formulation: scale such that 6 maps to 60
        # scale = 6/60 = 0.1; quantized(x) = round(x/scale) + zp
        # quantized(6) = round(6/0.1) + 0 = 60
        relu6_max = 60  # quantized value of 6.0 with scale=0.1, zp=0
        make_pattern(
            "p03", "relu6_zero_zp",
            "ReLU6: zero_point=0, activation_max=60 (=quantized(6) at scale=0.1). "
            "Both lower (0) and upper (60) rails exercised.",
            inp, activation_min=0, activation_max=relu6_max,
            input_zero_point=0, output_zero_point=0, out_dir=out_dir)

    # ── P04 ── ReLU6 with non-zero zero_point
    # zp = 10, scale = 0.04; quantized(0) = 10, quantized(6) = 10 + 150 = 160 > 127
    # so clamp to 127 for upper rail.  activation_min=10, activation_max=127.
    # Actually keep it realistic: zp=10, scale=0.1; quantized(6)=10+60=70.
    if want("p04"):
        inp = RNG.integers(-128, 128, size=64, dtype=np.int8)
        zp = 10
        relu6_max_q = zp + 60  # quantized(6) = 10 + 60 = 70 (scale=0.1, zp=10)
        make_pattern(
            "p04", "relu6_nonzero_zp",
            "ReLU6: zero_point=10, activation_min=10, activation_max=70 "
            "(=quantized(6) at scale=0.1, zp=10). Both rails stressed.",
            inp, activation_min=zp, activation_max=relu6_max_q,
            input_zero_point=zp, output_zero_point=zp, out_dir=out_dir)

    # ── P05 ── Generic clamp — full int8 range (identity)
    # activation_min=-128, activation_max=127: no element should be clamped.
    # Golden must equal input exactly.  Verifies the passthrough case.
    if want("p05"):
        inp = RNG.integers(-128, 128, size=64, dtype=np.int8)
        make_pattern(
            "p05", "clamp_full_range",
            "Clamp [-128,127]: identity, no element clamped. "
            "Golden must equal input exactly.",
            inp, activation_min=-128, activation_max=127,
            input_zero_point=0, output_zero_point=0, out_dir=out_dir)

    # ── P06 ── Generic clamp — narrow window
    # activation_min=-10, activation_max=10: most elements should be clamped.
    if want("p06"):
        inp = RNG.integers(-128, 128, size=64, dtype=np.int8)
        make_pattern(
            "p06", "clamp_narrow",
            "Clamp [-10,10]: majority of elements expected to hit the rails. "
            "Tests heavy-saturation throughput.",
            inp, activation_min=-10, activation_max=10,
            input_zero_point=0, output_zero_point=0, out_dir=out_dir)

    # ── P07 ── All-negative input, ReLU
    # Every element is negative → every output must equal activation_min (=0).
    # Any implementation that processes elements selectively will show errors.
    if want("p07"):
        inp = RNG.integers(-128, -1, size=64, dtype=np.int8)
        make_pattern(
            "p07", "all_negative_relu",
            "All inputs negative; ReLU activation_min=0 saturates every element. "
            "Output must be all-zeros.",
            inp, activation_min=0, activation_max=127,
            input_zero_point=0, output_zero_point=0, out_dir=out_dir)

    # ── P08 ── All-positive input below ReLU6 upper rail
    # Every element in [1, 60]; ReLU6 with activation_max=60 → all pass unchanged.
    if want("p08"):
        inp = RNG.integers(1, 61, size=64, dtype=np.int8)
        make_pattern(
            "p08", "all_positive_below_relu6",
            "All inputs in [1,60]; ReLU6 clamp=[0,60] does not clip any. "
            "Output must equal input exactly.",
            inp, activation_min=0, activation_max=60,
            input_zero_point=0, output_zero_point=0, out_dir=out_dir)

    # ── P09 ── Boundary values exactly at activation_min and activation_max
    # Constructs an array with values at -128, activation_min, activation_max, 127
    # and values just outside each boundary.  Verifies off-by-one in clamp.
    if want("p09"):
        act_min, act_max = -20, 50
        boundary_vals = np.array([
            -128, act_min - 1, act_min, act_min + 1,
            act_max - 1, act_max, act_max + 1, 127,
            0, -1, 1, act_min, act_max, -128, 127, act_min + 5,
        ], dtype=np.int8)
        make_pattern(
            "p09", "boundary_values",
            "Values exactly at and adjacent to activation_min=-20, activation_max=50. "
            "Catches off-by-one errors in clamp comparison.",
            boundary_vals, activation_min=act_min, activation_max=act_max,
            input_zero_point=0, output_zero_point=0, out_dir=out_dir)

    # ── P10 ── Large vector (N=1024) — loop throughput
    # Exercises any unrolling or SIMD path; also validates large allocation/parse.
    if want("p10"):
        inp = RNG.integers(-128, 128, size=1024, dtype=np.int8)
        make_pattern(
            "p10", "large_vector",
            "N=1024 elements; ReLU clamp=[0,127]. "
            "Stresses loop throughput and large fixture parsing.",
            inp, activation_min=0, activation_max=127,
            input_zero_point=0, output_zero_point=0, out_dir=out_dir)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate asym8->asym8 activation .in_out test fixtures.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python3 gen_activation_patterns.py\n"
            "  python3 gen_activation_patterns.py p01 p03 p09\n"
        ),
    )
    parser.add_argument(
        "patterns", nargs="*",
        help="Optional pattern IDs to generate (e.g. p01 p03). Default: all.")
    parser.add_argument(
        "--out-dir", default="patterns",
        help="Directory to write .in_out files into (default: patterns/)")
    args = parser.parse_args()

    select = set(args.patterns) if args.patterns else None

    print("Generating activation test patterns ...\n")
    define_patterns(out_dir=args.out_dir, select=select)
    print("\nDone.")
