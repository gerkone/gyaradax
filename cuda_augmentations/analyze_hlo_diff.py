#!/usr/bin/env python3
"""Analyze HLO differences between fp32 and fp64 Z2Z implementations."""
import re
from pathlib import Path
from collections import defaultdict

root = Path(__file__).parent.parent
fp64_path = root / "hlo_dumps" / "z2z_fp64.hlo.txt"
fp32_path = root / "hlo_dumps" / "z2z_fp32.hlo.txt"

print(f"\n{'='*80}")
print(f"HLO Analysis: fp64 vs fp32 Z2Z")
print(f"{'='*80}")

with open(fp64_path, "r") as f:
    hlo_fp64 = f.read()
with open(fp32_path, "r") as f:
    hlo_fp32 = f.read()

def extract_operations(hlo_text, label):
    """Extract and categorize operations from HLO text."""
    ops = defaultdict(list)
    lines = hlo_text.split('\n')
    
    for i, line in enumerate(lines):
        if '= fft[' in line or 'fft.' in line:
            ops['fft'].append((i, line.strip()))
        elif 'convert_element_type' in line:
            ops['convert'].append((i, line.strip()))
        elif 'multiply' in line and '=' in line:
            ops['multiply'].append((i, line.strip()))
        elif 'add' in line and '=' in line and 'broadcast' not in line:
            ops['add'].append((i, line.strip()))
        elif 'subtract' in line:
            ops['subtract'].append((i, line.strip()))
        elif 'negate' in line:
            ops['negate'].append((i, line.strip()))
        elif 'complex' in line and '=' in line:
            ops['complex'].append((i, line.strip()))
        elif 'real' in line and '=' in line:
            ops['real'].append((i, line.strip()))
        elif 'imag' in line and '=' in line:
            ops['imag'].append((i, line.strip()))
    
    return ops

def count_dtype_usage(hlo_text, dtype):
    """Count occurrences of specific dtype."""
    pattern = rf'f{32 if dtype == "f32" else 64}\b'
    return len(re.findall(pattern, hlo_text))

def find_fft_details(hlo_text):
    """Find FFT operations with their dtypes."""
    fft_ops = []
    lines = hlo_text.split('\n')
    for i, line in enumerate(lines):
        if '= fft[' in line:
            if i+1 < len(lines):
                next_line = lines[i+1]
                if 'f64' in next_line or 'complex128' in next_line:
                    fft_ops.append(('fp64', line.strip()))
                elif 'f32' in next_line or 'complex64' in next_line:
                    fft_ops.append(('fp32', line.strip()))
                else:
                    fft_ops.append(('unknown', line.strip()))
    return fft_ops

print("\n1. Operation Count Summary")
print("-" * 60)

ops_fp64 = extract_operations(hlo_fp64, "fp64")
ops_fp32 = extract_operations(hlo_fp32, "fp32")

all_op_types = set(ops_fp64.keys()) | set(ops_fp32.keys())
print(f"{'Operation':15s} | {'fp64 count':12s} | {'fp32 count':12s} | {'Diff':>8s}")
print("-" * 60)
for op_type in sorted(all_op_types):
    c64 = len(ops_fp64[op_type])
    c32 = len(ops_fp32[op_type])
    diff = c32 - c64
    diff_str = f"+{diff}" if diff > 0 else str(diff)
    print(f"{op_type:15s} | {c64:12d} | {c32:12d} | {diff_str:>8s}")

print("\n2. FFT Operation Details")
print("-" * 60)

fft_fp64 = find_fft_details(hlo_fp64)
fft_fp32 = find_fft_details(hlo_fp32)

print(f"\nfp64 FFTs ({len(fft_fp64)} total):")
for dtype, op in fft_fp64[:5]:
    print(f"  [{dtype}] {op[:80]}...")

print(f"\nfp32 FFTs ({len(fft_fp32)} total):")
for dtype, op in fft_fp32[:5]:
    print(f"  [{dtype}] {op[:80]}...")

print("\n3. Dtype Usage Analysis")
print("-" * 60)

f64_count_fp64 = count_dtype_usage(hlo_fp64, "f64")
f32_count_fp64 = count_dtype_usage(hlo_fp64, "f32")
f64_count_fp32 = count_dtype_usage(hlo_fp32, "f64")
f32_count_fp32 = count_dtype_usage(hlo_fp32, "f32")

print(f"\nIn fp64 HLO:")
print(f"  f64 references: {f64_count_fp64:,}")
print(f"  f32 references: {f32_count_fp64:,}")

print(f"\nIn fp32 HLO:")
print(f"  f64 references: {f64_count_fp32:,}")
print(f"  f32 references: {f32_count_fp32:,}")

print("\n4. Key Differences - Extra Operations in fp32")
print("-" * 60)

extra_converts = set(ops_fp32['convert']) - set(ops_fp64['convert'])
if extra_converts:
    print(f"\nfp32 has {len(extra_converts)} extra convert operations:")
    for line_no, op in list(extra_converts)[:10]:
        print(f"  Line {line_no}: {op[:100]}")

extra_ffts = set(ops_fp32['fft']) - set(ops_fp64['fft'])
if extra_ffts:
    print(f"\nfp32 has {len(extra_ffts)} extra FFT operations:")
    for line_no, op in list(extra_ffts)[:10]:
        print(f"  Line {line_no}: {op[:100]}")

print("\n5. File Size Comparison")
print("-" * 60)

import os
fp64_size = os.path.getsize(fp64_path)
fp32_size = os.path.getsize(fp32_path)

print(f"fp64 HLO: {fp64_size:>10,} bytes")
print(f"fp32 HLO: {fp32_size:>10,} bytes")
print(f"Difference: {fp32_size - fp64_size:+,} bytes ({(fp32_size/fp64_size - 1)*100:+.1f}%)")

print(f"\n{'='*80}")
print("Analysis complete!")
print(f"{'='*80}\n")
