# FMA Assembly Optimization - Trading Math Functions

## Overview

This document describes the x86-64 assembly optimizations implemented for trading math calculations in the Market Maker (Project 25), using FMA (Fused Multiply-Add) instructions.

## Target CPU Features

- **FMA3**: Fused Multiply-Add (Haswell 2013+, AMD Piledriver 2012+)
- **AVX2**: Advanced Vector Extensions 2 (for future batch processing)

## Optimized Functions

### 1. Fair Value Calculation (`calculate_fair_value_asm`)

Calculates fair value from BBO data using a weighted mid-price formula.

**Algorithm:**
```
mid_price = (bid + ask) / 2
weighted_price = (bid * bid_shares + ask * ask_shares) / total_shares
fair_value = (mid_price + weighted_price) / 2
```

**FMA Optimization:**
```asm
# Traditional: vmulsd + vaddsd = 2 instructions, 8+ cycles
vmulsd  xmm6, xmm0, xmm2      # bid * bid_shares
vmulsd  xmm7, xmm1, xmm3      # ask * ask_shares
vaddsd  xmm6, xmm6, xmm7      # sum

# FMA: vfmadd231sd = 1 instruction, 5 cycles
vmulsd  xmm6, xmm0, xmm2      # bid * bid_shares
vfmadd231sd xmm6, xmm1, xmm3  # xmm6 += ask * ask_shares (FMA!)
```

**Benchmark Results:**
| Implementation | Time (ns) | Speedup |
|---------------|-----------|---------|
| C++ (-O3)     | 1.90      | baseline |
| ASM (FMA)     | 1.83      | **1.04x (3.6% faster)** |

### 2. Quote Price Generation (`calculate_quote_prices_asm`)

Generates bid/ask quotes from fair value with edge and inventory skew.

**Algorithm:**
```
edge = fair_value * (edge_bps / 10000)
skew = fair_value * (skew_bps / 10000) * inventory_ratio
bid = fair_value - edge + skew
ask = fair_value + edge + skew
```

**Benchmark Results:**
| Implementation | Time (ns) | Result |
|---------------|-----------|--------|
| C++ (-O3)     | 1.17      | faster |
| ASM (FMA)     | 3.16      | slower (function call overhead) |

### 3. PnL Calculation (`calculate_pnl_asm`)

Simple: `(current_price - entry_price) * shares`

**Benchmark Results:**
| Implementation | Time (ns) | Result |
|---------------|-----------|--------|
| C++ (-O3)     | 0.36      | faster |
| ASM (FMA)     | 1.43      | slower (too simple for external call) |

### 4. VWAP Calculation (`calculate_vwap_asm`)

Volume-Weighted Average Price update.

**Benchmark Results:**
| Implementation | Time (ns) | Result |
|---------------|-----------|--------|
| C++ (-O3)     | 1.51      | faster |
| ASM (FMA)     | 2.06      | slower |

### 5. Weighted Average Entry (`calculate_weighted_avg_entry_asm`)

Updates average entry price when adding to position.

**Benchmark Results:**
| Implementation | Time (ns) | Result |
|---------------|-----------|--------|
| C++ (-O3)     | 1.51      | faster |
| ASM (FMA)     | 2.06      | slower |

## Key Findings

### When FMA Assembly Helps

1. **Complex calculations with multiple FMA opportunities**: The fair value calculation benefits from FMA because it has dependent multiply-add chains that can be fused.

2. **Hot inner loops**: FMA instructions would shine in batch processing where function call overhead is amortized.

### When FMA Assembly Doesn't Help

1. **Simple operations**: PnL calculation is just `(a-b)*c` - too simple to benefit from external function call overhead (~1-2ns).

2. **GCC optimization**: Modern GCC (13+) with `-O3 -march=native` already:
   - Inlines simple functions
   - Uses FMA instructions automatically when beneficial
   - Applies constant propagation and dead code elimination

3. **Function call overhead**: External assembly functions incur:
   - Call instruction (~1 cycle)
   - Stack frame setup
   - Register saves/restores (if needed)
   - Return instruction (~1 cycle)

## Compiler Output Comparison

GCC 15 with `-O3 -march=native` generates near-optimal code:

```asm
# GCC output for fair_value calculation (simplified)
vaddsd  xmm4, xmm0, xmm1      # mid = bid + ask
vmulsd  xmm4, xmm4, HALF      # mid /= 2
vmulsd  xmm0, xmm0, xmm2      # bid * bid_shares
vfmadd231sd xmm0, xmm1, xmm3  # += ask * ask_shares (GCC uses FMA!)
```

GCC automatically recognizes FMA opportunities and emits `vfmadd*` instructions.

## Recommendations

### For This Project

1. **Keep the FMA assembly for fair value**: The 3.6% improvement is measurable and the function is called frequently.

2. **Use C++ for simpler calculations**: Let GCC inline PnL, VWAP, and other simple functions.

3. **Consider inlining assembly**: Use `asm volatile` inline assembly for critical paths to avoid function call overhead.

### For Maximum Performance

```cpp
// Option 1: Trust the compiler (usually best)
inline double calculate_fair_value(double bid, double ask,
                                    uint32_t bid_sz, uint32_t ask_sz) {
    double mid = (bid + ask) * 0.5;
    double weighted = (bid * bid_sz + ask * ask_sz) / (bid_sz + ask_sz);
    return (mid + weighted) * 0.5;
}

// Option 2: Inline assembly for guaranteed FMA
inline double calculate_fair_value_inline(double bid, double ask,
                                          double bid_sz, double ask_sz) {
    double result;
    asm volatile(
        "vaddsd %1, %2, %%xmm4\n\t"       // mid = bid + ask
        "vmulsd %5, %%xmm4, %%xmm4\n\t"   // mid *= 0.5
        "vmulsd %1, %3, %%xmm5\n\t"       // bid * bid_sz
        "vfmadd231sd %2, %4, %%xmm5\n\t"  // += ask * ask_sz
        "vaddsd %3, %4, %%xmm6\n\t"       // total_sz
        "vdivsd %%xmm6, %%xmm5, %%xmm5\n\t"
        "vaddsd %%xmm4, %%xmm5, %%xmm5\n\t"
        "vmulsd %5, %%xmm5, %0"
        : "=x"(result)
        : "x"(bid), "x"(ask), "x"(bid_sz), "x"(ask_sz), "m"(HALF)
        : "xmm4", "xmm5", "xmm6"
    );
    return result;
}
```

## Build Integration

The assembly files are integrated into the build:

```cmake
# CMakeLists.txt
set(ASM_SOURCES
    src/asm/trading_math_asm.S
)
enable_language(ASM)
add_executable(market_maker ${SOURCES} ${ASM_SOURCES})
```

## CPU Feature Detection

Runtime detection ensures compatibility:

```cpp
inline bool has_fma() {
    unsigned int eax, ebx, ecx, edx;
    if (__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
        return (ecx & (1 << 12)) != 0;  // FMA bit
    }
    return false;
}
```

## Conclusion

FMA assembly optimization provides a **3.6% improvement** for the fair value calculation - the most complex trading math function. However, for simpler operations, modern GCC's optimization is sufficient or better due to function inlining.

The primary value of this assembly work is:
1. **Educational**: Understanding CPU instructions and calling conventions
2. **Documented baseline**: Documents exploration of low-level optimization
3. **Reference implementation**: Shows what optimal FMA usage looks like
4. **Portfolio demonstration**: Shows systems programming depth

For production, consider inline assembly or trust GCC's `-O3 -march=native` optimization.

---

**Author**: Adilson Dias
**Date**: December 2025
**Benchmark Environment**: Intel i9-14900K, GCC 15, Ubuntu 24.04
