#pragma once

/**
 * x86-64 Assembly Trading Math - FMA-Optimized Calculations
 *
 * Provides assembly-optimized functions for market maker calculations
 * using FMA (Fused Multiply-Add) instructions for maximum performance.
 *
 * FMA Benefits:
 *   - Single instruction for (a * b) + c
 *   - Higher precision (no intermediate rounding)
 *   - Better throughput (1 FMA/cycle on most modern CPUs)
 *
 * Author: Adilson Dias
 * Date: December 2025
 */

#include <cstdint>
#include <cpuid.h>

namespace mm {
namespace asm_opt {

// Assembly function declarations (defined in trading_math_asm.S)
extern "C" {
    /**
     * Calculate fair value using FMA-optimized weighted average
     *
     * Formula:
     *   mid = (bid + ask) / 2
     *   weighted = (bid * bid_shares + ask * ask_shares) / total_shares
     *   fair_value = (mid + weighted) / 2
     *
     * @param bid_price Best bid price
     * @param ask_price Best ask price
     * @param bid_shares Bid size at best bid
     * @param ask_shares Ask size at best ask
     * @return Fair value (size-weighted mid-price)
     */
    double calculate_fair_value_asm(double bid_price, double ask_price,
                                    uint32_t bid_shares, uint32_t ask_shares);

    /**
     * Calculate bid and ask quote prices from fair value
     *
     * @param fair_value Calculated fair value
     * @param edge_bps Edge in basis points (e.g., 2.0 for 2 bps)
     * @param skew_bps Position skew in basis points
     * @param inventory_ratio Position / max_position (range -1 to +1)
     * @param bid_price_out Output: calculated bid price
     * @param ask_price_out Output: calculated ask price
     */
    void calculate_quote_prices_asm(double fair_value, double edge_bps,
                                    double skew_bps, double inventory_ratio,
                                    double* bid_price_out, double* ask_price_out);

    /**
     * Calculate PnL for a position
     *
     * @param current_price Current market price
     * @param entry_price Average entry price
     * @param shares Position size (positive=long, negative=short)
     * @return PnL (positive=profit, negative=loss)
     */
    double calculate_pnl_asm(double current_price, double entry_price, int shares);

    /**
     * Calculate VWAP (Volume Weighted Average Price)
     *
     * @param old_vwap Previous VWAP
     * @param old_volume Previous cumulative volume
     * @param new_price Price of new trade
     * @param new_volume Volume of new trade
     * @return Updated VWAP
     */
    double calculate_vwap_asm(double old_vwap, uint64_t old_volume,
                              double new_price, uint64_t new_volume);

    /**
     * Calculate weighted average entry price when adding to position
     *
     * @param old_avg Previous average entry price
     * @param old_shares Previous position size
     * @param price Price of new fill
     * @param new_shares Size of new fill (same sign as position direction)
     * @return New average entry price
     */
    double calculate_weighted_avg_entry_asm(double old_avg, int old_shares,
                                            double price, int new_shares);

    /**
     * Batch calculate fair values for multiple BBOs using AVX2
     *
     * @param bid_prices Array of bid prices (should be 32-byte aligned)
     * @param ask_prices Array of ask prices (should be 32-byte aligned)
     * @param bid_shares Array of bid sizes
     * @param ask_shares Array of ask sizes
     * @param fair_values Output array for fair values
     * @param count Number of BBOs to process
     */
    void batch_fair_value_asm(const double* bid_prices, const double* ask_prices,
                              const uint32_t* bid_shares, const uint32_t* ask_shares,
                              double* fair_values, size_t count);
}

/**
 * Check if FMA instructions are supported on this CPU
 */
inline bool has_fma() {
    unsigned int eax, ebx, ecx, edx;
    if (__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
        return (ecx & (1 << 12)) != 0;  // FMA bit in ECX
    }
    return false;
}

/**
 * Check if AVX2 is supported (for batch operations)
 */
inline bool has_avx2() {
    unsigned int eax, ebx, ecx, edx;
    if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
        return (ebx & (1 << 5)) != 0;  // AVX2 bit in EBX
    }
    return false;
}

/**
 * High-level wrapper class for trading math operations
 */
class TradingMathAsm {
public:
    TradingMathAsm()
        : has_fma_(has_fma()), has_avx2_(has_avx2()) {}

    bool hasFma() const { return has_fma_; }
    bool hasAvx2() const { return has_avx2_; }

    // Wrapper methods that use assembly when available
    double fairValue(double bid, double ask, uint32_t bid_sz, uint32_t ask_sz) const {
        return calculate_fair_value_asm(bid, ask, bid_sz, ask_sz);
    }

    void quotePrice(double fair_value, double edge_bps, double skew_bps,
                     double inventory_ratio, double& bid_out, double& ask_out) const {
        calculate_quote_prices_asm(fair_value, edge_bps, skew_bps, inventory_ratio,
                                   &bid_out, &ask_out);
    }

    double pnl(double current, double entry, int shares) const {
        return calculate_pnl_asm(current, entry, shares);
    }

    double vwap(double old_vwap, uint64_t old_vol, double price, uint64_t vol) const {
        return calculate_vwap_asm(old_vwap, old_vol, price, vol);
    }

    double avgEntry(double old_avg, int old_shares, double price, int new_shares) const {
        return calculate_weighted_avg_entry_asm(old_avg, old_shares, price, new_shares);
    }

private:
    bool has_fma_;
    bool has_avx2_;
};

}  // namespace asm_opt
}  // namespace mm
