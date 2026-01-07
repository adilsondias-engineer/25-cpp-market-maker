/**
 * Trading Math Benchmark - C++ vs x86-64 Assembly (FMA)
 *
 * Compares performance of market maker calculations:
 *   1. Fair value calculation
 *   2. Quote price generation
 *   3. PnL calculation
 *   4. VWAP calculation
 *
 * Build:
 *   g++ -O3 -march=native -o trading_math_benchmark trading_math_benchmark.cpp \
 *       ../src/asm/trading_math_asm.S -lpthread
 *
 * Run:
 *   ./trading_math_benchmark
 *
 * Author: Adilson Dias
 * Date: December 2025
 */

#include <cstdio>
#include <cstdint>
#include <cmath>
#include <chrono>
#include <random>

#include "../include/trading_math_asm.h"

// ============================================================================
// C++ Reference Implementations
// ============================================================================

double calculate_fair_value_cpp(double bid_price, double ask_price,
                                uint32_t bid_shares, uint32_t ask_shares) {
    double mid_price = (bid_price + ask_price) / 2.0;

    uint32_t total_shares = bid_shares + ask_shares;
    if (total_shares == 0) {
        return mid_price;
    }

    double weighted_price = (bid_price * bid_shares + ask_price * ask_shares)
                           / static_cast<double>(total_shares);

    return (mid_price + weighted_price) / 2.0;
}

void calculate_quote_prices_cpp(double fair_value, double edge_bps,
                                double skew_bps, double inventory_ratio,
                                double* bid_out, double* ask_out) {
    double edge = fair_value * (edge_bps / 10000.0);
    double skew = fair_value * (skew_bps / 10000.0) * inventory_ratio;

    *bid_out = fair_value - edge + skew;
    *ask_out = fair_value + edge + skew;
}

double calculate_pnl_cpp(double current_price, double entry_price, int shares) {
    return (current_price - entry_price) * shares;
}

double calculate_vwap_cpp(double old_vwap, uint64_t old_volume,
                          double new_price, uint64_t new_volume) {
    uint64_t total_volume = old_volume + new_volume;
    if (total_volume == 0) return new_price;

    return (old_vwap * old_volume + new_price * new_volume)
           / static_cast<double>(total_volume);
}

double calculate_weighted_avg_entry_cpp(double old_avg, int old_shares,
                                        double price, int new_shares) {
    int total_shares = old_shares + new_shares;
    if (total_shares == 0) return price;

    double total_cost = (old_avg * std::abs(old_shares)) + (price * std::abs(new_shares));
    return total_cost / std::abs(total_shares);
}

// ============================================================================
// Test Data
// ============================================================================
struct TestBBO {
    double bid_price;
    double ask_price;
    uint32_t bid_shares;
    uint32_t ask_shares;
};

// Realistic BBO data
TestBBO test_bbos[] = {
    {149.50, 149.52, 300, 500},    // AAPL
    {420.25, 420.30, 100, 200},    // MSFT
    {175.80, 175.85, 400, 400},    // GOOGL
    {95.10, 95.12, 1000, 800},     // AMD
};

// ============================================================================
// Benchmark Functions
// ============================================================================

template<typename Func>
double benchmark_ns(Func&& f, int iterations) {
    // Warm-up
    for (int i = 0; i < 10000; i++) {
        f();
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        f();
        asm volatile("" ::: "memory");  // Prevent optimization
    }
    auto end = std::chrono::high_resolution_clock::now();

    return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()
           / static_cast<double>(iterations);
}

void verify_correctness() {
    printf("===========================================\n");
    printf("Correctness Verification\n");
    printf("===========================================\n\n");

    // Fair value test
    for (const auto& bbo : test_bbos) {
        double cpp_fv = calculate_fair_value_cpp(bbo.bid_price, bbo.ask_price,
                                                  bbo.bid_shares, bbo.ask_shares);
        double asm_fv = mm::asm_opt::calculate_fair_value_asm(bbo.bid_price, bbo.ask_price,
                                                              bbo.bid_shares, bbo.ask_shares);
        bool match = std::abs(cpp_fv - asm_fv) < 1e-10;
        printf("Fair Value: bid=%.2f, ask=%.2f, bid_sz=%u, ask_sz=%u\n",
               bbo.bid_price, bbo.ask_price, bbo.bid_shares, bbo.ask_shares);
        printf("  C++: %.6f, ASM: %.6f - %s\n\n", cpp_fv, asm_fv, match ? "PASS" : "FAIL");
    }

    // Quote prices test
    double fair_value = 150.0;
    double edge_bps = 2.0;
    double skew_bps = 1.0;
    double inventory_ratio = 0.3;

    double cpp_bid, cpp_ask, asm_bid, asm_ask;
    calculate_quote_prices_cpp(fair_value, edge_bps, skew_bps, inventory_ratio,
                               &cpp_bid, &cpp_ask);
    mm::asm_opt::calculate_quote_prices_asm(fair_value, edge_bps, skew_bps, inventory_ratio,
                                            &asm_bid, &asm_ask);
    bool quote_match = std::abs(cpp_bid - asm_bid) < 1e-10 &&
                       std::abs(cpp_ask - asm_ask) < 1e-10;
    printf("Quote Prices: fair=%.2f, edge=%.1f bps, skew=%.1f bps, inv_ratio=%.2f\n",
           fair_value, edge_bps, skew_bps, inventory_ratio);
    printf("  C++: bid=%.6f, ask=%.6f\n", cpp_bid, cpp_ask);
    printf("  ASM: bid=%.6f, ask=%.6f - %s\n\n", asm_bid, asm_ask, quote_match ? "PASS" : "FAIL");

    // PnL test
    double current = 152.0;
    double entry = 150.0;
    int shares = 100;
    double cpp_pnl = calculate_pnl_cpp(current, entry, shares);
    double asm_pnl = mm::asm_opt::calculate_pnl_asm(current, entry, shares);
    bool pnl_match = std::abs(cpp_pnl - asm_pnl) < 1e-10;
    printf("PnL: current=%.2f, entry=%.2f, shares=%d\n", current, entry, shares);
    printf("  C++: %.2f, ASM: %.2f - %s\n\n", cpp_pnl, asm_pnl, pnl_match ? "PASS" : "FAIL");

    // VWAP test
    double old_vwap = 150.0;
    uint64_t old_vol = 1000;
    double new_price = 151.0;
    uint64_t new_vol = 500;
    double cpp_vwap = calculate_vwap_cpp(old_vwap, old_vol, new_price, new_vol);
    double asm_vwap = mm::asm_opt::calculate_vwap_asm(old_vwap, old_vol, new_price, new_vol);
    bool vwap_match = std::abs(cpp_vwap - asm_vwap) < 1e-10;
    printf("VWAP: old_vwap=%.2f, old_vol=%lu, new_price=%.2f, new_vol=%lu\n",
           old_vwap, old_vol, new_price, new_vol);
    printf("  C++: %.6f, ASM: %.6f - %s\n\n", cpp_vwap, asm_vwap, vwap_match ? "PASS" : "FAIL");

    // Weighted avg entry test
    double old_avg = 150.0;
    int old_shares_pos = 100;
    double price = 152.0;
    int new_shares_pos = 50;
    double cpp_avg = calculate_weighted_avg_entry_cpp(old_avg, old_shares_pos, price, new_shares_pos);
    double asm_avg = mm::asm_opt::calculate_weighted_avg_entry_asm(old_avg, old_shares_pos, price, new_shares_pos);
    bool avg_match = std::abs(cpp_avg - asm_avg) < 1e-10;
    printf("Weighted Avg Entry: old_avg=%.2f, old_shares=%d, price=%.2f, new_shares=%d\n",
           old_avg, old_shares_pos, price, new_shares_pos);
    printf("  C++: %.6f, ASM: %.6f - %s\n\n", cpp_avg, asm_avg, avg_match ? "PASS" : "FAIL");
}

void run_benchmarks() {
    printf("===========================================\n");
    printf("Performance Benchmarks (1M iterations)\n");
    printf("===========================================\n\n");

    const int ITERATIONS = 1000000;
    const TestBBO& bbo = test_bbos[0];  // Use AAPL data

    // Fair value benchmark
    volatile double result;

    double cpp_fv_ns = benchmark_ns([&]() {
        result = calculate_fair_value_cpp(bbo.bid_price, bbo.ask_price,
                                          bbo.bid_shares, bbo.ask_shares);
    }, ITERATIONS);

    double asm_fv_ns = benchmark_ns([&]() {
        result = mm::asm_opt::calculate_fair_value_asm(bbo.bid_price, bbo.ask_price,
                                                       bbo.bid_shares, bbo.ask_shares);
    }, ITERATIONS);

    printf("Fair Value Calculation:\n");
    printf("  C++: %.2f ns\n", cpp_fv_ns);
    printf("  ASM (FMA): %.2f ns\n", asm_fv_ns);
    printf("  Speedup: %.2fx (%.1f%% %s)\n\n",
           cpp_fv_ns / asm_fv_ns,
           std::abs(1.0 - asm_fv_ns / cpp_fv_ns) * 100.0,
           asm_fv_ns < cpp_fv_ns ? "faster" : "slower");

    // Quote prices benchmark
    double fair_value = 150.0;
    double edge_bps = 2.0, skew_bps = 1.0, inv_ratio = 0.3;
    double bid_out, ask_out;

    double cpp_quote_ns = benchmark_ns([&]() {
        calculate_quote_prices_cpp(fair_value, edge_bps, skew_bps, inv_ratio,
                                   &bid_out, &ask_out);
    }, ITERATIONS);

    double asm_quote_ns = benchmark_ns([&]() {
        mm::asm_opt::calculate_quote_prices_asm(fair_value, edge_bps, skew_bps, inv_ratio,
                                                &bid_out, &ask_out);
    }, ITERATIONS);

    printf("Quote Price Generation:\n");
    printf("  C++: %.2f ns\n", cpp_quote_ns);
    printf("  ASM (FMA): %.2f ns\n", asm_quote_ns);
    printf("  Speedup: %.2fx (%.1f%% %s)\n\n",
           cpp_quote_ns / asm_quote_ns,
           std::abs(1.0 - asm_quote_ns / cpp_quote_ns) * 100.0,
           asm_quote_ns < cpp_quote_ns ? "faster" : "slower");

    // PnL benchmark
    double current = 152.0, entry = 150.0;
    int shares = 100;

    double cpp_pnl_ns = benchmark_ns([&]() {
        result = calculate_pnl_cpp(current, entry, shares);
    }, ITERATIONS);

    double asm_pnl_ns = benchmark_ns([&]() {
        result = mm::asm_opt::calculate_pnl_asm(current, entry, shares);
    }, ITERATIONS);

    printf("PnL Calculation:\n");
    printf("  C++: %.2f ns\n", cpp_pnl_ns);
    printf("  ASM: %.2f ns\n", asm_pnl_ns);
    printf("  Speedup: %.2fx (%.1f%% %s)\n\n",
           cpp_pnl_ns / asm_pnl_ns,
           std::abs(1.0 - asm_pnl_ns / cpp_pnl_ns) * 100.0,
           asm_pnl_ns < cpp_pnl_ns ? "faster" : "slower");

    // VWAP benchmark
    double old_vwap = 150.0;
    uint64_t old_vol = 1000, new_vol = 500;
    double new_price = 151.0;

    double cpp_vwap_ns = benchmark_ns([&]() {
        result = calculate_vwap_cpp(old_vwap, old_vol, new_price, new_vol);
    }, ITERATIONS);

    double asm_vwap_ns = benchmark_ns([&]() {
        result = mm::asm_opt::calculate_vwap_asm(old_vwap, old_vol, new_price, new_vol);
    }, ITERATIONS);

    printf("VWAP Calculation:\n");
    printf("  C++: %.2f ns\n", cpp_vwap_ns);
    printf("  ASM (FMA): %.2f ns\n", asm_vwap_ns);
    printf("  Speedup: %.2fx (%.1f%% %s)\n\n",
           cpp_vwap_ns / asm_vwap_ns,
           std::abs(1.0 - asm_vwap_ns / cpp_vwap_ns) * 100.0,
           asm_vwap_ns < cpp_vwap_ns ? "faster" : "slower");

    // Avg entry benchmark
    double old_avg = 150.0;
    int old_sh = 100, new_sh = 50;
    double pr = 152.0;

    double cpp_avg_ns = benchmark_ns([&]() {
        result = calculate_weighted_avg_entry_cpp(old_avg, old_sh, pr, new_sh);
    }, ITERATIONS);

    double asm_avg_ns = benchmark_ns([&]() {
        result = mm::asm_opt::calculate_weighted_avg_entry_asm(old_avg, old_sh, pr, new_sh);
    }, ITERATIONS);

    printf("Weighted Average Entry:\n");
    printf("  C++: %.2f ns\n", cpp_avg_ns);
    printf("  ASM (FMA): %.2f ns\n", asm_avg_ns);
    printf("  Speedup: %.2fx (%.1f%% %s)\n\n",
           cpp_avg_ns / asm_avg_ns,
           std::abs(1.0 - asm_avg_ns / cpp_avg_ns) * 100.0,
           asm_avg_ns < cpp_avg_ns ? "faster" : "slower");
}

int main() {
    printf("===========================================\n");
    printf("Trading Math Assembly Benchmark\n");
    printf("FMA (Fused Multiply-Add) Optimization\n");
    printf("===========================================\n\n");

    printf("CPU Features:\n");
    printf("  FMA:  %s\n", mm::asm_opt::has_fma() ? "YES" : "NO");
    printf("  AVX2: %s\n\n", mm::asm_opt::has_avx2() ? "YES" : "NO");

    verify_correctness();
    run_benchmarks();

    printf("===========================================\n");
    printf("Benchmark complete!\n");

    return 0;
}
