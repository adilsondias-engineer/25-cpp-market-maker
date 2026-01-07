#pragma once
#include <string>
#include <memory>
#include "order_types.h"
#include "position_tracker.h"
#include "order_producer.h"
#include <spdlog/spdlog.h>
#include "spdlog/sinks/stdout_color_sinks.h"

#ifdef HAVE_XGBOOST
#include "model/itch_predictor.hpp"
#endif

namespace mm {

// XGBoost inference configuration
struct XGBoostConfig {
    bool enabled = false;
    std::string model_path = "/opt/trading/model/itch_predictor.ubj";
    bool use_gpu = true;
    int gpu_device_id = 0;
    double prediction_weight = 0.5;  // Weight of ML prediction vs mid-price
    int verbosity = 2;  // 0=silent, 1=warning, 2=info, 3=debug (default=info for GPU debugging)
    int max_iterations = 0;  // Max trees for inference (0=all, N=first N trees) - latency/accuracy tradeoff
};

enum class State {
    IDLE,
    CALCULATE,
    QUOTE,
    RISK_CHECK,
    ORDER_GEN,
    WAIT_FILL
};

class MarketMakerFSM {
public:
    struct Config {
        double min_spread_bps = 5.0;
        double edge_bps = 2.0;
        int max_position = 500;
        double position_skew_bps = 1.0;
        int quote_size = 100;
        double max_notional = 100000.0;
        bool enable_order_execution = false;
        std::string order_ring_path = "order_ring_mm";
        std::string fill_ring_path = "fill_ring_oe";

        // XGBoost inference settings
        XGBoostConfig xgboost;
    };

    explicit MarketMakerFSM(const Config& config);

    // Main event handler
    void onBboUpdate(const BBO& bbo);

    // Process fill notifications (call this in main loop)
    void processFills();

    // State getters
    State getState() const { return state_; }
    const PositionTracker& getPosition() const { return position_; }

#ifdef HAVE_XGBOOST
    // Load XGBoost model for inference
    bool loadModel(const std::string& model_path);
    bool isModelLoaded() const { return predictor_ != nullptr; }
#endif

private:
    // State handlers
    void handleCalculate(const BBO& bbo);
    void handleQuote(const BBO& bbo);
    void handleRiskCheck();
    void handleOrderGen();
    void handleWaitFill();
    
    // Helper functions
    double calculateFairValue(const BBO& bbo);
    Quote generateQuote(double fair_value, const BBO& bbo);
    bool checkRiskLimits(const Quote& quote);
    void simulateFill(const BBO& bbo);

#ifdef HAVE_XGBOOST
    // ML-enhanced fair value calculation
    double applyMLPrediction(double base_fair_value, const BBO& bbo);
#endif
    
    // State
    State state_;
    Config config_;
    PositionTracker position_;
    Quote current_quote_;
    double cached_fair_value_;
    std::shared_ptr<spdlog::logger> logger_;
    std::unique_ptr<OrderProducer> order_producer_;
    uint64_t order_sequence_;

#ifdef HAVE_XGBOOST
    // XGBoost predictor for price direction inference
    std::unique_ptr<itch::ITCHPredictor> predictor_;
    double last_prediction_;  // Cached prediction value

    // Inference timing statistics (for performance measurement)
    uint64_t inference_count_ = 0;      // Number of predictions made
    double inference_total_us_ = 0.0;   // Sum of all inference times
    double inference_min_us_ = 0.0;     // Minimum inference time
    double inference_max_us_ = 0.0;     // Maximum inference time
#endif
};

} // namespace mm
