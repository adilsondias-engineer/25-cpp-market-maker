/**
 * @file test_itch_predictor.cpp
 * @brief Test program for ITCH Price Direction Predictor (XGBoost C API)
 *
 * Build:
 *   g++ -std=c++20 -O2 -I include \
 *       $(pkg-config --cflags xgboost) \
 *       -o test_predictor test_itch_predictor.cpp \
 *       $(pkg-config --libs xgboost)
 *
 * Run:
 *   ./test_predictor [model_path]
 *   ./test_predictor itch_predictor.ubj
 */

#include "itch_predictor.hpp"
#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <chrono>

using namespace itch;

/**
 * @brief Print feature vector
 */
void print_features(const FeatureVector &fv)
{
    std::cout << "Features:\n";
    for (size_t i = 0; i < FeatureVector::NUM_FEATURES; ++i)
    {
        std::cout << "  " << std::setw(20) << feature_name(i) << ": "
                  << std::fixed << std::setprecision(4) << fv[i] << "\n";
    }
}

/**
 * @brief Test feature extraction
 */
void test_feature_extraction()
{
    std::cout << "=== Feature Extraction Test ===\n\n";

    FeatureExtractor extractor;

    // Single snapshot
    OrderBookSnapshot snapshot{100.50, 100.52, 500, 300, 0};

    std::cout << "Order Book Snapshot:\n";
    std::cout << "  Bid: $" << snapshot.bid_price << " x " << snapshot.bid_size << "\n";
    std::cout << "  Ask: $" << snapshot.ask_price << " x " << snapshot.ask_size << "\n";
    std::cout << "  Mid: $" << snapshot.mid_price() << "\n";
    std::cout << "  Spread: $" << snapshot.spread()
              << " (" << snapshot.spread_bps() << " bps)\n";
    std::cout << "  Order Imbalance: " << snapshot.order_imbalance() << "\n\n";

    auto features = extractor.update(snapshot);
    print_features(features);
}

/**
 * @brief Test rolling buffer
 */
void test_rolling_buffer()
{
    std::cout << "\n=== Rolling Buffer Test ===\n\n";

    RollingBuffer<double, 5> buffer;

    std::cout << "Pushing values 1-7 into buffer of size 5:\n";
    for (int i = 1; i <= 7; ++i)
    {
        buffer.push(static_cast<double>(i));
        std::cout << "  Push " << i << ": count=" << buffer.count()
                  << ", sum=" << buffer.sum()
                  << ", mean=" << buffer.mean()
                  << ", get(0)=" << buffer.get(0) << "\n";
    }

    std::cout << "\nFinal buffer contents (most recent first):\n";
    for (size_t i = 0; i < buffer.count(); ++i)
    {
        std::cout << "  [" << i << " ticks ago]: " << buffer.get(i) << "\n";
    }
}

/**
 * @brief Test XGBoost model loading and prediction
 */
void test_model_loading(const std::string &model_path)
{
    std::cout << "\n=== XGBoost Model Test ===\n\n";

    XGBoostModel model;

    std::cout << "Loading model: " << model_path << "\n";

    if (!model.load(model_path))
    {
        std::cout << "  ERROR: Failed to load model: " << model.last_error() << "\n";
        std::cout << "  (This is expected if no model file is provided)\n";
        return;
    }

    std::cout << "  Model loaded successfully!\n\n";

    // Create test feature vector
    FeatureVector fv;
    fv[0] = 150.00f;    // bid_price
    fv[1] = 150.02f;    // ask_price
    fv[2] = 1500.0f;    // bid_size
    fv[3] = 500.0f;     // ask_size
    fv[4] = 0.02f;      // spread
    fv[5] = 1.33f;      // spread_bps
    fv[6] = 150.01f;    // mid_price
    fv[7] = 0.5f;       // order_imbalance (buy pressure)
    fv[8] = 2.0f;       // momentum_1 (slight upward)
    fv[9] = 5.0f;       // momentum_5 (upward trend)
    fv[10] = 0.4f;      // imbalance_ma
    fv[11] = 150.0075f; // microprice

    std::cout << "Test features (buy pressure scenario):\n";
    print_features(fv);

    std::vector<float> probs;
    if (model.predict_proba(fv, probs))
    {
        std::cout << "\nPrediction probabilities:\n";
        std::cout << "  P(DOWN):    " << std::fixed << std::setprecision(4) << probs[0] << "\n";
        std::cout << "  P(NEUTRAL): " << probs[1] << "\n";
        std::cout << "  P(UP):      " << probs[2] << "\n";

        int pred_class = model.predict_class(fv);
        std::cout << "\nPredicted class: " << pred_class << " (";
        switch (pred_class)
        {
        case 0:
            std::cout << "DOWN";
            break;
        case 1:
            std::cout << "NEUTRAL";
            break;
        case 2:
            std::cout << "UP";
            break;
        default:
            std::cout << "UNKNOWN";
            break;
        }
        std::cout << ")\n";
    }
    else
    {
        std::cout << "  ERROR: Prediction failed: " << model.last_error() << "\n";
    }
}

/**
 * @brief Test complete predictor with model
 */
void test_predictor_with_model(const std::string &model_path)
{
    std::cout << "\n=== Complete Predictor Test ===\n\n";

    ITCHPredictor predictor;

    if (!predictor.load_model(model_path))
    {
        std::cout << "Failed to load model: " << predictor.last_error() << "\n";
        std::cout << "(Skipping model-based tests)\n";
        return;
    }

    std::cout << "Model loaded successfully!\n\n";

    // Simulate order book updates
    struct TestCase
    {
        double bid, ask;
        uint32_t bid_sz, ask_sz;
        const char *description;
    };

    std::vector<TestCase> test_cases = {
        {150.00, 150.02, 1000, 1000, "Balanced book"},
        {150.01, 150.02, 1500, 500, "Buy pressure"},
        {150.02, 150.03, 1800, 200, "Strong buy pressure"},
        {150.03, 150.04, 2000, 100, "Very strong buy"},
        {150.04, 150.05, 1500, 500, "Continued buy"},
        {150.03, 150.05, 500, 1500, "Sell pressure"},
        {150.02, 150.04, 200, 1800, "Strong sell pressure"},
        {150.01, 150.03, 100, 2000, "Very strong sell"},
        {150.00, 150.02, 500, 1500, "Continued sell"},
        {150.00, 150.01, 1000, 1000, "Back to balanced"},
    };

    std::cout << std::setw(25) << "Description"
              << std::setw(10) << "Bid"
              << std::setw(10) << "Ask"
              << std::setw(8) << "BidSz"
              << std::setw(8) << "AskSz"
              << std::setw(12) << "Prediction"
              << std::setw(8) << "Conf"
              << std::setw(8) << "P(UP)"
              << std::setw(8) << "P(DN)"
              << "\n";
    std::cout << std::string(97, '-') << "\n";

    for (const auto &tc : test_cases)
    {
        auto [direction, confidence] = predictor.update(
            tc.bid, tc.ask, tc.bid_sz, tc.ask_sz);

        const auto &probs = predictor.last_probabilities();

        std::cout << std::setw(25) << tc.description
                  << std::setw(10) << std::fixed << std::setprecision(2) << tc.bid
                  << std::setw(10) << tc.ask
                  << std::setw(8) << tc.bid_sz
                  << std::setw(8) << tc.ask_sz
                  << std::setw(12) << to_string(direction)
                  << std::setw(8) << std::setprecision(3) << confidence
                  << std::setw(8) << probs[2]
                  << std::setw(8) << probs[0]
                  << "\n";
    }

    std::cout << "\nLast features:\n";
    print_features(predictor.last_features());
}

/**
 * @brief Performance benchmark
 */
void test_performance(const std::string &model_path)
{
    std::cout << "\n=== Performance Benchmark ===\n\n";

    ITCHPredictor predictor;

    bool has_model = predictor.load_model(model_path);
    if (!has_model)
    {
        std::cout << "Note: Running without model (feature extraction only)\n\n";
    }

    // Warmup
    for (int i = 0; i < 100; ++i)
    {
        predictor.update(150.0 + i * 0.001, 150.02 + i * 0.001, 1000, 1000);
    }
    predictor.reset();

    // Benchmark
    std::mt19937 rng(42);
    std::normal_distribution<double> price_noise(0.0, 0.01);
    std::normal_distribution<double> size_noise(0.0, 100);

    double base_price = 150.00;

    const int NUM_UPDATES = 10000;
    int up_count = 0, down_count = 0, neutral_count = 0;

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < NUM_UPDATES; ++i)
    {
        base_price += price_noise(rng);

        double bid = base_price;
        double ask = base_price + 0.01 + std::abs(price_noise(rng));

        int base_size = 500;
        uint32_t bid_size = std::max(100, static_cast<int>(base_size + size_noise(rng)));
        uint32_t ask_size = std::max(100, static_cast<int>(base_size + size_noise(rng)));

        auto [direction, confidence] = predictor.update(bid, ask, bid_size, ask_size);

        switch (direction)
        {
        case PriceDirection::UP:
            ++up_count;
            break;
        case PriceDirection::DOWN:
            ++down_count;
            break;
        case PriceDirection::NEUTRAL:
            ++neutral_count;
            break;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    std::cout << "Processed " << NUM_UPDATES << " updates in "
              << duration.count() << " μs\n";
    std::cout << "Throughput: " << std::fixed << std::setprecision(0)
              << (NUM_UPDATES * 1000000.0 / duration.count())
              << " updates/sec\n";
    std::cout << "Latency:    " << std::setprecision(2)
              << (static_cast<double>(duration.count()) / NUM_UPDATES)
              << " μs/update\n\n";

    std::cout << "Prediction distribution:\n";
    std::cout << "  UP:      " << up_count << " ("
              << (100.0 * up_count / NUM_UPDATES) << "%)\n";
    std::cout << "  DOWN:    " << down_count << " ("
              << (100.0 * down_count / NUM_UPDATES) << "%)\n";
    std::cout << "  NEUTRAL: " << neutral_count << " ("
              << (100.0 * neutral_count / NUM_UPDATES) << "%)\n";
}

/**
 * @brief Project 25 integration example
 */
void test_project25_integration()
{
    std::cout << "\n=== Project 25 Integration Pattern ===\n\n";

    std::cout << R"(
 // In Project 25 MarketMakerFSM (market_maker_fsm.hpp):
 
 #include "itch_predictor.hpp"
 
 class MarketMakerFSM {
     itch::ITCHPredictor predictor_;
     
 public:
     bool init(const std::string& model_path) {
         if (!predictor_.load_model(model_path)) {
             LOG_ERROR("Failed to load AI model: {}", predictor_.last_error());
             return false;
         }
         LOG_INFO("AI predictor loaded successfully");
         return true;
     }
     
     void on_bbo_update(const BboUpdate& bbo) {
         // Get AI prediction
         auto [direction, confidence] = predictor_.update(
             bbo.bid_price, bbo.ask_price,
             bbo.bid_size, bbo.ask_size,
             bbo.timestamp_ns
         );
         
         // Get detailed probabilities
         const auto& probs = predictor_.last_probabilities();
         
         // Adjust quoting based on prediction
         double skew = 0.0;
         
         if (confidence > 0.6) {  // High confidence threshold
             switch (direction) {
                 case itch::PriceDirection::UP:
                     // Price likely going up: be more aggressive on bid
                     skew = -0.0001 * confidence;  // Tighten bid
                     break;
                     
                 case itch::PriceDirection::DOWN:
                     // Price likely going down: be more aggressive on ask
                     skew = +0.0001 * confidence;  // Tighten ask
                     break;
                     
                 default:
                     // NEUTRAL: no adjustment
                     break;
             }
         }
         
         // Alternative: use probability-weighted skew
         double prob_skew = (probs[2] - probs[0]) * 0.0002;  // UP prob - DOWN prob
         
         // Apply skew to quote prices...
         double adjusted_bid = fair_value_ - half_spread_ + skew;
         double adjusted_ask = fair_value_ + half_spread_ + skew;
     }
 };
 )";
    std::cout << "\n";
}

void print_usage(const char *prog)
{
    std::cout << "Usage: " << prog << " [model_path]\n\n";
    std::cout << "  model_path: Path to XGBoost model file (.ubj, .json, or .bin)\n";
    std::cout << "              If not provided, runs feature extraction tests only\n\n";
    std::cout << "Example:\n";
    std::cout << "  " << prog << " itch_predictor.ubj\n";
}

int main(int argc, char *argv[])
{
    std::cout << "ITCH Price Direction Predictor - Test Suite\n";
    std::cout << "Using XGBoost C API\n";
    std::cout << "============================================\n\n";

    // Get XGBoost version
    int major, minor, patch;
    XGBoostVersion(&major, &minor, &patch);
    std::cout << "XGBoost version: " << major << "." << minor << "." << patch << "\n\n";

    std::string model_path = (argc > 1) ? argv[1] : "itch_predictor.ubj";

    test_rolling_buffer();
    test_feature_extraction();
    test_model_loading(model_path);
    test_predictor_with_model(model_path);
    test_performance(model_path);
    test_project25_integration();

    std::cout << "\n============================================\n";
    std::cout << "All tests completed!\n";

    return 0;
}
