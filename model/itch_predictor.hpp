#pragma once
/**
 * @file itch_predictor.hpp
 * @brief ITCH Price Direction Predictor using XGBoost C API
 *
 * Predicts short-term price direction (UP/DOWN/NEUTRAL) based on
 * order book features extracted from ITCH 5.0 market data.
 *
 * Model: XGBoost trained on NASDAQ ITCH historical data
 * Accuracy: 81% on test set (3-class classification)
 * Format: Native XGBoost .ubj (Universal Binary JSON)
 *
 * Features used (12 total):
 *   - Bid/Ask prices and sizes
 *   - Spread (absolute and percentage)
 *   - Order imbalance
 *   - Price momentum (rolling changes)
 *   - Volume-weighted metrics
 *
 * Build:
 *   g++ -std=c++17 -O2 -I include $(pkg-config --cflags xgboost) \
 *       -o predictor main.cpp $(pkg-config --libs xgboost)
 */

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <xgboost/c_api.h>

namespace itch
{

    /**
     * @brief Prediction result from the model
     */
    enum class PriceDirection : int8_t
    {
        DOWN = 0,    // Class 0 in model
        NEUTRAL = 1, // Class 1 in model
        UP = 2       // Class 2 in model
    };

    /**
     * @brief Convert PriceDirection to string
     */
    inline const char *to_string(PriceDirection dir)
    {
        switch (dir)
        {
        case PriceDirection::DOWN:
            return "DOWN";
        case PriceDirection::NEUTRAL:
            return "NEUTRAL";
        case PriceDirection::UP:
            return "UP";
        default:
            return "UNKNOWN";
        }
    }

    /**
     * @brief Order book snapshot for feature extraction
     */
    struct OrderBookSnapshot
    {
        double bid_price = 0.0;
        double ask_price = 0.0;
        uint32_t bid_size = 0;
        uint32_t ask_size = 0;
        uint64_t timestamp_ns = 0;

        double mid_price() const
        {
            return (bid_price + ask_price) / 2.0;
        }

        double spread() const
        {
            return ask_price - bid_price;
        }

        double spread_bps() const
        {
            double mid = mid_price();
            return (mid > 0) ? (spread() / mid) * 10000.0 : 0.0;
        }

        double order_imbalance() const
        {
            uint32_t total = bid_size + ask_size;
            if (total == 0)
                return 0.0;
            // Cast to signed to avoid underflow when bid_size < ask_size
            int64_t diff = static_cast<int64_t>(bid_size) - static_cast<int64_t>(ask_size);
            return static_cast<double>(diff) / static_cast<double>(total);
        }
    };

    /**
     * @brief Feature vector for XGBoost model
     *
     * 12 features matching the trained model:
     *   0: bid_price
     *   1: ask_price
     *   2: bid_size
     *   3: ask_size
     *   4: spread
     *   5: spread_bps
     *   6: mid_price
     *   7: order_imbalance
     *   8: price_momentum_1 (1-tick price change in bps)
     *   9: price_momentum_5 (5-tick price change in bps)
     *  10: volume_imbalance_ma (moving average)
     *  11: microprice
     */
    struct FeatureVector
    {
        static constexpr size_t NUM_FEATURES = 12;
        std::array<float, NUM_FEATURES> features{};

        float &operator[](size_t idx) { return features[idx]; }
        const float &operator[](size_t idx) const { return features[idx]; }
        const float *data() const { return features.data(); }
        float *data() { return features.data(); }
    };

    /**
     * @brief Feature names for debugging
     */
    inline const char *feature_name(size_t idx)
    {
        static const char *names[] = {
            "bid_price", "ask_price", "bid_size", "ask_size",
            "spread", "spread_bps", "mid_price", "order_imbalance",
            "price_momentum_1", "price_momentum_5", "volume_imbalance_ma", "microprice"};
        return (idx < 12) ? names[idx] : "unknown";
    }

    /**
     * @brief Circular buffer for rolling calculations
     */
    template <typename T, size_t N>
    class RollingBuffer
    {
    public:
        void push(T value)
        {
            buffer_[write_idx_] = value;
            write_idx_ = (write_idx_ + 1) % N;
            if (count_ < N)
                ++count_;
        }

        T get(size_t ticks_ago) const
        {
            if (ticks_ago >= count_)
                return T{};
            size_t idx = (write_idx_ + N - 1 - ticks_ago) % N;
            return buffer_[idx];
        }

        size_t count() const { return count_; }
        bool full() const { return count_ == N; }

        T sum() const
        {
            T total{};
            for (size_t i = 0; i < count_; ++i)
            {
                total += buffer_[i];
            }
            return total;
        }

        double mean() const
        {
            return count_ > 0 ? static_cast<double>(sum()) / count_ : 0.0;
        }

        void clear()
        {
            buffer_ = {};
            write_idx_ = 0;
            count_ = 0;
        }

    private:
        std::array<T, N> buffer_{};
        size_t write_idx_ = 0;
        size_t count_ = 0;
    };

    /**
     * @brief Feature extractor for order book data
     */
    class FeatureExtractor
    {
    public:
        static constexpr size_t MOMENTUM_WINDOW = 10;
        static constexpr size_t IMBALANCE_MA_WINDOW = 5;

        /**
         * @brief Update with new order book snapshot and extract features
         */
        FeatureVector update(const OrderBookSnapshot &snapshot)
        {
            FeatureVector fv;

            // Basic features
            fv[0] = static_cast<float>(snapshot.bid_price);
            fv[1] = static_cast<float>(snapshot.ask_price);
            fv[2] = static_cast<float>(snapshot.bid_size);
            fv[3] = static_cast<float>(snapshot.ask_size);
            fv[4] = static_cast<float>(snapshot.spread());
            fv[5] = static_cast<float>(snapshot.spread_bps());
            fv[6] = static_cast<float>(snapshot.mid_price());
            fv[7] = static_cast<float>(snapshot.order_imbalance());

            // Momentum features
            double current_mid = snapshot.mid_price();
            double prev_1 = mid_prices_.get(0);
            double prev_5 = mid_prices_.get(4);

            fv[8] = (prev_1 > 0) ? static_cast<float>((current_mid - prev_1) / prev_1 * 10000.0) : 0.0f;
            fv[9] = (prev_5 > 0) ? static_cast<float>((current_mid - prev_5) / prev_5 * 10000.0) : 0.0f;

            // Update rolling buffers
            mid_prices_.push(current_mid);
            imbalances_.push(snapshot.order_imbalance());

            // Moving average of order imbalance
            fv[10] = static_cast<float>(imbalances_.mean());

            // Microprice (size-weighted mid)
            uint32_t total_size = snapshot.bid_size + snapshot.ask_size;
            if (total_size > 0)
            {
                fv[11] = static_cast<float>(
                    (snapshot.bid_price * snapshot.ask_size +
                     snapshot.ask_price * snapshot.bid_size) /
                    total_size);
            }
            else
            {
                fv[11] = fv[6]; // Fall back to mid price
            }

            return fv;
        }

        void reset()
        {
            mid_prices_.clear();
            imbalances_.clear();
        }

    private:
        RollingBuffer<double, MOMENTUM_WINDOW> mid_prices_;
        RollingBuffer<double, IMBALANCE_MA_WINDOW> imbalances_;
    };

    /**
     * @brief XGBoost model wrapper using C API
     *
     * Loads native XGBoost model files (.ubj, .json, .bin)
     */
    class XGBoostModel
    {
    public:
        XGBoostModel() = default;

        ~XGBoostModel()
        {
            if (booster_)
            {
                XGBoosterFree(booster_);
                booster_ = nullptr;
            }
        }

        // Non-copyable
        XGBoostModel(const XGBoostModel &) = delete;
        XGBoostModel &operator=(const XGBoostModel &) = delete;

        // Movable
        XGBoostModel(XGBoostModel &&other) noexcept : booster_(other.booster_)
        {
            other.booster_ = nullptr;
        }

        XGBoostModel &operator=(XGBoostModel &&other) noexcept
        {
            if (this != &other)
            {
                if (booster_)
                    XGBoosterFree(booster_);
                booster_ = other.booster_;
                other.booster_ = nullptr;
            }
            return *this;
        }

        /**
         * @brief Load model from file (.ubj, .json, or .bin format)
         *
         * @param model_path Path to XGBoost model file
         * @param use_gpu Enable GPU inference (requires CUDA-enabled XGBoost)
         * @param gpu_id GPU device ID (default 0)
         * @return true on success, false on failure
         */
        bool load(const std::string &model_path, bool use_gpu = false, int gpu_id = 0)
        {
            if (booster_)
            {
                XGBoosterFree(booster_);
                booster_ = nullptr;
            }

            // Create empty booster
            int ret = XGBoosterCreate(nullptr, 0, &booster_);
            if (ret != 0)
            {
                last_error_ = XGBGetLastError();
                return false;
            }

            // Load model from file
            ret = XGBoosterLoadModel(booster_, model_path.c_str());
            if (ret != 0)
            {
                last_error_ = XGBGetLastError();
                XGBoosterFree(booster_);
                booster_ = nullptr;
                return false;
            }

            // Configure device for inference (GPU or CPU)
            if (use_gpu)
            {
                // XGBoost 2.0+ uses "device" parameter with "cuda:N" format
                std::string device = "cuda:" + std::to_string(gpu_id);
                ret = XGBoosterSetParam(booster_, "device", device.c_str());
                if (ret != 0)
                {
                    // Fallback to older "gpu_id" parameter for XGBoost < 2.0
                    XGBoosterSetParam(booster_, "predictor", "gpu_predictor");
                    XGBoosterSetParam(booster_, "gpu_id", std::to_string(gpu_id).c_str());
                }
                use_gpu_ = true;
            }
            else
            {
                //XGBoosterSetParam(booster_, "device", "cpu");
                use_gpu_ = false;
                last_error_ = "CPU inference not supported";
                return false;
            }

            loaded_ = true;
            return true;
        }

        /**
         * @brief Check if GPU inference is enabled
         */
        bool is_using_gpu() const { return use_gpu_; }

        /**
         * @brief Check if model is loaded
         */
        bool is_loaded() const { return loaded_ && booster_ != nullptr; }

        /**
         * @brief Get last error message
         */
        const std::string &last_error() const { return last_error_; }

        /**
         * @brief Predict class probabilities for a single sample
         *
         * @param features Input feature vector
         * @param out_probs Output probability array (size = num_classes)
         * @return true on success
         */
        bool predict_proba(const FeatureVector &features, std::vector<float> &out_probs)
        {
            if (!is_loaded())
            {
                last_error_ = "Model not loaded";
                return false;
            }

            // Create DMatrix from single row
            DMatrixHandle dmat = nullptr;
            int ret = XGDMatrixCreateFromMat(
                features.data(),
                1,                           // nrow
                FeatureVector::NUM_FEATURES, // ncol
                std::nanf(""),               // missing value marker
                &dmat);

            if (ret != 0)
            {
                last_error_ = XGBGetLastError();
                return false;
            }

            // Predict
            bst_ulong out_len = 0;
            const float *out_result = nullptr;

            // Use predict with output_margin=0 to get probabilities
            char const config[] =
                "{\"type\": 0, \"training\": false, \"iteration_range\": [0, 0], \"strict_shape\": false}";

            uint64_t const *out_shape = nullptr;
            uint64_t out_dim = 0;

            ret = XGBoosterPredictFromDMatrix(
                booster_,
                dmat,
                config,
                &out_shape,
                &out_dim,
                &out_result);

            XGDMatrixFree(dmat);

            if (ret != 0)
            {
                last_error_ = XGBGetLastError();
                return false;
            }

            // Copy results
            out_probs.clear();
            for (bst_ulong i = 0; i < out_shape[0] * (out_dim > 1 ? out_shape[1] : 1); ++i)
            {
                out_probs.push_back(out_result[i]);
            }

            return true;
        }

        /**
         * @brief Predict class for a single sample
         *
         * @param features Input feature vector
         * @return Predicted class index, or -1 on error
         */
        int predict_class(const FeatureVector &features)
        {
            std::vector<float> probs;
            if (!predict_proba(features, probs))
            {
                return -1;
            }

            // For multi-class, find argmax
            if (probs.size() >= 3)
            {
                int max_idx = 0;
                float max_val = probs[0];
                for (size_t i = 1; i < probs.size(); ++i)
                {
                    if (probs[i] > max_val)
                    {
                        max_val = probs[i];
                        max_idx = static_cast<int>(i);
                    }
                }
                return max_idx;
            }

            // For binary classification, threshold at 0.5
            return (probs[0] > 0.5f) ? 1 : 0;
        }

    private:
        BoosterHandle booster_ = nullptr;
        bool loaded_ = false;
        bool use_gpu_ = false;
        std::string last_error_;
    };

    /**
     * @brief Complete ITCH predictor combining feature extraction and XGBoost model
     */
    class ITCHPredictor
    {
    public:
        static constexpr size_t NUM_CLASSES = 3; // DOWN, NEUTRAL, UP

        ITCHPredictor() = default;

        /**
         * @brief Load trained XGBoost model from file
         *
         * @param model_path Path to .ubj, .json, or .bin model file
         * @param use_gpu Enable GPU inference (requires CUDA-enabled XGBoost)
         * @param gpu_id GPU device ID (default 0)
         * @return true on success
         */
        bool load_model(const std::string &model_path, bool use_gpu = false, int gpu_id = 0)
        {
            return model_.load(model_path, use_gpu, gpu_id);
        }

        /**
         * @brief Check if model is loaded
         */
        bool is_loaded() const { return model_.is_loaded(); }

        /**
         * @brief Check if GPU inference is enabled
         */
        bool is_using_gpu() const { return model_.is_using_gpu(); }

        /**
         * @brief Get last error message
         */
        const std::string &last_error() const { return model_.last_error(); }

        /**
         * @brief Update with new BBO and get prediction
         *
         * @param bid_price Best bid price
         * @param ask_price Best ask price
         * @param bid_size Best bid size
         * @param ask_size Best ask size
         * @param timestamp_ns Nanosecond timestamp (optional)
         * @return Predicted direction and confidence
         */
        std::pair<PriceDirection, float> update(
            double bid_price, double ask_price,
            uint32_t bid_size, uint32_t ask_size,
            uint64_t timestamp_ns = 0)
        {
            OrderBookSnapshot snapshot{bid_price, ask_price, bid_size, ask_size, timestamp_ns};
            last_features_ = extractor_.update(snapshot);
            ++update_count_;

            // Need some history for momentum features
            if (update_count_ < 5)
            {
                return {PriceDirection::NEUTRAL, 0.0f};
            }

            if (!model_.is_loaded())
            {
                return {PriceDirection::NEUTRAL, 0.0f};
            }

            // Get prediction probabilities
            std::vector<float> probs;
            if (!model_.predict_proba(last_features_, probs))
            {
                return {PriceDirection::NEUTRAL, 0.0f};
            }

            // Store probabilities
            if (probs.size() >= NUM_CLASSES)
            {
                last_probs_[0] = probs[0]; // DOWN
                last_probs_[1] = probs[1]; // NEUTRAL
                last_probs_[2] = probs[2]; // UP
            }

            // Find max probability class
            int max_class = 0;
            float max_prob = last_probs_[0];
            for (size_t i = 1; i < NUM_CLASSES; ++i)
            {
                if (last_probs_[i] > max_prob)
                {
                    max_prob = last_probs_[i];
                    max_class = static_cast<int>(i);
                }
            }

            return {static_cast<PriceDirection>(max_class), max_prob};
        }

        /**
         * @brief Get last extracted features (for debugging)
         */
        const FeatureVector &last_features() const { return last_features_; }

        /**
         * @brief Get last prediction probabilities
         */
        const std::array<float, NUM_CLASSES> &last_probabilities() const { return last_probs_; }

        /**
         * @brief Get probability for specific direction
         */
        float probability(PriceDirection dir) const
        {
            return last_probs_[static_cast<size_t>(dir)];
        }

        /**
         * @brief Get number of updates processed
         */
        uint64_t update_count() const { return update_count_; }

        /**
         * @brief Reset state
         */
        void reset()
        {
            extractor_.reset();
            update_count_ = 0;
            last_features_ = {};
            last_probs_ = {0.33f, 0.34f, 0.33f};
        }

    private:
        FeatureExtractor extractor_;
        XGBoostModel model_;
        FeatureVector last_features_;
        std::array<float, NUM_CLASSES> last_probs_ = {0.33f, 0.34f, 0.33f};
        uint64_t update_count_ = 0;
    };

} // namespace itch