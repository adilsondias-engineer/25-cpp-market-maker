#include "market_maker_fsm.h"
#include <cmath>
#include <chrono>
#include <vector>

namespace mm {

MarketMakerFSM::MarketMakerFSM(const Config& config)
    : state_(State::IDLE), config_(config), order_sequence_(0)
#ifdef HAVE_XGBOOST
    , last_prediction_(0.0)
#endif
{
    logger_ = spdlog::get("market_maker");
    if (!logger_) {
        logger_ = spdlog::stdout_color_mt("market_maker");
    }
    logger_->info("MarketMakerFSM initialized with config: spread={} bps, edge={} bps, max_pos={}, skew={} bps",
                  config_.min_spread_bps, config_.edge_bps, config_.max_position, config_.position_skew_bps);

#ifdef HAVE_XGBOOST
    if (config_.xgboost.enabled) {
        logger_->info("XGBoost inference enabled: model={}, gpu={}, device={}",
                     config_.xgboost.model_path, config_.xgboost.use_gpu, config_.xgboost.gpu_device_id);
        if (!loadModel(config_.xgboost.model_path)) {
            if (config_.xgboost.use_gpu) {
                // CRITICAL: If GPU was requested but model loading failed, abort
                logger_->critical("FATAL: XGBoost GPU mode requested but model loading failed!");
                logger_->critical("Market Maker requires GPU inference for pipeline parallelism.");
                logger_->critical("Check: CUDA installed, GPU available, model file exists.");
                throw std::runtime_error("XGBoost GPU initialization failed - cannot continue");
            } else {
                logger_->error("Failed to load XGBoost model - continuing without ML inference");
            }
        } else if (config_.xgboost.use_gpu && predictor_ && !predictor_->is_using_gpu()) {
            // Model loaded but GPU not actually being used
            logger_->critical("FATAL: XGBoost GPU mode requested but running on CPU!");
            logger_->critical("This defeats pipeline parallelism and gives false latency results.");
            logger_->critical("Check CUDA/GPU configuration or set use_gpu=false in config.");
            throw std::runtime_error("XGBoost GPU validation failed - running on CPU instead of GPU");
        }

        // Initialize async prediction worker if enabled
        if (config_.xgboost.async_mode && predictor_) {
            logger_->info("Async prediction mode enabled (predict on tick N-1, use for tick N)");
            async_ = std::make_unique<AsyncPrediction>();
            async_->running.store(true);
            async_->worker_thread = std::thread(&MarketMakerFSM::asyncPredictionWorker, this);
        }
    }
#else
    if (config_.xgboost.enabled) {
        logger_->warn("XGBoost requested but not compiled in (HAVE_XGBOOST not defined)");
        if (config_.xgboost.use_gpu) {
            logger_->critical("FATAL: XGBoost GPU mode requested but XGBoost support not compiled!");
            throw std::runtime_error("XGBoost not available - compile with HAVE_XGBOOST");
        }
    }
#endif

    // Initialize order producer if order execution is enabled
    if (config_.enable_order_execution) {
        try {
            order_producer_ = std::make_unique<OrderProducer>(
                config_.order_ring_path, config_.fill_ring_path);
            logger_->info("Order execution enabled (Project 26 integration)");
        } catch (const std::exception& e) {
            logger_->error("Failed to initialize order producer: {}", e.what());
            logger_->warn("Continuing without order execution");
        }
    }
}

MarketMakerFSM::~MarketMakerFSM() {
#ifdef HAVE_XGBOOST
    shutdownAsyncWorker();
#endif
}

void MarketMakerFSM::onBboUpdate(const BBO& bbo) {
    if (!bbo.valid) {
        return;
    }

    // Performance optimization: Skip processing if BBO hasn't changed significantly
    static BBO last_bbo;
    static uint64_t skip_count = 0;
    static uint64_t process_count = 0;
    
    if (last_bbo.valid && last_bbo.symbol == bbo.symbol) {
        double price_change = std::abs(bbo.bid_price - last_bbo.bid_price) + 
                             std::abs(bbo.ask_price - last_bbo.ask_price);
        double spread_change = std::abs(bbo.spread - last_bbo.spread);
        
        // Skip if price change < 0.01 and spread change < 0.01 (1 cent threshold)
        if (price_change < 0.01 && spread_change < 0.01) {
            if (++skip_count % 10000 == 0) {
                logger_->debug("Skipped {} BBO updates (minimal price change)", skip_count);
            }
            return;
        }
    }
    
    // Throttling: Only process every 10th update for high-frequency data
    // This gives 10% processing rate - enough for console visibility while managing load
    if (++process_count % 10 != 0) {
        return;
    }
    
    last_bbo = bbo;

    switch (state_) {
        case State::IDLE:
            state_ = State::CALCULATE;
            handleCalculate(bbo);
            break;

        case State::CALCULATE:
            handleCalculate(bbo);
            break;

        case State::QUOTE:
            handleQuote(bbo);
            break;

        case State::RISK_CHECK:
            handleRiskCheck();
            break;

        case State::ORDER_GEN:
            handleOrderGen();
            break;

        case State::WAIT_FILL:
            handleWaitFill();
            state_ = State::CALCULATE;
            handleCalculate(bbo);
            break;
    }
}

void MarketMakerFSM::handleCalculate(const BBO& bbo) {
    cached_fair_value_ = calculateFairValue(bbo);

    logger_->debug("CALCULATE: symbol={}, fair_value={:.4f}, spread={:.4f}",
                   bbo.symbol, cached_fair_value_, bbo.spread);

    state_ = State::QUOTE;
    handleQuote(bbo);
}

void MarketMakerFSM::handleQuote(const BBO& bbo) {
    // Use cached fair value from handleCalculate to avoid recalculation
    current_quote_ = generateQuote(cached_fair_value_, bbo);

    if (!current_quote_.valid) {
        logger_->warn("QUOTE: Invalid quote generated, returning to IDLE");
        state_ = State::IDLE;
        return;
    }

    // Console output similar to P24 order_gateway with ML prediction info
    std::string prediction_str = "N/A";
#ifdef HAVE_XGBOOST
    if (predictor_ && config_.xgboost.enabled) {
        if (last_prediction_ > 0.1) prediction_str = "BULLISH";
        else if (last_prediction_ < -0.1) prediction_str = "BEARISH";
        else prediction_str = "NEUTRAL";
    }
#endif
    logger_->debug("[{}] Bid: {:.4f} ({}) | Ask: {:.4f} ({}) | Spread: {:.4f} | Fair: {:.4f} | ML: {}",
                  bbo.symbol, bbo.bid_price, bbo.bid_shares,
                  bbo.ask_price, bbo.ask_shares, bbo.spread, cached_fair_value_, prediction_str);

    logger_->debug("QUOTE: symbol={}, bid={:.4f}x{}, ask={:.4f}x{}, fair={:.4f}",
                   current_quote_.symbol, current_quote_.bid_price, current_quote_.bid_size,
                   current_quote_.ask_price, current_quote_.ask_size, current_quote_.fair_value);

    state_ = State::RISK_CHECK;
    handleRiskCheck();
}

void MarketMakerFSM::handleRiskCheck() {
    bool risk_ok = checkRiskLimits(current_quote_);

    if (!risk_ok) {
        logger_->warn("RISK_CHECK: Risk limits exceeded, skipping quote");
        state_ = State::IDLE;
        return;
    }

    // Additional performance optimization: Skip order generation if position is near limits
    Position pos = position_.getPosition(current_quote_.symbol);
    if (std::abs(pos.shares) > config_.max_position * 0.8) {  // 80% of max position
        logger_->debug("RISK_CHECK: Position near limit ({}), skipping order generation", pos.shares);
        state_ = State::IDLE;
        return;
    }

    logger_->debug("RISK_CHECK: Passed");
    state_ = State::ORDER_GEN;
    handleOrderGen();
}

void MarketMakerFSM::handleOrderGen() {
    logger_->debug("ORDER_GEN: Sending quote: {}@{:.4f} / {:.4f}@{}",
                  current_quote_.bid_size, current_quote_.bid_price,
                  current_quote_.ask_price, current_quote_.ask_size);

    // Send orders to Project 26 if enabled
    if (order_producer_) {
        // Generate order ID (MM prefix + sequence number)
        char order_id[32];
        snprintf(order_id, sizeof(order_id), "MM%010lu", order_sequence_++);

        // Send buy order (bid)
        trading::OrderRequest buy_order;
        buy_order.set_order_id(order_id);
        buy_order.set_symbol(current_quote_.symbol.c_str());
        buy_order.side = 'B';
        buy_order.order_type = 'L';  // Limit order
        buy_order.time_in_force = 'D';  // Day order
        buy_order.price = current_quote_.bid_price;
        buy_order.quantity = current_quote_.bid_size;
        buy_order.timestamp_ns = current_quote_.timestamp_ns;
        buy_order.valid = true;

        order_producer_->send_order(buy_order);
        logger_->debug("Sent buy order {} to Project 26", order_id);

        // Generate new order ID for sell order
        snprintf(order_id, sizeof(order_id), "MM%010lu", order_sequence_++);

        // Send sell order (ask)
        trading::OrderRequest sell_order;
        sell_order.set_order_id(order_id);
        sell_order.set_symbol(current_quote_.symbol.c_str());
        sell_order.side = 'S';
        sell_order.order_type = 'L';
        sell_order.time_in_force = 'D';
        sell_order.price = current_quote_.ask_price;
        sell_order.quantity = current_quote_.ask_size;
        sell_order.timestamp_ns = current_quote_.timestamp_ns;
        sell_order.valid = true;

        order_producer_->send_order(sell_order);
        logger_->debug("Sent sell order {} to Project 26", order_id);
    }

    state_ = State::WAIT_FILL;
}

void MarketMakerFSM::handleWaitFill() {
    logger_->debug("WAIT_FILL: Simulating fill");
}

void MarketMakerFSM::processFills() {
    if (!order_producer_) {
        return;
    }

    trading::FillNotification fill;
    while (order_producer_->try_read_fill(fill)) {
        logger_->debug("Received fill: {} {} {} shares @ {:.4f} (cumQty={}, complete={})",
                     fill.get_order_id(), fill.side == 'B' ? "BUY" : "SELL",
                     fill.fill_qty, fill.avg_price, fill.cum_qty,
                     fill.is_complete ? "yes" : "no");

        // Update position tracker
        int shares = (fill.side == 'B') ? static_cast<int>(fill.fill_qty) : -static_cast<int>(fill.fill_qty);
        position_.addFill(fill.get_symbol(), shares, fill.avg_price);

        Position pos = position_.getPosition(fill.get_symbol());
        logger_->debug("Updated position: {} shares, realized_pnl={:.2f}",
                     pos.shares, pos.realized_pnl);
    }
}

double MarketMakerFSM::calculateFairValue(const BBO& bbo) {
    if (bbo.bid_price <= 0.0 || bbo.ask_price <= 0.0) {
        return 0.0;
    }

    double mid_price = (bbo.bid_price + bbo.ask_price) / 2.0;

    uint32_t total_size = bbo.bid_shares + bbo.ask_shares;
    if (total_size == 0) {
#ifdef HAVE_XGBOOST
        // Use ML prediction if available
        if (predictor_ && config_.xgboost.enabled) {
            return applyMLPrediction(mid_price, bbo);
        }
#endif
        return mid_price;
    }

    double weighted_price = (bbo.bid_price * bbo.bid_shares + bbo.ask_price * bbo.ask_shares) / total_size;
    double fair_value = (mid_price + weighted_price) / 2.0;

#ifdef HAVE_XGBOOST
    // Blend ML prediction with traditional fair value
    if (predictor_ && config_.xgboost.enabled) {
        return applyMLPrediction(fair_value, bbo);
    }
#endif

    return fair_value;
}

#ifdef HAVE_XGBOOST
bool MarketMakerFSM::loadModel(const std::string& model_path) {
    try {
        predictor_ = std::make_unique<itch::ITCHPredictor>();

        logger_->info("Loading XGBoost model with verbosity={} (0=silent, 1=warning, 2=info, 3=debug)",
                     config_.xgboost.verbosity);

        if (!predictor_->load_model(model_path,
                                    config_.xgboost.use_gpu,
                                    config_.xgboost.gpu_device_id,
                                    config_.xgboost.verbosity)) {
            logger_->error("Failed to load XGBoost model: {}", predictor_->last_error());
            predictor_.reset();
            return false;
        }

        // Set max iterations for latency/accuracy tradeoff (0 = use all trees)
        if (config_.xgboost.max_iterations > 0) {
            predictor_->set_max_iterations(config_.xgboost.max_iterations);
            logger_->info("XGBoost max_iterations set to {} (limiting trees for faster inference)",
                         config_.xgboost.max_iterations);
        }

        logger_->info("XGBoost model loaded successfully: {} (GPU: {}, max_iterations: {})",
                     model_path, predictor_->is_using_gpu() ? "yes" : "no",
                     config_.xgboost.max_iterations > 0 ? std::to_string(config_.xgboost.max_iterations) : "all");
        return true;
    } catch (const std::exception& e) {
        logger_->error("Failed to load XGBoost model: {}", e.what());
        predictor_.reset();
        return false;
    }
}

double MarketMakerFSM::applyMLPrediction(double base_fair_value, const BBO& bbo) {
    try {
        // Use async prediction if enabled (effectively zero latency on hot path)
        if (async_ && config_.xgboost.async_mode) {
            double adjustment = getAsyncPrediction(bbo);
            return base_fair_value + adjustment;
        }

        // Synchronous mode: Measure XGBoost inference timing
        auto inference_start = std::chrono::high_resolution_clock::now();

        // Update predictor with BBO data and get direction prediction
        auto [direction, confidence] = predictor_->update(
            bbo.bid_price, bbo.ask_price,
            static_cast<uint32_t>(bbo.bid_shares),
            static_cast<uint32_t>(bbo.ask_shares),
            bbo.timestamp_ns
        );

        auto inference_end = std::chrono::high_resolution_clock::now();
        auto inference_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            inference_end - inference_start).count();
        double inference_us = inference_ns / 1000.0;

        // Only track timing when actual inference happened (confidence > 0 means XGBoost ran)
        // The first few updates return early with confidence=0 for feature warmup
        bool actual_inference = (predictor_->update_count() >= 5);

        if (actual_inference) {
            inference_count_++;
            inference_total_us_ += inference_us;
            if (inference_us < inference_min_us_ || inference_min_us_ == 0.0) {
                inference_min_us_ = inference_us;
            }
            if (inference_us > inference_max_us_) {
                inference_max_us_ = inference_us;
            }

            // Log inference timing periodically (every 100 actual predictions)
            if (inference_count_ % 100 == 0) {
                double avg_us = inference_total_us_ / inference_count_;
                logger_->debug("[XGBoost Inference] count={}, avg={:.2f}us, min={:.2f}us, max={:.2f}us (GPU: {})",
                             inference_count_, avg_us, inference_min_us_, inference_max_us_,
                             predictor_->is_using_gpu() ? "yes" : "NO!");
            }
        }

        // Convert direction to prediction value (-1, 0, +1)
        float prediction = 0.0f;
        switch (direction) {
            case itch::PriceDirection::UP:
                prediction = confidence;
                break;
            case itch::PriceDirection::DOWN:
                prediction = -confidence;
                break;
            case itch::PriceDirection::NEUTRAL:
            default:
                prediction = 0.0f;
                break;
        }
        last_prediction_ = prediction;

        // Apply prediction as adjustment to fair value
        // Scale prediction by spread to get price adjustment
        double spread = bbo.ask_price - bbo.bid_price;
        double adjustment = prediction * spread * config_.xgboost.prediction_weight;

        double adjusted_fair_value = base_fair_value + adjustment;

        logger_->debug("ML prediction: {} (conf={:.2f}, {:.2f}us), adjustment: {:.6f}, fair_value: {:.4f} -> {:.4f}",
                      itch::to_string(direction), confidence, inference_us, adjustment,
                      base_fair_value, adjusted_fair_value);

        return adjusted_fair_value;
    } catch (const std::exception& e) {
        logger_->warn("ML prediction failed: {} - using base fair value", e.what());
        return base_fair_value;
    }
}
#endif

Quote MarketMakerFSM::generateQuote(double fair_value, const BBO& bbo) {
    Quote quote;
    quote.symbol = bbo.symbol;
    quote.fair_value = fair_value;
    quote.timestamp_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()
    ).count();

    if (fair_value <= 0.0) {
        quote.valid = false;
        return quote;
    }

    double edge = fair_value * (config_.edge_bps / 10000.0);

    Position pos = position_.getPosition(bbo.symbol);
    double skew = 0.0;
    if (pos.shares != 0 && config_.max_position > 0) {
        double inventory_ratio = static_cast<double>(pos.shares) / config_.max_position;
        skew = fair_value * (config_.position_skew_bps / 10000.0) * inventory_ratio;
    }

    quote.bid_price = fair_value - edge + skew;
    quote.ask_price = fair_value + edge + skew;
    quote.bid_size = config_.quote_size;
    quote.ask_size = config_.quote_size;

    double min_spread = fair_value * (config_.min_spread_bps / 10000.0);
    if (quote.ask_price - quote.bid_price < min_spread) {
        quote.bid_price = fair_value - min_spread / 2.0;
        quote.ask_price = fair_value + min_spread / 2.0;
    }

    quote.bid_price = std::max(quote.bid_price, 0.01);
    quote.ask_price = std::max(quote.ask_price, quote.bid_price + 0.01);

    quote.valid = true;
    return quote;
}

bool MarketMakerFSM::checkRiskLimits(const Quote& quote) {
    Position pos = position_.getPosition(quote.symbol);

    int bid_new_shares = pos.shares + quote.bid_size;
    int ask_new_shares = pos.shares - quote.ask_size;

    if (std::abs(bid_new_shares) > config_.max_position) {
        logger_->warn("Risk check failed: bid would exceed max position ({} > {})",
                      std::abs(bid_new_shares), config_.max_position);
        return false;
    }

    if (std::abs(ask_new_shares) > config_.max_position) {
        logger_->warn("Risk check failed: ask would exceed max position ({} > {})",
                      std::abs(ask_new_shares), config_.max_position);
        return false;
    }

    double bid_notional = std::abs(bid_new_shares) * quote.bid_price;
    double ask_notional = std::abs(ask_new_shares) * quote.ask_price;

    if (bid_notional > config_.max_notional) {
        logger_->warn("Risk check failed: bid notional exceeds limit ({:.2f} > {:.2f})",
                      bid_notional, config_.max_notional);
        return false;
    }

    if (ask_notional > config_.max_notional) {
        logger_->warn("Risk check failed: ask notional exceeds limit ({:.2f} > {:.2f})",
                      ask_notional, config_.max_notional);
        return false;
    }

    return true;
}

void MarketMakerFSM::simulateFill(const BBO& bbo) {
    if (std::rand() % 2 == 0) {
        Fill fill;
        fill.symbol = current_quote_.symbol;
        fill.side = Side::BUY;
        fill.price = current_quote_.bid_price;
        fill.shares = current_quote_.bid_size;
        fill.timestamp_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch()
        ).count();

        position_.addFill(fill.symbol, fill.shares, fill.price);
        logger_->debug("FILL: BUY {} shares of {} @ {:.4f}",
                      fill.shares, fill.symbol, fill.price);
    } else {
        Fill fill;
        fill.symbol = current_quote_.symbol;
        fill.side = Side::SELL;
        fill.price = current_quote_.ask_price;
        fill.shares = -current_quote_.ask_size;
        fill.timestamp_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch()
        ).count();

        position_.addFill(fill.symbol, fill.shares, fill.price);
        logger_->debug("FILL: SELL {} shares of {} @ {:.4f}",
                      std::abs(fill.shares), fill.symbol, fill.price);
    }

    Position pos = position_.getPosition(current_quote_.symbol);
    logger_->debug("POSITION: {} shares, realized_pnl={:.2f}, unrealized_pnl={:.2f}",
                  pos.shares, pos.realized_pnl, pos.unrealized_pnl);
}

#ifdef HAVE_XGBOOST
void MarketMakerFSM::asyncPredictionWorker() {
    logger_->info("[Async] Prediction worker thread started");

    while (async_->running.load()) {
        BBO bbo_to_process;
        bool has_work = false;

        {
            std::unique_lock<std::mutex> lock(async_->mutex);

            // Wait for work or shutdown
            async_->cv.wait(lock, [this] {
                return async_->has_pending || !async_->running.load();
            });

            if (!async_->running.load()) {
                break;
            }

            if (async_->has_pending) {
                bbo_to_process = async_->pending_bbo;
                async_->has_pending = false;
                has_work = true;
            }
        }

        if (has_work && predictor_) {
            auto start = std::chrono::high_resolution_clock::now();

            // Run prediction
            auto [direction, confidence] = predictor_->update(
                bbo_to_process.bid_price, bbo_to_process.ask_price,
                static_cast<uint32_t>(bbo_to_process.bid_shares),
                static_cast<uint32_t>(bbo_to_process.ask_shares),
                bbo_to_process.timestamp_ns
            );

            auto end = std::chrono::high_resolution_clock::now();
            double latency_us = std::chrono::duration<double, std::micro>(end - start).count();

            // Convert direction to prediction value
            double prediction = 0.0;
            switch (direction) {
                case itch::PriceDirection::UP:
                    prediction = confidence;
                    break;
                case itch::PriceDirection::DOWN:
                    prediction = -confidence;
                    break;
                default:
                    prediction = 0.0;
                    break;
            }

            // Store result
            {
                std::lock_guard<std::mutex> lock(async_->mutex);
                async_->ready_prediction = prediction;
                async_->ready_direction = direction;
                async_->ready_confidence = confidence;
                async_->has_ready = true;
                async_->predictions_completed++;
                async_->total_latency_us += latency_us;
            }

            // Log periodically
            if (async_->predictions_completed % 100 == 0) {
                double avg_us = async_->total_latency_us / async_->predictions_completed;
                logger_->debug("[Async] predictions={}, avg_latency={:.2f}us",
                             async_->predictions_completed, avg_us);
            }
        }
    }

    logger_->info("[Async] Prediction worker thread stopped");
}

void MarketMakerFSM::submitAsyncPrediction(const BBO& bbo) {
    if (!async_) return;

    {
        std::lock_guard<std::mutex> lock(async_->mutex);
        async_->pending_bbo = bbo;
        async_->has_pending = true;
    }
    async_->cv.notify_one();
}

double MarketMakerFSM::getAsyncPrediction(const BBO& bbo) {
    if (!async_) return 0.0;

    double prediction = 0.0;
    {
        std::lock_guard<std::mutex> lock(async_->mutex);
        if (async_->has_ready) {
            prediction = async_->ready_prediction;
            last_prediction_ = prediction;  // Store for state display
        }
    }

    // Submit current BBO for next prediction (pipeline)
    submitAsyncPrediction(bbo);

    // Apply prediction as adjustment to fair value
    double spread = bbo.ask_price - bbo.bid_price;
    return prediction * spread * config_.xgboost.prediction_weight;
}

void MarketMakerFSM::shutdownAsyncWorker() {
    if (!async_) return;

    logger_->info("[Async] Shutting down prediction worker...");

    async_->running.store(false);
    async_->cv.notify_all();

    if (async_->worker_thread.joinable()) {
        async_->worker_thread.join();
    }

    if (async_->predictions_completed > 0) {
        double avg_us = async_->total_latency_us / async_->predictions_completed;
        logger_->info("[Async] Final stats: predictions={}, avg_latency={:.2f}us",
                     async_->predictions_completed, avg_us);
    }

    async_.reset();
}
#endif

} // namespace mm
