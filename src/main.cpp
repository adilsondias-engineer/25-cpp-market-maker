#include <iostream>
#include <csignal>
#include <fstream>
#include <chrono>
#include <filesystem>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/rotating_file_sink.h>
#include <spdlog/async.h>
#include <nlohmann/json.hpp>

namespace fs = std::filesystem;
#include "market_maker_fsm.h"
#include "disruptor_client.h"
#include "bbo_parser.h"
#include "common/perf_monitor.h"
#ifdef HAVE_XGBOOST
#include "model/itch_predictor.hpp"
#endif
#include <sys/stat.h>

#ifdef __linux__
#include <sched.h>
#include <pthread.h>
#include <cstring>

void enableRealTimeScheduling() {
    struct sched_param param;
    param.sched_priority = 50;

    if (sched_setscheduler(0, SCHED_FIFO, &param) == -1) {
        spdlog::warn("Failed to set RT scheduling: {}", strerror(errno));
    } else {
        spdlog::info("RT scheduling enabled (SCHED_FIFO, priority 50)");
    }
}

void setCpuAffinity(const std::vector<int>& cores) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);

    for (int core : cores) {
        CPU_SET(core, &cpuset);
    }

    if (pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset) == 0) {
        std::string cores_str;
        for (size_t i = 0; i < cores.size(); ++i) {
            if (i > 0) cores_str += ",";
            cores_str += std::to_string(cores[i]);
        }
        spdlog::info("CPU affinity set to cores: {}", cores_str);
    }
}
#endif

volatile bool g_running = true;
std::unique_ptr<gateway::DisruptorClient> g_client;
gateway::PerfMonitor g_parse_latency;
std::atomic<uint64_t> g_quotes_generated{0};

void signalHandler(int signal) {
    spdlog::info("Received signal {}, shutting down...", signal);
    g_running = false;
}

mm::BBO convertBboData(const gateway::BBOData& bbo_data) {
    mm::BBO bbo;
    bbo.symbol = bbo_data.get_symbol();
    bbo.bid_price = bbo_data.bid_price;
    bbo.bid_shares = bbo_data.bid_shares;
    bbo.ask_price = bbo_data.ask_price;
    bbo.ask_shares = bbo_data.ask_shares;
    bbo.spread = bbo_data.spread;
    bbo.timestamp_ns = static_cast<uint64_t>(bbo_data.timestamp_ns);
    bbo.valid = bbo_data.valid;
    return bbo;
}

int main(int argc, char** argv) {
    // Initialize async logging thread pool (8K queue, 1 background thread)
    spdlog::init_thread_pool(8192, 1);

    // Console sink (sync - console writes are fast)
    auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    console_sink->set_level(spdlog::level::info);

    // Default log file path (can be overridden by config)
    std::string log_file_path = "/opt/trading/logs/p25_market_maker.log";

    // Ensure log directory exists
    fs::create_directories(fs::path(log_file_path).parent_path());

    // Rotating file sink: 10MB max, 5 rotated files
    auto file_sink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
        log_file_path, 10 * 1024 * 1024, 5);
    file_sink->set_level(spdlog::level::debug);  // File gets more detail

    // Create async logger - file I/O happens on background thread
    std::vector<spdlog::sink_ptr> sinks{console_sink, file_sink};
    auto logger = std::make_shared<spdlog::async_logger>(
        "market_maker", sinks.begin(), sinks.end(),
        spdlog::thread_pool(),
        spdlog::async_overflow_policy::overrun_oldest);  // Don't block on full queue
    logger->set_level(spdlog::level::debug);
    spdlog::set_default_logger(logger);

    spdlog::info("==============================================");
    spdlog::info("  Project 25: Market Maker");
    spdlog::info("  Disruptor Consumer from P24");
    spdlog::info("==============================================");
    spdlog::info("Log file: {}", log_file_path);

    std::string config_file = "config.json";
    bool run_benchmark = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            spdlog::info("Usage: {} [options] [config_file]", argv[0]);
            spdlog::info("");
            spdlog::info("Project 25: Market Maker");
            spdlog::info("  Consumes BBO data from Project 24 via Disruptor shared memory");
            spdlog::info("");
            spdlog::info("Options:");
            spdlog::info("  --benchmark    Run XGBoost CPU vs GPU inference benchmark and exit");
            spdlog::info("  -h, --help     Show this help message");
            return 0;
        } else if (arg == "--benchmark") {
            run_benchmark = true;
        } else if (arg[0] != '-') {
            config_file = arg;
        }
    }

    mm::MarketMakerFSM::Config mm_config;
    std::string disruptor_shm_name = "gateway";
    bool enable_rt = false;
    std::vector<int> cpu_cores = {9, 10};  // Isolated cores (4-23)

    std::ifstream config_stream(config_file);
    if (config_stream.is_open()) {
        try {
            nlohmann::json config_json;
            config_stream >> config_json;

            // Market maker config
            if (config_json.contains("min_spread_bps")) {
                mm_config.min_spread_bps = config_json["min_spread_bps"];
            }
            if (config_json.contains("edge_bps")) {
                mm_config.edge_bps = config_json["edge_bps"];
            }
            if (config_json.contains("max_position")) {
                mm_config.max_position = config_json["max_position"];
            }
            if (config_json.contains("position_skew_bps")) {
                mm_config.position_skew_bps = config_json["position_skew_bps"];
            }
            if (config_json.contains("quote_size")) {
                mm_config.quote_size = config_json["quote_size"];
            }
            if (config_json.contains("max_notional")) {
                mm_config.max_notional = config_json["max_notional"];
            }

            // Disruptor config
            if (config_json.contains("disruptor")) {
                auto& disruptor = config_json["disruptor"];
                if (disruptor.contains("shm_name")) {
                    disruptor_shm_name = disruptor["shm_name"].get<std::string>();
                }
            }

            // Order execution config (P26 integration)
            if (config_json.contains("enable_order_execution")) {
                mm_config.enable_order_execution = config_json["enable_order_execution"];
            }
            if (config_json.contains("order_ring_path")) {
                mm_config.order_ring_path = config_json["order_ring_path"].get<std::string>();
            }
            if (config_json.contains("fill_ring_path")) {
                mm_config.fill_ring_path = config_json["fill_ring_path"].get<std::string>();
            }

            // Performance config
            if (config_json.contains("performance")) {
                auto& perf = config_json["performance"];
                if (perf.contains("enable_rt")) {
                    enable_rt = perf["enable_rt"];
                }
                if (perf.contains("cpu_cores")) {
                    cpu_cores = perf["cpu_cores"].get<std::vector<int>>();
                }
            }

            // XGBoost inference config
            if (config_json.contains("xgboost")) {
                auto& xgb = config_json["xgboost"];
                if (xgb.contains("enabled")) {
                    mm_config.xgboost.enabled = xgb["enabled"];
                }
                if (xgb.contains("model_path")) {
                    mm_config.xgboost.model_path = xgb["model_path"].get<std::string>();
                }
                if (xgb.contains("use_gpu")) {
                    mm_config.xgboost.use_gpu = xgb["use_gpu"];
                }
                if (xgb.contains("gpu_device_id")) {
                    mm_config.xgboost.gpu_device_id = xgb["gpu_device_id"];
                }
                if (xgb.contains("prediction_weight")) {
                    mm_config.xgboost.prediction_weight = xgb["prediction_weight"];
                }
                if (xgb.contains("verbosity")) {
                    mm_config.xgboost.verbosity = xgb["verbosity"];
                }
                if (xgb.contains("max_iterations")) {
                    mm_config.xgboost.max_iterations = xgb["max_iterations"];
                }
                if (xgb.contains("async_mode")) {
                    mm_config.xgboost.async_mode = xgb["async_mode"];
                }
            }

            // Legacy flat config (backward compatibility)
            if (config_json.contains("enable_rt")) {
                enable_rt = config_json["enable_rt"];
            }
            if (config_json.contains("cpu_cores")) {
                cpu_cores = config_json["cpu_cores"].get<std::vector<int>>();
            }

            if (config_json.contains("log_level")) {
                std::string log_level = config_json["log_level"].get<std::string>();
                if (log_level == "debug") spdlog::set_level(spdlog::level::debug);
                else if (log_level == "info") spdlog::set_level(spdlog::level::info);
                else if (log_level == "warn") spdlog::set_level(spdlog::level::warn);
                else if (log_level == "error") spdlog::set_level(spdlog::level::err);
            }

            spdlog::info("Loaded config from {}", config_file);
        } catch (const std::exception& e) {
            spdlog::warn("Failed to parse config file: {}, using defaults", e.what());
        }
    } else {
        spdlog::info("Config file not found, using defaults");
    }

    if (enable_rt) {
#ifdef __linux__
        enableRealTimeScheduling();
        setCpuAffinity(cpu_cores);
#else
        spdlog::warn("RT optimization only supported on Linux");
#endif
    }

    // Run benchmark if requested
    if (run_benchmark) {
#ifdef HAVE_XGBOOST
        spdlog::info("");
        spdlog::info("=== XGBoost Inference Benchmark ===");
        spdlog::info("Model: {}", mm_config.xgboost.model_path);
        spdlog::info("Max iterations: {}", mm_config.xgboost.max_iterations > 0 ? std::to_string(mm_config.xgboost.max_iterations) : "all");
        spdlog::info("");

        const int BENCHMARK_ITERATIONS = 1000;

        // Test CPU inference
        spdlog::info("Loading model for CPU inference...");
        itch::ITCHPredictor cpu_predictor;
        if (cpu_predictor.load_model(mm_config.xgboost.model_path, false, 0, 0)) {
            if (mm_config.xgboost.max_iterations > 0) {
                cpu_predictor.set_max_iterations(mm_config.xgboost.max_iterations);
            }
            spdlog::info("Running CPU benchmark ({} iterations)...", BENCHMARK_ITERATIONS);
            auto cpu_result = cpu_predictor.benchmark(BENCHMARK_ITERATIONS);
            if (cpu_result.success) {
                spdlog::info("CPU Results:");
                spdlog::info("  avg: {:.2f} us", cpu_result.avg_us);
                spdlog::info("  min: {:.2f} us", cpu_result.min_us);
                spdlog::info("  max: {:.2f} us", cpu_result.max_us);
                spdlog::info("  p50: {:.2f} us", cpu_result.p50_us);
                spdlog::info("  p99: {:.2f} us", cpu_result.p99_us);
            } else {
                spdlog::error("CPU benchmark failed: {}", cpu_predictor.last_error());
            }
        } else {
            spdlog::error("Failed to load model for CPU: {}", cpu_predictor.last_error());
        }

        spdlog::info("");

        // Test GPU inference
        spdlog::info("Loading model for GPU inference...");
        itch::ITCHPredictor gpu_predictor;
        if (gpu_predictor.load_model(mm_config.xgboost.model_path, true,
                                      mm_config.xgboost.gpu_device_id,
                                      mm_config.xgboost.verbosity)) {
            if (mm_config.xgboost.max_iterations > 0) {
                gpu_predictor.set_max_iterations(mm_config.xgboost.max_iterations);
            }
            spdlog::info("Running GPU benchmark ({} iterations)...", BENCHMARK_ITERATIONS);
            auto gpu_result = gpu_predictor.benchmark(BENCHMARK_ITERATIONS);
            if (gpu_result.success) {
                spdlog::info("GPU Results:");
                spdlog::info("  avg: {:.2f} us", gpu_result.avg_us);
                spdlog::info("  min: {:.2f} us", gpu_result.min_us);
                spdlog::info("  max: {:.2f} us", gpu_result.max_us);
                spdlog::info("  p50: {:.2f} us", gpu_result.p50_us);
                spdlog::info("  p99: {:.2f} us", gpu_result.p99_us);
            } else {
                spdlog::error("GPU benchmark failed: {}", gpu_predictor.last_error());
            }
        } else {
            spdlog::warn("Failed to load model for GPU: {}", gpu_predictor.last_error());
            spdlog::warn("(GPU may not be available)");
        }

        spdlog::info("");
        spdlog::info("=== Benchmark Complete ===");
#else
        spdlog::error("XGBoost not compiled in - cannot run benchmark");
#endif
        return 0;
    }

    mm::MarketMakerFSM fsm(mm_config);

    try {
        // Connect to Project 24 via Disruptor shared memory
        spdlog::info("Connecting to Order Gateway via Disruptor (shm: {})...", disruptor_shm_name);
        g_client = std::make_unique<gateway::DisruptorClient>(disruptor_shm_name);
        g_client->connect();
        spdlog::info("Connected to Order Gateway (Disruptor Mode)");

        std::signal(SIGINT, signalHandler);
        std::signal(SIGTERM, signalHandler);

        spdlog::info("");
        spdlog::info("Market Maker FSM running");
        spdlog::info("Press Ctrl+C to stop");
        spdlog::info("");

        // For periodic stats writing
        auto last_stats_time = std::chrono::steady_clock::now();

        while (g_running) {
            try {
                // Process any fill notifications from Project 26
                fsm.processFills();

                // Read BBO from Disruptor (10ms timeout)
                gateway::BBOData bbo_data = g_client->read_bbo(10000);

                // Convert and process BBO
                mm::BBO bbo = convertBboData(bbo_data);
                if (bbo.valid) {
                    // Measure actual FSM processing latency (includes XGBoost inference)
                    static uint64_t sample_counter = 0;
                    if (++sample_counter % 100 == 0) {
                        gateway::LatencyMeasurement measure(g_parse_latency);
                        fsm.onBboUpdate(bbo);
                    } else {
                        fsm.onBboUpdate(bbo);
                    }
                    g_quotes_generated.fetch_add(1);
                }

                // Write stats for Trading UI every 500ms
                auto now = std::chrono::steady_clock::now();
                if (std::chrono::duration_cast<std::chrono::milliseconds>(now - last_stats_time).count() >= 500) {
                    // Write to both current directory (dev) and production path
                    g_parse_latency.saveStatsJson("p25_stats.json", g_quotes_generated.load());
                    mkdir("/opt/trading/stats", 0755);
                    g_parse_latency.saveStatsJson("/opt/trading/stats/p25_stats.json", g_quotes_generated.load());
                    last_stats_time = now;
                }

            } catch (const std::exception& e) {
                if (g_running) {
                    std::string error_msg = e.what();
                    if (error_msg.find("timeout") != std::string::npos) {
                        continue;  // Timeout is normal, keep polling
                    }
                    spdlog::error("Error processing BBO: {}", e.what());
                    break;
                }
            }
        }

        spdlog::info("");
        spdlog::info("Shutting down...");

        if (g_client) {
            g_client->disconnect();
        }

        // Print performance statistics
        if (g_parse_latency.count() > 0) {
            g_parse_latency.printSummary("Project 25 (Disruptor)");
            g_parse_latency.saveToFile("project25_latency.csv");
        }

        spdlog::info("Shutdown complete");

    } catch (const std::exception& e) {
        spdlog::error("Fatal error: {}", e.what());
        return 1;
    }

    return 0;
}
