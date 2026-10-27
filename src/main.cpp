#include "forgetune/compiler.hpp"
#include <iostream>
#include <fstream>
#include <sstream>

using namespace forgetune;

static std::string read_file(const std::string& path) {
    std::ifstream ifs(path);
    if (!ifs.is_open()) {
        throw std::runtime_error("Failed to open file for reading: " + path);
    }
    std::ostringstream ss;
    ss << ifs.rdbuf();
    return ss.str();
}

static void write_file(const std::string& path, const std::string& content) {
    std::ofstream ofs(path);
    if (!ofs.is_open()) {
        throw std::runtime_error("Failed to open file for writing: " + path);
    }
    ofs << content;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: forgetune_engine <command> [options]\n"
                  << "Commands:\n"
                  << "  --compile <graph.json> -o <output.cpp> [options]\n"
                  << "  --verify <graph.json>\n"
                  << "  --optimize <graph.json> -o <opt_graph.json> [--fusion]\n"
                  << "  --version\n";
        return 1;
    }

    std::string cmd = argv[1];

    if (cmd == "--version") {
        std::cout << "ForgeTune ML Compiler Engine v" << IR_SCHEMA_VERSION_MAJOR << "."
                  << IR_SCHEMA_VERSION_MINOR << " (C++17)\n";
        return 0;
    }

    try {
        if (cmd == "--verify") {
            if (argc < 3) {
                std::cerr << "Error: --verify requires <graph.json>\n";
                return 1;
            }
            std::string json_content = read_file(argv[2]);
            IRGraph graph = json_to_graph(json_content);
            auto vres = verify_graph(graph, true);
            std::cout << "Verification PASSED\n";
            return 0;
        }

        if (cmd == "--compile") {
            if (argc < 5) {
                std::cerr << "Error: --compile requires <graph.json> -o <output.cpp>\n";
                return 1;
            }
            std::string input_json = argv[2];
            std::string output_cpp;
            KernelConfig config;

            for (int i = 3; i < argc; ++i) {
                std::string arg = argv[i];
                if (arg == "-o" && i + 1 < argc) {
                    output_cpp = argv[++i];
                } else if (arg == "--variant" && i + 1 < argc) {
                    config.matmul_variant = string_to_matmul_variant(argv[++i]);
                } else if (arg == "--fusion") {
                    config.enable_fusion = true;
                } else if (arg == "--candidate-id" && i + 1 < argc) {
                    config.candidate_id = argv[++i];
                } else if (arg == "--tile-m" && i + 1 < argc) {
                    try { config.tile_m = std::stoi(argv[++i]); } catch (...) { config.tile_m = 16; }
                } else if (arg == "--tile-n" && i + 1 < argc) {
                    try { config.tile_n = std::stoi(argv[++i]); } catch (...) { config.tile_n = 16; }
                } else if (arg == "--tile-k" && i + 1 < argc) {
                    try { config.tile_k = std::stoi(argv[++i]); } catch (...) { config.tile_k = 16; }
                }
            }

            std::string err;
            if (!config.validate(&err)) {
                std::cerr << "Error: invalid kernel configuration: " << err << "\n";
                return 1;
            }

            if (output_cpp.empty()) {
                std::cerr << "Error: missing -o <output.cpp>\n";
                return 1;
            }

            std::string json_content = read_file(input_json);
            IRGraph graph = json_to_graph(json_content);

            Compiler compiler(config);
            auto res = compiler.compile(graph);

            if (!res.success) {
                std::cerr << "Compilation failed: " << res.error_message << "\n";
                return 1;
            }

            write_file(output_cpp, res.generated_cpp_source);
            std::cout << "Successfully generated " << output_cpp << "\n"
                      << res.optimization_report.summary << "\n";
            return 0;
        }

        if (cmd == "--optimize") {
            if (argc < 5) {
                std::cerr << "Error: --optimize requires <graph.json> -o <opt_graph.json>\n";
                return 1;
            }
            std::string input_json = argv[2];
            std::string output_json;
            bool enable_fusion = false;

            for (int i = 3; i < argc; ++i) {
                std::string arg = argv[i];
                if (arg == "-o" && i + 1 < argc) {
                    output_json = argv[++i];
                } else if (arg == "--fusion") {
                    enable_fusion = true;
                }
            }

            std::string json_content = read_file(input_json);
            IRGraph graph = json_to_graph(json_content);

            Optimizer opt;
            auto report = opt.optimize(graph, enable_fusion);

            std::string opt_json = graph_to_json(graph);
            write_file(output_json, opt_json);
            std::cout << "Successfully optimized graph to " << output_json << "\n"
                      << report.summary << "\n";
            return 0;
        }

        std::cerr << "Unknown command: " << cmd << "\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "ForgeTune Engine Error: " << e.what() << "\n";
        return 1;
    }
}
