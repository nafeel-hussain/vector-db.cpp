// main.cpp
// Parses args and starts the CLI.

#include "CLI.h"

#include <cstdlib>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    uint32_t dim = 128; // SIFT1M default; override with --dim
    std::string benchmarkPath = "BENCHMARKS.md";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--dim" && i + 1 < argc) {
            dim = static_cast<uint32_t>(std::stoul(argv[++i]));
        } else if (arg == "--benchmarks" && i + 1 < argc) {
            benchmarkPath = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "usage: " << argv[0] << " [--dim N] [--benchmarks path]\n";
            return 0;
        }
    }

    CLI cli(dim, benchmarkPath);
    cli._run();

    return 0;
}