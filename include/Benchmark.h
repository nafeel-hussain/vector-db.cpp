#pragma once
#include "vector.hpp"
#include "search.hpp"
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

struct LatencyStats {
    double p50Ms = 0.0;
    double p95Ms = 0.0;
    double p99Ms = 0.0;
    double avgMs = 0.0;
    double minMs = 0.0;
    double maxMs = 0.0;
    size_t sampleCount = 0;
};

struct BenchmarkResult {
    std::string label;
    std::string indexType;            // searching method (brute-force, ivf, hnsw)
    size_t numVectors = 0;
    uint32_t dimension = 0;
    uint32_t k = 0;
    LatencyStats queryLatency;
    double memoryBytesEstimate = 0.0; // size * dim * sizeof(float); collection has no memory getter
    double recallAtK = -1.0;          // -1.0 means not measured 
    std::vector<std::pair<std::string, std::string>> params; // free-form notes.
};

class BenchmarkRunner {
private:
    std::string outputPath;
    void ensure_file_exists();

public:
    explicit BenchmarkRunner(const std::string& outputMarkdownPath = "BENCHMARKS.md");

    // Deterministic random vectors generating func.
    static std::vector<std::vector<float>> generate_random_vectors(size_t count, uint32_t dim, unsigned seed = 42);
    static std::vector<std::string> generate_Ids(size_t count, const std::string& prefix = "vector", unsigned seed = 42);

    // Inserts `numVectors` random vectors into `coll` (timed per-call), then
    // runs `numQueries` random top-k queries through `index` (timed
    // per-call). `index` must already be bound to `coll` (e.g.
    // BruteForceSearch(coll)) — this function does not construct one.
    // NOTE: this mutates `coll` — it inserts real rows, same as any other
    // insert. Dimension is read from coll._dim(), not passed in, since
    // collection's dimension is fixed at construction.
    BenchmarkResult run_benchmark(collection& coll,
                                       Index& index,
                                       const std::string& label,
                                       size_t numVectors,
                                       uint32_t k,
                                       size_t numQueries = 100,
                                       unsigned seed = 67);

    // Compares one approximate-index result set against one brute-force
    // ground-truth result set for the SAME query, by id overlap.
    static double compute_recall_at_K(const std::vector<Neighbor>& approxResults,
                                    const std::vector<Neighbor>& groundTruth,
                                    uint32_t k);

    static double average_recall_at_K(const std::vector<std::vector<Neighbor>>& approxPerQuery,
                                    const std::vector<std::vector<Neighbor>>& groundTruthPerQuery,
                                    uint32_t k);

    static LatencyStats compute_stats(std::vector<double>& latenciesMs); // sorts in place

    void append_result_to_markdown(const BenchmarkResult& result);
    void append_section(const std::string& heading);
    void append_note(const std::string& note);
};