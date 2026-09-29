#include "CLI.h"
#include "Benchmark.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

CLI::CLI(uint32_t dim, std::string benchmarkOutputPath_)
    : coll(dim),
      index(coll),
      benchmarkOutputPath(std::move(benchmarkOutputPath_)) {}

std::vector<float> CLI::parse_vector_csv(const std::string& token) {
    std::vector<float> v;
    std::stringstream ss(token);
    std::string piece;
    while (std::getline(ss, piece, ',')) {
        if (piece.empty()) continue;
        v.push_back(std::stof(piece));
    }
    return v;
}

void CLI::print_neighbors(const std::vector<Neighbor>& results) {
    if (results.empty()) {
        std::cout << "  (no results)\n";
        return;
    }
    std::cout << "  rank  id                distance\n";
    for (size_t i = 0; i < results.size(); ++i) {
        std::cout << "  " << (i + 1) << "     " << results[i].id << "     " << results[i].distance << "\n";
    }
}

void CLI::print_help() const {
    std::cout <<
        "Commands:\n"
        "  insert <id> <v1,v2,...,vN>          insert one vector\n"
        "  bulk_insert <path.csv>              insert many; each line \"id,v1,v2,...\"\n"
        "  remove <id>                         delete by id\n"
        "  get <id>                            print the stored vector for id\n"
        "  query <v1,v2,...,vN> <k>            top-k nearest neighbors (brute-force)\n"
        "  benchmark <numVectors> <k> [label]  scale benchmark -> BENCHMARKS.md\n"
        "  load <path>                   bulk-load a .fvecs file (ids = row index)\n"
        "  recall <q.fvecs> <gt.ivecs> <k> [n]  recall@k against groundtruth\n"
        "  stats                               size / dimension / estimated memory\n"
        "  help                                show this list\n"
        "  exit | quit                         leave the REPL\n";
}

void CLI::cmd_insert(std::istringstream& args) {
    std::string id, vecToken;
    if (!(args >> id >> vecToken)) {
        std::cout << "usage: insert <id> <v1,v2,...,vN>\n";
        return;
    }
    try {
        std::vector<float> v = parse_vector_csv(vecToken);
        auto start = std::chrono::high_resolution_clock::now();
        bool ok = coll.insert_vector(id, v);
        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();

        if (ok) {
            std::cout << "OK inserted id=" << id << " dim=" << v.size() << " (" << ms << " ms)\n";
        } else {
            std::cout << "FAILED to insert id=" << id << " (dimension mismatch or duplicate id?)\n";
        }
    } catch (const std::exception& e) {
        std::cout << "ERROR parsing vector: " << e.what() << "\n";
    }
}

void CLI::cmd_bulkInsert(std::istringstream& args) {
    std::string path;
    if (!(args >> path)) {
        std::cout << "usage: bulk_insert <path/to/file.csv>   (each line: id,v1,v2,...)\n";
        return;
    }

    std::ifstream file(path);
    if (!file.is_open()) {
        std::cout << "ERROR: could not open file '" << path << "'\n";
        return;
    }

    // collection has no bulk-insert API, so this loops insert_vector() per
    // line and times the whole batch.
    size_t lineNo = 0, inserted = 0, parseErrors = 0, total = 0;
    std::string line;
    auto start = std::chrono::high_resolution_clock::now();
    while (std::getline(file, line)) {
        ++lineNo;
        if (line.empty()) continue;

        std::stringstream ss(line);
        std::string id;
        if (!std::getline(ss, id, ',')) { ++parseErrors; continue; }

        try {
            std::vector<float> v;
            std::string comp;
            while (std::getline(ss, comp, ',')) {
                if (comp.empty()) continue;
                v.push_back(std::stof(comp));
            }
            if (v.empty()) { ++parseErrors; continue; }
            ++total;
            if (coll.insert_vector(id, v)) ++inserted;
        } catch (const std::exception&) {
            ++parseErrors;
            std::cout << "  (skipping malformed line " << lineNo << ")\n";
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();

    std::cout << "Bulk insert: " << inserted << "/" << total << " rows inserted"
              << (parseErrors ? (", " + std::to_string(parseErrors) + " lines skipped (parse errors)") : "")
              << " (" << ms << " ms total"
              << (total ? (", " + std::to_string(ms / static_cast<double>(total)) + " ms/vector avg)") : ")")
              << "\n";
}

std::vector<std::vector<float>> CLI::read_fvecs(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        throw std::runtime_error("could not open '" + path + "'");
    }

    std::vector<std::vector<float>> vectors;
    int32_t dim = 0;
    while (f.read(reinterpret_cast<char*>(&dim), sizeof(int32_t))) {
        if (dim <= 0) throw std::runtime_error("corrupt .fvecs: non-positive dim in '" + path + "'");
        std::vector<float> v(static_cast<size_t>(dim));
        if (!f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(dim) * sizeof(float))) {
            throw std::runtime_error("truncated .fvecs file '" + path + "'");
        }
        vectors.push_back(std::move(v));
    }
    return vectors;
}

std::vector<std::vector<int32_t>> CLI::read_ivecs(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        throw std::runtime_error("could not open '" + path + "'");
    }

    std::vector<std::vector<int32_t>> rows;
    int32_t count = 0;
    while (f.read(reinterpret_cast<char*>(&count), sizeof(int32_t))) {
        if (count <= 0) throw std::runtime_error("corrupt .ivecs: non-positive count in '" + path + "'");
        std::vector<int32_t> row(static_cast<size_t>(count));
        if (!f.read(reinterpret_cast<char*>(row.data()), static_cast<std::streamsize>(count) * sizeof(int32_t))) {
            throw std::runtime_error("truncated .ivecs file '" + path + "'");
        }
        rows.push_back(std::move(row));
    }
    return rows;
}

void CLI::cmd_loadFvecs(std::istringstream& args) {
    std::string path;
    std::string label = "sift_load";
    if (!(args >> path)) {
        std::cout << "usage: load <path/to/sift_base.fvecs> [label]\n";
        return;
    }
    args >> label; // optional

    try {
        std::cout << "Reading '" << path << "'...\n";
        auto vectors = read_fvecs(path);
        std::cout << "Read " << vectors.size() << " vectors";
        if (!vectors.empty()) std::cout << " (dim=" << vectors[0].size() << ")";
        std::cout << ". Inserting (ids = row index, required for recall_fvecs to line up "
                     "with the groundtruth file)...\n";

        size_t inserted = 0;
        std::vector<double> insertLatencies;
        insertLatencies.reserve(vectors.size());

        auto start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < vectors.size(); ++i) {
            if (vectors[i].size() != coll._dim()) {
                std::cout << "  skipping row " << i << ": dim " << vectors[i].size()
                          << " != collection dim " << coll._dim() << "\n";
                continue;
            }
            auto rowStart = std::chrono::high_resolution_clock::now();
            bool ok = coll.insert_vector(std::to_string(i), vectors[i]);
            auto rowEnd = std::chrono::high_resolution_clock::now();
            insertLatencies.push_back(std::chrono::duration<double, std::milli>(rowEnd - rowStart).count());
            if (ok) ++inserted;
        }
        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();

        std::cout << "Inserted " << inserted << "/" << vectors.size() << " vectors ("
                  << ms << " ms total, " << (ms / static_cast<double>(std::max<size_t>(1, vectors.size())))
                  << " ms/vector avg)\n";

        auto insertStats = BenchmarkRunner::compute_stats(insertLatencies);

        BenchmarkResult result;
        result.label = label;
        result.indexType = "brute-force";
        result.numVectors = vectors.size();
        result.dimension = coll._dim();
        result.k = 0; // not a query benchmark
        result.insertLatency = insertStats;
        result.recallAtK = -1.0; // n/a — load_fvecs doesn't run queries
        result.memoryBytesEstimate = static_cast<double>(coll._size()) * coll._dim() * sizeof(float);
        result.params.emplace_back("source_file", path);
        result.params.emplace_back("inserted", std::to_string(inserted) + "/" + std::to_string(vectors.size()));

        BenchmarkRunner runner(benchmarkOutputPath);
        runner.append_result_to_markdown(result);
        std::cout << "Appended results to " << benchmarkOutputPath << "\n";
    } catch (const std::exception& e) {
        std::cout << "ERROR: " << e.what() << "\n";
    }
}

void CLI::cmd_recallFvecs(std::istringstream& args) {
    std::string queryPath, groundtruthPath;
    uint32_t k;
    size_t numQueries = 100;
    std::string label = "sift_recall";
    if (!(args >> queryPath >> groundtruthPath >> k)) {
        std::cout << "usage: recall <queries.fvecs> <groundtruth.ivecs> <k> [numQueries] [label]\n";
        return;
    }
    args >> numQueries; // optional
    args >> label;      // optional

    try {
        std::cout << "Reading queries from '" << queryPath << "' and groundtruth from '"
                  << groundtruthPath << "'...\n";
        auto queries = read_fvecs(queryPath);
        auto groundtruth = read_ivecs(groundtruthPath);

        if (queries.size() != groundtruth.size()) {
            std::cout << "WARNING: " << queries.size() << " queries but " << groundtruth.size()
                      << " groundtruth rows — using the smaller count.\n";
        }
        size_t available = std::min(queries.size(), groundtruth.size());
        numQueries = std::min(numQueries, available);

        if (numQueries == 0) {
            std::cout << "Nothing to evaluate.\n";
            return;
        }

        double recallSum = 0.0;
        std::vector<double> queryLatencies;
        queryLatencies.reserve(numQueries);

        for (size_t i = 0; i < numQueries; ++i) {
            if (queries[i].size() != coll._dim()) {
                std::cout << "  skipping query " << i << ": dim " << queries[i].size()
                          << " != collection dim " << coll._dim() << "\n";
                continue;
            }

            auto start = std::chrono::high_resolution_clock::now();
            std::vector<Neighbor> approx = index.search(queries[i], k);
            auto end = std::chrono::high_resolution_clock::now();
            queryLatencies.push_back(std::chrono::duration<double, std::milli>(end - start).count());

            std::vector<Neighbor> truth;
            size_t truthLimit = std::min<size_t>(k, groundtruth[i].size());
            truth.reserve(truthLimit);
            for (size_t j = 0; j < truthLimit; ++j) {
                truth.push_back(Neighbor{0.0f, std::to_string(groundtruth[i][j])});
            }

            recallSum += BenchmarkRunner::compute_recall_at_K(approx, truth, k);
        }

        double avgRecall = recallSum / static_cast<double>(numQueries);
        auto latStats = BenchmarkRunner::compute_stats(queryLatencies);

        std::cout << "recall@" << k << " over " << numQueries << " queries = " << avgRecall << "\n";
        std::cout << "query latency p50=" << latStats.p50Ms << "ms  p95=" << latStats.p95Ms << "ms\n";

        BenchmarkResult result;
        result.label = label;
        result.indexType = "brute-force";
        result.numVectors = coll._size();
        result.dimension = coll._dim();
        result.k = k;
        result.queryLatency = latStats;
        result.recallAtK = avgRecall;
        result.memoryBytesEstimate = static_cast<double>(coll._size()) * coll._dim() * sizeof(float);
        result.params.emplace_back("queries_evaluated", std::to_string(numQueries));
        result.params.emplace_back("query_file", queryPath);
        result.params.emplace_back("groundtruth_file", groundtruthPath);

        BenchmarkRunner runner(benchmarkOutputPath);
        runner.append_result_to_markdown(result);
        std::cout << "Appended results to " << benchmarkOutputPath << "\n";
    } catch (const std::exception& e) {
        std::cout << "ERROR: " << e.what() << "\n";
    }
}

void CLI::cmd_remove(std::istringstream& args) {
    std::string id;
    if (!(args >> id)) {
        std::cout << "usage: remove <id>\n";
        return;
    }
    bool ok = coll.remove_vector(id);
    std::cout << (ok ? "OK removed id=" : "NOT FOUND id=") << id << "\n";
}

void CLI::cmd_get(std::istringstream& args) {
    std::string id;
    if (!(args >> id)) {
        std::cout << "usage: get <id>\n";
        return;
    }
    try {
        std::vector<float> v = coll.get_vector(id);
        if (v.empty()) {
            std::cout << "NOT FOUND id=" << id << "\n";
            return;
        }
        std::cout << "id=" << id << " dim=" << v.size() << " vector=[";
        for (size_t i = 0; i < v.size(); ++i) {
            std::cout << v[i];
            if (i + 1 < v.size()) std::cout << ", ";
        }
        std::cout << "]\n";
    } catch (const std::exception& e) {
        std::cout << "NOT FOUND id=" << id << " (" << e.what() << ")\n";
    }
}

void CLI::cmd_query(std::istringstream& args) {
    std::string vecToken;
    uint32_t k;
    if (!(args >> vecToken >> k)) {
        std::cout << "usage: query <v1,v2,...,vN> <k>\n";
        return;
    }
    try {
        std::vector<float> q = parse_vector_csv(vecToken);

        auto start = std::chrono::high_resolution_clock::now();
        std::vector<Neighbor> results = index.search(q, k);
        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();

        std::cout << "Top-" << k << ", " << ms << " ms:\n";
        print_neighbors(results);
    } catch (const std::exception& e) {
        std::cout << "ERROR: " << e.what() << "\n";
    }
}

void CLI::cmd_benchmark(std::istringstream& args) {
    size_t numVectors;
    uint32_t k;
    std::string label = "adhoc";
    if (!(args >> numVectors >> k)) {
        std::cout << "usage: benchmark <numVectors> <k> [label]\n";
        return;
    }
    args >> label; // optional

    std::cout << "Running benchmark: N=" << numVectors << " dim=" << coll._dim() << " k=" << k
              << " label=\"" << label << "\" ... (this inserts real data into the live collection)\n";

    BenchmarkRunner runner(benchmarkOutputPath);
    BenchmarkResult result = runner.run_benchmark(coll, index, label, numVectors, k);
    runner.append_result_to_markdown(result);

    std::cout << "Insert latency  p50=" << result.insertLatency.p50Ms
              << "ms  p95=" << result.insertLatency.p95Ms << "ms\n";
    std::cout << "Query  latency  p50=" << result.queryLatency.p50Ms
              << "ms  p95=" << result.queryLatency.p95Ms << "ms\n";
    std::cout << "Estimated memory: " << (result.memoryBytesEstimate / (1024.0 * 1024.0)) << " MB\n";
    std::cout << "Appended results to " << benchmarkOutputPath << "\n";
}

void CLI::cmd_stats() const {
    uint32_t dim = coll._dim();
    uint32_t size = coll._size();
    double memMB = (static_cast<double>(size) * dim * sizeof(float)) / (1024.0 * 1024.0);
    std::cout << "size=" << size << "  dimension=" << dim
              << "  estimated_memory=" << memMB << " MB\n";
}

bool CLI::execute_command(const std::string& line) {
    std::istringstream iss(line);
    std::string cmd;
    iss >> cmd;

    if (cmd.empty()) return true;

    if (cmd == "exit" || cmd == "quit") {
        return false;
    } else if (cmd == "help" || cmd == "?") {
        print_help();
    } else if (cmd == "insert") {
        cmd_insert(iss);
    } else if (cmd == "bulk_insert") {
        cmd_bulkInsert(iss);
    } else if (cmd == "remove") {
        cmd_remove(iss);
    } else if (cmd == "get") {
        cmd_get(iss);
    } else if (cmd == "query") {
        cmd_query(iss);
    } else if (cmd == "benchmark") {
        cmd_benchmark(iss);
    } else if (cmd == "load") {
        cmd_loadFvecs(iss);
    } else if (cmd == "recall") {
        cmd_recallFvecs(iss);
    } else if (cmd == "stats") {
        cmd_stats();
    } else {
        std::cout << "Unknown command '" << cmd << "'. Type 'help' for the command list.\n";
    }
    return true;
}

void CLI::_run() {
    std::cout << "Vector DB CLI: dim=" << coll._dim() << ", type 'help' for commands, 'exit' to quit.\n";
    std::string line;
    while (true) {
        std::cout << "vecdb> ";
        if (!std::getline(std::cin, line)) {
            std::cout << "\n";
            break; // EOF (Ctrl-D)
        }
        if (!execute_command(line)) break;
    }
}