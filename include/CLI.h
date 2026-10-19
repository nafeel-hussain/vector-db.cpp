#pragma once

/*
Interactive REPL during development

Commands:
  insert <id> <v1,v2,...,vN>          insert one vector

  bulk_insert <path/to/file.csv>      insert many; each line "id,v1,v2,..."

  remove <id>                         delete by id

  get <id>                            print the stored vector for id

  query <bf|ivf> <v1,v2,...,vN> <k>   top-k nearest neighbors via the chosen method

  build_ivf <nlist> <nprobe>          (re)build the IVF index from the current collection;
                                       must be run after data is loaded, before 'query ivf'
                                       or 'recall ivf' will work

  set_nprobe <nprobe>                  change nprobe for the existing IVF index

  benchmark <numVectors> <k> [label]   scale benchmark -> BENCHMARKS.md (dim fixed at startup,
                                       always runs against brute-force)

  load <path> [label]                  bulk-load a .fvecs file; ids assigned as "0","1",...,"N-1"
                                       matching each vector's row index (needed for recall);
                                       appends per-insert latency stats to BENCHMARKS.md

  recall <bf|ivf> <queries.fvecs> <groundtruth.ivecs> <k> [numQueries] [label]
                                       run queries through the chosen method, report average
                                       recall@k against the groundtruth neighbor ids, and
                                       append the result to BENCHMARKS.md

  stats                               size / dimension / estimated memory

  help                                show this list

  exit | quit                         leave the REPL
*/

#include "vector.hpp"
#include "search.hpp"
#include "ivf.hpp"
#include <cstdint>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

class CLI {
public:
    explicit CLI(uint32_t dim, std::string benchmarkOutputPath = "BENCHMARKS.md");

    void _run();                                     // blocking REPL loop over stdin
    bool execute_command(const std::string& line);   // false => exit/quit requested

private:
    collection coll;                               // constructed first
    BruteForceSearch index;                        // holds a reference to coll, must be declared after it
    std::unique_ptr<InvertedFileIndex> ivfIndex;   // null until 'build_ivf' is run; rebuild 
    std::string benchmarkOutputPath;

    void cmd_insert(std::istringstream& args);
    void cmd_bulkInsert(std::istringstream& args);
    void cmd_remove(std::istringstream& args);
    void cmd_get(std::istringstream& args);
    void cmd_buildIvf(std::istringstream& args);
    void cmd_setnprobe(std::istringstream& args);
    void cmd_query(std::istringstream& args);
    void cmd_benchmark(std::istringstream& args);
    void cmd_loadFvecs(std::istringstream& args);
    void cmd_recallFvecs(std::istringstream& args);
    void cmd_stats() const;
    void print_help() const;

    Index* select_index(const std::string& method);
    static std::vector<float> parse_vector_csv(const std::string& token);
    static void print_neighbors(const std::vector<Neighbor>& results);

    // Binary readers for the texmex corpus format used by SIFT1M and friends.
    // Format: repeated records of [int32 dim][dim x float32] for .fvecs,
    // or [int32 count][count x int32] for .ivecs (groundtruth neighbor ids).
    static std::vector<std::vector<float>> read_fvecs(const std::string& path);
    static std::vector<std::vector<int32_t>> read_ivecs(const std::string& path);
};