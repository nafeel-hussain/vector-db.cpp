# SIFT1M benchmark: commands

All commands run from the project root in PowerShell.
Requires a 64-bit compiler: `g++ -dumpmachine` should print `x86_64-w64-mingw32`.

## 1. Build

Release build (use this for benchmarks):
```
g++ -std=c++17 -O2 -Wall -Wextra -D_GLIBCXX_ASSERTIONS -I include src/*.cpp -o build/vectordb
```

Debug build (use this with gdb; much slower):
```
g++ -std=c++17 -g -O0 -Wall -Wextra -D_GLIBCXX_ASSERTIONS -I include src/*.cpp -o build/vectordb_gdb
```

## 2. Run

Normal:
```
.\build\vectordb --dim 128 --benchmarks BENCHMARKS.md
```

Under gdb (at the `(gdb)` prompt type `run`, and after a crash type `bt`):
```
gdb --args .\build\vectordb_gdb --dim 128 --benchmarks BENCHMARKS.md
```

## 3. Benchmark session (type these at the `vecdb>` prompt)

Load the base vectors:
```
load sift1m/sift_base.fvecs
```

Brute-force baseline (k=10, 1000 queries):
```
recall bf sift1m/sift_query.fvecs sift1m/sift_groundtruth.ivecs 10 1000 sift1m_k10
```

Build the IVF index (nlist, nprobe):
```
build_ivf 1000 32
```

IVF recall (put nlist and nprobe in the label):
```
recall ivf sift1m/sift_query.fvecs sift1m/sift_groundtruth.ivecs 10 1000 sift1m_nl1000_np32_k10
```

Show collection and index info:
```
stats
```

Quit:
```
exit
```

## Notes

- Argument order for `recall`: `<bf|ivf> <queries.fvecs> <groundtruth.ivecs> <k> <numQueries> <label>`
- Use 1000 queries for every row so recall and p95 are comparable.
- `build_ivf` build time is roughly proportional to nlist and maxIter (maxIter = 20).
- Do not use the `-O0` build for timing; it is much slower.
- Label format: `sift1m_nl<nlist>_np<nprobe>_k<k>`