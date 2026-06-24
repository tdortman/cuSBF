# cuSBF

[![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://tdortman.github.io/cuSBF/)
[![arXiv](https://img.shields.io/badge/arXiv-2606.24417-b31b1b.svg)](https://arxiv.org/abs/2606.24417)

## Overview

cuSBF is a high-performance GPU implementation of the [Super Bloom filter](https://www.biorxiv.org/content/10.64898/2026.03.17.712354v1.article-info), optimized for high-throughput batch k-mer insertion and query on nucleotide (DNA) and protein sequences (or any other sequence type as long as a valid alphabet is provided).

It exploits the streaming nature of sequence-derived k-mers by using **minimizers** to group consecutive k-mers sharing the same minimiser into super-k-mers, assigning all k-mers of a super-k-mer to the same 256-bit memory shard. This amortizes random memory accesses across consecutive k-mer queries, reducing memory-bandwidth pressure. The **findere** scheme further reduces false positives dramatically by inserting overlapping s-mers and requiring a full run of consecutive s-mer matches.

This library is the companion code for the paper **"cuSBF: A Minimizer-Aware Bloom Filter for Genomic Sequence Data on Modern GPUs"**.

## Features

- CUDA-accelerated batch k-mer insert and query from sequences
- Configurable k-mer length, minimiser width, s-mer width, and hash function count
- Minimizer-based shard selection for cache-efficient streaming queries
- Findere false-positive reduction via overlapping s-mer membership
- Header-only library design
- FASTA/FASTQ stream and file support

## Performance

![image](./docs/throughput_comparison.png)

Benchmarks use `Config<31, 28, 16, 4>` on an NVIDIA RTX PRO 6000 Blackwell GPU. CPU Super Bloom runs on an Intel Xeon W9-3595X with 120 threads.

Compared against:

- [CPU Super Bloom](https://github.com/EtienneC-K/SuperBloom)
- [GPU Blocked Bloom filter (GBBF)](https://github.com/NVIDIA/cuCollections)
- [GPU Cuckoo-GPU](https://github.com/tdortman/Cuckoo-GPU)
- [GPU Bulk Two-Choice Filter (TCF)](https://github.com/saltsystemslab/gpu-filters/tree/main/bulk-tcf)
- [GPU Counting Quotient Filter (GQF)](https://github.com/saltsystemslab/gpu-filters/tree/main/gqf)

### Smaller Filter (C. elegans, ~100M k-mers)

| Comparison           | Insert      | Query       |
| -------------------- | ----------- | ----------- |
| cuSBF vs Super Bloom | 92× faster  | 234× faster |
| cuSBF vs GBBF        | 9.1× faster | 7.7× faster |
| cuSBF vs Cuckoo-GPU  | 80× faster  | 8.0× faster |
| cuSBF vs TCF         | 12× faster  | 52× faster  |
| cuSBF vs GQF         | 69× faster  | 13× faster  |

### Large Filter (CHM13, ~3.1B k-mers)

| Comparison           | Insert       | Query       |
| -------------------- | ------------ | ----------- |
| cuSBF vs Super Bloom | 59× faster   | 165× faster |
| cuSBF vs GBBF        | 8.2× faster  | 7.6× faster |
| cuSBF vs Cuckoo-GPU  | 3427× faster | 7.8× faster |
| cuSBF vs TCF         | 12× faster   | 67× faster  |
| cuSBF vs GQF         | 42× faster   | 11× faster  |

### False Positive Rate

| Bits/k-mer | cuSBF `s=28` | cuSBF `s=30` | cuSBF `s=31` | GBBF    |
| ---------- | ------------ | ------------ | ------------ | ------- |
| 21.4       | 0.848%       | 0.951%       | 1.593%       | 3.069%  |
| 85.7       | 0.091%       | 0.107%       | 0.210%       | 0.126%  |
| 342.6      | 0.0095%      | 0.0114%      | 0.0264%      | 0.0273% |

## Requirements

- Linux (x86_64 or aarch64) with an NVIDIA GPU and driver
- CUDA Toolkit >= 13.1
- GCC or Clang host compiler (C++20)
- Meson and Ninja
- NVIDIA GPU with compute capability 8.0+ (Ampere, Lovelace, Hopper, Blackwell)

### Platform support

cuSBF is developed and tested on **Linux** only.

- **WSL2** on Windows with is a reasonable dev environment (See [NVIDIA docs](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)).
- **Native Windows and macOS** are not supported or tested. The build uses Linux-specific FASTX paths (for example `mmap`) and host tooling assumptions (GCC/Clang, GNU statement expressions in `CUSBF_TRY`/`CUSBF_UNWRAP`).

## Building

```bash
meson setup build
ninja -C build
```

When this repo is the root Meson project, **benchmarks**, **tests**, and **examples** build by default. As a subproject they are skipped unless you force them on.

| Option                 | Type    | Default    | Description                                                 |
| ---------------------- | ------- | ---------- | ----------------------------------------------------------- |
| `benchmarks`           | feature | `auto`     | Google Benchmark binaries                                   |
| `tests`                | feature | `auto`     | GoogleTest suite                                            |
| `examples`             | feature | `auto`     | Example CLI                                                 |
| `param_sweep`          | feature | `disabled` | Parameter-sweep binaries (large, see below)                 |
| `param_sweep_alphabet` | combo   | `dna`      | `dna` or `protein` when `param_sweep` is enabled            |
| `large_fastx_tests`    | feature | `disabled` | Large generated FASTX test (`CUSBF_LARGE_FASTX_*` env vars) |

Each **feature** option accepts `auto`, `enabled`, or `disabled`:

- `auto` — on for a standalone checkout, off when cuSBF is a subproject
- `enabled` / `disabled` — override regardless of project layout

> [!IMPORTANT]
> Enabling `param_sweep` builds many binaries (208 for the DNA alphabet). Leave it disabled unless you need that sweep.

```bash
# Default standalone build
meson setup build

# Faster configure: library + examples only
meson setup build -Dbenchmarks=disabled -Dtests=disabled

# Subproject consumer forcing tests on
meson setup build -Dtests=enabled

# Parameter sweep
meson setup build -Dparam_sweep=enabled
meson setup build -Dparam_sweep=enabled -Dparam_sweep_alphabet=protein
```

## Usage

Fallible APIs return `cusbf::Result<T>` (a thin wrapper over `cuda::std::expected<T, Error>`). Use `return Err(error)` (`cuda::std::unexpected<Error>`, deduces `Result<T>`) or `return Ok()` / `return {}` for `Result<void>`. For success with a value, `return value` is enough. Two helpers unwrap results:

| Macro                | On failure                                                                                        | Use when                                                          |
| -------------------- | ------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| `CUSBF_TRY(expr)`    | Copies the error, then `return cuda::std::unexpected<Error>(...)` from the **enclosing** function | The caller returns `Result` (library glue, `examples/cusbf-main`) |
| `CUSBF_UNWRAP(expr)` | `throw std::runtime_error(message())`                                                             | Tests, `main`, or other code that does not return `Result`        |

Both work as statements or in initializers (`auto x = CUSBF_UNWRAP(...)`). For full control (typed errors, exit codes), use `if (!result)` instead.

### Quick example (`CUSBF_UNWRAP`)

```cpp
#include <cusbf/filter.cuh>

using Config = cusbf::Config<31, 28, 16, 4>;

int main() {
    cusbf::filter<Config> filter(1 << 24);

    CUSBF_UNWRAP(filter.insert_sequence("ACGTACGTACGTACGTACGTACGTACGTACGT"));
    const auto hits = CUSBF_UNWRAP(filter.contains_sequence("ACGTACGTACGTACGTACGTACGTACGTACGT"));

    CUSBF_UNWRAP(filter.insert_fastx_file("reference.fasta"));
    const auto summary = CUSBF_UNWRAP(filter.query_fastx_file("queries.fastq"));

    (void)hits;
    (void)summary;
    return 0;
}
```

### Propagating errors (`CUSBF_TRY`)

When the caller already returns `Result`, use `CUSBF_TRY` so failures propagate without exceptions:

```cpp
[[nodiscard]] cusbf::Result<void> run(cusbf::filter<Config>& filter) {
    CUSBF_TRY(filter.insert_fastx_file("reference.fasta"));
    const auto summary = CUSBF_TRY(filter.query_fastx_file("queries.fastq"));
    (void)summary;
    return cusbf::Ok();
}
```

Async device APIs, record batches, and streaming FASTX callbacks follow the same pattern. `filter.load_factor()` and `filter.filter_bits()` are synchronous and do not return `Result`.

### Inspecting errors

```cpp
if (const auto result = filter.query_fastx_file("queries.fastq"); !result) {
    const cusbf::Error& err = result.error();
    std::cerr << err.message() << '\n';
    if (const cusbf::FastxParseError* parse = err.as_fastx_parse()) {
        // parse->location.file / .line / .column
    }
    return 1;
}
```

`CUSBF_CUDA_TRY` wraps CUDA runtime calls into `Result<void>`; `CUSBF_CUDA_CALL` / `CUSBF_CUDA_ABORT` are for throw/abort paths only.

### Configuration Options

The `Config` template accepts the following parameters:

| Parameter       | Description                                            | Default       |
| --------------- | ------------------------------------------------------ | ------------- |
| `K`             | k-mer length (max depends on alphabet)                 | -             |
| `S`             | s-mer width for findere Bloom hash seed (1-K)          | -             |
| `M`             | Minimiser width for shard selection (1-K)              | -             |
| `HashCount`     | Number of independent Bloom hash functions (4,8,12,16) | 4             |
| `CudaBlockSize` | CUDA threads per block                                 | 256           |
| `Alphabet`      | Symbol encoding (DNA or protein)                       | `DnaAlphabet` |

### Protein Alphabet Support

```cpp
#include <cusbf/filter.cuh>

using ProteinConfig = cusbf::Config<12, 10, 6, 4, 256, cusbf::ProteinAlphabet>;

[[nodiscard]] cusbf::Result<void> run_protein() {
    cusbf::filter<ProteinConfig> filter(1 << 24);
    CUSBF_TRY(filter.insert_sequence("ACDEFGHIKLMNPQRSTVWY"));
    const auto hits = CUSBF_TRY(filter.contains_sequence("ACDEFGHIKLMNPQRSTVWY"));
    (void)hits;
    return cusbf::Ok();
}
```
