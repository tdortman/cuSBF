#pragma once

#include <benchmark/benchmark.h>
#include <cuda/__cmath/ceil_div.h>
#include <cuda_runtime.h>
#include <thrust/count.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/random.h>
#include <thrust/transform.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <unistd.h>

#include <cuda/std/bit>

#include <cusbf/device_span.cuh>
#include <cusbf/filter.cuh>
#include <cusbf/helpers.cuh>
#include <cusbf/superbloom_ffi.hpp>

#include "fastx_workload.hpp"
namespace benchmark_common {

inline bool g_cpuFastxParallelizeRecords = false;
inline uint64_t g_cpuFastxNumRecords =
    std::max<uint64_t>(1, static_cast<uint64_t>(std::thread::hardware_concurrency()));

// FASTX insert workload shared by filter-comparison and fpr-fastx benchmarks.
inline std::string g_insertFastxPath;
// 0 = use std::thread::hardware_concurrency() when building the CPU parallel FASTA.
inline uint64_t g_fastxCpuNumRecords = 0;

enum class FastxGpuPrepareKind {
    HostOnly,
    SequenceOnDevice,
    PackedKmers,
};

struct FastxInsertWorkload {
    std::vector<char> host_insert_sequence;
    thrust::device_vector<char> d_insert_sequence;
    thrust::device_vector<uint64_t> d_insert_packed_kmers;
    uint64_t insert_kmers = 0;
    std::string cpu_insert_fastx_path;
    FastxGpuPrepareKind gpuPrepareLevel = FastxGpuPrepareKind::HostOnly;
};

inline std::unique_ptr<FastxInsertWorkload> g_fastxInsertWorkload;

// Chunk size for hash-filter encode/insert/query (avoids multi-GB opKeys buffers).
inline uint64_t g_fastxChunkKmers = 0;
inline bool g_fastxChunkKmersUserSet = false;
inline uint64_t g_fastxChunkKmersResolved = 0;
inline uint64_t g_fastxChunkKmersResolvedReserved = 0;
inline uint64_t g_fastxChunkKmersResolvedScratchBpp = 0;

constexpr uint64_t kFastxChunkBytesPerKmer = sizeof(uint64_t) + sizeof(uint8_t);
// TCF per-chunk device footprint: harness opKeys/queryHits, cached bulk scratch, and
// thrust::sort_by_key temp inside attach_lossy_buffers{,_recovery} (~2x uint64_t keys).
constexpr uint64_t kTcfSortTempBytesPerKmer = sizeof(uint64_t) * 2;
constexpr uint64_t kTcfFastxChunkBytesPerKmer = kFastxChunkBytesPerKmer + sizeof(uint16_t) +
                                                sizeof(bool) + sizeof(uint64_t) +
                                                kTcfSortTempBytesPerKmer;
constexpr uint64_t kFastxChunkFloorKmers = 1ULL << 20;

inline void cudaDeviceMemInfo(size_t& freeBytes, size_t& totalBytes) {
    CUSBF_CUDA_CALL(cudaMemGetInfo(&freeBytes, &totalBytes));
}

inline size_t cudaDeviceFreeBytes() {
    size_t freeBytes = 0;
    size_t totalBytes = 0;
    cudaDeviceMemInfo(freeBytes, totalBytes);
    return freeBytes;
}

inline uint64_t fastxInsertSequenceDeviceBytes() {
    return g_fastxInsertWorkload ? g_fastxInsertWorkload->host_insert_sequence.size() : 0;
}

inline uint64_t fastxPackedKmersDeviceBytes() {
    if (!g_fastxInsertWorkload) {
        return 0;
    }
    if (g_fastxInsertWorkload->gpuPrepareLevel < FastxGpuPrepareKind::PackedKmers) {
        return 0;
    }
    return g_fastxInsertWorkload->insert_kmers * sizeof(uint64_t);
}

inline uint64_t resolveFastxChunkKmers(
    uint64_t totalItems,
    uint64_t reservedGpuBytes,
    uint64_t scratchBytesPerKmer = kFastxChunkBytesPerKmer
) {
    if (g_fastxChunkKmersUserSet) {
        g_fastxChunkKmersResolved = std::min(g_fastxChunkKmers, totalItems);
        g_fastxChunkKmersResolvedReserved = reservedGpuBytes;
        g_fastxChunkKmersResolvedScratchBpp = scratchBytesPerKmer;
        return g_fastxChunkKmersResolved;
    }
    if (g_fastxChunkKmersResolved != 0 && g_fastxChunkKmersResolvedReserved == reservedGpuBytes &&
        g_fastxChunkKmersResolvedScratchBpp == scratchBytesPerKmer) {
        return std::min(g_fastxChunkKmersResolved, totalItems);
    }

    // Target ~80% of total device memory for filter + sequence + encode/insert/query scratch.
    constexpr double kTargetDeviceFraction = 0.80;
    constexpr uint64_t kHeadroomBytes = 256ULL << 20;

    size_t freeBytes = 0;
    size_t totalBytes = 0;
    cudaDeviceMemInfo(freeBytes, totalBytes);

    size_t budget = 0;
    if (totalBytes > reservedGpuBytes + kHeadroomBytes) {
        const size_t targetUsed =
            static_cast<size_t>(static_cast<double>(totalBytes) * kTargetDeviceFraction);
        if (targetUsed > reservedGpuBytes + kHeadroomBytes) {
            budget = targetUsed - static_cast<size_t>(reservedGpuBytes) - kHeadroomBytes;
        }
    }
    if (freeBytes > reservedGpuBytes + kHeadroomBytes) {
        const size_t maxScratchFromFree =
            freeBytes - static_cast<size_t>(reservedGpuBytes) - kHeadroomBytes;
        budget = std::min(budget, maxScratchFromFree);
    }

    uint64_t chunk = budget / scratchBytesPerKmer;
    chunk = std::max(chunk, kFastxChunkFloorKmers);
    chunk = std::min(chunk, totalItems);

    g_fastxChunkKmersResolved = chunk;
    g_fastxChunkKmersResolvedReserved = reservedGpuBytes;
    g_fastxChunkKmersResolvedScratchBpp = scratchBytesPerKmer;

    const uint64_t scratchGiB = (chunk * scratchBytesPerKmer + (1ULL << 30) - 1) >> 30;
    const uint64_t totalGiB = (totalBytes + (1ULL << 30) - 1) >> 30;
    std::cerr << "FASTX chunk kmers: " << chunk << " (~" << scratchGiB
              << " GiB per-chunk scratch @ " << scratchBytesPerKmer << " B/kmer, target "
              << static_cast<int>(kTargetDeviceFraction * 100) << "% of " << totalGiB
              << " GiB device, " << (freeBytes >> 30) << " GiB free)\n";

    return chunk;
}

inline uint64_t fastxWorkloadChunkKmers(uint64_t totalItems = UINT64_MAX) {
    const uint64_t chunk = g_fastxChunkKmersResolved != 0 ? g_fastxChunkKmersResolved
                           : g_fastxChunkKmersUserSet     ? g_fastxChunkKmers
                                                          : (64ULL << 20);
    if (totalItems == UINT64_MAX) {
        return chunk;
    }
    return std::min(chunk, totalItems);
}

inline uint64_t effectiveFastxCpuNumRecords() {
    if (g_fastxCpuNumRecords != 0) {
        return g_fastxCpuNumRecords;
    }
    return std::max<uint64_t>(1, static_cast<uint64_t>(std::thread::hardware_concurrency()));
}

inline uint64_t g_fastxFilterBitsOverride = 0;
inline uint64_t g_fastxBitsPerItem = 16;

inline uint64_t resolveFastxFilterBits(uint64_t insertKmers) {
    if (g_fastxFilterBitsOverride != 0) {
        return g_fastxFilterBitsOverride;
    }
    return cuda::std::bit_ceil(std::max(insertKmers, uint64_t{1}) * g_fastxBitsPerItem);
}

inline uint64_t splitSequenceKmers(uint64_t sequenceLength, uint64_t numRecords, uint64_t k) {
    if (numRecords == 0) {
        return 0;
    }

    const uint64_t perRecordBases = sequenceLength / numRecords;
    uint64_t totalKmers = 0;
    uint64_t pos = 0;
    for (uint64_t r = 0; r < numRecords; ++r) {
        const uint64_t thisLen = (r == numRecords - 1) ? sequenceLength - pos : perRecordBases;
        totalKmers += thisLen >= k ? thisLen - k + 1 : 0;
        pos += thisLen;
    }
    return totalKmers;
}

inline std::string
writeGeneratedFasta(const std::vector<char>& sequence, uint64_t numRecords, const char* prefix) {
    if (numRecords == 0) {
        throw std::runtime_error("FASTA record count must be >= 1");
    }

    const auto tempDir = std::filesystem::temp_directory_path();
    const auto path = tempDir / (std::string(prefix) + "-" + std::to_string(getpid()) + "-" +
                                 std::to_string(numRecords) + ".fasta");

    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        throw std::runtime_error("failed to create temporary FASTA file");
    }

    const uint64_t perRecordBases = sequence.size() / numRecords;
    uint64_t pos = 0;
    for (uint64_t r = 0; r < numRecords; ++r) {
        const uint64_t thisLen = (r == numRecords - 1) ? sequence.size() - pos : perRecordBases;
        out << ">record_" << r << '\n';
        out.write(sequence.data() + pos, static_cast<std::streamsize>(thisLen));
        out.put('\n');
        pos += thisLen;
    }

    if (!out) {
        throw std::runtime_error("failed while writing temporary FASTA file");
    }

    return path.string();
}

class GPUTimer {
   public:
    GPUTimer() {
        CUSBF_CUDA_CALL(cudaEventCreate(&start_));
        CUSBF_CUDA_CALL(cudaEventCreate(&stop_));
    }

    ~GPUTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

    GPUTimer(const GPUTimer&) = delete;
    GPUTimer& operator=(const GPUTimer&) = delete;

    void start(cudaStream_t stream = {}) {
        CUSBF_CUDA_CALL(cudaEventRecord(start_, stream));
    }

    [[nodiscard]] double elapsed(cudaStream_t stream = {}) {
        CUSBF_CUDA_CALL(cudaEventRecord(stop_, stream));
        CUSBF_CUDA_CALL(cudaEventSynchronize(stop_));

        float milliseconds = 0.0f;
        CUSBF_CUDA_CALL(cudaEventElapsedTime(&milliseconds, start_, stop_));
        return static_cast<double>(milliseconds) / 1000.0;
    }

   private:
    cudaEvent_t start_{};
    cudaEvent_t stop_{};
};

class CPUTimer {
   public:
    void start() {
        start_ = std::chrono::high_resolution_clock::now();
    }

    [[nodiscard]] double elapsed() {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double>(end - start_).count();
    }

   private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_{};
};

// Concatenate all records in a FASTA/FASTQ file into a single sequence,
// Shared FASTX reader helper. Prefer benchmark_common::fastx_workload::load_fastx_sequence()
// when both host and device views are needed.
inline std::vector<char> readFastxConcatenated(
    const std::filesystem::path& path,
    char separator = cusbf::DnaAlphabet::separator
) {
    return fastx_workload::read_fastx_concatenated(path, separator);
}

/// Split @p sequence into @p numRecords FASTA records (contiguous partitions, no extra
/// separators). Used so CPU SuperBloom can parallelise over records while GPU/hash filters
/// still use the same flattened buffer and contiguous k-mer count.
inline std::string writeGeneratedFastaFromSequence(
    const std::vector<char>& sequence,
    uint64_t numRecords,
    const char* prefix
) {
    return writeGeneratedFasta(sequence, numRecords, prefix);
}

struct FastxBenchmarkCli {
    std::vector<char*> benchmark_argv;
};

inline FastxBenchmarkCli parseFastxBenchmarkCli(int argc, char** argv) {
    FastxBenchmarkCli cli;
    cli.benchmark_argv.reserve(argc);
    cli.benchmark_argv.push_back(argv[0]);

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        constexpr const char* fastxPrefix = "--insert-fastx=";
        if (std::strncmp(arg.c_str(), fastxPrefix, std::strlen(fastxPrefix)) == 0) {
            g_insertFastxPath = arg.substr(std::strlen(fastxPrefix));
            continue;
        }
        if (arg == "--insert-fastx") {
            if (i + 1 < argc) {
                ++i;
                g_insertFastxPath = argv[i];
            } else {
                std::cerr << "Missing value for --insert-fastx\n";
                std::exit(1);
            }
            continue;
        }

        constexpr const char* numRecordsPrefix = "--num-records=";
        if (std::strncmp(arg.c_str(), numRecordsPrefix, std::strlen(numRecordsPrefix)) == 0) {
            g_fastxCpuNumRecords = std::stoull(arg.substr(std::strlen(numRecordsPrefix)));
            continue;
        }
        if (arg == "--num-records") {
            if (i + 1 < argc) {
                ++i;
                g_fastxCpuNumRecords = std::stoull(argv[i]);
            } else {
                std::cerr << "Missing value for --num-records\n";
                std::exit(1);
            }
            continue;
        }

        constexpr const char* filterBitsPrefix = "--filter-bits=";
        if (std::strncmp(arg.c_str(), filterBitsPrefix, std::strlen(filterBitsPrefix)) == 0) {
            g_fastxFilterBitsOverride = std::stoull(arg.substr(std::strlen(filterBitsPrefix)));
            continue;
        }
        if (arg == "--filter-bits") {
            if (i + 1 < argc) {
                ++i;
                g_fastxFilterBitsOverride = std::stoull(argv[i]);
            } else {
                std::cerr << "Missing value for --filter-bits\n";
                std::exit(1);
            }
            continue;
        }

        constexpr const char* chunkKmersPrefix = "--fastx-chunk-kmers=";
        if (std::strncmp(arg.c_str(), chunkKmersPrefix, std::strlen(chunkKmersPrefix)) == 0) {
            g_fastxChunkKmers = std::stoull(arg.substr(std::strlen(chunkKmersPrefix)));
            g_fastxChunkKmersUserSet = true;
            continue;
        }
        if (arg == "--fastx-chunk-kmers") {
            if (i + 1 < argc) {
                ++i;
                g_fastxChunkKmers = std::stoull(argv[i]);
                g_fastxChunkKmersUserSet = true;
            } else {
                std::cerr << "Missing value for --fastx-chunk-kmers\n";
                std::exit(1);
            }
            continue;
        }

        cli.benchmark_argv.push_back(argv[i]);
    }

    if (g_fastxCpuNumRecords == 0) {
        g_fastxCpuNumRecords = effectiveFastxCpuNumRecords();
    }

    return cli;
}

inline void setCommonCounters(
    benchmark::State& state,
    uint64_t memoryBytes,
    uint64_t itemsProcessed,
    uint64_t sequenceBases
) {
    // itemsProcessed is k-mer count for throughput (GKmer/s = items_per_second / 1e9).
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations() * itemsProcessed));
    state.counters["sequence_bases"] = benchmark::Counter(static_cast<double>(sequenceBases));
    state.counters["memory_bytes"] = benchmark::Counter(
        static_cast<double>(memoryBytes), benchmark::Counter::kDefaults, benchmark::Counter::kIs1024
    );
    state.counters["bits_per_item"] = benchmark::Counter(
        static_cast<double>(memoryBytes * 8) / static_cast<double>(itemsProcessed),
        benchmark::Counter::kDefaults,
        benchmark::Counter::kIs1024
    );
    state.counters["fpr_percentage"] = 0.0;
    state.counters["false_positives"] = 0.0;
}

inline void
setFprCounters(benchmark::State& state, uint64_t falsePositives, uint64_t fprDenominator) {
    state.counters["false_positives"] = benchmark::Counter(static_cast<double>(falsePositives));
    state.counters["fpr_percentage"] = benchmark::Counter(
        100.0 * static_cast<double>(falsePositives) / static_cast<double>(fprDenominator)
    );
}

// Shared sizing/metrics for cross-filter GPU benchmarks.
namespace filter_benchmark {

constexpr double kLoadFactor = 0.95;
constexpr std::size_t kFprTestSize = 1'000'000;
constexpr std::size_t kBitsPerTag = 16;

constexpr uint64_t kDnaK = 31;
constexpr uint32_t kInsertSequenceSeed = 42;
constexpr uint32_t kFprQuerySequenceSeed = 1337;

inline std::size_t numItemsForTargetMemory(std::size_t targetMemoryBytes, std::size_t bitsPerSlot) {
    const std::size_t capacity = (targetMemoryBytes * 8) / bitsPerSlot;
    return static_cast<std::size_t>(static_cast<double>(capacity) * kLoadFactor);
}

template <typename T>
void generateKeysGpuRange(
    thrust::device_vector<T>& output,
    std::size_t count,
    T minValue,
    T maxValue,
    unsigned int seed = 99999
) {
    output.resize(count);
    thrust::transform(
        thrust::counting_iterator<std::size_t>(0),
        thrust::counting_iterator<std::size_t>(count),
        output.begin(),
        [=] __device__(std::size_t idx) {
            thrust::default_random_engine rng(seed);
            thrust::uniform_int_distribution<T> dist(minValue, maxValue);
            rng.discard(idx);
            return dist(rng);
        }
    );
}

inline void
setFilterBenchmarkCounters(benchmark::State& state, uint64_t memoryBytes, uint64_t numItems) {
    // Each item is one k-mer; items_per_second in CSV is k-mers/s (divide by 1e9 for GKmer/s).
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations() * numItems));
    state.counters["memory_bytes"] = benchmark::Counter(
        static_cast<double>(memoryBytes), benchmark::Counter::kDefaults, benchmark::Counter::kIs1024
    );
    state.counters["bits_per_item"] =
        benchmark::Counter(static_cast<double>(memoryBytes * 8) / static_cast<double>(numItems));
    state.counters["num_items"] = benchmark::Counter(static_cast<double>(numItems));
    state.counters["num_kmers"] = benchmark::Counter(static_cast<double>(numItems));
    state.counters["fpr_percentage"] = 0.0;
    state.counters["false_positives"] = 0.0;
}

inline void setFilterFprCounters(
    benchmark::State& state,
    uint64_t memoryBytes,
    uint64_t numItems,
    uint64_t falsePositives,
    uint64_t fprDenominator = kFprTestSize
) {
    setFilterBenchmarkCounters(state, memoryBytes, numItems);
    setFprCounters(state, falsePositives, fprDenominator);
}

}  // namespace filter_benchmark

inline uint64_t fastxFilterBitsBudget() {
    if (!g_fastxInsertWorkload) {
        return 0;
    }
    return resolveFastxFilterBits(g_fastxInsertWorkload->insert_kmers);
}

inline uint64_t fastxBenchKmersAtLoadFactor() {
    const uint64_t bits = fastxFilterBitsBudget();
    if (bits == 0 || !g_fastxInsertWorkload) {
        return 0;
    }
    return std::min(
        g_fastxInsertWorkload->insert_kmers,
        static_cast<uint64_t>(
            filter_benchmark::numItemsForTargetMemory(bits / 8, filter_benchmark::kBitsPerTag)
        )
    );
}

inline uint64_t fastxBenchSequenceLength(uint64_t benchKmers, uint64_t k = 31) {
    if (!g_fastxInsertWorkload || benchKmers == 0) {
        return 0;
    }
    const uint64_t maxLen = g_fastxInsertWorkload->host_insert_sequence.size();
    const uint64_t needed = benchKmers + k - 1;
    return std::min(maxLen, needed);
}

/// Shared FASTX throughput sizing for filter-comparison (target load factor).
struct FastxThroughputConfig {
    uint64_t genome_kmers = 0;
    uint64_t bench_kmers = 0;
    uint64_t bench_seq_len = 0;
    uint64_t filter_bits = 0;
};

inline FastxThroughputConfig resolveFastxThroughputConfig(uint64_t k = 31) {
    FastxThroughputConfig cfg;
    if (!g_fastxInsertWorkload) {
        return cfg;
    }
    cfg.genome_kmers = g_fastxInsertWorkload->insert_kmers;
    cfg.filter_bits = resolveFastxFilterBits(cfg.genome_kmers);
    cfg.bench_kmers = fastxBenchKmersAtLoadFactor();
    cfg.bench_seq_len = fastxBenchSequenceLength(cfg.bench_kmers, k);
    return cfg;
}

inline uint64_t fastxBenchCapacityItems() {
    const uint64_t benchKmers = fastxBenchKmersAtLoadFactor();
    if (benchKmers == 0) {
        return 0;
    }
    return static_cast<uint64_t>(
        std::ceil(static_cast<double>(benchKmers) / filter_benchmark::kLoadFactor)
    );
}

inline void setBenchmarkCounters(
    benchmark::State& state,
    uint64_t memoryBytes,
    uint64_t sequenceLength,
    uint64_t numKmers
) {
    setCommonCounters(state, memoryBytes, numKmers, sequenceLength);
    state.counters["num_kmers"] = benchmark::Counter(static_cast<double>(numKmers));
}

namespace detail {

__device__ __forceinline__ char randomDnaBase(uint64_t idx, uint32_t seed) {
    thrust::default_random_engine rng(seed);
    thrust::uniform_int_distribution<uint32_t> dist(0, 3);
    rng.discard(idx);
    constexpr char bases[] = {'A', 'C', 'G', 'T'};
    return bases[dist(rng)];
}

__device__ __forceinline__ char randomProteinSymbol(uint64_t idx, uint32_t seed) {
    thrust::default_random_engine rng(seed);
    thrust::uniform_int_distribution<uint32_t> dist(0, 19);
    rng.discard(idx);
    constexpr char symbols[] = {'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
                                'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'};
    return symbols[dist(rng)];
}

}  // namespace detail

inline void
gpuGenerateDna(thrust::device_vector<char>& d_seq, uint64_t length, uint32_t seed = 42) {
    d_seq.resize(length);
    thrust::transform(
        thrust::counting_iterator<uint64_t>(0),
        thrust::counting_iterator<uint64_t>(length),
        d_seq.begin(),
        [seed] __device__(uint64_t idx) { return detail::randomDnaBase(idx, seed); }
    );
}

inline void
gpuGenerateProtein(thrust::device_vector<char>& d_seq, uint64_t length, uint32_t seed = 42) {
    d_seq.resize(length);
    thrust::transform(
        thrust::counting_iterator<uint64_t>(0),
        thrust::counting_iterator<uint64_t>(length),
        d_seq.begin(),
        [seed] __device__(uint64_t idx) { return detail::randomProteinSymbol(idx, seed); }
    );
}

inline void gpuGeneratePackedKmers(
    thrust::device_vector<uint64_t>& d_kmers,
    uint64_t count,
    uint32_t seed = 42
) {
    d_kmers.resize(count);
    thrust::transform(
        thrust::counting_iterator<uint64_t>(0),
        thrust::counting_iterator<uint64_t>(count),
        d_kmers.begin(),
        [seed] __device__(uint64_t idx) {
            thrust::default_random_engine rng(seed);
            thrust::uniform_int_distribution<uint64_t> dist(0, UINT64_MAX);
            rng.discard(idx);
            return dist(rng);
        }
    );
}

template <uint64_t K, typename Alphabet = cusbf::DnaAlphabet>
__global__ void encodePackedKmersKernel(
    const char* sequence,
    uint64_t kmerStart,
    uint64_t numKmers,
    uint64_t* output
) {
    constexpr uint64_t symbolBits = cuda::std::bit_width(Alphabet::symbolCount - 1);
    constexpr uint64_t symbolMask = (uint64_t{1} << symbolBits) - 1;
    const uint64_t idx = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= numKmers) {
        return;
    }

    const uint64_t kmerIndex = kmerStart + idx;
    uint64_t packed = 0;
    for (uint64_t i = 0; i < K; ++i) {
        const uint8_t encoded =
            Alphabet::encode(sequence + (kmerIndex + i) * Alphabet::symbolWidth);
        packed = (packed << symbolBits) | (encoded & symbolMask);
    }
    output[idx] = packed;
}

template <uint64_t K, typename Alphabet = cusbf::DnaAlphabet>
inline void gpuEncodePackedKmers(
    const char* d_sequence,
    uint64_t sequenceLength,
    uint64_t* d_output,
    cudaStream_t stream = {},
    uint64_t kmerStart = 0,
    uint64_t numKmers = 0
) {
    const uint64_t symbols = sequenceLength / Alphabet::symbolWidth;
    const uint64_t totalKmers = symbols >= K ? symbols - K + 1 : 0;
    if (kmerStart >= totalKmers) {
        return;
    }
    const uint64_t available = totalKmers - kmerStart;
    const uint64_t encodeCount = numKmers == 0 ? available : std::min(numKmers, available);
    if (encodeCount == 0) {
        return;
    }
    constexpr uint64_t blockSize = 256;
    const uint64_t gridSize = cuda::ceil_div(encodeCount, blockSize);
    encodePackedKmersKernel<K, Alphabet>
        <<<gridSize, blockSize, 0, stream>>>(d_sequence, kmerStart, encodeCount, d_output);
}

inline void uploadFastxSequenceToDevice(FastxInsertWorkload& workload) {
    if (workload.gpuPrepareLevel >= FastxGpuPrepareKind::SequenceOnDevice) {
        return;
    }
    workload.d_insert_sequence.resize(workload.host_insert_sequence.size());
    if (!workload.host_insert_sequence.empty()) {
        CUSBF_CUDA_CALL(cudaMemcpy(
            thrust::raw_pointer_cast(workload.d_insert_sequence.data()),
            workload.host_insert_sequence.data(),
            workload.host_insert_sequence.size(),
            cudaMemcpyHostToDevice
        ));
    }
    workload.gpuPrepareLevel = FastxGpuPrepareKind::SequenceOnDevice;
}

template <uint64_t K = 31>
inline void encodeFastxPackedKmersOnDevice(FastxInsertWorkload& workload) {
    if (workload.gpuPrepareLevel >= FastxGpuPrepareKind::PackedKmers) {
        return;
    }
    uploadFastxSequenceToDevice(workload);
    workload.d_insert_packed_kmers.resize(workload.insert_kmers);
    if (workload.insert_kmers != 0) {
        fastx_workload::encode_packed_kmers<K, cusbf::DnaAlphabet>(
            thrust::raw_pointer_cast(workload.d_insert_sequence.data()),
            workload.host_insert_sequence.size(),
            thrust::raw_pointer_cast(workload.d_insert_packed_kmers.data())
        );
        CUSBF_CUDA_CALL(cudaDeviceSynchronize());
    }
    workload.gpuPrepareLevel = FastxGpuPrepareKind::PackedKmers;
}

template <uint64_t K = 31>
inline void prepareFastxInsertWorkload(
    char separator = cusbf::DnaAlphabet::separator,
    FastxGpuPrepareKind gpuPrepare = FastxGpuPrepareKind::PackedKmers
) {
    if (g_fastxInsertWorkload) {
        if (gpuPrepare == FastxGpuPrepareKind::SequenceOnDevice) {
            uploadFastxSequenceToDevice(*g_fastxInsertWorkload);
        } else if (gpuPrepare == FastxGpuPrepareKind::PackedKmers) {
            encodeFastxPackedKmersOnDevice<K>(*g_fastxInsertWorkload);
        }
        return;
    }
    if (g_insertFastxPath.empty()) {
        std::cerr << "Error: --insert-fastx is required\n";
        std::exit(1);
    }

    auto prepared =
        fastx_workload::load_fastx_sequence<K, cusbf::DnaAlphabet>(g_insertFastxPath, separator);
    if (prepared.host_sequence.empty()) {
        std::cerr << "Error: FASTX file is empty or contains no sequences\n";
        std::exit(1);
    }

    auto workload = std::make_unique<FastxInsertWorkload>();
    workload->host_insert_sequence = std::move(prepared.host_sequence);
    workload->insert_kmers = prepared.kmers;
    workload->gpuPrepareLevel = FastxGpuPrepareKind::HostOnly;

    if (gpuPrepare == FastxGpuPrepareKind::SequenceOnDevice) {
        uploadFastxSequenceToDevice(*workload);
    } else if (gpuPrepare == FastxGpuPrepareKind::PackedKmers) {
        encodeFastxPackedKmersOnDevice<K>(*workload);
    }

    g_fastxInsertWorkload = std::move(workload);
}

inline void ensureFastxCpuInsertFasta(const char* prefix = "bloom-filter-comparison-cpu-insert") {
    prepareFastxInsertWorkload<31>(cusbf::DnaAlphabet::separator, FastxGpuPrepareKind::HostOnly);
    auto& workload = *g_fastxInsertWorkload;
    if (!workload.cpu_insert_fastx_path.empty()) {
        return;
    }
    workload.cpu_insert_fastx_path = writeGeneratedFastaFromSequence(
        workload.host_insert_sequence, effectiveFastxCpuNumRecords(), prefix
    );
}

inline void clearFastxInsertWorkload() {
    if (g_fastxInsertWorkload && !g_fastxInsertWorkload->cpu_insert_fastx_path.empty()) {
        std::filesystem::remove(g_fastxInsertWorkload->cpu_insert_fastx_path);
    }
    g_fastxInsertWorkload.reset();
}

namespace filter_benchmark {

struct DnaKmerWorkload {
    uint64_t numItems{};
    uint64_t insertSequenceLength{};
    uint64_t fprQuerySequenceLength{};
    uint64_t fprQueryKmers{};
    thrust::device_vector<char> insertSequence;
    thrust::device_vector<char> fprQuerySequence;
    thrust::device_vector<uint64_t> insertPackedKmers;
    thrust::device_vector<uint64_t> fprQueryPackedKmers;

    void initFromItemCount(uint64_t itemCount) {
        numItems = itemCount;
        insertSequenceLength = numItems + kDnaK - 1;
        fprQuerySequenceLength = kFprTestSize + kDnaK - 1;
        fprQueryKmers = fprQuerySequenceLength - kDnaK + 1;
    }

    void prepareSequences() {
        gpuGenerateDna(insertSequence, insertSequenceLength, kInsertSequenceSeed);
        gpuGenerateDna(fprQuerySequence, fprQuerySequenceLength, kFprQuerySequenceSeed);
        CUSBF_CUDA_CALL(cudaDeviceSynchronize());
    }

    void encodeInsert() {
        gpuGenerateDna(insertSequence, insertSequenceLength, kInsertSequenceSeed);
        insertPackedKmers.resize(numItems);
        gpuEncodePackedKmers<kDnaK, cusbf::DnaAlphabet>(
            thrust::raw_pointer_cast(insertSequence.data()),
            insertSequenceLength,
            thrust::raw_pointer_cast(insertPackedKmers.data())
        );
        CUSBF_CUDA_CALL(cudaDeviceSynchronize());
    }

    void encodeFprQuery() {
        gpuGenerateDna(fprQuerySequence, fprQuerySequenceLength, kFprQuerySequenceSeed);
        fprQueryPackedKmers.resize(fprQueryKmers);
        gpuEncodePackedKmers<kDnaK, cusbf::DnaAlphabet>(
            thrust::raw_pointer_cast(fprQuerySequence.data()),
            fprQuerySequenceLength,
            thrust::raw_pointer_cast(fprQueryPackedKmers.data())
        );
        CUSBF_CUDA_CALL(cudaDeviceSynchronize());
    }

    void initFromTargetMemory(uint64_t targetMemory) {
        initFromItemCount(numItemsForTargetMemory(targetMemory, kBitsPerTag));
        prepareSequences();
    }
};

}  // namespace filter_benchmark

template <uint64_t K, typename Alphabet = cusbf::DnaAlphabet>
struct BenchmarkData {
    uint64_t sequenceLength{};
    uint64_t numKmers{};

    thrust::device_vector<char> d_throughputSequence;
    thrust::device_vector<uint64_t> d_throughputPackedKmers;

    thrust::device_vector<char> d_fprInsertSequence;
    thrust::device_vector<uint64_t> d_fprInsertPackedKmers;
    thrust::device_vector<char> d_zeroOverlapSequence;
    thrust::device_vector<uint64_t> d_zeroOverlapPackedKmers;
    bool fprDataReady = false;

    void generateThroughputData() {
        if constexpr (std::is_same_v<Alphabet, cusbf::ProteinAlphabet>) {
            gpuGenerateProtein(d_throughputSequence, sequenceLength, 42);
        } else {
            gpuGenerateDna(d_throughputSequence, sequenceLength, 42);
        }
        numKmers = sequenceLength >= K ? sequenceLength - K + 1 : 0;
        d_throughputPackedKmers.resize(numKmers);
    }

    void ensureFprData() const {
        if (fprDataReady) {
            return;
        }
        const_cast<BenchmarkData*>(this)->generateFprData();
    }

   private:
    void generateFprData() {
        if constexpr (std::is_same_v<Alphabet, cusbf::ProteinAlphabet>) {
            gpuGenerateProtein(d_fprInsertSequence, sequenceLength, 7);
            gpuGenerateProtein(d_zeroOverlapSequence, sequenceLength, 1337);
        } else {
            gpuGenerateDna(d_fprInsertSequence, sequenceLength, 7);
            gpuGenerateDna(d_zeroOverlapSequence, sequenceLength, 1337);
        }
        const uint64_t fprNumKmers = sequenceLength >= K ? sequenceLength - K + 1 : 0;
        d_fprInsertPackedKmers.resize(fprNumKmers);
        gpuEncodePackedKmers<K, Alphabet>(
            thrust::raw_pointer_cast(d_fprInsertSequence.data()),
            sequenceLength,
            thrust::raw_pointer_cast(d_fprInsertPackedKmers.data())
        );

        d_zeroOverlapPackedKmers.resize(fprNumKmers);
        gpuEncodePackedKmers<K, Alphabet>(
            thrust::raw_pointer_cast(d_zeroOverlapSequence.data()),
            sequenceLength,
            thrust::raw_pointer_cast(d_zeroOverlapPackedKmers.data())
        );

        CUSBF_CUDA_CALL(cudaDeviceSynchronize());

        fprDataReady = true;
    }
};

template <uint64_t K, typename Alphabet = cusbf::DnaAlphabet>
BenchmarkData<K, Alphabet>& getBenchmarkData(uint64_t length) {
    static std::unordered_map<uint64_t, BenchmarkData<K, Alphabet>> cache;

    auto it = cache.find(length);
    if (it != cache.end()) {
        return it->second;
    }

    cache.clear();

    BenchmarkData<K, Alphabet> data;
    data.sequenceLength = length;
    data.generateThroughputData();
    CUSBF_CUDA_CALL(cudaDeviceSynchronize());

    return cache.emplace(length, std::move(data)).first->second;
}

template <typename Config>
class CuSbfFixtureBase : public benchmark::Fixture {
   public:
    static constexpr uint64_t k = Config::k;

    void setupCommon(const benchmark::State& state) {
        targetMemoryBytes = static_cast<uint64_t>(state.range(0));
        numItems = filter_benchmark::numItemsForTargetMemory(
            targetMemoryBytes, filter_benchmark::kBitsPerTag
        );

        const uint64_t requestedFilterBits =
            cuda::std::bit_ceil(std::max(targetMemoryBytes, uint64_t{1}) * 8);
        filter = std::make_unique<cusbf::filter<Config>>(requestedFilterBits);
        filterMemory = filter->filter_bits() / 8;

        if constexpr (std::is_same_v<typename Config::Alphabet, cusbf::DnaAlphabet>) {
            dnaWorkload.initFromTargetMemory(targetMemoryBytes);
            d_insertSequence = std::move(dnaWorkload.insertSequence);
            d_fprQuerySequence = std::move(dnaWorkload.fprQuerySequence);
            insertSequenceLength = dnaWorkload.insertSequenceLength;
            fprQuerySequenceLength = dnaWorkload.fprQuerySequenceLength;
            fprQueryKmers = dnaWorkload.fprQueryKmers;
            numItems = dnaWorkload.numItems;
        } else if constexpr (std::is_same_v<typename Config::Alphabet, cusbf::ProteinAlphabet>) {
            insertSequenceLength = numItems + Config::k - 1;
            fprQuerySequenceLength = filter_benchmark::kFprTestSize + Config::k - 1;
            fprQueryKmers = fprQuerySequenceLength - Config::k + 1;
            d_insertSequence.resize(insertSequenceLength);
            d_fprQuerySequence.resize(fprQuerySequenceLength);
            gpuGenerateProtein(
                d_insertSequence, insertSequenceLength, filter_benchmark::kInsertSequenceSeed
            );
            gpuGenerateProtein(
                d_fprQuerySequence, fprQuerySequenceLength, filter_benchmark::kFprQuerySequenceSeed
            );
            CUSBF_CUDA_CALL(cudaDeviceSynchronize());
        } else {
            static_assert(false, "unsupported alphabet");
        }

        numKmers = numItems;
        numSmers = insertSequenceLength - Config::s + 1;
        d_output.resize(fprQueryKmers);
    }

    void tearDownCommon() {
        filter.reset();
        d_insertSequence.clear();
        d_insertSequence.shrink_to_fit();
        d_fprQuerySequence.clear();
        d_fprQuerySequence.shrink_to_fit();
        d_output.clear();
        d_output.shrink_to_fit();
    }

    void setCounters(benchmark::State& state) const {
        filter_benchmark::setFilterBenchmarkCounters(state, filterMemory, numItems);
        state.counters["s"] = benchmark::Counter(static_cast<double>(Config::s));
        state.counters["hashes"] = benchmark::Counter(static_cast<double>(Config::hashCount));
    }

    uint64_t targetMemoryBytes{};
    uint64_t numItems{};
    uint64_t insertSequenceLength{};
    uint64_t fprQuerySequenceLength{};
    uint64_t fprQueryKmers{};
    uint64_t numKmers{};
    uint64_t numSmers{};
    uint64_t filterMemory{};
    filter_benchmark::DnaKmerWorkload dnaWorkload;
    thrust::device_vector<char> d_insertSequence;
    thrust::device_vector<char> d_fprQuerySequence;
    thrust::device_vector<uint8_t> d_output;
    std::unique_ptr<cusbf::filter<Config>> filter;
    GPUTimer timer;
};

template <typename Config>
class CuSbfConfigFixture : public CuSbfFixtureBase<Config> {
    using benchmark::Fixture::SetUp;
    using benchmark::Fixture::TearDown;

   public:
    void SetUp(const benchmark::State& state) override {
        this->setupCommon(state);
    }

    void TearDown(const benchmark::State&) override {
        this->tearDownCommon();
    }
};

template <typename Fixture>
void runCuSbfInsert(Fixture& fixture, benchmark::State& state) {
    const cusbf::device_span<const char> insertSpan{
        thrust::raw_pointer_cast(fixture.d_insertSequence.data()),
        fixture.insertSequenceLength,
    };
    for (auto _ : state) {
        CUSBF_UNWRAP(fixture.filter->clear());
        CUSBF_CUDA_CALL(cudaDeviceSynchronize());

        fixture.timer.start();
        benchmark::DoNotOptimize(fixture.filter->insert_sequence_async(insertSpan));
        const double elapsed = fixture.timer.elapsed();
        state.SetIterationTime(elapsed);
    }
    fixture.setCounters(state);
}

template <typename Fixture>
void runCuSbfQuery(Fixture& fixture, benchmark::State& state) {
    const cusbf::device_span<const char> insertSpan{
        thrust::raw_pointer_cast(fixture.d_insertSequence.data()),
        fixture.insertSequenceLength,
    };
    CUSBF_UNWRAP(fixture.filter->clear());
    benchmark::DoNotOptimize(fixture.filter->insert_sequence_async(insertSpan));
    CUSBF_CUDA_CALL(cudaDeviceSynchronize());

    thrust::device_vector<uint8_t> queryOutput(fixture.numItems);
    const cusbf::device_span<const char> querySpan{
        thrust::raw_pointer_cast(fixture.d_insertSequence.data()),
        fixture.insertSequenceLength,
    };

    for (auto _ : state) {
        fixture.timer.start();
        cusbf::require_void(fixture.filter->contains_sequence_async(
            querySpan,
            cusbf::device_span<uint8_t>{
                thrust::raw_pointer_cast(queryOutput.data()),
                queryOutput.size(),
            }
        ));
        const double elapsed = fixture.timer.elapsed();
        state.SetIterationTime(elapsed);
        benchmark::DoNotOptimize(thrust::raw_pointer_cast(queryOutput.data()));
    }
    fixture.setCounters(state);
}

template <typename Fixture>
void runCuSbfFpr(Fixture& fixture, benchmark::State& state) {
    const cusbf::device_span<const char> insertSpan{
        thrust::raw_pointer_cast(fixture.d_insertSequence.data()),
        fixture.insertSequenceLength,
    };
    const cusbf::device_span<const char> querySpan{
        thrust::raw_pointer_cast(fixture.d_fprQuerySequence.data()),
        fixture.fprQuerySequenceLength,
    };

    CUSBF_UNWRAP(fixture.filter->clear());
    benchmark::DoNotOptimize(fixture.filter->insert_sequence_async(insertSpan));
    CUSBF_CUDA_CALL(cudaDeviceSynchronize());

    for (auto _ : state) {
        fixture.timer.start();
        cusbf::require_void(fixture.filter->contains_sequence_async(
            querySpan,
            cusbf::device_span<uint8_t>{
                thrust::raw_pointer_cast(fixture.d_output.data()),
                fixture.d_output.size(),
            }
        ));
        const double elapsed = fixture.timer.elapsed();
        state.SetIterationTime(elapsed);
        benchmark::DoNotOptimize(thrust::raw_pointer_cast(fixture.d_output.data()));
    }

    const uint64_t falsePositives = static_cast<uint64_t>(
        thrust::count(fixture.d_output.begin(), fixture.d_output.end(), uint8_t{1})
    );
    fixture.setCounters(state);
    filter_benchmark::setFilterFprCounters(
        state, fixture.filterMemory, fixture.numItems, falsePositives, fixture.fprQueryKmers
    );
}

/// Compute the bit_vector_size_exponent and block_size_exponent for a CPU
/// SuperBloom filter that gives at least 16 bits per item while satisfying
/// the Rust library's SHARD_COUNT constraint (nb_blocks >= 1024).
inline void cpuFilterExponents(uint64_t numKmers, uint8_t& bitExp, uint8_t& blockExp) {
    constexpr uint64_t kBitsPerItem = filter_benchmark::kBitsPerTag;
    uint64_t filter_bits = cuda::std::bit_ceil(numKmers * kBitsPerItem);
    constexpr uint64_t kMinFilterBits = uint64_t{1} << 19;
    filter_bits = std::max(filter_bits, kMinFilterBits);
    bitExp = static_cast<uint8_t>(cuda::std::bit_width(filter_bits) - 1);
    blockExp = 9;
}

// CPU SuperBloom FASTX fixture (filter-comparison).

template <typename Config>
class SuperBloomCpuFastxFixture : public benchmark::Fixture {
   public:
    using benchmark::Fixture::SetUp;
    using benchmark::Fixture::TearDown;

    static constexpr uint64_t k = Config::k;
    static constexpr uint64_t m = Config::m;
    static constexpr uint64_t s = Config::s;
    static constexpr uint64_t hashCount = Config::hashCount;

    void SetUp(const benchmark::State&) override {
        initFailed_ = false;
        prepareFastxInsertWorkload<Config::k>(
            cusbf::DnaAlphabet::separator, FastxGpuPrepareKind::HostOnly
        );

        const FastxThroughputConfig cfg = resolveFastxThroughputConfig(Config::k);
        numKmers = cfg.bench_kmers;
        sequenceLength = cfg.bench_seq_len;
        filter_bits = cfg.filter_bits;

        auto& workload = *g_fastxInsertWorkload;
        if (!workload.cpu_insert_fastx_path.empty()) {
            std::filesystem::remove(workload.cpu_insert_fastx_path);
            workload.cpu_insert_fastx_path.clear();
        }
        const std::vector<char> benchSequence(
            workload.host_insert_sequence.begin(),
            workload.host_insert_sequence.begin() + static_cast<std::ptrdiff_t>(sequenceLength)
        );
        workload.cpu_insert_fastx_path = writeGeneratedFastaFromSequence(
            benchSequence, effectiveFastxCpuNumRecords(), "bloom-filter-comparison-cpu-insert"
        );

        cpuFilterExponents(fastxBenchCapacityItems(), bitVectorSizeExp, blockSizeExp);

        unsigned n = std::thread::hardware_concurrency();
        threadCount_ = n > 0 ? static_cast<size_t>(n) : 0;

        recreateFilter();
        if (!handle_) {
            initFailed_ = true;
            return;
        }

        filterMemory = superbloom_filter_bits(handle_) / 8;
    }

    void TearDown(const benchmark::State&) override {
        if (handle_) {
            superbloom_destroy(handle_);
        }
        handle_ = nullptr;
        if (g_fastxInsertWorkload && !g_fastxInsertWorkload->cpu_insert_fastx_path.empty()) {
            std::filesystem::remove(g_fastxInsertWorkload->cpu_insert_fastx_path);
            g_fastxInsertWorkload->cpu_insert_fastx_path.clear();
        }
    }

    void setCounters(benchmark::State& state) const {
        filter_benchmark::setFilterBenchmarkCounters(state, filterMemory, numKmers);
        state.counters["s"] = static_cast<double>(Config::s);
        state.counters["filter_bits"] = benchmark::Counter(static_cast<double>(filter_bits));
    }

    void recreateFilter() {
        superbloom_destroy(handle_);
        handle_ = superbloom_create(
            Config::k, Config::m, Config::s, Config::hashCount, bitVectorSizeExp, blockSizeExp
        );
        if (handle_ && threadCount_ > 0) {
            superbloom_set_threads(handle_, threadCount_);
        }
    }

    uint64_t sequenceLength{};
    uint64_t numKmers{};
    uint64_t filter_bits{};
    uint64_t filterMemory{};
    uint8_t bitVectorSizeExp{};
    uint8_t blockSizeExp{};
    void* handle_{};
    bool initFailed_ = false;
    size_t threadCount_ = 0;
    CPUTimer timer;
};

template <typename Fixture>
void runSuperBloomCpuFastxInsert(Fixture& fixture, benchmark::State& state) {
    ensureFastxCpuInsertFasta();

    for (auto _ : state) {
        fixture.recreateFilter();
        if (!fixture.handle_) {
            state.SkipWithError("superbloom_create failed during iteration");
            return;
        }

        fixture.timer.start();
        const int64_t added = superbloom_insert_fastx_path(
            fixture.handle_, g_fastxInsertWorkload->cpu_insert_fastx_path.c_str()
        );
        if (added < 0) {
            state.SkipWithError("superbloom insert failed");
            return;
        }
        state.SetIterationTime(fixture.timer.elapsed());
        benchmark::DoNotOptimize(added);
    }
    fixture.setCounters(state);
}

template <typename Fixture>
void runSuperBloomCpuFastxQuery(Fixture& fixture, benchmark::State& state) {
    ensureFastxCpuInsertFasta();

    fixture.recreateFilter();
    if (!fixture.handle_) {
        state.SkipWithError("superbloom_create failed");
        return;
    }
    if (superbloom_insert_fastx_path(
            fixture.handle_, g_fastxInsertWorkload->cpu_insert_fastx_path.c_str()
        ) < 0) {
        state.SkipWithError("superbloom insert failed");
        return;
    }
    superbloom_freeze(fixture.handle_);

    for (auto _ : state) {
        fixture.timer.start();
        const int64_t positives = superbloom_query_fastx_path(
            fixture.handle_, g_fastxInsertWorkload->cpu_insert_fastx_path.c_str()
        );
        if (positives < 0) {
            state.SkipWithError("superbloom query failed");
            return;
        }
        state.SetIterationTime(fixture.timer.elapsed());
        benchmark::DoNotOptimize(positives);
    }
    fixture.setCounters(state);
}

#define BENCHMARK_DEFINE_SUPERBLOOM_CPU_FASTX_INSERT(FixtureName)       \
    BENCHMARK_DEFINE_F(FixtureName, Insert)(benchmark::State & state) { \
        benchmark_common::runSuperBloomCpuFastxInsert(*this, state);    \
    };

#define BENCHMARK_DEFINE_SUPERBLOOM_CPU_FASTX_QUERY(FixtureName)       \
    BENCHMARK_DEFINE_F(FixtureName, Query)(benchmark::State & state) { \
        benchmark_common::runSuperBloomCpuFastxQuery(*this, state);    \
    };

#define BENCHMARK_DEFINE_SUPERBLOOM_CPU_FASTX_ALL(FixtureName) \
    BENCHMARK_DEFINE_SUPERBLOOM_CPU_FASTX_INSERT(FixtureName)  \
    BENCHMARK_DEFINE_SUPERBLOOM_CPU_FASTX_QUERY(FixtureName)

#define BENCHMARK_REGISTER_SUPERBLOOM_CPU_FASTX_ALL(FixtureName) \
    REGISTER_BENCHMARK_THROUGHPUT_FASTX(FixtureName, Insert);    \
    REGISTER_BENCHMARK_THROUGHPUT_FASTX(FixtureName, Query);

template <typename Config>
class SuperBloomCpuFixture : public benchmark::Fixture {
   public:
    using benchmark::Fixture::SetUp;
    using benchmark::Fixture::TearDown;

    static constexpr uint64_t k = Config::k;
    static constexpr uint64_t m = Config::m;
    static constexpr uint64_t s = Config::s;
    static constexpr uint64_t hashCount = Config::hashCount;
    using Alphabet = typename Config::Alphabet;
    using ConfigType = Config;

    void SetUp(const benchmark::State& state) override {
        initFailed_ = false;
        sequenceLength = static_cast<uint64_t>(state.range(0));
        benchData = &getBenchmarkData<Config::k, Alphabet>(sequenceLength);
        numKmers = g_cpuFastxParallelizeRecords
                       ? splitSequenceKmers(sequenceLength, g_cpuFastxNumRecords, Config::k)
                       : benchData->numKmers;

        cpuFilterExponents(numKmers, bitVectorSizeExp, blockSizeExp);

        unsigned n = std::thread::hardware_concurrency();
        threadCount_ = n > 0 ? static_cast<size_t>(n) : 0;

        recreateFilter();
        if (!handle_) {
            initFailed_ = true;
            return;
        }

        filter_bits = uint64_t{1} << bitVectorSizeExp;
        h_output.resize(numKmers);
    }

    void TearDown(const benchmark::State&) override {
        if (handle_) superbloom_destroy(handle_);
        handle_ = nullptr;
        benchData = nullptr;
        h_output.clear();
        h_sequence.clear();
        if (!throughputFastxPath.empty()) {
            std::filesystem::remove(throughputFastxPath);
            throughputFastxPath.clear();
        }
        if (!fprInsertFastxPath.empty()) {
            std::filesystem::remove(fprInsertFastxPath);
            fprInsertFastxPath.clear();
        }
        if (!fprQueryFastxPath.empty()) {
            std::filesystem::remove(fprQueryFastxPath);
            fprQueryFastxPath.clear();
        }
    }

    void setCounters(benchmark::State& state) const {
        setBenchmarkCounters(state, filter_bits / 8, sequenceLength, numKmers);
        state.counters["s"] = static_cast<double>(Config::s);
        state.counters["hashes"] = static_cast<double>(Config::hashCount);
    }

    void ensureHostSequence() {
        if (!h_sequence.empty() || sequenceLength == 0) return;
        h_sequence.resize(sequenceLength);
        CUSBF_CUDA_CALL(cudaMemcpy(
            h_sequence.data(),
            thrust::raw_pointer_cast(benchData->d_throughputSequence.data()),
            sequenceLength,
            cudaMemcpyDeviceToHost
        ));
    }

    void recreateFilter() {
        superbloom_destroy(handle_);
        handle_ = superbloom_create(
            Config::k, Config::m, Config::s, Config::hashCount, bitVectorSizeExp, blockSizeExp
        );
        if (handle_ && threadCount_ > 0) {
            superbloom_set_threads(handle_, threadCount_);
        }
    }

    void ensureThroughputFastxPath() {
        if (!g_cpuFastxParallelizeRecords || !throughputFastxPath.empty()) {
            return;
        }
        ensureHostSequence();
        throughputFastxPath = writeGeneratedFasta(
            h_sequence, g_cpuFastxNumRecords, "bloom-cpu-filter-comparison-throughput"
        );
    }

    void ensureFprFastxPaths() {
        if (!g_cpuFastxParallelizeRecords ||
            (!fprInsertFastxPath.empty() && !fprQueryFastxPath.empty())) {
            return;
        }

        benchData->ensureFprData();

        std::vector<char> fprHostSeq(sequenceLength);
        CUSBF_CUDA_CALL(cudaMemcpy(
            fprHostSeq.data(),
            thrust::raw_pointer_cast(benchData->d_fprInsertSequence.data()),
            sequenceLength,
            cudaMemcpyDeviceToHost
        ));

        std::vector<char> zeroHostSeq(sequenceLength);
        CUSBF_CUDA_CALL(cudaMemcpy(
            zeroHostSeq.data(),
            thrust::raw_pointer_cast(benchData->d_zeroOverlapSequence.data()),
            sequenceLength,
            cudaMemcpyDeviceToHost
        ));

        fprInsertFastxPath = writeGeneratedFasta(
            fprHostSeq, g_cpuFastxNumRecords, "bloom-cpu-filter-comparison-fpr-insert"
        );
        fprQueryFastxPath = writeGeneratedFasta(
            zeroHostSeq, g_cpuFastxNumRecords, "bloom-cpu-filter-comparison-fpr-query"
        );
    }

    uint64_t sequenceLength{};
    uint64_t numKmers{};
    uint64_t filter_bits{};
    uint8_t bitVectorSizeExp{};
    uint8_t blockSizeExp{};
    BenchmarkData<Config::k, Alphabet>* benchData{};
    std::vector<uint8_t> h_output;
    std::vector<char> h_sequence;
    std::string throughputFastxPath;
    std::string fprInsertFastxPath;
    std::string fprQueryFastxPath;
    void* handle_{};
    bool initFailed_ = false;
    size_t threadCount_ = 0;
    CPUTimer timer;
};

template <typename Fixture>
void runSuperBloomCpuInsert(Fixture& fixture, benchmark::State& state) {
    if (g_cpuFastxParallelizeRecords) {
        fixture.ensureThroughputFastxPath();
    } else {
        fixture.ensureHostSequence();
    }

    for (auto _ : state) {
        fixture.recreateFilter();
        if (!fixture.handle_) {
            state.SkipWithError("superbloom_create failed during iteration");
            return;
        }

        fixture.timer.start();
        auto added =
            g_cpuFastxParallelizeRecords
                ? superbloom_insert_fastx_path(fixture.handle_, fixture.throughputFastxPath.c_str())
                : superbloom_insert_sequence(
                      fixture.handle_,
                      reinterpret_cast<const uint8_t*>(fixture.h_sequence.data()),
                      fixture.sequenceLength
                  );
        if (added < 0) {
            state.SkipWithError("superbloom insert failed");
            return;
        }
        state.SetIterationTime(fixture.timer.elapsed());
        benchmark::DoNotOptimize(added);
    }
    fixture.setCounters(state);
}

template <typename Fixture>
void runSuperBloomCpuQuery(Fixture& fixture, benchmark::State& state) {
    if (g_cpuFastxParallelizeRecords) {
        fixture.ensureThroughputFastxPath();
    } else {
        fixture.ensureHostSequence();
    }

    fixture.recreateFilter();
    if (!fixture.handle_) {
        state.SkipWithError("superbloom_create failed");
        return;
    }
    if (g_cpuFastxParallelizeRecords) {
        if (superbloom_insert_fastx_path(fixture.handle_, fixture.throughputFastxPath.c_str()) <
            0) {
            state.SkipWithError("superbloom insert failed");
            return;
        }
    } else {
        if (superbloom_insert_sequence(
                fixture.handle_,
                reinterpret_cast<const uint8_t*>(fixture.h_sequence.data()),
                fixture.sequenceLength
            ) < 0) {
            state.SkipWithError("superbloom insert failed");
            return;
        }
    }
    superbloom_freeze(fixture.handle_);

    for (auto _ : state) {
        fixture.timer.start();
        auto positives =
            g_cpuFastxParallelizeRecords
                ? superbloom_query_fastx_path(fixture.handle_, fixture.throughputFastxPath.c_str())
                : superbloom_query_sequence(
                      fixture.handle_,
                      reinterpret_cast<const uint8_t*>(fixture.h_sequence.data()),
                      fixture.sequenceLength
                  );
        if (positives < 0) {
            state.SkipWithError("superbloom query failed");
            return;
        }
        state.SetIterationTime(fixture.timer.elapsed());
        benchmark::DoNotOptimize(positives);
    }
    fixture.setCounters(state);
}

template <typename Fixture>
void runSuperBloomCpuFpr(Fixture& fixture, benchmark::State& state) {
    if (g_cpuFastxParallelizeRecords) {
        fixture.ensureFprFastxPaths();
    } else {
        fixture.benchData->ensureFprData();
    }

    std::vector<char> fprHostSeq(fixture.sequenceLength);
    if (!g_cpuFastxParallelizeRecords) {
        CUSBF_CUDA_CALL(cudaMemcpy(
            fprHostSeq.data(),
            thrust::raw_pointer_cast(fixture.benchData->d_fprInsertSequence.data()),
            fixture.sequenceLength,
            cudaMemcpyDeviceToHost
        ));
    }

    std::vector<char> zeroHostSeq(fixture.sequenceLength);
    if (!g_cpuFastxParallelizeRecords) {
        CUSBF_CUDA_CALL(cudaMemcpy(
            zeroHostSeq.data(),
            thrust::raw_pointer_cast(fixture.benchData->d_zeroOverlapSequence.data()),
            fixture.sequenceLength,
            cudaMemcpyDeviceToHost
        ));
    }

    fixture.recreateFilter();
    if (!fixture.handle_) {
        state.SkipWithError("superbloom_create failed");
        return;
    }
    if (g_cpuFastxParallelizeRecords) {
        if (superbloom_insert_fastx_path(fixture.handle_, fixture.fprInsertFastxPath.c_str()) < 0) {
            state.SkipWithError("superbloom insert failed");
            return;
        }
    } else {
        if (superbloom_insert_sequence(
                fixture.handle_,
                reinterpret_cast<const uint8_t*>(fprHostSeq.data()),
                fixture.sequenceLength
            ) < 0) {
            state.SkipWithError("superbloom insert failed");
            return;
        }
    }
    superbloom_freeze(fixture.handle_);

    uint64_t falsePositives = 0;
    for (auto _ : state) {
        fixture.timer.start();
        auto positives =
            g_cpuFastxParallelizeRecords
                ? superbloom_query_fastx_path(fixture.handle_, fixture.fprQueryFastxPath.c_str())
                : superbloom_query_sequence(
                      fixture.handle_,
                      reinterpret_cast<const uint8_t*>(zeroHostSeq.data()),
                      fixture.sequenceLength
                  );
        if (positives < 0) {
            state.SkipWithError("superbloom query failed");
            return;
        }
        state.SetIterationTime(fixture.timer.elapsed());
        falsePositives = static_cast<uint64_t>(positives);
        benchmark::DoNotOptimize(falsePositives);
    }

    fixture.setCounters(state);
    setFprCounters(state, falsePositives, fixture.numKmers);
}

}  // namespace benchmark_common

#define BENCHMARK_CUSBF_CONFIG_SYMBOL(K, S, M, H) CuSBF_K##K##_S##S##_M##M##_H##H##_Config
#define BENCHMARK_CUSBF_FIXTURE_SYMBOL(K, S, M, H) CuSBF_K##K##_S##S##_M##M##_H##H##_Fixture
#define BENCHMARK_SUPERBLOOM_CPU_FIXTURE_SYMBOL(K, S, M, H) \
    SuperBloomCpu_K##K##_S##S##_M##M##_H##H##_Fixture

#define BENCHMARK_DEFINE_CUSBF_CONFIG_AND_FIXTURE(K, S, M, H)                         \
    using BENCHMARK_CUSBF_CONFIG_SYMBOL(K, S, M, H) = cusbf::Config<K, S, M, H, 256>; \
    using BENCHMARK_CUSBF_FIXTURE_SYMBOL(K, S, M, H) =                                \
        benchmark_common::CuSbfConfigFixture<BENCHMARK_CUSBF_CONFIG_SYMBOL(K, S, M, H)>;

#define BENCHMARK_DEFINE_CUSBF_ALL(FixtureName)                         \
    BENCHMARK_DEFINE_F(FixtureName, Insert)(benchmark::State & state) { \
        benchmark_common::runCuSbfInsert(*this, state);                 \
    }                                                                   \
    BENCHMARK_DEFINE_F(FixtureName, Query)(benchmark::State & state) {  \
        benchmark_common::runCuSbfQuery(*this, state);                  \
    }                                                                   \
    BENCHMARK_DEFINE_F(FixtureName, FPR)(benchmark::State & state) {    \
        benchmark_common::runCuSbfFpr(*this, state);                    \
    }

#define BENCHMARK_DEFINE_CUSBF_FPR_ONLY(FixtureName)                 \
    BENCHMARK_DEFINE_F(FixtureName, FPR)(benchmark::State & state) { \
        benchmark_common::runCuSbfFpr(*this, state);                 \
    }

#define BENCHMARK_DEFINE_SUPERBLOOM_CPU_ALL(FixtureName)                \
    BENCHMARK_DEFINE_F(FixtureName, Insert)(benchmark::State & state) { \
        benchmark_common::runSuperBloomCpuInsert(*this, state);         \
    }                                                                   \
    BENCHMARK_DEFINE_F(FixtureName, Query)(benchmark::State & state) {  \
        benchmark_common::runSuperBloomCpuQuery(*this, state);          \
    }                                                                   \
    BENCHMARK_DEFINE_F(FixtureName, FPR)(benchmark::State & state) {    \
        benchmark_common::runSuperBloomCpuFpr(*this, state);            \
    }

#define BENCHMARK_DEFINE_SUPERBLOOM_CPU_FPR_ONLY(FixtureName)        \
    BENCHMARK_DEFINE_F(FixtureName, FPR)(benchmark::State & state) { \
        benchmark_common::runSuperBloomCpuFpr(*this, state);         \
    }

#define BENCHMARK_REGISTER_CUSBF_ALL(FixtureName) \
    REGISTER_BENCHMARK(FixtureName, Insert);      \
    REGISTER_BENCHMARK(FixtureName, Query);       \
    REGISTER_BENCHMARK(FixtureName, FPR);

#define BENCHMARK_REGISTER_CUSBF_FPR_ONLY(FixtureName) REGISTER_BENCHMARK(FixtureName, FPR);

#define BENCHMARK_REGISTER_SUPERBLOOM_CPU_ALL(FixtureName) \
    REGISTER_BENCHMARK(FixtureName, Insert);               \
    REGISTER_BENCHMARK(FixtureName, Query);                \
    REGISTER_BENCHMARK(FixtureName, FPR);

#define BENCHMARK_REGISTER_SUPERBLOOM_CPU_FPR_ONLY(FixtureName) \
    REGISTER_BENCHMARK(FixtureName, FPR);

#define BENCHMARK_CONFIG                \
    ->RangeMultiplier(2)                \
        ->Range(1 << 16, 1ULL << 28)    \
        ->Unit(benchmark::kMillisecond) \
        ->UseManualTime()               \
        ->Iterations(10)                \
        ->Repetitions(5)                \
        ->ReportAggregatesOnly(true)

#define BENCHMARK_CONFIG_FPR_FASTX   \
    ->Unit(benchmark::kMillisecond)  \
        ->UseManualTime()            \
        ->Iterations(1)              \
        ->Repetitions(3)             \
        ->ReportAggregatesOnly(true) \
        ->ArgName("filter_bits_exp") \
        ->DenseRange(22, 31, 1)

#define REGISTER_BENCHMARK(FixtureName, BenchName) \
    BENCHMARK_REGISTER_F(FixtureName, BenchName)   \
    BENCHMARK_CONFIG

#define REGISTER_BENCHMARK_FPR_FASTX(FixtureName, BenchName) \
    BENCHMARK_REGISTER_F(FixtureName, BenchName)             \
    BENCHMARK_CONFIG_FPR_FASTX

// Single-point throughput comparison from one FASTX insert file (g_fastxBitsPerItem).
#define BENCHMARK_CONFIG_THROUGHPUT_FASTX \
    ->Unit(benchmark::kMillisecond)       \
        ->UseManualTime()                 \
        ->Iterations(10)                  \
        ->Repetitions(5)                  \
        ->ReportAggregatesOnly(true)

#define REGISTER_BENCHMARK_THROUGHPUT_FASTX(FixtureName, BenchName) \
    BENCHMARK_REGISTER_F(FixtureName, BenchName)                    \
    BENCHMARK_CONFIG_THROUGHPUT_FASTX

#define STANDARD_BENCHMARK_MAIN()                                   \
    int main(int argc, char** argv) {                               \
        ::benchmark::Initialize(&argc, argv);                       \
        if (::benchmark::ReportUnrecognizedArguments(argc, argv)) { \
            return 1;                                               \
        }                                                           \
        ::benchmark::RunSpecifiedBenchmarks();                      \
        ::benchmark::Shutdown();                                    \
        fflush(stdout);                                             \
        std::_Exit(0);                                              \
    }
