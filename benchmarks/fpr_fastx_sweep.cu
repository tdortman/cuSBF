#include <benchmark/benchmark.h>
#include <cuda/__cmath/ceil_div.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <unistd.h>

#include <cuco/bloom_filter.cuh>
#include <cusbf/device_span.cuh>
#include <cusbf/filter.cuh>
#include <cusbf/helpers.cuh>
#include <cusbf/superbloom_ffi.hpp>

#include "benchmark_common.cuh"

namespace bm = benchmark;

struct FastxData {
    thrust::device_vector<char> d_insert_sequence;
    uint64_t insertKmers = 0;

    thrust::device_vector<char> d_querySequence;
    uint64_t queryKmers = 0;

    // CPU SuperBloom parallelizes one thread per FASTX record, so keep an
    // explicit FASTA stream with a user-controlled record count.
    std::string query_fastx_path;
    uint64_t query_fastxKmers = 0;

    // Pre-encoded packed k-mers for Cuco
    thrust::device_vector<uint64_t> d_insertPackedKmers;
    thrust::device_vector<uint64_t> d_queryPackedKmers;
};

static std::unique_ptr<FastxData> g_fastxData;
static std::string g_insert_fastx_path;

static constexpr uint64_t kQueryLength = 1'000'000'000ULL;
static constexpr uint64_t kQuerySeed = 0xDEADBEEF;
static uint64_t g_numQueryRecords =
    std::max<uint64_t>(1, static_cast<uint64_t>(std::thread::hardware_concurrency()));

using CucoBloom = cuco::bloom_filter<uint64_t>;

static std::string writeGeneratedQueryFasta(
    const std::vector<char>& sequence,
    uint64_t numRecords,
    uint64_t& totalKmers
) {
    if (numRecords == 0) {
        std::cerr << "Error: --num-query-records must be >= 1" << std::endl;
        std::exit(1);
    }

    const auto tempDir = std::filesystem::temp_directory_path();
    const auto fileName = "bloom-fpr-fastx-sweep-" + std::to_string(getpid()) + "-" +
                          std::to_string(numRecords) + ".fasta";
    const auto path = tempDir / fileName;

    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        std::cerr << "Error: failed to create temporary FASTA at " << path << std::endl;
        std::exit(1);
    }

    totalKmers = 0;
    const uint64_t perRecordBases = sequence.size() / numRecords;
    uint64_t pos = 0;
    for (uint64_t r = 0; r < numRecords; ++r) {
        const uint64_t thisLen = (r == numRecords - 1) ? sequence.size() - pos : perRecordBases;
        out << ">query_" << r << '\n';
        out.write(sequence.data() + pos, static_cast<std::streamsize>(thisLen));
        out.put('\n');
        totalKmers += thisLen >= 31 ? thisLen - 31 + 1 : 0;
        pos += thisLen;
    }

    if (!out) {
        std::cerr << "Error: failed while writing temporary FASTA at " << path << std::endl;
        std::exit(1);
    }

    return path.string();
}

static void prepareFastxData() {
    if (g_fastxData) {
        return;
    }
    if (g_insert_fastx_path.empty()) {
        std::cerr << "Error: --insert-fastx is required" << std::endl;
        std::exit(1);
    }

    g_fastxData = std::make_unique<FastxData>();

    // Read insert FASTX
    std::vector<char> hostInsert = benchmark_common::readFastxConcatenated(g_insert_fastx_path);
    if (hostInsert.empty()) {
        std::cerr << "Error: FASTX file is empty or contains no sequences" << std::endl;
        std::exit(1);
    }

    g_fastxData->d_insert_sequence.resize(hostInsert.size());
    CUSBF_CUDA_CALL(cudaMemcpy(
        thrust::raw_pointer_cast(g_fastxData->d_insert_sequence.data()),
        hostInsert.data(),
        hostInsert.size(),
        cudaMemcpyHostToDevice
    ));
    g_fastxData->insertKmers = hostInsert.size() >= 31 ? hostInsert.size() - 31 + 1 : 0;

    // Generate full query sequence on device, then split into records
    thrust::device_vector<char> d_fullQuery;
    benchmark_common::gpuGenerateDna(d_fullQuery, kQueryLength, kQuerySeed);

    std::vector<char> hostFullQuery(kQueryLength);
    CUSBF_CUDA_CALL(cudaMemcpy(
        hostFullQuery.data(),
        thrust::raw_pointer_cast(d_fullQuery.data()),
        kQueryLength,
        cudaMemcpyDeviceToHost
    ));

    // Build concatenated GPU query with 'N' separators.
    const uint64_t perRecordBases = kQueryLength / g_numQueryRecords;
    std::vector<char> hostConcat;
    hostConcat.reserve(kQueryLength + g_numQueryRecords - 1);

    uint64_t pos = 0;
    for (uint64_t r = 0; r < g_numQueryRecords; ++r) {
        uint64_t thisLen = (r == g_numQueryRecords - 1) ? kQueryLength - pos : perRecordBases;
        hostConcat.insert(
            hostConcat.end(), hostFullQuery.begin() + pos, hostFullQuery.begin() + pos + thisLen
        );
        pos += thisLen;

        if (r + 1 < g_numQueryRecords) {
            hostConcat.push_back('N');
        }
    }

    // Copy concatenated query to GPU
    g_fastxData->d_querySequence.resize(hostConcat.size());
    CUSBF_CUDA_CALL(cudaMemcpy(
        thrust::raw_pointer_cast(g_fastxData->d_querySequence.data()),
        hostConcat.data(),
        hostConcat.size(),
        cudaMemcpyHostToDevice
    ));
    g_fastxData->queryKmers = hostConcat.size() >= 31 ? hostConcat.size() - 31 + 1 : 0;
    g_fastxData->query_fastx_path =
        writeGeneratedQueryFasta(hostFullQuery, g_numQueryRecords, g_fastxData->query_fastxKmers);

    // Pre-encode packed k-mers for Cuco
    g_fastxData->d_insertPackedKmers.resize(g_fastxData->insertKmers);
    benchmark_common::gpuEncodePackedKmers<31>(
        thrust::raw_pointer_cast(g_fastxData->d_insert_sequence.data()),
        hostInsert.size(),
        thrust::raw_pointer_cast(g_fastxData->d_insertPackedKmers.data())
    );

    g_fastxData->d_queryPackedKmers.resize(g_fastxData->queryKmers);
    benchmark_common::gpuEncodePackedKmers<31>(
        thrust::raw_pointer_cast(g_fastxData->d_querySequence.data()),
        hostConcat.size(),
        thrust::raw_pointer_cast(g_fastxData->d_queryPackedKmers.data())
    );

    CUSBF_CUDA_CALL(cudaDeviceSynchronize());
}

static void setFprFastxCounters(
    bm::State& state,
    uint64_t filter_bits,
    uint64_t memoryBytes,
    uint64_t insertKmers,
    uint64_t queryKmers
) {
    state.counters["filter_bits"] = bm::Counter(static_cast<double>(filter_bits));
    state.counters["memory_bytes"] =
        bm::Counter(static_cast<double>(memoryBytes), bm::Counter::kDefaults, bm::Counter::kIs1024);
    state.counters["insert_kmers"] = bm::Counter(static_cast<double>(insertKmers));
    state.counters["query_kmers"] = bm::Counter(static_cast<double>(queryKmers));
    state.counters["bits_per_item"] = bm::Counter(
        insertKmers > 0 ? static_cast<double>(filter_bits) / static_cast<double>(insertKmers) : 0.0
    );
}

template <typename Config>
class CuSbfFprFastxFixture : public bm::Fixture {
    using bm::Fixture::SetUp;
    using bm::Fixture::TearDown;

   public:
    void SetUp(const bm::State& state) override {
        prepareFastxData();

        filter_bits = static_cast<uint64_t>(state.range(0));
        filter = std::make_unique<cusbf::filter<Config>>(filter_bits);
        filterMemory = filter->filter_bits() / 8;

        d_output.resize(g_fastxData->queryKmers);
    }

    void TearDown(const bm::State&) override {
        filter.reset();
        d_output.clear();
        d_output.shrink_to_fit();
    }

    uint64_t filter_bits = 0;
    uint64_t filterMemory = 0;
    thrust::device_vector<uint8_t> d_output;
    std::unique_ptr<cusbf::filter<Config>> filter;
    benchmark_common::GPUTimer timer;
};

#define CUSBF_FPR_FASTX_CONFIG_SYMBOL(S) CuSBF_K31_S##S##_M21_H4_FprFastxConfig
#define CUSBF_FPR_FASTX_FIXTURE_SYMBOL(S) CuSBF_K31_S##S##_M21_H4_FprFastxFixture

#define DEFINE_CUSBF_FPR_FASTX_CONFIG_AND_FIXTURE(S)                           \
    using CUSBF_FPR_FASTX_CONFIG_SYMBOL(S) = cusbf::Config<31, S, 21, 4, 256>; \
    using CUSBF_FPR_FASTX_FIXTURE_SYMBOL(S) =                                  \
        CuSbfFprFastxFixture<CUSBF_FPR_FASTX_CONFIG_SYMBOL(S)>;

#define FOR_EACH_CUSBF_FPR_FASTX_CONFIG(X) \
    X(20)                                  \
    X(22)                                  \
    X(24)                                  \
    X(26)                                  \
    X(28)                                  \
    X(30)                                  \
    X(31)

FOR_EACH_CUSBF_FPR_FASTX_CONFIG(DEFINE_CUSBF_FPR_FASTX_CONFIG_AND_FIXTURE)

#undef DEFINE_CUSBF_FPR_FASTX_CONFIG_AND_FIXTURE

class CucoBloomFprFastxFixture : public bm::Fixture {
    using bm::Fixture::SetUp;
    using bm::Fixture::TearDown;

   public:
    void SetUp(const bm::State& state) override {
        prepareFastxData();

        filter_bits = static_cast<uint64_t>(state.range(0));
        constexpr auto bitsPerBlock =
            CucoBloom::words_per_block * sizeof(typename CucoBloom::word_type) * 8;
        uint64_t blocks = cuda::ceil_div(filter_bits, bitsPerBlock);
        if (blocks == 0) {
            blocks = 1;
        }
        filter = std::make_unique<CucoBloom>(blocks);
        filterMemory = filter->block_extent() * CucoBloom::words_per_block *
                       sizeof(typename CucoBloom::word_type);
        actualFilterBits = filterMemory * 8;

        d_output.resize(g_fastxData->queryKmers);
    }

    void TearDown(const bm::State&) override {
        filter.reset();
        d_output.clear();
        d_output.shrink_to_fit();
    }

    uint64_t filter_bits = 0;
    uint64_t actualFilterBits = 0;
    uint64_t filterMemory = 0;
    thrust::device_vector<uint8_t> d_output;
    std::unique_ptr<CucoBloom> filter;
    benchmark_common::GPUTimer timer;
};

class SuperBloomCpuFprFastxFixture : public bm::Fixture {
    using bm::Fixture::SetUp;
    using bm::Fixture::TearDown;

   public:
    void SetUp(const bm::State& state) override {
        prepareFastxData();

        filter_bits = static_cast<uint64_t>(state.range(0));

        // Copy insert sequence to host
        h_insert_sequence.resize(g_fastxData->d_insert_sequence.size());
        CUSBF_CUDA_CALL(cudaMemcpy(
            h_insert_sequence.data(),
            thrust::raw_pointer_cast(g_fastxData->d_insert_sequence.data()),
            g_fastxData->d_insert_sequence.size(),
            cudaMemcpyDeviceToHost
        ));

        // Compute CPU filter exponents from requested filter_bits
        uint64_t targetBits = std::max(filter_bits, uint64_t{1} << 22);
        bitExp_ = static_cast<uint8_t>(cuda::std::bit_width(targetBits) - 1);
        blockExp_ = 9;

        createFilter();
    }

    void TearDown(const bm::State&) override {
        if (handle_) {
            superbloom_destroy(handle_);
        }
        handle_ = nullptr;
        h_insert_sequence.clear();
    }

    void createFilter() {
        superbloom_destroy(handle_);
        handle_ = superbloom_create(31, 21, 27, 8, bitExp_, blockExp_);
        if (handle_) {
            unsigned n = std::thread::hardware_concurrency();
            if (n > 0) {
                superbloom_set_threads(handle_, n);
            }
            actualFilterBits = superbloom_filter_bits(handle_);
            filterMemory = actualFilterBits / 8;
        } else {
            actualFilterBits = 0;
            filterMemory = 0;
        }
    }

    uint64_t filter_bits = 0;
    uint64_t actualFilterBits = 0;
    uint64_t filterMemory = 0;
    uint8_t bitExp_ = 0;
    uint8_t blockExp_ = 0;
    void* handle_ = nullptr;
    std::vector<char> h_insert_sequence;
    benchmark_common::CPUTimer timer;
};

template <typename Fixture>
void runCuSbfFprFastxBenchmark(Fixture& fixture, bm::State& state) {
    fixture.filter->clear();
    benchmark::DoNotOptimize(fixture.filter->insert_sequence_async(
        cusbf::device_span<const char>{
            thrust::raw_pointer_cast(g_fastxData->d_insert_sequence.data()),
            g_fastxData->d_insert_sequence.size()
        }
    ));
    CUSBF_CUDA_CALL(cudaDeviceSynchronize());

    for (auto _ : state) {
        fixture.timer.start();
        fixture.filter->contains_sequence_async(
            cusbf::device_span<const char>{
                thrust::raw_pointer_cast(g_fastxData->d_querySequence.data()),
                g_fastxData->d_querySequence.size()
            },
            cusbf::device_span<uint8_t>{
                thrust::raw_pointer_cast(fixture.d_output.data()), fixture.d_output.size()
            }
        );
        const double elapsed = fixture.timer.elapsed();
        state.SetIterationTime(elapsed);
        benchmark::DoNotOptimize(thrust::raw_pointer_cast(fixture.d_output.data()));
    }

    const auto falsePositives = static_cast<uint64_t>(
        thrust::count(fixture.d_output.begin(), fixture.d_output.end(), uint8_t{1})
    );

    setFprFastxCounters(
        state,
        fixture.filter->filter_bits(),
        fixture.filterMemory,
        g_fastxData->insertKmers,
        g_fastxData->queryKmers
    );
    benchmark_common::setFprCounters(state, falsePositives, g_fastxData->queryKmers);
}

void runCucoFprFastxBenchmark(CucoBloomFprFastxFixture& fixture, bm::State& state) {
    fixture.filter->clear();
    fixture.filter->add(
        g_fastxData->d_insertPackedKmers.begin(), g_fastxData->d_insertPackedKmers.end()
    );
    CUSBF_CUDA_CALL(cudaDeviceSynchronize());

    for (auto _ : state) {
        fixture.timer.start();
        fixture.filter->contains(
            g_fastxData->d_queryPackedKmers.begin(),
            g_fastxData->d_queryPackedKmers.end(),
            reinterpret_cast<bool*>(thrust::raw_pointer_cast(fixture.d_output.data()))
        );
        const double elapsed = fixture.timer.elapsed();
        state.SetIterationTime(elapsed);
        benchmark::DoNotOptimize(thrust::raw_pointer_cast(fixture.d_output.data()));
    }

    const uint64_t falsePositives = static_cast<uint64_t>(
        thrust::count(fixture.d_output.begin(), fixture.d_output.end(), uint8_t{1})
    );

    setFprFastxCounters(
        state,
        fixture.actualFilterBits,
        fixture.filterMemory,
        g_fastxData->insertKmers,
        g_fastxData->queryKmers
    );
    benchmark_common::setFprCounters(state, falsePositives, g_fastxData->queryKmers);
}

void runSuperBloomCpuFprFastxBenchmark(SuperBloomCpuFprFastxFixture& fixture, bm::State& state) {
    fixture.createFilter();
    if (!fixture.handle_) {
        state.SkipWithError("superbloom_create failed");
        return;
    }

    superbloom_insert_sequence(
        fixture.handle_,
        reinterpret_cast<const uint8_t*>(fixture.h_insert_sequence.data()),
        fixture.h_insert_sequence.size()
    );
    superbloom_freeze(fixture.handle_);

    uint64_t falsePositives = 0;
    for (auto _ : state) {
        fixture.timer.start();
        const int64_t iterationPositives =
            superbloom_query_fastx_path(fixture.handle_, g_fastxData->query_fastx_path.c_str());
        if (iterationPositives < 0) {
            state.SkipWithError("superbloom_query_fastx_path failed");
            return;
        }
        state.SetIterationTime(fixture.timer.elapsed());
        falsePositives = static_cast<uint64_t>(iterationPositives);
        benchmark::DoNotOptimize(falsePositives);
    }

    setFprFastxCounters(
        state,
        fixture.actualFilterBits,
        fixture.filterMemory,
        g_fastxData->insertKmers,
        g_fastxData->query_fastxKmers
    );
    benchmark_common::setFprCounters(state, falsePositives, g_fastxData->query_fastxKmers);
}

#define DEFINE_CUSBF_FPR_FASTX_BENCHMARK(S)                    \
    BENCHMARK_DEFINE_F(CUSBF_FPR_FASTX_FIXTURE_SYMBOL(S), FPR) \
    (bm::State & state) {                                      \
        runCuSbfFprFastxBenchmark(*this, state);               \
    }

FOR_EACH_CUSBF_FPR_FASTX_CONFIG(DEFINE_CUSBF_FPR_FASTX_BENCHMARK)

#undef DEFINE_CUSBF_FPR_FASTX_BENCHMARK

BENCHMARK_DEFINE_F(CucoBloomFprFastxFixture, FPR)(bm::State& state) {
    runCucoFprFastxBenchmark(*this, state);
}

BENCHMARK_DEFINE_F(SuperBloomCpuFprFastxFixture, FPR)(bm::State& state) {
    runSuperBloomCpuFprFastxBenchmark(*this, state);
}

#define REGISTER_CUSBF_FPR_FASTX_BENCHMARK(S) \
    REGISTER_BENCHMARK_FPR_FASTX_SWEEP(CUSBF_FPR_FASTX_FIXTURE_SYMBOL(S), FPR);

FOR_EACH_CUSBF_FPR_FASTX_CONFIG(REGISTER_CUSBF_FPR_FASTX_BENCHMARK)

#undef REGISTER_CUSBF_FPR_FASTX_BENCHMARK

REGISTER_BENCHMARK_FPR_FASTX_SWEEP(CucoBloomFprFastxFixture, FPR);
REGISTER_BENCHMARK_FPR_FASTX_SWEEP(SuperBloomCpuFprFastxFixture, FPR);

#undef FOR_EACH_CUSBF_FPR_FASTX_CONFIG
#undef CUSBF_FPR_FASTX_FIXTURE_SYMBOL
#undef CUSBF_FPR_FASTX_CONFIG_SYMBOL

void parseCustomArgs(int argc, char** argv, std::vector<char*>& benchmarkArgv) {
    benchmarkArgv.clear();
    benchmarkArgv.reserve(argc);
    benchmarkArgv.push_back(argv[0]);

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        constexpr const char* fastxPrefix = "--insert-fastx=";
        if (std::strncmp(arg.c_str(), fastxPrefix, std::strlen(fastxPrefix)) == 0) {
            g_insert_fastx_path = arg.substr(std::strlen(fastxPrefix));
            continue;
        }
        if (arg == "--insert-fastx") {
            if (i + 1 < argc) {
                ++i;
                g_insert_fastx_path = argv[i];
            } else {
                std::cerr << "Missing value for --insert-fastx" << std::endl;
                std::exit(1);
            }
            continue;
        }

        constexpr const char* numRecordsPrefix = "--num-query-records=";
        if (std::strncmp(arg.c_str(), numRecordsPrefix, std::strlen(numRecordsPrefix)) == 0) {
            g_numQueryRecords = std::stoul(arg.substr(std::strlen(numRecordsPrefix)));
            continue;
        }
        if (arg == "--num-query-records") {
            if (i + 1 < argc) {
                ++i;
                g_numQueryRecords = std::stoul(argv[i]);
            } else {
                std::cerr << "Missing value for --num-query-records" << std::endl;
                std::exit(1);
            }
            continue;
        }

        benchmarkArgv.push_back(argv[i]);
    }
}

int main(int argc, char** argv) {
    std::vector<char*> benchmarkArgv;
    parseCustomArgs(argc, argv, benchmarkArgv);

    int benchmarkArgc = static_cast<int>(benchmarkArgv.size());
    ::benchmark::Initialize(&benchmarkArgc, benchmarkArgv.data());
    if (::benchmark::ReportUnrecognizedArguments(benchmarkArgc, benchmarkArgv.data())) {
        return 1;
    }
    ::benchmark::RunSpecifiedBenchmarks();
    ::benchmark::Shutdown();
    fflush(stdout);
    std::_Exit(0);
}
