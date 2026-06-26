#include <benchmark/benchmark.h>
#include <cuda/__cmath/ceil_div.h>
#include <cuda_runtime.h>
#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>

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
#include "fastx_workload.hpp"

#include <cuda/std/functional>

#include <cuckoogpu/CuckooFilter.cuh>
#include <cuco/bloom_filter.cuh>
#include <cusbf/device_span.cuh>
#include <cusbf/filter.cuh>
#include <cusbf/helpers.cuh>
#include <cusbf/superbloom_ffi.hpp>

#include "benchmark_common.cuh"
#include "gpu_filter_gqf_tcf.cuh"

namespace gqf_tcf = gpu_filter_gqf_tcf;

namespace bm = benchmark;
namespace fb = benchmark_common::filter_benchmark;

using CucoBloom = cuco::bloom_filter<uint64_t>;

static constexpr uint32_t kFprQuerySeed = 0xDEADBEEF;
static constexpr uint64_t kDnaK = 31;

using CuckooGpuConfig =
    cuckoogpu::Config<uint64_t, 16, 500, 256, 16, cuckoogpu::XorAltBucketPolicy>;
using CuckooGpuFilter = cuckoogpu::Filter<CuckooGpuConfig>;

struct FastxData {
    std::vector<char> host_insert;
    thrust::device_vector<char> d_insert;
    thrust::device_vector<uint64_t> d_insert_packed;
    uint64_t insert_kmers = 0;

    uint64_t num_records = 0;
    std::string insert_fasta;
    std::string query_fasta;

    uint64_t fpr_query_kmers = 0;
    uint64_t query_chunk_kmers = 0;
    thrust::device_vector<uint64_t> d_query_keys;
    thrust::device_vector<char> d_query_seq;
    thrust::device_vector<uint8_t> d_query_hits;
};

static std::unique_ptr<FastxData> g_data;
static std::string g_insert_fastx_path;
static uint64_t g_num_records = 0;
static uint64_t g_fpr_query_kmers = 1'000'000'000ULL;
static uint64_t g_fpr_query_chunk_kmers = 32'000'000ULL;

struct TempFastaPath {
    std::string path;

    explicit TempFastaPath(std::string pathValue) : path(std::move(pathValue)) {}
    TempFastaPath(const TempFastaPath&) = delete;
    TempFastaPath& operator=(const TempFastaPath&) = delete;
    TempFastaPath(TempFastaPath&& other) noexcept : path(std::move(other.path)) {
        other.path.clear();
    }
    ~TempFastaPath() {
        if (!path.empty()) {
            std::remove(path.c_str());
        }
    }
};

static TempFastaPath makeTempFastaPath(const char* prefix) {
    std::string templ = std::string("/tmp/") + prefix + "-XXXXXX";
    std::vector<char> buf(templ.begin(), templ.end());
    buf.push_back('\0');
    const int fd = mkstemp(buf.data());
    if (fd == -1) {
        throw std::runtime_error("failed to create temporary FASTA file");
    }
    close(fd);
    return TempFastaPath{buf.data()};
}

static uint64_t effectiveNumRecords() {
    if (g_num_records != 0) {
        return g_num_records;
    }
    return std::max<uint64_t>(1, static_cast<uint64_t>(std::thread::hardware_concurrency()));
}

// Even split of @p total into @p numRecords parts; last record takes the remainder.
static uint64_t recordShare(uint64_t total, uint64_t numRecords, uint64_t recordIndex) {
    const uint64_t start = (total * recordIndex) / numRecords;
    const uint64_t end =
        (recordIndex + 1 == numRecords) ? total : (total * (recordIndex + 1)) / numRecords;
    return end - start;
}

static void writePartitionedInsertFasta(const FastxData& data) {
    std::ofstream out(data.insert_fasta, std::ios::binary | std::ios::trunc);
    if (!out) {
        throw std::runtime_error("failed to open insert FASTA for writing");
    }
    for (uint64_t r = 0; r < data.num_records; ++r) {
        const uint64_t start = (data.host_insert.size() * r) / data.num_records;
        const uint64_t len = recordShare(data.host_insert.size(), data.num_records, r);
        out << ">insert_" << r << '\n';
        out.write(data.host_insert.data() + start, static_cast<std::streamsize>(len));
        out.put('\n');
    }
}

static void writePartitionedQueryFasta(FastxData& data) {
    std::ofstream out(data.query_fasta, std::ios::binary | std::ios::trunc);
    if (!out) {
        throw std::runtime_error("failed to open query FASTA for writing");
    }
    thrust::device_vector<char> d_record;
    std::vector<char> host_record;
    for (uint64_t r = 0; r < data.num_records; ++r) {
        const uint64_t kmers = recordShare(data.fpr_query_kmers, data.num_records, r);
        const uint64_t bases = kmers + kDnaK - 1;
        d_record.resize(bases);
        benchmark_common::gpuGenerateDna(
            d_record, bases, kFprQuerySeed + static_cast<uint32_t>(r & 0xFFFFFFFFu)
        );
        host_record.resize(bases);
        CUSBF_CUDA_CALL(cudaMemcpy(
            host_record.data(),
            thrust::raw_pointer_cast(d_record.data()),
            bases,
            cudaMemcpyDeviceToHost
        ));
        out << ">query_" << r << '\n';
        out.write(host_record.data(), static_cast<std::streamsize>(bases));
        out.put('\n');
    }
}

static void prepareFastxData() {
    if (g_data) {
        return;
    }
    if (g_insert_fastx_path.empty()) {
        std::cerr << "Error: --insert-fastx is required\n";
        std::exit(1);
    }
    if (g_fpr_query_kmers == 0) {
        std::cerr << "Error: --fpr-query-kmers must be > 0\n";
        std::exit(1);
    }
    if (g_fpr_query_chunk_kmers == 0 || g_fpr_query_chunk_kmers > g_fpr_query_kmers) {
        g_fpr_query_chunk_kmers = std::min(g_fpr_query_kmers, uint64_t{32'000'000});
    }

    auto data = std::make_unique<FastxData>();
    auto prepared =
        benchmark_common::fastx_workload::load_fastx_sequence<kDnaK, cusbf::DnaAlphabet>(
            g_insert_fastx_path
        );
    if (prepared.host_sequence.empty()) {
        std::cerr << "Error: FASTX file is empty or contains no sequences\n";
        std::exit(1);
    }

    data->num_records = effectiveNumRecords();
    data->insert_kmers = prepared.kmers;
    data->fpr_query_kmers = g_fpr_query_kmers;
    data->query_chunk_kmers = g_fpr_query_chunk_kmers;

    // upload_sequence copies prepared.host_sequence -> prepared.d_sequence, so it must run
    // before the host buffer is moved out of `prepared` (otherwise the device buffer ends
    // up empty while encode_packed_kmers is still launched against host_insert.size()).
    benchmark_common::fastx_workload::upload_sequence(prepared);
    data->host_insert = std::move(prepared.host_sequence);
    data->d_insert = std::move(prepared.d_sequence);
    data->d_insert_packed.resize(data->insert_kmers);
    if (data->insert_kmers != 0) {
        benchmark_common::fastx_workload::encode_packed_kmers<kDnaK, cusbf::DnaAlphabet>(
            thrust::raw_pointer_cast(data->d_insert.data()),
            data->host_insert.size(),
            thrust::raw_pointer_cast(data->d_insert_packed.data())
        );
    }

    data->d_query_keys.resize(data->query_chunk_kmers);
    data->d_query_seq.resize(data->query_chunk_kmers + kDnaK - 1);
    data->d_query_hits.resize(data->query_chunk_kmers);

    TempFastaPath insertPath = makeTempFastaPath("fpr-insert");
    data->insert_fasta = std::move(insertPath.path);
    writePartitionedInsertFasta(*data);

    TempFastaPath queryPath = makeTempFastaPath("fpr-query");
    data->query_fasta = std::move(queryPath.path);
    writePartitionedQueryFasta(*data);

    CUSBF_CUDA_CALL(cudaDeviceSynchronize());
    g_data = std::move(data);
}

static uint64_t requestedFilterBitsForState(const bm::State& state) {
    if (benchmark_common::g_fastxFilterBitsOverride != 0) {
        return benchmark_common::g_fastxFilterBitsOverride;
    }
    const int64_t bitsExp = state.range(0);
    if (bitsExp <= 0) {
        return benchmark_common::resolveFastxFilterBits(g_data->insert_kmers);
    }
    return uint64_t{1} << static_cast<unsigned>(bitsExp);
}

static uint64_t countDeviceHits(const thrust::device_vector<uint8_t>& hits, uint64_t count) {
    return static_cast<uint64_t>(
        thrust::count(hits.begin(), hits.begin() + static_cast<int64_t>(count), uint8_t{1})
    );
}

static uint32_t chunkSeed(uint64_t offset) {
    return kFprQuerySeed + static_cast<uint32_t>(offset & 0xFFFFFFFFu);
}

static void generateQueryChunkKeys(uint64_t offset, uint64_t chunkKmers) {
    benchmark_common::gpuGeneratePackedKmers(g_data->d_query_keys, chunkKmers, chunkSeed(offset));
}

static void generateQueryChunkSequence(uint64_t offset, uint64_t chunkKmers) {
    benchmark_common::gpuGenerateDna(
        g_data->d_query_seq, chunkKmers + kDnaK - 1, chunkSeed(offset)
    );
}

static void setFprFastxCounters(
    bm::State& state,
    uint64_t filter_bits,
    uint64_t memoryBytes,
    uint64_t insert_kmers,
    uint64_t query_kmers
) {
    state.counters["filter_bits"] = bm::Counter(static_cast<double>(filter_bits));
    state.counters["memory_bytes"] =
        bm::Counter(static_cast<double>(memoryBytes), bm::Counter::kDefaults, bm::Counter::kIs1024);
    state.counters["insert_kmers"] = bm::Counter(static_cast<double>(insert_kmers));
    state.counters["query_kmers"] = bm::Counter(static_cast<double>(query_kmers));
    state.counters["bits_per_item"] = bm::Counter(
        insert_kmers > 0 ? static_cast<double>(filter_bits) / static_cast<double>(insert_kmers)
                         : 0.0
    );
}

template <typename Config>
class CuSbfFprFastxFixture : public bm::Fixture {
    using bm::Fixture::SetUp;
    using bm::Fixture::TearDown;

   public:
    void SetUp(const bm::State& state) override {
        prepareFastxData();

        filter_bits = requestedFilterBitsForState(state);
        filter = std::make_unique<cusbf::filter<Config>>(filter_bits);
        filterMemory = filter->filter_bits() / 8;
    }

    void TearDown(const bm::State&) override {
        filter.reset();
    }

    uint64_t filter_bits = 0;
    uint64_t filterMemory = 0;
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
    X(28)                                  \
    X(30)                                  \
    X(31)

#define FOR_EACH_SUPERBLOOM_CPU_FPR_FASTX_CONFIG(X) \
    X(27)                                           \
    X(28)

FOR_EACH_CUSBF_FPR_FASTX_CONFIG(DEFINE_CUSBF_FPR_FASTX_CONFIG_AND_FIXTURE)

#undef DEFINE_CUSBF_FPR_FASTX_CONFIG_AND_FIXTURE

class CucoBloomFprFastxFixture : public bm::Fixture {
    using bm::Fixture::SetUp;
    using bm::Fixture::TearDown;

   public:
    void SetUp(const bm::State& state) override {
        prepareFastxData();

        filter_bits = requestedFilterBitsForState(state);
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
    }

    void TearDown(const bm::State&) override {
        filter.reset();
    }

    uint64_t filter_bits = 0;
    uint64_t actualFilterBits = 0;
    uint64_t filterMemory = 0;
    std::unique_ptr<CucoBloom> filter;
    benchmark_common::GPUTimer timer;
};

class GqfFprFastxFixture : public bm::Fixture {
    using bm::Fixture::SetUp;
    using bm::Fixture::TearDown;

   public:
    void SetUp(const bm::State& state) override {
        prepareFastxData();

        filter_bits = requestedFilterBitsForState(state);
        filter.createForFilterBits(filter_bits);
        filterMemory = filter.filterBytes();
        actualFilterBits = filter.filterBits();

        d_gqfResults.resize(g_data->query_chunk_kmers);
        d_scratchKeys.resize(g_data->insert_kmers);
    }

    void TearDown(const bm::State&) override {
        filter.destroy();
        d_gqfResults.clear();
        d_gqfResults.shrink_to_fit();
        d_scratchKeys.clear();
        d_scratchKeys.shrink_to_fit();
    }

    uint64_t filter_bits = 0;
    uint64_t actualFilterBits = 0;
    uint64_t filterMemory = 0;
    thrust::device_vector<uint64_t> d_gqfResults;
    thrust::device_vector<uint64_t> d_scratchKeys;
    gqf_tcf::GqfHandle filter;
    benchmark_common::GPUTimer timer;
};

class TcfFprFastxFixture : public bm::Fixture {
    using bm::Fixture::SetUp;
    using bm::Fixture::TearDown;

   public:
    void SetUp(const bm::State& state) override {
        prepareFastxData();

        filter_bits = requestedFilterBitsForState(state);
        filter.createForFilterBits(filter_bits);
        filterMemory = filter.filterBytes();
        actualFilterBits = filter.filterBits();

        d_scratchKeys.resize(g_data->insert_kmers);

        const uint64_t reservedGpuBytes = filterMemory + g_data->d_insert.size() +
                                          g_data->d_insert_packed.size() * sizeof(uint64_t) +
                                          d_scratchKeys.size() * sizeof(uint64_t);
        benchmark_common::resolveFastxChunkKmers(
            g_data->insert_kmers, reservedGpuBytes, benchmark_common::kTcfFastxChunkBytesPerKmer
        );
        filter.bindWorkload(g_data->insert_kmers);
    }

    void TearDown(const bm::State&) override {
        filter.destroy();
        d_scratchKeys.clear();
        d_scratchKeys.shrink_to_fit();
    }

    uint64_t filter_bits = 0;
    uint64_t actualFilterBits = 0;
    uint64_t filterMemory = 0;
    thrust::device_vector<uint64_t> d_scratchKeys;
    gqf_tcf::TcfHandle filter;
    benchmark_common::GPUTimer timer;
};

template <uint8_t BlockWindowS>
class SuperBloomCpuFprFastxFixture : public bm::Fixture {
    using bm::Fixture::SetUp;
    using bm::Fixture::TearDown;

   public:
    static constexpr uint8_t kBlockWindowS = BlockWindowS;

    void SetUp(const bm::State& state) override {
        prepareFastxData();

        filter_bits = requestedFilterBitsForState(state);

        const uint64_t targetBits = std::max(filter_bits, uint64_t{1} << 22);
        bitExp_ = static_cast<uint8_t>(cuda::std::bit_width(targetBits) - 1);
        blockExp_ = 9;

        createFilter();
    }

    void TearDown(const bm::State&) override {
        if (handle_) {
            superbloom_destroy(handle_);
        }
        handle_ = nullptr;
    }

    void createFilter() {
        superbloom_destroy(handle_);
        handle_ = superbloom_create(31, 21, kBlockWindowS, 8, bitExp_, blockExp_);
        if (handle_) {
            const unsigned n = std::thread::hardware_concurrency();
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
    benchmark_common::CPUTimer timer;
};

#define SUPERBLOOM_CPU_FPR_FASTX_FIXTURE_SYMBOL(S) SuperBloomCpuFprFastxFixture##S

#define DEFINE_SUPERBLOOM_CPU_FPR_FASTX_FIXTURE(S) \
    using SUPERBLOOM_CPU_FPR_FASTX_FIXTURE_SYMBOL(S) = SuperBloomCpuFprFastxFixture<S>;

FOR_EACH_SUPERBLOOM_CPU_FPR_FASTX_CONFIG(DEFINE_SUPERBLOOM_CPU_FPR_FASTX_FIXTURE)

#undef DEFINE_SUPERBLOOM_CPU_FPR_FASTX_FIXTURE

class CuckooGpuFprFastxFixture : public bm::Fixture {
    using bm::Fixture::SetUp;
    using bm::Fixture::TearDown;

   public:
    void SetUp(const bm::State& state) override {
        prepareFastxData();

        filter_bits = requestedFilterBitsForState(state);
        const uint64_t capacity = std::max(filter_bits / 16, uint64_t{1});
        filter = std::make_unique<CuckooGpuFilter>(capacity);
        filterMemory = filter->sizeInBytes();
        actualFilterBits = filterMemory * 8;

        queryOutput.resize(g_data->query_chunk_kmers);
    }

    void TearDown(const bm::State&) override {
        filter.reset();
        queryOutput.clear();
        queryOutput.shrink_to_fit();
    }

    uint64_t filter_bits = 0;
    uint64_t actualFilterBits = 0;
    uint64_t filterMemory = 0;
    std::unique_ptr<CuckooGpuFilter> filter;
    thrust::device_vector<uint8_t> queryOutput;
    benchmark_common::GPUTimer timer;
};

static uint64_t queryChunkKmerCount(uint64_t offset) {
    return std::min(g_data->query_chunk_kmers, g_data->fpr_query_kmers - offset);
}

template <typename Config>
void runCuSbfFprFastxBenchmark(CuSbfFprFastxFixture<Config>& fixture, bm::State& state) {
    (void)fixture.filter->clear();
    benchmark::DoNotOptimize(fixture.filter->insert_sequence_async(
        cusbf::device_span<const char>{
            thrust::raw_pointer_cast(g_data->d_insert.data()), g_data->d_insert.size()
        }
    ));
    CUSBF_CUDA_CALL(cudaDeviceSynchronize());

    uint64_t falsePositives = 0;
    for (auto _ : state) {
        fixture.timer.start();
        uint64_t positives = 0;
        for (uint64_t offset = 0; offset < g_data->fpr_query_kmers;
             offset += g_data->query_chunk_kmers) {
            const uint64_t chunkKmers = queryChunkKmerCount(offset);
            generateQueryChunkSequence(offset, chunkKmers);
            cusbf::require_void(fixture.filter->contains_sequence_async(
                cusbf::device_span<const char>{
                    thrust::raw_pointer_cast(g_data->d_query_seq.data()), g_data->d_query_seq.size()
                },
                cusbf::device_span<uint8_t>{
                    thrust::raw_pointer_cast(g_data->d_query_hits.data()), chunkKmers
                }
            ));
            CUSBF_CUDA_CALL(cudaDeviceSynchronize());
            positives += countDeviceHits(g_data->d_query_hits, chunkKmers);
        }
        state.SetIterationTime(fixture.timer.elapsed());
        falsePositives = positives;
        benchmark::DoNotOptimize(falsePositives);
    }

    setFprFastxCounters(
        state,
        fixture.filter->filter_bits(),
        fixture.filterMemory,
        g_data->insert_kmers,
        g_data->fpr_query_kmers
    );
    benchmark_common::setFprCounters(state, falsePositives, g_data->fpr_query_kmers);
    state.counters["s"] = benchmark::Counter(static_cast<double>(Config::s));
}

void runCucoFprFastxBenchmark(CucoBloomFprFastxFixture& fixture, bm::State& state) {
    (void)fixture.filter->clear();
    fixture.filter->add(g_data->d_insert_packed.begin(), g_data->d_insert_packed.end());
    CUSBF_CUDA_CALL(cudaDeviceSynchronize());

    uint64_t falsePositives = 0;
    for (auto _ : state) {
        fixture.timer.start();
        uint64_t positives = 0;
        for (uint64_t offset = 0; offset < g_data->fpr_query_kmers;
             offset += g_data->query_chunk_kmers) {
            const uint64_t chunkKmers = queryChunkKmerCount(offset);
            generateQueryChunkKeys(offset, chunkKmers);
            fixture.filter->contains(
                g_data->d_query_keys.begin(),
                g_data->d_query_keys.begin() + static_cast<int64_t>(chunkKmers),
                reinterpret_cast<bool*>(thrust::raw_pointer_cast(g_data->d_query_hits.data()))
            );
            positives += countDeviceHits(g_data->d_query_hits, chunkKmers);
        }
        state.SetIterationTime(fixture.timer.elapsed());
        falsePositives = positives;
        benchmark::DoNotOptimize(falsePositives);
    }

    setFprFastxCounters(
        state,
        fixture.actualFilterBits,
        fixture.filterMemory,
        g_data->insert_kmers,
        g_data->fpr_query_kmers
    );
    benchmark_common::setFprCounters(state, falsePositives, g_data->fpr_query_kmers);
}

void runGqfFprFastxBenchmark(GqfFprFastxFixture& fixture, bm::State& state) {
    if (!gqf_tcf::gqfSupportsItemsForFilterBits(fixture.actualFilterBits, g_data->insert_kmers)) {
        const std::string error =
            "GQF FPR benchmark requires at least " +
            std::to_string(gqf_tcf::gqfMinFilterBitsForItems(g_data->insert_kmers)) +
            " filter bits for " + std::to_string(g_data->insert_kmers) +
            " insert kmers at 0.95 load; got " + std::to_string(fixture.actualFilterBits);
        state.SkipWithError(error);
        return;
    }

    thrust::device_vector<uint64_t> d_insertScratch(g_data->insert_kmers);
    gqf_tcf::copyPackedKmers(g_data->d_insert_packed, d_insertScratch);
    gqf_tcf::gqfBulkInsert(
        fixture.filter, thrust::raw_pointer_cast(d_insertScratch.data()), g_data->insert_kmers
    );
    CUSBF_CUDA_CALL(cudaDeviceSynchronize());

    thrust::device_vector<uint64_t> d_queryScratch(g_data->query_chunk_kmers);
    uint64_t falsePositives = 0;
    for (auto _ : state) {
        fixture.timer.start();
        uint64_t positives = 0;
        for (uint64_t offset = 0; offset < g_data->fpr_query_kmers;
             offset += g_data->query_chunk_kmers) {
            const uint64_t chunkKmers = queryChunkKmerCount(offset);
            generateQueryChunkKeys(offset, chunkKmers);
            gqf_tcf::copyPackedKmers(g_data->d_query_keys, d_queryScratch);
            gqf_tcf::gqfBulkGet(
                fixture.filter,
                chunkKmers,
                thrust::raw_pointer_cast(d_queryScratch.data()),
                thrust::raw_pointer_cast(fixture.d_gqfResults.data())
            );
            gqf_tcf::convertGqfResults(fixture.d_gqfResults);
            positives += static_cast<uint64_t>(thrust::reduce(
                fixture.d_gqfResults.begin(),
                fixture.d_gqfResults.begin() + static_cast<int64_t>(chunkKmers),
                uint64_t{0},
                cuda::std::plus<uint64_t>{}
            ));
        }
        state.SetIterationTime(fixture.timer.elapsed());
        falsePositives = positives;
        benchmark::DoNotOptimize(falsePositives);
    }

    setFprFastxCounters(
        state,
        fixture.actualFilterBits,
        fixture.filterMemory,
        g_data->insert_kmers,
        g_data->fpr_query_kmers
    );
    benchmark_common::setFprCounters(state, falsePositives, g_data->fpr_query_kmers);
}

void runTcfFprFastxBenchmark(TcfFprFastxFixture& fixture, bm::State& state) {
    if (!gqf_tcf::tcfSupportsItemsForFilterBits(fixture.actualFilterBits, g_data->insert_kmers)) {
        const std::string error =
            "TCF FPR benchmark requires at least " +
            std::to_string(gqf_tcf::tcfMinFilterBitsForItems(g_data->insert_kmers)) +
            " filter bits for " + std::to_string(g_data->insert_kmers) +
            " insert kmers at 0.95 load; got " + std::to_string(fixture.actualFilterBits);
        state.SkipWithError(error);
        return;
    }

    thrust::device_vector<uint64_t> d_insertScratch(g_data->insert_kmers);
    gqf_tcf::copyPackedKmers(g_data->d_insert_packed, d_insertScratch);
    fixture.filter.bulkInsert(
        thrust::raw_pointer_cast(d_insertScratch.data()), g_data->insert_kmers
    );
    CUSBF_CUDA_CALL(cudaDeviceSynchronize());

    thrust::device_vector<uint64_t> d_queryScratch(g_data->query_chunk_kmers);
    thrust::device_vector<uint8_t> d_queryHits(g_data->query_chunk_kmers);
    uint64_t falsePositives = 0;
    for (auto _ : state) {
        fixture.timer.start();
        uint64_t positives = 0;
        for (uint64_t offset = 0; offset < g_data->fpr_query_kmers;
             offset += g_data->query_chunk_kmers) {
            const uint64_t chunkKmers = queryChunkKmerCount(offset);
            generateQueryChunkKeys(offset, chunkKmers);
            gqf_tcf::copyPackedKmers(g_data->d_query_keys, d_queryScratch);
            fixture.filter.bulkQueryInto(
                thrust::raw_pointer_cast(d_queryScratch.data()),
                chunkKmers,
                reinterpret_cast<bool*>(thrust::raw_pointer_cast(d_queryHits.data()))
            );
            positives += countDeviceHits(d_queryHits, chunkKmers);
        }
        state.SetIterationTime(fixture.timer.elapsed());
        falsePositives = positives;
        benchmark::DoNotOptimize(falsePositives);
    }

    setFprFastxCounters(
        state,
        fixture.actualFilterBits,
        fixture.filterMemory,
        g_data->insert_kmers,
        g_data->fpr_query_kmers
    );
    benchmark_common::setFprCounters(state, falsePositives, g_data->fpr_query_kmers);
}

void runCuckooGpuFprFastxBenchmark(CuckooGpuFprFastxFixture& fixture, bm::State& state) {
    (void)fixture.filter->clear();
    fixture.filter->insertMany(g_data->d_insert_packed);
    CUSBF_CUDA_CALL(cudaDeviceSynchronize());

    uint64_t falsePositives = 0;
    for (auto _ : state) {
        fixture.timer.start();
        uint64_t positives = 0;
        for (uint64_t offset = 0; offset < g_data->fpr_query_kmers;
             offset += g_data->query_chunk_kmers) {
            const uint64_t chunkKmers = queryChunkKmerCount(offset);
            generateQueryChunkKeys(offset, chunkKmers);
            g_data->d_query_keys.resize(chunkKmers);
            fixture.filter->containsMany(g_data->d_query_keys, fixture.queryOutput);
            positives += countDeviceHits(fixture.queryOutput, chunkKmers);
        }
        state.SetIterationTime(fixture.timer.elapsed());
        falsePositives = positives;
        benchmark::DoNotOptimize(falsePositives);
    }

    setFprFastxCounters(
        state,
        fixture.actualFilterBits,
        fixture.filterMemory,
        g_data->insert_kmers,
        g_data->fpr_query_kmers
    );
    benchmark_common::setFprCounters(state, falsePositives, g_data->fpr_query_kmers);
}

template <typename Fixture>
void runSuperBloomCpuFprFastxBenchmark(Fixture& fixture, bm::State& state) {
    fixture.createFilter();
    if (!fixture.handle_) {
        state.SkipWithError("superbloom_create failed");
        return;
    }

    if (superbloom_insert_fastx_path(fixture.handle_, g_data->insert_fasta.c_str()) < 0) {
        state.SkipWithError("superbloom insert failed");
        return;
    }
    superbloom_freeze(fixture.handle_);

    uint64_t falsePositives = 0;
    for (auto _ : state) {
        fixture.timer.start();
        const int64_t positives =
            superbloom_query_fastx_path(fixture.handle_, g_data->query_fasta.c_str());
        if (positives < 0) {
            state.SkipWithError("superbloom query failed");
            return;
        }
        state.SetIterationTime(fixture.timer.elapsed());
        falsePositives = static_cast<uint64_t>(positives);
        benchmark::DoNotOptimize(falsePositives);
    }

    setFprFastxCounters(
        state,
        fixture.actualFilterBits,
        fixture.filterMemory,
        g_data->insert_kmers,
        g_data->fpr_query_kmers
    );
    benchmark_common::setFprCounters(state, falsePositives, g_data->fpr_query_kmers);
    state.counters["s"] = benchmark::Counter(static_cast<double>(Fixture::kBlockWindowS));
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

BENCHMARK_DEFINE_F(CuckooGpuFprFastxFixture, FPR)(bm::State& state) {
    runCuckooGpuFprFastxBenchmark(*this, state);
}

#define DEFINE_SUPERBLOOM_CPU_FPR_FASTX_BENCHMARK(S)                    \
    BENCHMARK_DEFINE_F(SUPERBLOOM_CPU_FPR_FASTX_FIXTURE_SYMBOL(S), FPR) \
    (bm::State & state) {                                               \
        runSuperBloomCpuFprFastxBenchmark(*this, state);                \
    }

FOR_EACH_SUPERBLOOM_CPU_FPR_FASTX_CONFIG(DEFINE_SUPERBLOOM_CPU_FPR_FASTX_BENCHMARK)

#undef DEFINE_SUPERBLOOM_CPU_FPR_FASTX_BENCHMARK

BENCHMARK_DEFINE_F(GqfFprFastxFixture, FPR)(bm::State& state) {
    runGqfFprFastxBenchmark(*this, state);
}

BENCHMARK_DEFINE_F(TcfFprFastxFixture, FPR)(bm::State& state) {
    runTcfFprFastxBenchmark(*this, state);
}

#define REGISTER_CUSBF_FPR_FASTX_BENCHMARK(S) \
    REGISTER_BENCHMARK_FPR_FASTX(CUSBF_FPR_FASTX_FIXTURE_SYMBOL(S), FPR);

FOR_EACH_CUSBF_FPR_FASTX_CONFIG(REGISTER_CUSBF_FPR_FASTX_BENCHMARK)

#undef REGISTER_CUSBF_FPR_FASTX_BENCHMARK

REGISTER_BENCHMARK_FPR_FASTX(CucoBloomFprFastxFixture, FPR);
REGISTER_BENCHMARK_FPR_FASTX(CuckooGpuFprFastxFixture, FPR);
REGISTER_BENCHMARK_FPR_FASTX(GqfFprFastxFixture, FPR);
REGISTER_BENCHMARK_FPR_FASTX(TcfFprFastxFixture, FPR);

#define REGISTER_SUPERBLOOM_CPU_FPR_FASTX_BENCHMARK(S) \
    REGISTER_BENCHMARK_FPR_FASTX(SUPERBLOOM_CPU_FPR_FASTX_FIXTURE_SYMBOL(S), FPR);

FOR_EACH_SUPERBLOOM_CPU_FPR_FASTX_CONFIG(REGISTER_SUPERBLOOM_CPU_FPR_FASTX_BENCHMARK)

#undef REGISTER_SUPERBLOOM_CPU_FPR_FASTX_BENCHMARK
#undef FOR_EACH_CUSBF_FPR_FASTX_CONFIG
#undef FOR_EACH_SUPERBLOOM_CPU_FPR_FASTX_CONFIG
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
                g_insert_fastx_path = argv[++i];
            } else {
                std::cerr << "Missing value for --insert-fastx\n";
                std::exit(1);
            }
            continue;
        }

        constexpr const char* numRecordsPrefix = "--num-records=";
        if (std::strncmp(arg.c_str(), numRecordsPrefix, std::strlen(numRecordsPrefix)) == 0) {
            g_num_records = std::stoull(arg.substr(std::strlen(numRecordsPrefix)));
            continue;
        }
        if (arg == "--num-records") {
            if (i + 1 < argc) {
                g_num_records = std::stoull(argv[++i]);
            } else {
                std::cerr << "Missing value for --num-records\n";
                std::exit(1);
            }
            continue;
        }

        constexpr const char* queryKmersPrefix = "--fpr-query-kmers=";
        if (std::strncmp(arg.c_str(), queryKmersPrefix, std::strlen(queryKmersPrefix)) == 0) {
            g_fpr_query_kmers = std::stoull(arg.substr(std::strlen(queryKmersPrefix)));
            continue;
        }
        if (arg == "--fpr-query-kmers") {
            if (i + 1 < argc) {
                ++i;
                g_fpr_query_kmers = std::stoull(argv[i]);
            } else {
                std::cerr << "Missing value for --fpr-query-kmers" << std::endl;
                std::exit(1);
            }
            continue;
        }

        constexpr const char* chunkPrefix = "--fpr-query-chunk-kmers=";
        if (std::strncmp(arg.c_str(), chunkPrefix, std::strlen(chunkPrefix)) == 0) {
            g_fpr_query_chunk_kmers = std::stoull(arg.substr(std::strlen(chunkPrefix)));
            continue;
        }
        if (arg == "--fpr-query-chunk-kmers") {
            if (i + 1 < argc) {
                ++i;
                g_fpr_query_chunk_kmers = std::stoull(argv[i]);
            } else {
                std::cerr << "Missing value for --fpr-query-chunk-kmers" << std::endl;
                std::exit(1);
            }
            continue;
        }

        constexpr const char* filterBitsPrefix = "--filter-bits=";
        if (std::strncmp(arg.c_str(), filterBitsPrefix, std::strlen(filterBitsPrefix)) == 0) {
            benchmark_common::g_fastxFilterBitsOverride =
                std::stoull(arg.substr(std::strlen(filterBitsPrefix)));
            continue;
        }
        if (arg == "--filter-bits") {
            if (i + 1 < argc) {
                ++i;
                benchmark_common::g_fastxFilterBitsOverride = std::stoull(argv[i]);
            } else {
                std::cerr << "Missing value for --filter-bits" << std::endl;
                std::exit(1);
            }
            continue;
        }

        benchmarkArgv.push_back(argv[i]);
    }

    if (g_num_records == 0) {
        g_num_records = effectiveNumRecords();
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
    g_data.reset();
    fflush(stdout);
    std::_Exit(0);
}
