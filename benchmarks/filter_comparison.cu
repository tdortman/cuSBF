#include <benchmark/benchmark.h>
#include <cuda/__cmath/ceil_div.h>
#include <cuda_runtime.h>
#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <cuda/std/functional>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <unistd.h>

#include <cuckoogpu/CuckooFilter.cuh>
#include <cuco/bloom_filter.cuh>
#include <cusbf/device_span.cuh>
#include <cusbf/filter.cuh>
#include <cusbf/helpers.cuh>
#include <cusbf/superbloom_ffi.hpp>

#include "benchmark_common.cuh"
#include "gpu_filter_gqf_tcf.cuh"

namespace fb = benchmark_common::filter_benchmark;
namespace gqf_tcf = gpu_filter_gqf_tcf;

namespace bm = benchmark;

using CucoBloom = cuco::bloom_filter<uint64_t>;
using CuckooGpuConfig =
    cuckoogpu::Config<uint64_t, 16, 500, 128, 16, cuckoogpu::XorAltBucketPolicy>;
using CuckooGpuFilter = cuckoogpu::Filter<CuckooGpuConfig>;

using CuSbfThroughputConfig = cusbf::Config<31, 28, 16, 4, 256>;
using SuperBloomCpuThroughputConfig = cusbf::Config<31, 28, 16, 4>;
using SuperBloomCpuFastxFixture =
    benchmark_common::SuperBloomCpuFastxFixture<SuperBloomCpuThroughputConfig>;

static void ensureThroughputFastxReady() {
    benchmark_common::prepareFastxInsertWorkload();
}

static void setThroughputCounters(
    bm::State& state,
    uint64_t filter_bits,
    uint64_t memoryBytes,
    uint64_t numKmers
) {
    fb::setFilterBenchmarkCounters(state, memoryBytes, numKmers);
    state.counters["filter_bits"] = bm::Counter(static_cast<double>(filter_bits));
}

struct CucoBloomSetup {
    uint64_t filter_bits = 0;
    uint64_t numKmers = 0;
    uint64_t filterMemory = 0;
    thrust::device_vector<uint8_t> output;
    std::unique_ptr<CucoBloom> filter;

    void init() {
        ensureThroughputFastxReady();
        filter_bits = benchmark_common::resolveFastxFilterBits(
            benchmark_common::g_fastxInsertWorkload->insert_kmers
        );
        numKmers = benchmark_common::g_fastxInsertWorkload->insert_kmers;

        constexpr auto bitsPerBlock =
            CucoBloom::words_per_block * sizeof(typename CucoBloom::word_type) * 8;
        uint64_t blocks = cuda::ceil_div(filter_bits, bitsPerBlock);
        if (blocks == 0) {
            blocks = 1;
        }

        filter = std::make_unique<CucoBloom>(blocks);
        filterMemory = filter->block_extent() * CucoBloom::words_per_block *
                       sizeof(typename CucoBloom::word_type);
        output.resize(numKmers);
    }
};

struct CuckooGpuSetup {
    uint64_t filter_bits = 0;
    uint64_t numKmers = 0;
    uint64_t filterMemory = 0;
    std::unique_ptr<CuckooGpuFilter> filter;

    void init() {
        ensureThroughputFastxReady();
        filter_bits = benchmark_common::resolveFastxFilterBits(
            benchmark_common::g_fastxInsertWorkload->insert_kmers
        );
        numKmers = benchmark_common::g_fastxInsertWorkload->insert_kmers;

        const uint64_t capacity = std::max(filter_bits / fb::kBitsPerTag, uint64_t{1});
        filter = std::make_unique<CuckooGpuFilter>(capacity);
        filterMemory = filter->sizeInBytes();
    }
};

struct GqfSetup {
    uint64_t filter_bits = 0;
    uint64_t numKmers = 0;
    uint64_t filterMemory = 0;
    thrust::device_vector<uint64_t> queryResults;
    gqf_tcf::GqfHandle handle;

    void init() {
        ensureThroughputFastxReady();
        filter_bits = benchmark_common::resolveFastxFilterBits(
            benchmark_common::g_fastxInsertWorkload->insert_kmers
        );
        numKmers = benchmark_common::g_fastxInsertWorkload->insert_kmers;

        handle.createForFilterBits(filter_bits);
        filterMemory = handle.filterBytes();
        queryResults.resize(numKmers);
    }

    void destroy() {
        handle.destroy();
    }
};

struct TcfSetup {
    uint64_t filter_bits = 0;
    uint64_t numKmers = 0;
    uint64_t filterMemory = 0;
    gqf_tcf::TcfHandle handle;

    void init() {
        ensureThroughputFastxReady();
        filter_bits = benchmark_common::resolveFastxFilterBits(
            benchmark_common::g_fastxInsertWorkload->insert_kmers
        );
        numKmers = benchmark_common::g_fastxInsertWorkload->insert_kmers;

        handle.createForFilterBits(filter_bits);
        filterMemory = handle.filterBytes();
    }

    void destroy() {
        handle.destroy();
    }
};

class CucoBloomFixture : public bm::Fixture {
    using bm::Fixture::SetUp;
    using bm::Fixture::TearDown;

   public:
    void SetUp(const bm::State&) override {
        setup.init();
    }

    void TearDown(const bm::State&) override {
        setup.filter.reset();
        setup.output.clear();
        setup.output.shrink_to_fit();
    }

    CucoBloomSetup setup;
    benchmark_common::GPUTimer timer;
};

class CuckooGpuFixture : public bm::Fixture {
    using bm::Fixture::SetUp;
    using bm::Fixture::TearDown;

   public:
    void SetUp(const bm::State&) override {
        setup.init();
    }

    void TearDown(const bm::State&) override {
        setup.filter.reset();
    }

    CuckooGpuSetup setup;
    benchmark_common::GPUTimer timer;
};

class GqfFixture : public bm::Fixture {
    using bm::Fixture::SetUp;
    using bm::Fixture::TearDown;

   public:
    void SetUp(const bm::State&) override {
        setup.init();
    }

    void TearDown(const bm::State&) override {
        setup.destroy();
        setup.queryResults.clear();
        setup.queryResults.shrink_to_fit();
    }

    GqfSetup setup;
    benchmark_common::GPUTimer timer;
};

class TcfFixture : public bm::Fixture {
    using bm::Fixture::SetUp;
    using bm::Fixture::TearDown;

   public:
    void SetUp(const bm::State&) override {
        setup.init();
    }

    void TearDown(const bm::State&) override {
        setup.destroy();
    }

    TcfSetup setup;
    benchmark_common::GPUTimer timer;
};

class CuSbfFixture : public bm::Fixture {
    using bm::Fixture::SetUp;
    using bm::Fixture::TearDown;

   public:
    void SetUp(const bm::State&) override {
        ensureThroughputFastxReady();
        filter_bits = benchmark_common::resolveFastxFilterBits(
            benchmark_common::g_fastxInsertWorkload->insert_kmers
        );
        numKmers = benchmark_common::g_fastxInsertWorkload->insert_kmers;
        filter = std::make_unique<cusbf::filter<CuSbfThroughputConfig>>(filter_bits);
        filterMemory = filter->filter_bits() / 8;
        queryOutput.resize(numKmers);
    }

    void TearDown(const bm::State&) override {
        filter.reset();
        queryOutput.clear();
        queryOutput.shrink_to_fit();
    }

    uint64_t filter_bits = 0;
    uint64_t numKmers = 0;
    uint64_t filterMemory = 0;
    thrust::device_vector<uint8_t> queryOutput;
    std::unique_ptr<cusbf::filter<CuSbfThroughputConfig>> filter;
    benchmark_common::GPUTimer timer;
};

BENCHMARK_DEFINE_F(CucoBloomFixture, Insert)(bm::State& state) {
    auto& fix = *static_cast<CucoBloomFixture*>(this);
    auto& s = fix.setup;
    for (auto _ : state) {
        (void)s.filter->clear();
        CUSBF_CUDA_CALL(cudaDeviceSynchronize());
        fix.timer.start();
        s.filter->add(
            benchmark_common::g_fastxInsertWorkload->d_insert_packed_kmers.begin(), benchmark_common::g_fastxInsertWorkload->d_insert_packed_kmers.end()
        );
        state.SetIterationTime(fix.timer.elapsed());
    }
    setThroughputCounters(state, s.filter_bits, s.filterMemory, s.numKmers);
}

BENCHMARK_DEFINE_F(CucoBloomFixture, Query)(bm::State& state) {
    auto& fix = *static_cast<CucoBloomFixture*>(this);
    auto& s = fix.setup;
    (void)s.filter->clear();
    s.filter->add(
        benchmark_common::g_fastxInsertWorkload->d_insert_packed_kmers.begin(), benchmark_common::g_fastxInsertWorkload->d_insert_packed_kmers.end()
    );
    CUSBF_CUDA_CALL(cudaDeviceSynchronize());
    for (auto _ : state) {
        fix.timer.start();
        s.filter->contains(
            benchmark_common::g_fastxInsertWorkload->d_insert_packed_kmers.begin(),
            benchmark_common::g_fastxInsertWorkload->d_insert_packed_kmers.end(),
            reinterpret_cast<bool*>(thrust::raw_pointer_cast(s.output.data()))
        );
        state.SetIterationTime(fix.timer.elapsed());
        bm::DoNotOptimize(thrust::raw_pointer_cast(s.output.data()));
    }
    setThroughputCounters(state, s.filter_bits, s.filterMemory, s.numKmers);
}

BENCHMARK_DEFINE_F(CuckooGpuFixture, Insert)(bm::State& state) {
    auto& fix = *static_cast<CuckooGpuFixture*>(this);
    auto& s = fix.setup;
    for (auto _ : state) {
        (void)s.filter->clear();
        CUSBF_CUDA_CALL(cudaDeviceSynchronize());
        fix.timer.start();
        s.filter->insertMany(benchmark_common::g_fastxInsertWorkload->d_insert_packed_kmers);
        state.SetIterationTime(fix.timer.elapsed());
    }
    setThroughputCounters(state, s.filter_bits, s.filterMemory, s.numKmers);
}

BENCHMARK_DEFINE_F(CuckooGpuFixture, Query)(bm::State& state) {
    auto& fix = *static_cast<CuckooGpuFixture*>(this);
    auto& s = fix.setup;
    thrust::device_vector<uint8_t> queryOutput(s.numKmers);
    (void)s.filter->clear();
    s.filter->insertMany(benchmark_common::g_fastxInsertWorkload->d_insert_packed_kmers);
    CUSBF_CUDA_CALL(cudaDeviceSynchronize());
    for (auto _ : state) {
        fix.timer.start();
        s.filter->containsMany(benchmark_common::g_fastxInsertWorkload->d_insert_packed_kmers, queryOutput);
        state.SetIterationTime(fix.timer.elapsed());
        bm::DoNotOptimize(thrust::raw_pointer_cast(queryOutput.data()));
    }
    setThroughputCounters(state, s.filter_bits, s.filterMemory, s.numKmers);
}

BENCHMARK_DEFINE_F(GqfFixture, Insert)(bm::State& state) {
    auto& fix = *static_cast<GqfFixture*>(this);
    auto& s = fix.setup;
    for (auto _ : state) {
        s.destroy();
        s.init();
        fix.timer.start();
        gqf_tcf::gqfBulkInsert(
            s.handle,
            thrust::raw_pointer_cast(benchmark_common::g_fastxInsertWorkload->d_insert_packed_kmers.data()),
            s.numKmers
        );
        state.SetIterationTime(fix.timer.elapsed());
        s.filterMemory = s.handle.filterBytes();
    }
    setThroughputCounters(state, s.filter_bits, s.filterMemory, s.numKmers);
}

BENCHMARK_DEFINE_F(GqfFixture, Query)(bm::State& state) {
    auto& fix = *static_cast<GqfFixture*>(this);
    auto& s = fix.setup;
    gqf_tcf::gqfBulkInsert(
        s.handle, thrust::raw_pointer_cast(benchmark_common::g_fastxInsertWorkload->d_insert_packed_kmers.data()), s.numKmers
    );
    CUSBF_CUDA_CALL(cudaDeviceSynchronize());
    for (auto _ : state) {
        fix.timer.start();
        gqf_tcf::gqfBulkGet(
            s.handle,
            s.numKmers,
            thrust::raw_pointer_cast(benchmark_common::g_fastxInsertWorkload->d_insert_packed_kmers.data()),
            thrust::raw_pointer_cast(s.queryResults.data())
        );
        state.SetIterationTime(fix.timer.elapsed());
        bm::DoNotOptimize(thrust::raw_pointer_cast(s.queryResults.data()));
    }
    setThroughputCounters(state, s.filter_bits, s.filterMemory, s.numKmers);
}

BENCHMARK_DEFINE_F(TcfFixture, Insert)(bm::State& state) {
    auto& fix = *static_cast<TcfFixture*>(this);
    auto& s = fix.setup;
    for (auto _ : state) {
        s.destroy();
        s.init();
        fix.timer.start();
        s.handle.bulkInsert(
            thrust::raw_pointer_cast(benchmark_common::g_fastxInsertWorkload->d_insert_packed_kmers.data()), s.numKmers
        );
        state.SetIterationTime(fix.timer.elapsed());
        s.filterMemory = s.handle.filterBytes();
    }
    setThroughputCounters(state, s.filter_bits, s.filterMemory, s.numKmers);
}

BENCHMARK_DEFINE_F(TcfFixture, Query)(bm::State& state) {
    auto& fix = *static_cast<TcfFixture*>(this);
    auto& s = fix.setup;
    s.handle.bulkInsert(
        thrust::raw_pointer_cast(benchmark_common::g_fastxInsertWorkload->d_insert_packed_kmers.data()), s.numKmers
    );
    CUSBF_CUDA_CALL(cudaDeviceSynchronize());
    for (auto _ : state) {
        fix.timer.start();
        bool* hits = s.handle.bulkQuery(
            thrust::raw_pointer_cast(benchmark_common::g_fastxInsertWorkload->d_insert_packed_kmers.data()), s.numKmers
        );
        state.SetIterationTime(fix.timer.elapsed());
        bm::DoNotOptimize(hits);
        CUSBF_CUDA_CALL(cudaFree(hits));
    }
    setThroughputCounters(state, s.filter_bits, s.filterMemory, s.numKmers);
}

BENCHMARK_DEFINE_F(CuSbfFixture, Insert)(bm::State& state) {
    auto& fix = *static_cast<CuSbfFixture*>(this);
    const cusbf::device_span<const char> insertSpan{
        thrust::raw_pointer_cast(benchmark_common::g_fastxInsertWorkload->d_insert_sequence.data()),
        benchmark_common::g_fastxInsertWorkload->d_insert_sequence.size(),
    };
    for (auto _ : state) {
        CUSBF_UNWRAP(fix.filter->clear());
        CUSBF_CUDA_CALL(cudaDeviceSynchronize());
        fix.timer.start();
        benchmark::DoNotOptimize(fix.filter->insert_sequence_async(insertSpan));
        state.SetIterationTime(fix.timer.elapsed());
    }
    setThroughputCounters(state, fix.filter_bits, fix.filterMemory, fix.numKmers);
    state.counters["s"] = bm::Counter(static_cast<double>(CuSbfThroughputConfig::s));
}

BENCHMARK_DEFINE_F(CuSbfFixture, Query)(bm::State& state) {
    auto& fix = *static_cast<CuSbfFixture*>(this);
    const cusbf::device_span<const char> insertSpan{
        thrust::raw_pointer_cast(benchmark_common::g_fastxInsertWorkload->d_insert_sequence.data()),
        benchmark_common::g_fastxInsertWorkload->d_insert_sequence.size(),
    };
    CUSBF_UNWRAP(fix.filter->clear());
    benchmark::DoNotOptimize(fix.filter->insert_sequence_async(insertSpan));
    CUSBF_CUDA_CALL(cudaDeviceSynchronize());

    for (auto _ : state) {
        fix.timer.start();
        cusbf::require_void(fix.filter->contains_sequence_async(
            insertSpan,
            cusbf::device_span<uint8_t>{
                thrust::raw_pointer_cast(fix.queryOutput.data()),
                fix.numKmers,
            }
        ));
        state.SetIterationTime(fix.timer.elapsed());
        bm::DoNotOptimize(thrust::raw_pointer_cast(fix.queryOutput.data()));
    }
    setThroughputCounters(state, fix.filter_bits, fix.filterMemory, fix.numKmers);
    state.counters["s"] = bm::Counter(static_cast<double>(CuSbfThroughputConfig::s));
}

REGISTER_BENCHMARK_THROUGHPUT_FASTX(CucoBloomFixture, Insert);
REGISTER_BENCHMARK_THROUGHPUT_FASTX(CucoBloomFixture, Query);
REGISTER_BENCHMARK_THROUGHPUT_FASTX(CuckooGpuFixture, Insert);
REGISTER_BENCHMARK_THROUGHPUT_FASTX(CuckooGpuFixture, Query);
REGISTER_BENCHMARK_THROUGHPUT_FASTX(GqfFixture, Insert);
REGISTER_BENCHMARK_THROUGHPUT_FASTX(GqfFixture, Query);
REGISTER_BENCHMARK_THROUGHPUT_FASTX(TcfFixture, Insert);
REGISTER_BENCHMARK_THROUGHPUT_FASTX(TcfFixture, Query);
REGISTER_BENCHMARK_THROUGHPUT_FASTX(CuSbfFixture, Insert);
REGISTER_BENCHMARK_THROUGHPUT_FASTX(CuSbfFixture, Query);

BENCHMARK_DEFINE_SUPERBLOOM_CPU_FASTX_ALL(SuperBloomCpuFastxFixture);
BENCHMARK_REGISTER_SUPERBLOOM_CPU_FASTX_ALL(SuperBloomCpuFastxFixture);

int main(int argc, char** argv) {
    auto cli = benchmark_common::parseFastxBenchmarkCli(argc, argv);
    std::vector<char*> benchmarkArgv = std::move(cli.benchmark_argv);
    int benchmarkArgc = static_cast<int>(benchmarkArgv.size());

    ::benchmark::Initialize(&benchmarkArgc, benchmarkArgv.data());
    if (::benchmark::ReportUnrecognizedArguments(benchmarkArgc, benchmarkArgv.data())) {
        return 1;
    }
    ::benchmark::RunSpecifiedBenchmarks();
    ::benchmark::Shutdown();
    benchmark_common::clearFastxInsertWorkload();
    fflush(stdout);
    std::_Exit(0);
}
