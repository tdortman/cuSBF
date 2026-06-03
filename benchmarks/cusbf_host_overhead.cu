#include <benchmark/benchmark.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>

#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include <unistd.h>

#include <cusbf/device_span.cuh>
#include <cusbf/filter.cuh>
#include <cusbf/helpers.cuh>

#include "benchmark_common.cuh"

namespace fb = benchmark_common::filter_benchmark;
namespace bm = benchmark;

using CuSbfThroughputConfig = cusbf::Config<31, 28, 16, 4, 256>;

static void ensureThroughputFastxReady() {
    benchmark_common::prepareFastxInsertWorkload<31>(
        cusbf::DnaAlphabet::separator, benchmark_common::FastxGpuPrepareKind::SequenceOnDevice
    );
}

static benchmark_common::FastxThroughputConfig throughputConfig() {
    return benchmark_common::resolveFastxThroughputConfig(31);
}

static const char* benchDeviceSequencePtr() {
    return thrust::raw_pointer_cast(
        benchmark_common::g_fastxInsertWorkload->d_insert_sequence.data()
    );
}

static uint64_t benchSequenceLength() {
    return throughputConfig().bench_seq_len;
}

static std::string_view benchHostSequenceView() {
    const auto& sequence = benchmark_common::g_fastxInsertWorkload->host_insert_sequence;
    return std::string_view{sequence.data(), benchSequenceLength()};
}

static void setThroughputCounters(
    bm::State& state,
    uint64_t filter_bits,
    uint64_t memoryBytes,
    uint64_t numKmers,
    double pipelineMode
) {
    fb::setFilterBenchmarkCounters(state, memoryBytes, numKmers);
    state.counters["filter_bits"] = bm::Counter(static_cast<double>(filter_bits));
    state.counters["pipeline_mode"] = bm::Counter(pipelineMode);
}

class CuSbfHostFixture : public bm::Fixture {
    using bm::Fixture::SetUp;
    using bm::Fixture::TearDown;

   public:
    void SetUp(const bm::State&) override {
        ensureThroughputFastxReady();
        const auto cfg = throughputConfig();
        filter_bits = cfg.filter_bits;
        numKmers = cfg.bench_kmers;
        filter = std::make_unique<cusbf::filter<CuSbfThroughputConfig>>(filter_bits);
        filterMemory = filter->filter_bits() / 8;
    }

    void TearDown(const bm::State&) override {
        filter.reset();
    }

    uint64_t filter_bits = 0;
    uint64_t numKmers = 0;
    uint64_t filterMemory = 0;
    std::unique_ptr<cusbf::filter<CuSbfThroughputConfig>> filter;
    benchmark_common::CPUTimer timer;
};

class CuSbfDeviceFixture : public bm::Fixture {
    using bm::Fixture::SetUp;
    using bm::Fixture::TearDown;

   public:
    void SetUp(const bm::State&) override {
        ensureThroughputFastxReady();
        const auto cfg = throughputConfig();
        filter_bits = cfg.filter_bits;
        numKmers = cfg.bench_kmers;
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

BENCHMARK_DEFINE_F(CuSbfHostFixture, Insert)(bm::State& state) {
    auto& fix = *static_cast<CuSbfHostFixture*>(this);
    const std::string_view sequence = benchHostSequenceView();

    for (auto _ : state) {
        fix.filter = std::make_unique<cusbf::filter<CuSbfThroughputConfig>>(fix.filter_bits);
        fix.filterMemory = fix.filter->filter_bits() / 8;

        fix.timer.start();
        const auto kmersInserted = fix.filter->insert_sequence(sequence);
        if (!kmersInserted) {
            state.SkipWithError(kmersInserted.error().message().c_str());
            return;
        }
        state.SetIterationTime(fix.timer.elapsed());
        benchmark::DoNotOptimize(uint64_t{*kmersInserted});
    }
    setThroughputCounters(state, fix.filter_bits, fix.filterMemory, fix.numKmers, 0.0);
}

BENCHMARK_DEFINE_F(CuSbfHostFixture, Query)(bm::State& state) {
    auto& fix = *static_cast<CuSbfHostFixture*>(this);
    const std::string_view sequence = benchHostSequenceView();

    const auto insertKmers = fix.filter->insert_sequence(sequence);
    if (!insertKmers) {
        state.SkipWithError(insertKmers.error().message().c_str());
        return;
    }
    CUSBF_CUDA_CALL(cudaDeviceSynchronize());

    for (auto _ : state) {
        fix.timer.start();
        const auto queryResults = fix.filter->contains_sequence(sequence);
        if (!queryResults) {
            state.SkipWithError(queryResults.error().message().c_str());
            return;
        }
        state.SetIterationTime(fix.timer.elapsed());
        benchmark::DoNotOptimize(queryResults->data());
        benchmark::DoNotOptimize(queryResults->size());
    }
    setThroughputCounters(state, fix.filter_bits, fix.filterMemory, fix.numKmers, 0.0);
}

BENCHMARK_DEFINE_F(CuSbfDeviceFixture, Insert)(bm::State& state) {
    auto& fix = *static_cast<CuSbfDeviceFixture*>(this);
    const cusbf::device_span<const char> insertSpan{benchDeviceSequencePtr(), benchSequenceLength()};

    for (auto _ : state) {
        CUSBF_UNWRAP(fix.filter->clear());
        CUSBF_CUDA_CALL(cudaDeviceSynchronize());
        fix.timer.start();
        benchmark::DoNotOptimize(fix.filter->insert_sequence_async(insertSpan));
        state.SetIterationTime(fix.timer.elapsed());
    }
    setThroughputCounters(state, fix.filter_bits, fix.filterMemory, fix.numKmers, 1.0);
}

BENCHMARK_DEFINE_F(CuSbfDeviceFixture, Query)(bm::State& state) {
    auto& fix = *static_cast<CuSbfDeviceFixture*>(this);
    const cusbf::device_span<const char> insertSpan{benchDeviceSequencePtr(), benchSequenceLength()};

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
        benchmark::DoNotOptimize(thrust::raw_pointer_cast(fix.queryOutput.data()));
    }
    setThroughputCounters(state, fix.filter_bits, fix.filterMemory, fix.numKmers, 1.0);
}

REGISTER_BENCHMARK_THROUGHPUT_FASTX(CuSbfHostFixture, Insert);
REGISTER_BENCHMARK_THROUGHPUT_FASTX(CuSbfHostFixture, Query);
REGISTER_BENCHMARK_THROUGHPUT_FASTX(CuSbfDeviceFixture, Insert);
REGISTER_BENCHMARK_THROUGHPUT_FASTX(CuSbfDeviceFixture, Query);

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
