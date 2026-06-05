#include <benchmark/benchmark.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <cuda/std/bit>

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <cusbf/dense_packed.hpp>
#include <cusbf/device_span.cuh>
#include <cusbf/filter.cuh>
#include <cusbf/helpers.cuh>

#include "benchmark_common.cuh"

namespace bm = benchmark;

using DensePackedConfig = cusbf::Config<31, 28, 16, 4, 256>;

struct DensePackedThroughputInputs {
    uint64_t num_symbols{};
    uint64_t num_kmers{};
    uint64_t filter_memory{};
    thrust::device_vector<char> d_sequence;
    thrust::device_vector<uint64_t> d_dense_words;
    std::vector<char> h_sequence;
    std::vector<uint64_t> h_dense_words;
    thrust::device_vector<uint8_t> query_output;
    std::unique_ptr<cusbf::filter<DensePackedConfig>> filter;
};

static void
prepareDensePackedThroughputInputs(DensePackedThroughputInputs& inputs, uint64_t num_symbols) {
    inputs.num_symbols = num_symbols;
    inputs.num_kmers =
        num_symbols >= DensePackedConfig::k ? num_symbols - DensePackedConfig::k + 1 : 0;
    if (inputs.num_kmers == 0) {
        return;
    }

    const uint64_t filter_bits = cuda::std::bit_ceil(
        std::max(inputs.num_kmers, uint64_t{1}) * benchmark_common::g_fastxBitsPerItem
    );
    inputs.filter = std::make_unique<cusbf::filter<DensePackedConfig>>(filter_bits);
    inputs.filter_memory = inputs.filter->filter_bits() / 8;

    benchmark_common::gpuGenerateDna(inputs.d_sequence, num_symbols);

    inputs.h_sequence.resize(num_symbols);
    CUSBF_CUDA_CALL(cudaMemcpy(
        inputs.h_sequence.data(),
        thrust::raw_pointer_cast(inputs.d_sequence.data()),
        num_symbols,
        cudaMemcpyDeviceToHost
    ));

    const uint64_t word_count = cusbf::dense_packed_word_count<DensePackedConfig>(num_symbols);
    inputs.d_dense_words.resize(word_count);
    cusbf::pack_dense_sequence_device<DensePackedConfig>(
        thrust::raw_pointer_cast(inputs.d_sequence.data()),
        num_symbols,
        thrust::raw_pointer_cast(inputs.d_dense_words.data())
    );

    inputs.h_dense_words = cusbf::pack_dense_sequence<DensePackedConfig>(
        std::string_view{inputs.h_sequence.data(), inputs.h_sequence.size()}
    );

    inputs.query_output.resize(inputs.num_kmers);
    CUSBF_CUDA_CALL(cudaDeviceSynchronize());
}

static void setThroughputCounters(bm::State& state, const DensePackedThroughputInputs& inputs) {
    benchmark_common::filter_benchmark::setFilterBenchmarkCounters(
        state, inputs.filter_memory, inputs.num_kmers
    );
    state.counters["num_symbols"] = bm::Counter(static_cast<double>(inputs.num_symbols));
}

class DensePackedDeviceThroughputFixture : public bm::Fixture {
    using bm::Fixture::SetUp;
    using bm::Fixture::TearDown;

   public:
    void SetUp(const bm::State& state) override {
        prepareDensePackedThroughputInputs(inputs_, static_cast<uint64_t>(state.range(0)));
    }

    void TearDown(const bm::State&) override {
        inputs_ = {};
    }

    DensePackedThroughputInputs inputs_;
    benchmark_common::GPUTimer timer;
};

class DensePackedHostThroughputFixture : public bm::Fixture {
    using bm::Fixture::SetUp;
    using bm::Fixture::TearDown;

   public:
    void SetUp(const bm::State& state) override {
        prepareDensePackedThroughputInputs(inputs_, static_cast<uint64_t>(state.range(0)));
    }

    void TearDown(const bm::State&) override {
        inputs_ = {};
    }

    DensePackedThroughputInputs inputs_;
    benchmark_common::CPUTimer timer;
};

BENCHMARK_DEFINE_F(DensePackedDeviceThroughputFixture, ByteInsert)(bm::State& state) {
    auto& fix = *static_cast<DensePackedDeviceThroughputFixture*>(this);
    if (fix.inputs_.num_kmers == 0) {
        state.SkipWithError("sequence shorter than k");
        return;
    }

    const cusbf::device_span<const char> insert_span{
        thrust::raw_pointer_cast(fix.inputs_.d_sequence.data()),
        fix.inputs_.num_symbols,
    };

    for (auto _ : state) {
        CUSBF_UNWRAP(fix.inputs_.filter->clear());
        CUSBF_CUDA_CALL(cudaDeviceSynchronize());
        fix.timer.start();
        benchmark::DoNotOptimize(fix.inputs_.filter->insert_sequence_async(insert_span));
        state.SetIterationTime(fix.timer.elapsed());
    }
    setThroughputCounters(state, fix.inputs_);
}

BENCHMARK_DEFINE_F(DensePackedDeviceThroughputFixture, DenseInsert)(bm::State& state) {
    auto& fix = *static_cast<DensePackedDeviceThroughputFixture*>(this);
    if (fix.inputs_.num_kmers == 0) {
        state.SkipWithError("sequence shorter than k");
        return;
    }

    const cusbf::device_span<const uint64_t> insert_words{
        thrust::raw_pointer_cast(fix.inputs_.d_dense_words.data()),
        fix.inputs_.d_dense_words.size(),
    };

    for (auto _ : state) {
        CUSBF_UNWRAP(fix.inputs_.filter->clear());
        CUSBF_CUDA_CALL(cudaDeviceSynchronize());
        fix.timer.start();
        benchmark::DoNotOptimize(
            fix.inputs_.filter->insert_dense_packed_async(insert_words, fix.inputs_.num_symbols)
        );
        state.SetIterationTime(fix.timer.elapsed());
    }
    setThroughputCounters(state, fix.inputs_);
}

BENCHMARK_DEFINE_F(DensePackedDeviceThroughputFixture, ByteQuery)(bm::State& state) {
    auto& fix = *static_cast<DensePackedDeviceThroughputFixture*>(this);
    if (fix.inputs_.num_kmers == 0) {
        state.SkipWithError("sequence shorter than k");
        return;
    }

    const cusbf::device_span<const char> query_span{
        thrust::raw_pointer_cast(fix.inputs_.d_sequence.data()),
        fix.inputs_.num_symbols,
    };
    const cusbf::device_span<uint8_t> output_span{
        thrust::raw_pointer_cast(fix.inputs_.query_output.data()),
        fix.inputs_.query_output.size(),
    };

    CUSBF_UNWRAP(fix.inputs_.filter->clear());
    benchmark::DoNotOptimize(fix.inputs_.filter->insert_sequence_async(query_span));
    CUSBF_CUDA_CALL(cudaDeviceSynchronize());

    for (auto _ : state) {
        fix.timer.start();
        cusbf::require_void(fix.inputs_.filter->contains_sequence_async(query_span, output_span));
        state.SetIterationTime(fix.timer.elapsed());
        benchmark::DoNotOptimize(thrust::raw_pointer_cast(fix.inputs_.query_output.data()));
    }
    setThroughputCounters(state, fix.inputs_);
}

BENCHMARK_DEFINE_F(DensePackedDeviceThroughputFixture, DenseQuery)(bm::State& state) {
    auto& fix = *static_cast<DensePackedDeviceThroughputFixture*>(this);
    if (fix.inputs_.num_kmers == 0) {
        state.SkipWithError("sequence shorter than k");
        return;
    }

    const cusbf::device_span<const uint64_t> query_words{
        thrust::raw_pointer_cast(fix.inputs_.d_dense_words.data()),
        fix.inputs_.d_dense_words.size(),
    };
    const cusbf::device_span<uint8_t> output_span{
        thrust::raw_pointer_cast(fix.inputs_.query_output.data()),
        fix.inputs_.query_output.size(),
    };

    CUSBF_UNWRAP(fix.inputs_.filter->clear());
    benchmark::DoNotOptimize(
        fix.inputs_.filter->insert_dense_packed_async(query_words, fix.inputs_.num_symbols)
    );
    CUSBF_CUDA_CALL(cudaDeviceSynchronize());

    for (auto _ : state) {
        fix.timer.start();
        cusbf::require_void(fix.inputs_.filter->contains_dense_packed_async(
            query_words, fix.inputs_.num_symbols, output_span
        ));
        state.SetIterationTime(fix.timer.elapsed());
        benchmark::DoNotOptimize(thrust::raw_pointer_cast(fix.inputs_.query_output.data()));
    }
    setThroughputCounters(state, fix.inputs_);
}

BENCHMARK_DEFINE_F(DensePackedHostThroughputFixture, ByteInsert)(bm::State& state) {
    auto& fix = *static_cast<DensePackedHostThroughputFixture*>(this);
    if (fix.inputs_.num_kmers == 0) {
        state.SkipWithError("sequence shorter than k");
        return;
    }

    const std::string_view sequence{fix.inputs_.h_sequence.data(), fix.inputs_.num_symbols};

    for (auto _ : state) {
        CUSBF_UNWRAP(fix.inputs_.filter->clear());
        fix.timer.start();
        const auto inserted = fix.inputs_.filter->insert_sequence(sequence);
        if (!inserted) {
            state.SkipWithError(inserted.error().message());
            return;
        }
        state.SetIterationTime(fix.timer.elapsed());
        benchmark::DoNotOptimize(uint64_t{*inserted});
    }
    setThroughputCounters(state, fix.inputs_);
}

BENCHMARK_DEFINE_F(DensePackedHostThroughputFixture, DenseInsert)(bm::State& state) {
    auto& fix = *static_cast<DensePackedHostThroughputFixture*>(this);
    if (fix.inputs_.num_kmers == 0) {
        state.SkipWithError("sequence shorter than k");
        return;
    }

    const std::span<const uint64_t> insert_words{fix.inputs_.h_dense_words};

    for (auto _ : state) {
        CUSBF_UNWRAP(fix.inputs_.filter->clear());
        fix.timer.start();
        const auto inserted =
            fix.inputs_.filter->insert_dense_packed(insert_words, fix.inputs_.num_symbols);
        if (!inserted) {
            state.SkipWithError(inserted.error().message());
            return;
        }
        state.SetIterationTime(fix.timer.elapsed());
        benchmark::DoNotOptimize(uint64_t{*inserted});
    }
    setThroughputCounters(state, fix.inputs_);
}

BENCHMARK_DEFINE_F(DensePackedHostThroughputFixture, ByteQuery)(bm::State& state) {
    auto& fix = *static_cast<DensePackedHostThroughputFixture*>(this);
    if (fix.inputs_.num_kmers == 0) {
        state.SkipWithError("sequence shorter than k");
        return;
    }

    const std::string_view sequence{fix.inputs_.h_sequence.data(), fix.inputs_.num_symbols};

    const auto insertKmers = fix.inputs_.filter->insert_sequence(sequence);
    if (!insertKmers) {
        state.SkipWithError(insertKmers.error().message());
        return;
    }
    CUSBF_CUDA_CALL(cudaDeviceSynchronize());

    for (auto _ : state) {
        fix.timer.start();
        const auto queryResults = fix.inputs_.filter->contains_sequence(sequence);
        if (!queryResults) {
            state.SkipWithError(queryResults.error().message());
            return;
        }
        state.SetIterationTime(fix.timer.elapsed());
        benchmark::DoNotOptimize(queryResults->data());
        benchmark::DoNotOptimize(queryResults->size());
    }
    setThroughputCounters(state, fix.inputs_);
}

BENCHMARK_DEFINE_F(DensePackedHostThroughputFixture, DenseQuery)(bm::State& state) {
    auto& fix = *static_cast<DensePackedHostThroughputFixture*>(this);
    if (fix.inputs_.num_kmers == 0) {
        state.SkipWithError("sequence shorter than k");
        return;
    }

    const std::span<const uint64_t> query_words{fix.inputs_.h_dense_words};

    const auto insertKmers =
        fix.inputs_.filter->insert_dense_packed(query_words, fix.inputs_.num_symbols);
    if (!insertKmers) {
        state.SkipWithError(insertKmers.error().message());
        return;
    }
    CUSBF_CUDA_CALL(cudaDeviceSynchronize());

    for (auto _ : state) {
        fix.timer.start();
        const auto queryResults =
            fix.inputs_.filter->contains_dense_packed(query_words, fix.inputs_.num_symbols);
        if (!queryResults) {
            state.SkipWithError(queryResults.error().message());
            return;
        }
        state.SetIterationTime(fix.timer.elapsed());
        benchmark::DoNotOptimize(queryResults->data());
        benchmark::DoNotOptimize(queryResults->size());
    }
    setThroughputCounters(state, fix.inputs_);
}

REGISTER_BENCHMARK(DensePackedDeviceThroughputFixture, ByteInsert);
REGISTER_BENCHMARK(DensePackedDeviceThroughputFixture, DenseInsert);
REGISTER_BENCHMARK(DensePackedDeviceThroughputFixture, ByteQuery);
REGISTER_BENCHMARK(DensePackedDeviceThroughputFixture, DenseQuery);
REGISTER_BENCHMARK(DensePackedHostThroughputFixture, ByteInsert);
REGISTER_BENCHMARK(DensePackedHostThroughputFixture, DenseInsert);
REGISTER_BENCHMARK(DensePackedHostThroughputFixture, ByteQuery);
REGISTER_BENCHMARK(DensePackedHostThroughputFixture, DenseQuery);

STANDARD_BENCHMARK_MAIN()
