#include <cuda/__cmath/ceil_div.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <CLI/CLI.hpp>

#include <cstdint>
#include <iostream>
#include <memory>
#include <string>

#include <cuckoogpu/CuckooFilter.cuh>
#include <cuco/bloom_filter.cuh>
#include <cusbf/device_span.cuh>
#include <cusbf/filter.cuh>
#include <cusbf/helpers.cuh>

#include "benchmark_common.cuh"

using CuSbfConfig = cusbf::Config<31, 28, 16, 4>;
using CuSbfFilter = cusbf::filter<CuSbfConfig>;
using CucoBloom = cuco::bloom_filter<uint64_t>;
using CuckooGpuConfig = cuckoogpu::Config<uint64_t, 16, 500, 256, 16>;
using CuckooGpuFilter = cuckoogpu::Filter<CuckooGpuConfig>;

constexpr uint64_t kBitsPerItem = 16;

uint64_t cucoNumBlocks(uint64_t numItems) {
    constexpr auto bitsPerWord = sizeof(typename CucoBloom::word_type) * 8;
    return cuda::ceil_div(numItems * kBitsPerItem, CucoBloom::words_per_block * bitsPerWord);
}

struct BenchmarkInput {
    explicit BenchmarkInput(uint64_t numKmers) : sequenceLength(numKmers + CuSbfConfig::k - 1) {
        benchmark_common::gpuGenerateDna(d_sequence, sequenceLength, 42);
        d_packed_kmers.resize(numKmers);
        benchmark_common::gpuEncodePackedKmers<CuSbfConfig::k, cusbf::DnaAlphabet>(
            thrust::raw_pointer_cast(d_sequence.data()),
            sequenceLength,
            thrust::raw_pointer_cast(d_packed_kmers.data())
        );
        CUSBF_CUDA_CALL(cudaDeviceSynchronize());
    }

    uint64_t sequenceLength{};
    thrust::device_vector<char> d_sequence;
    thrust::device_vector<uint64_t> d_packed_kmers;
};

void benchmarkCuSbfInsert(uint64_t capacity, double load_factor) {
    const auto n = static_cast<uint64_t>(capacity * load_factor);
    BenchmarkInput input(n);
    CuSbfFilter filter(cuda::std::bit_ceil(n * kBitsPerItem));

    benchmark::DoNotOptimize(filter.insert_sequence_async(
        cusbf::device_span<const char>{
            thrust::raw_pointer_cast(input.d_sequence.data()), input.sequenceLength
        }
    ));
    CUSBF_CUDA_CALL(cudaDeviceSynchronize());

    filter.clear();
    benchmark::DoNotOptimize(filter.insert_sequence_async(
        cusbf::device_span<const char>{
            thrust::raw_pointer_cast(input.d_sequence.data()), input.sequenceLength
        }
    ));
    CUSBF_CUDA_CALL(cudaDeviceSynchronize());
}

void benchmarkCuSbfQuery(uint64_t capacity, double load_factor) {
    const auto n = static_cast<uint64_t>(capacity * load_factor);
    BenchmarkInput input(n);
    thrust::device_vector<uint8_t> d_output(n);
    CuSbfFilter filter(cuda::std::bit_ceil(n * kBitsPerItem));

    benchmark::DoNotOptimize(filter.insert_sequence_async(
        cusbf::device_span<const char>{
            thrust::raw_pointer_cast(input.d_sequence.data()), input.sequenceLength
        }
    ));
    CUSBF_CUDA_CALL(cudaDeviceSynchronize());

    filter.contains_sequence_async(
        cusbf::device_span<const char>{
            thrust::raw_pointer_cast(input.d_sequence.data()), input.sequenceLength
        },
        cusbf::device_span<uint8_t>{thrust::raw_pointer_cast(d_output.data()), d_output.size()}
    );
    CUSBF_CUDA_CALL(cudaDeviceSynchronize());
    benchmark::DoNotOptimize(thrust::raw_pointer_cast(d_output.data()));
}

void benchmarkCucoBloomInsert(uint64_t capacity, double load_factor) {
    const auto n = static_cast<uint64_t>(capacity * load_factor);
    BenchmarkInput input(n);
    CucoBloom filter(cucoNumBlocks(n));

    filter.add(input.d_packed_kmers.begin(), input.d_packed_kmers.end());
    CUSBF_CUDA_CALL(cudaDeviceSynchronize());

    filter.clear();
    CUSBF_CUDA_CALL(cudaDeviceSynchronize());
    filter.add(input.d_packed_kmers.begin(), input.d_packed_kmers.end());
    CUSBF_CUDA_CALL(cudaDeviceSynchronize());
}

void benchmarkCucoBloomQuery(uint64_t capacity, double load_factor) {
    const auto n = static_cast<uint64_t>(capacity * load_factor);
    BenchmarkInput input(n);
    thrust::device_vector<uint8_t> d_output(n);
    CucoBloom filter(cucoNumBlocks(n));

    filter.add(input.d_packed_kmers.begin(), input.d_packed_kmers.end());
    CUSBF_CUDA_CALL(cudaDeviceSynchronize());

    filter.contains(
        input.d_packed_kmers.begin(),
        input.d_packed_kmers.end(),
        reinterpret_cast<bool*>(thrust::raw_pointer_cast(d_output.data()))
    );
    CUSBF_CUDA_CALL(cudaDeviceSynchronize());
    benchmark::DoNotOptimize(thrust::raw_pointer_cast(d_output.data()));
}

void benchmarkCuckooGpuInsert(uint64_t capacity, double load_factor) {
    const auto n = static_cast<uint64_t>(capacity * load_factor);
    BenchmarkInput input(n);
    CuckooGpuFilter filter(capacity);

    filter.insertMany(input.d_packed_kmers);
    CUSBF_CUDA_CALL(cudaDeviceSynchronize());

    filter.clear();
    CUSBF_CUDA_CALL(cudaDeviceSynchronize());
    filter.insertMany(input.d_packed_kmers);
    CUSBF_CUDA_CALL(cudaDeviceSynchronize());
}

void benchmarkCuckooGpuQuery(uint64_t capacity, double load_factor) {
    const auto n = static_cast<uint64_t>(capacity * load_factor);
    BenchmarkInput input(n);
    thrust::device_vector<uint8_t> d_output(n);
    CuckooGpuFilter filter(capacity);

    filter.insertMany(input.d_packed_kmers);
    CUSBF_CUDA_CALL(cudaDeviceSynchronize());

    filter.containsMany(input.d_packed_kmers, d_output);
    CUSBF_CUDA_CALL(cudaDeviceSynchronize());
    benchmark::DoNotOptimize(thrust::raw_pointer_cast(d_output.data()));
}

int main(int argc, char** argv) {
    CLI::App app{"GPU filter hardware profiler benchmark"};

    std::string filter = "cusbf";
    std::string operation = "insert";
    uint64_t exponent = 24;
    double load_factor = 0.95;

    app.add_option("filter", filter, "Filter type: cusbf, cucobloom, cuckoogpu")
        ->required()
        ->check(CLI::IsMember({"cusbf", "cucobloom", "cuckoogpu"}));
    app.add_option("operation", operation, "Operation: insert, query")
        ->required()
        ->check(CLI::IsMember({"insert", "query"}));
    app.add_option("exponent", exponent, "Exponent for capacity = 2^x")
        ->required()
        ->check(CLI::PositiveNumber);
    app.add_option("-l,--load-factor", load_factor, "Load factor (0.0-1.0)")
        ->default_val(0.95)
        ->check(CLI::Range(0.0, 1.0));

    CLI11_PARSE(app, argc, argv);

    const uint64_t capacity = uint64_t{1} << exponent;
    const auto n = static_cast<uint64_t>(capacity * load_factor);

    std::cout << "Filter: " << filter << '\n';
    std::cout << "Operation: " << operation << '\n';
    std::cout << "Capacity: " << capacity << '\n';
    std::cout << "Load Factor: " << load_factor << '\n';
    std::cout << "Number of keys: " << n << '\n';

    if (filter == "cusbf") {
        if (operation == "insert") {
            benchmarkCuSbfInsert(capacity, load_factor);
        } else {
            benchmarkCuSbfQuery(capacity, load_factor);
        }
    } else if (filter == "cucobloom") {
        if (operation == "insert") {
            benchmarkCucoBloomInsert(capacity, load_factor);
        } else {
            benchmarkCucoBloomQuery(capacity, load_factor);
        }
    } else if (filter == "cuckoogpu") {
        if (operation == "insert") {
            benchmarkCuckooGpuInsert(capacity, load_factor);
        } else {
            benchmarkCuckooGpuQuery(capacity, load_factor);
        }
    }

    return 0;
}
