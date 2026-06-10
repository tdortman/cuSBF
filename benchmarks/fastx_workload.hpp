#pragma once

#include <algorithm>
#include <filesystem>
#include <string_view>
#include <vector>

#include <cuda/__cmath/ceil_div.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <cuda/std/bit>

#include <cusbf/Alphabet.cuh>
#include <cusbf/Fastx.hpp>
#include <cusbf/helpers.cuh>

namespace benchmark_common::fastx_workload {

template <uint64_t K, typename Alphabet = cusbf::DnaAlphabet>
struct PreparedFastxSequence {
    std::vector<char> host_sequence{};
    thrust::device_vector<char> d_sequence{};
    thrust::device_vector<uint64_t> d_packed_kmers{};
    uint64_t kmers{};
    bool packed_ready{};
};

template <uint64_t K, typename Alphabet = cusbf::DnaAlphabet>
[[nodiscard]] constexpr uint64_t sequence_kmer_count(uint64_t sequence_length) noexcept {
    const uint64_t symbols = sequence_length / Alphabet::symbolWidth;
    return symbols < K ? 0 : symbols - K + 1;
}

[[nodiscard]] inline std::vector<char> read_fastx_concatenated(
    const std::filesystem::path& path,
    char separator = cusbf::DnaAlphabet::separator
) {
    auto input = CUSBF_UNWRAP(cusbf::detail::openFastxFile(path));
    cusbf::detail::FastxReader reader(*input, std::string_view{path.native()});

    std::vector<char> sequence;
    cusbf::detail::FastxRecord record;
    bool first = true;
    while (CUSBF_UNWRAP(reader.nextRecord(record))) {
        if (!first) {
            sequence.push_back(separator);
        }
        first = false;
        sequence.insert(sequence.end(), record.sequence.begin(), record.sequence.end());
    }
    return sequence;
}

template <uint64_t K, typename Alphabet = cusbf::DnaAlphabet>
__global__ void encode_packed_kmers_kernel(
    const char* sequence,
    uint64_t kmer_start,
    uint64_t num_kmers,
    uint64_t* output
) {
    constexpr uint64_t symbol_bits = cuda::std::bit_width(Alphabet::symbolCount - 1);
    constexpr uint64_t symbol_mask = (uint64_t{1} << symbol_bits) - 1;
    const uint64_t idx = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= num_kmers) {
        return;
    }

    const uint64_t kmer_index = kmer_start + idx;
    uint64_t packed = 0;
    for (uint64_t i = 0; i < K; ++i) {
        const uint8_t encoded =
            Alphabet::encode(sequence + (kmer_index + i) * Alphabet::symbolWidth);
        packed = (packed << symbol_bits) | (encoded & symbol_mask);
    }
    output[idx] = packed;
}

template <uint64_t K, typename Alphabet = cusbf::DnaAlphabet>
inline void encode_packed_kmers(
    const char* d_sequence,
    uint64_t sequence_length,
    uint64_t* d_output,
    cudaStream_t stream = {},
    uint64_t kmer_start = 0,
    uint64_t num_kmers = 0
) {
    const uint64_t total_kmers = sequence_kmer_count<K, Alphabet>(sequence_length);
    if (kmer_start >= total_kmers) {
        return;
    }

    const uint64_t available = total_kmers - kmer_start;
    const uint64_t encode_count = num_kmers == 0 ? available : std::min(num_kmers, available);
    if (encode_count == 0) {
        return;
    }

    constexpr uint64_t block_size = 256;
    const uint64_t grid_size = cuda::ceil_div(encode_count, block_size);
    encode_packed_kmers_kernel<K, Alphabet>
        <<<grid_size, block_size, 0, stream>>>(d_sequence, kmer_start, encode_count, d_output);
    CUSBF_CUDA_CALL(cudaGetLastError());
}

template <uint64_t K, typename Alphabet = cusbf::DnaAlphabet>
inline void upload_sequence(PreparedFastxSequence<K, Alphabet>& sequence) {
    sequence.d_sequence.resize(sequence.host_sequence.size());
    if (!sequence.host_sequence.empty()) {
        CUSBF_CUDA_CALL(cudaMemcpy(
            thrust::raw_pointer_cast(sequence.d_sequence.data()),
            sequence.host_sequence.data(),
            sequence.host_sequence.size(),
            cudaMemcpyHostToDevice
        ));
    }
}

template <uint64_t K, typename Alphabet = cusbf::DnaAlphabet>
inline void ensure_packed_kmers(PreparedFastxSequence<K, Alphabet>& sequence) {
    if (sequence.packed_ready) {
        return;
    }
    if (sequence.d_sequence.size() != sequence.host_sequence.size()) {
        upload_sequence(sequence);
    }
    sequence.d_packed_kmers.resize(sequence.kmers);
    if (sequence.kmers != 0) {
        encode_packed_kmers<K, Alphabet>(
            thrust::raw_pointer_cast(sequence.d_sequence.data()),
            sequence.host_sequence.size(),
            thrust::raw_pointer_cast(sequence.d_packed_kmers.data())
        );
        CUSBF_CUDA_CALL(cudaDeviceSynchronize());
    }
    sequence.packed_ready = true;
}
template <uint64_t K, typename Alphabet = cusbf::DnaAlphabet>
[[nodiscard]] inline PreparedFastxSequence<K, Alphabet>
load_fastx_sequence(const std::filesystem::path& path, char separator = Alphabet::separator) {
    PreparedFastxSequence<K, Alphabet> sequence;
    sequence.host_sequence = read_fastx_concatenated(path, separator);
    sequence.kmers = sequence_kmer_count<K, Alphabet>(sequence.host_sequence.size());
    return sequence;
}

}  // namespace benchmark_common::fastx_workload
