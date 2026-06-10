#pragma once

#include <cuda/__cmath/ceil_div.h>
#include <cuda_runtime.h>

#include <thrust/count.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>

#include <cstdint>

#include <cusbf/config.cuh>
#include <cusbf/detail/query_layout.cuh>
#include <cusbf/device_span.cuh>
#include <cusbf/error.hpp>
#include <cusbf/helpers.cuh>

namespace cusbf::detail {

/// @brief Per-record kernel: sums @c hits[hit_offset ..] for each @ref QueryLayoutRecord.
template <typename Config>
__global__ void count_positive_kmers_per_record_kernel(
    const uint8_t* hits,
    const QueryLayoutRecord* records,
    uint64_t* positive_kmers_out,
    uint64_t record_count
) {
    const uint64_t record_index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (record_index >= record_count) {
        return;
    }

    const QueryLayoutRecord& record = records[record_index];
    if (record.hit_count == 0) {
        positive_kmers_out[record_index] = 0;
        return;
    }

    uint64_t positive = 0;
    const uint64_t begin = record.hit_offset;
    for (uint64_t i = 0; i < record.hit_count; ++i) {
        positive += hits[begin + i];
    }
    positive_kmers_out[record_index] = positive;
}

/// @brief Device-wide count of set bits in a per-k-mer hit buffer.
template <typename Config>
[[nodiscard]] inline uint64_t
count_positive_kmers_total(device_span<const uint8_t> hits, cuda::stream_ref stream) {
    if (hits.empty()) {
        return 0;
    }

    const auto execution = thrust::cuda::par.on(stream.get());
    return static_cast<uint64_t>(thrust::count(
        execution,
        thrust::device_pointer_cast(hits.data()),
        thrust::device_pointer_cast(hits.data()) + hits.size(),
        uint8_t{1}
    ));
}

/**
 * @brief Fills @p positive_kmers_out with per-record positive k-mer counts.
 *
 * @p positive_kmers_out must hold at least @p records.size() elements.
 */
template <typename Config>
[[nodiscard]] inline Result<void> count_positive_kmers_per_record(
    device_span<const uint8_t> hits,
    device_span<const QueryLayoutRecord> records,
    device_span<uint64_t> positive_kmers_out,
    cuda::stream_ref stream
) {
    if (records.empty()) {
        return {};
    }
    if (positive_kmers_out.size() < records.size()) {
        return Err(Error::invalid_argument("positive k-mer output buffer is too small"));
    }

    const uint32_t block_size = 256;
    const uint32_t grid_size = cuda::ceil_div(records.size(), static_cast<uint64_t>(block_size));
    count_positive_kmers_per_record_kernel<Config><<<grid_size, block_size, 0, stream.get()>>>(
        hits.data(), records.data(), positive_kmers_out.data(), records.size()
    );
    CUSBF_CUDA_TRY(cudaGetLastError());
    return {};
}

}  // namespace cusbf::detail
