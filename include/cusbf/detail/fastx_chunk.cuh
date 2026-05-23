#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <format>
#include <stdexcept>
#include <string_view>

#include <cusbf/config.cuh>
#include <cusbf/detail/fastx_host_limits.cuh>
#include <cusbf/error.hpp>

#if defined(__linux__)
    #include <sys/stat.h>
#endif

namespace cusbf::detail {

/// @brief FASTX mode used to size GPU staging buffers.
enum class fastx_chunk_mode { insert, query };

/// @brief Driver-reported free device memory (bytes available for new allocations).
struct cuda_free_memory {
    /// Bytes reported free by @c cudaMemGetInfo.
    size_t free_bytes{};
};

/// @brief Queries current device free memory via @c cudaMemGetInfo.
[[nodiscard]] inline Result<cuda_free_memory> query_cuda_free_memory() {
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    const cudaError_t error = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (error != cudaSuccess) {
        return cuda::std::unexpected(
            Error::io(std::format("cudaMemGetInfo failed: {}", cudaGetErrorString(error)))
        );
    }
    return cuda_free_memory{free_bytes};
}

/// @brief Reserved device memory left for allocator and kernel temporaries.
[[nodiscard]] constexpr size_t fastx_chunk_slack_bytes() noexcept {
    return 64u << 20;
}

[[nodiscard]] inline uint64_t fastx_file_bytes(const std::filesystem::path& path) {
#if defined(__linux__)
    const std::string path_string = path.string();
    struct stat file_status{};
    if (::stat(path_string.c_str(), &file_status) != 0 || file_status.st_size < 0) {
        return 0;
    }
    return static_cast<uint64_t>(file_status.st_size);
#else
    (void)path;
    return 0;
#endif
}

template <typename Config>
[[nodiscard]] constexpr uint64_t fastx_record_symbol_count(uint64_t bases) noexcept {
    return bases / Config::symbolWidth;
}

template <typename Config>
[[nodiscard]] constexpr uint64_t fastx_record_kmer_count(uint64_t bases) noexcept {
    const uint64_t symbols = fastx_record_symbol_count<Config>(bases);
    return symbols < Config::k ? 0 : symbols - Config::k + 1;
}

/// @brief Upper bound on normalized sequence bytes for a raw host chunk.
template <typename Config>
[[nodiscard]] constexpr uint64_t
estimate_normalized_sequence_bytes(uint64_t raw_bytes, uint64_t record_count) noexcept {
    return raw_bytes + record_count * 2 * Config::symbolWidth;
}

/// @brief Peak device bytes for insert staging (@c d_sequence_) for a host chunk.
template <typename Config>
[[nodiscard]] constexpr uint64_t
estimate_insert_staging_bytes(uint64_t raw_bytes, uint64_t record_count) noexcept {
    return estimate_normalized_sequence_bytes<Config>(raw_bytes, record_count);
}

/// @brief Peak device bytes for query staging (@c d_sequence_ + @c d_resultBuffer_).
template <typename Config>
[[nodiscard]] constexpr uint64_t
estimate_query_staging_bytes(uint64_t raw_bytes, uint64_t record_count) noexcept {
    const uint64_t normalized = estimate_normalized_sequence_bytes<Config>(raw_bytes, record_count);
    return normalized + fastx_record_kmer_count<Config>(normalized);
}

/// @brief Device staging byte budget derived from free VRAM and @p fill_fraction.
template <typename Config>
[[nodiscard]] constexpr size_t
fastx_staging_budget_bytes(double fill_fraction, size_t free_bytes) noexcept {
    if (fill_fraction <= 0.0) {
        return 0;
    }

    const size_t available =
        free_bytes > fastx_chunk_slack_bytes() ? free_bytes - fastx_chunk_slack_bytes() : size_t{0};
    const double budget = static_cast<double>(available) * fill_fraction;
    return budget <= 0.0 ? size_t{0} : static_cast<size_t>(budget);
}

/// @brief Whether estimated device staging meets or exceeds @p staging_budget_bytes.
template <typename Config>
[[nodiscard]] constexpr bool fastx_chunk_reached_staging_budget(
    fastx_chunk_mode mode,
    size_t staging_budget_bytes,
    uint64_t raw_bytes,
    uint64_t record_count
) noexcept {
    if (raw_bytes == 0) {
        return false;
    }
    if (staging_budget_bytes == 0) {
        return true;
    }

    const uint64_t staging = mode == fastx_chunk_mode::insert
                                 ? estimate_insert_staging_bytes<Config>(raw_bytes, record_count)
                                 : estimate_query_staging_bytes<Config>(raw_bytes, record_count);
    return staging >= staging_budget_bytes;
}

/// @brief Per-chunk flush budget for dual-stream ping-pong (two device sequence buffers).
[[nodiscard]] constexpr size_t
fastx_pipelined_chunk_budget(fastx_chunk_mode mode, size_t staging_budget_bytes) noexcept {
    if (staging_budget_bytes == 0) {
        return 0;
    }
    if (mode == fastx_chunk_mode::insert) {
        return staging_budget_bytes / 2;
    }
    return staging_budget_bytes / 3;
}

/// @brief Whether a host chunk should flush based on GPU staging and host byte limits.
template <typename Config>
[[nodiscard]] inline bool fastx_chunk_should_flush(
    fastx_chunk_mode mode,
    size_t gpu_staging_budget_bytes,
    size_t host_chunk_max_bytes,
    uint64_t raw_chunk_bytes,
    uint64_t record_count
) noexcept {
    return fastx_chunk_reached_staging_budget<Config>(
               mode, gpu_staging_budget_bytes, raw_chunk_bytes, record_count
           ) ||
           fastx_chunk_reached_host_byte_limit(host_chunk_max_bytes, raw_chunk_bytes);
}

/// @brief Whether the entire uncompressed file fits in one GPU staging pass.
template <typename Config>
[[nodiscard]] inline bool fastx_file_fits_gpu_staging(
    const std::filesystem::path& path,
    fastx_chunk_mode mode,
    double fill_fraction
) {
    const uint64_t file_bytes = fastx_file_bytes(path);
    if (file_bytes == 0) {
        return true;
    }

    const auto gpu_memory = query_cuda_free_memory();
    if (!gpu_memory) {
        return false;
    }
    const size_t staging_budget_bytes =
        fastx_staging_budget_bytes<Config>(fill_fraction, gpu_memory->free_bytes);
    return !fastx_chunk_reached_staging_budget<Config>(mode, staging_budget_bytes, file_bytes, 1);
}

/// @return Resource error if @p raw_bytes / @p record_count exceed the GPU staging budget.
template <typename Config>
[[nodiscard]] inline Result<void> validate_fastx_staging_fits(
    fastx_chunk_mode mode,
    double fill_fraction,
    uint64_t raw_bytes,
    uint64_t record_count,
    std::string_view source_name
) {
    const auto gpu_memory = query_cuda_free_memory();
    if (!gpu_memory) {
        return cuda::std::unexpected(gpu_memory.error());
    }
    const size_t staging_budget_bytes =
        fastx_staging_budget_bytes<Config>(fill_fraction, gpu_memory->free_bytes);
    if (!fastx_chunk_reached_staging_budget<Config>(
            mode, staging_budget_bytes, raw_bytes, record_count
        )) {
        return {};
    }

    return cuda::std::unexpected(
        Error::resource(
            std::format(
                "{}: FASTX input requires more GPU memory than available at fill_fraction={} "
                "(free staging budget {} bytes)",
                source_name,
                fill_fraction,
                staging_budget_bytes
            )
        )
    );
}

}  // namespace cusbf::detail
