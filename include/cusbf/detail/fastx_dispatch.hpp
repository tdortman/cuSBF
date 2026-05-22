#pragma once

#include <cstdint>
#include <functional>
#include <istream>
#include <memory>
#include <string>
#include <string_view>

#include <cusbf/detail/fastx_buffer_reader.hpp>
#include <cusbf/detail/fastx_chunk.cuh>
#include <cusbf/detail/fastx_file_buffer.hpp>
#include <cusbf/detail/fastx_host_memory.cuh>
#include <cusbf/Fastx.hpp>
#include <cusbf/gzstreambuf.hpp>

#if defined(__linux__)
    #include <sys/stat.h>
#endif

namespace cusbf::detail {

/// @brief How a FASTX file is read and chunked for GPU processing.
enum class fastx_dispatch_path {
    /// @brief Whole file in one GPU chunk, stream via @c istream (no mmap).
    single_chunk_stream,
    /// @brief Whole file in one GPU chunk, mmap'd when it fits in host RAM.
    single_chunk_mmap,
    /// @brief Multiple GPU chunks, file mmap'd when it fits in host RAM.
    chunked_mmap,
    /// @brief Multiple GPU chunks, stream via @c istream (gzip or larger than RAM).
    chunked_stream,
};

[[nodiscard]] constexpr bool fastx_is_single_chunk_path(fastx_dispatch_path path) noexcept {
    return path == fastx_dispatch_path::single_chunk_stream ||
           path == fastx_dispatch_path::single_chunk_mmap;
}

[[nodiscard]] constexpr bool fastx_uses_mmap_reader(fastx_dispatch_path path) noexcept {
    return path == fastx_dispatch_path::single_chunk_mmap ||
           path == fastx_dispatch_path::chunked_mmap;
}

/// @brief Whether the entire file fits in a single GPU staging chunk at @p fill_fraction.
template <typename Config>
[[nodiscard]] inline bool
fastx_fits_single_gpu_chunk(fastx_chunk_mode mode, double fill_fraction, uint64_t file_bytes) {
    if (file_bytes == 0) {
        return true;
    }

    const auto gpu_memory = query_cuda_free_memory();
    const size_t staging_budget_bytes =
        fastx_staging_budget_bytes<Config>(fill_fraction, gpu_memory.free_bytes);
    if (staging_budget_bytes == 0) {
        return true;
    }

    return !fastx_chunk_reached_staging_budget<Config>(mode, staging_budget_bytes, file_bytes, 1);
}

/// @brief Max raw file size for @ref fastx_dispatch_path::single_chunk_stream (istream, no mmap).
///
/// Larger files that still fit one GPU chunk use @ref fastx_dispatch_path::single_chunk_mmap.
[[nodiscard]] inline uint64_t fastx_single_chunk_stream_max_bytes() {
    constexpr uint64_t kDefaultBytes = 32u << 20;
    const char* value = std::getenv("CUSBF_FASTX_SINGLE_CHUNK_STREAM_MAX_MB");
    if (value == nullptr || value[0] == '\0') {
        return kDefaultBytes;
    }

    char* end = nullptr;
    const auto mebibytes = std::strtoull(value, &end, 10);
    if (end == value) {
        return kDefaultBytes;
    }
    return mebibytes << 20;
}

/// @brief Selects mmap vs stream and single- vs multi-chunk processing from file size.
template <typename Config>
[[nodiscard]] inline fastx_dispatch_path select_fastx_dispatch_path_for_file_bytes(
    uint64_t file_bytes,
    fastx_chunk_mode mode,
    double fill_fraction,
    bool file_fits_in_memory
) {
    if (file_bytes > 0 && fastx_fits_single_gpu_chunk<Config>(mode, fill_fraction, file_bytes) &&
        file_bytes <= fastx_single_chunk_stream_max_bytes()) {
        return fastx_dispatch_path::single_chunk_stream;
    }

    if (file_bytes > 0 && fastx_fits_single_gpu_chunk<Config>(mode, fill_fraction, file_bytes) &&
        file_fits_in_memory) {
        return fastx_dispatch_path::single_chunk_mmap;
    }

    if (file_fits_in_memory) {
        return fastx_dispatch_path::chunked_mmap;
    }

    return fastx_dispatch_path::chunked_stream;
}

/// @brief Selects mmap vs stream and single- vs multi-chunk processing for a path.
template <typename Config>
[[nodiscard]] inline fastx_dispatch_path
select_fastx_dispatch_path(std::string_view path, fastx_chunk_mode mode, double fill_fraction) {
    if (isGzipFile(path)) {
        return fastx_dispatch_path::chunked_stream;
    }

    return select_fastx_dispatch_path_for_file_bytes<Config>(
        fastx_file_bytes(path), mode, fill_fraction, fastx_file_fits_in_memory(path)
    );
}

/// @brief Opens a FASTX path and invokes @p handler with a reader and dispatch path.
///
/// @p handler receives the reader and how the file was opened. Use
/// Small files use @ref fastx_dispatch_path::single_chunk_stream, GPU-sized inputs use
/// @ref fastx_dispatch_path::single_chunk_mmap, larger inputs use pipelined mmap or stream.
template <typename Config, typename Handler>
void dispatch_fastx_file(
    std::string_view path,
    fastx_chunk_mode mode,
    double fill_fraction,
    Handler&& handler
) {
    const fastx_dispatch_path dispatch_path =
        select_fastx_dispatch_path<Config>(path, mode, fill_fraction);

    if (fastx_uses_mmap_reader(dispatch_path)) {
        const auto buffer = FastxFileBuffer::load(path);
        FastxBufferReader reader(buffer->data(), path);
        handler(reader, dispatch_path);
        return;
    }

    const auto input = openFastxFile(path);
    FastxReader reader(*input, path);
    handler(reader, dispatch_path);
}

}  // namespace cusbf::detail
