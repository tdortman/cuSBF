#pragma once

#include <cstdlib>
#include <cstring>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace cusbf::detail {

static_assert(sizeof(size_t) == sizeof(uint64_t), "cuSBF assumes size_t is 64-bit");

[[nodiscard]] inline size_t parse_host_chunk_max_bytes(const char* env_name) {
    const char* value = std::getenv(env_name);
    if (value == nullptr || value[0] == '\0') {
        return 0;
    }

    char* end = nullptr;
    const auto mebibytes = std::strtoull(value, &end, 10);
    if (end == value || mebibytes == 0) {
        return 0;
    }
    return static_cast<size_t>(mebibytes) << 20;
}

/// @brief Optional host assembly byte cap before flush (debug / low-RAM safety valve).
///
/// Returns @c SIZE_MAX when unset so @ref fastx_chunk_should_flush uses GPU staging only.
[[nodiscard]] inline size_t fastx_host_chunk_max_bytes() {
    if (const size_t override_bytes = parse_host_chunk_max_bytes("CUSBF_FASTX_MAX_HOST_CHUNK_MB");
        override_bytes != 0) {
        return override_bytes;
    }

    if (const size_t large_bytes = parse_host_chunk_max_bytes("CUSBF_LARGE_FASTX_HOST_CHUNK_MB");
        large_bytes != 0) {
        return large_bytes;
    }

    return std::numeric_limits<size_t>::max();
}

[[nodiscard]] inline bool fastx_chunk_reached_host_byte_limit(
    size_t host_chunk_max_bytes,
    uint64_t raw_chunk_bytes
) noexcept {
    return host_chunk_max_bytes != std::numeric_limits<size_t>::max() &&
           raw_chunk_bytes >= host_chunk_max_bytes;
}

}  // namespace cusbf::detail
