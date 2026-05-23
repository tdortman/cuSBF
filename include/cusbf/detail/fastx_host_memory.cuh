#pragma once

#include <cstdlib>
#include <cstring>

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

#if defined(__linux__)
    #include <sys/stat.h>
    #include <unistd.h>
#endif

namespace cusbf::detail {

/// @brief Default headroom left for the OS and other processes when sizing mmap.
inline constexpr size_t kDefaultFastxHostRamSlackBytes = 4u << 30;

[[nodiscard]] inline size_t fastx_host_ram_slack_bytes() {
    const char* value = std::getenv("CUSBF_FASTX_HOST_RAM_SLACK_MB");
    if (value == nullptr || value[0] == '\0') {
        return kDefaultFastxHostRamSlackBytes;
    }

    char* end = nullptr;
    const auto mebibytes = std::strtoull(value, &end, 10);
    if (end == value) {
        return kDefaultFastxHostRamSlackBytes;
    }
    return static_cast<size_t>(mebibytes) << 20;
}

/// @brief Available physical RAM (bytes) for mmap budgeting.
[[nodiscard]] inline size_t query_available_host_bytes() {
#if defined(__linux__)
    const long page_size = ::sysconf(_SC_PAGESIZE);
    const long avail_pages = ::sysconf(_SC_AVPHYS_PAGES);
    if (page_size > 0 && avail_pages > 0) {
        return static_cast<size_t>(page_size) * static_cast<size_t>(avail_pages);
    }
#endif
    return 0;
}

/// @brief Upper bound on file bytes that may be mmap'd (env cap and available RAM minus slack).
[[nodiscard]] inline uint64_t fastx_memory_map_max_bytes() {
    const char* value = std::getenv("CUSBF_FASTX_MMAP_MAX_MB");
    uint64_t cap_bytes = UINT64_MAX;
    if (value != nullptr && value[0] != '\0') {
        char* end = nullptr;
        const auto mebibytes = std::strtoull(value, &end, 10);
        if (end != value) {
            cap_bytes = mebibytes << 20;
        }
    }

    const size_t available = query_available_host_bytes();
    if (available == 0) {
        return cap_bytes;
    }

    const size_t slack = fastx_host_ram_slack_bytes();
    const size_t ram_budget = available > slack ? available - slack : size_t{0};
    return std::min(cap_bytes, static_cast<uint64_t>(ram_budget));
}

/// @brief True when uncompressed @p path size is within @ref fastx_memory_map_max_bytes.
[[nodiscard]] inline bool fastx_file_fits_in_memory(std::string_view path) {
#if defined(__linux__)
    const std::string path_string(path);
    struct stat file_status{};
    if (::stat(path_string.c_str(), &file_status) != 0 || file_status.st_size < 0) {
        return false;
    }
    const auto file_bytes = static_cast<uint64_t>(file_status.st_size);
    return file_bytes <= fastx_memory_map_max_bytes();
#else
    (void)path;
    return false;
#endif
}

}  // namespace cusbf::detail
