#pragma once

#include <charconv>
#include <cstdint>
#include <cstdlib>
#include <string_view>

namespace cusbf::detail {

/**
 * @brief Parses a decimal mebibyte count from @p value.
 *
 * Accepts a leading decimal prefix (same spirit as @c strtoull). Returns @c 0 when @p value is
 * empty or does not start with digits.
 */
[[nodiscard]] inline uint64_t parse_env_mebibytes(std::string_view value) {
    if (value.empty()) {
        return 0;
    }

    uint64_t mebibytes = 0;
    const auto* begin = value.data();
    const auto* end = begin + value.size();
    const auto [ptr, ec] = std::from_chars(begin, end, mebibytes);
    if (ec != std::errc{} || ptr == begin) {
        return 0;
    }
    return mebibytes;
}

/// @brief Reads @p env_name via @c getenv, or an empty view when unset.
[[nodiscard]] inline std::string_view getenv_value(const char* env_name) {
    const char* value = std::getenv(env_name);
    if (value == nullptr || value[0] == '\0') {
        return {};
    }
    return value;
}

}  // namespace cusbf::detail
