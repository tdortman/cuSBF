#pragma once

#include <cstdint>

namespace cusbf::detail {

template <typename Config>
[[nodiscard]] constexpr __host__ __device__ uint64_t record_symbol_count(uint64_t bases) noexcept {
    return bases / Config::symbolWidth;
}

template <typename Config>
[[nodiscard]] constexpr __host__ __device__ uint64_t record_kmer_count(uint64_t bases) noexcept {
    const uint64_t symbols = record_symbol_count<Config>(bases);
    return symbols < Config::k ? 0 : symbols - Config::k + 1;
}

}  // namespace cusbf::detail
