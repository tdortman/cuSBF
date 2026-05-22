#pragma once

#include <cuda/std/bit>

#include <cstdint>

#include <cusbf/config.cuh>
#include <cusbf/detail/filter_common.cuh>

namespace cusbf {

/**
 * @brief One 256-bit filter block stored as an array of Config::blockWordCount words.
 *
 * Each shard is addressed as a unit: a minimizer hash selects the shard,
 * and the s-mer-derived hashes set/test bits within it.
 */
template <typename Config>
struct alignas(32) filter_block {
    static constexpr uint64_t wordCount = Config::blockWordCount;
    static constexpr uint64_t wordBits = Config::wordBits;
    static constexpr int wordBitsLog2 = cuda::std::bit_width(wordBits) - 1;
    static constexpr uint64_t wordMask = (1ULL << wordBitsLog2) - 1;
    static constexpr int hashShift = 64 - wordBitsLog2;
    static constexpr uint64_t sliceWidth = 64 / Config::hashCount;
    static constexpr bool useBitSlicing = sliceWidth >= wordBitsLog2;

    uint64_t words[wordCount];

    template <uint64_t HashIndex>
    [[nodiscard]] constexpr __host__ __device__ static uint64_t sectorizedBitAddress(
        uint64_t baseHash
    ) {
        static_assert(HashIndex < Config::hashCount, "Hash index out of range");
        if constexpr (useBitSlicing) {
            return (baseHash >> (sliceWidth * HashIndex)) & wordMask;
        } else {
            const uint64_t mixed = baseHash * detail::multiplicativeSaltLiteral<HashIndex>();
            return mixed >> hashShift;
        }
    }

    __device__ __forceinline__ static void sectorizedHashToMasks(
        uint64_t baseHash,
        uint64_t& mask0,
        uint64_t& mask1,
        uint64_t& mask2,
        uint64_t& mask3
    ) {
        detail::forEachHashIndex<Config>(
            [&]<uint64_t HashIndex>(std::integral_constant<uint64_t, HashIndex>) {
                constexpr uint64_t s = HashIndex % Config::blockWordCount;
                const uint64_t bit_pos = sectorizedBitAddress<HashIndex>(baseHash);
                const uint64_t bit = uint64_t{1} << bit_pos;
                if constexpr (s == 0) {
                    mask0 |= bit;
                } else if constexpr (s == 1) {
                    mask1 |= bit;
                } else if constexpr (s == 2) {
                    mask2 |= bit;
                } else {
                    mask3 |= bit;
                }
            }
        );
    }
};

}  // namespace cusbf
