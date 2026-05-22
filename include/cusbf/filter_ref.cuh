#pragma once

#include <cuda/std/bit>

#include <cstdint>

#include <cusbf/config.cuh>
#include <cusbf/detail/filter_common.cuh>
#include <cusbf/detail/filter_impl.cuh>
#include <cusbf/detail/sequence_kmer.cuh>
#include <cusbf/device_span.cuh>

namespace cusbf {

/**
 * @brief Non-owning device reference to sectorized filter storage.
 *
 * Trivially copyable; intended to be passed by value into device code.
 */
template <typename Config>
class filter_ref {
   public:
    using block_type = filter_block<Config>;

    __host__ __device__ constexpr filter_ref() noexcept : blocks_{}, num_blocks_{} {
    }

    __host__ __device__ constexpr filter_ref(block_type* blocks, uint64_t num_blocks) noexcept
        : blocks_(blocks), num_blocks_(num_blocks) {
    }

    [[nodiscard]] __host__ __device__ block_type* blocks() const noexcept {
        return blocks_;
    }

    [[nodiscard]] __host__ __device__ uint64_t num_blocks() const noexcept {
        return num_blocks_;
    }

    [[nodiscard]] __host__ __device__ static uint64_t
    shard_index(uint64_t minimizer_hash, uint64_t num_blocks) noexcept {
        return minimizer_hash & (num_blocks - 1);
    }

    [[nodiscard]] __device__ uint64_t shard_index(uint64_t minimizer_hash) const noexcept {
        return shard_index(minimizer_hash, num_blocks_);
    }

    [[nodiscard]] __device__ static bool
    sectorized_contains_packed_kmer(uint64_t packed_kmer, const uint64_t* shard_words) {
        bool present = true;
        _Pragma("unroll")
        for (uint64_t smer_offset = 0; smer_offset < Config::findereSpan; ++smer_offset) {
            const uint64_t smer_hash =
                detail::packed_kmer_smer_hash<Config>(packed_kmer, smer_offset);
            detail::forEachHashIndex<Config>(
                [&]<uint64_t HashIndex>(std::integral_constant<uint64_t, HashIndex>) {
                    constexpr uint64_t s = HashIndex % Config::blockWordCount;
                    const uint64_t bit_pos =
                        block_type::template sectorizedBitAddress<HashIndex>(smer_hash);
                    present &= ((shard_words[s] >> bit_pos) & 1) != 0;
                }
            );
        }
        return present;
    }

    __device__ void
    apply_word_masks(block_type& block, uint64_t m0, uint64_t m1, uint64_t m2, uint64_t m3) const {
        if (m0 != 0) {
            detail::atomicOrWord(&block.words[0], m0);
        }
        if (m1 != 0) {
            detail::atomicOrWord(&block.words[1], m1);
        }
        if (m2 != 0) {
            detail::atomicOrWord(&block.words[2], m2);
        }
        if (m3 != 0) {
            detail::atomicOrWord(&block.words[3], m3);
        }
    }

   private:
    block_type* blocks_{};
    uint64_t num_blocks_{};
};

}  // namespace cusbf
