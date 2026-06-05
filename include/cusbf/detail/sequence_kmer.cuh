#pragma once

#include <cuda_runtime.h>

#include <cub/warp/warp_reduce.cuh>

#include <cstdint>

#include <cusbf/config.cuh>
#include <cusbf/detail/filter_common.cuh>
#include <cusbf/detail/filter_impl.cuh>
#include <cusbf/device_span.cuh>
#include <cusbf/hashutil.cuh>
#include <cusbf/helpers.cuh>

namespace cusbf::detail {

/**
 * @brief Device-side view of an encoded sequence for k-mer / s-mer counting.
 */
template <typename Config>
struct SequenceKmerInput {
    /// Encoded sequence bytes (@ref Config::symbolWidth per symbol).
    device_span<const char> sequence;

    /// @brief Number of k-mer windows in @ref sequence.
    [[nodiscard]] constexpr __host__ __device__ uint64_t kmerCount() const {
        const uint64_t symbols = sequence.size() / Config::symbolWidth;
        return symbols < Config::k ? 0 : (symbols - Config::k + 1);
    }

    /// @brief Number of s-mer windows in @ref sequence.
    [[nodiscard]] constexpr __host__ __device__ uint64_t smerCount() const {
        const uint64_t symbols = sequence.size() / Config::symbolWidth;
        return symbols < Config::s ? 0 : (symbols - Config::s + 1);
    }
};

/// @brief Minimum minimizer hash over all @c m-mers in a packed k-mer.
template <typename Config>
[[nodiscard]] __device__ __forceinline__ uint64_t packed_kmer_minimizer_hash(uint64_t packed_kmer) {
    uint64_t minimizer_hash = kInvalidHash;
    _Pragma("unroll")
    for (uint64_t offset = 0; offset < Config::minimizerSpan; ++offset) {
        const uint64_t packed_mmer =
            extractPackedSubwindow<Config, Config::m, Config::k>(packed_kmer, offset);
        minimizer_hash = min(minimizer_hash, minimizer_hash64(packed_mmer));
    }
    return minimizer_hash;
}

/// @brief Bloom hash for the s-mer at @p start within a packed k-mer.
template <typename Config>
[[nodiscard]] __device__ __forceinline__ uint64_t
packed_kmer_smer_hash(uint64_t packed_kmer, uint64_t start) {
    const uint64_t packed_smer =
        extractPackedSubwindow<Config, Config::s, Config::k>(packed_kmer, start);
    return hash64(packed_smer);
}

/// @brief Loads four 64-bit shard words with 256-bit (sm_100+) or 128-bit vector loads.
template <typename Config>
__device__ __forceinline__ void
load_shard_words4(const filter_block<Config>* shards, uint64_t shard_index, uint64_t* w) {
#if __CUDA_ARCH__ >= 1000
    load256BitGlobalNC(shards[shard_index].words, w[0], w[1], w[2], w[3]);
#else
    load128BitGlobalNC(shards[shard_index].words + 0, w[0], w[1]);
    load128BitGlobalNC(shards[shard_index].words + 2, w[2], w[3]);
#endif
}

/// @brief Packs @c K encoded symbols from a shared-memory tile starting at @p start.
template <typename Config, uint64_t K>
__device__ __forceinline__ uint64_t pack_kmer_from_tile(const uint8_t* tile, uint64_t start) {
    uint64_t packed = 0;
    _Pragma("unroll")
    for (uint64_t i = 0; i < K; ++i) {
        packed = (packed << Config::symbolBits) | (tile[start + i] & Config::symbolMask);
    }
    return packed;
}

/// @brief Slides a packed k-mer window by one encoded base.
template <typename Config, uint64_t K>
__device__ __forceinline__ uint64_t advance_packed_kmer(uint64_t packed, uint8_t new_base) {
    return ((packed << Config::symbolBits) | (new_base & Config::symbolMask)) &
           packedWindowMask<Config, K>();
}

/// @brief True when no symbol in the k-mer window is the alphabet invalid sentinel.
template <typename Config>
__device__ __forceinline__ bool kmer_is_valid(const uint8_t* tile, uint64_t start) {
    _Pragma("unroll")
    for (uint64_t i = 0; i < Config::k; ++i) {
        if (tile[start + i] == Config::Alphabet::invalidSymbol) {
            return false;
        }
    }
    return true;
}

/**
 * @brief Encodes a block's sequence slice into @p sequence_tile and reports global validity.
 *
 * @return @c true when every encoded base in the tile is valid.
 */
template <typename Config>
__device__ __forceinline__ bool prepare_sequence_hash_tiles(
    const char* sequence,
    uint64_t block_start_kmer,
    uint64_t block_kmers,
    uint8_t* sequence_tile
) {
    const uint64_t tile_bases = block_kmers + Config::k - 1;

    bool local_invalid_base = false;
    for (uint64_t idx = threadIdx.x; idx < tile_bases; idx += Config::cudaBlockSize) {
        const uint8_t encoded_base =
            Config::Alphabet::encode(sequence + (block_start_kmer + idx) * Config::symbolWidth);
        sequence_tile[idx] = encoded_base;
        local_invalid_base |= (encoded_base == Config::Alphabet::invalidSymbol);
    }
    return __syncthreads_count(local_invalid_base) == 0;
}

/// @brief Builds the per-thread validity bitmask for strided query kernels.
template <uint32_t k_stride, typename Config>
__device__ __forceinline__ uint32_t build_stride_kmer_valid_mask(
    uint64_t thread_offset,
    uint64_t block_kmers,
    bool block_all_valid,
    const uint8_t* sequence_tile
) {
    uint32_t kmer_valid_mask = 0;
    _Pragma("unroll")
    for (uint32_t s = 0; s < k_stride; ++s) {
        if ((thread_offset + s) < block_kmers) {
            kmer_valid_mask |= (1u << s);
        }
    }

    if (!block_all_valid) {
        _Pragma("unroll")
        for (uint32_t s = 0; s < k_stride; ++s) {
            if (!(kmer_valid_mask & (1u << s))) {
                continue;
            }
            const uint64_t local_idx = thread_offset + s;
            if (!kmer_is_valid<Config>(sequence_tile, local_idx)) {
                kmer_valid_mask &= ~(1u << s);
            }
        }
    }
    return kmer_valid_mask;
}

}  // namespace cusbf::detail
