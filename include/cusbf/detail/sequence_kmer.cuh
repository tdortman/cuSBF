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

template <typename Config>
struct SequenceKmerInput {
    device_span<const char> sequence;

    [[nodiscard]] constexpr __host__ __device__ uint64_t kmerCount() const {
        const uint64_t symbols = sequence.size() / Config::symbolWidth;
        return symbols < Config::k ? 0 : (symbols - Config::k + 1);
    }

    [[nodiscard]] constexpr __host__ __device__ uint64_t smerCount() const {
        const uint64_t symbols = sequence.size() / Config::symbolWidth;
        return symbols < Config::s ? 0 : (symbols - Config::s + 1);
    }
};

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

template <typename Config>
[[nodiscard]] __device__ __forceinline__ uint64_t
packed_kmer_smer_hash(uint64_t packed_kmer, uint64_t start) {
    const uint64_t packed_smer =
        extractPackedSubwindow<Config, Config::s, Config::k>(packed_kmer, start);
    return hash64(packed_smer);
}

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

template <typename Config, uint64_t K>
__device__ __forceinline__ uint64_t pack_kmer_from_tile(const uint8_t* tile, uint64_t start) {
    uint64_t packed = 0;
    _Pragma("unroll")
    for (uint64_t i = 0; i < K; ++i) {
        packed = (packed << Config::symbolBits) | (tile[start + i] & Config::symbolMask);
    }
    return packed;
}

template <typename Config, uint64_t K>
__device__ __forceinline__ uint64_t advance_packed_kmer(uint64_t packed, uint8_t new_base) {
    return ((packed << Config::symbolBits) | (new_base & Config::symbolMask)) &
           packedWindowMask<Config, K>();
}

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

}  // namespace cusbf::detail
