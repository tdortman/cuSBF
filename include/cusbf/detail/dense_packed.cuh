#pragma once

#include <cuda/__cmath/ceil_div.h>

#include <cstdint>

#include <cusbf/config.cuh>
#include <cusbf/device_span.cuh>

namespace cusbf::detail {

/// @brief Encoded symbols stored in each @c uint64_t word for @p Config.
template <typename Config>
[[nodiscard]] constexpr uint64_t dense_packed_symbols_per_word() {
    return 64 / Config::symbolBits;
}

/// @brief Returns the number of @c uint64_t words required for @p num_symbols encoded symbols.
template <typename Config>
[[nodiscard]] constexpr uint64_t dense_packed_word_count(uint64_t num_symbols) {
    return cuda::ceil_div(num_symbols, dense_packed_symbols_per_word<Config>());
}

/// @brief Returns the number of k-mer windows in a dense packed symbol sequence.
template <typename Config>
[[nodiscard]] constexpr uint64_t dense_packed_kmer_count(uint64_t num_symbols) {
    return num_symbols < Config::k ? 0 : (num_symbols - Config::k + 1);
}

/**
 * @brief Device-side view of a dense packed symbol sequence.
 *
 * Layout: word @c w holds symbols @c [S*w, S*w + S - 1] where @c S =
 * @ref dense_packed_symbols_per_word. Symbol @c i occupies bits
 * @c [symbolBits*(i % S), symbolBits*(i % S) + symbolBits - 1] of word @c i/S
 * (LSB = earlier symbol). Values use @ref Config::Alphabet encoding masked with
 * @ref Config::symbolMask. Trailing bits in the final word beyond
 * @c symbolBits*num_symbols are ignored.
 *
 * Adjacent k-mers overlap in the same words; kernels decode a per-block symbol tile
 * and slide packed k-mers like the byte @ref SequenceKmerInput path.
 */
template <typename Config>
struct DensePackedKmerInput {
    /// Dense packed words (device memory).
    device_span<const uint64_t> words{};
    /// Number of valid encoded symbols represented in @ref words.
    uint64_t num_symbols{};

    /// @brief Number of k-mer windows in this sequence.
    [[nodiscard]] constexpr __host__ __device__ uint64_t kmerCount() const {
        return dense_packed_kmer_count<Config>(num_symbols);
    }
};

/// @brief Decodes one packed symbol at global index @p symbol_index.
template <typename Config>
[[nodiscard]] __device__ __forceinline__ uint8_t
dense_packed_symbol_at(const uint64_t* words, uint64_t symbol_index) {
    constexpr uint64_t symbols_per_word = dense_packed_symbols_per_word<Config>();
    const uint64_t word_index = symbol_index / symbols_per_word;
    const auto bit_offset =
        static_cast<unsigned>((symbol_index % symbols_per_word) * Config::symbolBits);
    return static_cast<uint8_t>((words[word_index] >> bit_offset) & Config::symbolMask);
}

/**
 * @brief Fills @p sequence_tile with encoded symbols for k-mers starting at @p block_start_kmer.
 *
 * @return @c true when every symbol in the tile is valid.
 */
template <typename Config>
__device__ __forceinline__ bool prepare_dense_packed_tiles(
    const uint64_t* words,
    uint64_t block_start_kmer,
    uint64_t block_kmers,
    uint8_t* sequence_tile
) {
    const uint64_t tile_symbols = block_kmers + Config::k - 1;

    bool local_invalid_symbol = false;
    for (uint64_t idx = threadIdx.x; idx < tile_symbols; idx += Config::cudaBlockSize) {
        const uint8_t encoded = dense_packed_symbol_at<Config>(words, block_start_kmer + idx);
        sequence_tile[idx] = encoded;
        local_invalid_symbol |= (encoded == Config::Alphabet::invalidSymbol);
    }
    return __syncthreads_count(local_invalid_symbol) == 0;
}

}  // namespace cusbf::detail
