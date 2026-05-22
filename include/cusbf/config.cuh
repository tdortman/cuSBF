#pragma once

#include <cuda/std/bit>

#include "Alphabet.cuh"

namespace cusbf {

template <typename Config>
class filter;

/**
 * @brief Compile-time configuration for a cusbf::filter.
 *
 * All filter behaviour (k-mer length, minimizer width, s-mer width, hash
 * count, CUDA block size, and alphabet) is encoded in this struct so that
 * separate configurations produce completely independent Filter types with
 * zero run-time overhead.
 *
 * @tparam K_             k-mer length (1-32).
 * @tparam S_             s-mer width used as the Bloom hash seed (1-k).
 * @tparam M_             minimizer width used for shard selection (1-k).
 * @tparam HashCount_     number of independent Bloom hash functions (1-16, default 4).
 * @tparam CudaBlockSize_ CUDA threads per block (multiple of 32, default 256).
 * @tparam Alphabet_      byte-to-symbol alphabet encoding.
 */
template <
    uint16_t K_,
    uint16_t S_,
    uint16_t M_,
    uint64_t HashCount_ = 4,
    uint64_t CudaBlockSize_ = 256,
    Alphabet Alphabet_ = DnaAlphabet>
struct Config {
    using Alphabet = Alphabet_;

    static constexpr uint16_t k = K_;
    static constexpr uint16_t m = M_;
    static constexpr uint16_t s = S_;
    static constexpr uint64_t hashCount = HashCount_;
    static constexpr uint64_t alphabetSize = Alphabet::symbolCount;
    static constexpr uint64_t symbolWidth = Alphabet::symbolWidth;
    static constexpr uint64_t symbolBits = cuda::std::bit_width(alphabetSize - 1);
    static constexpr uint64_t symbolMask = (uint64_t{1} << symbolBits) - 1;
    static constexpr uint64_t filterBlockBits = 256;
    static constexpr uint64_t cudaBlockSize = CudaBlockSize_;

    static constexpr uint64_t wordBits = 64;
    static constexpr uint64_t blockWordCount = filterBlockBits / wordBits;
    static constexpr uint64_t minimizerSpan = k - m + 1;
    static constexpr uint64_t findereSpan = k - s + 1;
    static constexpr uint64_t insertGroupSize = blockWordCount;
    static constexpr uint64_t queryGroupSize = 1;
    static constexpr uint64_t maxRunKmers = cudaBlockSize;

    static_assert(k > 0, "k must be positive");
    static_assert(symbolWidth > 0, "alphabet symbolWidth must be positive");
    static_assert(m > 0 && m <= k, "m must satisfy 0 < m <= k");
    static_assert(s > 0 && s <= k, "s must satisfy 0 < s <= k");
    static_assert(k * symbolBits <= 64, "k-mer must fit in one packed uint64_t");
    static_assert(m * symbolBits <= 64, "m-mer must fit in one packed uint64_t");
    static_assert(s * symbolBits <= 64, "s-mer must fit in one packed uint64_t");
    static_assert(hashCount > 0, "At least one Bloom hash is required");
    static_assert(hashCount <= 16, "This implementation provides 16 multiplicative salts");
    static_assert(filterBlockBits >= wordBits, "Filter block must contain at least one word");
    static_assert(
        cuda::std::has_single_bit(filterBlockBits),
        "Filter block size must be a power of two"
    );
    static_assert(filterBlockBits % wordBits == 0, "Filter block size must align to the word size");
    static_assert(blockWordCount <= 32, "At most one warp may cooperate on a filter block");
    static_assert(
        cuda::std::has_single_bit(blockWordCount),
        "blockWordCount must be a power of two"
    );
    static_assert(insertGroupSize <= 32, "insertGroupSize must fit in one warp");
    static_assert(queryGroupSize <= 32, "queryGroupSize must fit in one warp");
    static_assert(
        cuda::std::has_single_bit(insertGroupSize),
        "insertGroupSize must be a power of two"
    );
    static_assert(
        cuda::std::has_single_bit(queryGroupSize),
        "queryGroupSize must be a power of two"
    );
    static_assert(
        hashCount >= blockWordCount,
        "Sectorized layout requires hashCount >= blockWordCount"
    );
    static_assert(
        hashCount % blockWordCount == 0,
        "Hash count must distribute evenly across shard words"
    );
    static_assert(cudaBlockSize % 32 == 0, "CUDA block size must be a multiple of one warp");
    static_assert(
        cudaBlockSize % insertGroupSize == 0,
        "cudaBlockSize must divide insertGroupSize"
    );
    static_assert(cudaBlockSize % queryGroupSize == 0, "cudaBlockSize must divide queryGroupSize");
};

}  // namespace cusbf
