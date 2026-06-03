#pragma once

#include <cstdint>
#include <string_view>
#include <vector>

#include <cusbf/config.cuh>
#include <cusbf/detail/dense_packed.cuh>

namespace cusbf {

/// @copydoc cusbf::detail::dense_packed_word_count
template <typename Config>
[[nodiscard]] constexpr uint64_t dense_packed_word_count(uint64_t num_symbols) {
    return detail::dense_packed_word_count<Config>(num_symbols);
}

/// @copydoc cusbf::detail::dense_packed_kmer_count
template <typename Config>
[[nodiscard]] constexpr uint64_t dense_packed_kmer_count(uint64_t num_symbols) {
    return detail::dense_packed_kmer_count<Config>(num_symbols);
}

/**
 * @brief Packs an ASCII sequence into dense @c symbolBits-wide @c uint64_t words.
 *
 * Reads one symbol every @ref Config::symbolWidth input bytes via @ref Config::Alphabet.
 * Trailing input shorter than one symbol is ignored. Invalid bytes are stored as their
 * low @ref Config::symbolBits (same as masking @ref Config::Alphabet::invalidSymbol).
 */
template <typename Config>
[[nodiscard]] std::vector<uint64_t> pack_dense_sequence(std::string_view sequence) {
    const uint64_t num_symbols = sequence.size() / Config::symbolWidth;
    std::vector<uint64_t> words(dense_packed_word_count<Config>(num_symbols), 0);
    constexpr uint64_t symbols_per_word = detail::dense_packed_symbols_per_word<Config>();
    for (uint64_t symbol_index = 0; symbol_index < num_symbols; ++symbol_index) {
        const uint8_t symbol =
            Config::Alphabet::encode(sequence.data() + symbol_index * Config::symbolWidth);
        const uint64_t word_index = symbol_index / symbols_per_word;
        const auto bit_offset =
            static_cast<unsigned>((symbol_index % symbols_per_word) * Config::symbolBits);
        words[word_index] |= (static_cast<uint64_t>(symbol & Config::symbolMask) << bit_offset);
    }
    return words;
}

}  // namespace cusbf
