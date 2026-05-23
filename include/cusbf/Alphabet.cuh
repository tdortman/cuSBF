#pragma once

#include <cuda/std/__bit/integral.h>
#include <cuda_runtime.h>

#include <concepts>
#include <cstdint>

namespace cusbf {

namespace detail {

template <typename T>
consteval uint64_t validByteCount() {
    uint64_t count = 0;
    while (T::validBytes[count] != '\0') {
        ++count;
    }
    return count;
}

/**
 * @brief Recursively tests whether placing the separator byte at any position in an input of valid
 * bytes always results in an invalid encoding. This ensures that the separator cannot be confused
 * with valid symbols when concatenating sequences.
 *
 * @tparam T Alphabet type to test.
 * @param input Buffer to construct input strings for encoding. Must have length at least
 * `T::symbolWidth`.
 * @param separatorPosition Position at which to place the separator byte in the input.
 * @param index Current index being set in the input. Should be called with 0 initially.
 * @return bool True if the separator byte always produces an invalid encoding, false if any
 * combination of valid bytes with the separator produces a valid encoding.
 */
template <typename T>
consteval bool
separatorPositionAlwaysEncodesInvalid(char* input, uint64_t separatorPosition, uint64_t index) {
    if (index == T::symbolWidth) {
        return T::encode(input) == static_cast<uint8_t>(T::invalidSymbol);
    }

    if (index == separatorPosition) {
        input[index] = static_cast<char>(T::separator);
        return separatorPositionAlwaysEncodesInvalid<T>(input, separatorPosition, index + 1);
    }

    for (uint64_t byteIndex = 0; byteIndex < validByteCount<T>(); ++byteIndex) {
        input[index] = T::validBytes[byteIndex];
        if (!separatorPositionAlwaysEncodesInvalid<T>(input, separatorPosition, index + 1)) {
            return false;
        }
    }
    return true;
}

/**
 * @brief Tests that for every position in the input, placing the separator byte at that position
 * always results in an invalid encoding. This is a necessary condition for the separator to
 * function correctly when concatenating sequences, as it prevents the creation of valid symbols
 * that span across sequence boundaries.
 *
 * @tparam T Alphabet type to test.
 * @return bool True if the separator byte always produces an invalid encoding at every position,
 * false if any position allows the separator to be part of a valid encoding.
 */
template <typename T>
consteval bool separatorByteAlwaysEncodesInvalid() {
    for (uint64_t separatorPosition = 0; separatorPosition < T::symbolWidth; ++separatorPosition) {
        char input[T::symbolWidth]{};
        if (!separatorPositionAlwaysEncodesInvalid<T>(input, separatorPosition, 0)) {
            return false;
        }
    }
    return true;
}

}  // namespace detail

/**
 * @brief Concept for alphabet-like types used to encode bytes as symbol indices.
 *
 * A type satisfies `Alphabet` if it provides:
 *
 * - `T::symbolCount`: number of encoded symbols in the alphabet.
 * - `T::symbolWidth`: number of input bytes consumed per encoded symbol.
 * - `T::invalidSymbol`: sentinel value for invalid symbols.
 * - `T::separator`: sentinel value for separators when concatenating sequences.
 * - `T::validBytes`: null-terminated representative valid input bytes.
 * - `T::encode(const char*)`: maps `symbolWidth` input bytes to a symbol index,
 *   or `invalidSymbol` if the input bytes are not valid in the alphabet.
 *
 * @tparam T Alphabet type to validate.
 */
template <typename T>
concept Alphabet = requires(const char* input) {
    { T::symbolCount } -> std::convertible_to<uint64_t>;
    { T::symbolWidth } -> std::convertible_to<uint64_t>;
    { T::invalidSymbol } -> std::convertible_to<uint8_t>;
    { T::separator } -> std::convertible_to<uint8_t>;
    { T::validBytes } -> std::convertible_to<const char*>;
    { T::encode(input) } -> std::same_as<uint8_t>;
} && requires {
    requires T::symbolCount > 0 && T::symbolCount <= 255;
    requires T::symbolWidth > 0;
    requires detail::validByteCount<T>() > 0;
    requires detail::separatorByteAlwaysEncodesInvalid<T>();
};

/**
 * @brief An alphabet for encoding DNA sequences, consisting of the symbols A, C, G, and T.
 * Each symbol is encoded as a 2-bit value: A=0, C=1, T=2, G=3. Invalid bytes are encoded as 0xFF.
 */
struct DnaAlphabet {
    /// Input bytes per encoded symbol.
    static constexpr uint64_t symbolWidth = 1;
    /// Number of valid symbols (A, C, G, T).
    static constexpr uint64_t symbolCount = 4;
    /// Sentinel returned by @ref encode for invalid input bytes.
    static constexpr uint8_t invalidSymbol = 0xFFu;
    /// Byte value used between concatenated records (must not encode as a valid symbol).
    static constexpr uint8_t separator = 'N';
    /// Representative valid bytes for compile-time separator checks.
    static constexpr char validBytes[] = "ACGT";

    /**
     * @brief Maps one byte to a 2-bit symbol index, or @ref invalidSymbol.
     *
     * Accepts A/C/G/T in either case.
     *
     * @param input Pointer to one raw byte.
     */
    [[nodiscard]] constexpr __host__ __device__ __forceinline__ static uint8_t encode(
        const char* input
    ) {
        const auto byte = static_cast<uint8_t>(input[0]);
        const uint8_t upper = byte & 0xDFu;   // force upper for validation only
        const uint8_t x = (byte >> 1u) & 3u;  // A=0, C=1, T=2, G=3
        const uint8_t valid = (upper == 'A') | (upper == 'C') | (upper == 'G') | (upper == 'T');
        const uint8_t mask = -valid;
        return (x & mask) | (invalidSymbol & ~mask);
    }
};

/**
 * @brief An alphabet that encodes non-overlapping DNA triplets as single symbols.
 *
 * Each triplet is encoded as a 6-bit value. Invalid bytes in any triplet position
 * produce invalidSymbol.
 */
struct DnaTripletAlphabet {
    /// Input bytes per encoded symbol (one DNA triplet).
    static constexpr uint64_t symbolWidth = 3;
    /// Number of packed triplet symbols.
    static constexpr uint64_t symbolCount = 64;
    /// Sentinel returned by @ref encode for invalid triplets.
    static constexpr uint8_t invalidSymbol = 0xFFu;
    /// Separator byte between concatenated records.
    static constexpr uint8_t separator = 'N';
    /// Representative valid bytes for compile-time separator checks.
    static constexpr char validBytes[] = "ACGT";

    /**
     * @brief Maps three DNA bytes to a 6-bit triplet index, or @ref invalidSymbol.
     *
     * @param input Pointer to three raw bytes.
     */
    [[nodiscard]] constexpr __host__ __device__ __forceinline__ static uint8_t encode(
        const char* input
    ) {
        const uint8_t a = DnaAlphabet::encode(input + 0);
        const uint8_t b = DnaAlphabet::encode(input + 1);
        const uint8_t c = DnaAlphabet::encode(input + 2);
        const uint8_t valid = (a != invalidSymbol) & (b != invalidSymbol) & (c != invalidSymbol);
        const uint8_t packed = (a << 4u) | (b << 2u) | c;
        const uint8_t mask = -valid;
        return (packed & mask) | (invalidSymbol & ~mask);
    }
};

/**
 * @brief An alphabet for encoding protein sequences, consisting of the 20 standard amino acids
 * plus common ambiguous and rare residue symbols:
 *
 *  A through Z.
 *
 *  Each symbol is encoded as a unique 5-bit value from 0 to 25. Invalid bytes are encoded as
 *  0xFF.
 */
struct ProteinAlphabet {
    /// Input bytes per encoded symbol.
    static constexpr uint64_t symbolWidth = 1;
    /// Number of letter symbols (A–Z).
    static constexpr uint64_t symbolCount = 26;
    /// Sentinel returned by @ref encode for invalid input bytes.
    static constexpr uint8_t invalidSymbol = 0xFFu;
    /// Separator byte between concatenated records.
    static constexpr uint8_t separator = '*';
    /// Representative valid bytes for compile-time separator checks.
    static constexpr char validBytes[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";

    /**
     * @brief Maps one amino-acid letter to a 5-bit index, or @ref invalidSymbol.
     *
     * Accepts A–Z in either case.
     *
     * @param input Pointer to one raw byte.
     */
    [[nodiscard]] constexpr __host__ __device__ __forceinline__ static uint8_t encode(
        const char* input
    ) {
        const auto byte = static_cast<uint8_t>(input[0]);
        const uint8_t upper = byte & 0xDFu;
        const uint8_t letterIndex = upper - 'A';
        const uint8_t valid = letterIndex < 26;
        const uint8_t mask = -valid;
        return (letterIndex & mask) | (invalidSymbol & ~mask);
    }
};

}  // namespace cusbf
