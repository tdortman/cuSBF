#pragma once

#include <cstddef>
#include <cstdint>

extern "C" {

/**
 * @brief Creates a SuperBloom filter with the given configuration.
 *
 * @return Opaque handle, or @c NULL on failure.
 */
void* superbloom_create(
    uint16_t k,
    uint16_t m,
    uint16_t s,
    size_t n_hashes,
    uint8_t bit_vector_size_exponent,
    uint8_t block_size_exponent
);

/**
 * @brief Inserts a raw DNA sequence into the filter (mutable mode).
 *
 * @return Number of k-mers added, or @c -1 on error.
 */
int64_t superbloom_insert_sequence(void* handle, const uint8_t* seq, size_t len);

/**
 * @brief Inserts every FASTA/FASTQ record in a file into the filter (mutable mode).
 *
 * @return Total k-mers added, or @c -1 on error.
 */
int64_t superbloom_insert_fastx_path(void* handle, const char* path);

/**
 * @brief Queries a raw DNA sequence (freezes the filter first if needed).
 *
 * @return Number of k-mers reported present, or @c -1 on error.
 */
int64_t superbloom_query_sequence(const void* handle, const uint8_t* seq, size_t len);

/**
 * @brief Queries every FASTA/FASTQ record in a file (freezes first if needed).
 *
 * @return Total positive k-mers, or @c -1 on error.
 */
int64_t superbloom_query_fastx_path(const void* handle, const char* path);

/**
 * @brief Freezes the filter for query-only access.
 *
 * @return @c 0 on success, @c -1 on error.
 */
int32_t superbloom_freeze(const void* handle);

/**
 * @brief Sets the Rayon thread-pool size (call before insert/query).
 *
 * @return @c 0 on success, @c -1 on error.
 */
int32_t superbloom_set_threads(void* handle, size_t n);

/** @brief Total filter bits (@c 2^bit_vector_size_exponent from create). */
uint64_t superbloom_filter_bits(const void* handle);

/** @brief Destroys a handle from @ref superbloom_create. */
void superbloom_destroy(void* handle);
}
