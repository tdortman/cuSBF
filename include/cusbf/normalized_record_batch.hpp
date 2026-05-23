#pragma once

#include <cuda/std/span>

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <cusbf/config.cuh>
#include <cusbf/error.hpp>
#include <cusbf/Fastx.hpp>

namespace cusbf {

/// @brief Per-record metadata inside a normalized record batch.
struct NormalizedRecord {
    /// Index in the source @ref RecordBatchView.
    uint64_t record_index{};
    /// Byte offset in the dense input sequence.
    uint64_t input_offset{};
    /// Symbol offset in the normalized sequence (k-mer index base).
    uint64_t output_offset{};
    /// Record payload size in bytes after normalization.
    uint64_t size{};
    /// K-mer windows with no invalid symbols in this record.
    uint64_t valid_kmers{};
};

/**
 * @brief Host-resident Normalized record batch ready for bulk sequence insert/query.
 *
 * The dense sequence includes alphabet separators between records. Per-record metadata
 * maps input record ranges to k-mer offsets in the normalized sequence.
 */
class NormalizedRecordBatch {
   public:
    /// @brief Dense normalized sequence (includes inter-record separators).
    [[nodiscard]] const std::string& sequence() const noexcept {
        return sequence_;
    }

    /// @brief Per-record metadata in source order.
    [[nodiscard]] cuda::std::span<const NormalizedRecord> records() const noexcept {
        return cuda::std::span<const NormalizedRecord>{records_.data(), records_.size()};
    }

    /// @brief Sum of @ref NormalizedRecord::valid_kmers across all records.
    [[nodiscard]] uint64_t total_valid_kmers() const noexcept {
        uint64_t total = 0;
        for (const NormalizedRecord& record : records_) {
            total += record.valid_kmers;
        }
        return total;
    }

   private:
    template <typename Config>
    friend Result<NormalizedRecordBatch> normalize_record_batch(RecordBatchView batch);

    template <typename Config>
    [[nodiscard]] static Result<NormalizedRecordBatch> from_batch(RecordBatchView batch) {
        NormalizedRecordBatch prepared;
        CUSBF_TRY(
            normalize_record_batch_into<Config>(batch, prepared.sequence_, prepared.records_)
        );
        return prepared;
    }

    std::string sequence_;
    std::vector<NormalizedRecord> records_;
};

namespace detail {

template <typename Config>
[[nodiscard]] static uint64_t record_symbol_count(uint64_t bases) {
    return bases / Config::symbolWidth;
}

template <typename Config>
[[nodiscard]] static uint64_t record_kmer_count(uint64_t bases) {
    const uint64_t symbols = record_symbol_count<Config>(bases);
    return symbols < Config::k ? 0 : symbols - Config::k + 1;
}

template <typename Config>
[[nodiscard]] static bool sequence_may_have_invalid_symbols(std::string_view sequence) {
    for (uint64_t offset = 0; offset < sequence.size(); offset += Config::symbolWidth) {
        if (Config::Alphabet::encode(sequence.data() + offset) == Config::Alphabet::invalidSymbol) {
            return true;
        }
    }
    return false;
}

template <typename Config>
[[nodiscard]] static uint64_t valid_record_kmer_count(std::string_view sequence) {
    if (record_symbol_count<Config>(sequence.size()) < Config::k) {
        return 0;
    }

    uint64_t invalid_symbols = 0;
    for (uint64_t i = 0; i < Config::k; ++i) {
        invalid_symbols += Config::Alphabet::encode(sequence.data() + i * Config::symbolWidth) ==
                           Config::Alphabet::invalidSymbol;
    }

    uint64_t valid_kmers = invalid_symbols == 0 ? 1 : 0;
    for (uint64_t start = 1; start < record_kmer_count<Config>(sequence.size()); ++start) {
        invalid_symbols -=
            Config::Alphabet::encode(sequence.data() + (start - 1) * Config::symbolWidth) ==
            Config::Alphabet::invalidSymbol;
        invalid_symbols += Config::Alphabet::encode(
                               sequence.data() + (start + Config::k - 1) * Config::symbolWidth
                           ) == Config::Alphabet::invalidSymbol;
        valid_kmers += invalid_symbols == 0;
    }
    return valid_kmers;
}

template <typename Config>
static void appendRecordBoundary(std::string& sequence) {
    const uint64_t remainder = sequence.size() % Config::symbolWidth;
    if (remainder != 0) {
        sequence.append(
            Config::symbolWidth - remainder, static_cast<char>(Config::Alphabet::separator)
        );
    }
    sequence.append(Config::symbolWidth, static_cast<char>(Config::Alphabet::separator));
}

template <typename Config>
[[nodiscard]] static Result<void> validateRecordBatch(RecordBatchView batch) {
    uint64_t next_offset = 0;
    for (const RecordRange& record : batch.records) {
        if (record.sequenceOffset < next_offset) {
            return Err(
                Error::invalid_argument("record batch ranges must be ordered and non-overlapping")
            );
        }
        if (record.sequenceOffset > batch.sequence.size() ||
            record.sequenceBytes > batch.sequence.size() - record.sequenceOffset) {
            return Err(Error::invalid_argument("record batch range exceeds the source sequence"));
        }
        if (record.sequenceOffset % Config::symbolWidth != 0 ||
            record.sequenceBytes % Config::symbolWidth != 0) {
            return Err(
                Error::invalid_argument(
                    "record batch ranges must align to the configured alphabet symbol width"
                )
            );
        }
        next_offset = record.sequenceOffset + record.sequenceBytes;
    }
    return {};
}

template <typename Config>
static void appendPreparedRecord(
    std::string& output,
    std::vector<NormalizedRecord>& ranges,
    uint64_t record_index,
    uint64_t input_offset,
    std::string_view record_sequence
) {
    if (!output.empty()) {
        appendRecordBoundary<Config>(output);
    }
    const uint64_t output_offset = record_symbol_count<Config>(output.size());
    output.append(record_sequence);
    const uint64_t valid_kmers = sequence_may_have_invalid_symbols<Config>(record_sequence)
                                     ? valid_record_kmer_count<Config>(record_sequence)
                                     : record_kmer_count<Config>(record_sequence.size());
    ranges.push_back(
        NormalizedRecord{
            record_index,
            input_offset,
            output_offset,
            static_cast<uint64_t>(record_sequence.size()),
            valid_kmers,
        }
    );
}

template <typename Config>
[[nodiscard]] static uint64_t normalized_batch_logical_size(uint64_t current_size) noexcept {
    return record_symbol_count<Config>(current_size);
}

template <typename Config>
[[nodiscard]] inline uint64_t estimate_normalized_batch_bytes(RecordBatchView batch) noexcept {
    uint64_t size = 0;
    for (uint64_t record_index = 0; record_index < batch.records.size(); ++record_index) {
        const RecordRange& record = batch.records[record_index];
        if (record_index != 0 || size != 0) {
            const uint64_t remainder = size % Config::symbolWidth;
            if (remainder != 0) {
                size += Config::symbolWidth - remainder;
            }
            size += Config::symbolWidth;
        }
        size += record.sequenceBytes;
    }
    return size;
}

template <typename Config>
[[nodiscard]] static bool records_have_only_valid_symbols(RecordBatchView batch) {
    for (const RecordRange& record : batch.records) {
        const std::string_view record_sequence =
            batch.sequence.substr(record.sequenceOffset, record.sequenceBytes);
        if (sequence_may_have_invalid_symbols<Config>(record_sequence)) {
            return false;
        }
    }
    return true;
}

template <typename Config>
static char* write_record_boundary(char* write_cursor, uint64_t& logical_size) {
    if (logical_size == 0) {
        return write_cursor;
    }

    const uint64_t remainder = logical_size % Config::symbolWidth;
    if (remainder != 0) {
        const uint64_t pad = Config::symbolWidth - remainder;
        std::memset(
            write_cursor, static_cast<int>(Config::Alphabet::separator), static_cast<size_t>(pad)
        );
        write_cursor += pad;
        logical_size += pad;
    }

    std::memset(
        write_cursor,
        static_cast<int>(Config::Alphabet::separator),
        static_cast<size_t>(Config::symbolWidth)
    );
    write_cursor += Config::symbolWidth;
    logical_size += Config::symbolWidth;
    return write_cursor;
}

template <typename Config>
inline void normalize_record_batch_into_buffer(
    RecordBatchView batch,
    char* sequence_out,
    size_t& sequence_out_bytes,
    std::vector<NormalizedRecord>& records_out
) {
    records_out.clear();
    if (batch.records.size() <= 4096) {
        records_out.reserve(batch.records.size());
    }

    const bool records_are_clean = records_have_only_valid_symbols<Config>(batch);

    char* write_cursor = sequence_out;
    uint64_t logical_size = 0;
    for (uint64_t record_index = 0; record_index < batch.records.size(); ++record_index) {
        const RecordRange& record = batch.records[record_index];
        const std::string_view record_sequence =
            batch.sequence.substr(record.sequenceOffset, record.sequenceBytes);

        write_cursor = write_record_boundary<Config>(write_cursor, logical_size);
        const uint64_t output_offset = normalized_batch_logical_size<Config>(logical_size);
        std::memcpy(
            write_cursor, record_sequence.data(), static_cast<size_t>(record_sequence.size())
        );
        write_cursor += record_sequence.size();
        logical_size += record_sequence.size();

        const uint64_t valid_kmers = records_are_clean
                                         ? record_kmer_count<Config>(record_sequence.size())
                                         : valid_record_kmer_count<Config>(record_sequence);
        records_out.push_back(
            NormalizedRecord{
                record_index,
                record.sequenceOffset,
                output_offset,
                static_cast<uint64_t>(record_sequence.size()),
                valid_kmers,
            }
        );
    }

    sequence_out_bytes = static_cast<size_t>(write_cursor - sequence_out);
}

}  // namespace detail

/**
 * @brief Builds a normalized record batch into reusable host buffers.
 *
 * @tparam Config  Filter configuration (alphabet and k-mer sizing).
 * @param batch    Dense input batch without embedded separators.
 * @param sequence_out  Output normalized sequence buffer (resized in place).
 * @param records_out   Output per-record metadata (cleared then filled).
 */
template <typename Config>
[[nodiscard]] Result<void> normalize_record_batch_into(
    RecordBatchView batch,
    std::string& sequence_out,
    std::vector<NormalizedRecord>& records_out
) {
    CUSBF_TRY(detail::validateRecordBatch<Config>(batch));

    const uint64_t estimated_bytes = detail::estimate_normalized_batch_bytes<Config>(batch);
    sequence_out.resize(static_cast<size_t>(estimated_bytes));
    size_t sequence_out_bytes = 0;
    detail::normalize_record_batch_into_buffer<Config>(
        batch, sequence_out.data(), sequence_out_bytes, records_out
    );
    sequence_out.resize(sequence_out_bytes);
    return {};
}

/**
 * @brief Builds a normalized record batch from a dense @ref RecordBatchView.
 *
 * @tparam Config Filter configuration (alphabet and k-mer sizing).
 * @param batch   Dense input batch without embedded separators.
 * @return Owning normalized batch ready for @ref filter::insert_record_batch or query APIs.
 */
template <typename Config>
[[nodiscard]] Result<NormalizedRecordBatch> normalize_record_batch(RecordBatchView batch) {
    return NormalizedRecordBatch::from_batch<Config>(batch);
}

}  // namespace cusbf
