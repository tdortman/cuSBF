#pragma once

#include <algorithm>
#include <concepts>
#include <cstdint>
#include <filesystem>
#include <format>
#include <fstream>
#include <istream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <cuda/std/span>

#include <cusbf/error.hpp>
#include <cusbf/gzstreambuf.hpp>

namespace cusbf {

/// @brief Ordered non-overlapping byte range for one record inside a dense sequence batch.
struct RecordRange {
    /// Byte offset into @ref RecordBatchView::sequence.
    uint64_t sequenceOffset{};
    /// Payload length in bytes (symbol-aligned).
    uint64_t sequenceBytes{};
};

/// @brief Dense host-resident sequence batch plus explicit record boundaries.
struct RecordBatchView {
    /// Raw record payloads concatenated.
    std::string_view sequence{};
    /// Ordered, non-overlapping record ranges.
    cuda::std::span<const RecordRange> records{};
};

/// @brief Per-record query payload emitted by query_record_batch().
struct RecordQueryView {
    /// Index in the source batch.
    uint64_t record_index{};
    /// Record sequence slice (normalized layout).
    std::string_view sequence{};
    /// Bases included in the query window.
    uint64_t queriedBases{};
    /// K-mer windows evaluated (valid symbols only).
    uint64_t queriedKmers{};
    /// K-mers reported present in the filter.
    uint64_t positive_kmers{};
    /// Per-k-mer hits (1 = present), valid only in callback.
    cuda::std::span<const uint8_t> hits{};
};

/// @brief Summary statistics returned by Filter insert operations on FASTX and record-batch input.
struct FastxInsertReport {
    /// Records parsed and indexed.
    uint64_t recordsIndexed{};
    /// Total sequence bytes indexed.
    uint64_t indexedBases{};
    /// K-mer windows inserted (valid symbols only).
    uint64_t insertedKmers{};
};

/// @brief Summary statistics returned by Filter query operations on FASTX and record-batch input.
struct FastxQueryReport {
    /// Records parsed and queried.
    uint64_t recordsQueried{};
    /// Total sequence bytes queried.
    uint64_t queriedBases{};
    /// K-mer windows evaluated.
    uint64_t queriedKmers{};
    /// K-mers reported present in the filter.
    uint64_t positive_kmers{};
};

/// @brief Per-record query payload emitted by FASTX streaming query APIs.
struct FastxRecordView {
    /// Index in the input stream or file.
    uint64_t record_index{};
    /// FASTA/FASTQ header (without leading @c > or @c @).
    std::string_view header{};
    /// Record sequence bytes.
    std::string_view sequence{};
    /// Bases included in the query window.
    uint64_t queriedBases{};
    /// K-mer windows evaluated.
    uint64_t queriedKmers{};
    /// K-mers reported present.
    uint64_t positive_kmers{};
    /// Per-k-mer hits (1 = present), valid only for the callback duration.
    cuda::std::span<const uint8_t> hits{};
};

/// @brief Detailed per-record query results returned by Filter FASTX detail APIs.
struct FastxDetailedQueryRecord {
    /// Index in the input stream or file.
    uint64_t record_index{};
    /// Owning copy of the record header.
    std::string header;
    /// Owning copy of the record sequence.
    std::string sequence;
    /// Bases included in the query window.
    uint64_t queriedBases{};
    /// K-mer windows evaluated.
    uint64_t queriedKmers{};
    /// K-mers reported present.
    uint64_t positive_kmers{};
    /// Per-k-mer hits retained after the API returns.
    std::vector<uint8_t> hits;
};

/// @brief Aggregate and per-record results returned by Filter FASTX detail APIs.
struct FastxDetailedQueryReport {
    /// Totals across all records.
    FastxQueryReport summary{};
    /// One entry per record in source order.
    std::vector<FastxDetailedQueryRecord> records;
};

/**
 * @brief Callable invoked once per record by @ref Filter::query_record_batch overloads that take
 * a callback.
 *
 * Must accept `const RecordQueryView&` and return void. The @ref RecordQueryView::hits span is
 * valid only for the duration of the call.
 */
template <typename F>
concept RecordQueryConsumer = std::invocable<F, const RecordQueryView&> &&
                              std::same_as<std::invoke_result_t<F, const RecordQueryView&>, void>;

/**
 * @brief Callable invoked once per record by @ref Filter::query_fastx_records and related FASTX
 * streaming query APIs.
 *
 * Must accept `const FastxRecordView&` and return void. The @ref FastxRecordView::hits span is
 * valid only for the duration of the call.
 */
template <typename F>
concept FastxRecordConsumer = std::invocable<F, const FastxRecordView&> &&
                              std::same_as<std::invoke_result_t<F, const FastxRecordView&>, void>;

namespace detail {

/// @brief Detected file format for a FASTA/FASTQ stream.
enum class FastxFormat : uint8_t {
    /// Format not yet determined from the first header.
    unknown,
    /// FASTA (@c > headers).
    fasta,
    /// FASTQ (@c @ headers).
    fastq,
};

/// @brief A single sequence record extracted from a FASTA/FASTQ stream.
struct FastxRecord {
    /// Header line without the leading @c > or @c @.
    std::string header;
    /// Concatenated sequence lines (qualities skipped for FASTQ).
    std::string sequence;
};

/// @brief Removes a trailing carriage return from @p line if present (Windows line endings).
inline void trimTrailingCarriageReturn(std::string& line) {
    if (!line.empty() && line.back() == '\r') {
        line.pop_back();
    }
}

/// @brief 1-based column at @p byte_index within @p line (clamped to the line end).
[[nodiscard]] inline uint32_t fastx_column_at(std::string_view line, size_t byte_index) {
    if (line.empty()) {
        return 1;
    }
    return static_cast<uint32_t>(std::min(byte_index, line.size() - 1) + 1);
}

/// @brief 1-based column of the first quality byte that exceeds @p expected_length.
[[nodiscard]] inline uint32_t fastx_quality_excess_column(
    uint64_t quality_length,
    uint64_t expected_length,
    std::string_view line
) {
    const uint64_t before = quality_length - line.size();
    const size_t byte_index = expected_length > before ? expected_length - before : line.size();
    return fastx_column_at(line, byte_index);
}

/// @brief 1-based column where a quality run ends too short (position after the last byte).
[[nodiscard]] inline uint32_t fastx_quality_short_column(std::string_view line) {
    return static_cast<uint32_t>(line.empty() ? 1 : line.size() + 1);
}

/**
 * @brief Streaming FASTA/FASTQ parser.
 *
 * Reads one record at a time via @ref nextRecord. Supports both FASTA and
 * FASTQ formats, auto-detected from the first header character. Mixed
 * formats within a single stream are rejected with an exception.
 */
class FastxReader {
   public:
    /**
     * @brief Constructs a reader over an open input stream.
     *
     * @param input        Input stream positioned at the first record.
     * @param source_name  Label used in parse error messages.
     */
    explicit FastxReader(std::istream& input, std::string_view source_name = "<stream>")
        : input_(input), source_name_(source_name) {}

    /**
     * @brief Reads the next FASTA/FASTQ record into @p record.
     *
     * @param record Output record, cleared before fill.
     * @return @c false at end-of-stream, @c true when a record was read, or an error.
     */
    [[nodiscard]] Result<bool> nextRecord(FastxRecord& record) {
        record.header.clear();
        record.sequence.clear();

        const auto header = readHeaderLine();
        if (!header) {
            return Err(header.error());
        }
        if (header->empty()) {
            return false;
        }

        const char headerTag = header->front();

        if (format_ == FastxFormat::unknown) {
            if (headerTag == '>') {
                format_ = FastxFormat::fasta;
            } else if (headerTag == '@') {
                format_ = FastxFormat::fastq;
            } else {
                return Err(
                    parseError("expected FASTA or FASTQ header", fastx_column_at(*header, 0))
                );
            }
        }

        const char expectedHeader = format_ == FastxFormat::fasta ? '>' : '@';
        if (headerTag != expectedHeader) {
            return Err(parseError(
                "mixed FASTA and FASTQ records are not supported", fastx_column_at(*header, 0)
            ));
        }

        record.header.assign(header->substr(1));
        if (format_ == FastxFormat::fasta) {
            CUSBF_TRY(readFastaSequence(record.sequence));
        } else {
            CUSBF_TRY(readFastqSequence(record.sequence));
        }
        return true;
    }

   private:
    std::istream& input_;
    std::string_view source_name_;
    std::string pendingHeader_;
    std::string lineBuffer_;
    FastxFormat format_{FastxFormat::unknown};
    uint64_t lineNumber_{};

    [[nodiscard]] Error parseError(std::string_view message, uint32_t column) const {
        return Error::fastx_parse(
            SourceLocation::fastx(source_name_, static_cast<uint32_t>(lineNumber_), column), message
        );
    }

    [[nodiscard]] Result<std::string_view> readHeaderLine() {
        if (!pendingHeader_.empty()) {
            lineBuffer_ = std::move(pendingHeader_);
            pendingHeader_.clear();
            return std::string_view{lineBuffer_};
        }

        while (std::getline(input_, lineBuffer_)) {
            ++lineNumber_;
            trimTrailingCarriageReturn(lineBuffer_);
            if (!lineBuffer_.empty()) {
                return std::string_view{lineBuffer_};
            }
        }

        if (input_.bad()) {
            return Err(
                Error::io(std::format("Failed to read FASTA/FASTQ input from {}", source_name_))
            );
        }
        return std::string_view{};
    }

    [[nodiscard]] Result<void> readFastaSequence(std::string& sequence) {
        while (std::getline(input_, lineBuffer_)) {
            ++lineNumber_;
            trimTrailingCarriageReturn(lineBuffer_);
            if (!lineBuffer_.empty() && lineBuffer_.front() == '>') {
                pendingHeader_ = std::move(lineBuffer_);
                return {};
            }
            sequence += lineBuffer_;
        }

        if (input_.bad()) {
            return Err(
                Error::io(std::format("Failed to read FASTA/FASTQ input from {}", source_name_))
            );
        }
        return {};
    }

    [[nodiscard]] Result<void> readFastqSequence(std::string& sequence) {
        while (std::getline(input_, lineBuffer_)) {
            ++lineNumber_;
            trimTrailingCarriageReturn(lineBuffer_);
            if (!lineBuffer_.empty() && lineBuffer_.front() == '+') {
                CUSBF_TRY(readFastqQualities(sequence.size()));
                return {};
            }
            sequence += lineBuffer_;
        }

        if (input_.bad()) {
            return Err(
                Error::io(std::format("Failed to read FASTA/FASTQ input from {}", source_name_))
            );
        }
        return Err(parseError(
            "unterminated FASTQ record: missing '+' separator",
            fastx_column_at(lineBuffer_, lineBuffer_.size() > 0 ? lineBuffer_.size() - 1 : 0)
        ));
    }

    [[nodiscard]] Result<void> readFastqQualities(uint64_t expectedLength) {
        uint64_t qualityLength = 0;
        uint32_t short_column = 1;
        while (qualityLength < expectedLength && std::getline(input_, lineBuffer_)) {
            ++lineNumber_;
            trimTrailingCarriageReturn(lineBuffer_);
            short_column = fastx_quality_short_column(lineBuffer_);
            qualityLength += lineBuffer_.size();
            if (qualityLength > expectedLength) {
                return Err(parseError(
                    "FASTQ quality length exceeds sequence length",
                    fastx_quality_excess_column(qualityLength, expectedLength, lineBuffer_)
                ));
            }
        }

        if (qualityLength == expectedLength) {
            return {};
        }
        if (input_.bad()) {
            return Err(
                Error::io(std::format("Failed to read FASTA/FASTQ input from {}", source_name_))
            );
        }
        return Err(parseError("FASTQ quality length does not match sequence length", short_column));
    }
};

/**
 * @brief Opens a FASTA/FASTQ file for reading.
 *
 * @param path  File path.
 * @return Open input file stream, or an I/O error.
 */
[[nodiscard]] inline Result<std::unique_ptr<std::istream>> openFastxFile(
    const std::filesystem::path& path
) {
    if (isGzipFile(path)) {
        return GzIstream::open(path);
    }
    auto input = std::make_unique<std::ifstream>(path);
    if (!input->is_open()) {
        return Err(Error::io(std::format("Failed to open FASTA/FASTQ file: {}", path.string())));
    }
    return input;
}

}  // namespace detail

}  // namespace cusbf
