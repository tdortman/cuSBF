#pragma once

#include <algorithm>
#include <concepts>
#include <cstdint>
#include <filesystem>
#include <format>
#include <fstream>
#include <istream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

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
    std::span<const RecordRange> records{};
};

/**
 * @brief Accumulates parsed FASTX records into a dense @ref RecordBatchView.
 *
 * Payloads are stored back-to-back without alphabet separators. Normalization
 * (separator injection and per-record metadata) is performed separately via
 * @ref normalize_record_batch_into.
 */
class DenseRecordBatchBuilder {
   public:
    explicit DenseRecordBatchBuilder(uint64_t reserve_bytes = 0) {
        if (reserve_bytes != 0) {
            sequence_.reserve(static_cast<size_t>(reserve_bytes));
        }
    }

    /// @brief Dense sequence buffer backing @ref view (owned or mmap-external).
    [[nodiscard]] std::string_view sequence_view() const noexcept {
        return external_sequence_.empty() ? std::string_view{sequence_} : external_sequence_;
    }

    /// @brief Non-owning @ref RecordBatchView over the accumulated records.
    [[nodiscard]] RecordBatchView view() const noexcept {
        return RecordBatchView{
            sequence_view(),
            std::span<const RecordRange>{ranges_.data(), ranges_.size()},
        };
    }

    /// @brief Appends one record payload and records its byte range.
    void appendRecord(std::string_view record_sequence) {
        assert(external_sequence_.empty() && "cannot appendRecord when external_sequence_ is set");
        ranges_.push_back(
            RecordRange{
                static_cast<uint64_t>(sequence_.size()),
                static_cast<uint64_t>(record_sequence.size()),
            }
        );
        sequence_.append(record_sequence);
    }

    /// @brief Records a range already present in @ref sequence_view.
    void push_range(RecordRange range) {
        ranges_.push_back(range);
    }

    [[nodiscard]] std::string& sequence_buffer() noexcept {
        return sequence_;
    }

    [[nodiscard]] std::string_view& external_sequence_slot() noexcept {
        return external_sequence_;
    }

    [[nodiscard]] bool empty() const noexcept {
        return ranges_.empty();
    }

    [[nodiscard]] uint64_t recordCount() const noexcept {
        return static_cast<uint64_t>(ranges_.size());
    }

    /// @brief Sum of record payload bytes (ignores mmap storage outside ranges).
    [[nodiscard]] uint64_t raw_sequence_bytes() const noexcept {
        if (!external_sequence_.empty()) {
            uint64_t total = 0;
            for (const RecordRange& range : ranges_) {
                total += range.sequenceBytes;
            }
            return total;
        }
        return static_cast<uint64_t>(sequence_.size());
    }

    [[nodiscard]] const std::vector<RecordRange>& ranges() const noexcept {
        return ranges_;
    }

    void clear() {
        sequence_.clear();
        ranges_.clear();
        external_sequence_ = {};
    }

    void clear_and_shrink() {
        clear();
        std::string{}.swap(sequence_);
        std::vector<RecordRange>{}.swap(ranges_);
    }

   private:
    std::string sequence_;
    std::string_view external_sequence_{};
    std::vector<RecordRange> ranges_;
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
    std::span<const uint8_t> hits{};
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
    std::span<const uint8_t> hits{};
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
template <typename Functor>
concept RecordQueryConsumer = requires(Functor f, const RecordQueryView& view) {
    { f(view) } -> std::same_as<void>;
};

/**
 * @brief Callable invoked once per record by @ref Filter::query_fastx_records and related FASTX
 * streaming query APIs.
 *
 * Must accept `const FastxRecordView&` and return void. The @ref FastxRecordView::hits span is
 * valid only for the duration of the call.
 */
template <typename Functor>
concept FastxRecordConsumer = requires(Functor f, const FastxRecordView& record) {
    { f(record) } -> std::same_as<void>;
};

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
 * Reads one record at a time via @ref nextRecord. Errors reading the
 * stream are returned as error results.
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
    std::string source_name_;
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
        sequence.reserve(4096);
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
        sequence.reserve(4096);
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
