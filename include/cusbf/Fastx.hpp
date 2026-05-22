#pragma once

#include <cstdint>
#include <fstream>
#include <istream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <cuda/std/span>

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
        : input_(input), source_name_(source_name) {
    }

    /**
     * @brief Reads the next FASTA/FASTQ record into @p record.
     *
     * @param record Output record, cleared before fill.
     * @return @c false at end-of-stream, @c true when a record was read.
     * @throws std::runtime_error on parse or I/O errors.
     */
    [[nodiscard]] bool nextRecord(FastxRecord& record) {
        record.header.clear();
        record.sequence.clear();

        const std::string_view header = readHeaderLine();
        if (header.empty()) {
            return false;
        }

        const char headerTag = header.front();

        if (format_ == FastxFormat::unknown) {
            if (headerTag == '>') {
                format_ = FastxFormat::fasta;
            } else if (headerTag == '@') {
                format_ = FastxFormat::fastq;
            } else {
                throwParseError("expected FASTA or FASTQ header");
            }
        }

        const char expectedHeader = format_ == FastxFormat::fasta ? '>' : '@';
        if (headerTag != expectedHeader) {
            throwParseError("mixed FASTA and FASTQ records are not supported");
        }

        record.header.assign(header.substr(1));
        if (format_ == FastxFormat::fasta) {
            readFastaSequence(record.sequence);
        } else {
            readFastqSequence(record.sequence);
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

    [[noreturn]] void throwParseError(std::string_view message) const {
        throw std::runtime_error(
            std::string(source_name_) + ":" + std::to_string(lineNumber_) + ": " +
            std::string(message)
        );
    }

    [[nodiscard]] std::string_view readHeaderLine() {
        if (!pendingHeader_.empty()) {
            lineBuffer_ = std::move(pendingHeader_);
            pendingHeader_.clear();
            return lineBuffer_;
        }

        while (std::getline(input_, lineBuffer_)) {
            ++lineNumber_;
            trimTrailingCarriageReturn(lineBuffer_);
            if (!lineBuffer_.empty()) {
                return lineBuffer_;
            }
        }

        if (input_.bad()) {
            throw std::runtime_error(
                std::string("Failed to read FASTA/FASTQ input from ") + std::string(source_name_)
            );
        }
        return {};
    }

    void readFastaSequence(std::string& sequence) {
        while (std::getline(input_, lineBuffer_)) {
            ++lineNumber_;
            trimTrailingCarriageReturn(lineBuffer_);
            if (!lineBuffer_.empty() && lineBuffer_.front() == '>') {
                pendingHeader_ = std::move(lineBuffer_);
                return;
            }
            sequence += lineBuffer_;
        }

        if (input_.bad()) {
            throw std::runtime_error(
                std::string("Failed to read FASTA/FASTQ input from ") + std::string(source_name_)
            );
        }
    }

    void readFastqSequence(std::string& sequence) {
        while (std::getline(input_, lineBuffer_)) {
            ++lineNumber_;
            trimTrailingCarriageReturn(lineBuffer_);
            if (!lineBuffer_.empty() && lineBuffer_.front() == '+') {
                readFastqQualities(sequence.size());
                return;
            }
            sequence += lineBuffer_;
        }

        if (input_.bad()) {
            throw std::runtime_error(
                std::string("Failed to read FASTA/FASTQ input from ") + std::string(source_name_)
            );
        }
        throwParseError("unterminated FASTQ record: missing '+' separator");
    }

    void readFastqQualities(uint64_t expectedLength) {
        uint64_t qualityLength = 0;
        while (qualityLength < expectedLength && std::getline(input_, lineBuffer_)) {
            ++lineNumber_;
            trimTrailingCarriageReturn(lineBuffer_);
            qualityLength += lineBuffer_.size();
            if (qualityLength > expectedLength) {
                throwParseError("FASTQ quality length exceeds sequence length");
            }
        }

        if (qualityLength == expectedLength) {
            return;
        }
        if (input_.bad()) {
            throw std::runtime_error(
                std::string("Failed to read FASTA/FASTQ input from ") + std::string(source_name_)
            );
        }
        throwParseError("FASTQ quality length does not match sequence length");
    }
};

/**
 * @brief Opens a FASTA/FASTQ file for reading.
 *
 * @param path  File path.
 * @return Open input file stream.
 * @throws std::runtime_error if the file cannot be opened.
 */
inline std::unique_ptr<std::istream> openFastxFile(std::string_view path) {
    if (isGzipFile(path)) {
        return std::make_unique<GzIstream>(path);
    }
    auto input = std::make_unique<std::ifstream>(std::string(path));
    if (!input->is_open()) {
        throw std::runtime_error("Failed to open FASTA/FASTQ file: " + std::string(path));
    }
    return input;
}

}  // namespace detail

}  // namespace cusbf
