#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

#include <cusbf/error.hpp>
#include <cusbf/Fastx.hpp>

namespace cusbf::detail {

/// @brief FASTA/FASTQ parser over a contiguous in-memory buffer.
class FastxBufferReader {
   public:
    /**
     * @brief Constructs a reader over a contiguous in-memory FASTA/FASTQ buffer.
     *
     * @param data         Entire file or chunk payload.
     * @param source_name  Label used in parse error messages.
     */
    explicit FastxBufferReader(std::string_view data, std::string_view source_name = "<buffer>")
        : data_(data), source_name_(source_name) {}

    /**
     * @brief Reads the next record into @p record.
     *
     * @param record Output record, cleared before fill.
     * @return @c false at end-of-buffer, @c true when a record was read, or an error.
     */
    [[nodiscard]] Result<bool> nextRecord(FastxRecord& record) {
        record.header.clear();
        record.sequence.clear();

        const auto header = readHeaderLine();
        if (!header) {
            return cuda::std::unexpected(header.error());
        }
        if (header->empty()) {
            return false;
        }

        const char header_tag = header->front();
        if (format_ == FastxFormat::unknown) {
            if (header_tag == '>') {
                format_ = FastxFormat::fasta;
            } else if (header_tag == '@') {
                format_ = FastxFormat::fastq;
            } else {
                return cuda::std::unexpected(
                    parseError("expected FASTA or FASTQ header", fastx_column_at(*header, 0))
                );
            }
        }

        const char expected_header = format_ == FastxFormat::fasta ? '>' : '@';
        if (header_tag != expected_header) {
            return cuda::std::unexpected(parseError(
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

    /// @brief Entire mmap or owned buffer backing this reader.
    [[nodiscard]] std::string_view buffer() const noexcept {
        return data_;
    }

    /**
     * @brief Parses one record with optional zero-copy sequence views for single-line FASTA.
     *
     * When the sequence fits one mmap line, returns a @ref RecordRange into @p buffer instead of
     * appending to @p sequence. Otherwise appends sequence bytes to @p sequence and returns an
     * owned range offset.
     *
     * @param record   Output header (sequence may stay empty on zero-copy path).
     * @param sequence Growing buffer for multi-line or owned FASTA sequence data.
     * @param buffer   Set to the full mmap view on first zero-copy record.
     * @return Record byte range, @c std::nullopt at end-of-buffer, or an error.
     */
    [[nodiscard]] Result<std::optional<RecordRange>>
    appendNextRecord(FastxRecord& record, std::string& sequence, std::string_view& buffer) {
        record.header.clear();
        record.sequence.clear();

        const auto header = readHeaderLine();
        if (!header) {
            return cuda::std::unexpected(header.error());
        }
        if (header->empty()) {
            return std::nullopt;
        }

        const char header_tag = header->front();
        if (format_ == FastxFormat::unknown) {
            if (header_tag == '>') {
                format_ = FastxFormat::fasta;
            } else if (header_tag == '@') {
                format_ = FastxFormat::fastq;
            } else {
                return cuda::std::unexpected(
                    parseError("expected FASTA or FASTQ header", fastx_column_at(*header, 0))
                );
            }
        }

        const char expected_header = format_ == FastxFormat::fasta ? '>' : '@';
        if (header_tag != expected_header) {
            return cuda::std::unexpected(parseError(
                "mixed FASTA and FASTQ records are not supported", fastx_column_at(*header, 0)
            ));
        }

        record.header.assign(header->substr(1));
        if (format_ == FastxFormat::fasta) {
            const uint64_t sequence_offset = static_cast<uint64_t>(position_);
            const std::string_view line = readLine();
            if (line.empty()) {
                return cuda::std::unexpected(
                    parseError("FASTA record missing sequence", fastx_column_at(line, 0))
                );
            }
            if (!line.empty() && line.front() == '>') {
                pending_header_.assign(line);
                return cuda::std::unexpected(
                    parseError("FASTA record missing sequence", fastx_column_at(line, 0))
                );
            }

            if (position_ < data_.size() && data_[position_] != '>') {
                const uint64_t owned_offset = static_cast<uint64_t>(sequence.size());
                sequence.append(line.data(), line.size());
                CUSBF_TRY(readFastaSequence(sequence));
                return RecordRange{
                    owned_offset,
                    static_cast<uint64_t>(sequence.size()) - owned_offset,
                };
            }

            if (position_ < data_.size() && data_[position_] == '>') {
                pending_header_.assign(readLine());
            }

            if (buffer.empty()) {
                buffer = data_;
            }
            return RecordRange{sequence_offset, static_cast<uint64_t>(line.size())};
        }

        const uint64_t sequence_offset = static_cast<uint64_t>(sequence.size());
        CUSBF_TRY(readFastqSequence(sequence));
        return RecordRange{
            sequence_offset,
            static_cast<uint64_t>(sequence.size()) - sequence_offset,
        };
    }

   private:
    std::string_view data_;
    std::string_view source_name_;
    size_t position_{0};
    std::string pending_header_;
    std::string header_line_;
    FastxFormat format_{FastxFormat::unknown};
    uint64_t line_number_{};

    [[nodiscard]] Error parseError(std::string_view message, uint32_t column) const {
        return Error::fastx_parse(
            SourceLocation::fastx(source_name_, static_cast<uint32_t>(line_number_), column),
            message
        );
    }

    [[nodiscard]] std::string_view readLine() {
        if (position_ >= data_.size()) {
            return {};
        }

        size_t end = position_;
        while (end < data_.size() && data_[end] != '\n' && data_[end] != '\r') {
            ++end;
        }

        const std::string_view line = data_.substr(position_, end - position_);
        position_ = end;
        if (position_ < data_.size() && data_[position_] == '\r') {
            ++position_;
        }
        if (position_ < data_.size() && data_[position_] == '\n') {
            ++position_;
        }
        ++line_number_;
        return line;
    }

    [[nodiscard]] Result<std::string_view> readHeaderLine() {
        if (!pending_header_.empty()) {
            header_line_ = std::move(pending_header_);
            pending_header_.clear();
            return std::string_view{header_line_};
        }

        while (position_ < data_.size()) {
            const std::string_view line = readLine();
            if (!line.empty()) {
                header_line_.assign(line.data(), line.size());
                return std::string_view{header_line_};
            }
        }
        return std::string_view{};
    }

    [[nodiscard]] Result<void> readFastaSequence(std::string& sequence) {
        while (position_ < data_.size()) {
            const std::string_view line = readLine();
            if (!line.empty() && line.front() == '>') {
                pending_header_.assign(line);
                return {};
            }
            sequence.append(line.data(), line.size());
        }
        return {};
    }

    [[nodiscard]] Result<void> readFastqSequence(std::string& sequence) {
        std::string_view last_line;
        while (position_ < data_.size()) {
            const std::string_view line = readLine();
            last_line = line;
            if (!line.empty() && line.front() == '+') {
                CUSBF_TRY(readFastqQualities(sequence.size()));
                return {};
            }
            sequence.append(line.data(), line.size());
        }
        return cuda::std::unexpected(parseError(
            "unterminated FASTQ record: missing '+' separator",
            fastx_column_at(last_line, last_line.size() > 0 ? last_line.size() - 1 : 0)
        ));
    }

    [[nodiscard]] Result<void> readFastqQualities(uint64_t expected_length) {
        uint64_t quality_length = 0;
        std::string_view last_line;
        while (quality_length < expected_length && position_ < data_.size()) {
            const std::string_view line = readLine();
            last_line = line;
            quality_length += line.size();
            if (quality_length > expected_length) {
                return cuda::std::unexpected(parseError(
                    "FASTQ quality length exceeds sequence length",
                    fastx_quality_excess_column(quality_length, expected_length, line)
                ));
            }
        }
        if (quality_length != expected_length) {
            return cuda::std::unexpected(parseError(
                "FASTQ quality length does not match sequence length",
                fastx_quality_short_column(last_line)
            ));
        }
        return {};
    }
};

}  // namespace cusbf::detail
