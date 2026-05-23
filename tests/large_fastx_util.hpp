#pragma once

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>

namespace large_fastx_test {

struct GeneratedFastaStats {
    uint64_t records{};
    uint64_t indexed_bases{};
    uint64_t expected_kmers{};
    uint64_t file_bytes{};
};

struct LargeFastaFile {
    std::string path;
    GeneratedFastaStats stats{};

    LargeFastaFile() = default;

    LargeFastaFile(std::string path_value, GeneratedFastaStats stats_value)
        : path(std::move(path_value)), stats(stats_value) {}

    LargeFastaFile(const LargeFastaFile&) = delete;
    LargeFastaFile& operator=(const LargeFastaFile&) = delete;

    LargeFastaFile(LargeFastaFile&& other) noexcept
        : path(std::move(other.path)), stats(other.stats) {
        other.path.clear();
    }

    LargeFastaFile& operator=(LargeFastaFile&& other) noexcept {
        if (this != &other) {
            remove();
            path = std::move(other.path);
            stats = other.stats;
            other.path.clear();
        }
        return *this;
    }

    ~LargeFastaFile() {
        remove();
    }

    void remove() {
        if (!path.empty()) {
            std::error_code error;
            std::filesystem::remove(path, error);
            path.clear();
        }
    }
};

[[nodiscard]] inline std::string default_output_directory() {
    if (const char* env = std::getenv("CUSBF_LARGE_FASTX_DIR"); env != nullptr && env[0] != '\0') {
        return env;
    }
    return "build/test_artifacts";
}

[[nodiscard]] inline uint64_t available_bytes_on_filesystem(const std::filesystem::path& path) {
    const std::filesystem::space_info space = std::filesystem::space(path);
    return space.available;
}

[[nodiscard]] inline LargeFastaFile generate_fasta(
    const std::filesystem::path& output_path,
    uint64_t target_file_bytes,
    uint64_t sequence_bytes_per_record,
    uint64_t k
) {
    std::filesystem::create_directories(output_path.parent_path());

    std::ofstream output(output_path, std::ios::binary | std::ios::trunc);
    if (!output.is_open()) {
        throw std::runtime_error("Failed to open large FASTA output: " + output_path.string());
    }

    constexpr std::string_view sequence_chunk = "ACGTACGTACGTACGTACGTACGTACGTACGT";
    const std::string header = ">rec\n";
    std::string sequence_line;
    sequence_line.reserve(sequence_bytes_per_record);
    while (sequence_line.size() < sequence_bytes_per_record) {
        const uint64_t append_bytes =
            std::min(sequence_bytes_per_record - sequence_line.size(), sequence_chunk.size());
        sequence_line.append(sequence_chunk.data(), append_bytes);
    }

    GeneratedFastaStats stats;
    while (stats.file_bytes < target_file_bytes) {
        output.write(header.data(), static_cast<std::streamsize>(header.size()));
        output.write(sequence_line.data(), static_cast<std::streamsize>(sequence_line.size()));
        output.put('\n');
        stats.file_bytes += header.size() + sequence_line.size() + 1;

        stats.indexed_bases += sequence_bytes_per_record;
        if (sequence_bytes_per_record >= k) {
            stats.expected_kmers += sequence_bytes_per_record - k + 1;
        }
        stats.records += 1;
    }

    output.close();
    if (!output) {
        std::filesystem::remove(output_path);
        throw std::runtime_error("Failed while writing large FASTA: " + output_path.string());
    }

    stats.file_bytes = std::filesystem::file_size(output_path);
    return LargeFastaFile{output_path.string(), stats};
}

[[nodiscard]] inline LargeFastaFile generate_fasta_in_directory(
    const std::filesystem::path& directory,
    uint64_t target_file_bytes,
    uint64_t sequence_bytes_per_record,
    uint64_t k
) {
    const auto timestamp = std::chrono::steady_clock::now().time_since_epoch().count();
    const std::filesystem::path output_path =
        directory / ("large-input-" + std::to_string(target_file_bytes) + "-" +
                     std::to_string(timestamp) + ".fa");
    return generate_fasta(output_path, target_file_bytes, sequence_bytes_per_record, k);
}

}  // namespace large_fastx_test
