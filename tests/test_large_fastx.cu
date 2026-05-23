#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>

#include <cusbf/detail/fastx_dispatch.hpp>
#include <cusbf/detail/fastx_host_memory.cuh>
#include <cusbf/filter.cuh>

#include "large_fastx_util.hpp"

using LargeTestConfig = cusbf::Config<31, 28, 16, 4>;

namespace {

[[nodiscard]] bool cuda_device_available() {
    int device_count = 0;
    return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
}

[[nodiscard]] uint64_t parse_positive_env_bytes(const char* name) {
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return 0;
    }

    char* end = nullptr;
    const auto parsed = std::strtoull(value, &end, 10);
    if (end == value || parsed == 0) {
        return 0;
    }
    return parsed;
}

/// @brief Target generated file size from @c CUSBF_LARGE_FASTX_BYTES, @c _MB, or @c _GB.
[[nodiscard]] uint64_t parse_target_file_bytes() {
    if (const uint64_t bytes = parse_positive_env_bytes("CUSBF_LARGE_FASTX_BYTES"); bytes != 0) {
        return bytes;
    }

    if (const char* mebibytes_value = std::getenv("CUSBF_LARGE_FASTX_MB");
        mebibytes_value != nullptr && mebibytes_value[0] != '\0') {
        char* end = nullptr;
        const auto mebibytes = std::strtoull(mebibytes_value, &end, 10);
        if (end != mebibytes_value && mebibytes > 0) {
            return mebibytes << 20;
        }
        return 0;
    }

    const char* gib_value = std::getenv("CUSBF_LARGE_FASTX_GB");
    if (gib_value == nullptr || gib_value[0] == '\0') {
        return 0;
    }

    char* end = nullptr;
    const double gib = std::strtod(gib_value, &end);
    if (end == gib_value || gib <= 0.0) {
        return 0;
    }
    return static_cast<uint64_t>(gib * 1024.0 * 1024.0 * 1024.0);
}

[[nodiscard]] uint64_t parse_record_bytes_env() {
    constexpr uint64_t default_record_bytes = 256u << 10;
    const char* value = std::getenv("CUSBF_LARGE_FASTX_RECORD_BYTES");
    if (value == nullptr || value[0] == '\0') {
        return default_record_bytes;
    }

    char* end = nullptr;
    const auto parsed = std::strtoull(value, &end, 10);
    if (end == value || parsed == 0) {
        return 0;
    }
    return parsed;
}

struct GpuMemorySnapshot {
    size_t free_bytes{};
    size_t total_bytes{};
};

[[nodiscard]] bool large_fastx_timing_enabled() {
    const char* value = std::getenv("CUSBF_LARGE_FASTX_TIME");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

template <typename Duration>
void maybe_log_duration(const char* label, const Duration& duration) {
    if (!large_fastx_timing_enabled()) {
        return;
    }
    std::cerr << label << ": " << std::chrono::duration<double>(duration).count() << " s\n";
}

[[nodiscard]] GpuMemorySnapshot query_gpu_memory() {
    GpuMemorySnapshot snapshot;
    if (cudaMemGetInfo(&snapshot.free_bytes, &snapshot.total_bytes) != cudaSuccess) {
        return {};
    }
    return snapshot;
}

[[nodiscard]] const char* dispatch_path_name(cusbf::detail::fastx_dispatch_path path) {
    switch (path) {
        case cusbf::detail::fastx_dispatch_path::single_chunk_stream:
            return "single_chunk_stream";
        case cusbf::detail::fastx_dispatch_path::single_chunk_mmap:
            return "single_chunk_mmap";
        case cusbf::detail::fastx_dispatch_path::chunked_mmap:
            return "chunked_mmap";
        case cusbf::detail::fastx_dispatch_path::chunked_stream:
            return "chunked_stream";
    }
    return "unknown";
}

}  // namespace

TEST(LargeFastxOutOfCore, ProcessesGeneratedFastaAtConfiguredSize) {
    if (!cuda_device_available()) {
        GTEST_SKIP() << "CUDA device unavailable";
    }

    const uint64_t target_file_bytes = parse_target_file_bytes();
    if (target_file_bytes == 0) {
        GTEST_SKIP()
            << "Set CUSBF_LARGE_FASTX_BYTES, CUSBF_LARGE_FASTX_MB, or CUSBF_LARGE_FASTX_GB";
    }

    const uint64_t record_bytes = parse_record_bytes_env();
    if (record_bytes == 0) {
        GTEST_FAIL() << "CUSBF_LARGE_FASTX_RECORD_BYTES must be a positive integer";
    }
    if (record_bytes < LargeTestConfig::k) {
        GTEST_FAIL() << "CUSBF_LARGE_FASTX_RECORD_BYTES must be at least k=" << LargeTestConfig::k;
    }

    const std::filesystem::path output_dir = large_fastx_test::default_output_directory();
    std::error_code directory_error;
    std::filesystem::create_directories(output_dir, directory_error);
    if (directory_error) {
        GTEST_FAIL() << "Failed to create output directory " << output_dir.string() << ": "
                     << directory_error.message();
    }

    const uint64_t available_disk_bytes =
        large_fastx_test::available_bytes_on_filesystem(output_dir);
    const uint64_t disk_headroom = std::max<uint64_t>(64u << 20, target_file_bytes / 10);
    const uint64_t required_disk_bytes = target_file_bytes + disk_headroom;
    if (available_disk_bytes < required_disk_bytes) {
        GTEST_SKIP() << "Need at least " << required_disk_bytes << " bytes free under "
                     << output_dir.string() << ", found " << available_disk_bytes;
    }

    const GpuMemorySnapshot gpu_before = query_gpu_memory();
    if (gpu_before.total_bytes == 0) {
        GTEST_SKIP() << "cudaMemGetInfo unavailable";
    }

    constexpr double fill_fraction = 0.7;

    SCOPED_TRACE("Generating large FASTA");
    const auto generate_begin = std::chrono::steady_clock::now();
    large_fastx_test::LargeFastaFile input = large_fastx_test::generate_fasta_in_directory(
        output_dir, target_file_bytes, record_bytes, LargeTestConfig::k
    );
    maybe_log_duration("generate", std::chrono::steady_clock::now() - generate_begin);

    ASSERT_GE(input.stats.file_bytes, target_file_bytes);
    ASSERT_GT(input.stats.records, 0u);
    ASSERT_GT(input.stats.indexed_bases, 0u);
    ASSERT_GT(input.stats.expected_kmers, 0u);

    const cusbf::detail::fastx_dispatch_path insert_path =
        cusbf::detail::select_fastx_dispatch_path<LargeTestConfig>(
            input.path, cusbf::detail::fastx_chunk_mode::insert, fill_fraction
        );
    if (large_fastx_timing_enabled()) {
        std::cerr << "insert_dispatch: " << dispatch_path_name(insert_path) << '\n';
    }

    SCOPED_TRACE("Insert large FASTA");
    cusbf::filter<LargeTestConfig> filter(1u << 24);
    const auto insert_begin = std::chrono::steady_clock::now();
    const auto insert_report = filter.insert_fastx_file(input.path, fill_fraction);
    maybe_log_duration("insert", std::chrono::steady_clock::now() - insert_begin);
    EXPECT_EQ(insert_report.recordsIndexed, input.stats.records);
    EXPECT_EQ(insert_report.indexedBases, input.stats.indexed_bases);
    EXPECT_EQ(insert_report.insertedKmers, input.stats.expected_kmers);

    SCOPED_TRACE("Query large FASTA");
    const auto query_begin = std::chrono::steady_clock::now();
    const auto query_report = filter.query_fastx_file(input.path, fill_fraction);
    maybe_log_duration("query", std::chrono::steady_clock::now() - query_begin);
    EXPECT_EQ(query_report.recordsQueried, input.stats.records);
    EXPECT_EQ(query_report.queriedBases, input.stats.indexed_bases);
    EXPECT_EQ(query_report.queriedKmers, input.stats.expected_kmers);
    EXPECT_EQ(query_report.positive_kmers, input.stats.expected_kmers);
}
