#include <gtest/gtest.h>

#include <fstream>
#include <string>
#include <vector>

#include <unistd.h>

#include <cusbf/detail/fastx_dispatch.hpp>

using DispatchConfig = cusbf::Config<31, 28, 16, 4>;

namespace {

struct TempFile {
    std::string path;

    explicit TempFile(std::string path_value) : path(std::move(path_value)) {
    }

    ~TempFile() {
        if (!path.empty()) {
            std::remove(path.c_str());
        }
    }
};

[[nodiscard]] TempFile write_temp_fasta(std::string_view contents) {
    std::string path_template = "/tmp/cusbf-dispatch-XXXXXX";
    std::vector<char> path_buffer(path_template.begin(), path_template.end());
    path_buffer.push_back('\0');

    const int fd = mkstemp(path_buffer.data());
    if (fd == -1) {
        throw std::runtime_error("Failed to create temporary FASTA file");
    }
    close(fd);

    std::ofstream output(path_buffer.data(), std::ios::binary | std::ios::trunc);
    if (!output.is_open()) {
        std::remove(path_buffer.data());
        throw std::runtime_error("Failed to open temporary FASTA file");
    }
    output << contents;
    output.close();

    return TempFile{path_buffer.data()};
}

}  // namespace

TEST(FastxDispatchTest, SmallFileSelectsSingleChunkStream) {
    const TempFile file = write_temp_fasta(
        ">a\n"
        "ACGTACGTACGTACGT\n"
    );
    const uint64_t file_bytes = cusbf::detail::fastx_file_bytes(file.path);
    ASSERT_GT(file_bytes, 0u);
    EXPECT_TRUE(cusbf::detail::fastx_fits_single_gpu_chunk<DispatchConfig>(
        cusbf::detail::fastx_chunk_mode::insert, 0.7, file_bytes
    ));
    EXPECT_EQ(
        cusbf::detail::select_fastx_dispatch_path<DispatchConfig>(
            file.path, cusbf::detail::fastx_chunk_mode::insert, 0.7
        ),
        cusbf::detail::fastx_dispatch_path::single_chunk_stream
    );
}

TEST(FastxDispatchTest, LargeFileDoesNotSelectSingleChunkStream) {
    const uint64_t huge_bytes = static_cast<uint64_t>(1ull << 40);
    EXPECT_FALSE(cusbf::detail::fastx_fits_single_gpu_chunk<DispatchConfig>(
        cusbf::detail::fastx_chunk_mode::insert, 0.7, huge_bytes
    ));
}

TEST(FastxDispatchTest, SingleChunkStreamMaxBelowMultiGibInputs) {
    EXPECT_LT(cusbf::detail::fastx_single_chunk_stream_max_bytes(), 1u << 30);
}

TEST(FastxDispatchTest, GpuSizedFileSelectsSingleChunkMmap) {
    const uint64_t two_gib = 2u << 30;
    if (!cusbf::detail::fastx_fits_single_gpu_chunk<DispatchConfig>(
            cusbf::detail::fastx_chunk_mode::insert, 0.7, two_gib
        )) {
        GTEST_SKIP() << "GPU staging too small to fit 2 GiB in one chunk on this device";
    }

    EXPECT_EQ(
        cusbf::detail::select_fastx_dispatch_path_for_file_bytes<DispatchConfig>(
            two_gib, cusbf::detail::fastx_chunk_mode::insert, 0.7, true
        ),
        cusbf::detail::fastx_dispatch_path::single_chunk_mmap
    );
    EXPECT_EQ(
        cusbf::detail::select_fastx_dispatch_path_for_file_bytes<DispatchConfig>(
            two_gib, cusbf::detail::fastx_chunk_mode::insert, 0.7, false
        ),
        cusbf::detail::fastx_dispatch_path::chunked_stream
    );
}
