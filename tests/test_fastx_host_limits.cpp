#include <gtest/gtest.h>

#include <cstdlib>
#include <limits>

#include <cusbf/detail/fastx_host_limits.cuh>
#include <cusbf/detail/fastx_host_memory.cuh>

TEST(FastxHostLimitsTest, DefaultHostChunkCapUsesGpuStagingOnly) {
    unsetenv("CUSBF_FASTX_MAX_HOST_CHUNK_MB");
    unsetenv("CUSBF_LARGE_FASTX_HOST_CHUNK_MB");
    EXPECT_EQ(cusbf::detail::fastx_host_chunk_max_bytes(), std::numeric_limits<size_t>::max());
}

TEST(FastxHostLimitsTest, LargeFastxGbDoesNotRaiseHostChunkCap) {
    setenv("CUSBF_LARGE_FASTX_GB", "16", 1);
    unsetenv("CUSBF_FASTX_MAX_HOST_CHUNK_MB");
    unsetenv("CUSBF_LARGE_FASTX_HOST_CHUNK_MB");
    EXPECT_EQ(cusbf::detail::fastx_host_chunk_max_bytes(), std::numeric_limits<size_t>::max());
    unsetenv("CUSBF_LARGE_FASTX_GB");
}

TEST(FastxHostLimitsTest, LargeFastxHostChunkMbOverride) {
    setenv("CUSBF_LARGE_FASTX_HOST_CHUNK_MB", "2048", 1);
    EXPECT_EQ(cusbf::detail::fastx_host_chunk_max_bytes(), static_cast<size_t>(2048ull << 20));
    unsetenv("CUSBF_LARGE_FASTX_HOST_CHUNK_MB");
}

TEST(FastxHostLimitsTest, FastxMaxHostChunkMbOverride) {
    setenv("CUSBF_FASTX_MAX_HOST_CHUNK_MB", "128", 1);
    EXPECT_EQ(cusbf::detail::fastx_host_chunk_max_bytes(), 128u << 20);
    unsetenv("CUSBF_FASTX_MAX_HOST_CHUNK_MB");
}

TEST(FastxHostLimitsTest, HostByteLimitDisabledWhenCapIsMax) {
    EXPECT_FALSE(cusbf::detail::fastx_chunk_reached_host_byte_limit(
        std::numeric_limits<size_t>::max(), 1u << 30
    ));
}

#if defined(__linux__)
TEST(FastxHostLimitsTest, MmapBudgetTracksAvailableRam) {
    unsetenv("CUSBF_FASTX_MMAP_MAX_MB");
    const size_t available = cusbf::detail::query_available_host_bytes();
    ASSERT_GT(available, 0u);
    const uint64_t mmap_cap = cusbf::detail::fastx_memory_map_max_bytes();
    EXPECT_LE(mmap_cap, static_cast<uint64_t>(available));
}
#endif
