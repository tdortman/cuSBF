#include <gtest/gtest.h>

#include <cusbf/detail/fastx_chunk.cuh>

using TestConfig = cusbf::Config<31, 28, 16, 4>;

TEST(FastxChunkTest, ZeroFillFractionForcesImmediateFlush) {
    EXPECT_TRUE(
        cusbf::detail::fastx_chunk_reached_staging_budget<TestConfig>(
            cusbf::detail::fastx_chunk_mode::insert, 0, 8, 1
        )
    );
    EXPECT_TRUE(
        cusbf::detail::fastx_chunk_reached_staging_budget<TestConfig>(
            cusbf::detail::fastx_chunk_mode::query, 0, 8, 1
        )
    );
    EXPECT_FALSE(
        cusbf::detail::fastx_chunk_reached_staging_budget<TestConfig>(
            cusbf::detail::fastx_chunk_mode::insert, 0, 0, 0
        )
    );
}

TEST(FastxChunkTest, QueryStagingExceedsInsertStaging) {
    constexpr uint64_t raw_bytes = 1u << 20;
    constexpr uint64_t records = 16;
    const uint64_t insertStaging =
        cusbf::detail::estimate_insert_staging_bytes<TestConfig>(raw_bytes, records);
    const uint64_t queryStaging =
        cusbf::detail::estimate_query_staging_bytes<TestConfig>(raw_bytes, records);
    EXPECT_GT(queryStaging, insertStaging);
}

TEST(FastxChunkTest, StagingBudgetReservesSlack) {
    constexpr size_t free_bytes = 1u << 30;
    const size_t budget = cusbf::detail::fastx_staging_budget_bytes<TestConfig>(0.5, free_bytes);
    const size_t expected = static_cast<size_t>(
        0.5 * static_cast<double>(free_bytes - cusbf::detail::fastx_chunk_slack_bytes())
    );
    EXPECT_EQ(budget, expected);
}

TEST(FastxChunkTest, QueryModeReachesBudgetBeforeInsertMode) {
    constexpr size_t budget = 1u << 20;
    constexpr uint64_t raw_bytes = 700000;
    constexpr uint64_t records = 4;
    const bool insertReached = cusbf::detail::fastx_chunk_reached_staging_budget<TestConfig>(
        cusbf::detail::fastx_chunk_mode::insert, budget, raw_bytes, records
    );
    const bool queryReached = cusbf::detail::fastx_chunk_reached_staging_budget<TestConfig>(
        cusbf::detail::fastx_chunk_mode::query, budget, raw_bytes, records
    );
    EXPECT_FALSE(insertReached);
    EXPECT_TRUE(queryReached);
}
