#include <gtest/gtest.h>

#include <array>

#include <cusbf/detail/query_layout.cuh>

using TestConfig = cusbf::Config<3, 2, 2, 4>;

TEST(QueryLayoutTest, DerivesPerRecordHitSpansFromNormalizedLayout) {
    static_assert(TestConfig::symbolWidth == 1);
    static_assert(TestConfig::k == 3);

    const std::array<cusbf::NormalizedRecord, 2> records{{
        cusbf::NormalizedRecord{0, 0, 6, 1},
        cusbf::NormalizedRecord{1, 6, 8, 2},
    }};

    const auto layout = cusbf::detail::QueryLayout::build<TestConfig>(records);
    const auto layout_records = layout.records();

    ASSERT_EQ(layout_records.size(), 2U);
    EXPECT_EQ(layout.total_hit_count(), 13U);

    EXPECT_EQ(layout_records[0].record_index, 0U);
    EXPECT_EQ(layout_records[0].input_offset, 0U);
    EXPECT_EQ(layout_records[0].size, 6U);
    EXPECT_EQ(layout_records[0].valid_kmers, 1U);
    EXPECT_EQ(layout_records[0].hit_offset, 0U);
    EXPECT_EQ(layout_records[0].hit_count, 4U);

    EXPECT_EQ(layout_records[1].record_index, 1U);
    EXPECT_EQ(layout_records[1].input_offset, 6U);
    EXPECT_EQ(layout_records[1].size, 8U);
    EXPECT_EQ(layout_records[1].valid_kmers, 2U);
    EXPECT_EQ(layout_records[1].hit_offset, 7U);
    EXPECT_EQ(layout_records[1].hit_count, 6U);

    const std::array<uint8_t, 13> hits{{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}};

    const auto first_hits = layout.hits_for_record(hits, 0);
    ASSERT_EQ(first_hits.size(), 4U);
    EXPECT_EQ(first_hits[0], 0U);
    EXPECT_EQ(first_hits[1], 1U);
    EXPECT_EQ(first_hits[2], 2U);
    EXPECT_EQ(first_hits[3], 3U);

    const auto second_hits = layout.hits_for_record(hits, 1);
    ASSERT_EQ(second_hits.size(), 6U);
    EXPECT_EQ(second_hits[0], 7U);
    EXPECT_EQ(second_hits[1], 8U);
    EXPECT_EQ(second_hits[2], 9U);
    EXPECT_EQ(second_hits[3], 10U);
    EXPECT_EQ(second_hits[4], 11U);
    EXPECT_EQ(second_hits[5], 12U);
}

TEST(QueryLayoutTest, KeepsValidKmersSeparateFromHitCount) {
    const std::array<cusbf::NormalizedRecord, 1> records{{
        cusbf::NormalizedRecord{0, 0, 14, 3},
    }};

    const auto layout = cusbf::detail::QueryLayout::build<TestConfig>(records);
    const auto layout_records = layout.records();

    ASSERT_EQ(layout_records.size(), 1U);
    EXPECT_EQ(layout.total_hit_count(), 12U);
    EXPECT_EQ(layout_records[0].valid_kmers, 3U);
    EXPECT_EQ(layout_records[0].hit_count, 12U);
}

TEST(QueryLayoutTest, TotalHitCountIncludesBoundaryWindows) {
    const std::array<cusbf::NormalizedRecord, 2> records{{
        cusbf::NormalizedRecord{0, 0, 4, 0},
        cusbf::NormalizedRecord{1, 4, 4, 0},
    }};

    const auto layout = cusbf::detail::QueryLayout::build<TestConfig>(records);
    const auto layout_records = layout.records();

    ASSERT_EQ(layout_records.size(), 2U);
    EXPECT_EQ(layout_records[0].hit_count, 2U);
    EXPECT_EQ(layout_records[1].hit_count, 2U);
    EXPECT_EQ(layout.total_hit_count(), 7U);
}
