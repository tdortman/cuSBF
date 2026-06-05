#include <gtest/gtest.h>
#include <cusbf/error.hpp>

#include <array>

#include <cusbf/normalized_record_batch.hpp>

using TestConfig = cusbf::Config<3, 2, 2, 4>;

TEST(NormalizedRecordBatchTest, InjectsSeparatorsBetweenRecords) {
    const std::array<cusbf::RecordRange, 2> ranges{{{0, 4}, {4, 4}}};
    const cusbf::RecordBatchView batch{"ACGTACGTACGT", ranges};

    const auto normalized = CUSBF_UNWRAP(cusbf::normalize_record_batch<TestConfig>(batch));
    ASSERT_EQ(normalized.records().size(), 2U);
    EXPECT_GT(normalized.sequence().size(), 8U);
    EXPECT_EQ(normalized.total_valid_kmers(), 4U);
}

TEST(NormalizedRecordBatchTest, RejectsOverlappingRanges) {
    const std::array<cusbf::RecordRange, 2> ranges{{{4, 4}, {0, 4}}};
    const cusbf::RecordBatchView batch{"ACGTACGT", ranges};

    const auto normalized = cusbf::normalize_record_batch<TestConfig>(batch);
    ASSERT_FALSE(normalized);
    EXPECT_EQ(normalized.error().category(), cusbf::ErrorCategory::invalid_argument);
}

TEST(NormalizedRecordBatchTest, PreservesRecordOrderMetadata) {
    const std::array<cusbf::RecordRange, 2> ranges{{{0, 4}, {4, 4}}};
    const cusbf::RecordBatchView batch{"ACGTACGTACGT", ranges};

    const auto normalized = CUSBF_UNWRAP(cusbf::normalize_record_batch<TestConfig>(batch));
    ASSERT_EQ(normalized.records().size(), 2U);
    EXPECT_EQ(normalized.records()[0].record_index, 0U);
    EXPECT_EQ(normalized.records()[1].record_index, 1U);
    EXPECT_EQ(normalized.records()[0].input_offset, 0U);
    EXPECT_EQ(normalized.records()[1].input_offset, 4U);
}
