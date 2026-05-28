#include <gtest/gtest.h>

#include <cusbf/Fastx.hpp>

TEST(DenseRecordBatchBuilderTest, ViewMatchesAppendedRecords) {
    cusbf::DenseRecordBatchBuilder batch;
    batch.appendRecord("ACGT");
    batch.appendRecord("TGCA");

    const cusbf::RecordBatchView view = batch.view();
    ASSERT_EQ(view.records.size(), 2u);
    EXPECT_EQ(view.records[0].sequenceOffset, 0u);
    EXPECT_EQ(view.records[0].sequenceBytes, 4u);
    EXPECT_EQ(view.records[1].sequenceOffset, 4u);
    EXPECT_EQ(view.records[1].sequenceBytes, 4u);
    EXPECT_EQ(view.sequence, "ACGTTGCA");
    EXPECT_EQ(batch.raw_sequence_bytes(), 8u);
}

TEST(DenseRecordBatchBuilderTest, ClearAndShrinkResetsState) {
    cusbf::DenseRecordBatchBuilder batch;
    batch.appendRecord("ACGT");
    batch.clear_and_shrink();
    EXPECT_TRUE(batch.empty());
    EXPECT_EQ(batch.view().records.size(), 0u);
}
