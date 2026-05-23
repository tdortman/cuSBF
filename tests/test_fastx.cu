#include <unistd.h>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "test_support.cuh"

struct StreamedRecord {
    uint64_t record_index{};
    std::string header;
    std::string sequence;
    uint64_t queriedBases{};
    uint64_t queriedKmers{};
    uint64_t positive_kmers{};
    std::vector<uint8_t> hits;
};

TEST_F(BloomFilterTest, InsertFastxFileParsesWrappedFastaRecords) {
    cusbf::filter<TestConfig> filter(1 << 12);

    const std::string sequence = "ACGTACGTACGT";
    const auto file = writeTempFile(
        ">wrapped\n"
        "ACGT\n"
        "ACGT\n"
        "ACGT\n"
    );

    const auto report = CUSBF_UNWRAP(filter.insert_fastx_file(file.path));
    const auto hits = CUSBF_UNWRAP(filter.contains_sequence(sequence));

    EXPECT_EQ(report.recordsIndexed, 1);
    EXPECT_EQ(report.indexedBases, sequence.size());
    EXPECT_EQ(report.insertedKmers, sequence.size() - TestConfig::k + 1);
    EXPECT_TRUE(allOnes(hits));
}

TEST_F(BloomFilterTest, QueryFastxFileParsesWrappedFastqWithCrLf) {
    cusbf::filter<TestConfig> filter(1 << 12);

    const std::string sequence = "ACGTACGTACGT";
    (void)CUSBF_UNWRAP(filter.insert_sequence(sequence));

    const auto file = writeTempFile(
        "@wrapped\r\n"
        "ACGTAC\r\n"
        "GTACGT\r\n"
        "+\r\n"
        "IIIIII\r\n"
        "IIIIII\r\n"
    );

    const auto report = CUSBF_UNWRAP(filter.query_fastx_file(file.path));

    EXPECT_EQ(report.recordsQueried, 1);
    EXPECT_EQ(report.queriedBases, sequence.size());
    EXPECT_EQ(report.queriedKmers, sequence.size() - TestConfig::k + 1);
    EXPECT_EQ(report.positive_kmers, report.queriedKmers);
}

TEST_F(BloomFilterTest, QueryFastxFileRecordsParsesWrappedFastqWithCrLf) {
    cusbf::filter<TestConfig> filter(1 << 12);

    const std::string sequence = "ACGTACGTACGT";
    (void)CUSBF_UNWRAP(filter.insert_sequence(sequence));

    const auto file = writeTempFile(
        "@wrapped\r\n"
        "ACGTAC\r\n"
        "GTACGT\r\n"
        "+\r\n"
        "IIIIII\r\n"
        "IIIIII\r\n"
    );

    std::vector<StreamedRecord> records;
    const auto summary = CUSBF_UNWRAP(filter.query_fastx_file_records(
        file.path,
        [&](const cusbf::FastxRecordView& record) {
            records.push_back(
                StreamedRecord{
                    record.record_index,
                    std::string(record.header),
                    std::string(record.sequence),
                    record.queriedBases,
                    record.queriedKmers,
                    record.positive_kmers,
                    std::vector<uint8_t>(record.hits.begin(), record.hits.end()),
                }
            );
        }));

    ASSERT_EQ(records.size(), 1u);
    const auto& record = records.front();

    EXPECT_EQ(summary.recordsQueried, 1);
    EXPECT_EQ(summary.queriedBases, sequence.size());
    EXPECT_EQ(summary.queriedKmers, sequence.size() - TestConfig::k + 1);
    EXPECT_EQ(summary.positive_kmers, summary.queriedKmers);
    EXPECT_EQ(record.record_index, 0u);
    EXPECT_EQ(record.header, "wrapped");
    EXPECT_EQ(record.sequence, sequence);
    EXPECT_EQ(record.queriedBases, sequence.size());
    EXPECT_EQ(record.queriedKmers, sequence.size() - TestConfig::k + 1);
    EXPECT_EQ(record.positive_kmers, record.queriedKmers);
    EXPECT_TRUE(allOnes(record.hits));
}

TEST_F(BloomFilterTest, QueryFastxFileDoesNotCreateCrossRecordKmers) {
    cusbf::filter<TestConfig> filter(1 << 12);

    const std::string sequenceA = "ACGTACGT";
    const std::string sequenceB = "TGCATGCA";
    (void)CUSBF_UNWRAP(filter.insert_sequence(sequenceA));
    (void)CUSBF_UNWRAP(filter.insert_sequence(sequenceB));

    const auto file = writeTempFile(
        ">first\n"
        "ACGT\n"
        "ACGT\n"
        ">second\n"
        "TGCA\n"
        "TGCA\n"
    );

    const auto report = CUSBF_UNWRAP(filter.query_fastx_file(file.path));

    EXPECT_EQ(report.recordsQueried, 2);
    EXPECT_EQ(report.queriedBases, sequenceA.size() + sequenceB.size());
    EXPECT_EQ(
        report.queriedKmers,
        (sequenceA.size() - TestConfig::k + 1) + (sequenceB.size() - TestConfig::k + 1)
    );
    EXPECT_EQ(report.positive_kmers, report.queriedKmers);
}

TEST_F(BloomFilterTest, RecordBatchInsertAndQueryInjectRecordBoundaries) {
    cusbf::filter<TestConfig> filter(1 << 12);

    const std::string sequenceA = "ACGTACGT";
    const std::string sequenceB = "TGCATGCA";
    const std::string denseSequence = sequenceA + sequenceB;
    const std::vector<cusbf::RecordRange> ranges{
        {0, sequenceA.size()},
        {sequenceA.size(), sequenceB.size()},
    };
    const auto batch = cusbf::RecordBatchView{
        denseSequence,
        cuda::std::span<const cusbf::RecordRange>{ranges.data(), ranges.size()},
    };

    const auto insertReport = CUSBF_UNWRAP(filter.insert_record_batch(batch));

    std::vector<StreamedRecord> records;
    const auto summary = CUSBF_UNWRAP(filter.query_record_batch(
        batch,
        [&](const cusbf::RecordQueryView& record) {
            records.push_back(
                StreamedRecord{
                    record.record_index,
                    {},
                    std::string(record.sequence),
                    record.queriedBases,
                    record.queriedKmers,
                    record.positive_kmers,
                    std::vector<uint8_t>(record.hits.begin(), record.hits.end()),
                }
            );
        }));

    ASSERT_EQ(records.size(), 2u);
    EXPECT_EQ(insertReport.recordsIndexed, 2);
    EXPECT_EQ(insertReport.indexedBases, denseSequence.size());
    EXPECT_EQ(
        insertReport.insertedKmers,
        (sequenceA.size() - TestConfig::k + 1) + (sequenceB.size() - TestConfig::k + 1)
    );
    EXPECT_EQ(summary.recordsQueried, 2);
    EXPECT_EQ(summary.queriedBases, denseSequence.size());
    EXPECT_EQ(summary.queriedKmers, insertReport.insertedKmers);
    EXPECT_EQ(summary.positive_kmers, summary.queriedKmers);
    EXPECT_EQ(records[0].record_index, 0u);
    EXPECT_EQ(records[0].sequence, sequenceA);
    EXPECT_TRUE(allOnes(records[0].hits));
    EXPECT_EQ(records[1].record_index, 1u);
    EXPECT_EQ(records[1].sequence, sequenceB);
    EXPECT_TRUE(allOnes(records[1].hits));
}

TEST_F(BloomFilterTest, TripletQueryFastxFileDoesNotCreateCrossRecordKmers) {
    cusbf::filter<TripletTestConfig> filter(1 << 12);

    const std::string sequenceA = "ACGTACGTT";
    const std::string sequenceB = "GGGTTTAAA";
    (void)CUSBF_UNWRAP(filter.insert_sequence(sequenceA));
    (void)CUSBF_UNWRAP(filter.insert_sequence(sequenceB));

    const auto file = writeTempFile(
        ">first\n"
        "ACGTACGTT\n"
        ">second\n"
        "GGGTTTAAA\n"
    );

    const auto report = CUSBF_UNWRAP(filter.query_fastx_file(file.path));

    EXPECT_EQ(report.recordsQueried, 2);
    EXPECT_EQ(report.queriedBases, sequenceA.size() + sequenceB.size());
    EXPECT_EQ(report.queriedKmers, 2u);
    EXPECT_EQ(report.positive_kmers, report.queriedKmers);
}

TEST_F(BloomFilterTest, TripletQueryFastxFileRecordsDoesNotCreateCrossRecordKmers) {
    cusbf::filter<TripletTestConfig> filter(1 << 12);

    const std::string sequenceA = "ACGTACGTT";
    const std::string sequenceB = "GGGTTTAAA";
    (void)CUSBF_UNWRAP(filter.insert_sequence(sequenceA));
    (void)CUSBF_UNWRAP(filter.insert_sequence(sequenceB));

    const auto file = writeTempFile(
        ">first\n"
        "ACGTACGTT\n"
        ">second\n"
        "GGGTTTAAA\n"
    );

    std::vector<StreamedRecord> records;
    const auto summary = CUSBF_UNWRAP(filter.query_fastx_file_records(
        file.path,
        [&](const cusbf::FastxRecordView& record) {
            records.push_back(
                StreamedRecord{
                    record.record_index,
                    std::string(record.header),
                    std::string(record.sequence),
                    record.queriedBases,
                    record.queriedKmers,
                    record.positive_kmers,
                    std::vector<uint8_t>(record.hits.begin(), record.hits.end()),
                }
            );
        }));

    ASSERT_EQ(records.size(), 2u);
    EXPECT_EQ(summary.recordsQueried, 2);
    EXPECT_EQ(summary.queriedBases, sequenceA.size() + sequenceB.size());
    EXPECT_EQ(summary.queriedKmers, 2u);
    EXPECT_EQ(summary.positive_kmers, 2u);
    EXPECT_EQ(records[0].record_index, 0u);
    EXPECT_EQ(records[0].header, "first");
    EXPECT_EQ(records[0].sequence, sequenceA);
    EXPECT_EQ(records[0].queriedKmers, 1u);
    EXPECT_EQ(records[0].positive_kmers, 1u);
    EXPECT_EQ((records[0].hits), (std::vector<uint8_t>{1}));
    EXPECT_EQ(records[1].record_index, 1u);
    EXPECT_EQ(records[1].header, "second");
    EXPECT_EQ(records[1].sequence, sequenceB);
    EXPECT_EQ(records[1].queriedKmers, 1u);
    EXPECT_EQ(records[1].positive_kmers, 1u);
    EXPECT_EQ((records[1].hits), (std::vector<uint8_t>{1}));
}

TEST_F(BloomFilterTest, TripletQueryFastxDetailedDoesNotCreateCrossRecordKmers) {
    cusbf::filter<TripletTestConfig> filter(1 << 12);

    const std::string sequenceA = "ACGTACGTT";
    const std::string sequenceB = "GGGTTTAAA";
    (void)CUSBF_UNWRAP(filter.insert_sequence(sequenceA));
    (void)CUSBF_UNWRAP(filter.insert_sequence(sequenceB));

    const auto file = writeTempFile(
        ">first\n"
        "ACGTACGTT\n"
        ">second\n"
        "GGGTTTAAA\n"
    );

    const auto report = CUSBF_UNWRAP(filter.query_fastx_file_detailed(file.path));

    ASSERT_EQ(report.records.size(), 2u);
    EXPECT_EQ(report.summary.recordsQueried, 2);
    EXPECT_EQ(report.summary.queriedBases, sequenceA.size() + sequenceB.size());
    EXPECT_EQ(report.summary.queriedKmers, 2u);
    EXPECT_EQ(report.summary.positive_kmers, 2u);
    EXPECT_EQ(report.records[0].record_index, 0u);
    EXPECT_EQ(report.records[0].header, "first");
    EXPECT_EQ(report.records[0].sequence, sequenceA);
    EXPECT_EQ(report.records[0].queriedBases, sequenceA.size());
    EXPECT_EQ(report.records[0].queriedKmers, 1u);
    EXPECT_EQ(report.records[0].positive_kmers, 1u);
    EXPECT_EQ((report.records[0].hits), (std::vector<uint8_t>{1}));
    EXPECT_EQ(report.records[1].record_index, 1u);
    EXPECT_EQ(report.records[1].header, "second");
    EXPECT_EQ(report.records[1].sequence, sequenceB);
    EXPECT_EQ(report.records[1].queriedBases, sequenceB.size());
    EXPECT_EQ(report.records[1].queriedKmers, 1u);
    EXPECT_EQ(report.records[1].positive_kmers, 1u);
    EXPECT_EQ((report.records[1].hits), (std::vector<uint8_t>{1}));
}

TEST_F(BloomFilterTest, FastxReportsOnlyValidKmers) {
    cusbf::filter<TestConfig> filter(1 << 12);

    const auto file = writeTempFile(
        ">with-invalid\n"
        "ACGTNACGTACGTA\n"
    );

    const auto insertReport = CUSBF_UNWRAP(filter.insert_fastx_file(file.path));
    const auto queryReport = CUSBF_UNWRAP(filter.query_fastx_file(file.path));

    EXPECT_EQ(insertReport.recordsIndexed, 1);
    EXPECT_EQ(insertReport.indexedBases, 14);
    EXPECT_EQ(insertReport.insertedKmers, 5);
    EXPECT_EQ(queryReport.recordsQueried, 1);
    EXPECT_EQ(queryReport.queriedBases, 14);
    EXPECT_EQ(queryReport.queriedKmers, 5);
    EXPECT_EQ(queryReport.positive_kmers, 5);
}

TEST_F(BloomFilterTest, QueryFastxFileDetailedParsesWrappedFastqWithCrLf) {
    cusbf::filter<TestConfig> filter(1 << 12);

    const std::string sequence = "ACGTACGTACGT";
    (void)CUSBF_UNWRAP(filter.insert_sequence(sequence));

    const auto file = writeTempFile(
        "@wrapped\r\n"
        "ACGTAC\r\n"
        "GTACGT\r\n"
        "+\r\n"
        "IIIIII\r\n"
        "IIIIII\r\n"
    );

    const auto report = CUSBF_UNWRAP(filter.query_fastx_file_detailed(file.path));

    ASSERT_EQ(report.records.size(), 1u);
    const auto& record = report.records.front();

    EXPECT_EQ(report.summary.recordsQueried, 1);
    EXPECT_EQ(report.summary.queriedBases, sequence.size());
    EXPECT_EQ(report.summary.queriedKmers, sequence.size() - TestConfig::k + 1);
    EXPECT_EQ(report.summary.positive_kmers, report.summary.queriedKmers);
    EXPECT_EQ(record.record_index, 0u);
    EXPECT_EQ(record.header, "wrapped");
    EXPECT_EQ(record.sequence, sequence);
    EXPECT_EQ(record.queriedBases, sequence.size());
    EXPECT_EQ(record.queriedKmers, sequence.size() - TestConfig::k + 1);
    EXPECT_EQ(record.positive_kmers, record.queriedKmers);
    EXPECT_EQ(record.hits.size(), record.queriedBases - TestConfig::k + 1);
    EXPECT_TRUE(allOnes(record.hits));
}

TEST_F(BloomFilterTest, QueryFastxRecordsPreservesWrappedFastaRecordOrder) {
    cusbf::filter<TestConfig> filter(1 << 12);

    const std::string sequenceA = "ACGTACGT";
    const std::string sequenceB = "TGCATGCA";
    (void)CUSBF_UNWRAP(filter.insert_sequence(sequenceA));
    (void)CUSBF_UNWRAP(filter.insert_sequence(sequenceB));

    std::istringstream input(
        ">first\n"
        "ACGT\n"
        "ACGT\n"
        ">second\n"
        "TGCA\n"
        "TGCA\n"
    );

    std::vector<StreamedRecord> records;
    const auto summary = CUSBF_UNWRAP(filter.query_fastx_records(
        input,
        [&](const cusbf::FastxRecordView& record) {
            records.push_back(
                StreamedRecord{
                    record.record_index,
                    std::string(record.header),
                    std::string(record.sequence),
                    record.queriedBases,
                    record.queriedKmers,
                    record.positive_kmers,
                    std::vector<uint8_t>(record.hits.begin(), record.hits.end()),
                }
            );
        }));

    ASSERT_EQ(records.size(), 2u);
    EXPECT_EQ(summary.recordsQueried, 2);
    EXPECT_EQ(summary.queriedBases, sequenceA.size() + sequenceB.size());
    EXPECT_EQ(
        summary.queriedKmers,
        (sequenceA.size() - TestConfig::k + 1) + (sequenceB.size() - TestConfig::k + 1)
    );
    EXPECT_EQ(summary.positive_kmers, summary.queriedKmers);
    EXPECT_EQ(records[0].record_index, 0u);
    EXPECT_EQ(records[0].header, "first");
    EXPECT_EQ(records[0].sequence, sequenceA);
    EXPECT_EQ(records[0].queriedBases, sequenceA.size());
    EXPECT_EQ(records[0].queriedKmers, sequenceA.size() - TestConfig::k + 1);
    EXPECT_EQ(records[0].positive_kmers, records[0].queriedKmers);
    EXPECT_TRUE(allOnes(records[0].hits));
    EXPECT_EQ(records[1].record_index, 1u);
    EXPECT_EQ(records[1].header, "second");
    EXPECT_EQ(records[1].sequence, sequenceB);
    EXPECT_EQ(records[1].queriedBases, sequenceB.size());
    EXPECT_EQ(records[1].queriedKmers, sequenceB.size() - TestConfig::k + 1);
    EXPECT_EQ(records[1].positive_kmers, records[1].queriedKmers);
    EXPECT_TRUE(allOnes(records[1].hits));
}

TEST_F(BloomFilterTest, QueryFastxDetailedPreservesWrappedFastaRecordOrder) {
    cusbf::filter<TestConfig> filter(1 << 12);

    const std::string sequenceA = "ACGTACGT";
    const std::string sequenceB = "TGCATGCA";
    (void)CUSBF_UNWRAP(filter.insert_sequence(sequenceA));
    (void)CUSBF_UNWRAP(filter.insert_sequence(sequenceB));

    std::istringstream input(
        ">first\n"
        "ACGT\n"
        "ACGT\n"
        ">second\n"
        "TGCA\n"
        "TGCA\n"
    );

    const auto report = CUSBF_UNWRAP(filter.query_fastx_detailed(input));

    ASSERT_EQ(report.records.size(), 2u);
    const auto& first = report.records[0];
    const auto& second = report.records[1];

    EXPECT_EQ(report.summary.recordsQueried, 2);
    EXPECT_EQ(report.summary.queriedBases, sequenceA.size() + sequenceB.size());
    EXPECT_EQ(
        report.summary.queriedKmers,
        (sequenceA.size() - TestConfig::k + 1) + (sequenceB.size() - TestConfig::k + 1)
    );
    EXPECT_EQ(report.summary.positive_kmers, report.summary.queriedKmers);
    EXPECT_EQ(first.record_index, 0u);
    EXPECT_EQ(first.header, "first");
    EXPECT_EQ(first.sequence, sequenceA);
    EXPECT_EQ(first.queriedBases, sequenceA.size());
    EXPECT_EQ(first.queriedKmers, sequenceA.size() - TestConfig::k + 1);
    EXPECT_EQ(first.positive_kmers, first.queriedKmers);
    EXPECT_EQ(first.hits.size(), first.queriedBases - TestConfig::k + 1);
    EXPECT_TRUE(allOnes(first.hits));
    EXPECT_EQ(second.record_index, 1u);
    EXPECT_EQ(second.header, "second");
    EXPECT_EQ(second.sequence, sequenceB);
    EXPECT_EQ(second.queriedBases, sequenceB.size());
    EXPECT_EQ(second.queriedKmers, sequenceB.size() - TestConfig::k + 1);
    EXPECT_EQ(second.positive_kmers, second.queriedKmers);
    EXPECT_EQ(second.hits.size(), second.queriedBases - TestConfig::k + 1);
    EXPECT_TRUE(allOnes(second.hits));
}

TEST_F(BloomFilterTest, FastxDetailedQueryReportsInvalidWindowsAsMisses) {
    cusbf::filter<TestConfig> filter(1 << 12);

    const auto file = writeTempFile(
        ">with-invalid\n"
        "ACGTNACGTACGTA\n"
    );

    (void)CUSBF_UNWRAP(filter.insert_fastx_file(file.path));

    const auto report = CUSBF_UNWRAP(filter.query_fastx_file_detailed(file.path));

    ASSERT_EQ(report.records.size(), 1u);
    const auto& record = report.records.front();

    EXPECT_EQ(report.summary.recordsQueried, 1);
    EXPECT_EQ(report.summary.queriedBases, 14);
    EXPECT_EQ(report.summary.queriedKmers, 5);
    EXPECT_EQ(report.summary.positive_kmers, 5);
    EXPECT_EQ(record.record_index, 0u);
    EXPECT_EQ(record.header, "with-invalid");
    EXPECT_EQ(record.sequence, "ACGTNACGTACGTA");
    EXPECT_EQ(record.queriedBases, 14);
    EXPECT_EQ(record.queriedKmers, 5);
    EXPECT_EQ(record.positive_kmers, 5);
    EXPECT_EQ((record.hits), (std::vector<uint8_t>{0, 0, 0, 0, 0, 1, 1, 1, 1, 1}));
}

TEST_F(BloomFilterTest, QueryFastxFileRecordsReportInvalidWindowsAsMisses) {
    cusbf::filter<TestConfig> filter(1 << 12);

    const auto file = writeTempFile(
        ">with-invalid\n"
        "ACGTNACGTACGTA\n"
    );

    (void)CUSBF_UNWRAP(filter.insert_fastx_file(file.path));

    std::vector<StreamedRecord> records;
    const auto summary = CUSBF_UNWRAP(filter.query_fastx_file_records(
        file.path,
        [&](const cusbf::FastxRecordView& record) {
            records.push_back(
                StreamedRecord{
                    record.record_index,
                    std::string(record.header),
                    std::string(record.sequence),
                    record.queriedBases,
                    record.queriedKmers,
                    record.positive_kmers,
                    std::vector<uint8_t>(record.hits.begin(), record.hits.end()),
                }
            );
        }));

    ASSERT_EQ(records.size(), 1u);
    const auto& record = records.front();

    EXPECT_EQ(summary.recordsQueried, 1);
    EXPECT_EQ(summary.queriedBases, 14);
    EXPECT_EQ(summary.queriedKmers, 5);
    EXPECT_EQ(summary.positive_kmers, 5);
    EXPECT_EQ(record.record_index, 0u);
    EXPECT_EQ(record.header, "with-invalid");
    EXPECT_EQ(record.sequence, "ACGTNACGTACGTA");
    EXPECT_EQ(record.queriedBases, 14);
    EXPECT_EQ(record.queriedKmers, 5);
    EXPECT_EQ(record.positive_kmers, 5);
    EXPECT_EQ((record.hits), (std::vector<uint8_t>{0, 0, 0, 0, 0, 1, 1, 1, 1, 1}));
}

TEST_F(BloomFilterTest, FastxDetailedAndAggregateReportsMatch) {
    cusbf::filter<TestConfig> filter(1 << 12);

    const auto file = writeTempFile(
        ">first\n"
        "ACGTNACGTACGTA\n"
        ">second\n"
        "TGCATGCATGCA\n"
    );

    const auto insertReport = CUSBF_UNWRAP(filter.insert_fastx_file(file.path));
    const auto aggregate = CUSBF_UNWRAP(filter.query_fastx_file(file.path));
    const auto detailed = CUSBF_UNWRAP(filter.query_fastx_file_detailed(file.path));

    EXPECT_EQ(aggregate.recordsQueried, detailed.summary.recordsQueried);
    EXPECT_EQ(aggregate.queriedBases, detailed.summary.queriedBases);
    EXPECT_EQ(aggregate.queriedKmers, detailed.summary.queriedKmers);
    EXPECT_EQ(aggregate.positive_kmers, detailed.summary.positive_kmers);
    EXPECT_EQ(insertReport.insertedKmers, detailed.summary.positive_kmers);

    uint64_t detailedBases = 0;
    uint64_t detailedKmers = 0;
    uint64_t detailedPositiveKmers = 0;
    for (const auto& record : detailed.records) {
        detailedBases += record.queriedBases;
        detailedKmers += record.queriedKmers;
        const auto countedPositives =
            static_cast<uint64_t>(std::count(record.hits.begin(), record.hits.end(), uint8_t{1}));
        EXPECT_EQ(record.positive_kmers, countedPositives);
        detailedPositiveKmers += record.positive_kmers;
    }

    EXPECT_EQ(detailed.records.size(), 2u);
    EXPECT_EQ(detailed.records[0].record_index, 0u);
    EXPECT_EQ(detailed.records[1].record_index, 1u);
    EXPECT_EQ(detailedBases, detailed.summary.queriedBases);
    EXPECT_EQ(detailedKmers, detailed.summary.queriedKmers);
    EXPECT_EQ(detailedPositiveKmers, detailed.summary.positive_kmers);
}

TEST_F(BloomFilterTest, ForcedFastxChunkFlushPreservesRecordOrderAndCounts) {
    cusbf::filter<TestConfig> filter(1 << 12);
    constexpr double fill_fraction = 0.0;

    const std::string sequenceA = "ACGTACGT";
    const std::string sequenceB = "ACGTNACGTACGTA";
    const std::string sequenceC = "TGCATGCA";
    const std::vector<uint8_t> sequenceBHits{0, 0, 0, 0, 0, 1, 1, 1, 1, 1};
    const uint64_t expectedInsertKmers =
        (sequenceA.size() - TestConfig::k + 1) +
        static_cast<uint64_t>(std::count(sequenceBHits.begin(), sequenceBHits.end(), uint8_t{1})) +
        (sequenceC.size() - TestConfig::k + 1);

    const auto file = writeTempFile(
        ">first\n"
        "ACGTACGT\n"
        ">second\n"
        "ACGTNACGTACGTA\n"
        ">third\n"
        "TGCATGCA\n"
    );

    const auto insertReport = CUSBF_UNWRAP(filter.insert_fastx_file(file.path, fill_fraction));
    const auto aggregate = CUSBF_UNWRAP(filter.query_fastx_file(file.path, fill_fraction));

    std::vector<StreamedRecord> streamedRecords;
    const auto streamed = CUSBF_UNWRAP(filter.query_fastx_file_records(
        file.path,
        [&](const cusbf::FastxRecordView& record) {
            streamedRecords.push_back(
                StreamedRecord{
                    record.record_index,
                    std::string(record.header),
                    std::string(record.sequence),
                    record.queriedBases,
                    record.queriedKmers,
                    record.positive_kmers,
                    std::vector<uint8_t>(record.hits.begin(), record.hits.end()),
                }
            );
        },
        fill_fraction
    ));
    const auto detailed = CUSBF_UNWRAP(filter.query_fastx_file_detailed(file.path, fill_fraction));

    ASSERT_EQ(streamedRecords.size(), 3u);
    ASSERT_EQ(detailed.records.size(), 3u);

    EXPECT_EQ(insertReport.recordsIndexed, 3);
    EXPECT_EQ(insertReport.indexedBases, sequenceA.size() + sequenceB.size() + sequenceC.size());
    EXPECT_EQ(insertReport.insertedKmers, expectedInsertKmers);
    EXPECT_EQ(aggregate.recordsQueried, 3);
    EXPECT_EQ(aggregate.queriedBases, insertReport.indexedBases);
    EXPECT_EQ(aggregate.queriedKmers, insertReport.insertedKmers);
    EXPECT_EQ(aggregate.positive_kmers, insertReport.insertedKmers);
    EXPECT_EQ(streamed.recordsQueried, aggregate.recordsQueried);
    EXPECT_EQ(streamed.queriedBases, aggregate.queriedBases);
    EXPECT_EQ(streamed.queriedKmers, aggregate.queriedKmers);
    EXPECT_EQ(streamed.positive_kmers, aggregate.positive_kmers);
    EXPECT_EQ(detailed.summary.recordsQueried, aggregate.recordsQueried);
    EXPECT_EQ(detailed.summary.queriedBases, aggregate.queriedBases);
    EXPECT_EQ(detailed.summary.queriedKmers, aggregate.queriedKmers);
    EXPECT_EQ(detailed.summary.positive_kmers, aggregate.positive_kmers);

    EXPECT_EQ(streamedRecords[0].record_index, 0u);
    EXPECT_EQ(streamedRecords[0].header, "first");
    EXPECT_EQ(streamedRecords[0].sequence, sequenceA);
    EXPECT_TRUE(allOnes(streamedRecords[0].hits));
    EXPECT_EQ(streamedRecords[1].record_index, 1u);
    EXPECT_EQ(streamedRecords[1].header, "second");
    EXPECT_EQ(streamedRecords[1].sequence, sequenceB);
    EXPECT_EQ(streamedRecords[1].hits, sequenceBHits);
    EXPECT_EQ(streamedRecords[2].record_index, 2u);
    EXPECT_EQ(streamedRecords[2].header, "third");
    EXPECT_EQ(streamedRecords[2].sequence, sequenceC);
    EXPECT_TRUE(allOnes(streamedRecords[2].hits));

    EXPECT_EQ(detailed.records[0].record_index, 0u);
    EXPECT_EQ(detailed.records[0].header, "first");
    EXPECT_EQ(detailed.records[0].sequence, sequenceA);
    EXPECT_TRUE(allOnes(detailed.records[0].hits));
    EXPECT_EQ(detailed.records[1].record_index, 1u);
    EXPECT_EQ(detailed.records[1].header, "second");
    EXPECT_EQ(detailed.records[1].sequence, sequenceB);
    EXPECT_EQ(detailed.records[1].hits, sequenceBHits);
    EXPECT_EQ(detailed.records[2].record_index, 2u);
    EXPECT_EQ(detailed.records[2].header, "third");
    EXPECT_EQ(detailed.records[2].sequence, sequenceC);
    EXPECT_TRUE(allOnes(detailed.records[2].hits));
}

TEST_F(BloomFilterTest, MalformedFastqThrowsOnQualityLengthMismatch) {
    cusbf::filter<TestConfig> filter(1 << 12);

    const auto file = writeTempFile(
        "@broken\n"
        "ACGTACGT\n"
        "+\n"
        "IIIIIII\n"
    );

    const auto query_result = filter.query_fastx_file(file.path);
    ASSERT_FALSE(query_result);
    EXPECT_EQ(query_result.error().category, cusbf::ErrorCategory::fastx_parse);
}
