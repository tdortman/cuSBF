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
    uint64_t recordIndex{};
    std::string header;
    std::string sequence;
    uint64_t queriedBases{};
    uint64_t queriedKmers{};
    uint64_t positiveKmers{};
    std::vector<uint8_t> hits;
};

TEST_F(BloomFilterTest, InsertFastxFileParsesWrappedFastaRecords) {
    cusbf::Filter<TestConfig> filter(1 << 12);

    const std::string sequence = "ACGTACGTACGT";
    const auto file = writeTempFile(
        ">wrapped\n"
        "ACGT\n"
        "ACGT\n"
        "ACGT\n"
    );

    const auto report = filter.insertFastxFile(file.path);
    const auto hits = filter.containsSequence(sequence);

    EXPECT_EQ(report.recordsIndexed, 1);
    EXPECT_EQ(report.indexedBases, sequence.size());
    EXPECT_EQ(report.insertedKmers, sequence.size() - TestConfig::k + 1);
    EXPECT_TRUE(allOnes(hits));
}

TEST_F(BloomFilterTest, QueryFastxFileParsesWrappedFastqWithCrLf) {
    cusbf::Filter<TestConfig> filter(1 << 12);

    const std::string sequence = "ACGTACGTACGT";
    (void)filter.insertSequence(sequence);

    const auto file = writeTempFile(
        "@wrapped\r\n"
        "ACGTAC\r\n"
        "GTACGT\r\n"
        "+\r\n"
        "IIIIII\r\n"
        "IIIIII\r\n"
    );

    const auto report = filter.queryFastxFile(file.path);

    EXPECT_EQ(report.recordsQueried, 1);
    EXPECT_EQ(report.queriedBases, sequence.size());
    EXPECT_EQ(report.queriedKmers, sequence.size() - TestConfig::k + 1);
    EXPECT_EQ(report.positiveKmers, report.queriedKmers);
}

TEST_F(BloomFilterTest, QueryFastxFileRecordsParsesWrappedFastqWithCrLf) {
    cusbf::Filter<TestConfig> filter(1 << 12);

    const std::string sequence = "ACGTACGTACGT";
    (void)filter.insertSequence(sequence);

    const auto file = writeTempFile(
        "@wrapped\r\n"
        "ACGTAC\r\n"
        "GTACGT\r\n"
        "+\r\n"
        "IIIIII\r\n"
        "IIIIII\r\n"
    );

    std::vector<StreamedRecord> records;
    const auto summary =
        filter.queryFastxFileRecords(file.path, [&](const cusbf::FastxRecordView& record) {
            records.push_back(
                StreamedRecord{
                    record.recordIndex,
                    std::string(record.header),
                    std::string(record.sequence),
                    record.queriedBases,
                    record.queriedKmers,
                    record.positiveKmers,
                    std::vector<uint8_t>(record.hits.begin(), record.hits.end()),
                }
            );
        });

    ASSERT_EQ(records.size(), 1u);
    const auto& record = records.front();

    EXPECT_EQ(summary.recordsQueried, 1);
    EXPECT_EQ(summary.queriedBases, sequence.size());
    EXPECT_EQ(summary.queriedKmers, sequence.size() - TestConfig::k + 1);
    EXPECT_EQ(summary.positiveKmers, summary.queriedKmers);
    EXPECT_EQ(record.recordIndex, 0u);
    EXPECT_EQ(record.header, "wrapped");
    EXPECT_EQ(record.sequence, sequence);
    EXPECT_EQ(record.queriedBases, sequence.size());
    EXPECT_EQ(record.queriedKmers, sequence.size() - TestConfig::k + 1);
    EXPECT_EQ(record.positiveKmers, record.queriedKmers);
    EXPECT_TRUE(allOnes(record.hits));
}

TEST_F(BloomFilterTest, QueryFastxFileDoesNotCreateCrossRecordKmers) {
    cusbf::Filter<TestConfig> filter(1 << 12);

    const std::string sequenceA = "ACGTACGT";
    const std::string sequenceB = "TGCATGCA";
    (void)filter.insertSequence(sequenceA);
    (void)filter.insertSequence(sequenceB);

    const auto file = writeTempFile(
        ">first\n"
        "ACGT\n"
        "ACGT\n"
        ">second\n"
        "TGCA\n"
        "TGCA\n"
    );

    const auto report = filter.queryFastxFile(file.path);

    EXPECT_EQ(report.recordsQueried, 2);
    EXPECT_EQ(report.queriedBases, sequenceA.size() + sequenceB.size());
    EXPECT_EQ(
        report.queriedKmers,
        (sequenceA.size() - TestConfig::k + 1) + (sequenceB.size() - TestConfig::k + 1)
    );
    EXPECT_EQ(report.positiveKmers, report.queriedKmers);
}

TEST_F(BloomFilterTest, RecordBatchInsertAndQueryInjectRecordBoundaries) {
    cusbf::Filter<TestConfig> filter(1 << 12);

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

    const auto insertReport = filter.insertRecordBatch(batch);

    std::vector<StreamedRecord> records;
    const auto summary = filter.queryRecordBatch(batch, [&](const cusbf::RecordQueryView& record) {
        records.push_back(
            StreamedRecord{
                record.recordIndex,
                {},
                std::string(record.sequence),
                record.queriedBases,
                record.queriedKmers,
                record.positiveKmers,
                std::vector<uint8_t>(record.hits.begin(), record.hits.end()),
            }
        );
    });

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
    EXPECT_EQ(summary.positiveKmers, summary.queriedKmers);
    EXPECT_EQ(records[0].recordIndex, 0u);
    EXPECT_EQ(records[0].sequence, sequenceA);
    EXPECT_TRUE(allOnes(records[0].hits));
    EXPECT_EQ(records[1].recordIndex, 1u);
    EXPECT_EQ(records[1].sequence, sequenceB);
    EXPECT_TRUE(allOnes(records[1].hits));
}

TEST_F(BloomFilterTest, TripletQueryFastxFileDoesNotCreateCrossRecordKmers) {
    cusbf::Filter<TripletTestConfig> filter(1 << 12);

    const std::string sequenceA = "ACGTACGTT";
    const std::string sequenceB = "GGGTTTAAA";
    (void)filter.insertSequence(sequenceA);
    (void)filter.insertSequence(sequenceB);

    const auto file = writeTempFile(
        ">first\n"
        "ACGTACGTT\n"
        ">second\n"
        "GGGTTTAAA\n"
    );

    const auto report = filter.queryFastxFile(file.path);

    EXPECT_EQ(report.recordsQueried, 2);
    EXPECT_EQ(report.queriedBases, sequenceA.size() + sequenceB.size());
    EXPECT_EQ(report.queriedKmers, 2u);
    EXPECT_EQ(report.positiveKmers, report.queriedKmers);
}

TEST_F(BloomFilterTest, TripletQueryFastxFileRecordsDoesNotCreateCrossRecordKmers) {
    cusbf::Filter<TripletTestConfig> filter(1 << 12);

    const std::string sequenceA = "ACGTACGTT";
    const std::string sequenceB = "GGGTTTAAA";
    (void)filter.insertSequence(sequenceA);
    (void)filter.insertSequence(sequenceB);

    const auto file = writeTempFile(
        ">first\n"
        "ACGTACGTT\n"
        ">second\n"
        "GGGTTTAAA\n"
    );

    std::vector<StreamedRecord> records;
    const auto summary =
        filter.queryFastxFileRecords(file.path, [&](const cusbf::FastxRecordView& record) {
            records.push_back(
                StreamedRecord{
                    record.recordIndex,
                    std::string(record.header),
                    std::string(record.sequence),
                    record.queriedBases,
                    record.queriedKmers,
                    record.positiveKmers,
                    std::vector<uint8_t>(record.hits.begin(), record.hits.end()),
                }
            );
        });

    ASSERT_EQ(records.size(), 2u);
    EXPECT_EQ(summary.recordsQueried, 2);
    EXPECT_EQ(summary.queriedBases, sequenceA.size() + sequenceB.size());
    EXPECT_EQ(summary.queriedKmers, 2u);
    EXPECT_EQ(summary.positiveKmers, 2u);
    EXPECT_EQ(records[0].recordIndex, 0u);
    EXPECT_EQ(records[0].header, "first");
    EXPECT_EQ(records[0].sequence, sequenceA);
    EXPECT_EQ(records[0].queriedKmers, 1u);
    EXPECT_EQ(records[0].positiveKmers, 1u);
    EXPECT_EQ((records[0].hits), (std::vector<uint8_t>{1}));
    EXPECT_EQ(records[1].recordIndex, 1u);
    EXPECT_EQ(records[1].header, "second");
    EXPECT_EQ(records[1].sequence, sequenceB);
    EXPECT_EQ(records[1].queriedKmers, 1u);
    EXPECT_EQ(records[1].positiveKmers, 1u);
    EXPECT_EQ((records[1].hits), (std::vector<uint8_t>{1}));
}

TEST_F(BloomFilterTest, TripletQueryFastxDetailedDoesNotCreateCrossRecordKmers) {
    cusbf::Filter<TripletTestConfig> filter(1 << 12);

    const std::string sequenceA = "ACGTACGTT";
    const std::string sequenceB = "GGGTTTAAA";
    (void)filter.insertSequence(sequenceA);
    (void)filter.insertSequence(sequenceB);

    const auto file = writeTempFile(
        ">first\n"
        "ACGTACGTT\n"
        ">second\n"
        "GGGTTTAAA\n"
    );

    const auto report = filter.queryFastxFileDetailed(file.path);

    ASSERT_EQ(report.records.size(), 2u);
    EXPECT_EQ(report.summary.recordsQueried, 2);
    EXPECT_EQ(report.summary.queriedBases, sequenceA.size() + sequenceB.size());
    EXPECT_EQ(report.summary.queriedKmers, 2u);
    EXPECT_EQ(report.summary.positiveKmers, 2u);
    EXPECT_EQ(report.records[0].recordIndex, 0u);
    EXPECT_EQ(report.records[0].header, "first");
    EXPECT_EQ(report.records[0].sequence, sequenceA);
    EXPECT_EQ(report.records[0].queriedBases, sequenceA.size());
    EXPECT_EQ(report.records[0].queriedKmers, 1u);
    EXPECT_EQ(report.records[0].positiveKmers, 1u);
    EXPECT_EQ((report.records[0].hits), (std::vector<uint8_t>{1}));
    EXPECT_EQ(report.records[1].recordIndex, 1u);
    EXPECT_EQ(report.records[1].header, "second");
    EXPECT_EQ(report.records[1].sequence, sequenceB);
    EXPECT_EQ(report.records[1].queriedBases, sequenceB.size());
    EXPECT_EQ(report.records[1].queriedKmers, 1u);
    EXPECT_EQ(report.records[1].positiveKmers, 1u);
    EXPECT_EQ((report.records[1].hits), (std::vector<uint8_t>{1}));
}

TEST_F(BloomFilterTest, FastxReportsOnlyValidKmers) {
    cusbf::Filter<TestConfig> filter(1 << 12);

    const auto file = writeTempFile(
        ">with-invalid\n"
        "ACGTNACGTACGTA\n"
    );

    const auto insertReport = filter.insertFastxFile(file.path);
    const auto queryReport = filter.queryFastxFile(file.path);

    EXPECT_EQ(insertReport.recordsIndexed, 1);
    EXPECT_EQ(insertReport.indexedBases, 14);
    EXPECT_EQ(insertReport.insertedKmers, 5);
    EXPECT_EQ(queryReport.recordsQueried, 1);
    EXPECT_EQ(queryReport.queriedBases, 14);
    EXPECT_EQ(queryReport.queriedKmers, 5);
    EXPECT_EQ(queryReport.positiveKmers, 5);
}

TEST_F(BloomFilterTest, QueryFastxFileDetailedParsesWrappedFastqWithCrLf) {
    cusbf::Filter<TestConfig> filter(1 << 12);

    const std::string sequence = "ACGTACGTACGT";
    (void)filter.insertSequence(sequence);

    const auto file = writeTempFile(
        "@wrapped\r\n"
        "ACGTAC\r\n"
        "GTACGT\r\n"
        "+\r\n"
        "IIIIII\r\n"
        "IIIIII\r\n"
    );

    const auto report = filter.queryFastxFileDetailed(file.path);

    ASSERT_EQ(report.records.size(), 1u);
    const auto& record = report.records.front();

    EXPECT_EQ(report.summary.recordsQueried, 1);
    EXPECT_EQ(report.summary.queriedBases, sequence.size());
    EXPECT_EQ(report.summary.queriedKmers, sequence.size() - TestConfig::k + 1);
    EXPECT_EQ(report.summary.positiveKmers, report.summary.queriedKmers);
    EXPECT_EQ(record.recordIndex, 0u);
    EXPECT_EQ(record.header, "wrapped");
    EXPECT_EQ(record.sequence, sequence);
    EXPECT_EQ(record.queriedBases, sequence.size());
    EXPECT_EQ(record.queriedKmers, sequence.size() - TestConfig::k + 1);
    EXPECT_EQ(record.positiveKmers, record.queriedKmers);
    EXPECT_EQ(record.hits.size(), record.queriedBases - TestConfig::k + 1);
    EXPECT_TRUE(allOnes(record.hits));
}

TEST_F(BloomFilterTest, QueryFastxRecordsPreservesWrappedFastaRecordOrder) {
    cusbf::Filter<TestConfig> filter(1 << 12);

    const std::string sequenceA = "ACGTACGT";
    const std::string sequenceB = "TGCATGCA";
    (void)filter.insertSequence(sequenceA);
    (void)filter.insertSequence(sequenceB);

    std::istringstream input(
        ">first\n"
        "ACGT\n"
        "ACGT\n"
        ">second\n"
        "TGCA\n"
        "TGCA\n"
    );

    std::vector<StreamedRecord> records;
    const auto summary = filter.queryFastxRecords(input, [&](const cusbf::FastxRecordView& record) {
        records.push_back(
            StreamedRecord{
                record.recordIndex,
                std::string(record.header),
                std::string(record.sequence),
                record.queriedBases,
                record.queriedKmers,
                record.positiveKmers,
                std::vector<uint8_t>(record.hits.begin(), record.hits.end()),
            }
        );
    });

    ASSERT_EQ(records.size(), 2u);
    EXPECT_EQ(summary.recordsQueried, 2);
    EXPECT_EQ(summary.queriedBases, sequenceA.size() + sequenceB.size());
    EXPECT_EQ(
        summary.queriedKmers,
        (sequenceA.size() - TestConfig::k + 1) + (sequenceB.size() - TestConfig::k + 1)
    );
    EXPECT_EQ(summary.positiveKmers, summary.queriedKmers);
    EXPECT_EQ(records[0].recordIndex, 0u);
    EXPECT_EQ(records[0].header, "first");
    EXPECT_EQ(records[0].sequence, sequenceA);
    EXPECT_EQ(records[0].queriedBases, sequenceA.size());
    EXPECT_EQ(records[0].queriedKmers, sequenceA.size() - TestConfig::k + 1);
    EXPECT_EQ(records[0].positiveKmers, records[0].queriedKmers);
    EXPECT_TRUE(allOnes(records[0].hits));
    EXPECT_EQ(records[1].recordIndex, 1u);
    EXPECT_EQ(records[1].header, "second");
    EXPECT_EQ(records[1].sequence, sequenceB);
    EXPECT_EQ(records[1].queriedBases, sequenceB.size());
    EXPECT_EQ(records[1].queriedKmers, sequenceB.size() - TestConfig::k + 1);
    EXPECT_EQ(records[1].positiveKmers, records[1].queriedKmers);
    EXPECT_TRUE(allOnes(records[1].hits));
}

TEST_F(BloomFilterTest, QueryFastxDetailedPreservesWrappedFastaRecordOrder) {
    cusbf::Filter<TestConfig> filter(1 << 12);

    const std::string sequenceA = "ACGTACGT";
    const std::string sequenceB = "TGCATGCA";
    (void)filter.insertSequence(sequenceA);
    (void)filter.insertSequence(sequenceB);

    std::istringstream input(
        ">first\n"
        "ACGT\n"
        "ACGT\n"
        ">second\n"
        "TGCA\n"
        "TGCA\n"
    );

    const auto report = filter.queryFastxDetailed(input);

    ASSERT_EQ(report.records.size(), 2u);
    const auto& first = report.records[0];
    const auto& second = report.records[1];

    EXPECT_EQ(report.summary.recordsQueried, 2);
    EXPECT_EQ(report.summary.queriedBases, sequenceA.size() + sequenceB.size());
    EXPECT_EQ(
        report.summary.queriedKmers,
        (sequenceA.size() - TestConfig::k + 1) + (sequenceB.size() - TestConfig::k + 1)
    );
    EXPECT_EQ(report.summary.positiveKmers, report.summary.queriedKmers);
    EXPECT_EQ(first.recordIndex, 0u);
    EXPECT_EQ(first.header, "first");
    EXPECT_EQ(first.sequence, sequenceA);
    EXPECT_EQ(first.queriedBases, sequenceA.size());
    EXPECT_EQ(first.queriedKmers, sequenceA.size() - TestConfig::k + 1);
    EXPECT_EQ(first.positiveKmers, first.queriedKmers);
    EXPECT_EQ(first.hits.size(), first.queriedBases - TestConfig::k + 1);
    EXPECT_TRUE(allOnes(first.hits));
    EXPECT_EQ(second.recordIndex, 1u);
    EXPECT_EQ(second.header, "second");
    EXPECT_EQ(second.sequence, sequenceB);
    EXPECT_EQ(second.queriedBases, sequenceB.size());
    EXPECT_EQ(second.queriedKmers, sequenceB.size() - TestConfig::k + 1);
    EXPECT_EQ(second.positiveKmers, second.queriedKmers);
    EXPECT_EQ(second.hits.size(), second.queriedBases - TestConfig::k + 1);
    EXPECT_TRUE(allOnes(second.hits));
}

TEST_F(BloomFilterTest, FastxDetailedQueryReportsInvalidWindowsAsMisses) {
    cusbf::Filter<TestConfig> filter(1 << 12);

    const auto file = writeTempFile(
        ">with-invalid\n"
        "ACGTNACGTACGTA\n"
    );

    (void)filter.insertFastxFile(file.path);

    const auto report = filter.queryFastxFileDetailed(file.path);

    ASSERT_EQ(report.records.size(), 1u);
    const auto& record = report.records.front();

    EXPECT_EQ(report.summary.recordsQueried, 1);
    EXPECT_EQ(report.summary.queriedBases, 14);
    EXPECT_EQ(report.summary.queriedKmers, 5);
    EXPECT_EQ(report.summary.positiveKmers, 5);
    EXPECT_EQ(record.recordIndex, 0u);
    EXPECT_EQ(record.header, "with-invalid");
    EXPECT_EQ(record.sequence, "ACGTNACGTACGTA");
    EXPECT_EQ(record.queriedBases, 14);
    EXPECT_EQ(record.queriedKmers, 5);
    EXPECT_EQ(record.positiveKmers, 5);
    EXPECT_EQ((record.hits), (std::vector<uint8_t>{0, 0, 0, 0, 0, 1, 1, 1, 1, 1}));
}

TEST_F(BloomFilterTest, QueryFastxFileRecordsReportInvalidWindowsAsMisses) {
    cusbf::Filter<TestConfig> filter(1 << 12);

    const auto file = writeTempFile(
        ">with-invalid\n"
        "ACGTNACGTACGTA\n"
    );

    (void)filter.insertFastxFile(file.path);

    std::vector<StreamedRecord> records;
    const auto summary =
        filter.queryFastxFileRecords(file.path, [&](const cusbf::FastxRecordView& record) {
            records.push_back(
                StreamedRecord{
                    record.recordIndex,
                    std::string(record.header),
                    std::string(record.sequence),
                    record.queriedBases,
                    record.queriedKmers,
                    record.positiveKmers,
                    std::vector<uint8_t>(record.hits.begin(), record.hits.end()),
                }
            );
        });

    ASSERT_EQ(records.size(), 1u);
    const auto& record = records.front();

    EXPECT_EQ(summary.recordsQueried, 1);
    EXPECT_EQ(summary.queriedBases, 14);
    EXPECT_EQ(summary.queriedKmers, 5);
    EXPECT_EQ(summary.positiveKmers, 5);
    EXPECT_EQ(record.recordIndex, 0u);
    EXPECT_EQ(record.header, "with-invalid");
    EXPECT_EQ(record.sequence, "ACGTNACGTACGTA");
    EXPECT_EQ(record.queriedBases, 14);
    EXPECT_EQ(record.queriedKmers, 5);
    EXPECT_EQ(record.positiveKmers, 5);
    EXPECT_EQ((record.hits), (std::vector<uint8_t>{0, 0, 0, 0, 0, 1, 1, 1, 1, 1}));
}

TEST_F(BloomFilterTest, FastxDetailedAndAggregateReportsMatch) {
    cusbf::Filter<TestConfig> filter(1 << 12);

    const auto file = writeTempFile(
        ">first\n"
        "ACGTNACGTACGTA\n"
        ">second\n"
        "TGCATGCATGCA\n"
    );

    const auto insertReport = filter.insertFastxFile(file.path);
    const auto aggregate = filter.queryFastxFile(file.path);
    const auto detailed = filter.queryFastxFileDetailed(file.path);

    EXPECT_EQ(aggregate.recordsQueried, detailed.summary.recordsQueried);
    EXPECT_EQ(aggregate.queriedBases, detailed.summary.queriedBases);
    EXPECT_EQ(aggregate.queriedKmers, detailed.summary.queriedKmers);
    EXPECT_EQ(aggregate.positiveKmers, detailed.summary.positiveKmers);
    EXPECT_EQ(insertReport.insertedKmers, detailed.summary.positiveKmers);

    uint64_t detailedBases = 0;
    uint64_t detailedKmers = 0;
    uint64_t detailedPositiveKmers = 0;
    for (const auto& record : detailed.records) {
        detailedBases += record.queriedBases;
        detailedKmers += record.queriedKmers;
        const auto countedPositives =
            static_cast<uint64_t>(std::count(record.hits.begin(), record.hits.end(), uint8_t{1}));
        EXPECT_EQ(record.positiveKmers, countedPositives);
        detailedPositiveKmers += record.positiveKmers;
    }

    EXPECT_EQ(detailed.records.size(), 2u);
    EXPECT_EQ(detailed.records[0].recordIndex, 0u);
    EXPECT_EQ(detailed.records[1].recordIndex, 1u);
    EXPECT_EQ(detailedBases, detailed.summary.queriedBases);
    EXPECT_EQ(detailedKmers, detailed.summary.queriedKmers);
    EXPECT_EQ(detailedPositiveKmers, detailed.summary.positiveKmers);
}

TEST_F(BloomFilterTest, ForcedFastxChunkFlushPreservesRecordOrderAndCounts) {
    cusbf::Filter<TestConfig> filter(1 << 12);
    constexpr double fillFraction = 0.0;

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

    const auto insertReport = filter.insertFastxFile(file.path, fillFraction);
    const auto aggregate = filter.queryFastxFile(file.path, fillFraction);

    std::vector<StreamedRecord> streamedRecords;
    const auto streamed = filter.queryFastxFileRecords(
        file.path,
        [&](const cusbf::FastxRecordView& record) {
            streamedRecords.push_back(
                StreamedRecord{
                    record.recordIndex,
                    std::string(record.header),
                    std::string(record.sequence),
                    record.queriedBases,
                    record.queriedKmers,
                    record.positiveKmers,
                    std::vector<uint8_t>(record.hits.begin(), record.hits.end()),
                }
            );
        },
        fillFraction
    );
    const auto detailed = filter.queryFastxFileDetailed(file.path, fillFraction);

    ASSERT_EQ(streamedRecords.size(), 3u);
    ASSERT_EQ(detailed.records.size(), 3u);

    EXPECT_EQ(insertReport.recordsIndexed, 3);
    EXPECT_EQ(insertReport.indexedBases, sequenceA.size() + sequenceB.size() + sequenceC.size());
    EXPECT_EQ(insertReport.insertedKmers, expectedInsertKmers);
    EXPECT_EQ(aggregate.recordsQueried, 3);
    EXPECT_EQ(aggregate.queriedBases, insertReport.indexedBases);
    EXPECT_EQ(aggregate.queriedKmers, insertReport.insertedKmers);
    EXPECT_EQ(aggregate.positiveKmers, insertReport.insertedKmers);
    EXPECT_EQ(streamed.recordsQueried, aggregate.recordsQueried);
    EXPECT_EQ(streamed.queriedBases, aggregate.queriedBases);
    EXPECT_EQ(streamed.queriedKmers, aggregate.queriedKmers);
    EXPECT_EQ(streamed.positiveKmers, aggregate.positiveKmers);
    EXPECT_EQ(detailed.summary.recordsQueried, aggregate.recordsQueried);
    EXPECT_EQ(detailed.summary.queriedBases, aggregate.queriedBases);
    EXPECT_EQ(detailed.summary.queriedKmers, aggregate.queriedKmers);
    EXPECT_EQ(detailed.summary.positiveKmers, aggregate.positiveKmers);

    EXPECT_EQ(streamedRecords[0].recordIndex, 0u);
    EXPECT_EQ(streamedRecords[0].header, "first");
    EXPECT_EQ(streamedRecords[0].sequence, sequenceA);
    EXPECT_TRUE(allOnes(streamedRecords[0].hits));
    EXPECT_EQ(streamedRecords[1].recordIndex, 1u);
    EXPECT_EQ(streamedRecords[1].header, "second");
    EXPECT_EQ(streamedRecords[1].sequence, sequenceB);
    EXPECT_EQ(streamedRecords[1].hits, sequenceBHits);
    EXPECT_EQ(streamedRecords[2].recordIndex, 2u);
    EXPECT_EQ(streamedRecords[2].header, "third");
    EXPECT_EQ(streamedRecords[2].sequence, sequenceC);
    EXPECT_TRUE(allOnes(streamedRecords[2].hits));

    EXPECT_EQ(detailed.records[0].recordIndex, 0u);
    EXPECT_EQ(detailed.records[0].header, "first");
    EXPECT_EQ(detailed.records[0].sequence, sequenceA);
    EXPECT_TRUE(allOnes(detailed.records[0].hits));
    EXPECT_EQ(detailed.records[1].recordIndex, 1u);
    EXPECT_EQ(detailed.records[1].header, "second");
    EXPECT_EQ(detailed.records[1].sequence, sequenceB);
    EXPECT_EQ(detailed.records[1].hits, sequenceBHits);
    EXPECT_EQ(detailed.records[2].recordIndex, 2u);
    EXPECT_EQ(detailed.records[2].header, "third");
    EXPECT_EQ(detailed.records[2].sequence, sequenceC);
    EXPECT_TRUE(allOnes(detailed.records[2].hits));
}

TEST_F(BloomFilterTest, MalformedFastqThrowsOnQualityLengthMismatch) {
    cusbf::Filter<TestConfig> filter(1 << 12);

    const auto file = writeTempFile(
        "@broken\n"
        "ACGTACGT\n"
        "+\n"
        "IIIIIII\n"
    );

    EXPECT_THROW((void)filter.queryFastxFile(file.path), std::runtime_error);
}
