#include <thrust/device_vector.h>
#include <cuda/std/span>
#include <string>

#include <cusbf/dense_packed.hpp>
#include <cusbf/device_span.cuh>

#include "test_support.cuh"

TEST_F(BloomFilterTest, InsertAndQuerySameSequenceHasNoFalseNegatives) {
    cusbf::filter<TestConfig> filter(1 << 12);

    const std::string sequence = "ACGTACGTACGTACGT";
    const uint64_t inserted = CUSBF_UNWRAP(filter.insert_sequence(sequence));
    const auto hits = CUSBF_UNWRAP(filter.contains_sequence(sequence));

    ASSERT_EQ(inserted, sequence.size() - TestConfig::k + 1);
    ASSERT_EQ(hits.size(), inserted);
    EXPECT_TRUE(allOnes(hits));
}

TEST_F(BloomFilterTest, DensePackedMatchesByteSequenceInsertAndQuery) {
    cusbf::filter<TestConfig> filter(1 << 12);

    const std::string sequence = "ACGTACGTACGTACGTACGTACGT";
    const std::vector<uint64_t> packed = cusbf::pack_dense_sequence<TestConfig>(sequence);
    const uint64_t num_symbols = sequence.size();

    const uint64_t inserted = CUSBF_UNWRAP(filter.insert_dense_packed(packed, num_symbols));
    const auto hits = CUSBF_UNWRAP(filter.contains_dense_packed(packed, num_symbols));

    ASSERT_EQ(inserted, num_symbols - TestConfig::k + 1);
    ASSERT_EQ(hits.size(), inserted);
    EXPECT_TRUE(allOnes(hits));

    const auto byteHits = CUSBF_UNWRAP(filter.contains_sequence(sequence));
    EXPECT_EQ(hits, byteHits);
}

TEST_F(BloomFilterTest, DensePackedDeviceAsyncMatchesHostPath) {
    cusbf::filter<TestConfig> filter(1 << 12);

    const std::string sequence = "ACGTACGTACGTACGT";
    const std::vector<uint64_t> packed = cusbf::pack_dense_sequence<TestConfig>(sequence);
    thrust::device_vector<uint64_t> d_packed(packed.begin(), packed.end());

    const uint64_t inserted = CUSBF_UNWRAP(filter.insert_dense_packed_async(
        cuda::std::span<const uint64_t>{
            thrust::raw_pointer_cast(d_packed.data()), d_packed.size()
        },
        sequence.size()
    ));
    ASSERT_EQ(inserted, sequence.size() - TestConfig::k + 1);
    CUSBF_CUDA_CALL(cudaDeviceSynchronize());

    thrust::device_vector<uint8_t> d_output(inserted);
    cusbf::require_void(filter.contains_dense_packed_async(
        cuda::std::span<const uint64_t>{
            thrust::raw_pointer_cast(d_packed.data()), d_packed.size()
        },
        sequence.size(),
        cusbf::device_span<uint8_t>{
            thrust::raw_pointer_cast(d_output.data()), d_output.size()
        }
    ));
    CUSBF_CUDA_CALL(cudaDeviceSynchronize());

    std::vector<uint8_t> hits(inserted);
    CUSBF_CUDA_CALL(cudaMemcpy(
        hits.data(),
        thrust::raw_pointer_cast(d_output.data()),
        hits.size(),
        cudaMemcpyDeviceToHost
    ));
    EXPECT_TRUE(allOnes(hits));
}

TEST_F(BloomFilterTest, DensePackedProteinMatchesByteSequenceInsertAndQuery) {
    cusbf::filter<ProteinTestConfig> filter(1 << 14);

    const std::string sequence = "ACDEFGHIKLMNPQRSTVWYACDEF";
    const std::vector<uint64_t> packed = cusbf::pack_dense_sequence<ProteinTestConfig>(sequence);
    const uint64_t num_symbols = sequence.size();

    const uint64_t inserted = CUSBF_UNWRAP(filter.insert_dense_packed(packed, num_symbols));
    const auto hits = CUSBF_UNWRAP(filter.contains_dense_packed(packed, num_symbols));

    ASSERT_EQ(inserted, num_symbols - ProteinTestConfig::k + 1);
    ASSERT_EQ(hits.size(), inserted);
    EXPECT_TRUE(allOnes(hits));

    const auto byteHits = CUSBF_UNWRAP(filter.contains_sequence(sequence));
    EXPECT_EQ(hits, byteHits);
}

TEST_F(BloomFilterTest, InvalidBasesResetForwardWindows) {
    cusbf::filter<TestConfig> filter(1 << 12);

    const std::string sequence = "ACGTNACGTACGTA";
    const auto inserted = CUSBF_UNWRAP(filter.insert_sequence(sequence));
    const auto hits = CUSBF_UNWRAP(filter.contains_sequence(sequence));

    EXPECT_EQ(inserted, hits.size());
    const std::vector<uint8_t> expected = {0, 0, 0, 0, 0, 1, 1, 1, 1, 1};
    EXPECT_EQ(hits, expected);
}

TEST_F(BloomFilterTest, RepeatedInsertionIsIdempotent) {
    cusbf::filter<TestConfig> filter(1 << 12);

    const std::string sequence = "ACGTACGTACGTACGT";
    const auto firstInserted = CUSBF_UNWRAP(filter.insert_sequence(sequence));
    const float firstLoadFactor = filter.load_factor();

    const auto secondInserted = CUSBF_UNWRAP(filter.insert_sequence(sequence));
    const float secondLoadFactor = filter.load_factor();

    EXPECT_EQ(firstInserted, secondInserted);
    EXPECT_FLOAT_EQ(firstLoadFactor, secondLoadFactor);
}

TEST_F(BloomFilterTest, ShortSequenceInsertAndQueryReturnEmpty) {
    cusbf::filter<TestConfig> filter(1 << 12);

    const std::string shortSequence = "ACGT";
    const uint64_t inserted = CUSBF_UNWRAP(filter.insert_sequence(shortSequence));
    const auto hits = CUSBF_UNWRAP(filter.contains_sequence(shortSequence));

    EXPECT_EQ(inserted, 0);
    EXPECT_TRUE(hits.empty());
}

TEST_F(BloomFilterTest, ShortSequenceDeviceOutputBufferRemainsUnchanged) {
    cusbf::filter<TestConfig> filter(1 << 12);

    thrust::device_vector<char> d_sequence({'A', 'C', 'G', 'T'});
    thrust::device_vector<uint8_t> d_output(1, uint8_t{0xAB});

    cusbf::require_void(filter.contains_sequence_async(
        cusbf::device_span<const char>{
            thrust::raw_pointer_cast(d_sequence.data()), d_sequence.size()
        },
        cusbf::device_span<uint8_t>{thrust::raw_pointer_cast(d_output.data()), d_output.size()}
    ));

    uint8_t after = 0;
    ASSERT_EQ(
        cudaMemcpy(
            &after,
            thrust::raw_pointer_cast(d_output.data()),
            sizeof(uint8_t),
            cudaMemcpyDeviceToHost
        ),
        cudaSuccess
    );
    EXPECT_EQ(after, uint8_t{0xAB});
}

TEST_F(BloomFilterTest, ClearResetsMembership) {
    cusbf::filter<TestConfig> filter(1 << 12);

    const std::string sequence = "ACGTACGTACGTACGT";
    (void)CUSBF_UNWRAP(filter.insert_sequence(sequence));
    (void)CUSBF_UNWRAP(filter.clear());

    const auto hits = CUSBF_UNWRAP(filter.contains_sequence(sequence));
    EXPECT_TRUE(std::all_of(hits.begin(), hits.end(), [](uint8_t value) { return value == 0; }));
}

TEST_F(BloomFilterTest, DeviceOutputMatchesHostContainsResults) {
    cusbf::filter<TestConfig> filter(1 << 13);

    const std::string insertedSequence = "ACGTACGTACGTACGTACGTACGT";
    const std::string querySequence = "TACGTACGTACGTACGTACGTACG";
    (void)CUSBF_UNWRAP(filter.insert_sequence(insertedSequence));

    const auto hostHits = CUSBF_UNWRAP(filter.contains_sequence(querySequence));
    ASSERT_FALSE(hostHits.empty());

    thrust::device_vector<char> d_query(querySequence.begin(), querySequence.end());
    thrust::device_vector<uint8_t> d_output(hostHits.size());

    cusbf::require_void(filter.contains_sequence_async(
        cusbf::device_span<const char>{thrust::raw_pointer_cast(d_query.data()), d_query.size()},
        cusbf::device_span<uint8_t>{thrust::raw_pointer_cast(d_output.data()), d_output.size()}
    ));
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<uint8_t> deviceHits(hostHits.size());
    thrust::copy(d_output.begin(), d_output.end(), deviceHits.begin());

    EXPECT_EQ(deviceHits, hostHits);
}

TEST_F(BloomFilterTest, MultipleInsertionsRemainQueryable) {
    cusbf::filter<TestConfig> filter(1 << 14);

    const std::string sequenceA = "ACGTACGTACGTACGT";
    const std::string sequenceB = "TGCATGCATGCATGCA";

    (void)CUSBF_UNWRAP(filter.insert_sequence(sequenceA));
    (void)CUSBF_UNWRAP(filter.insert_sequence(sequenceB));

    const auto hitsA = CUSBF_UNWRAP(filter.contains_sequence(sequenceA));
    const auto hitsB = CUSBF_UNWRAP(filter.contains_sequence(sequenceB));

    EXPECT_TRUE(allOnes(hitsA));
    EXPECT_TRUE(allOnes(hitsB));
}

TEST_F(BloomFilterTest, LowercaseInsertionMatchesUppercaseQuery) {
    cusbf::filter<TestConfig> filter(1 << 12);

    const std::string lowerSequence = "acgtacgtacgtacgt";
    const std::string upperSequence = "ACGTACGTACGTACGT";

    (void)CUSBF_UNWRAP(filter.insert_sequence(lowerSequence));

    const auto upperHits = CUSBF_UNWRAP(filter.contains_sequence(upperSequence));
    const auto lowerHits = CUSBF_UNWRAP(filter.contains_sequence(lowerSequence));

    EXPECT_TRUE(allOnes(upperHits));
    EXPECT_TRUE(allOnes(lowerHits));
}

TEST_F(BloomFilterTest, ProteinAlphabetInsertAndQuerySameSequenceHasNoFalseNegatives) {
    cusbf::filter<ProteinTestConfig> filter(1 << 12);

    const std::string sequence = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    const uint64_t inserted = CUSBF_UNWRAP(filter.insert_sequence(sequence));
    const auto hits = CUSBF_UNWRAP(filter.contains_sequence(sequence));

    ASSERT_EQ(inserted, sequence.size() - ProteinTestConfig::k + 1);
    ASSERT_EQ(hits.size(), inserted);
    EXPECT_TRUE(allOnes(hits));
}

TEST_F(BloomFilterTest, ProteinAlphabetLowercaseMatchesUppercaseQuery) {
    cusbf::filter<ProteinTestConfig> filter(1 << 12);

    const std::string lowerSequence = "abcdefghijklmnopqrstuvwxyz";
    const std::string upperSequence = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";

    (void)CUSBF_UNWRAP(filter.insert_sequence(lowerSequence));

    EXPECT_TRUE(allOnes(CUSBF_UNWRAP(filter.contains_sequence(upperSequence))));
    EXPECT_TRUE(allOnes(CUSBF_UNWRAP(filter.contains_sequence(lowerSequence))));
}

TEST_F(BloomFilterTest, ProteinAlphabetInvalidSymbolsResetForwardWindows) {
    cusbf::filter<ProteinTestConfig> filter(1 << 12);

    const std::string sequence = "ACDE*ACDEFGHI";
    const auto inserted = CUSBF_UNWRAP(filter.insert_sequence(sequence));
    const auto hits = CUSBF_UNWRAP(filter.contains_sequence(sequence));

    EXPECT_EQ(inserted, hits.size());
    const std::vector<uint8_t> expected = {0, 0, 0, 0, 0, 1, 1, 1, 1};
    EXPECT_EQ(hits, expected);
}

TEST_F(BloomFilterTest, CustomAlphabetUsesExplicitInvalidSentinel) {
    cusbf::filter<CustomAlphabetTestConfig> filter(1 << 12);

    const std::string sequence = "xyz!xyz";
    const auto inserted = CUSBF_UNWRAP(filter.insert_sequence(sequence));
    const auto hits = CUSBF_UNWRAP(filter.contains_sequence(sequence));

    EXPECT_EQ(inserted, hits.size());
    const std::vector<uint8_t> expected = {1, 0, 0, 0, 1};
    EXPECT_EQ(hits, expected);
}

TEST_F(BloomFilterTest, DnaTripletAlphabetInsertAndQuerySameSequenceHasNoFalseNegatives) {
    cusbf::filter<TripletTestConfig> filter(1 << 12);

    const std::string sequence = "ACGTACGTTAAACCCGGGTTT";
    const uint64_t inserted = CUSBF_UNWRAP(filter.insert_sequence(sequence));
    const auto hits = CUSBF_UNWRAP(filter.contains_sequence(sequence));

    ASSERT_EQ(
        inserted, sequence.size() / TripletTestConfig::symbolWidth - TripletTestConfig::k + 1
    );
    ASSERT_EQ(hits.size(), inserted);
    EXPECT_TRUE(allOnes(hits));
}

TEST_F(BloomFilterTest, DnaTripletAlphabetLowercaseMatchesUppercaseQuery) {
    cusbf::filter<TripletTestConfig> filter(1 << 12);

    const std::string lowerSequence = "acgtacgttaaacccgggttt";
    const std::string upperSequence = "ACGTACGTTAAACCCGGGTTT";

    (void)CUSBF_UNWRAP(filter.insert_sequence(lowerSequence));

    EXPECT_TRUE(allOnes(CUSBF_UNWRAP(filter.contains_sequence(upperSequence))));
    EXPECT_TRUE(allOnes(CUSBF_UNWRAP(filter.contains_sequence(lowerSequence))));
}

TEST_F(BloomFilterTest, DnaTripletAlphabetInvalidTripletsResetForwardWindows) {
    cusbf::filter<TripletTestConfig> filter(1 << 12);

    const std::string sequence = "ACGTACNNNGGGTTTAAA";
    const auto inserted = CUSBF_UNWRAP(filter.insert_sequence(sequence));
    const auto hits = CUSBF_UNWRAP(filter.contains_sequence(sequence));

    EXPECT_EQ(inserted, hits.size());
    const std::vector<uint8_t> expected = {0, 0, 0, 1};
    EXPECT_EQ(hits, expected);
}

TEST_F(BloomFilterTest, DnaTripletAlphabetIgnoresTrailingIncompleteTriplet) {
    cusbf::filter<TripletTestConfig> filter(1 << 12);

    const std::string sequence = "ACGTACGTTAAACCCGGGTTTAA";
    const auto inserted = CUSBF_UNWRAP(filter.insert_sequence(sequence));
    const auto hits = CUSBF_UNWRAP(filter.contains_sequence(sequence));

    EXPECT_EQ(inserted, 5u);
    EXPECT_EQ(hits.size(), 5u);
    EXPECT_TRUE(allOnes(hits));
}
