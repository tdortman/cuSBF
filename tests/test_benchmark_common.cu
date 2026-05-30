#include <gtest/gtest.h>

#include "../benchmarks/benchmark_common.cuh"

namespace {

class FastxThroughputConfigTest : public ::testing::Test {
  protected:
    void TearDown() override {
        benchmark_common::clearFastxInsertWorkload();
    }
};

TEST_F(FastxThroughputConfigTest, ClampsBenchKmersToAvailableInsertKmers) {
    auto workload = std::make_unique<benchmark_common::FastxInsertWorkload>();
    workload->insert_kmers = 437500;
    workload->host_insert_sequence.assign(700000, 'A');
    benchmark_common::g_fastxInsertWorkload = std::move(workload);

    const auto cfg = benchmark_common::resolveFastxThroughputConfig(31);

    EXPECT_EQ(cfg.genome_kmers, 437500);
    EXPECT_EQ(cfg.filter_bits, 8388608);
    EXPECT_EQ(cfg.bench_kmers, 437500);
    EXPECT_EQ(cfg.bench_seq_len, 437530);
}

}  // namespace
