#pragma once

#include <span>
#include <vector>

#include <cstddef>
#include <cstdint>
#include <cusbf/device_span.cuh>
#include <cusbf/normalized_record_batch.hpp>

namespace cusbf::detail {

struct QueryLayoutRecord {
    uint64_t record_index{};
    uint64_t input_offset{};
    uint64_t size{};
    uint64_t valid_kmers{};
    uint64_t hit_offset{};
    uint64_t hit_count{};
};

class QueryLayout {
   public:
    [[nodiscard]] std::span<const QueryLayoutRecord> records() const noexcept {
        return std::span<const QueryLayoutRecord>{records_.data(), records_.size()};
    }

    [[nodiscard]] uint64_t total_hit_count() const noexcept {
        return total_hit_count_;
    }

    [[nodiscard]] std::span<const uint8_t>
    hits_for_record(std::span<const uint8_t> hits, size_t layout_index) const noexcept {
        const QueryLayoutRecord& record = records_[layout_index];
        return std::span<const uint8_t>{
            hits.data() + static_cast<ptrdiff_t>(record.hit_offset),
            static_cast<size_t>(record.hit_count)
        };
    }

    template <typename Config>
    [[nodiscard]] static QueryLayout build(std::span<const NormalizedRecord> records) {
        QueryLayout layout;
        layout.records_.reserve(records.size());

        uint64_t logical_size = 0;
        for (const NormalizedRecord& record : records) {
            if (logical_size != 0) {
                const uint64_t remainder = logical_size % Config::symbolWidth;
                if (remainder != 0) {
                    logical_size += Config::symbolWidth - remainder;
                }
                logical_size += Config::symbolWidth;
            }

            const uint64_t hit_count = record_kmer_count<Config>(record.size);
            layout.records_.push_back(
                QueryLayoutRecord{
                    record.record_index,
                    record.input_offset,
                    record.size,
                    record.valid_kmers,
                    logical_size / Config::symbolWidth,
                    hit_count,
                }
            );
            logical_size += record.size;
        }

        layout.total_hit_count_ = record_kmer_count<Config>(logical_size);
        return layout;
    }

   private:
    std::vector<QueryLayoutRecord> records_{};
    uint64_t total_hit_count_{};
};

}  // namespace cusbf::detail
