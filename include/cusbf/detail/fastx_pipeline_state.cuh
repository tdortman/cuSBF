#pragma once

#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <thrust/device_vector.h>

#include <cusbf/detail/fastx_pinned_buffer.hpp>
#include <cusbf/detail/query_layout.cuh>
#include <cusbf/device_span.cuh>
#include <cusbf/error.hpp>
#include <cusbf/normalized_record_batch.hpp>
namespace cusbf::detail {

class FastxPipelineState {
   public:
    FastxPipelineState() = default;
    FastxPipelineState(const FastxPipelineState&) = delete;
    FastxPipelineState& operator=(const FastxPipelineState&) = delete;
    FastxPipelineState(FastxPipelineState&&) noexcept = default;
    FastxPipelineState& operator=(FastxPipelineState&&) noexcept = default;
    ~FastxPipelineState() = default;

    [[nodiscard]] thrust::device_vector<char>& sequence_device() noexcept {
        return d_sequence_;
    }

    [[nodiscard]] const thrust::device_vector<char>& sequence_device() const noexcept {
        return d_sequence_;
    }

    [[nodiscard]] thrust::device_vector<uint64_t>& dense_packed_words_device() noexcept {
        return d_dense_packed_words_;
    }

    [[nodiscard]] const thrust::device_vector<uint64_t>&
    dense_packed_words_device() const noexcept {
        return d_dense_packed_words_;
    }

    [[nodiscard]] thrust::device_vector<uint8_t>& result_buffer_device() noexcept {
        return d_resultBuffer_;
    }

    [[nodiscard]] const thrust::device_vector<uint8_t>& result_buffer_device() const noexcept {
        return d_resultBuffer_;
    }

    [[nodiscard]] thrust::device_vector<QueryLayoutRecord>& query_layout_records_device() noexcept {
        return d_query_layout_records_;
    }

    [[nodiscard]] const thrust::device_vector<QueryLayoutRecord>&
    query_layout_records_device() const noexcept {
        return d_query_layout_records_;
    }

    [[nodiscard]] thrust::device_vector<uint64_t>& record_positive_kmers_device() noexcept {
        return d_record_positive_kmers_;
    }

    [[nodiscard]] const thrust::device_vector<uint64_t>&
    record_positive_kmers_device() const noexcept {
        return d_record_positive_kmers_;
    }

    [[nodiscard]] FastxPinnedSequenceBuffer& normalized_sequence_ping(size_t ping) noexcept {
        return normalized_sequence_pings_[ping & 1U];
    }

    [[nodiscard]] const FastxPinnedSequenceBuffer& normalized_sequence_ping(
        size_t ping
    ) const noexcept {
        return normalized_sequence_pings_[ping & 1U];
    }

    [[nodiscard]] std::vector<NormalizedRecord>& normalized_records_ping(size_t ping) noexcept {
        return normalized_records_pings_[ping & 1U];
    }

    [[nodiscard]] const std::vector<NormalizedRecord>& normalized_records_ping(
        size_t ping
    ) const noexcept {
        return normalized_records_pings_[ping & 1U];
    }

    [[nodiscard]] std::string& normalized_sequence_scratch() noexcept {
        return normalized_sequence_scratch_;
    }

    [[nodiscard]] const std::string& normalized_sequence_scratch() const noexcept {
        return normalized_sequence_scratch_;
    }

    [[nodiscard]] std::vector<NormalizedRecord>& normalized_records_scratch() noexcept {
        return normalized_records_scratch_;
    }

    [[nodiscard]] const std::vector<NormalizedRecord>& normalized_records_scratch() const noexcept {
        return normalized_records_scratch_;
    }

    [[nodiscard]] std::vector<uint64_t>& record_positive_kmers_scratch() noexcept {
        return record_positive_kmers_scratch_;
    }

    [[nodiscard]] const std::vector<uint64_t>& record_positive_kmers_scratch() const noexcept {
        return record_positive_kmers_scratch_;
    }

    [[nodiscard]] std::vector<uint8_t>& result_hits_scratch() noexcept {
        return result_hits_scratch_;
    }

    [[nodiscard]] const std::vector<uint8_t>& result_hits_scratch() const noexcept {
        return result_hits_scratch_;
    }

    void ensure_sequence_capacity(uint64_t bases) {
        if (bases > d_sequence_.size()) {
            d_sequence_.resize(bases);
        }
    }

    void ensure_dense_packed_capacity(uint64_t words) {
        if (words > d_dense_packed_words_.size()) {
            d_dense_packed_words_.resize(words);
        }
    }

    void ensure_result_capacity(uint64_t kmers) {
        if (kmers > d_resultBuffer_.size()) {
            d_resultBuffer_.resize(kmers);
        }
    }

    [[nodiscard]] Result<void>
    stage_sequence(std::span<const char> sequence, cuda::stream_ref stream) {
        ensure_sequence_capacity(sequence.size());
        CUSBF_CUDA_TRY(cudaMemcpyAsync(
            thrust::raw_pointer_cast(d_sequence_.data()),
            sequence.data(),
            sequence.size_bytes(),
            cudaMemcpyHostToDevice,
            stream.get()
        ));
        return {};
    }

    [[nodiscard]] Result<device_span<const char>>
    staged_sequence_view(std::span<const char> sequence, cuda::stream_ref stream) {
        CUSBF_TRY(stage_sequence(sequence, stream));
        return device_span<const char>{
            thrust::raw_pointer_cast(d_sequence_.data()), sequence.size()
        };
    }

    [[nodiscard]] Result<void>
    stage_dense_packed(std::span<const uint64_t> words, cuda::stream_ref stream) {
        ensure_dense_packed_capacity(words.size());
        CUSBF_CUDA_TRY(cudaMemcpyAsync(
            thrust::raw_pointer_cast(d_dense_packed_words_.data()),
            words.data(),
            words.size_bytes(),
            cudaMemcpyHostToDevice,
            stream.get()
        ));
        return {};
    }

    [[nodiscard]] Result<device_span<const uint64_t>>
    staged_dense_packed_view(std::span<const uint64_t> words, cuda::stream_ref stream) {
        CUSBF_TRY(stage_dense_packed(words, stream));
        return device_span<const uint64_t>{
            thrust::raw_pointer_cast(d_dense_packed_words_.data()), words.size()
        };
    }

    [[nodiscard]] Result<device_span<const char>>
    stage_sequence_ping(size_t ping, std::string_view sequence, cuda::stream_ref stream) {
        auto& buffer = d_sequence_pings_[ping & 1U];
        if (sequence.size() > buffer.size()) {
            buffer.resize(sequence.size());
        }
        CUSBF_CUDA_TRY(cudaMemcpyAsync(
            thrust::raw_pointer_cast(buffer.data()),
            sequence.data(),
            sequence.size(),
            cudaMemcpyHostToDevice,
            stream.get()
        ));
        return device_span<const char>{thrust::raw_pointer_cast(buffer.data()), sequence.size()};
    }

    void release_host_pings() {
        for (FastxPinnedSequenceBuffer& sequence : normalized_sequence_pings_) {
            sequence.release();
        }
        for (std::vector<NormalizedRecord>& records : normalized_records_pings_) {
            std::vector<NormalizedRecord>{}.swap(records);
        }
    }

    void release_host_scratch() {
        std::string{}.swap(normalized_sequence_scratch_);
        std::vector<NormalizedRecord>{}.swap(normalized_records_scratch_);
        std::vector<uint64_t>{}.swap(record_positive_kmers_scratch_);
        std::vector<uint8_t>{}.swap(result_hits_scratch_);
    }

    void release_device_staging() {
        thrust::device_vector<char>{}.swap(d_sequence_);
        thrust::device_vector<uint64_t>{}.swap(d_dense_packed_words_);
        for (thrust::device_vector<char>& buffer : d_sequence_pings_) {
            thrust::device_vector<char>{}.swap(buffer);
        }
        thrust::device_vector<uint8_t>{}.swap(d_resultBuffer_);
        thrust::device_vector<QueryLayoutRecord>{}.swap(d_query_layout_records_);
        thrust::device_vector<uint64_t>{}.swap(d_record_positive_kmers_);
    }

    void release_all() {
        release_host_pings();
        release_host_scratch();
        release_device_staging();
    }

    void shrink_host_scratch() {
        release_host_scratch();
    }

    void shrink_device_staging() {
        release_device_staging();
    }

    void shrink_all() {
        release_all();
    }

   private:
    thrust::device_vector<char> d_sequence_;
    thrust::device_vector<uint64_t> d_dense_packed_words_;
    std::array<thrust::device_vector<char>, 2> d_sequence_pings_;
    thrust::device_vector<uint8_t> d_resultBuffer_;
    thrust::device_vector<QueryLayoutRecord> d_query_layout_records_;
    thrust::device_vector<uint64_t> d_record_positive_kmers_;
    std::array<FastxPinnedSequenceBuffer, 2> normalized_sequence_pings_;
    std::array<std::vector<NormalizedRecord>, 2> normalized_records_pings_;
    std::string normalized_sequence_scratch_;
    std::vector<NormalizedRecord> normalized_records_scratch_;
    std::vector<uint64_t> record_positive_kmers_scratch_;
    std::vector<uint8_t> result_hits_scratch_;
};

}  // namespace cusbf::detail
