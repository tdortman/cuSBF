#pragma once

#include <cstddef>
#include <string_view>
#include <utility>

#include <cuda/stream>

#include <cusbf/detail/chunk_stream_pair.cuh>
#include <cusbf/detail/fastx_chunk.cuh>
#include <cusbf/detail/fastx_dense_batch.hpp>
#include <cusbf/detail/fastx_dispatch.hpp>
#include <cusbf/detail/fastx_pipeline_state.cuh>
#include <cusbf/error.hpp>

namespace cusbf::detail {

class FastxPipelineReleaseGuard {
   public:
    explicit FastxPipelineReleaseGuard(FastxPipelineState& state) : state_(state) {}
    FastxPipelineReleaseGuard(const FastxPipelineReleaseGuard&) = delete;
    FastxPipelineReleaseGuard& operator=(const FastxPipelineReleaseGuard&) = delete;

    ~FastxPipelineReleaseGuard() {
        state_.release_all();
    }

   private:
    FastxPipelineState& state_;
};

template <typename Config, typename FastxReaderType, typename Adapter>
[[nodiscard]] Result<typename Adapter::report_type> run_fastx_pipeline(
    FastxReaderType& reader,
    std::string_view source_name,
    double fill_fraction,
    cuda::stream_ref stream,
    fastx_dispatch_path dispatch_path,
    FastxPipelineState& state,
    Adapter&& adapter
) {
    FastxPipelineReleaseGuard release_guard{state};

    const auto gpu_memory = query_cuda_free_memory();
    if (!gpu_memory) {
        return Err(Error::resource(gpu_memory.error().message()));
    }

    const size_t staging_budget_bytes =
        fastx_staging_budget_bytes<Config>(fill_fraction, gpu_memory->free_bytes);
    const size_t host_chunk_max_bytes = fastx_host_chunk_max_bytes();
    const uint64_t sequence_reserve_bytes =
        fastx_uses_mmap_reader(dispatch_path) ? fastx_file_bytes(source_name) : 0;

    DenseRecordBatchBuilder chunk(sequence_reserve_bytes);
    FastxRecord record;

    auto collect_all = [&](auto&& maybe_flush) -> Result<void> {
        for (;;) {
            const uint64_t local_index = chunk.recordCount();
            if (!CUSBF_TRY(collect_next_fastx_record(reader, record, chunk))) {
                break;
            }
            adapter.on_record_collected(record, local_index, chunk);
            CUSBF_TRY(maybe_flush());
        }
        return {};
    };

    if (fastx_is_single_chunk_path(dispatch_path)) {
        CUSBF_TRY(collect_all([&]() -> Result<void> { return {}; }));
        CUSBF_TRY(adapter.flush_sync(chunk, stream));
        return adapter.finish();
    }

    if (stream.get() == nullptr && adapter.supports_pipelined()) {
        const size_t pipelined_chunk_budget =
            fastx_pipelined_chunk_budget(adapter.chunk_mode(), staging_budget_bytes);
        ChunkStreamPair chunk_streams;
        size_t ping = 0;
        bool has_inflight = false;

        CUSBF_TRY(collect_all([&]() -> Result<void> {
            if (!fastx_chunk_should_flush<Config>(
                    adapter.chunk_mode(),
                    pipelined_chunk_budget,
                    host_chunk_max_bytes,
                    chunk.raw_sequence_bytes(),
                    chunk.recordCount()
                )) {
                return {};
            }
            return adapter.flush_pipelined(chunk, chunk_streams, ping, has_inflight);
        }));

        CUSBF_TRY(adapter.flush_pipelined(chunk, chunk_streams, ping, has_inflight));
        CUSBF_TRY(chunk_streams.sync_all());
        CUSBF_TRY(adapter.finish_pipelined(chunk_streams, ping, has_inflight));
        return adapter.finish();
    }

    const size_t sync_chunk_budget =
        stream.get() == nullptr && !adapter.supports_pipelined()
            ? fastx_pipelined_chunk_budget(adapter.chunk_mode(), staging_budget_bytes)
            : staging_budget_bytes;

    CUSBF_TRY(collect_all([&]() -> Result<void> {
        if (!fastx_chunk_should_flush<Config>(
                adapter.chunk_mode(),
                sync_chunk_budget,
                host_chunk_max_bytes,
                chunk.raw_sequence_bytes(),
                chunk.recordCount()
            )) {
            return {};
        }
        return adapter.flush_sync(chunk, stream);
    }));

    CUSBF_TRY(adapter.flush_sync(chunk, stream));
    return adapter.finish();
}

}  // namespace cusbf::detail
