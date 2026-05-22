#pragma once

#include <cuda_runtime.h>
#include <cuda/stream>

#include <array>
#include <utility>

#include <cusbf/helpers.cuh>

namespace cusbf::detail {

/// @brief Two non-blocking CUDA streams for overlapping chunk H2D and kernel work.
class ChunkStreamPair {
   public:
    ChunkStreamPair() {
        for (cudaStream_t& stream : streams_) {
            CUSBF_CUDA_CALL(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        }
    }

    ChunkStreamPair(const ChunkStreamPair&) = delete;
    ChunkStreamPair& operator=(const ChunkStreamPair&) = delete;

    ChunkStreamPair(ChunkStreamPair&& other) noexcept : streams_{other.streams_} {
        for (cudaStream_t& stream : other.streams_) {
            stream = nullptr;
        }
    }

    ChunkStreamPair& operator=(ChunkStreamPair&& other) noexcept {
        if (this != &other) {
            destroy();
            streams_ = other.streams_;
            for (cudaStream_t& stream : other.streams_) {
                stream = nullptr;
            }
        }
        return *this;
    }

    ~ChunkStreamPair() {
        destroy();
    }

    [[nodiscard]] cuda::stream_ref operator[](size_t index) const noexcept {
        return {streams_.at(index)};
    }

    void sync_all() const {
        for (cudaStream_t stream : streams_) {
            if (stream != nullptr) {
                CUSBF_CUDA_CALL(cudaStreamSynchronize(stream));
            }
        }
    }

   private:
    std::array<cudaStream_t, 2> streams_{nullptr, nullptr};

    void destroy() {
        for (cudaStream_t& stream : streams_) {
            if (stream != nullptr) {
                CUSBF_CUDA_CALL(cudaStreamDestroy(stream));
                stream = nullptr;
            }
        }
    }
};

}  // namespace cusbf::detail
