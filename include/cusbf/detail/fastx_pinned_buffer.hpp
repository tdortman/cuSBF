#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string_view>

#include <cusbf/cuda_error.hpp>

namespace cusbf::detail {

/// @brief Page-locked host buffer for fused FASTX normalize + H2D staging.
class FastxPinnedSequenceBuffer {
   public:
    FastxPinnedSequenceBuffer() = default;
    FastxPinnedSequenceBuffer(const FastxPinnedSequenceBuffer&) = delete;
    FastxPinnedSequenceBuffer& operator=(const FastxPinnedSequenceBuffer&) = delete;

    FastxPinnedSequenceBuffer(FastxPinnedSequenceBuffer&& other) noexcept {
        *this = std::move(other);
    }

    FastxPinnedSequenceBuffer& operator=(FastxPinnedSequenceBuffer&& other) noexcept {
        if (this != &other) {
            release();
            data_ = other.data_;
            size_ = other.size_;
            capacity_ = other.capacity_;
            other.data_ = nullptr;
            other.size_ = 0;
            other.capacity_ = 0;
        }
        return *this;
    }

    ~FastxPinnedSequenceBuffer() {
        release();
    }

    /// @brief Writable pinned host pointer (may be @c nullptr before @ref reserve).
    [[nodiscard]] char* data() noexcept {
        return data_;
    }

    /// @brief Read-only view of @ref data().
    [[nodiscard]] const char* data() const noexcept {
        return data_;
    }

    /// @brief Logical byte length (≤ @ref capacity).
    [[nodiscard]] size_t size() const noexcept {
        return size_;
    }

    /// @brief Allocated pinned bytes.
    [[nodiscard]] size_t capacity() const noexcept {
        return capacity_;
    }

    /// @brief @ref data() as a string view of length @ref size().
    [[nodiscard]] std::string_view view() const noexcept {
        return std::string_view{data_, size_};
    }

    /// @brief Sets logical size to zero without freeing capacity.
    void clear() noexcept {
        size_ = 0;
    }

    /// @brief Frees pinned host memory and resets size and capacity.
    void release() {
        if (data_ != nullptr) {
            CUSBF_CUDA_CALL(cudaFreeHost(data_));
            data_ = nullptr;
        }
        size_ = 0;
        capacity_ = 0;
    }

    /// @brief Grows pinned allocation to at least @p nbytes, preserving existing bytes.
    void reserve(size_t nbytes) {
        if (nbytes <= capacity_) {
            return;
        }

        char* reallocated = nullptr;
        CUSBF_CUDA_CALL(cudaHostAlloc(&reallocated, nbytes, cudaHostAllocDefault));
        if (data_ != nullptr) {
            if (size_ != 0) {
                std::memcpy(reallocated, data_, size_);
            }
            CUSBF_CUDA_CALL(cudaFreeHost(data_));
        }
        data_ = reallocated;
        capacity_ = nbytes;
    }

    /// @brief Sets logical size to @p nbytes, reserving if needed.
    void set_size(size_t nbytes) {
        if (nbytes > capacity_) {
            reserve(nbytes);
        }
        size_ = nbytes;
    }

   private:
    char* data_{nullptr};
    size_t size_{0};
    size_t capacity_{0};
};

}  // namespace cusbf::detail
