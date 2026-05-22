#pragma once

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace cusbf {

/// @brief Exception thrown on CUDA runtime errors.
class CudaError : public std::runtime_error {
   public:
    CudaError(cudaError_t code, const char* file, int line)
        : std::runtime_error(
              std::string(file) + ":" + std::to_string(line) + " " + cudaGetErrorString(code)
          ),
          code_(code) {
    }

    [[nodiscard]] cudaError_t code() const noexcept {
        return code_;
    }

   private:
    cudaError_t code_;
};

}  // namespace cusbf

/// @brief Macro for checking CUDA runtime errors. Safe to include from host C++ translation units.
#define CUSBF_CUDA_CALL(err)                              \
    do {                                                  \
        cudaError_t err_ = (err);                         \
        if (err_ == cudaSuccess) [[likely]] {             \
            break;                                        \
        }                                                 \
        throw cusbf::CudaError(err_, __FILE__, __LINE__); \
    } while (0)
