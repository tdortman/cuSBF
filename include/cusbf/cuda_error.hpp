#pragma once

#include <cusbf/error.hpp>

#include <stdexcept>

/// @brief Checks a CUDA runtime call and throws @c std::runtime_error on failure.
///
/// Use in non-@ref cusbf::Result functions (benchmarks, RAII helpers). Prefer @ref CUSBF_CUDA_TRY
/// inside functions that return @ref cusbf::Result.
#define CUSBF_CUDA_CALL(err)                                                      \
    do {                                                                          \
        if (cudaError_t _cusbf_err = (err); _cusbf_err != cudaSuccess) {          \
            throw std::runtime_error(::cusbf::Error::cuda(_cusbf_err).message()); \
        }                                                                         \
    } while (0)
