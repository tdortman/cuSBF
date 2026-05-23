#pragma once

#include <cuda_runtime.h>
#include <cuda/std/expected>

#include <cstdint>
#include <cstdio>
#include <format>
#include <source_location>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

namespace cusbf {

/// @brief Formats a call site as @c file:line:col (compiler-diagnostic style).
[[nodiscard]] inline std::string format_source_location(std::source_location location) {
    return std::format("{}:{}:{}", location.file_name(), location.line(), location.column());
}

/// @brief Error category for @ref Result failures.
enum class ErrorCategory : uint8_t {
    /// CUDA runtime API failure (@ref Error::cuda_code).
    cuda,
    /// File or stream I/O failure.
    io,
    /// FASTA/FASTQ parse failure (@ref Error::line).
    fastx_parse,
    /// Invalid caller input (batch layout, buffer sizes, etc.).
    invalid_argument,
    /// Resource limit (typically GPU staging budget).
    resource,
};

/// @brief Error payload carried in @ref Result on failure.
struct Error {
    /// Failure kind.
    ErrorCategory category{};
    /// CUDA error code when @ref category is @ref ErrorCategory::cuda.
    cudaError_t cuda_code{cudaSuccess};
    /// Source line for FASTX parse errors.
    uint64_t line{};
    /// Human-readable message.
    std::string message;

    [[nodiscard]] static Error
    cuda(cudaError_t code, std::source_location location = std::source_location::current()) {
        return Error{
            ErrorCategory::cuda,
            code,
            static_cast<uint64_t>(location.line()),
            std::format("{}: {}", format_source_location(location), cudaGetErrorString(code)),
        };
    }

    [[nodiscard]] static Error io(std::string message) {
        return Error{ErrorCategory::io, cudaSuccess, 0, std::move(message)};
    }

    [[nodiscard]] static Error
    fastx_parse(std::string_view source, uint64_t line_number, std::string_view detail) {
        return Error{
            ErrorCategory::fastx_parse,
            cudaSuccess,
            line_number,
            std::format("{}:{}: {}", source, line_number, detail),
        };
    }

    [[nodiscard]] static Error invalid_argument(std::string message) {
        return Error{ErrorCategory::invalid_argument, cudaSuccess, 0, std::move(message)};
    }

    [[nodiscard]] static Error resource(std::string message) {
        return Error{ErrorCategory::resource, cudaSuccess, 0, std::move(message)};
    }
};

/// @brief Success/failure result using libcudacxx @c cuda::std::expected.
template <typename T>
using Result = cuda::std::expected<T, Error>;

/// @brief Checks a CUDA runtime call and returns an error on failure.
[[nodiscard]] inline Result<void>
cuda_try(cudaError_t error, std::source_location location = std::source_location::current()) {
    if (error == cudaSuccess) {
        return {};
    }
    return cuda::std::unexpected(Error::cuda(error, location));
}

/// @brief Aborts when a @ref Result is unsuccessful (benchmarks and tests).
inline void require_void(const Result<void>& result) {
    if (!result) {
        std::fputs(result.error().message.c_str(), stderr);
        std::fputc('\n', stderr);
        std::abort();
    }
}

/// @brief Terminates the process when @p error is not @c cudaSuccess (for RAII teardown only).
inline void cuda_abort_on_error(
    cudaError_t error,
    std::source_location location = std::source_location::current()
) {
    if (error == cudaSuccess) {
        return;
    }
    const Error err = Error::cuda(error, location);
    std::fputs(err.message.c_str(), stderr);
    std::fputc('\n', stderr);
    std::abort();
}

namespace detail {

template <typename T>
[[nodiscard]] T try_unwrap_success(Result<T>&& result) {
    return std::move(*result);
}

inline void try_unwrap_success(Result<void>&& result) {
    (void)result;
}

}  // namespace detail

}  // namespace cusbf

/// @brief Propagates a @ref cusbf::Result failure from the enclosing function (GNU statement expression).
///
/// On failure, returns @c cuda::std::unexpected(...) from the caller. On success, yields the value for
/// valued results, or nothing for @c Result<void>. Usable as a statement (@c CUSBF_TRY(expr);) or in
/// initializers (@c auto x = CUSBF_TRY(expr);).
#define CUSBF_TRY(expr)                                                          \
    ({                                                                           \
        auto _cusbf_result = (expr);                                             \
        if (!_cusbf_result) {                                                    \
            return cuda::std::unexpected(std::move(_cusbf_result).error());      \
        }                                                                        \
        ::cusbf::detail::try_unwrap_success(std::move(_cusbf_result));            \
    })

/// @brief Unwraps a @ref cusbf::Result or throws @c std::runtime_error on failure (tests and apps).
#define CUSBF_UNWRAP(expr)                                                       \
    ({                                                                           \
        auto _cusbf_result = (expr);                                             \
        if (!_cusbf_result) {                                                    \
            throw std::runtime_error(std::move(_cusbf_result).error().message);  \
        }                                                                        \
        ::cusbf::detail::try_unwrap_success(std::move(_cusbf_result));            \
    })

/// @brief Checks a CUDA call and aborts on failure (destructors and RAII only).
#define CUSBF_CUDA_ABORT(expr) ::cusbf::cuda_abort_on_error((expr))

/// @brief Propagates a CUDA error wrapped in @ref cusbf::Result<void>.
#define CUSBF_CUDA_TRY(expr) CUSBF_TRY(::cusbf::cuda_try((expr)))
