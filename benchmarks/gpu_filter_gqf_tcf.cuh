#pragma once

#include <cuda/__cmath/ceil_div.h>
#include <cuda_runtime.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <cuda/std/functional>

#include <algorithm>
#include <cmath>
#include <cstdint>

#include <bulk_tcf_host.cuh>
#include <gqf.cuh>
#include <gqf_int.cuh>

#include <cusbf/helpers.cuh>

namespace gpu_filter_gqf_tcf {

constexpr double kLoadFactor = 0.95;

using TcfFilter = host_bulk_tcf<uint64_t, uint16_t>;

inline size_t gqfFilterBytes(const QF* devQf) {
    QF hostQf{};
    CUSBF_CUDA_CALL(cudaMemcpy(&hostQf, devQf, sizeof(QF), cudaMemcpyDeviceToHost));

    qfmetadata metadata{};
    CUSBF_CUDA_CALL(
        cudaMemcpy(&metadata, hostQf.metadata, sizeof(qfmetadata), cudaMemcpyDeviceToHost)
    );
    return metadata.total_size_in_bytes;
}

inline void convertGqfResults(thrust::device_vector<uint64_t>& results) {
    thrust::transform(
        results.begin(), results.end(), results.begin(), [] __device__(uint64_t value) {
            return value > 0 ? 1ULL : 0ULL;
        }
    );
}

inline uint32_t gqfExponent(uint64_t minCapacity) {
    if (minCapacity <= 1) {
        return 0;
    }
    return 64 - static_cast<uint32_t>(__builtin_clzll(minCapacity - 1));
}

inline uint64_t gqfCapacity(uint32_t exponent) {
    return 1ULL << exponent;
}

inline uint64_t gqfMinCapacityForItems(uint64_t numItems) {
    return static_cast<uint64_t>(std::ceil(static_cast<double>(numItems) / kLoadFactor));
}

inline uint64_t gqfCapacityForFilterBits(uint64_t filterBits) {
    const uint64_t minCapacity =
        cuda::ceil_div(std::max(filterBits, uint64_t{1}), static_cast<uint64_t>(QF_BITS_PER_SLOT));
    return 1ULL << gqfExponent(minCapacity);
}

inline uint64_t tcfCapacityForItems(uint64_t numItems) {
    return static_cast<uint64_t>(std::ceil(static_cast<double>(numItems) / kLoadFactor));
}

inline uint64_t tcfCapacityForFilterBits(uint64_t filterBits) {
    return cuda::ceil_div(
        std::max(filterBits, uint64_t{1}), static_cast<uint64_t>(sizeof(uint16_t) * 8)
    );
}

struct GqfHandle {
    QF* filter = nullptr;
    uint32_t exponent = 0;
    uint64_t capacity = 0;

    void createForExponent(uint32_t q) {
        destroy();
        exponent = q;
        capacity = gqfCapacity(exponent);
        qf_malloc_device(&filter, exponent, true);
        CUSBF_CUDA_CALL(cudaDeviceSynchronize());
    }

    void createForCapacity(uint64_t minCapacity) {
        createForExponent(gqfExponent(minCapacity));
    }

    void createForItems(uint64_t numItems) {
        createForCapacity(std::max(gqfMinCapacityForItems(numItems), uint64_t{1}));
    }

    void createForFilterBits(uint64_t filterBits) {
        const uint64_t minCapacity = cuda::ceil_div(
            std::max(filterBits, uint64_t{1}), static_cast<uint64_t>(QF_BITS_PER_SLOT)
        );
        createForCapacity(minCapacity);
    }

    [[nodiscard]] size_t filterBytes() const {
        return filter == nullptr ? 0 : gqfFilterBytes(filter);
    }

    [[nodiscard]] size_t logicalFilterBytes() const {
        return static_cast<size_t>(filterBits() / 8);
    }

    [[nodiscard]] uint64_t filterBits() const {
        return capacity * static_cast<uint64_t>(QF_BITS_PER_SLOT);
    }

    void destroy() {
        if (filter != nullptr) {
            qf_destroy_device(filter);
            CUSBF_CUDA_CALL(cudaFree(filter));
            filter = nullptr;
        }
        exponent = 0;
        capacity = 0;
    }

    ~GqfHandle() {
        destroy();
    }
};

struct TcfHandle {
    TcfFilter* filter = nullptr;
    uint64_t capacity = 0;
    uint64_t* misses = nullptr;

    void createForCapacity(uint64_t newCapacity) {
        destroy();
        capacity = newCapacity;
        filter = TcfFilter::host_build_tcf(capacity);
        CUSBF_CUDA_CALL(cudaMalloc(&misses, sizeof(uint64_t)));
        CUSBF_CUDA_CALL(cudaDeviceSynchronize());
    }

    void createForItems(uint64_t numItems) {
        createForCapacity(tcfCapacityForItems(numItems));
    }

    void createForFilterBits(uint64_t filterBits) {
        createForCapacity(tcfCapacityForFilterBits(filterBits));
    }

    [[nodiscard]] size_t filterBytes() const {
        return capacity * sizeof(uint16_t);
    }

    [[nodiscard]] uint64_t filterBits() const {
        return capacity * sizeof(uint16_t) * 8;
    }

    void destroy() {
        if (misses != nullptr) {
            CUSBF_CUDA_CALL(cudaFree(misses));
            misses = nullptr;
        }
        if (filter != nullptr) {
            TcfFilter::host_free_tcf(filter);
            filter = nullptr;
        }
        capacity = 0;
    }

    ~TcfHandle() {
        destroy();
    }

    void bulkInsert(uint64_t* keys, uint64_t count) const {
        CUSBF_CUDA_CALL(cudaMemset(misses, 0, sizeof(uint64_t)));
        filter->bulk_insert(keys, count, misses);
        CUSBF_CUDA_CALL(cudaDeviceSynchronize());
    }

    bool* bulkQuery(uint64_t* keys, uint64_t count) const {
        bool* hits = filter->bulk_query(keys, count);
        CUSBF_CUDA_CALL(cudaDeviceSynchronize());
        return hits;
    }
};

inline void copyPackedKmers(
    const thrust::device_vector<uint64_t>& source,
    thrust::device_vector<uint64_t>& scratch
) {
    scratch.resize(source.size());
    thrust::copy(source.begin(), source.end(), scratch.begin());
}

inline void gqfBulkInsert(GqfHandle& handle, uint64_t* keys, uint64_t count) {
    bulk_insert(handle.filter, count, keys, 0);
    CUSBF_CUDA_CALL(cudaDeviceSynchronize());
}

inline void gqfBulkGet(GqfHandle& handle, uint64_t count, uint64_t* keys, uint64_t* results) {
    bulk_get(handle.filter, count, keys, results);
    CUSBF_CUDA_CALL(cudaDeviceSynchronize());
}

}  // namespace gpu_filter_gqf_tcf
