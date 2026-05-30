#pragma once

#include <type_traits>

#include <cusbf/Fastx.hpp>
#include <cusbf/detail/fastx_buffer_reader.hpp>
#include <cusbf/error.hpp>

namespace cusbf::detail {

/// @brief Appends the next FASTX record from @p reader into @p chunk.
template <typename FastxReaderType>
[[nodiscard]] inline Result<bool> collect_next_fastx_record(
    FastxReaderType& reader,
    FastxRecord& record,
    DenseRecordBatchBuilder& chunk
) {
    if constexpr (std::is_same_v<std::decay_t<FastxReaderType>, FastxBufferReader>) {
        const auto range = CUSBF_TRY(
            reader.appendNextRecord(record, chunk.sequence_buffer(), chunk.external_sequence_slot())
        );
        if (!range) {
            return false;
        }
        chunk.push_range(*range);
        return true;
    }

    const bool has_record = CUSBF_TRY(reader.nextRecord(record));
    if (!has_record) {
        return false;
    }
    chunk.appendRecord(record.sequence);
    return true;
}

}  // namespace cusbf::detail
