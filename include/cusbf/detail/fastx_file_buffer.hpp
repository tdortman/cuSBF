#pragma once

#include <cstdint>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#if defined(__linux__)
    #include <fcntl.h>
    #include <sys/mman.h>
    #include <sys/stat.h>
    #include <unistd.h>
#endif

#include <cusbf/gzstreambuf.hpp>

namespace cusbf::detail {

/// @brief Read-only contiguous file payload for in-memory FASTX parsing.
class FastxFileBuffer {
   public:
    FastxFileBuffer() = default;
    FastxFileBuffer(const FastxFileBuffer&) = delete;
    FastxFileBuffer& operator=(const FastxFileBuffer&) = delete;
    FastxFileBuffer(FastxFileBuffer&& other) noexcept {
        *this = std::move(other);
    }

    FastxFileBuffer& operator=(FastxFileBuffer&& other) noexcept {
        if (this != &other) {
            release();
            owned_storage_ = std::move(other.owned_storage_);
            data_ = other.data_;
            size_ = other.size_;
#if defined(__linux__)
            mapped_ = other.mapped_;
            mapped_size_ = other.mapped_size_;
            other.mapped_ = nullptr;
            other.mapped_size_ = 0;
#endif
            other.data_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    ~FastxFileBuffer() {
        release();
    }

    [[nodiscard]] static std::unique_ptr<FastxFileBuffer> load(std::string_view path) {
        auto buffer = std::make_unique<FastxFileBuffer>();
        buffer->load_from_path(path);
        return buffer;
    }

    [[nodiscard]] std::string_view data() const noexcept {
        return std::string_view{data_, size_};
    }

    [[nodiscard]] bool empty() const noexcept {
        return size_ == 0;
    }

   private:
    std::vector<char> owned_storage_;
    const char* data_{nullptr};
    size_t size_{0};

#if defined(__linux__)
    void* mapped_{nullptr};
    size_t mapped_size_{0};
#endif

    void release() {
#if defined(__linux__)
        if (mapped_ != nullptr) {
            munmap(mapped_, mapped_size_);
            mapped_ = nullptr;
            mapped_size_ = 0;
        }
#endif
        owned_storage_.clear();
        data_ = nullptr;
        size_ = 0;
    }

    void load_from_path(std::string_view path) {
        const std::string path_string(path);

#if defined(__linux__)
        const int fd = ::open(path_string.c_str(), O_RDONLY);
        if (fd == -1) {
            throw std::runtime_error("Failed to open FASTA/FASTQ file: " + path_string);
        }

        struct stat file_status{};
        if (::fstat(fd, &file_status) != 0 || file_status.st_size < 0) {
            ::close(fd);
            throw std::runtime_error("Failed to stat FASTA/FASTQ file: " + path_string);
        }

        if (file_status.st_size == 0) {
            ::close(fd);
            return;
        }

        mapped_size_ = static_cast<size_t>(file_status.st_size);
        mapped_ = ::mmap(nullptr, mapped_size_, PROT_READ, MAP_PRIVATE, fd, 0);
        ::close(fd);
        if (mapped_ == MAP_FAILED) {
            mapped_ = nullptr;
            mapped_size_ = 0;
            throw std::runtime_error("Failed to mmap FASTA/FASTQ file: " + path_string);
        }

        data_ = static_cast<const char*>(mapped_);
        size_ = mapped_size_;
        return;
#endif

        std::ifstream input(path_string, std::ios::binary);
        if (!input.is_open()) {
            throw std::runtime_error("Failed to open FASTA/FASTQ file: " + path_string);
        }
        input.seekg(0, std::ios::end);
        const auto file_size = input.tellg();
        if (file_size < 0) {
            throw std::runtime_error("Failed to size FASTA/FASTQ file: " + path_string);
        }
        owned_storage_.resize(static_cast<size_t>(file_size));
        input.seekg(0, std::ios::beg);
        if (!owned_storage_.empty()) {
            input.read(owned_storage_.data(), static_cast<std::streamsize>(owned_storage_.size()));
            if (!input) {
                throw std::runtime_error("Failed to read FASTA/FASTQ file: " + path_string);
            }
        }
        data_ = owned_storage_.data();
        size_ = owned_storage_.size();
    }
};

[[nodiscard]] inline bool fastx_file_supports_memory_map(std::string_view path) {
    return !isGzipFile(path);
}

}  // namespace cusbf::detail
