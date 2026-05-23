#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <istream>
#include <memory>
#include <stdexcept>
#include <streambuf>
#include <string>
#include <string_view>

#include <zlib.h>

namespace cusbf::detail {

/// @brief @c std::streambuf over a gzip file (zlib @c gzread).
class GzStreambuf : public std::streambuf {
   public:
    /**
     * @brief Opens @p path for binary gzip read.
     *
     * @param path Filesystem path to a @c .gz file.
     * @throws std::runtime_error if zlib cannot open the file.
     */
    explicit GzStreambuf(std::string_view path) : file_(gzopen(std::string(path).c_str(), "rb")) {
        if (!file_) {
            throw std::runtime_error("Failed to open gzip file: " + std::string(path));
        }
        setg(buffer_, buffer_, buffer_);
    }

    ~GzStreambuf() override {
        if (file_) {
            gzclose(file_);
        }
    }

    GzStreambuf(const GzStreambuf&) = delete;
    GzStreambuf& operator=(const GzStreambuf&) = delete;
    GzStreambuf(GzStreambuf&&) = delete;
    GzStreambuf& operator=(GzStreambuf&&) = delete;

   protected:
    int_type underflow() override {
        if (gptr() < egptr()) {
            return traits_type::to_int_type(*gptr());
        }

        auto bytesRead = gzread(file_, buffer_, kBufferSize);
        if (bytesRead <= 0) {
            return traits_type::eof();
        }

        setg(buffer_, buffer_, buffer_ + bytesRead);
        return traits_type::to_int_type(*gptr());
    }

   private:
    static constexpr std::size_t kBufferSize = 8192;
    gzFile file_;
    char buffer_[kBufferSize];
};

/// @brief @c std::istream adapter for @ref GzStreambuf.
class GzIstream : public std::istream {
   public:
    /**
     * @brief Opens a gzip file as an input stream.
     *
     * @param path Filesystem path to a @c .gz file.
     */
    explicit GzIstream(std::string_view path)
        : std::istream(nullptr), sb_(std::make_unique<GzStreambuf>(path)) {
        rdbuf(sb_.get());
    }

   private:
    std::unique_ptr<GzStreambuf> sb_;
};

/// @brief True when @p path begins with the gzip magic bytes (@c 0x1F, @c 0x8B).
inline bool isGzipFile(std::string_view path) {
    FILE* f = std::fopen(std::string(path).c_str(), "rb");
    if (!f) {
        return false;
    }
    uint8_t magic[2];
    size_t n = std::fread(magic, 1, 2, f);
    std::fclose(f);
    return n == 2 && magic[0] == 0x1F && magic[1] == 0x8B;
}

}  // namespace cusbf::detail
