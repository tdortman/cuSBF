#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <format>
#include <istream>
#include <memory>
#include <streambuf>
#include <string_view>

#include <zlib.h>
#include <cusbf/error.hpp>

namespace cusbf::detail {

/// @brief @c std::streambuf over a gzip file (zlib @c gzread).
class GzStreambuf : public std::streambuf {
   public:
    GzStreambuf(const GzStreambuf&) = delete;
    GzStreambuf& operator=(const GzStreambuf&) = delete;
    GzStreambuf(GzStreambuf&&) = delete;
    GzStreambuf& operator=(GzStreambuf&&) = delete;

    /**
     * @brief Opens @p path for binary gzip read.
     *
     * @param path Filesystem path to a @c .gz file.
     */
    [[nodiscard]] static Result<std::unique_ptr<GzStreambuf>> open(
        const std::filesystem::path& path
    ) {
        auto streambuf = std::unique_ptr<GzStreambuf>(new GzStreambuf(path));
        if (!streambuf->file_) {
            return Err(Error::io(std::format("Failed to open gzip file: {}", path.string())));
        }
        return streambuf;
    }

    ~GzStreambuf() override {
        if (file_) {
            gzclose(file_);
        }
    }

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

    explicit GzStreambuf(const std::filesystem::path& path)
        : file_(gzopen(path.string().c_str(), "rb")) {
        setg(buffer_, buffer_, buffer_);
    }

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
    [[nodiscard]] static Result<std::unique_ptr<GzIstream>> open(
        const std::filesystem::path& path
    ) {
        auto streambuf = GzStreambuf::open(path);
        if (!streambuf) {
            return Err(streambuf.error());
        }
        return std::unique_ptr<GzIstream>(new GzIstream(std::move(*streambuf)));
    }

   private:
    explicit GzIstream(std::unique_ptr<GzStreambuf> streambuf)
        : std::istream(nullptr), sb_(std::move(streambuf)) {
        rdbuf(sb_.get());
    }

    std::unique_ptr<GzStreambuf> sb_;
};

/// @brief True when @p path begins with the gzip magic bytes (@c 0x1F, @c 0x8B).
inline bool isGzipFile(const std::filesystem::path& path) {
    FILE* f = std::fopen(path.string().c_str(), "rb");
    if (!f) {
        return false;
    }
    uint8_t magic[2];
    size_t n = std::fread(magic, 1, 2, f);
    std::fclose(f);
    return n == 2 && magic[0] == 0x1F && magic[1] == 0x8B;
}

}  // namespace cusbf::detail
