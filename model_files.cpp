#include "model_files.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <vector>

#include <zlib.h>

namespace modelfiles {
namespace {

#ifdef _WIN32
using NativeString = std::wstring;
const NativeString kNullRedirect = L" > NUL 2>&1";

NativeString native(const std::string& text) { return std::wstring(text.begin(), text.end()); }
NativeString native(const std::filesystem::path& path) { return path.wstring(); }
NativeString quote(const NativeString& text) { return L"\"" + text + L"\""; }
int run(const NativeString& command) { return _wsystem(command.c_str()); }

std::filesystem::path env_path(const wchar_t* name) {
    const wchar_t* value = _wgetenv(name);
    return value != nullptr && *value != L'\0' ? std::filesystem::path(value)
                                               : std::filesystem::path();
}
#else
using NativeString = std::string;
const NativeString kNullRedirect = " > /dev/null 2>&1";

NativeString native(const std::string& text) { return text; }
NativeString native(const std::filesystem::path& path) { return path.string(); }
NativeString quote(const NativeString& text) {
    NativeString quoted = "'";
    for (char ch : text) {
        if (ch == '\'') {
            quoted += "'\\''";
        } else {
            quoted += ch;
        }
    }
    quoted += "'";
    return quoted;
}
int run(const NativeString& command) { return std::system(command.c_str()); }

std::filesystem::path env_path(const char* name) {
    const char* value = std::getenv(name);
    return value != nullptr && *value != '\0' ? std::filesystem::path(value)
                                              : std::filesystem::path();
}
#endif

bool command_exists(const NativeString& command) {
    return run(command + native(std::string(" --version")) + kNullRedirect) == 0;
}

uint16_t read_le16(const unsigned char* data) {
    return static_cast<uint16_t>(data[0] | (data[1] << 8));
}

uint32_t read_le32(const unsigned char* data) {
    return static_cast<uint32_t>(data[0]) | (static_cast<uint32_t>(data[1]) << 8) |
           (static_cast<uint32_t>(data[2]) << 16) | (static_cast<uint32_t>(data[3]) << 24);
}

bool ends_with_onnx(const std::string& name) {
    constexpr const char* kSuffix = ".onnx";
    constexpr size_t kSuffixLen = 5;
    if (name.size() < kSuffixLen) {
        return false;
    }
    const size_t offset = name.size() - kSuffixLen;
    for (size_t i = 0; i < kSuffixLen; ++i) {
        if (std::tolower(static_cast<unsigned char>(name[offset + i])) != kSuffix[i]) {
            return false;
        }
    }
    return true;
}

struct ZipEntry {
    uint16_t method = 0;
    uint32_t compressed_size = 0;
    uint32_t uncompressed_size = 0;
    uint32_t local_offset = 0;
    std::string name;
};

ZipEntry find_onnx_entry(std::ifstream& in, uint64_t file_size) {
    constexpr uint64_t kMaxEocdSearch = 65557;  // 22 字节固定尾部 + 最长 65535 字节注释
    const uint64_t tail_size = std::min<uint64_t>(file_size, kMaxEocdSearch);
    std::vector<unsigned char> tail(static_cast<size_t>(tail_size));
    in.seekg(static_cast<std::streamoff>(file_size - tail_size));
    in.read(reinterpret_cast<char*>(tail.data()), static_cast<std::streamsize>(tail_size));
    if (!in) {
        throw std::runtime_error("读取压缩包失败");
    }

    std::streamoff eocd = -1;
    for (std::streamoff i = static_cast<std::streamoff>(tail_size) - 22; i >= 0; --i) {
        const unsigned char* p = tail.data() + i;
        if (p[0] == 0x50 && p[1] == 0x4b && p[2] == 0x05 && p[3] == 0x06) {
            eocd = i;
            break;
        }
    }
    if (eocd < 0) {
        throw std::runtime_error("不是有效的 zip 文件");
    }

    const uint64_t directory_offset = read_le32(tail.data() + eocd + 16);
    in.clear();
    in.seekg(static_cast<std::streamoff>(directory_offset));

    unsigned char header[46];
    while (true) {
        in.read(reinterpret_cast<char*>(header), sizeof(header));
        if (in.gcount() != static_cast<std::streamsize>(sizeof(header)) ||
            read_le32(header) != 0x02014b50u) {
            break;
        }
        ZipEntry entry;
        entry.method = read_le16(header + 10);
        entry.compressed_size = read_le32(header + 20);
        entry.uncompressed_size = read_le32(header + 24);
        const uint16_t name_len = read_le16(header + 28);
        const uint16_t extra_len = read_le16(header + 30);
        const uint16_t comment_len = read_le16(header + 32);
        entry.local_offset = read_le32(header + 42);
        entry.name.resize(name_len);
        in.read(entry.name.data(), static_cast<std::streamsize>(name_len));
        if (!in) {
            break;
        }
        in.seekg(static_cast<std::streamoff>(extra_len) + comment_len, std::ios::cur);
        if (!in) {
            break;
        }
        if (ends_with_onnx(entry.name)) {
            return entry;
        }
    }
    throw std::runtime_error("压缩包中没有 .onnx 文件");
}

void inflate_entry(std::ifstream& in, const ZipEntry& entry, std::ofstream& out) {
    z_stream stream{};
    if (inflateInit2(&stream, -MAX_WBITS) != Z_OK) {
        throw std::runtime_error("初始化 zlib 失败");
    }
    struct Guard {
        z_stream* stream;
        ~Guard() { inflateEnd(stream); }
    } guard{&stream};

    std::vector<unsigned char> input(64 * 1024);
    std::vector<unsigned char> output(64 * 1024);
    uint64_t remaining = entry.compressed_size;
    uint64_t written = 0;
    int status = Z_OK;

    while (status != Z_STREAM_END) {
        if (stream.avail_in == 0) {
            const size_t chunk = static_cast<size_t>(std::min<uint64_t>(remaining, input.size()));
            if (chunk == 0) {
                break;
            }
            in.read(reinterpret_cast<char*>(input.data()), static_cast<std::streamsize>(chunk));
            if (in.gcount() != static_cast<std::streamsize>(chunk)) {
                throw std::runtime_error("读取压缩数据失败");
            }
            remaining -= chunk;
            stream.next_in = input.data();
            stream.avail_in = static_cast<uInt>(chunk);
        }
        stream.next_out = output.data();
        stream.avail_out = static_cast<uInt>(output.size());
        status = inflate(&stream, Z_NO_FLUSH);
        if (status != Z_OK && status != Z_STREAM_END) {
            throw std::runtime_error("解压失败，压缩包可能已损坏");
        }
        const size_t produced = output.size() - stream.avail_out;
        out.write(reinterpret_cast<const char*>(output.data()),
                  static_cast<std::streamsize>(produced));
        written += produced;
    }

    if (status != Z_STREAM_END || written != entry.uncompressed_size) {
        throw std::runtime_error("解压结果校验失败，压缩包可能已损坏");
    }
}

void write_entry(std::ifstream& in, const ZipEntry& entry, std::ofstream& out) {
    if (entry.compressed_size == 0xFFFFFFFFu || entry.uncompressed_size == 0xFFFFFFFFu) {
        throw std::runtime_error("暂不支持 ZIP64 压缩包");
    }

    in.clear();
    in.seekg(static_cast<std::streamoff>(entry.local_offset));
    unsigned char header[30];
    in.read(reinterpret_cast<char*>(header), sizeof(header));
    if (in.gcount() != static_cast<std::streamsize>(sizeof(header)) ||
        read_le32(header) != 0x04034b50u) {
        throw std::runtime_error("压缩包结构损坏");
    }
    const uint16_t name_len = read_le16(header + 26);
    const uint16_t extra_len = read_le16(header + 28);
    in.seekg(static_cast<std::streamoff>(entry.local_offset) + 30 + name_len + extra_len);
    if (!in) {
        throw std::runtime_error("压缩包结构损坏");
    }

    if (entry.method == 0) {
        std::vector<char> buffer(64 * 1024);
        uint64_t remaining = entry.compressed_size;
        while (remaining > 0) {
            const size_t chunk = static_cast<size_t>(std::min<uint64_t>(remaining, buffer.size()));
            in.read(buffer.data(), static_cast<std::streamsize>(chunk));
            if (in.gcount() != static_cast<std::streamsize>(chunk)) {
                throw std::runtime_error("读取压缩数据失败");
            }
            out.write(buffer.data(), static_cast<std::streamsize>(chunk));
            remaining -= chunk;
        }
    } else if (entry.method == 8) {
        inflate_entry(in, entry, out);
    } else {
        throw std::runtime_error("不支持的 zip 压缩方式: " + std::to_string(entry.method));
    }
}

}  // namespace

std::filesystem::path data_dir() {
    if (const auto path = env_path(
#ifdef _WIN32
            L"CNOCR_HOME"
#else
            "CNOCR_HOME"
#endif
            );
        !path.empty()) {
        return path;
    }
#ifdef _WIN32
    if (const auto path = env_path(L"APPDATA"); !path.empty()) {
        return path / "cnocr";
    }
    if (const auto path = env_path(L"USERPROFILE"); !path.empty()) {
        return path / ".cnocr";
    }
#else
    if (const auto path = env_path("HOME"); !path.empty()) {
        return path / ".cnocr";
    }
#endif
    return std::filesystem::temp_directory_path() / "cnocr";
}

void download(const std::string& url, const std::filesystem::path& dest) {
    std::error_code ec;
    std::filesystem::create_directories(dest.parent_path(), ec);

    std::filesystem::path part = dest;
    part += ".part";
    std::filesystem::remove(part, ec);

    const NativeString url_arg = quote(native(url));
    const NativeString out_arg = quote(native(part));

    NativeString command;
    if (command_exists(native(std::string("curl")))) {
        command = native(std::string("curl")) +
                  native(std::string(" -fL --retry 3 --connect-timeout 20 -sS -o ")) + out_arg +
                  native(std::string(" ")) + url_arg;
    } else if (command_exists(native(std::string("wget")))) {
        command = native(std::string("wget")) + native(std::string(" -q -O ")) + out_arg +
                  native(std::string(" ")) + url_arg;
    } else {
        throw std::runtime_error(
            "未找到 curl 或 wget，无法自动下载模型；请安装其一，或手动把模型放到 modle/ 目录");
    }

    const int exit_code = run(command);
    const bool ok = exit_code == 0 && std::filesystem::is_regular_file(part, ec) &&
                    std::filesystem::file_size(part, ec) > 0;
    if (!ok) {
        std::filesystem::remove(part, ec);
        throw std::runtime_error("下载失败: " + url);
    }

    std::filesystem::remove(dest, ec);
    std::filesystem::rename(part, dest, ec);
    if (ec) {
        std::filesystem::remove(part, ec);
        throw std::runtime_error("保存模型失败: " + dest.u8string());
    }
}

void extract_onnx(const std::filesystem::path& zip_path,
                  const std::filesystem::path& out_path) {
    std::ifstream in(zip_path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("无法打开压缩包: " + zip_path.u8string());
    }
    in.seekg(0, std::ios::end);
    const uint64_t file_size = static_cast<uint64_t>(in.tellg());
    in.seekg(0, std::ios::beg);

    const ZipEntry entry = find_onnx_entry(in, file_size);

    std::error_code ec;
    std::filesystem::create_directories(out_path.parent_path(), ec);
    std::filesystem::path part = out_path;
    part += ".part";
    {
        std::ofstream out(part, std::ios::binary | std::ios::trunc);
        if (!out) {
            throw std::runtime_error("无法写入模型文件: " + part.u8string());
        }
        write_entry(in, entry, out);
        if (!out) {
            throw std::runtime_error("写入模型文件失败: " + part.u8string());
        }
    }

    std::filesystem::remove(out_path, ec);
    std::filesystem::rename(part, out_path, ec);
    if (ec) {
        std::filesystem::remove(part, ec);
        throw std::runtime_error("保存模型失败: " + out_path.u8string());
    }
}

}  // namespace modelfiles
