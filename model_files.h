#ifndef CNOCR_MODEL_FILES_H
#define CNOCR_MODEL_FILES_H

#include <filesystem>
#include <string>

namespace modelfiles {

/// @brief 模型缓存根目录
/// @note 优先使用环境变量 CNOCR_HOME；否则 Windows 为 %APPDATA%/cnocr，其他平台为 ~/.cnocr
std::filesystem::path data_dir();

/// @brief 下载 url 到 dest（先写 .part 再改名），依次尝试 curl / wget
/// @throw std::runtime_error 下载失败
void download(const std::string& url, const std::filesystem::path& dest);

/// @brief 从 zip 中解出第一个 .onnx 文件到 out_path
/// @throw std::runtime_error 压缩包损坏或解压失败
void extract_onnx(const std::filesystem::path& zip_path,
                  const std::filesystem::path& out_path);

}  // namespace modelfiles

#endif  // CNOCR_MODEL_FILES_H
