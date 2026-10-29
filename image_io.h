#ifndef CNOCR_IMAGE_IO_H
#define CNOCR_IMAGE_IO_H

#include <filesystem>
#include <opencv2/core.hpp>

namespace imageio {

/// @brief 读取图片为 BGR，路径按 UTF-8 解释（兼容 Windows 非 ASCII 路径）
cv::Mat imread_utf8(const std::filesystem::path& path);

}  // namespace imageio

#endif  // CNOCR_IMAGE_IO_H
