#ifndef CNOCR_MODEL_REGISTRY_H
#define CNOCR_MODEL_REGISTRY_H

#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

/// @brief 参考 Python cnocr / cnstd 的模型清单与下载方式：
///        模型压缩包来自 HuggingFace 仓库 breezedeus/cnstd-cnocr-models，
///        默认优先使用国内镜像 hf-mirror.com，可用环境变量覆盖。
namespace cnocr_models {

struct ModelSpec {
    const char* name;         // cnocr 中的模型名
    int channels;             // 输入通道：1 灰度（x/255）；3 彩色（(x/255-0.5)/0.5）
    int height;               // 输入高度：cnocr 自训练模型与 PaddleOCR mobile 系为 32，PP-OCRv3 为 48
    int blank_index;          // CTC blank 下标；负数表示 blank 为最后一类（字典长度）
    const char* onnx_file;    // 本地文件名（modle/ 与缓存目录共用）
    const char* dict_file;    // dict/ 下的字典文件；检测模型为空
    const char* subfolder;    // 仓库内子目录
    const char* archive;      // 压缩包名；空串表示不提供自动下载
};

const std::vector<ModelSpec>& all_recognizers();

const ModelSpec* find_recognizer(std::string_view name);

/// @brief 找到可用的模型文件；modle/ 优先，其次是缓存目录，缺失时按需下载
/// @throw std::runtime_error 模型缺失且无法下载
std::filesystem::path ensure_model(const ModelSpec& spec, bool auto_download);

/// @brief 字典文件路径（相对 dict/）
std::filesystem::path dict_path(const ModelSpec& spec);

/// @brief PP-OCRv3 检测模型（ch_PP-OCRv3_det_infer.onnx）
std::filesystem::path ensure_det_model(bool auto_download);

}  // namespace cnocr_models

#endif  // CNOCR_MODEL_REGISTRY_H
