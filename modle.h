#ifndef CNOCR_MODELE_H
#define CNOCR_MODELE_H

#include <onnxruntime_cxx_api.h>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace cnocrmodle {

enum class USE_DEVICE {
    CPU,
    CUDA,
    TensorRT,
};

struct rec_return {
    // [seq_len, num_classes]，行优先
    std::vector<float> logits;
    // 有效长度（模型 output_lengths）；为空表示与 seq_len 相同
    std::vector<int64_t> lengths;
    int64_t seq_len = 0;
    int64_t num_classes = 0;
    int64_t valid_len = 0;
};

struct det_return {
    // [height, width]
    std::vector<float> prob;
    int64_t height = 0;
    int64_t width = 0;
};

class onnxmodle {
public:
    explicit onnxmodle(const std::wstring& model_path, USE_DEVICE device = USE_DEVICE::CPU);
    ~onnxmodle();

    onnxmodle(const onnxmodle&) = delete;
    onnxmodle& operator=(const onnxmodle&) = delete;

    /// @brief 识别模型推理
    /// @param image HWC 布局的 8bit 图像数据
    /// @param height 图像高度（模型输入高度）
    /// @param width 图像宽度
    /// @param channels 1：灰度（x/255）；3：BGR（(x/255-0.5)/0.5，PaddleOCR 系模型）
    rec_return run_rec(const uint8_t* image, int height, int width, int channels);

    /// @brief 检测模型推理
    /// @param chw 已归一化的 [3, height, width] float 数据
    det_return run_det(float* chw, int height, int width);

private:
    std::vector<Ort::Value> run_model(std::vector<Ort::Value>& inputs) const;

    Ort::Env env_;
    Ort::SessionOptions session_options_;
    std::unique_ptr<Ort::Session> session_;
    std::vector<std::string> input_name_storage_;
    std::vector<std::string> output_name_storage_;
    std::vector<const char*> input_names_;
    std::vector<const char*> output_names_;
    std::vector<std::vector<int64_t>> input_shapes_;
};

}  // namespace cnocrmodle

#endif  // CNOCR_MODELE_H
