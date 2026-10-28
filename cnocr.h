#ifndef CNOCR_CNOCR_H
#define CNOCR_CNOCR_H

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/opencv.hpp>

#include "cnstd.h"
#include "modle.h"

class cnocr {
public:
    enum USE_MODLE {
        cnocr136fc,   // 中文识别模型
        en_number,    // 英文数字识别模型
        chinese_cht,  // 繁体中文识别模型
    };

    /// @param device 推理设备；当前 onnxruntime 包为 CPU 版，CUDA/TensorRT 不可用时自动回退
    /// @param use_detection 是否先用 PP-OCRv3 检测模型切分文本行（关闭后使用投影切分）
    explicit cnocr(USE_MODLE mode = USE_MODLE::cnocr136fc,
                   cnocrmodle::USE_DEVICE device = cnocrmodle::USE_DEVICE::CPU,
                   bool use_detection = true);
    ~cnocr();

    cnocr(const cnocr&) = delete;
    cnocr& operator=(const cnocr&) = delete;

    /// @brief 识别整张图片，返回按阅读顺序排列的（文本, 置信度）
    std::vector<std::pair<std::wstring, float>> ocr(const std::string& path);
    std::vector<std::pair<std::wstring, float>> ocr(const cv::Mat& bgr);

    /// @brief 识别单行图片（BGR 或灰度）
    std::vector<std::pair<std::wstring, float>> ocr_for_single_lines(
        const std::vector<cv::Mat>& line_imgs);

private:
    std::vector<cv::Mat> line_split(const cv::Mat& gray);
    std::pair<std::wstring, float> recognize_line(const cv::Mat& img);

    std::unique_ptr<cnocrmodle::onnxmodle> modle;
    std::unique_ptr<CnStd> detector;
    std::vector<std::wstring> ctc_data;
    int input_channels_ = 1;
    int input_height_ = 32;
    int blank_index_ = -1;
};

#endif  // CNOCR_CNOCR_H
