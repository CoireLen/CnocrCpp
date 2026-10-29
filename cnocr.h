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
        cnocr136fc,                 // 早期中文识别模型（cnocr 2.2，字典 label_cn.txt）
        en_number,                  // 英文数字识别模型（PaddleOCR）
        chinese_cht,                // 繁体中文识别模型（PaddleOCR）
        densenet_lite_136_gru,      // 通用中文识别模型（cnocr 2.3）
        scene_densenet_136_gru,     // 场景文字识别模型（cnocr 2.3）
        doc_densenet_136_gru,       // 文档识别模型（cnocr 2.3）
        number_densenet_136_fc,     // 数字识别模型（cnocr 2.3）
        ch_ppocrv3,                 // 简体中英文识别模型（PaddleOCR PP-OCRv3）
        ch_ppocr_mobile_v2,         // 简体中英文识别模型（PaddleOCR mobile v2.0）
    };

    /// @param device 推理设备；当前 onnxruntime 包为 CPU 版，CUDA/TensorRT 不可用时自动回退
    /// @param use_detection 是否先用 PP-OCRv3 检测模型切分文本行（关闭后使用投影切分）
    /// @param auto_download 模型文件缺失时是否自动下载（缓存到 CNOCR_HOME 或平台默认目录）
    explicit cnocr(USE_MODLE mode = USE_MODLE::cnocr136fc,
                   cnocrmodle::USE_DEVICE device = cnocrmodle::USE_DEVICE::CPU,
                   bool use_detection = true, bool auto_download = true);
    ~cnocr();

    cnocr(const cnocr&) = delete;
    cnocr& operator=(const cnocr&) = delete;

    /// @brief 识别整张图片，返回按阅读顺序排列的（文本, 置信度）
    /// @note 文本统一为 UTF-8 编码；路径按 UTF-8 解释
    std::vector<std::pair<std::string, float>> ocr(const std::string& path);
    std::vector<std::pair<std::string, float>> ocr(const cv::Mat& bgr);

    /// @brief 识别单行图片（BGR 或灰度），文本为 UTF-8
    std::vector<std::pair<std::string, float>> ocr_for_single_lines(
        const std::vector<cv::Mat>& line_imgs);

private:
    std::vector<cv::Mat> line_split(const cv::Mat& gray);
    std::pair<std::string, float> recognize_line(const cv::Mat& img);

    std::unique_ptr<cnocrmodle::onnxmodle> modle;
    std::unique_ptr<CnStd> detector;
    std::vector<std::string> ctc_data;
    int input_channels_ = 1;
    int input_height_ = 32;
    int blank_index_ = -1;
};

#endif  // CNOCR_CNOCR_H
