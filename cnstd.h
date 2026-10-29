#ifndef CNOCR_CNSTD_H
#define CNOCR_CNSTD_H

#include <opencv2/opencv.hpp>
#include <memory>
#include <string>
#include <vector>

#include "det_box.h"
#include "modle.h"

class CnStd {
public:
    /// @param auto_download 检测模型缺失时是否自动下载
    explicit CnStd(cnocrmodle::USE_DEVICE device = cnocrmodle::USE_DEVICE::CPU,
                   bool auto_download = true);
    ~CnStd();

    std::vector<DetBox> detect(const cv::Mat& bgr, cv::Size resized_shape = cv::Size(768, 768),
                               bool preserve_aspect_ratio = true, float box_score_thresh = 0.3f,
                               int min_box_size = 4);
    /// @param path 图片路径，按 UTF-8 解释
    std::vector<DetBox> detect(const std::string& path);

    /// @brief 按四点透视变换裁剪文本框，必要时旋转为水平
    static cv::Mat crop(const cv::Mat& bgr, const DetBox& box);

private:
    std::unique_ptr<cnocrmodle::onnxmodle> modle;
};

#endif  // CNOCR_CNSTD_H
