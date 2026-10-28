#ifndef CNOCR_DB_POSTPROCESS_H
#define CNOCR_DB_POSTPROCESS_H

#include <opencv2/core.hpp>
#include <vector>

#include "det_box.h"

namespace db_postprocess {

struct Params {
    float bin_thresh = 0.3f;   // 概率图二值化阈值
    float box_thresh = 0.3f;   // 文本框得分阈值
    float unclip_ratio = 1.5f; // 文本框外扩比例
    int min_box_size = 4;      // 最小文本框边长
    int max_candidates = 1000; // 最多处理的轮廓数
};

/// @brief DBNet 概率图转文本框（坐标已还原到原图，并按阅读顺序排序）
/// @param prob 模型输出的单通道概率图，CV_32FC1
/// @param src_size 原图尺寸
std::vector<DetBox> boxes_from_prob(const cv::Mat& prob, cv::Size src_size, const Params& params);

}  // namespace db_postprocess

#endif  // CNOCR_DB_POSTPROCESS_H
