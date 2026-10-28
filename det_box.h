#ifndef CNOCR_DET_BOX_H
#define CNOCR_DET_BOX_H

#include <cmath>
#include <opencv2/core.hpp>

struct DetBox {
    // 顺时针：左上、右上、右下、左下
    cv::Point2f points[4];
    float score = 0.f;
};

inline float box_point_distance(const cv::Point2f& a, const cv::Point2f& b) {
    return std::hypot(a.x - b.x, a.y - b.y);
}

#endif  // CNOCR_DET_BOX_H
