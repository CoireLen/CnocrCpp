#include "db_postprocess.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <opencv2/imgproc.hpp>
#include <utility>

namespace db_postprocess {
namespace {

std::vector<cv::Point2f> to_points(const std::vector<cv::Point>& contour) {
    std::vector<cv::Point2f> points(contour.size());
    for (size_t i = 0; i < contour.size(); ++i) {
        points[i] = cv::Point2f(static_cast<float>(contour[i].x),
                                static_cast<float>(contour[i].y));
    }
    return points;
}

// 与 PaddleOCR/cv2.minAreaRect 后处理一致：按 x 排序后还原左上、右上、右下、左下
std::pair<std::array<cv::Point2f, 4>, float> mini_boxes(
    const std::vector<cv::Point2f>& points) {
    cv::RotatedRect rect = cv::minAreaRect(points);
    cv::Point2f raw[4];
    rect.points(raw);
    std::stable_sort(raw, raw + 4, [](const cv::Point2f& a, const cv::Point2f& b) {
        return a.x < b.x;
    });

    int i1 = 0, i2 = 2, i3 = 3, i4 = 1;
    if (raw[1].y > raw[0].y) {
        i1 = 0;
        i4 = 1;
    } else {
        i1 = 1;
        i4 = 0;
    }
    if (raw[3].y > raw[2].y) {
        i2 = 2;
        i3 = 3;
    } else {
        i2 = 3;
        i3 = 2;
    }

    std::array<cv::Point2f, 4> box = {raw[i1], raw[i2], raw[i3], raw[i4]};
    return {box, std::min(rect.size.width, rect.size.height)};
}

std::array<cv::Point2f, 4> order_points_clockwise(const std::vector<cv::Point2f>& points) {
    std::array<int, 4> idx = {0, 1, 2, 3};
    std::stable_sort(idx.begin(), idx.end(),
                     [&](int a, int b) { return points[a].x < points[b].x; });

    int left[2] = {idx[0], idx[1]};
    int right[2] = {idx[2], idx[3]};
    if (points[left[1]].y < points[left[0]].y) std::swap(left[0], left[1]);
    if (points[right[1]].y < points[right[0]].y) std::swap(right[0], right[1]);

    return {points[left[0]], points[right[0]], points[right[1]], points[left[1]]};
}

// 多边形等距外扩（miter 拐角），等价于 pyclipper 的 JT_ROUND 近似
std::vector<cv::Point2f> offset_polygon(const std::vector<cv::Point2f>& poly, float distance) {
    const size_t n = poly.size();
    if (n < 3 || distance <= 0.f) {
        return poly;
    }
    double signed_area = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const auto& p = poly[i];
        const auto& q = poly[(i + 1) % n];
        signed_area += static_cast<double>(p.x) * q.y - static_cast<double>(q.x) * p.y;
    }
    const float orient = signed_area >= 0.0 ? 1.f : -1.f;

    auto outward = [orient](const cv::Point2f& a, const cv::Point2f& b) {
        cv::Point2f edge = b - a;
        const float length = std::hypot(edge.x, edge.y);
        if (length < 1e-6f) {
            return cv::Point2f(0.f, 0.f);
        }
        return cv::Point2f(orient * edge.y / length, -orient * edge.x / length);
    };

    std::vector<cv::Point2f> out(n);
    for (size_t i = 0; i < n; ++i) {
        const cv::Point2f& prev = poly[(i + n - 1) % n];
        const cv::Point2f& cur = poly[i];
        const cv::Point2f& next = poly[(i + 1) % n];
        const cv::Point2f normal_prev = outward(prev, cur);
        const cv::Point2f normal_next = outward(cur, next);
        const cv::Point2f line1_start = prev + normal_prev * distance;
        const cv::Point2f line2_start = cur + normal_next * distance;
        const cv::Point2f dir1 = cur - prev;
        const cv::Point2f dir2 = next - cur;

        const double denom =
            static_cast<double>(dir1.x) * dir2.y - static_cast<double>(dir1.y) * dir2.x;
        if (std::abs(denom) < 1e-6) {
            out[i] = cur + (normal_prev + normal_next) * (distance * 0.5f);
            continue;
        }
        const cv::Point2f diff = line2_start - line1_start;
        const double t =
            (static_cast<double>(diff.x) * dir2.y - static_cast<double>(diff.y) * dir2.x) /
            denom;
        out[i] = line1_start + dir1 * static_cast<float>(t);
    }
    return out;
}

float box_score_fast(const cv::Mat& prob, const std::array<cv::Point2f, 4>& box) {
    const int height = prob.rows;
    const int width = prob.cols;
    float min_x = box[0].x, max_x = box[0].x, min_y = box[0].y, max_y = box[0].y;
    for (int i = 1; i < 4; ++i) {
        min_x = std::min(min_x, box[i].x);
        max_x = std::max(max_x, box[i].x);
        min_y = std::min(min_y, box[i].y);
        max_y = std::max(max_y, box[i].y);
    }
    const int xmin = std::clamp(static_cast<int>(std::floor(min_x)), 0, width - 1);
    const int xmax = std::clamp(static_cast<int>(std::ceil(max_x)), 0, width - 1);
    const int ymin = std::clamp(static_cast<int>(std::floor(min_y)), 0, height - 1);
    const int ymax = std::clamp(static_cast<int>(std::ceil(max_y)), 0, height - 1);
    if (xmax < xmin || ymax < ymin) {
        return 0.f;
    }

    cv::Mat mask = cv::Mat::zeros(ymax - ymin + 1, xmax - xmin + 1, CV_8UC1);
    cv::Point polygon[4];
    for (int i = 0; i < 4; ++i) {
        polygon[i] =
            cv::Point(static_cast<int>(box[i].x - xmin), static_cast<int>(box[i].y - ymin));
    }
    const cv::Point* polygons[1] = {polygon};
    const int counts[1] = {4};
    cv::fillPoly(mask, polygons, counts, 1, cv::Scalar(1));

    const cv::Rect roi(xmin, ymin, xmax - xmin + 1, ymax - ymin + 1);
    return static_cast<float>(cv::mean(prob(roi), mask)[0]);
}

bool compare_boxes(const DetBox& a, const DetBox& b) {
    const float a_left = a.points[0].x, a_top = a.points[0].y;
    const float a_bottom = a.points[2].y;
    const float b_left = b.points[0].x, b_top = b.points[0].y;
    const float b_bottom = b.points[2].y;

    float y_overlap_ratio = 0.f;
    if (!(a_bottom <= b_top || b_bottom <= a_top)) {
        const float overlap = std::min(a_bottom, b_bottom) - std::max(a_top, b_top);
        y_overlap_ratio =
            overlap / std::max(1.f, std::min(a_bottom - a_top, b_bottom - b_top));
    }
    return y_overlap_ratio > 0.5f ? (a_left < b_left) : (a_top < b_top);
}

// compare_boxes 不是严格弱序（是否比较左右取决于两框的纵向重叠），不能用 std::sort；
// 这里用插入排序保证结果确定，与 Python 的实现思路一致。
void sort_reading_order(std::vector<DetBox>& boxes) {
    for (size_t i = 1; i < boxes.size(); ++i) {
        const DetBox key = boxes[i];
        size_t j = i;
        while (j > 0 && compare_boxes(key, boxes[j - 1])) {
            boxes[j] = boxes[j - 1];
            --j;
        }
        boxes[j] = key;
    }
}

}  // namespace

std::vector<DetBox> boxes_from_prob(const cv::Mat& prob, cv::Size src_size,
                                    const Params& params) {
    std::vector<DetBox> boxes;
    if (prob.empty() || src_size.width <= 0 || src_size.height <= 0) {
        return boxes;
    }

    const int height = prob.rows;
    const int width = prob.cols;
    cv::Mat bitmap(height, width, CV_8UC1);
    for (int y = 0; y < height; ++y) {
        const float* src = prob.ptr<float>(y);
        uchar* dst = bitmap.ptr<uchar>(y);
        for (int x = 0; x < width; ++x) {
            dst[x] = src[x] > params.bin_thresh ? 255 : 0;
        }
    }

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(bitmap, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

    const size_t candidate_count = std::min<size_t>(
        contours.size(), static_cast<size_t>(std::max(params.max_candidates, 0)));
    for (size_t i = 0; i < candidate_count; ++i) {
        const std::vector<cv::Point2f> contour = to_points(contours[i]);
        if (contour.size() < 4) {
            continue;
        }
        auto [box, side] = mini_boxes(contour);
        if (side < 3.f) {
            continue;
        }

        const float score = box_score_fast(prob, box);
        if (score < params.box_thresh) {
            continue;
        }

        const std::vector<cv::Point2f> box_points(box.begin(), box.end());
        const double area = std::abs(cv::contourArea(box_points));
        const double perimeter = cv::arcLength(box_points, true);
        const float expand = static_cast<float>(
            area * params.unclip_ratio / std::max(perimeter, 1e-6));
        const std::vector<cv::Point2f> expanded = offset_polygon(box_points, expand);
        if (expanded.size() < 3) {
            continue;
        }
        auto [expanded_box, expanded_side] = mini_boxes(expanded);
        if (expanded_side < 5.f) {
            continue;
        }

        // 概率图坐标还原到原图
        std::vector<cv::Point2f> scaled(4);
        for (int k = 0; k < 4; ++k) {
            scaled[k].x = std::clamp(std::round(expanded_box[k].x / width * src_size.width),
                                     0.f, static_cast<float>(src_size.width));
            scaled[k].y =
                std::clamp(std::round(expanded_box[k].y / height * src_size.height), 0.f,
                           static_cast<float>(src_size.height));
        }

        std::array<cv::Point2f, 4> ordered = order_points_clockwise(scaled);
        for (int k = 0; k < 4; ++k) {
            ordered[k].x =
                std::min(std::max(ordered[k].x, 0.f), static_cast<float>(src_size.width - 1));
            ordered[k].y =
                std::min(std::max(ordered[k].y, 0.f), static_cast<float>(src_size.height - 1));
        }
        const int box_width = static_cast<int>(box_point_distance(ordered[0], ordered[1]));
        const int box_height = static_cast<int>(box_point_distance(ordered[0], ordered[3]));
        if (std::min(box_width, box_height) < params.min_box_size) {
            continue;
        }

        DetBox result;
        std::copy(ordered.begin(), ordered.end(), result.points);
        result.score = score;
        boxes.push_back(result);
    }

    sort_reading_order(boxes);
    return boxes;
}

}  // namespace db_postprocess
