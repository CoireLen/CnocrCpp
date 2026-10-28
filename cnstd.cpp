#include "cnstd.h"

#include <algorithm>
#include <cmath>

#include "db_postprocess.h"

namespace {

constexpr int kSizeDivisor = 32;

cv::Size resized_shape_of(cv::Size src, cv::Size target, bool preserve_aspect_ratio) {
    if (!preserve_aspect_ratio) {
        return target;
    }
    const double target_ratio = static_cast<double>(target.height) / target.width;
    const double actual_ratio = static_cast<double>(src.height) / src.width;
    const double ratio = actual_ratio > target_ratio
                             ? static_cast<double>(target.height) / src.height
                             : static_cast<double>(target.width) / src.width;
    const int new_height = std::max(1, static_cast<int>(src.height * ratio));
    const int new_width = std::max(1, static_cast<int>(src.width * ratio));
    auto calibrate = [](int value) {
        return std::max(static_cast<int>(std::lround(value / static_cast<double>(kSizeDivisor))) *
                            kSizeDivisor,
                        kSizeDivisor);
    };
    return cv::Size(calibrate(new_width), calibrate(new_height));
}

// BGR -> ImageNet 归一化后的 CHW float 数据
void fill_normalized_chw(const cv::Mat& src, cv::Size target, std::vector<float>& chw) {
    cv::Mat bgr;
    if (src.channels() == 3) {
        bgr = src;
    } else if (src.channels() == 1) {
        cv::cvtColor(src, bgr, cv::COLOR_GRAY2BGR);
    } else {
        cv::cvtColor(src, bgr, cv::COLOR_BGRA2BGR);
    }

    cv::Mat resized;
    cv::resize(bgr, resized, target, 0, 0, cv::INTER_LINEAR);

    const float mean[3] = {0.485f, 0.456f, 0.406f};
    const float stddev[3] = {0.229f, 0.224f, 0.225f};
    const size_t plane = static_cast<size_t>(target.height) * target.width;
    chw.resize(plane * 3);

    for (int y = 0; y < target.height; ++y) {
        const cv::Vec3b* row = resized.ptr<cv::Vec3b>(y);
        for (int x = 0; x < target.width; ++x) {
            const size_t index = static_cast<size_t>(y) * target.width + x;
            for (int c = 0; c < 3; ++c) {
                const float value = row[x][c] / 255.0f;
                chw[static_cast<size_t>(c) * plane + index] = (value - mean[c]) / stddev[c];
            }
        }
    }
}

}  // namespace

CnStd::CnStd(cnocrmodle::USE_DEVICE device)
    : modle(std::make_unique<cnocrmodle::onnxmodle>(L"modle/ch_PP-OCRv3_det_infer.onnx", device)) {}

CnStd::~CnStd() = default;

std::vector<DetBox> CnStd::detect(const cv::Mat& bgr, cv::Size resized_shape,
                                  bool preserve_aspect_ratio, float box_score_thresh,
                                  int min_box_size) {
    if (bgr.empty()) {
        return {};
    }

    const cv::Size target = resized_shape_of(bgr.size(), resized_shape, preserve_aspect_ratio);
    std::vector<float> chw;
    fill_normalized_chw(bgr, target, chw);

    cnocrmodle::det_return out = modle->run_det(chw.data(), target.height, target.width);
    const cv::Mat prob(static_cast<int>(out.height), static_cast<int>(out.width), CV_32FC1,
                       out.prob.data());

    db_postprocess::Params params;
    params.box_thresh = box_score_thresh;
    params.min_box_size = min_box_size;
    return db_postprocess::boxes_from_prob(prob, bgr.size(), params);
}

std::vector<DetBox> CnStd::detect(const std::string& path) {
    const cv::Mat bgr = cv::imread(path, cv::IMREAD_COLOR);
    return detect(bgr);
}

cv::Mat CnStd::crop(const cv::Mat& bgr, const DetBox& box) {
    if (bgr.empty()) {
        return cv::Mat();
    }
    const int crop_width = static_cast<int>(
        std::max(box_point_distance(box.points[0], box.points[1]),
                 box_point_distance(box.points[2], box.points[3])));
    const int crop_height = static_cast<int>(
        std::max(box_point_distance(box.points[0], box.points[3]),
                 box_point_distance(box.points[1], box.points[2])));
    if (crop_width <= 0 || crop_height <= 0) {
        return cv::Mat();
    }

    const cv::Point2f destination[4] = {
        cv::Point2f(0.f, 0.f), cv::Point2f(static_cast<float>(crop_width), 0.f),
        cv::Point2f(static_cast<float>(crop_width), static_cast<float>(crop_height)),
        cv::Point2f(0.f, static_cast<float>(crop_height))};
    const cv::Mat transform = cv::getPerspectiveTransform(box.points, destination);
    cv::Mat cropped;
    cv::warpPerspective(bgr, cropped, transform, cv::Size(crop_width, crop_height),
                        cv::INTER_CUBIC, cv::BORDER_REPLICATE);
    if (cropped.rows * 1.0 / cropped.cols >= 1.5) {
        cv::rotate(cropped, cropped, cv::ROTATE_90_COUNTERCLOCKWISE);
    }
    return cropped;
}
