#include "cnocr.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <stdexcept>

#include "image_io.h"
#include "model_registry.h"

namespace {

const char* model_name_of(cnocr::USE_MODLE mode) {
    switch (mode) {
        case cnocr::USE_MODLE::cnocr136fc:
            return "cnocr136fc";
        case cnocr::USE_MODLE::en_number:
            return "en_number_mobile_v2.0";
        case cnocr::USE_MODLE::chinese_cht:
            return "chinese_cht_PP-OCRv3";
        case cnocr::USE_MODLE::densenet_lite_136_gru:
            return "densenet_lite_136-gru";
        case cnocr::USE_MODLE::scene_densenet_136_gru:
            return "scene-densenet_lite_136-gru";
        case cnocr::USE_MODLE::doc_densenet_136_gru:
            return "doc-densenet_lite_136-gru";
        case cnocr::USE_MODLE::number_densenet_136_fc:
            return "number-densenet_lite_136-fc";
        case cnocr::USE_MODLE::ch_ppocrv3:
            return "ch_PP-OCRv3";
        case cnocr::USE_MODLE::ch_ppocr_mobile_v2:
            return "ch_ppocr_mobile_v2.0";
    }
    return nullptr;
}

std::vector<std::string> load_dict(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("字符集文件不存在或无法读取");
    }
    std::vector<std::string> tokens;
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        std::string token = line;
        const size_t begin = token.find_first_not_of(" \t");
        const size_t end = token.find_last_not_of(" \t");
        const std::string trimmed =
            begin == std::string::npos ? std::string() : token.substr(begin, end - begin + 1);
        if (trimmed == "<space>") {
            token = " ";
        } else if (trimmed.rfind("<unused", 0) == 0) {
            token.clear();
        }
        tokens.push_back(std::move(token));
    }
    return tokens;
}

cv::Mat prepare_line_image(const cv::Mat& img, int channels, int height) {
    cv::Mat converted;
    if (channels == 1) {
        if (img.channels() == 1) {
            converted = img;
        } else if (img.channels() == 4) {
            cv::cvtColor(img, converted, cv::COLOR_BGRA2GRAY);
        } else {
            cv::cvtColor(img, converted, cv::COLOR_BGR2GRAY);
        }
    } else if (img.channels() == 3) {
        converted = img;
    } else if (img.channels() == 4) {
        cv::cvtColor(img, converted, cv::COLOR_BGRA2BGR);
    } else {
        cv::cvtColor(img, converted, cv::COLOR_GRAY2BGR);
    }

    const int target_w =
        std::max(static_cast<int>(converted.cols * static_cast<double>(height) / converted.rows), 8);
    cv::Mat resized;
    cv::resize(converted, resized, cv::Size(target_w, height), 0, 0, cv::INTER_LINEAR);
    return resized;
}

std::pair<std::string, float> decode_ctc(const cnocrmodle::rec_return& out,
                                         const std::vector<std::string>& dict,
                                         int blank_index) {
    const int seq_len = static_cast<int>(std::min(out.valid_len, out.seq_len));
    const int num_classes = static_cast<int>(out.num_classes);
    const int token_count = static_cast<int>(dict.size());

    std::string text;
    float confidence = 1.f;
    int previous = -1;
    for (int t = 0; t < seq_len; ++t) {
        const float* row = out.logits.data() + static_cast<size_t>(t) * num_classes;
        int best = 0;
        float max_value = row[0];
        for (int c = 1; c < num_classes; ++c) {
            if (row[c] > max_value) {
                max_value = row[c];
                best = c;
            }
        }

        double sum = 0.0;
        for (int c = 0; c < num_classes; ++c) {
            sum += std::exp(static_cast<double>(row[c] - max_value));
        }
        confidence = std::min(confidence, static_cast<float>(1.0 / sum));

        if (best != previous && best != blank_index && best >= 0 && best < token_count) {
            text += dict[static_cast<size_t>(best)];
        }
        previous = best;
    }
    return {text, confidence};
}

}  // namespace

cnocr::cnocr(USE_MODLE mode, cnocrmodle::USE_DEVICE device, bool use_detection,
             bool auto_download) {
    const char* name = model_name_of(mode);
    const cnocr_models::ModelSpec* spec =
        name == nullptr ? nullptr : cnocr_models::find_recognizer(name);
    if (spec == nullptr) {
        throw std::runtime_error("未注册的识别模型");
    }

    modle = std::make_unique<cnocrmodle::onnxmodle>(
        cnocr_models::ensure_model(*spec, auto_download), device);
    ctc_data = load_dict(cnocr_models::dict_path(*spec));
    if (ctc_data.empty()) {
        throw std::runtime_error("字符集为空");
    }
    input_channels_ = spec->channels;
    input_height_ = spec->height;
    // cnocr 自训练模型把 blank 放在最后一类；PaddleOCR 系模型 blank 为第 0 类
    blank_index_ =
        spec->blank_index < 0 ? static_cast<int>(ctc_data.size()) : spec->blank_index;

    if (use_detection) {
        detector = std::make_unique<CnStd>(device, auto_download);
    }
}

cnocr::~cnocr() = default;

std::vector<std::pair<std::string, float>> cnocr::ocr(const std::string& path) {
    const cv::Mat bgr = imageio::imread_utf8(std::filesystem::u8path(path));
    if (bgr.empty()) {
        throw std::runtime_error("无法读取图片: " + path);
    }
    return ocr(bgr);
}

std::vector<std::pair<std::string, float>> cnocr::ocr(const cv::Mat& bgr) {
    if (bgr.empty() || std::min(bgr.rows, bgr.cols) < 2) {
        return {};
    }

    std::vector<cv::Mat> lines;
    if (detector) {
        const std::vector<DetBox> boxes = detector->detect(bgr);
        lines.reserve(boxes.size());
        for (const DetBox& box : boxes) {
            cv::Mat crop = CnStd::crop(bgr, box);
            if (!crop.empty()) {
                lines.push_back(std::move(crop));
            }
        }
    } else {
        cv::Mat gray;
        if (bgr.channels() == 1) {
            gray = bgr;
        } else {
            cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
        }
        lines = line_split(gray);
    }

    if (lines.empty()) {
        return {};
    }
    return ocr_for_single_lines(lines);
}

std::vector<std::pair<std::string, float>> cnocr::ocr_for_single_lines(
    const std::vector<cv::Mat>& line_imgs) {
    std::vector<std::pair<std::string, float>> result;
    result.reserve(line_imgs.size());
    for (const cv::Mat& img : line_imgs) {
        result.push_back(recognize_line(img));
    }
    return result;
}

std::pair<std::string, float> cnocr::recognize_line(const cv::Mat& img) {
    if (img.empty() || std::min(img.rows, img.cols) < 2) {
        return {std::string(), 0.f};
    }

    const cv::Mat resized = prepare_line_image(img, input_channels_, input_height_);
    const cnocrmodle::rec_return out =
        modle->run_rec(resized.data, resized.rows, resized.cols, input_channels_);
    return decode_ctc(out, ctc_data, blank_index_);
}

std::vector<cv::Mat> cnocr::line_split(const cv::Mat& gray) {
    std::vector<cv::Mat> lines;
    cv::Mat inverted;
    cv::subtract(255, gray, inverted);
    cv::Mat row_max;
    cv::reduce(inverted, row_max, 1, cv::REDUCE_MAX);

    int start = -1;
    int end = -1;
    auto append = [&](int y0, int y1) {
        if (y1 - y0 <= 5) {
            return;
        }
        const int top = std::max(y0 - 1, 0);
        const int bottom = std::min(y1 + 1, gray.rows);
        lines.push_back(gray(cv::Rect(0, top, gray.cols, bottom - top)).clone());
    };

    for (int y = 0; y < gray.rows; ++y) {
        const bool has_text = row_max.at<uchar>(y, 0) > 100;
        if (has_text) {
            if (start < 0) {
                start = y;
            }
            end = y;
        } else if (start >= 0) {
            append(start, end);
            start = -1;
        }
    }
    if (start >= 0) {
        append(start, end);
    }
    return lines;
}
