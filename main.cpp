#include <chrono>
#include <exception>
#include <iostream>
#include <string>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#include "cnocr.h"
#include "cnstd.h"

using namespace std::chrono;

namespace {

struct ModelChoice {
    const char* alias;
    cnocr::USE_MODLE mode;
    const char* description;
};

const ModelChoice kModels[] = {
    {"zh", cnocr::USE_MODLE::cnocr136fc, "早期中文识别模型"},
    {"en", cnocr::USE_MODLE::en_number, "英文数字识别模型"},
    {"cht", cnocr::USE_MODLE::chinese_cht, "繁体中文识别模型"},
    {"zh-gru", cnocr::USE_MODLE::densenet_lite_136_gru, "通用中文识别（cnocr 2.3）"},
    {"scene", cnocr::USE_MODLE::scene_densenet_136_gru, "场景文字识别（cnocr 2.3）"},
    {"doc", cnocr::USE_MODLE::doc_densenet_136_gru, "文档识别（cnocr 2.3）"},
    {"number", cnocr::USE_MODLE::number_densenet_136_fc, "数字识别（cnocr 2.3）"},
    {"ppv3", cnocr::USE_MODLE::ch_ppocrv3, "简体中英文识别（PP-OCRv3）"},
    {"ppv2", cnocr::USE_MODLE::ch_ppocr_mobile_v2, "简体中英文识别（mobile v2.0）"},
};

void run_ocr(cnocr& ocr, const std::string& image_path) {
    const auto start = system_clock::now();
    const auto lines = ocr.ocr(image_path);
    const auto end = system_clock::now();

    int index = 0;
    for (const auto& line : lines) {
        std::cout << index++ << ": " << line.first
                  << "  (置信度 " << line.second << ")" << std::endl;
    }
    std::cout << image_path << " 识别耗时: " << duration_cast<milliseconds>(end - start).count()
              << "ms, 共 " << lines.size() << " 行" << std::endl;
}

void print_models() {
    std::cout << "用法: UseAiForWinML [det | list | <模型> <图片>]" << std::endl;
    std::cout << "模型别名:" << std::endl;
    for (const ModelChoice& item : kModels) {
        std::cout << "  " << item.alias << "\t" << item.description << std::endl;
    }
}

bool run_single(const std::string& alias, const std::string& image_path) {
    for (const ModelChoice& item : kModels) {
        if (alias == item.alias) {
            std::cout << "===== " << item.description << " =====" << std::endl;
            cnocr model(item.mode, cnocrmodle::USE_DEVICE::CPU);
            run_ocr(model, image_path);
            return true;
        }
    }
    return false;
}

}  // namespace

void run_cnocr() {
    std::cout << "===== 中文模型 =====" << std::endl;
    cnocr zh(cnocr::USE_MODLE::cnocr136fc, cnocrmodle::USE_DEVICE::CPU);
    run_ocr(zh, "cnocr.png");

    std::cout << "===== 英文数字模型 =====" << std::endl;
    cnocr en(cnocr::USE_MODLE::en_number, cnocrmodle::USE_DEVICE::CPU);
    run_ocr(en, "enocr.png");

    std::cout << "===== 繁体中文模型 =====" << std::endl;
    cnocr cht(cnocr::USE_MODLE::chinese_cht, cnocrmodle::USE_DEVICE::CPU);
    run_ocr(cht, "ctocr.png");
}

void run_cnstd(const std::string& image_path = "cnocr.png") {
    CnStd cnstd;
    const auto start = system_clock::now();
    const auto boxes = cnstd.detect(image_path);
    const auto end = system_clock::now();
    std::cout << "检测到 " << boxes.size() << " 个文本框, 耗时 "
              << duration_cast<milliseconds>(end - start).count() << "ms" << std::endl;
    for (size_t i = 0; i < boxes.size(); ++i) {
        const DetBox& box = boxes[i];
        std::cout << i << ": score=" << box.score << " points=[";
        for (int k = 0; k < 4; ++k) {
            std::cout << "(" << box.points[k].x << "," << box.points[k].y << ")"
                      << (k == 3 ? "" : ", ");
        }
        std::cout << "]" << std::endl;
    }
}

int main(int argc, char** argv) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif
    const auto time_start = system_clock::now();

    try {
        if (argc > 1) {
            const std::string arg = argv[1];
            if (arg == "det") {
                run_cnstd(argc > 2 ? argv[2] : "cnocr.png");
            } else if (arg == "list") {
                print_models();
            } else if (argc > 2 && run_single(arg, argv[2])) {
                // 指定模型与图片，已执行
            } else {
                print_models();
                return 1;
            }
        } else {
            run_cnocr();
        }
    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << std::endl;
        return 1;
    }

    const auto time_end = system_clock::now();
    std::cout << "程序总耗时: " << duration_cast<milliseconds>(time_end - time_start).count()
              << "ms" << std::endl;
    return 0;
}
