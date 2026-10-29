#include <chrono>
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

void run_cnstd() {
    CnStd cnstd;
    const auto start = system_clock::now();
    const auto boxes = cnstd.detect("cnocr.png");
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

    if (argc > 1 && std::string(argv[1]) == "det") {
        run_cnstd();
    } else {
        run_cnocr();
    }

    const auto time_end = system_clock::now();
    std::cout << "程序总耗时: " << duration_cast<milliseconds>(time_end - time_start).count()
              << "ms" << std::endl;
    return 0;
}
