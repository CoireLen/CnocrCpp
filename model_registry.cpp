#include "model_registry.h"

#include <cstdlib>
#include <iostream>
#include <stdexcept>

#include "model_files.h"

namespace cnocr_models {
namespace {

constexpr const char* kRepoPrefix = "breezedeus/cnstd-cnocr-models/resolve/main/";

std::vector<std::string> endpoints() {
    std::vector<std::string> list;
    if (const char* custom = std::getenv("CNOCR_MODEL_ENDPOINT"); custom != nullptr && *custom) {
        list.emplace_back(custom);
        return list;
    }
    if (const char* hf = std::getenv("HF_ENDPOINT"); hf != nullptr && *hf) {
        list.emplace_back(hf);
    }
    list.emplace_back("https://hf-mirror.com");
    list.emplace_back("https://huggingface.co");
    return list;
}

std::string trim_trailing_slash(std::string endpoint) {
    while (!endpoint.empty() && endpoint.back() == '/') {
        endpoint.pop_back();
    }
    return endpoint;
}

std::filesystem::path cache_dir() { return modelfiles::data_dir() / "models"; }

std::filesystem::path ensure_downloaded(const ModelSpec& spec, bool auto_download) {
    const std::filesystem::path local = std::filesystem::path("modle") / spec.onnx_file;
    if (std::filesystem::is_regular_file(local)) {
        return local;
    }

    const std::filesystem::path cached = cache_dir() / spec.onnx_file;
    if (std::filesystem::is_regular_file(cached)) {
        return cached;
    }

    if (spec.archive == nullptr || *spec.archive == '\0') {
        throw std::runtime_error("模型文件缺失: " + local.u8string() +
                                 "（该模型无自动下载地址，请手动放置）");
    }
    if (!auto_download) {
        throw std::runtime_error("模型文件缺失: " + local.u8string() +
                                 "（已禁用自动下载，可用 cnocr 构造参数 auto_download=true 开启）");
    }

    std::error_code ec;
    std::filesystem::create_directories(cached.parent_path(), ec);

    std::string last_error;
    for (const std::string& raw_endpoint : endpoints()) {
        const std::string endpoint = trim_trailing_slash(raw_endpoint);
        const std::string url =
            endpoint + "/" + kRepoPrefix + spec.subfolder + "/" + spec.archive;
        const std::filesystem::path zip =
            cached.parent_path() / (std::string(spec.archive) + ".download");
        try {
            std::cerr << "下载模型 " << spec.name << ": " << url << std::endl;
            modelfiles::download(url, zip);
            modelfiles::extract_onnx(zip, cached);
            std::filesystem::remove(zip, ec);
            std::cerr << "模型已保存到 " << cached.u8string() << std::endl;
            return cached;
        } catch (const std::exception& e) {
            last_error = e.what();
            std::filesystem::remove(zip, ec);
            std::cerr << "从 " << endpoint << " 获取模型失败: " << last_error << std::endl;
        }
    }

    throw std::runtime_error("无法获取模型 " + std::string(spec.name) + ": " + last_error +
                             "\n请手动下载后放到 " + local.u8string() + " 或 " + cached.u8string() +
                             "\n也可设置环境变量 CNOCR_MODEL_ENDPOINT 指向可用的镜像地址");
}

}  // namespace

const std::vector<ModelSpec>& all_recognizers() {
    static const std::vector<ModelSpec> specs = {
        {"cnocr136fc", 1, 32, -1, "cnocr136fc.onnx", "label_cn.txt", "models/cnocr/2.2",
         "densenet_lite_136-fc-onnx.zip"},
        {"densenet_lite_136-gru", 1, 32, -1, "densenet_lite_136-gru.onnx", "label_cn_2.3.txt",
         "models/cnocr/2.3", "densenet_lite_136-gru-onnx.zip"},
        {"scene-densenet_lite_136-gru", 1, 32, -1, "scene-densenet_lite_136-gru.onnx",
         "label_cn_2.3.txt", "models/cnocr/2.3", "scene-densenet_lite_136-gru-onnx.zip"},
        {"doc-densenet_lite_136-gru", 1, 32, -1, "doc-densenet_lite_136-gru.onnx",
         "label_cn_2.3.txt", "models/cnocr/2.3", "doc-densenet_lite_136-gru-onnx.zip"},
        {"number-densenet_lite_136-fc", 1, 32, -1, "number-densenet_lite_136-fc.onnx",
         "label_number.txt", "models/cnocr/2.3", "number-densenet_lite_136-fc-onnx.zip"},
        {"en_number_mobile_v2.0", 3, 32, 0, "en_number_mobile_v2.0_rec_infer.onnx", "en_dict.txt",
         "models/cnocr/2.3", "en_number_mobile_v2.0_rec_infer-onnx.zip"},
        {"chinese_cht_PP-OCRv3", 3, 32, 0, "chinese_cht_PP-OCRv3_rec_infer.onnx",
         "chinese_cht_dict.txt", "models/cnocr/2.3", "chinese_cht_PP-OCRv3_rec_infer-onnx.zip"},
        {"ch_PP-OCRv3", 3, 48, 0, "ch_PP-OCRv3_rec_infer.onnx", "ppocr_keys_v1.txt",
         "models/cnocr/2.3", "ch_PP-OCRv3_rec_infer-onnx.zip"},
        {"ch_ppocr_mobile_v2.0", 3, 32, 0, "ch_ppocr_mobile_v2.0_rec_infer.onnx",
         "ppocr_keys_v1.txt", "models/cnocr/2.3", "ch_ppocr_mobile_v2.0_rec_infer-onnx.zip"},
    };
    return specs;
}

const ModelSpec* find_recognizer(std::string_view name) {
    for (const ModelSpec& spec : all_recognizers()) {
        if (name == spec.name) {
            return &spec;
        }
    }
    return nullptr;
}

std::filesystem::path ensure_model(const ModelSpec& spec, bool auto_download) {
    return ensure_downloaded(spec, auto_download);
}

std::filesystem::path dict_path(const ModelSpec& spec) {
    if (spec.dict_file == nullptr || *spec.dict_file == '\0') {
        throw std::runtime_error("模型没有关联的字典文件");
    }
    return std::filesystem::path("dict") / spec.dict_file;
}

std::filesystem::path ensure_det_model(bool auto_download) {
    static const ModelSpec det = {"ch_PP-OCRv3_det",
                                  3,
                                  32,
                                  0,
                                  "ch_PP-OCRv3_det_infer.onnx",
                                  "",
                                  "models/cnstd/1.2",
                                  "ch_PP-OCRv3_det_infer-onnx.zip"};
    return ensure_downloaded(det, auto_download);
}

}  // namespace cnocr_models
