#include "modle.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace cnocrmodle {
namespace {

void append_execution_provider(Ort::SessionOptions& options, USE_DEVICE device) {
    if (device == USE_DEVICE::CPU) {
        return;
    }
    try {
        if (device == USE_DEVICE::CUDA) {
            OrtCUDAProviderOptions cuda_options{};
            options.AppendExecutionProvider_CUDA(cuda_options);
        } else {
            const OrtApi& api = Ort::GetApi();
            OrtTensorRTProviderOptionsV2* trt_options = nullptr;
            Ort::ThrowOnError(api.CreateTensorRTProviderOptions(&trt_options));
            try {
                Ort::ThrowOnError(api.SessionOptionsAppendExecutionProvider_TensorRT_V2(
                    static_cast<OrtSessionOptions*>(options), trt_options));
            } catch (...) {
                api.ReleaseTensorRTProviderOptions(trt_options);
                throw;
            }
            api.ReleaseTensorRTProviderOptions(trt_options);
        }
    } catch (const std::exception& e) {
        // 当前 onnxruntime 包可能未附带对应 provider，退回 CPU
        std::cerr << "执行提供器不可用，回退到 CPU: " << e.what() << std::endl;
    }
}

const Ort::Value* find_output(const std::vector<Ort::Value>& outputs, size_t rank,
                              ONNXTensorElementDataType type) {
    for (const Ort::Value& tensor : outputs) {
        auto info = tensor.GetTensorTypeAndShapeInfo();
        if (info.GetShape().size() == rank && info.GetElementType() == type) {
            return &tensor;
        }
    }
    return nullptr;
}

template <typename T>
void copy_tensor_data(const Ort::Value& tensor, std::vector<T>& target, size_t count) {
    const T* data = tensor.GetTensorData<T>();
    target.assign(data, data + count);
}

}  // namespace

onnxmodle::onnxmodle(const std::wstring& model_path, USE_DEVICE device)
    : env_(ORT_LOGGING_LEVEL_WARNING, "cnocr") {
    {
        std::ifstream probe(model_path, std::ios::binary);
        if (!probe.good()) {
            throw std::runtime_error("模型文件不存在或无法读取");
        }
    }

    session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    append_execution_provider(session_options_, device);
    session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), session_options_);

    Ort::AllocatorWithDefaultOptions allocator;
    input_name_storage_.reserve(session_->GetInputCount());
    for (size_t i = 0; i < session_->GetInputCount(); ++i) {
        auto name = session_->GetInputNameAllocated(i, allocator);
        input_name_storage_.emplace_back(name.get());
        auto type_info = session_->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo();
        input_shapes_.push_back(type_info.GetShape());
    }
    output_name_storage_.reserve(session_->GetOutputCount());
    for (size_t i = 0; i < session_->GetOutputCount(); ++i) {
        auto name = session_->GetOutputNameAllocated(i, allocator);
        output_name_storage_.emplace_back(name.get());
    }
    for (auto& name : input_name_storage_) {
        input_names_.push_back(name.c_str());
    }
    for (auto& name : output_name_storage_) {
        output_names_.push_back(name.c_str());
    }
}

onnxmodle::~onnxmodle() = default;

std::vector<Ort::Value> onnxmodle::run_model(std::vector<Ort::Value>& inputs) const {
    return session_->Run(Ort::RunOptions{nullptr}, input_names_.data(), inputs.data(),
                         inputs.size(), output_names_.data(), output_names_.size());
}

rec_return onnxmodle::run_rec(const uint8_t* image, int height, int width, int channels) {
    if (image == nullptr || height <= 0 || width <= 0 || (channels != 1 && channels != 3)) {
        throw std::invalid_argument("run_rec 输入参数非法");
    }

    const size_t plane = static_cast<size_t>(height) * static_cast<size_t>(width);
    std::vector<float> pixel_data(plane * static_cast<size_t>(channels));
    for (int c = 0; c < channels; ++c) {
        float* dst = pixel_data.data() + static_cast<size_t>(c) * plane;
        for (size_t i = 0; i < plane; ++i) {
            const float value = image[i * static_cast<size_t>(channels) + c] / 255.0f;
            dst[i] = channels == 3 ? (value - 0.5f) / 0.5f : value;
        }
    }

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    int64_t image_dims[4] = {1, channels, height, width};
    Ort::Value image_tensor = Ort::Value::CreateTensor<float>(
        memory_info, pixel_data.data(), pixel_data.size(), image_dims, 4);

    int64_t input_length = width;
    int64_t length_dims[1] = {1};

    // 按模型输入的实际顺序组装：一维输入为序列长度，其余为图像
    std::vector<Ort::Value> inputs;
    inputs.reserve(input_names_.size());
    bool image_used = false;
    for (size_t i = 0; i < input_names_.size(); ++i) {
        const bool is_length_input =
            i < input_shapes_.size() && input_shapes_[i].size() == 1;
        if (is_length_input) {
            inputs.push_back(Ort::Value::CreateTensor<int64_t>(memory_info, &input_length, 1,
                                                               length_dims, 1));
            continue;
        }
        if (image_used) {
            throw std::runtime_error("识别模型包含多个图像输入，暂不支持");
        }
        inputs.push_back(std::move(image_tensor));
        image_used = true;
    }
    if (!image_used) {
        throw std::runtime_error("识别模型缺少图像输入");
    }

    std::vector<Ort::Value> outputs = run_model(inputs);
    const Ort::Value* logits = find_output(outputs, 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    if (logits == nullptr) {
        throw std::runtime_error("识别模型未返回 3 维 logits");
    }

    auto logits_info = logits->GetTensorTypeAndShapeInfo();
    const std::vector<int64_t> logits_shape = logits_info.GetShape();
    rec_return result;
    result.seq_len = logits_shape[1];
    result.num_classes = logits_shape[2];
    copy_tensor_data(*logits, result.logits,
                     static_cast<size_t>(result.seq_len * result.num_classes));

    const Ort::Value* lengths = find_output(outputs, 1, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
    if (lengths != nullptr) {
        const int64_t count = lengths->GetTensorTypeAndShapeInfo().GetShape()[0];
        copy_tensor_data(*lengths, result.lengths, static_cast<size_t>(count));
    }
    result.valid_len = result.lengths.empty()
                           ? result.seq_len
                           : std::clamp<int64_t>(result.lengths.front(), 0, result.seq_len);
    return result;
}

det_return onnxmodle::run_det(float* chw, int height, int width) {
    if (chw == nullptr || height <= 0 || width <= 0) {
        throw std::invalid_argument("run_det 输入参数非法");
    }

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    int64_t dims[4] = {1, 3, height, width};
    std::vector<Ort::Value> inputs;
    inputs.push_back(Ort::Value::CreateTensor<float>(
        memory_info, chw, static_cast<size_t>(3) * height * width, dims, 4));

    std::vector<Ort::Value> outputs = run_model(inputs);
    const Ort::Value* prob = find_output(outputs, 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    if (prob == nullptr) {
        throw std::runtime_error("检测模型未返回 4 维概率图");
    }

    const std::vector<int64_t> shape = prob->GetTensorTypeAndShapeInfo().GetShape();
    det_return result;
    result.height = shape[2];
    result.width = shape[3];
    copy_tensor_data(*prob, result.prob,
                     static_cast<size_t>(result.height * result.width));
    return result;
}

}  // namespace cnocrmodle
