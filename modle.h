#include <onnxruntime_cxx_api.h>
#include <assert.h>
#include <vector>
#ifdef HAVE_TENSORRT_PROVIDER_FACTORY_H
#include <tensorrt_provider_factory.h>
#include <provider_options.h>

std::unique_ptr<OrtTensorRTProviderOptionsV2> get_default_trt_provider_options() {
  auto tensorrt_options = std::make_unique<OrtTensorRTProviderOptionsV2>();
  tensorrt_options->device_id = 0;
  tensorrt_options->has_user_compute_stream = 0;
  tensorrt_options->user_compute_stream = nullptr;
  tensorrt_options->trt_max_partition_iterations = 1000;
  tensorrt_options->trt_min_subgraph_size = 1;
  tensorrt_options->trt_max_workspace_size = 1 << 30;
  tensorrt_options->trt_fp16_enable = false;
  tensorrt_options->trt_int8_enable = false;
  tensorrt_options->trt_int8_calibration_table_name = "";
  tensorrt_options->trt_int8_use_native_calibration_table = false;
  tensorrt_options->trt_dla_enable = false;
  tensorrt_options->trt_dla_core = 0;
  tensorrt_options->trt_dump_subgraphs = false;
  tensorrt_options->trt_engine_cache_enable = false;
  tensorrt_options->trt_engine_cache_path = "";
  tensorrt_options->trt_engine_decryption_enable = false;
  tensorrt_options->trt_engine_decryption_lib_path = "";
  tensorrt_options->trt_force_sequential_engine_build = false;

  return tensorrt_options;
}
#endif


void run_ort_trt(long long input_lengths,long long x_length,unsigned char * x);

class onnxmodle
{
private:
    Ort::Env env;
    const OrtApi& api=Ort::GetApi();
    OrtTensorRTProviderOptionsV2* tensorrt_option;
    Ort::SessionOptions session_options;
    const wchar_t* model_path=L"cnocr136fc.onnx";
    Ort::Session *session;
    Ort::AllocatorWithDefaultOptions allocator;
    size_t num_input_nodes;
    std::vector<const char*> input_node_names;
    std::vector<int64_t> input_node_dims;
public:
  std::vector<void *> run(long long input_lengths,long long x_length,unsigned char * x);
  onnxmodle(/* args */);
  ~onnxmodle();
};


