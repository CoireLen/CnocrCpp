#include "modle.h"
#include <iostream>
onnxmodle::onnxmodle(/* args */)
{
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING,"test");
  this->env=std::move(env);
  this->session_options.SetInterOpNumThreads(1);
  this->session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
  this->api.CreateTensorRTProviderOptions(&tensorrt_option);
  std::unique_ptr<OrtTensorRTProviderOptionsV2,decltype(api.ReleaseTensorRTProviderOptions)>  rel_trt_options(tensorrt_option,api.ReleaseTensorRTProviderOptions);
  this->api.SessionOptionsAppendExecutionProvider_TensorRT_V2(static_cast<OrtSessionOptions*>(session_options), rel_trt_options.get());
  Ort::Session *session=new Ort::Session(this->env, this->model_path, this->session_options);
  this->session=session;
  this->num_input_nodes = this->session->GetInputCount();
  std::vector<const char*> input_node_names(this->num_input_nodes);
  this->input_node_names=std::move(input_node_names);
  printf("Number of inputs = %zu\n", this->num_input_nodes);
  char* input_names[]={"x","input_lengths"};
  for (int i = 0; i < this->num_input_nodes; i++) {
    // print input node names
    this->input_node_names[i] = input_names[i];

    // print input node types
    Ort::TypeInfo type_info = session->GetInputTypeInfo(i);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

    ONNXTensorElementDataType type = tensor_info.GetElementType();
    printf("Input %d : type=%d\n", i, type);

    // print input shapes/dims
    input_node_dims = tensor_info.GetShape();
    printf("Input %d : num_dims=%zu\n", i, input_node_dims.size());
    for (size_t j = 0; j < input_node_dims.size(); j++)
      printf("Input %d : dim %zu=%jd\n", i, j, input_node_dims[j]);
  }

}
std::vector<void*> onnxmodle::run(long long input_length,long long x_length,unsigned char * x) {
  size_t input_tensor_size = 32*input_length;  // simplify ... using known dim values to calculate size
                                             // use OrtGetTensorShapeElementCount() to get official size!

  std::vector<float> input_tensor_values(input_tensor_size);
  std::vector<const char*> output_node_names = {"output_lengths","logits"};

  // initialize input data with values in [0.0, 1.0]
  for (unsigned int i = 0; i < x_length; i++){
    input_tensor_values[i] = *(x+i)/255.0;
  }
    

  // create input tensor object from data values
  auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  input_node_dims={1,1,32,input_length};
  Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_size, input_node_dims.data(), 4);
  std::vector<int64_t> input_tensor_length(1);
  input_tensor_length[0]=input_length;
  std::vector<int64_t> input_length_dims={1};
  auto memory_info_length = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value input_tensor_lengths = Ort::Value::CreateTensor<int64_t>(memory_info_length, input_tensor_length.data(), 1,input_length_dims.data(),1);
  assert(input_tensor.IsTensor());
  assert(input_tensor_lengths.IsTensor());
  std::vector<Ort::Value> input_tensor_data;
  input_tensor_data.push_back(std::move(input_tensor));
  input_tensor_data.push_back(std::move(input_tensor_lengths));
  // score model & input tensor, get back output tensor
  auto output_tensors = this->session->Run(Ort::RunOptions{nullptr}, input_node_names.data(),input_tensor_data.data() , 2, output_node_names.data(), 2);
  assert(output_tensors.size() == 2 && output_tensors.front().IsTensor());
  for (auto i :output_node_names){
    std::cout <<i<<std::endl;
  }
  // Get pointer to output tensor float values
  int64_t*  output_lengths= output_tensors[0].GetTensorMutableData<int64_t>();
  float * logits=output_tensors[1].GetTensorMutableData<float>();
  // release buffers allocated by ORT alloctor
  std::vector<void *> ret={(void*)output_lengths,(void*)logits};
  printf("Done!\n");
  return ret;
}
onnxmodle::~onnxmodle()
{
  delete this->session;
}