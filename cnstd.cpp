#include "cnstd.h"
void filter_tag_det_res();
void sorted_boxes();
void get_rotate_crop_image();
void order_points_clockwise();
void clip_det_res();
CnStd::CnStd(/* args */)
{
    using namespace cnocrmodle;
    this->modle =std::unique_ptr<onnxmodle>(new onnxmodle(L"modle/ch_PP-OCRv3_det_infer.onnx",cnocrmodle::USE_DEVICE::TensorRT));
}

CnStd::~CnStd()
{
}

std::vector<cv::Mat> CnStd::detect(std::string path){
    //detect(cv::Mat img)
}
std::vector<cv::Mat> CnStd::detect(cv::Mat img){
    long long input_height=img.size[0];
    long long input_width=img.size[1];
    std::vector<void *>ret_data;
    ret_data=this->modle->run_en(input_width,input_height*input_width*3,img.data);
    int64_t length=(int64_t)ret_data[0];
    int64_t width=(int64_t)ret_data[1];
    auto ncdata=cv::Mat(length,width,CV_32FC1,(float*)ret_data[2]);
    //运行模型完毕
    filter_tag_det_res();
    sorted_boxes;
    std::vector<cv::Mat> retMat;
    //for 
    get_rotate_crop_image();
    return retMat;
}
void filter_tag_det_res(){
    order_points_clockwise();
    clip_det_res();
}
void sorted_boxes(){

}
void get_rotate_crop_image(){

}
void order_points_clockwise(){

}
void clip_det_res(){

}