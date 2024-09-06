#include "cnstd.h"
void filter_tag_det_res();
void sorted_boxes();
void get_rotate_crop_image();
void order_points_clockwise();
void clip_det_res();
CnStd::CnStd()
{
    std::locale lc("zh_CN.UTF-8");
    std::locale::global(lc);
    printf("Load cnstd Modle\n");
    using namespace cnocrmodle;
    this->modle =std::unique_ptr<onnxmodle>(new onnxmodle(L"modle/ch_PP-OCRv3_det_infer.onnx",cnocrmodle::USE_DEVICE::TensorRT));
}

CnStd::~CnStd()
{
}

std::vector<cv::Mat> CnStd::detect(std::string path){
    auto inimg =cv::imread(path,cv::IMREAD_GRAYSCALE);
    std::vector<cv::Mat> vmat;
    vmat.push_back(inimg);
    return this->detect(vmat);
}
std::vector<cv::Mat> CnStd::detect(std::vector<cv::Mat> imgs){
    cv::Size2i resized_shape {768,768};
    bool preserve_aspect_ratio {true};
    float box_score_thresh{0.3};
    int min_box_size{4};
    std::vector<cv::Mat> out;
    for (auto img:imgs){
        //img = self._preprocess_images(img)//格式化图片
        out.push_back(this->detect_one(img,resized_shape,preserve_aspect_ratio,box_score_thresh,min_box_size));
    }
    return out;
}
cv::Mat CnStd::detect_one(cv::Mat img,cv::Size2i resized_shape,bool preserve_aspect_ratio,float box_score_thresh,int min_box_size){
    long long input_height=img.size[0];
    long long input_width=img.size[1];
    auto ret_data=this->modle->run_std(input_height,input_width,input_height*input_width*3,img.data);
    std::for_each(ret_data.shape.begin(),ret_data.shape.end(),[](int64_t x){std::cout<<x<<" ";});
    int64_t length=(int64_t)ret_data.shape[0];
    int64_t width=(int64_t)ret_data.shape[1];
    auto ncdata=cv::Mat(length,width,CV_32FC1,(float*)ret_data.data);
    //运行模型完毕
    printf("(%d,%d)",length,width);
    for(int i=0;i<length*width;i++){
        std::cout<<ncdata.at<double>(i)<<"  , ";
        if (((i+1)%width)==0){
            putchar('\n');
        }
    }
    filter_tag_det_res();
    sorted_boxes;
    cv::Mat retMat;
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