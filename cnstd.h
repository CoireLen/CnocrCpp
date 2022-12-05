#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#ifndef _CNOCR_MODLE_H
#include "modle.h"
#endif

class CnStd
{
private:
    std::unique_ptr<cnocrmodle::onnxmodle> modle;
    cv::Mat detect_one(cv::Mat img,cv::Size2i resized_shape,bool preserve_aspect_ratio,float box_score_thresh,int min_box_size);
public:
    CnStd();
    std::vector<cv::Mat> detect(std::string path);
    std::vector<cv::Mat> detect(std::vector<cv::Mat> imgs);
    ~CnStd();
};

