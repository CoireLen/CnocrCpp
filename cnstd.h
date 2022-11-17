#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include "modle.h"
class CnStd
{
private:
    std::unique_ptr<cnocrmodle::onnxmodle> modle;
public:
    CnStd(/* args */);
    ~CnStd();
    std::vector<cv::Mat> detect(std::string path);
    std::vector<cv::Mat> detect(cv::Mat img);
};

