#include "modle.h"
#include <string>
#include <memory>
#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif
#include <opencv2/opencv.hpp>
#include <fstream>

class cnocr
{
public:
    enum USE_MODLE{
        cnocr136fc//中文ocr识别模型
        ,en_number //英文数字ocr识别模型
        ,chinese_cht //繁体中文ocr识别模型
    };
private:
    std::vector<cv::Mat> line_split(cv::Mat& inimg);
    std::vector<std::pair<std::wstring,float>> ocr_for_single_lines(std::vector<cv::Mat>& inimgs);
    std::wstring ctc_best(std::vector<uint16_t>);
    std::unique_ptr<onnxmodle> modle;
    wchar_t * ctc_path;
    std::vector<wchar_t> ctc_data;
    USE_MODLE use_modle;
    /* data */
public:
    cnocr(cnocr::USE_MODLE=USE_MODLE::cnocr136fc,USE_DEVICE device=USE_DEVICE::TensorRT);
    std::vector<std::pair<std::wstring,float>> ocr(std::string path);
    std::vector<std::pair<std::wstring,float>> ocr(cv::Mat& img);
    std::vector<std::wstring> ocrtable(cv::Mat img);
    ~cnocr();

};

