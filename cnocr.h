#include "modle.h"
#include <string>
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
private:
    std::vector<cv::Mat> line_split(cv::Mat& inimg);
    std::vector<std::wstring> ocr_for_single_lines(std::vector<cv::Mat>& inimgs);
    std::wstring ctc_best(std::vector<uint16_t>);
    onnxmodle modle;
    wchar_t * ctc_path=L"label_cn.txt";
    std::vector<wchar_t> ctc_data;
    /* data */
public:
    cnocr(/* args */);
    std::vector<std::wstring> ocr(std::string path);
    std::vector<std::wstring> ocr(cv::Mat& img);
    std::vector<std::wstring> ocrtable(cv::Mat img);
    ~cnocr();
};

