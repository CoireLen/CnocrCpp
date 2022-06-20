#include "modle.h"
#include <string>
#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif
#include <NumCpp.hpp>
#include <opencv2/opencv.hpp>
#include <fstream>

class cnocr
{
private:
    nc::NdArray<uint8_t> read_img(std::string path);
    nc::NdArray<uint8_t> toNdArray(cv::Mat img);
    void show_img(nc::NdArray<uint8_t> img);
    cv::Mat toMat(nc::NdArray<uint8_t> img);
    cv::Mat toMat(nc::NdArray<float> img);
    std::vector<nc::NdArray<uint8_t>> line_split(nc::NdArray<uint8_t> img);
    std::vector<std::wstring> ocr_for_single_lines(std::vector<nc::NdArray<uint8_t>>);
    std::wstring ctc_best(nc::NdArray<uint32_t>);
    onnxmodle modle;
    wchar_t * ctc_path=L"label_cn.txt";
    std::vector<wchar_t> ctc_data;
    /* data */
public:
    cnocr(/* args */);
    std::vector<std::wstring> ocr(std::string path);
    std::vector<std::wstring> ocr(cv::Mat img);
    ~cnocr();
};

