
#include "modle.h"
#include <string>
#include <NumCpp.hpp>
#include <opencv2/opencv.hpp>
#include <fstream>
using namespace std;

class cnocr
{
private:
    nc::NdArray<uint8_t> read_img(string path);
    nc::NdArray<uint8_t> toNdArray(cv::Mat img);
    void show_img(nc::NdArray<uint8_t> img);
    cv::Mat toMat(nc::NdArray<uint8_t> img);
    cv::Mat toMat(nc::NdArray<float> img);
    vector<nc::NdArray<uint8_t>> line_split(nc::NdArray<uint8_t> img);
    vector<wstring> ocr_for_single_lines(vector<nc::NdArray<uint8_t>>);
    wstring ctc_best(nc::NdArray<uint32_t>);
    onnxmodle modle;
    wchar_t * ctc_path=L"label_cn.txt";
    vector<wchar_t> ctc_data;
    /* data */
public:
    cnocr(/* args */);
    vector<wstring> ocr(string path);
    ~cnocr();
};

