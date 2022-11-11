#include "cnocr.h"
#include <algorithm>
#include <locale>
cnocr::cnocr(cnocr::USE_MODLE themodle,cnocrmodle::USE_DEVICE device)
{
    using namespace cnocrmodle;
    std::locale lc("zh_CN.UTF-8");
    std::locale::global(lc);
    this->use_modle=themodle;
    switch (themodle)
    {
    case USE_MODLE::cnocr136fc:
        this->ctc_path=L"dict/label_cn.txt";
        this->modle =std::unique_ptr<onnxmodle>(new onnxmodle(L"modle/cnocr136fc.onnx",device));
        break;
    case USE_MODLE::en_number:
        this->ctc_path=L"dict/en_dict.txt";
        this->modle =std::unique_ptr<onnxmodle>(new onnxmodle(L"modle/en_number_mobile_v2.0_rec_infer.onnx",device));
        break;
    case USE_MODLE::chinese_cht:
        this->ctc_path=L"dict/chinese_cht_dict.txt";
        this->modle =std::unique_ptr<onnxmodle>(new onnxmodle(L"modle/chinese_cht_PP-OCRv3_rec_infer.onnx",device));
        break;
    default:
        break;
    }
    //载入中文字符
    wchar_t buffer[100];
    std::wifstream ctc_char(ctc_path);
    while (!ctc_char.eof() )
    {
        ctc_char.getline(buffer,100);
        ctc_data.push_back(buffer[0]);
    }
}
/// @brief 读入图片路径返回图片中的字符串
/// @param path 图片文件路径
/// @return 识别到的中文字符串列表
std::vector<std::pair<std::wstring,float>> cnocr::ocr(std::string path){
    auto inimg =cv::imread(path,cv::IMREAD_GRAYSCALE);
    cv::Mat outimg=inimg;
    std::vector<std::pair<std::wstring,float>> res;
    auto imgcol=outimg.cols;
    auto imgrow=outimg.rows;
    if (std::min(imgrow,imgcol) < 2){
        return res;
    }
    if (cv::sum(outimg.col(0))[0]/outimg.rows < 145) // 把黑底白字的图片对调为白底黑字
    {
        outimg=255-outimg;
    }
    auto imgs=line_split(outimg);
    res=ocr_for_single_lines(imgs);
    return res;
}
std::vector<std::pair<std::wstring,float>> cnocr::ocr(cv::Mat& inimg){
    cv::Mat outimg;
    cv::cvtColor(inimg,outimg,cv::COLOR_RGB2GRAY);
    std::vector<std::pair<std::wstring,float>> res;
    auto imgcol=outimg.cols;
    auto imgrow=outimg.rows;
    if (std::min(imgrow,imgcol) < 2){
        return res;
    }
    if (cv::sum(outimg.col(0))[0] < 145) // 把黑底白字的图片对调为白底黑字
    {
        outimg=255-outimg;
    }
    auto imgs=line_split(outimg);
    res=ocr_for_single_lines(imgs);
    return res;
}
/// @brief 读取Mat格式的图片 横向切分成 单列有文字的 图片列表
/// @param inimg 
/// @return 单列有文字的 图片列表
std::vector<cv::Mat> cnocr::line_split(cv::Mat& inimg){
    std::vector<cv::Mat> list;
    auto imgcol=inimg.cols;
    auto imgrow=inimg.rows;
    auto bij=inimg.row(0);
    for (int i=0;i<imgrow;i++){
        *(bij.data+i)=255;
    }
    int lineforchar=0;
    int lineforcharstart=0;
    for (int i=0;i<imgrow;i++){
        cv::Mat res=bij-inimg.row(i);
        if (*std::max_element(res.begin<uchar>(),res.end<uchar>())>100){
            if (lineforchar+1==i){
                lineforchar++;
            }
            else{
                if (lineforchar-lineforcharstart>7){
                    int start=lineforcharstart;
                    int end=lineforchar;
                    if (start>0){
                        start-=1;
                    }
                    if (end<imgcol-1){
                        end+=1;
                    }
                    list.push_back(inimg(cv::Rect(0,start,inimg.cols,end-start)));
                }
                lineforcharstart=i;
                lineforchar=i;
            }
        }
    }
    if (lineforchar-lineforcharstart>5){
        int start=lineforcharstart;
        int end=lineforchar;
        if (start>0){
            start-=1;
        }
        if (end<imgcol-1){
            end+=1;
        }
        list.push_back(inimg(cv::Rect(0,start,inimg.cols,end-start)));
    }
    return list;
}
/// @brief 
/// @param input 
void softmax(cv::Mat &input){
    for (int i=0;i<input.rows;i++){
        auto ncdata=input.row(i);
        double t=(*std::max_element(ncdata.begin<float>(),ncdata.end<float>()));
        cv::exp(ncdata-t,ncdata);
        double t1=cv::sum(ncdata)[0];
        ncdata=ncdata/t1;
    }
}

template<class ForwardIterator>
inline size_t argmax(ForwardIterator first, ForwardIterator last)
{
    return std::distance(first, std::max_element(first, last));
}
std::vector<uint16_t> vargmax(cv::Mat input) {
    std::vector<uint16_t> res;
    for (int i=0;i<input.rows;i++){
        auto ncdata=input.row(i);
        res.push_back(argmax(ncdata.begin<float>(),ncdata.end<float>()));
    }
    return res;
}
std::vector<std::pair<std::wstring,float>> cnocr::ocr_for_single_lines(std::vector<cv::Mat>& imgs){
    std::vector<std::pair<std::wstring,float>>res;
    if (imgs.size()==0){
        return res;
    }
    std::cout<<"have lines x for ocr:"<<imgs.size()<<std::endl;
    //res = self.rec_model.recognize(img_list, batch_size=batch_size)
    for (auto img:imgs){
        auto imgcol=img.cols;
        auto imgrow=img.rows;
        std::cout <<"imgSize:"<<imgcol<<","<<imgrow<<std::endl;
        //等比例放图像至高度32
        auto imgmat=img;
        float ratio=imgrow/32.0;
        cv::Size sz((int)(imgcol/ratio),32);

        cv::Mat imgresize(sz,CV_8UC1);
        cv::resize(imgmat,imgresize,sz);
        //调用模型
        long long input_height=imgresize.size[0];
        long long input_width=imgresize.size[1];
        cv::Mat ncdata;
        std::vector<void *>ret_data;
        switch (this->use_modle)
        {
        case USE_MODLE::cnocr136fc:
            ret_data=this->modle->run(input_width,input_height*input_width,imgresize.data);
            ncdata=cv::Mat(*(int64_t*)ret_data[0],6674,CV_32FC1,(float*)ret_data[1]);
            softmax(ncdata);
            break;
        
        case USE_MODLE::en_number:
        case USE_MODLE::chinese_cht:
            ret_data=this->modle->run_en(input_width,input_height*input_width*3,imgresize.data);
            int64_t length=(int64_t)ret_data[0];
            int64_t width=(int64_t)ret_data[1];
            ncdata=cv::Mat(length,width,CV_32FC1,(float*)ret_data[2]);
            break;
        }
        
        auto matdata=ncdata.clone();
        std::vector<uint16_t> best_path=vargmax(ncdata);
        //计算准确率
        cv::reduce(matdata,matdata,1,cv::REDUCE_MAX);
        cv::reduce(matdata,matdata,0,cv::REDUCE_MIN);
        float zql=matdata.at<float>(0,0);//准确率
        //std::cout<<"Accuracy:"<<zql<< "<"<<matdata.cols<<","<<matdata.rows<<">"<<std::endl;
        res.push_back(std::pair<std::wstring,float>(ctc_best(best_path),zql));
    }
    return res;
}
std::wstring cnocr::ctc_best(std::vector<uint16_t> data){
    std::wstring res;
    std::vector<uint32_t> vui;//消除重复的
    for (auto i=data.begin();i!=data.end();i++){
        if (vui.size()!=0){
            if (vui[vui.size()-1]!=(uint32_t)*i){
                vui.push_back((uint32_t)*i);
            }
        }
        else{
            vui.push_back((uint32_t)*i);
        }
    }
    for (auto i:vui){
        if (i<ctc_data.size()){
            res.push_back(ctc_data[i]);
        }
    }
    return res;
}

cnocr::~cnocr()
{
}