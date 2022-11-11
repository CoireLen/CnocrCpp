#include "cnocr.h"
#include <boost/timer/timer.hpp>
#include <thread>
void run_cnocr();
int main(){
    //run_cnocr_1(ocr,"0123456789");
    boost::timer::auto_cpu_timer time1;
    run_cnocr();
}
void run_cnocr(){
    int i=0;
    cnocr ocr(cnocr::USE_MODLE::cnocr136fc,cnocrmodle::USE_DEVICE::CUDA);//使用中文模型，和GPU
    for (auto line:ocr.ocr("D:/Dev/CPP/CnocrCpp/cnocr.png")){
        std::wcout<<i++<<":"<<line.first<<","<<line.second<<std::endl;
    }
    cnocr enocr(cnocr::USE_MODLE::en_number);//使用英文识别模型
    for (auto line:enocr.ocr("D:/Dev/CPP/CnocrCpp/enocr.png")){
        std::wcout<<i++<<":"<<line.first<<","<<line.second<<std::endl;
    }
    cnocr ctocr(cnocr::USE_MODLE::chinese_cht);//使用繁体识别
    for (auto line:ctocr.ocr("D:/Dev/CPP/CnocrCpp/ctocr.png")){
        std::wcout<<i++<<":"<<line.first<<","<<line.second<<std::endl;
    }
}
