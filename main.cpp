#include "cnocr.h"
#include <chrono>
void run_cnocr();
using namespace std::chrono;
int main(){
    auto time_start=system_clock::now();
    run_cnocr();
    auto time_end=system_clock::now();
    std::cout <<"程序总耗时:"<<duration_cast<milliseconds>(time_end - time_start).count()<<"ms"<<std::endl;
}
void run_cnocr(){
    int i=0;
    auto time_start=system_clock::now();
    cnocr ocr(cnocr::USE_MODLE::cnocr136fc,cnocrmodle::USE_DEVICE::TensorRT);//使用中文模型，和GPU
    auto time_end=system_clock::now();
    std::cout <<"载入中文模型耗时:"<<duration_cast<milliseconds>(time_end - time_start).count()<<"ms"<<std::endl;
    auto cnRun_start=system_clock::now();
    for (auto line:ocr.ocr("D:/Dev/CPP/CnocrCpp/cnocr.png")){
        std::wcout<<i++<<":"<<line.first<<","<<line.second<<std::endl;
    }
    time_end=system_clock::now();
    std::cout <<"运行中文模型耗时:"<<duration_cast<milliseconds>(time_end - cnRun_start).count()<<"ms"<<std::endl;
    std::cout <<"中文模型总耗时:"<<duration_cast<milliseconds>(time_end - time_start).count()<<"ms"<<std::endl;
    cnocr enocr(cnocr::USE_MODLE::en_number);//使用英文识别模型
    for (auto line:enocr.ocr("D:/Dev/CPP/CnocrCpp/enocr.png")){
        std::wcout<<i++<<":"<<line.first<<","<<line.second<<std::endl;
    }
    cnocr ctocr(cnocr::USE_MODLE::chinese_cht);//使用繁体识别
    for (auto line:ctocr.ocr("D:/Dev/CPP/CnocrCpp/ctocr.png")){
        std::wcout<<i++<<":"<<line.first<<","<<line.second<<std::endl;
    }
}
