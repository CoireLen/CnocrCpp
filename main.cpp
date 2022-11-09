#include "cnocr.h"
void run_cnocr();
int main(){
    run_cnocr();
}
void run_cnocr(){
    cnocr ocr;
    cnocr enocr(cnocr::USE_MODLE::en_number);//使用英文识别模型
    int i=0;
    for (auto line:ocr.ocr("D:/Dev/CPP/CnocrCpp/cnocr.png")){
        std::wcout<<i++<<":"<<line.first<<","<<line.second<<std::endl;
    }
    for (auto line:enocr.ocr("D:/Dev/CPP/CnocrCpp/enocr.png")){
        std::wcout<<i++<<":"<<line.first<<","<<line.second<<std::endl;
    }
}