#include "cnocr.h"
void run_cnocr();
int main(){
    run_cnocr();
}
void run_cnocr(){
    cnocr ocr;
    int i=0;
    for (auto line:ocr.ocr("D:/Dev/CPP/CnocrCpp/cnocr.png")){
        std::wcout<<i++<<":"<<line.first<<","<<line.second<<std::endl;
    }
}