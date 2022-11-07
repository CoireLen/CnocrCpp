#include "cnocr.h"
void run_cnocr();
int main(){
    run_cnocr();
}
void run_cnocr(){
    cnocr ocr(cnocr::USE_MODLE::en_number);
    int i=0;
    for (auto line:ocr.ocr("D:/2.png")){
        std::wcout<<i++<<":"<<line.first<<","<<line.second<<std::endl;
    }
}