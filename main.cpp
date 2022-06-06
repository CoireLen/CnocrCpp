#include <NumCPP.hpp>
#include "cnocr.h"

int main(){
    cnocr ocr;
    int i=0;
    for (auto line:ocr.ocr("D:\\Code\\cpp\\aimodle\\cnocr.png")){
        wcout<<i++<<":"<<line<<endl;
    }
}