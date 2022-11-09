# CnocrCpp

- 一份来自python cnocr的cpp实现，官方文档->[CnOcr文档](https://cnocr.readthedocs.io/zh/latest/)
- onnxruntime 自 https://github.com/microsoft/onnxruntime/releases 下载后 放入此文件夹或自行更改CMakeListsCMAK
- 另 需求 vcpkg (opencv) 包
## 20221109
英 文 识 别 可 用，但 是 有  [*空格*] 很 多，并英文识别**准确率**不准。en_dict词典首行加了空格。
## 20221107
添加模型切换框架
修复黑底白字对调错误
## 20221102

添加准确率

## 20220701
  
移除numcpp包的需求转为opencv
也就是说 完全不需要numcpp包了

## 使用演示

```cpp
#include "cnocr.h"
int main(){
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
 
```
```sh
0:cnocr自V2.1.2之后，可直接使用的模型包含两类:1）cnocr自己训练的模型，通常会包含PyTorch和,0.511289
1:ONNX版本；2）从其他ocr引擎搬运过来的训练好的外部模型，ONNX化后用于 cnocr中。,0.721081

0: T h e  R u s t S t a n d a r d  L i b r a r y  i s t h e  f o u n d a t i o n  o f  p o r t a b l e  R u s t  s o f t w a r e , a  s e t  o f m i n i m a l a n d  b a t t l e - t e s t e d s h a r e d a b s t r a c t i o n s  f o r  t h e ,0.535793
1: b r o a d e r  R u s t  e c o s y s t e m . I t  o f f e r s  c o r e  t y p e s , l i k e  V e c < T >  a n d  O p t i o n < T > , l i b r a r y - d e f i n e d  o p e r a t i o n s  o n l a n g u a g e p r i m i t i v e s , ,0.515892
2: s t a n d a r d  m a c r o s , I / O  a n d m u l t i t h r e a d i n g ,  a m o n g  m a n y  o t h e r  t h i n g s . ,0.579248 
```