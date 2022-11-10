# CnocrCpp

- 一份来自python cnocr的cpp实现，官方文档->[CnOcr文档](https://cnocr.readthedocs.io/zh/latest/)
- onnxruntime 自 https://github.com/microsoft/onnxruntime/releases 下载后 放入此文件夹或自行更改CMakeListsCMAK
- 另 需求 vcpkg (opencv) 包
## 20221110
onnxruntime 可以调用CUDA进行模型计算
## 20221109
英 文 识 别 可 用，~~但 是 有  [*空格*] 很 多，~~并英文识别**准确率**不准。en_dict词典首行和末行加了空格。
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
#include <boost/timer/timer.hpp>
#include <thread>
void run_cnocr();
int main(){
    //run_cnocr_1(ocr,"0123456789");
    boost::timer::auto_cpu_timer time1;
    run_cnocr();
}
void run_cnocr(){
    cnocr enocr(cnocr::USE_MODLE::en_number);//使用英文识别模型
    cnocr ocr(cnocr::USE_MODLE::cnocr136fc,USE_DEVICE::CUDA);//使用中文模型，和GPU
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

2:The Rust Standard Library is the foundation of portable Rust software, a set of minimal and battle-tested shared abstractions for the,0.535793
3:broader Rust ecosystem. It offers core types, like Vec<T> and Option<T>, library-defined operations on language primitives,,0.515892
4:standard macros, I/O and multithreading, among many other things.,0.579248

3.633044s wall, 2.890625s user + 1.890625s system = 4.781250s CPU (131.6%)
```