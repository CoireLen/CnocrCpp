# CnocrCpp

- 一份来自python cnocr的cpp实现，官方文档->[CnOcr文档](https://cnocr.readthedocs.io/zh/latest/)
- onnxruntime 自 https://github.com/microsoft/onnxruntime/releases 下载后 放入此文件夹或自行更改CMakeListsCMAK
- 另 需求 vcpkg (opencv) 包

## 20220701

移除numcpp包的需求转为opencv
也就是说 完全不需要numcpp包了

## 使用演示

```cpp
#include "cnocr.h"
int main(){
    cnocr ocr;
    int i=0;
    for (auto line:ocr.ocr("img.png")){
        std::wcout<<i++<<":"<<line<<std::endl;
    }
}

```
