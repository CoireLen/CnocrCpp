# CnocrCpp

- 一份来自 Python [cnocr](https://cnocr.readthedocs.io/zh/latest/) 的 C++ 实现。
- 使用 OpenCV 做图像处理，使用 onnxruntime 做推理。
- 当前支持 **PP-OCRv3 文本检测 + 三种识别模型**，可直接识别整张图片并按阅读顺序返回文本。

## 环境依赖

| 依赖 | 说明 |
| --- | --- |
| onnxruntime | 1.31.1，默认路径 `E:/Lib/onnxruntime-win-x64-1.31.1`（需包含 `include/` 与 `lib/`） |
| OpenCV | 通过 vcpkg 安装，默认 vcpkg 根目录 `E:/Lib/vcpkg` |
| CMake | ≥ 3.21 |
| 编译器 | Visual Studio 2022/2026（MSVC），C++17 |

路径都可以在配置时通过缓存变量覆盖：

```sh
# 生成器按本机安装的 VS 版本选择，例如 "Visual Studio 17 2022"
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 \
  -DONNXRUNTIME_ROOT=<onnxruntime解压目录> \
  -DVCPKG_ROOT_DIR=<vcpkg目录>
cmake --build build --config Release
```

构建后会自动完成两件事：

1. 把 `onnxruntime.dll`、`onnxruntime_providers_shared.dll` 以及 OpenCV 的 DLL 部署到输出目录；
2. 把 `modle/*.onnx`、`dict/*.txt` 和示例图片复制到 `UseAiForWinML.exe` 旁边，程序按相对路径读取。

> CUDA / TensorRT 需要 onnxruntime 包内带有对应的 provider DLL；当前 1.31.1 CPU 包缺少它们，代码会自动回退到 CPU，不会崩溃。

## 模型与字典

| 模型文件 | 字典 | 说明 |
| --- | --- | --- |
| `modle/ch_PP-OCRv3_det_infer.onnx` | — | 文本检测（DBNet），`CnStd` 使用 |
| `modle/cnocr136fc.onnx` | `dict/label_cn.txt` | 中文识别，blank 为最后一类 |
| `modle/en_number_mobile_v2.0_rec_infer.onnx` | `dict/en_dict.txt` | 英文数字识别，3 通道 / blank 为 0 |
| `modle/chinese_cht_PP-OCRv3_rec_infer.onnx` | `dict/chinese_cht_dict.txt` | 繁体识别，3 通道 / blank 为 0 |

## 使用方法

```cpp
#include <iostream>
#include "cnocr.h"
#include "cnstd.h"

int main() {
    // 默认使用 PP-OCRv3 检测模型切分文本行；device 可用 CPU / CUDA / TensorRT
    cnocr ocr(cnocr::USE_MODLE::cnocr136fc, cnocrmodle::USE_DEVICE::CPU);
    for (const auto& line : ocr.ocr("cnocr.png")) {
        std::wcout << line.first << L"  (置信度 " << line.second << L")" << std::endl;
    }

    // 关闭检测，退回按行投影切分
    cnocr legacy(cnocr::USE_MODLE::cnocr136fc, cnocrmodle::USE_DEVICE::CPU, false);

    // 仅查看检测框
    CnStd cnstd;
    for (const DetBox& box : cnstd.detect("cnocr.png")) {
        // box.points[0..3] 为左上、右上、右下、左下；box.score 为置信度
    }
}
```

示例程序支持两种运行方式：

```sh
UseAiForWinML.exe        # 依次运行中文、英文、繁体三个识别模型
UseAiForWinML.exe det    # 只运行检测模型并打印文本框
```

中文模型实际输出（`cnocr.png`，CPU）：

```text
0: cnocr 自V2.1.2之后，可直接使用的模型包含两类:1) cnocr自己训练的模型，通常会包含PyTorch和  (置信度 0.526114)
1: ONNX版本；2）从其他ocr引擎搬运过来的训练好的外部模型，ONNX化后用于 cnocr中  (置信度 0.773391)
```

英文模型实际输出（`enocr.png`）：

```text
0: The Rust Standard Library is the foundation of portable Rust software, a set of minimal and battle-tested shared abstractions for the
1: broader Rust ecosystem. It offers core types, like Vec<T> and Option<T> , library-defined operations on language primitives
2: standard macros I/O and multithreading among many other things
```

## 实现说明

- **模块划分**：`modle.*` 负责 onnxruntime 会话与张量输入输出，`cnstd.*` 负责 DBNet 预处理与裁剪，`db_postprocess.*` 负责概率图后处理，`text_utils.*` 负责 UTF-8/宽字符转换，`cnocr.*` 负责模型配置、文本行准备与 CTC 解码。
- **检测**（`cnstd.cpp` + `db_postprocess.cpp`）：按 cnstd 的逻辑，把长边缩放到 768 且宽高取 32 的倍数，做 ImageNet 归一化后送入 DBNet；概率图 0.3 二值化、轮廓最小外接矩形、框内均值打分（阈值 0.3）、unclip 1.5 倍、还原坐标并按“从上到下、从左到右”排序；最后用透视变换裁剪，高宽比 ≥ 1.5 时旋转 90°。
- **识别**（`cnocr.cpp`）：文本行高度固定 32、宽度按比例（最小 8）；`cnocr136fc` 使用灰度 `x/255`，英文与繁体模型使用 3 通道 `(x/255-0.5)/0.5`。
- **CTC 解码**：按模型输出的有效长度解码，连续帧去重并跳过 blank；置信度取各时刻最大 softmax 概率的最小值。
- **字典加载**：按 UTF-8 读取并兼容 CRLF，处理 `<space>` 与 `<unused*>` 占位符。

## 更新记录

### 20261028

- 支持 `ch_PP-OCRv3_det_infer.onnx` 文本检测，识别流程改为“先检测后识别”。
- 重写推理封装：修复输出指针悬垂、输出节点名读取错误、3 通道输入取址错误，新增 CPU 设备与 provider 自动回退。
- 修复 CTC blank 索引与置信度计算，修复字典加载的 CRLF / 末尾重复项问题。
- 主程序改为相对路径，新增 `det` 模式，输出统一为 UTF-8。
- 拆分模块（`det_box.h`、`text_utils.*`、`db_postprocess.*`），复用文本行预处理与 CTC 解码，推理封装改为按模型输入顺序组装张量。

### 20240909

修改以支持生成 DLL。

### 20240906

更新支持 onnxruntime 1.19。

### 20240401

修改以支持新版 onnxruntime。

### 20221111

支持繁体中文识别。

### 20221110

opencv Mat 替换为 UMat，性能得到了一点点提升；onnxruntime 可以调用 CUDA 进行模型计算。

### 20221109

英文识别可用，但英文识别**准确率**不准。`en_dict` 词典首行和末行加了空格。

### 20221107

添加模型切换框架；修复黑底白字对调错误。

### 20221102

添加准确率。

### 20220701

移除 numcpp 包的需求转为 opencv，也就是说完全不需要 numcpp 包了。
