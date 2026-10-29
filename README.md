# CnocrCpp

- 一份来自 Python [cnocr](https://cnocr.readthedocs.io/zh/latest/) 的 C++ 实现。
- 使用 OpenCV 做图像处理，使用 onnxruntime 做推理。
- 当前支持 **PP-OCRv3 文本检测 + 九种识别模型**（cnocr 自训练模型与 PaddleOCR 模型），可直接识别整张图片并按阅读顺序返回文本；模型文件缺失时参考 cnocr 的做法自动下载。

## 环境依赖

| 依赖 | 说明 |
| --- | --- |
| onnxruntime | 1.31.1，CMake 自动查找；未找到时按当前平台自动下载（`ONNXRUNTIME_VERSION` 可改） |
| OpenCV | 自动查找 vcpkg / apt / Homebrew 安装的 OpenCV（vcpkg：`opencv4`；Ubuntu：`libopencv-dev`；macOS：`brew install opencv`） |
| zlib | 解压自动下载的模型压缩包；OpenCV 已依赖，通常无需单独安装 |
| curl / wget | 运行期自动下载模型时使用；Windows 10+ 自带 curl，Linux/macOS 一般已安装 |
| CMake | ≥ 3.21 |
| 编译器 | MSVC 2022/2026、GCC ≥ 9、Clang / AppleClang ≥ 11，C++17 |

配置时会依次尝试：

1. `-DONNXRUNTIME_ROOT=<目录>`、环境变量 `ONNXRUNTIME_ROOT`；
2. 项目内 `onnxruntime*` / `third_party/onnxruntime*`；
3. 常见安装目录（`E:/Lib/onnxruntime-*`、`$HOME/onnxruntime-*`、`/opt/onnxruntime-*` 等）；
4. 系统或 vcpkg 中已安装的 onnxruntime；
5. 仍未找到时，按平台下载官方包到 `<build>/_deps/onnxruntime` 并解压（可用 `ONNXRUNTIME_PACKAGE_URL` 换源，`ONNXRUNTIME_PACKAGE_ARCHIVE` 指定本地压缩包，`ONNXRUNTIME_AUTO_DOWNLOAD=OFF` 关闭下载）。

```sh
# Windows（自动检测 vcpkg 与 onnxruntime，通常无需额外参数）
cmake -S . -B build
cmake --build build --config Release

# Linux（OpenCV 用系统包，onnxruntime 自动下载或已安装在 /opt）
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# macOS（Homebrew OpenCV 自动定位）
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# 也可显式指定依赖
cmake -S . -B build -DONNXRUNTIME_ROOT=<onnxruntime解压目录> -DOpenCV_DIR=<OpenCVConfig.cmake目录>
```

构建后会自动完成两件事：

1. 把 onnxruntime 运行库部署到输出目录：Windows 拷 `onnxruntime*.dll`，Linux/macOS 拷 `libonnxruntime.*` 并设置 `$ORIGIN` / `@loader_path` RPATH；OpenCV 的 DLL 由 vcpkg applocal 自动部署；
2. 把 `modle/*.onnx`、`dict/*.txt` 和示例图片复制到可执行文件旁边，程序按相对路径读取。

> CUDA / TensorRT 需要 onnxruntime 包内带有对应的 provider 运行库；当前 1.31.1 CPU 包缺少它们，代码会自动回退到 CPU，不会崩溃。macOS 官方包不含 CUDA/TensorRT，默认使用 CPU。

## 模型与字典

| 枚举值 | 模型文件 | 字典 | 输入 |
| --- | --- | --- | --- |
| `cnocr136fc` | `cnocr136fc.onnx` | `dict/label_cn.txt` | 1 通道，高 32 |
| `densenet_lite_136_gru` | `densenet_lite_136-gru.onnx` | `dict/label_cn_2.3.txt` | 1 通道，高 32 |
| `scene_densenet_136_gru` | `scene-densenet_lite_136-gru.onnx` | `dict/label_cn_2.3.txt` | 1 通道，高 32 |
| `doc_densenet_136_gru` | `doc-densenet_lite_136-gru.onnx` | `dict/label_cn_2.3.txt` | 1 通道，高 32 |
| `number_densenet_136_fc` | `number-densenet_lite_136-fc.onnx` | `dict/label_number.txt` | 1 通道，高 32 |
| `en_number` | `en_number_mobile_v2.0_rec_infer.onnx` | `dict/en_dict.txt` | 3 通道，高 32 |
| `chinese_cht` | `chinese_cht_PP-OCRv3_rec_infer.onnx` | `dict/chinese_cht_dict.txt` | 3 通道，高 32 |
| `ch_ppocrv3` | `ch_PP-OCRv3_rec_infer.onnx` | `dict/ppocr_keys_v1.txt` | 3 通道，高 48 |
| `ch_ppocr_mobile_v2` | `ch_ppocr_mobile_v2.0_rec_infer.onnx` | `dict/ppocr_keys_v1.txt` | 3 通道，高 32 |

- cnocr 自训练模型（前 5 个）blank 为最后一类，灰度输入 `x/255`；PaddleOCR 模型 blank 为第 0 类，彩色输入 `(x/255-0.5)/0.5`。
- 检测模型为 `modle/ch_PP-OCRv3_det_infer.onnx`（DBNet），`CnStd` 与开启检测的 `cnocr` 使用。
- `ch_ppocrv3` 按 PaddleOCR 的标准输入高度 48，其余模型为 32；高度已作为模型属性内置，无需手动配置。

## 模型自动下载

参考 Python cnocr 的做法：本地找不到模型时，运行期自动从 HuggingFace 仓库 `breezedeus/cnstd-cnocr-models` 下载对应压缩包，解压出 `.onnx` 后缓存复用。

- 查找顺序：`modle/<模型文件>` → 缓存目录 → 自动下载。
- 缓存目录：环境变量 `CNOCR_HOME`；未设置时 Windows 为 `%APPDATA%/cnocr`，Linux/macOS 为 `~/.cnocr`，模型保存在其下 `models/`。
- 下载源：默认 `https://hf-mirror.com`（国内可用），失败后回退 `https://huggingface.co`。可用 `CNOCR_MODEL_ENDPOINT` 指定其它镜像（替换仓库地址前缀，便于自建代理或离线镜像），`HF_ENDPOINT` 同样会被识别。
- 下载依赖系统里的 `curl` 或 `wget`。
- 字典文件随项目放在 `dict/`，不参与下载；缺少字典会直接报错提示。
- `cnocr(mode, device, use_detection, auto_download)` 的 `auto_download=false`、`CnStd(device, false)` 可关闭自动下载，此时缺失模型会抛异常并给出应放置的路径。

| 模型 | 仓库内压缩包 |
| --- | --- |
| `cnocr136fc` | `models/cnocr/2.2/densenet_lite_136-fc-onnx.zip` |
| `densenet_lite_136_gru` | `models/cnocr/2.3/densenet_lite_136-gru-onnx.zip` |
| `scene_densenet_136_gru` | `models/cnocr/2.3/scene-densenet_lite_136-gru-onnx.zip` |
| `doc_densenet_136_gru` | `models/cnocr/2.3/doc-densenet_lite_136-gru-onnx.zip` |
| `number_densenet_136_fc` | `models/cnocr/2.3/number-densenet_lite_136-fc-onnx.zip` |
| `en_number` | `models/cnocr/2.3/en_number_mobile_v2.0_rec_infer-onnx.zip` |
| `chinese_cht` | `models/cnocr/2.3/chinese_cht_PP-OCRv3_rec_infer-onnx.zip` |
| `ch_ppocrv3` | `models/cnocr/2.3/ch_PP-OCRv3_rec_infer-onnx.zip` |
| `ch_ppocr_mobile_v2` | `models/cnocr/2.3/ch_ppocr_mobile_v2.0_rec_infer-onnx.zip` |
| 检测模型 | `models/cnstd/1.2/ch_PP-OCRv3_det_infer-onnx.zip` |

## 使用方法

```cpp
#include <iostream>
#include "cnocr.h"
#include "cnstd.h"

int main() {
    // 默认使用 PP-OCRv3 检测模型切分文本行；device 可用 CPU / CUDA / TensorRT
    cnocr ocr(cnocr::USE_MODLE::cnocr136fc, cnocrmodle::USE_DEVICE::CPU);
    for (const auto& line : ocr.ocr("cnocr.png")) {
        // line.first 为 UTF-8 文本，可直接输出或参与后续处理
        std::cout << line.first << "  (置信度 " << line.second << ")" << std::endl;
    }

    // cnocr 2.3 通用中文模型：本地缺失时自动下载（第 4 个参数为 false 可关闭）
    cnocr newer(cnocr::USE_MODLE::densenet_lite_136_gru, cnocrmodle::USE_DEVICE::CPU,
                true, true);

    // 关闭检测，退回按行投影切分
    cnocr legacy(cnocr::USE_MODLE::cnocr136fc, cnocrmodle::USE_DEVICE::CPU, false);

    // 仅查看检测框
    CnStd cnstd;
    for (const DetBox& box : cnstd.detect("cnocr.png")) {
        // box.points[0..3] 为左上、右上、右下、左下；box.score 为置信度
    }
}
```

示例程序支持以下运行方式：

```sh
UseAiForWinML                    # 依次运行中文、英文、繁体三个识别模型
UseAiForWinML det [图片]         # 只运行检测模型并打印文本框
UseAiForWinML list               # 查看可用的模型别名
UseAiForWinML <模型> <图片>      # 运行指定模型，例如：UseAiForWinML zh-gru cnocr.png
```

模型别名：`zh`（cnocr136fc）、`en`（en_number）、`cht`（chinese_cht）、`zh-gru`、`scene`、`doc`、`number`、`ppv3`（ch_PP-OCRv3）、`ppv2`（ch_ppocr_mobile_v2.0）。

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

cnocr 2.3 通用中文模型实际输出（`UseAiForWinML zh-gru cnocr.png`，首次运行会自动下载模型）：

```text
0: cnocr自V2.1.2 之后，可直接使用的模型包含两类：1)cnocr 自己训练的模型，通常会包含PyTorch和  (置信度 0.527891)
1: ONNX 版本：2）从其他ocr引擎搬运过来的训练好的外部模型，ONNX化后用于cnocr中  (置信度 0.492485)
```

## 实现说明

> 完整的调用链、每一步的数据格式与张量布局见 [docs/LEARNING_GUIDE.md](docs/LEARNING_GUIDE.md)。

- **模块划分**：`modle.*` 负责 onnxruntime 会话与张量输入输出，`cnstd.*` 负责 DBNet 预处理与裁剪，`db_postprocess.*` 负责概率图后处理，`image_io.*` 负责按 UTF-8 路径读图，`cnocr.*` 负责模型配置、文本行准备与 CTC 解码；`model_registry.*` 维护模型清单与查找顺序，`model_files.*` 负责下载（curl/wget）与用 zlib 解压 zip。
- **文本编码**：字典、识别结果与图片路径全链路使用 UTF-8；`cnocr` 返回 `std::string`，Windows 控制台仅在示例程序中切换为 UTF-8 代码页。
- **检测**（`cnstd.cpp` + `db_postprocess.cpp`）：按 cnstd 的逻辑，把长边缩放到 768 且宽高取 32 的倍数，做 ImageNet 归一化后送入 DBNet；概率图 0.3 二值化、轮廓最小外接矩形、框内均值打分（阈值 0.3）、unclip 1.5 倍、还原坐标并按“从上到下、从左到右”排序；最后用透视变换裁剪，高宽比 ≥ 1.5 时旋转 90°。
- **识别**（`cnocr.cpp`）：文本行高度与通道由模型决定（cnocr 自训练模型与 mobile 系为 32、灰度 `x/255`；PP-OCRv3 中文模型为 48、3 通道 `(x/255-0.5)/0.5`），宽度按比例且最小 8。
- **CTC 解码**：按模型输出的有效长度解码，连续帧去重并跳过 blank；置信度取各时刻最大 softmax 概率的最小值。
- **字典加载**：按 UTF-8 读取并兼容 CRLF，处理 `<space>` 与 `<unused*>` 占位符；PaddleOCR 系字典在文件里补上首行 blank 与末行空格，与模型类别数对齐。
- **模型下载**：与 cnocr 一致按“压缩包 → 解压 `ppocr/*.onnx` 或 `<模型>/*.onnx`”处理，只取压缩包中的第一个 `.onnx`，支持 ZIP64 之外的常见 deflate / store 两种压缩方式。

## 更新记录

### 20261029

- 参考 Python cnocr 实现模型自动下载：新增 `model_registry`（模型表、缓存路径、`CNOCR_HOME`）与 `model_files`（curl/wget 下载、内置 zlib 解压 zip），按 `modle/` → 缓存 → 下载的顺序查找模型，`cnocr` / `CnStd` 构造函数新增 `auto_download` 参数。
- 新增 6 种识别模型：`densenet_lite_136-gru`、`scene-densenet_lite_136-gru`、`doc-densenet_lite_136-gru`、`number-densenet_lite_136-fc`、`ch_PP-OCRv3`、`ch_ppocr_mobile_v2.0`；检测模型同样支持自动下载。
- 按模型配置输入通道与高度（`ch_PP-OCRv3` 使用 48，其余 32），blank 索引改由模型表给出；新增 `label_cn_2.3.txt`、`label_number.txt`，重建 `ppocr_keys_v1.txt` 以对齐 PP 字典。
- 示例程序新增 `list` 列出可选模型、`<模型> <图片>` 单模型识别、`det [图片]` 仅检测，并对异常统一输出错误信息；图片无法读取时抛出 `无法读取图片` 而不再静默返回空结果。
- 文本接口改为 UTF-8 `std::string`：字典、识别结果不再经过 `wchar_t` 转换，移除原文本转换模块。
- 模型与字典路径改用 `std::filesystem::path`，适配 onnxruntime 的 `ORTCHAR_T`（Windows 宽字符 / POSIX 窄字符）；新增 `image_io`，图片路径同样按 UTF-8 解释。
- `main.cpp` 中 Windows 专用 API 用 `_WIN32` 隔离，可在 Linux / macOS 编译。
- CMake 跨平台：仅 Windows 启用 DLL 导出与 `/utf-8`，onnxruntime 改用 `find_path` / `find_library` 查找，非 Windows 拷贝 `so`/`dylib` 并设置 RPATH。
- CMake 自动导入依赖：自动探测 vcpkg 工具链、onnxruntime 常见安装位置与 OpenCV；找不到 onnxruntime 时按平台下载官方包（支持自定义下载地址 / 本地压缩包 / 关闭自动下载）。

### 20261028

- 支持 `ch_PP-OCRv3_det_infer.onnx` 文本检测，识别流程改为“先检测后识别”。
- 重写推理封装：修复输出指针悬垂、输出节点名读取错误、3 通道输入取址错误，新增 CPU 设备与 provider 自动回退。
- 修复 CTC blank 索引与置信度计算，修复字典加载的 CRLF / 末尾重复项问题。
- 主程序改为相对路径，新增 `det` 模式，输出统一为 UTF-8。
- 拆分模块（`det_box.h`、`db_postprocess.*`），复用文本行预处理与 CTC 解码，推理封装改为按模型输入顺序组装张量。

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
