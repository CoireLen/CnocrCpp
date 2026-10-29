# CnocrCpp 代码学习指南

本文从「一次 `ocr()` 调用」出发，讲清楚项目的模块划分、调用链、每个函数做了什么，以及图像 / 张量在每一步的**数据类型、内存布局和尺寸变化**。

- 目标读者：第一次读这个仓库、想理解 OCR 推理流程或准备扩展模型的人。
- 配套文档：构建、依赖与环境变量见根目录 [README.md](../README.md)。

---

## 1. 项目定位

CnocrCpp 是 Python [cnocr](https://cnocr.readthedocs.io/zh/latest/) / [cnstd](https://github.com/breezedeus/cnstd) 的 C++ 精简实现：

- 图像处理：OpenCV（读图、缩放、翻转、透视变换、轮廓）。
- 推理：ONNX Runtime（CPU 默认，CUDA / TensorRT 可选，provider 不可用时自动回退 CPU）。
- 文本检测：PP-OCRv3 DBNet，输出旋转文本框。
- 文本识别：CTC 解码，内置 9 种识别模型（cnocr 自训练 4 种 + PaddleOCR 系 5 种）。
- 模型管理：参考 Python cnocr，缺失时自动从 HuggingFace 镜像下载并解压。
- 编码：内部全链路 UTF-8（`std::string` + `std::filesystem::path`）。

---

## 2. 代码地图

### 2.1 文件与职责

| 文件 | 层次 | 职责 | 关键符号 |
| --- | --- | --- | --- |
| `main.cpp` | 应用层 | 命令行入口、控制台 UTF-8、异常兜底 | `main`、`run_ocr`、`run_cnstd`、`kModels` |
| `cnocr.h/.cpp` | API 层 | 对外识别接口：切行 → 识别 → 解码 | `cnocr::ocr`、`recognize_line`、`decode_ctc`、`prepare_line_image`、`load_dict` |
| `cnstd.h/.cpp` | API 层 | 文本检测：预处理 → 推理 → 后处理 → 裁剪 | `CnStd::detect`、`CnStd::crop`、`resized_shape_of`、`fill_normalized_chw` |
| `db_postprocess.h/.cpp` | 后处理 | DBNet 概率图 → 文本框、打分、外扩、排序 | `boxes_from_prob`、`mini_boxes`、`offset_polygon`、`sort_reading_order` |
| `modle.h/.cpp` | 推理层 | ONNX Runtime 会话封装、张量组装与拷贝 | `onnxmodle::run_rec`、`run_det`、`find_output` |
| `model_registry.h/.cpp` | 资源层 | 模型清单、字典路径、缓存查找、下载调度 | `ModelSpec`、`ensure_model`、`ensure_det_model` |
| `model_files.h/.cpp` | 资源层 | HTTP 下载（curl/wget）、zip 解压（zlib） | `download`、`extract_onnx` |
| `image_io.h/.cpp` | 基础设施 | 按 UTF-8 路径读图 | `imread_utf8` |
| `det_box.h` | 数据结构 | 四点文本框与点距工具 | `DetBox`、`box_point_distance` |
| `CMakeLists.txt` | 构建 | 依赖探测、onnxruntime 下载、运行库与资源部署 | — |

### 2.2 依赖关系

```text
                 ┌─────────────┐
                 │  main.cpp   │  命令行：list / det [图] / <模型> <图>
                 └──────┬──────┘
                        │
        ┌───────────────┴────────────────┐
        ▼                                ▼
 ┌─────────────┐                  ┌─────────────┐
 │   cnocr     │─────────────────▶│   CnStd     │
 │ 识别 API    │  组合（可选检测） │  检测 API   │
 └──┬───┬───┬──┘                  └──┬───┬──────┘
    │   │   │                        │   │
    │   │   └──────────────┐         │   └───────────────┐
    │   │                  ▼         ▼                   ▼
    │   │           ┌─────────────────────┐      ┌──────────────────┐
    │   │           │ onnxmodle（ORT 封装）│      │ db_postprocess   │
    │   │           └─────────────────────┘      └──────────────────┘
    │   │                  ▲
    │   ▼                  │
    │ ┌────────────────────┴────┐
    │ │ model_registry（模型表）│──────▶ model_files（下载 + 解压）
    │ └─────────────────────────┘
    ▼
 image_io / OpenCV
```

**分层原则**：`main` 只负责展示；`cnocr` / `CnStd` 负责业务流程；`onnxmodle` 只管张量和 `session.Run`，不关心图像含义；`db_postprocess` 是纯算法（输入概率图，输出框）；`model_registry` / `model_files` 只处理文件。

---

## 3. 一次 `ocr()` 的完整工作流程

### 3.1 总览

```text
图片文件
  │ imread_utf8                       → cv::Mat BGR, CV_8UC3, H×W
  ▼
检测预处理 fill_normalized_chw        → vector<float> CHW, 3×H'×W'（ImageNet 归一化）
  ▼
检测推理 run_det                      → det_return{ prob, H_p, W_p }
  ▼
后处理 boxes_from_prob                → vector<DetBox>（原图坐标，阅读顺序）
  ▼
逐框裁剪 CnStd::crop                  → vector<cv::Mat>（水平文本行）
  ▼
识别预处理 prepare_line_image         → 每行 CV_8UC(1|3)，高 32/48，宽度等比
  ▼
识别推理 run_rec                      → rec_return{ logits[T×C], lengths, valid_len }
  ▼
CTC 解码 decode_ctc                   → vector<pair<string(UTF-8), float>>
```

关闭检测（`use_detection=false`）时，第 2~5 步替换为 `line_split`（行投影切分），其余不变。

### 3.2 启动阶段：构造函数做了什么

```cpp
cnocr model(cnocr::USE_MODLE::cnocr136fc, cnocrmodle::USE_DEVICE::CPU);
```

`cnocr::cnocr`（`cnocr.cpp:122`）依次执行：

1. `model_name_of(mode)`：枚举 → 字符串名（如 `cnocr136fc` → `"cnocr136fc"`）。
2. `find_recognizer(name)`：在 `all_recognizers()` 里找到 `ModelSpec`；找不到抛 `未注册的识别模型`。
3. `ensure_model(spec, auto_download)`：定位 `.onnx` 文件（详见第 6 节）。
4. `onnxmodle` 构造：创建 `Ort::Env`、`SessionOptions`（图优化 `ORT_ENABLE_ALL` + 可选 provider），加载模型，读取所有输入/输出名称与输入形状。
5. `load_dict(dict_path)`：读取字典 → `vector<string>`（UTF-8），空字典抛错。
6. 从 `ModelSpec` 写入 `input_channels_`、`input_height_`、`blank_index_`。
7. 若启用检测，构造 `CnStd(device, auto_download)`，同样先 `ensure_det_model` 再建会话。

> 注意：`CnStd` 与 `cnocr` **各自持有一个 ORT 会话**（检测模型 + 识别模型），互不共享。

`cnstd::CnStd` 构造（`cnstd.cpp:68`）只有一步：用检测模型的 `ModelSpec` 建 `onnxmodle`。

### 3.3 检测阶段

#### (1) 读图：`imageio::imread_utf8`

`image_io.cpp:11` 先用 `std::ifstream` 以二进制读入全部字节，再 `cv::imdecode(bytes, IMREAD_COLOR)`。

- 为什么不用 `cv::imread`：Windows 上 `imread(const std::string&)` 按本地 ANSI 编码解释路径，中文路径会失败；先自己读字节再 `imdecode` 可以完全绕开路径编码问题。
- 输出固定为 **BGR、3 通道、`CV_8UC3`**（灰度/带 alpha 的图会被 OpenCV 直接转成 3 通道）。
- 文件不存在或解码失败 → 空 `cv::Mat`；路径版接口抛 `无法读取图片: <path>`。

#### (2) 尺寸规划：`resized_shape_of`

`cnstd.cpp:16`，默认目标 `768×768`、保持长宽比：

```text
ratio = max(768/src.h, 768/src.w) 的倒数，即让较长边刚好缩到 768
new_w = int(src.w * ratio)
new_h = int(src.h * ratio)
new_w = round(new_w / 32) * 32   // 至少 32
new_h = round(new_h / 32) * 32   // 至少 32
```

即：长边缩到 768，两维都对齐到 32 的倍数（DBNet 下采样要求）。`preserve_aspect_ratio=false` 时直接返回目标尺寸。

以示例图 `cnocr.png`（747×57）为例：

```text
ratio = 768 / 747 ≈ 1.0281
new_w = 768
new_h = int(57 × 1.0281) = 58 → round(58/32)=2 → 64
target = 768 × 64
```

#### (3) 归一化：`fill_normalized_chw`

`cnstd.cpp:36`，把 `cv::Mat` 变成 ONNX 需要的 **CHW float**：

| 步骤 | 操作 | 数据形态 |
| --- | --- | --- |
| 通道统一 | 1 通道 → `GRAY2BGR`；4 通道 → `BGRA2BGR`；3 通道原样 | `CV_8UC3` |
| 缩放 | `cv::resize(..., target, INTER_LINEAR)` | `CV_8UC3, H'×W'` |
| 归一化 | `x/255`，再减均值除标准差（ImageNet） | float |
| 重排 | HWC → CHW，`chw[c*H'*W' + y*W' + x]` | `vector<float>`, 3·H'·W' |

均值/标准差：`mean = {0.485, 0.456, 0.406}`，`std = {0.229, 0.224, 0.225}`（BGR 顺序）。

768×64 时向量长度 = `3 × 768 × 64 = 147456` 个 float（约 576 KB）。

#### (4) 推理：`onnxmodle::run_det`

`modle.cpp:164`：

1. 用 `Ort::MemoryInfo::CreateCpu` 建 CPU 内存描述。
2. `CreateTensor<float>(chw.data(), 3*H*W, {1,3,H,W})`——**不复制数据**，直接引用 `chw` 缓冲区（因此 `chw` 必须活到 `Run` 结束，这里是局部变量，天然满足）。
3. `session.Run(输入名, 输入值, 输出名)`。
4. `find_output(outputs, 4, FLOAT)`：按「秩 = 4 且类型 float」挑概率图，而不是假设它是第 0 个输出（不同导出工具输出顺序不固定）。
5. 读输出的 `shape[2]`、`shape[3]` 作为概率图高宽，把整块数据 `copy` 到 `det_return.prob`。

为什么必须拷贝：`Ort::Value` 是输出结果的所有者，函数返回时会被析构，直接返回裸指针会悬垂。拷贝后 `prob` 是独立内存，后面可安全包成 `cv::Mat`。

#### (5) 后处理：`db_postprocess::boxes_from_prob`

`db_postprocess.cpp:177`，依次：

1. **二值化**：逐像素 `prob > 0.3` → `CV_8UC1`（0/255）。
2. **找轮廓**：`findContours(RETR_LIST, CHAIN_APPROX_SIMPLE)`，最多处理前 `max_candidates=1000` 个。
3. **最小外接矩形**：`mini_boxes()` → `cv::minAreaRect` 得 4 点，按 x 排序后还原 **左上、右上、右下、左下**；返回较短边作为过滤尺寸；`< 3px` 丢弃。
4. **框内均值打分**：`box_score_fast()` 在矩形 bbox 内建 mask（`fillPoly` 画四边形），`cv::mean(prob(roi), mask)`；`< 0.3` 丢弃。**分数在膨胀前计算**（与 PaddleOCR 一致）。
5. **外扩（unclip）**：`distance = area × 1.5 / perimeter`，`offset_polygon()` 沿各边法线做 miter 外扩；外扩后再取 `minAreaRect`，短边 `< 5px` 丢弃。
6. **坐标还原**：`x' = round(x / W_p × src.w)`，`y' = round(y / H_p × src.h)`，并 clamp 到图内。**用概率图尺寸做分母**，所以即使模型输出经过下采样也能正确还原。
7. **点序统一**：`order_points_clockwise()` 用「按 x 分左右两组，再按 y 定上下」得到左上/右上/右下/左下。
8. **尺寸过滤**：宽 = 上边距离、高 = 左边距离，任一边 `< min_box_size(4)` 丢弃。
9. **阅读顺序排序**：`sort_reading_order()` 用**插入排序**（不是 `std::sort`）。比较规则：两框纵向重叠比 > 0.5 时比左边界，否则比上边界。该比较不是严格弱序，`std::sort` 行为未定义，插入排序可保证结果稳定、与 Python 实现一致。

输出 `DetBox`：

```cpp
struct DetBox {
    cv::Point2f points[4];  // 左上、右上、右下、左下（原图坐标）
    float score;            // 框内概率均值
};
```

#### (6) 裁剪：`CnStd::crop`

`cnstd.cpp:103`：

- 目标宽 = 上下两条边长的较大值；目标高 = 左右两条边长的较大值（等价于透视变换后的正矩形尺寸）。
- `getPerspectiveTransform(box.points, 目标四角)` + `warpPerspective(..., INTER_CUBIC, BORDER_REPLICATE)`。
- 若裁剪结果 `高/宽 ≥ 1.5`（竖排文字），`cv::ROTATE_90_COUNTERCLOCKWISE` 旋转为横向。

### 3.4 识别阶段

#### (1) 单行预处理：`prepare_line_image`

`cnocr.cpp:64`：

| 模型通道 | 处理 | 结果 |
| --- | --- | --- |
| `channels == 1` | 1 通道原样；3 通道 `BGR2GRAY`；4 通道 `BGRA2GRAY` | 单通道 |
| `channels == 3` | 3 通道原样；4 通道 `BGRA2BGR`；1 通道 `GRAY2BGR` | 3 通道 |

然后：

```text
target_w = max(int(cols × H / rows), 8)     // H = 32 或 48
resize 到 (target_w, H)，INTER_LINEAR
```

即**高度固定、宽度等比、最小 8 像素**，不做厚度方向的 padding（与 cnocr 的 `resize_norm_img` 简化一致）。

#### (2) 推理：`onnxmodle::run_rec`

`modle.cpp:94`，输入是 `cv::Mat.data`（uint8，HWC 紧密排列）：

1. 手动完成 **HWC(uint8) → CHW(float)**：
   - 1 通道：`x / 255`
   - 3 通道：`(x/255 - 0.5) / 0.5`（映射到 `[-1,1]`，PaddleOCR 系列要求）
   - 写入 `pixel_data[c*H*W + i]`。
2. 建 4 维图像张量 `{1, C, H, W}`。
3. **按模型的输入顺序组装**：遍历 `input_names_`，若该输入在建模时记录的形状是 1 维，则填 `input_lengths = width`（int64，`{1}`）；否则填入图像张量。这样兼容：
   - `cnocr136fc` / `densenet_lite_136-gru` / `number-...`：两个输入 `x`、`input_lengths`；
   - PaddleOCR 模型：只有一个输入 `x`。
   出现第二个图像输入则抛错（当前模型都不需要）。
4. `Run` 后用 `find_output` 取：
   - **3 维 float** → `logits`，形状 `{1, T, C}`；记录 `seq_len=T`、`num_classes=C`；
   - **1 维 int64** → `output_lengths`（有效时间步）。
   两者都拷贝进 `rec_return`。
5. `valid_len = clamp(lengths[0], 0, T)`；没有 lengths 输出时等于 `T`。

`rec_return` 结构：

```cpp
struct rec_return {
    std::vector<float>  logits;      // [T, C] 行优先，已拷贝
    std::vector<int64_t> lengths;    // 通常 [1]
    int64_t seq_len, num_classes, valid_len;
};
```

#### (3) CTC 解码：`decode_ctc`

`cnocr.cpp:85`，对 `t = 0..valid_len-1` 的每一帧：

1. **argmax**：在该行 `C` 个 logits 中找最大类别 `best`。
2. **置信度**：`1 / Σ exp(row[c] - max)`，等于该帧最大 softmax 概率（用减最大值保证数值稳定）。整个文本的置信度取**所有帧的最小值**——所以长文本、生僻字会把整体分数拉得很低。
3. **去重 + 跳 blank**：`best != previous && best != blank_index` 时输出 `dict[best]`；`previous = best`。
4. 文本以 UTF-8 字节直接追加到 `std::string`（字典本身是 UTF-8）。

**blank 规则**（两类模型不同）：

| 模型族 | 字典内容 | 输出类别数 C | blank 下标 |
| --- | --- | --- | --- |
| cnocr 自训练（`fc`/`gru`） | 纯字符，共 N 行 | N + 1 | `N`（最后一类） |
| PaddleOCR（`*_PP-OCRv3`、mobile、en_number） | 第 0 行是 blank 占位，末行是空格 | N（= 字典行数） | `0` |

`cnocr.cpp:140` 用 `spec->blank_index < 0 ? dict.size() : spec->blank_index` 统一这两种情况。

### 3.5 无检测分支：`line_split`

`cnocr.cpp:210`，仅当 `use_detection=false`：

1. 转灰度（如需要）。
2. `inverted = 255 - gray`，按行取最大值 `cv::reduce(..., REDUCE_MAX)`。
3. 行最大值 `> 100` 视为有文字，连续行构成一段；段高 `> 5` 才保留，上下各扩 1 像素后裁出整行。

这是早期 cnocr 的简化行切分，适合扫描文档，不适合复杂排版。

### 3.6 输出

`ocr()` 返回 `vector<pair<string, float>>`：UTF-8 文本 + 置信度，顺序即 `DetBox` 的阅读顺序。`main.cpp` 中 Windows 用 `SetConsoleOutputCP(CP_UTF8)` 保证控制台正常显示。

---

## 4. 关键函数速查

| 函数 | 位置 | 输入 → 输出 | 关键点 |
| --- | --- | --- | --- |
| `model_name_of` | `cnocr.cpp:14` | 枚举 → 模型名 | 枚举与注册表之间的唯一映射 |
| `load_dict` | `cnocr.cpp:38` | 字典文件 → `vector<string>` | 去 CRLF；`<space>`→`" "`；`<unused*>`→`""` |
| `prepare_line_image` | `cnocr.cpp:64` | 行图 → 行图 | 高固定 32/48，宽等比，最小 8 |
| `decode_ctc` | `cnocr.cpp:85` | `rec_return` → `(text, conf)` | 逐帧 argmax、去重、跳 blank、置信度取最小 |
| `cnocr::ocr(path)` | `cnocr.cpp:150` | 路径 → 结果 | 读图失败抛异常 |
| `cnocr::ocr(Mat)` | `cnocr.cpp:158` | BGR 图 → 结果 | 检测 + 裁剪 或 投影切行 |
| `recognize_line` | `cnocr.cpp:199` | 行图 → `(text, conf)` | 预处理 → `run_rec` → `decode_ctc` |
| `line_split` | `cnocr.cpp:210` | 灰度图 → 行图列表 | 反色后行投影 |
| `resized_shape_of` | `cnstd.cpp:16` | 原尺寸 → 目标尺寸 | 长边 768、32 对齐 |
| `fill_normalized_chw` | `cnstd.cpp:36` | BGR → CHW float | ImageNet 归一化，HWC→CHW |
| `CnStd::detect` | `cnstd.cpp:74` | BGR → `vector<DetBox>` | 预处理 → `run_det` → 后处理 |
| `CnStd::crop` | `cnstd.cpp:103` | 图 + 框 → 行图 | 透视变换，竖排旋转 90° |
| `boxes_from_prob` | `db_postprocess.cpp:177` | 概率图 → 框 | 二值化、轮廓、打分、外扩、还原、排序 |
| `mini_boxes` | `db_postprocess.cpp:22` | 轮廓点 → 4 点 + 短边 | `minAreaRect` + 点序还原 |
| `box_score_fast` | `db_postprocess.cpp:114` | 概率图 + 框 → 分数 | bbox 内多边形 mask 求均值 |
| `offset_polygon` | `db_postprocess.cpp:65` | 多边形 + 距离 → 多边形 | 按朝向做 miter 外扩 |
| `order_points_clockwise` | `db_postprocess.cpp:51` | 4 点 → 4 点 | 左上/右上/右下/左下 |
| `sort_reading_order` | `db_postprocess.cpp:163` | 框列表 → 排序 | 插入排序，纵向重叠 > 0.5 比左右 |
| `onnxmodle::run_rec` | `modle.cpp:94` | uint8 HWC → `rec_return` | HWC→CHW、双输入组装、按秩找输出 |
| `onnxmodle::run_det` | `modle.cpp:164` | CHW float → `det_return` | 4 维概率图拷贝 |
| `find_output` | `modle.cpp:37` | 输出列表 → 张量指针 | 按秩 + 元素类型匹配 |
| `ensure_model` | `model_registry.cpp:121` | `ModelSpec` → 路径 | 本地 → 缓存 → 下载 |
| `ensure_det_model` | `model_registry.cpp:132` | `bool` → 路径 | 检测模型的固定 `ModelSpec` |
| `ensure_downloaded` | `model_registry.cpp:37` | 规格 → 路径 | 拼接 URL、多镜像重试、解压 |
| `modelfiles::download` | `model_files.cpp:264` | URL + 路径 | curl → wget，`.part` 原子替换 |
| `modelfiles::extract_onnx` | `model_files.cpp:304` | zip → onnx | 中央目录找第一个 `.onnx`，raw deflate |

---

## 5. 数据格式与内存布局速查

| 阶段 | C++ 类型 | 形状 / 布局 | 数值范围 |
| --- | --- | --- | --- |
| 读图 | `cv::Mat` | H×W，HWC，`CV_8UC3` | 0–255（BGR） |
| 检测输入 | `std::vector<float>` | CHW：`c·H'·W' + y·W' + x` | 约 `[-2.1, 2.6]`（ImageNet 归一化） |
| 检测张量 | `Ort::Value` | `{1, 3, H', W'}` float32 | 同上 |
| 概率图 | `cv::Mat` | H_p×W_p，`CV_32FC1`，行优先 | 0–1 |
| 检测框 | `DetBox` | 4 个 `Point2f` + score | 原图像素坐标 |
| 行图 | `cv::Mat` | h×w，`CV_8UC1/3` | 0–255 |
| 识别输入 | `std::vector<float>` | `{1, C, 32|48, W}` | 1 通道 0–1；3 通道 -1–1 |
| 序列长度输入 | `int64_t` | `{1}`，值 = 行图宽度 | — |
| logits | `std::vector<float>` | `[T, C]` 行优先，`t·C + c` | 未归一化 |
| 有效长度 | `std::vector<int64_t>` | 通常 `{1}` | `0..T` |
| 字典 | `std::vector<std::string>` | 下标 = 类别 | UTF-8 字节 |
| 结果 | `vector<pair<string,float>>` | 阅读顺序 | 置信度 0–1 |

### 5.1 一个完整数值示例（`cnocr.png`，默认中文模型）

```text
读图                 747 × 57 × 3 (BGR uint8)
检测尺寸规划         768 × 64（长边 768，高对齐 32）
检测输入             3 × 768 × 64 float = 147456 个
检测输出概率图       1 × 1 × 64 × 768（该导出与输入同尺寸）
文本框（示例）       (10,10)-(738,25)  score≈0.743
裁剪                 728 × 15 行图（高/宽 < 1.5，不旋转）
识别预处理           高 32，宽 = round(728×32/15) = 1553
识别输入             1 × 1 × 32 × 1553；input_lengths = 1553
logits               1 × 194 × 6674（宽 / 8 下采样）
output_lengths       194 → valid_len = 194（不是 1553）
CTC 后处理           跳过 blank(6673)、去重、查字典 → UTF-8 文本
```

---

## 6. 模型清单与自动下载

### 6.1 模型规格表

`model_registry.cpp:88` 中的 `ModelSpec` 字段：`name, channels, height, blank_index, onnx_file, dict_file, subfolder, archive`。

| 模型名 | 枚举 | 通道 | 高度 | blank | 输出类别 | 字典（行数） |
| --- | --- | --- | --- | --- | --- | --- |
| `cnocr136fc` | `cnocr136fc` | 1 | 32 | 最后一类 | 6674 | `label_cn.txt`（6673） |
| `densenet_lite_136-gru` | `densenet_lite_136_gru` | 1 | 32 | 最后一类 | 6683 | `label_cn_2.3.txt`（6682） |
| `scene-densenet_lite_136-gru` | `scene_densenet_136_gru` | 1 | 32 | 最后一类 | 6683 | `label_cn_2.3.txt` |
| `doc-densenet_lite_136-gru` | `doc_densenet_136_gru` | 1 | 32 | 最后一类 | 6683 | `label_cn_2.3.txt` |
| `number-densenet_lite_136-fc` | `number_densenet_136_fc` | 1 | 32 | 最后一类 | 11 | `label_number.txt`（10） |
| `en_number_mobile_v2.0` | `en_number` | 3 | 32 | 0 | 97 | `en_dict.txt`（97） |
| `chinese_cht_PP-OCRv3` | `chinese_cht` | 3 | 32 | 0 | 8423 | `chinese_cht_dict.txt`（8423） |
| `ch_PP-OCRv3` | `ch_ppocrv3` | 3 | **48** | 0 | 6625 | `ppocr_keys_v1.txt`（6625） |
| `ch_ppocr_mobile_v2.0` | `ch_ppocr_mobile_v2` | 3 | 32 | 0 | 6625 | `ppocr_keys_v1.txt` |
| `ch_PP-OCRv3_det` | （检测） | 3 | — | — | 1 | — |

规律：

- **cnocr 自训练模型**：`输出类别数 = 字典行数 + 1`，blank 排最后；输入 2 个（图像 + `input_lengths`）。
- **PaddleOCR 模型**：字典首行就是 blank，末行是空格，`输出类别数 = 字典行数`；输入 1 个。
- `ch_PP-OCRv3` 必须用 **高 48**：模型按 48 高训练，用 32 会丢字、并英文单词。
- `ppocr_keys_v1.txt` 是**重建过**的版本（blank 占位 + 6623 个字符 + 空格），以匹配模型的 6625 类。

> 校验方法：用 onnxruntime 打印输出最后一维，应等于「输出类别」列；字典行数与 blank 规则见上。
>
> ```python
> import onnxruntime as ort
> s = ort.InferenceSession("modle/cnocr136fc.onnx", providers=["CPUExecutionProvider"])
> print(s.get_inputs(), s.get_outputs())
> ```

### 6.2 查找顺序

`ensure_downloaded`（`model_registry.cpp:37`）：

```text
modle/<onnx_file>                存在 → 直接使用（仓库自带/用户手动放置）
        ↓ 不存在
<缓存>/models/<onnx_file>        存在 → 直接使用
        ↓ 不存在
auto_download == true ? 下载 : 抛错
```

缓存根目录 `modelfiles::data_dir()`：

| 平台 | 顺序 |
| --- | --- |
| Windows | `%CNOCR_HOME%` → `%APPDATA%/cnocr` → `%USERPROFILE%/.cnocr` → 临时目录 |
| Linux/macOS | `$CNOCR_HOME` → `$HOME/.cnocr` → 临时目录 |

### 6.3 下载 URL 与镜像

```text
<endpoint>/breezedeus/cnstd-cnocr-models/resolve/main/<subfolder>/<archive>
```

镜像选择（`model_registry.cpp:14`）：

1. `CNOCR_MODEL_ENDPOINT` 存在 → **只**用它；
2. 否则 `HF_ENDPOINT` → `https://hf-mirror.com` → `https://huggingface.co`，逐个尝试。

例如默认识别模型：

```text
https://hf-mirror.com/breezedeus/cnstd-cnocr-models/resolve/main/models/cnocr/2.3/densenet_lite_136-gru-onnx.zip
```

全部失败时抛错并提示可手动放置的两个路径与 `CNOCR_MODEL_ENDPOINT`。

### 6.4 下载与解压实现

`modelfiles::download`（`model_files.cpp:264`）：

1. 若 `curl --version` 成功 → `curl -fL --retry 3 --connect-timeout 20 -sS -o <part> <url>`；
2. 否则若 `wget --version` 成功 → `wget -q -O <part> <url>`；
3. 先写 `<dest>.part`，成功后删除旧文件并 `rename` 成正式文件（避免半截文件被当成可用模型）；
4. Windows 用 `_wsystem` + 宽字符串，POSIX 用 `std::system` + 单引号转义，两者都兼容非 ASCII 路径。

`modelfiles::extract_onnx`（`model_files.cpp:304`）不依赖第三方 zip 库，只依赖 zlib：

```text
读文件尾部最多 65557 字节，倒序找 EOCD 签名 0x06054b50
  → 取中央目录偏移（EOCD+16）
  → 遍历 46 字节中央目录项 0x02014b50（压缩方式/大小/名称长度/本地头偏移）
  → 取第一个以 .onnx 结尾的条目
  → 跳到本地头 0x04034b50 之后的数据区
  → method 0：原样拷贝；method 8：zlib raw inflate（inflateInit2(-MAX_WBITS)）
  → 校验解压字节数 == 中央目录中的 uncompressed_size
  → 写 <out>.part 再 rename
```

限制：不支持 ZIP64（大小字段为 `0xFFFFFFFF` 时直接报错）；不校验 CRC。

---

## 7. 常量与阈值

| 常量 | 值 | 位置 | 作用 |
| --- | --- | --- | --- |
| 检测目标尺寸 | 768×768 | `cnstd.cpp:76` | `detect` 默认参数 |
| 尺寸对齐 | 32 | `cnstd.cpp:14` | DBNet 下采样 |
| ImageNet mean/std | 见 3.3 | `cnstd.cpp:49` | 检测归一化 |
| 二值化阈值 | 0.3 | `db_postprocess.h:12` | 概率图 → 二值图 |
| 框得分阈值 | 0.3 | `db_postprocess.h:13` | 过滤低质量框 |
| unclip 比例 | 1.5 | `db_postprocess.h:14` | 文本框外扩 |
| 最小框尺寸 | 4 | `db_postprocess.h:15` | 最终过滤 |
| 最大轮廓数 | 1000 | `db_postprocess.h:16` | 控制耗时 |
| 轮廓短边下限 | 3 / 5 | `db_postprocess.cpp:206,225` | 外扩前后过滤 |
| 竖排旋转阈值 | 高/宽 ≥ 1.5 | `cnstd.cpp:125` | 旋转为横向 |
| 行切分阈值 | 行内最大灰度 > 100 | `cnocr.cpp:229` | 无检测模式 |
| 行最小高度 | > 5 | `cnocr.cpp:220` | 无检测模式 |
| 识别最小宽度 | 8 | `cnocr.cpp:79` | 极短行保护 |

---

## 8. 学习要点与易错点

1. **张量内存所有权**
   `CreateTensor` 不拷贝输入缓冲区，`chw` / `pixel_data` 必须活到 `Run` 返回；输出 `Ort::Value` 在函数结束时析构，必须 `copy_tensor_data` 拷出，不能返回裸指针。

2. **输出不能按顺序假设**
   `find_output` 用「秩 + 元素类型」定位：检测取 4 维 float，识别取 3 维 float（logits）和 1 维 int64（lengths）。不同导出工具的输出顺序不同，按顺序取会出错。

3. **blank 有两套规则**
   cnocr 自训练模型 blank 是最后一类（= 字典长度）；PaddleOCR 系 blank 是第 0 类。弄反会输出乱码或整段丢字。

4. **PP-OCRv3 高度是 48**
   用 32 推理不会崩溃，但准确率明显下降，这是最容易踩的「静默错误」。

5. **字典与类别数必须对齐**
   字典行数决定 `dict[best]` 是否越界，也决定 blank 下标。换模型时必须同时换字典；`load_dict` 只做了 `<space>` / `<unused*>` 的占位处理，不校验总类数。

6. **坐标还原用概率图尺寸**
   `x / prob.width * src.width`，而不是 `x / target.width * src.width`。模型若带下采样，两者不同。

7. **阅读顺序排序不能用 `std::sort`**
   `compare_boxes` 的比较规则依赖两框的相对位置，不满足严格弱序；用插入排序保证确定性（与 Python 版一致）。

8. **置信度是「最差帧」**
   `min` over frames 的策略会让长行、生僻字的分数很低（如英文、繁体输出 0.01–0.0001），这不代表整体识别质量，比较模型时应看文本本身。

9. **UTF-8 全链路**
   字典文件、`std::string` 结果、`std::filesystem::path`、控制台代码页都是 UTF-8；不要在中间引入 `wchar_t` / ANSI 转换。

10. **相对路径依赖工作目录**
    模型找 `modle/...`、字典找 `dict/...`、图片按传入路径。CMake 把这些资源复制到可执行文件旁，因此要在输出目录运行，或自行传入绝对路径。

11. **`.part` 原子写入**
    下载和解压都先写临时文件再改名，避免下载中断留下损坏的 `.onnx` 被后续误用。

---

## 9. 如何新增一个识别模型

以新增 `xxx` 为例：

1. **确认模型规格**：用 onnxruntime 打印输入输出，记录
   - 输入：通道数、高、是否带 `input_lengths`；
   - 输出：`logits` 最后一维（类别数）、是否有 `output_lengths`。
2. **准备字典**：确保 `行数 + blank` 规则与类别数一致；PP 系字典首行留空、末行空格。
3. **注册**：在 `model_registry.cpp:88` 的 `all_recognizers()` 增加一行 `ModelSpec`（名称、通道、高度、blank、文件名、字典、仓库子目录、压缩包名）。
4. **加枚举**：在 `cnocr.h` 的 `USE_MODLE` 增加枚举值，并在 `cnocr.cpp:14` 的 `model_name_of` 映射到注册名。
5. **加命令行别名**（可选）：`main.cpp` 的 `kModels`。
6. **验证**：
   - `UseAiForWinML list` 能看到；
   - 指定模型跑示例图，文本正确、无乱码；
   - 与 Python cnocr 同模型同图片结果对比。

> 若模型没有公开下载地址，把 `archive` 留空，并把 `.onnx` 放到 `modle/` 或缓存目录。

---

## 10. 构建与调试

```sh
cmake -S . -B build
cmake --build build --config Release
```

CMake 依次处理：vcpkg 工具链探测 → onnxruntime（显式路径/环境变量/常见目录/系统包，最后按平台下载）→ OpenCV → zlib → 生成 `cnocr` 共享库与 `UseAiForWinML` → 拷 onnxruntime 运行库 → 拷 `modle/`、`dict/` 与示例图片到输出目录。

常见问题：

| 现象 | 原因 / 处理 |
| --- | --- |
| `模型文件不存在或无法读取` | `modle/` 与缓存都没有；联网会自动下载，离线请手动放置 |
| `未找到 curl 或 wget` | 安装其一，或手动放置模型 |
| `下载失败: <url>` | 镜像不可用，设置 `CNOCR_MODEL_ENDPOINT`（如其他 HF 镜像） |
| `不是有效的 zip 文件` | 镜像返回了 HTML 错误页（HTTP 200），换镜像重试 |
| `暂不支持 ZIP64 压缩包` | 重新打包为普通 zip，或手动解压 |
| 输出乱码 | 控制台未切 UTF-8；Windows 终端执行 `chcp 65001` |
| 识别结果串行/丢字 | 检查模型高度（PP-OCRv3 必须 48）、字典与类别数是否匹配 |
| `执行提供器不可用，回退到 CPU` | 当前 onnxruntime 包不含 CUDA/TensorRT provider，属预期行为 |

命令行：

```sh
UseAiForWinML                 # 依次跑中文 / 英文 / 繁体
UseAiForWinML list            # 列出模型别名
UseAiForWinML zh cnocr.png    # 指定模型识别
UseAiForWinML det cnocr.png   # 只做检测，打印框和分数
```
