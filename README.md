# RepVGG 分类模型全流程教学项目 🚀

本项目以 RepVGG 为核心，覆盖了**模型训练、验证、推理、权重转换、ONNX 导出、C++高效部署**等全流程，适合深度学习入门与工程实战教学。

---

## 目录
- [环境准备](#环境准备)
- [Python 端训练/验证/推理/权重导出](#python-端训练验证推理权重导出)
- [C++ 端高效部署与推理](#c-端高效部署与推理)
- [Python & C++ 速度对比](#python--c-速度对比)
- [VGG家族与RepVGG模型原理详解](#vgg家族与repvgg模型原理详解)
- [结构重参数化宇宙简述](#结构重参数化宇宙简述)

---

## 环境准备 🛠️

### 1. 创建 conda 环境
```bash
conda create -n repvgg python=3.9 -y
conda activate repvgg
```

### 2. 安装 Python 依赖
```bash
pip install torch torchvision opencv-python onnx onnxruntime onnxsim tqdm
```

---

## C++ 依赖安装与环境配置

| 平台   | ONNX Runtime 安装                | OpenCV 安装                | CMake 安装                |
|--------|----------------------------------|----------------------------|---------------------------|
| **macOS** | `brew install onnxruntime`        | `brew install opencv`      | `brew install cmake`      |
| **Ubuntu** | `sudo apt update && sudo apt install -y onnxruntime libopencv-dev cmake g++` <br>（如需源码编译OpenCV/ONNX Runtime请参考官方文档） | `sudo apt install libopencv-dev` | `sudo apt install cmake` |

> **注意：**
> - Ubuntu下如需更高性能，可从[ONNX Runtime官方](https://onnxruntime.ai/)源码编译，支持OpenMP/MKL等。
> - OpenCV建议用系统包或源码编译，确保有C++头文件和动态库。

---

## Python 端训练/验证/推理/权重导出

### 1. 训练模型
```bash
python train.py
```

### 2. 验证模型
```bash
python main.py --mode val
```

### 3. 单张图片/文件夹推理
```bash
# --img_path 后面可以是图片文件路径，也可以是文件夹路径
python inference.py --img_path /path/to/image.jpg      # 推理单张图片
python inference.py --img_path /path/to/folder/        # 推理文件夹下所有图片
```
> `/path/to/image.jpg` 请替换为你自己的图片路径，`/path/to/folder/` 替换为包含图片的文件夹路径。

### 4. 权重转换（训练权重 → 部署权重）
```bash
# --input 指训练得到的权重文件路径（如 best.pth）
# --output 指转换后部署权重的保存路径（如 deploy.pth）
python convert.py --model RepVGG-A0 --input ./checkpoints/RepVGG-A0_4/best.pth --output ./checkpoints/RepVGG-A0_4/deploy.pth --num_classes 2
```
> 路径可根据实际训练输出位置调整。

### 5. 导出 ONNX 权重
```bash
# --weights 指 deploy.pth 路径，--output 指导出的 onnx 文件保存路径
python export_onnx.py --model RepVGG-A0 --weights ./checkpoints/RepVGG-A0_4/deploy.pth --output ./checkpoints/RepVGG-A0_4/deploy.onnx --num_classes 2
```
> 建议将 onnx 文件保存在与权重同目录，便于管理。

### 6. Python 端 Benchmark
```bash
python benchmark.py
```

---

## C++ 端高效部署与推理

### 1. 编译 C++ 代码

#### macOS
```bash
cd cpp_inference
mkdir -p build && cd build
cmake ..
make
```

#### Ubuntu (Linux)
```bash
cd cpp_inference
mkdir -p build && cd build
cmake ..
make
```
> 如遇到找不到 onnxruntime/opencv 的头文件或库，请在 CMakeLists.txt 中手动指定路径。

### 2. C++ 推理命令
- **单张图片/文件夹分类**
  ```bash
  # 第一个参数为 onnx 模型路径，第二个为图片或文件夹路径，第三个为图片尺寸（如224）
  ./repvgg_inference ../../checkpoints/RepVGG-A0_4/deploy.onnx /path/to/image_or_folder 224
  ```
  > `../../checkpoints/RepVGG-A0_4/deploy.onnx` 路径是相对于 build 目录的，注意根据实际情况调整。
  > `/path/to/image_or_folder` 可为单张图片或图片文件夹。
- **C++ Benchmark**
  ```bash
  # 第一个参数为 onnx 模型路径，第二个为图片尺寸，第三/四个为预热和测试次数
  ./benchmark_onnx ../../checkpoints/RepVGG-A0_4/deploy.onnx 224 20 100
  ```
  > 路径同上，参数可根据实际需求调整。

---

## Python & C++ 速度对比

| 版本         | 平均延迟 (ms) | 吞吐量 (FPS) |
|--------------|--------------|--------------|
| Python部署版 | 9.54         | 104.85       |
| C++ ONNX版   | 11.85        | 84.39        |

> **说明：** 按理说 C++ ONNX Runtime 推理应更快，但本项目测试中 C++ 版本略慢，原因可能包括：
> - ONNX Runtime 线程数/BLAS优化未完全调优
> - Python/PyTorch 默认多线程，BLAS库更优
> - ONNX模型结构未极致优化
> 
> **后续可进一步测试和优化。**

---

## VGG家族与RepVGG模型原理详解

### 1. VGG家族发展简述
- **VGG（2014）**：提出极简的卷积网络结构（3x3卷积+2x2池化+全连接），极大推动了深度学习模型的标准化。
- **VGG16/VGG19**：加深网络层数，提升表达能力，但参数量大、推理慢。
- **后续改进**：ResNet、DenseNet等引入残差/密集连接，提升性能和效率。

### 2. RepVGG原理与创新
- **核心思想**：训练时用多分支结构（3x3卷积、1x1卷积、Identity），推理时将多分支结构重参数化为单一3x3卷积，兼顾训练性能和推理速度。
- **结构重参数化**：训练时的多分支结构可提升表达能力和收敛速度，推理时融合为单分支，极大提升推理效率。
- **优势**：
  - 推理速度媲美甚至超越ResNet/VGG
  - 结构简单，易于部署
  - 支持多种硬件平台
- **详细解读**：可参考作者知乎文章 [《RepVGG：结构重参数化宇宙的起点》](https://zhuanlan.zhihu.com/p/344324470)

---

## 结构重参数化宇宙简述
- **起点：RepVGG**
- **核心思想**：训练时用多分支结构提升性能，推理时融合为单分支，兼顾表达力与推理效率。
- **后续发展**：RepResNet、YOLOv6/YOLOv9等众多模型均采用结构重参数化思想、作者后续还有RepMLP、RepLKNet等一系列工作。
- **意义**：极大推动了深度学习模型的工业部署和高效推理。

---

## 致谢
- 本项目参考了 RepVGG 官方实现及知乎等社区资料，感谢开源社区的贡献！

---

> **如有问题或建议，欢迎 issue 或 PR！**
