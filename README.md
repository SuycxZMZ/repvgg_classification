# RepVGG 图像分类模型训练部署简单流程 🚀

本项目以 [RepVGG](https://github.com/DingXiaoH/RepVGG) 为核心，简化了**模型训练、验证、推理、权重转换、ONNX 导出、C++高效部署**等流程，适合深度学习入门与简单工程实践。

---

## 目录

- [环境准备](#环境准备)
- [Python 端典型命令与说明](#python-端典型命令与说明)
- [C++ 端典型命令与说明](#c-端典型命令与说明)
- [C++ 依赖安装与环境配置](#c-依赖安装与环境配置)
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
# 训练或者推理验证过程中，缺啥包，用pip装啥包就行
pip install torch torchvision opencv-python onnx onnxruntime onnxsim tqdm
```

---

## Python 端典型命令与说明

> **注意：所有权重文件路径、数据集路径请根据你本地实际情况修改！**

### 1. 训练（指定ImageNet预训练权重）
```bash
python main.py --mode train --pretrained_path "/Users/yuansu/Code/repvgg_classification/pretrained/RepVGG-A0-train.pth"

# 数据集下载与目录结构说明
# 数据集下载地址：[猫狗识别数据集（ModelScope）](https://www.modelscope.cn/datasets/tany0699/cats_and_dogs)

# 解压后目录结构如下：
# cats_and_dogs/
#   ├── train/
#   │   ├── cat/
#   │   └── dog/
#   └── val/
#       ├── cat/
#       └── dog/
```
- `--pretrained_path` 指定ImageNet预训练权重路径。

### 2. 推理（训练权重）
```bash
python main.py --mode "inference" --input "/Users/yuansu/Desktop/codes/datasets/cats_and_dogs/val/cat"
```
- `--input` 可为单张图片或文件夹。
- 默认使用config.py中指定的训练权重。

### 3. 权重转换（训练权重→部署权重）
```bash
python convert.py --model "RepVGG-A0" --input "/Users/yuansu/Code/repvgg_classification/checkpoints/RepVGG-A0_1/RepVGG-A0_4/best.pth"
```
- `--input` 为训练得到的best.pth。
- `--output` 可省略，默认与输入同目录。

### 4. 使用重参数化权重推理
```bash
python main.py --mode "inference" --input "/Users/yuansu/Desktop/codes/datasets/cats_and_dogs/val/dog" --checkpoint_path "/Users/yuansu/Code/repvgg_classification/checkpoints/RepVGG-A0_1/RepVGG-A0_4/deploy.pth" --deploy True
```
- `--checkpoint_path` 指定deploy.pth。
- `--deploy True` 表示推理结构。

### 5. Python下转换前后推理速度对比
```bash
python benchmark.py --model RepVGG-A0 --train_weights /Users/yuansu/Code/repvgg_classification/checkpoints/RepVGG-A0_1/RepVGG-A0_4/best.pth --deploy_weights /Users/yuansu/Code/repvgg_classification/checkpoints/RepVGG-A0_1/RepVGG-A0_4/deploy.pth --num_classes 2
```
- 对比训练结构和deploy结构的推理速度。

### 6. 导出ONNX权重
```bash
python export_onnx.py --weights /Users/yuansu/Code/repvgg_classification/checkpoints/RepVGG-A0_1/RepVGG-A0_4/deploy.pth --model "RepVGG-A0"
```
- `--weights` 指deploy.pth。
- `--output` 可省略，默认同目录。

---

## C++ 端典型命令与说明

> **注意：C++推理和benchmark命令中的ONNX权重路径、图片路径等也需根据实际情况修改！**

### 1. 推理（编译后可执行文件）
```bash
./repvgg_inference /Users/yuansu/Code/repvgg_classification/cpp_inference/checkpoints/deploy.onnx /Users/yuansu/Desktop/codes/datasets/cats_and_dogs/val/dog 224
```
- 第一个参数为ONNX模型路径，第二个为图片或文件夹路径，第三个为图片尺寸。

### 2. C++版本速度测试
```bash
./benchmark_onnx /Users/yuansu/Code/repvgg_classification/cpp_inference/checkpoints/deploy.onnx 224 100 500
```
- 100为预热次数，500为测试次数。

> **提示：ONNX系统库报错可忽略，不影响实际推理和速度测试。**

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

### 1. VGG网络推理原理详解

VGG网络是一种典型的深层卷积神经网络，其推理流程如下：

1. **图片输入**
   - 输入一张RGB图片，通常会resize为224x224像素，归一化到[0,1]或[-1,1]。

2. **卷积层（Conv）**
   - 多个3x3卷积核滑动窗口提取局部特征，每个卷积核可看作一个特征检测器。
   - 每一层卷积后，输出特征图（feature map），通道数逐层增加（如64→128→256→512）。
   - 卷积操作本质：对输入的每个小区域与卷积核做加权求和，输出一个新像素。

3. **激活函数（ReLU）**
   - 每个卷积层后接ReLU激活，增加非线性表达能力。
   - 公式：`y = max(0, x)`，抑制负值，保留正值。

4. **池化层（Pooling）**
   - 通常为2x2最大池化（MaxPooling），每2x2区域取最大值，降低空间分辨率，减少参数和计算量。
   - 池化有助于提取更具鲁棒性的特征，抑制噪声。

5. **多层卷积+池化堆叠**
   - VGG16/19等网络会堆叠多组"卷积+ReLU+池化"，逐步提取从低级到高级的特征。

6. **展平（Flatten）**
   - 最后一个池化层输出的特征图展平成一维向量，作为全连接层输入。

7. **全连接层（FC）**
   - 一般有2-3个全连接层，模拟传统MLP，进一步融合全局特征。
   - 最后一层输出的向量长度等于类别数（如1000）。

8. **Softmax输出**
   - 对最后一层输出做softmax归一化，得到每个类别的概率。
   - 取概率最大者为最终预测类别。

**推理流程举例：**
> 一张猫的图片输入VGG16，经过多层卷积提取边缘、纹理、形状等特征，池化降维，全连接层融合全局信息，softmax输出"cat"类别概率最大，模型最终判定为"cat"。

---

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
- 本项目参考了 [RepVGG](https://github.com/DingXiaoH/RepVGG) 官方实现及知乎等社区资料，感谢开源社区的贡献！

---

> **如有问题或建议，欢迎 issue 或 PR！**
