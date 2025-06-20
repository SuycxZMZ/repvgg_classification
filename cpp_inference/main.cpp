#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include "onnxruntime_cxx_api.h"
#include <iomanip> // for std::setprecision
#include <cmath>   // for std::exp

// --------------------------------------------------------
// RepVGG C++ 推理脚本
// 支持单张图片或文件夹批量推理，自动输出类别和置信度
// 需配合 ONNX Runtime C++ API 和 OpenCV
// --------------------------------------------------------

namespace fs = std::filesystem;

// 类别名（需与训练时类别顺序一致）
const std::vector<std::string> class_names = {"cat", "dog"};

// 图像预处理函数：对输入图片进行resize、BGR转RGB、归一化、NCHW排列
std::vector<float> preprocess(const cv::Mat& img, int img_size) {
    // 1. resize
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(img_size, img_size));

    // 2. BGR to RGB
    cv::Mat rgb;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

    // 3. to float32, scale to [0,1]
    rgb.convertTo(rgb, CV_32F, 1.0 / 255);

    // 4. 归一化到 [-1, 1]，即 (x-0.5)/0.5
    std::vector<cv::Mat> channels(3);
    cv::split(rgb, channels);
    for (int i = 0; i < 3; ++i) {
        channels[i] = (channels[i] - 0.5f) / 0.5f;
    }

    // 5. 合并为NCHW
    std::vector<float> input_tensor_values(3 * img_size * img_size);
    int channel_size = img_size * img_size;
    for (int c = 0; c < 3; ++c) {
        memcpy(input_tensor_values.data() + c * channel_size, channels[c].data, channel_size * sizeof(float));
    }
    return input_tensor_values;
}

// 计算softmax概率分布
std::vector<float> softmax(const float* logits, size_t n) {
    std::vector<float> probs(n);
    float max_logit = *std::max_element(logits, logits + n);
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        probs[i] = std::exp(logits[i] - max_logit);
        sum += probs[i];
    }
    for (size_t i = 0; i < n; ++i) {
        probs[i] /= sum;
    }
    return probs;
}

// 单张图片推理与结果打印
void infer_image(const std::string& image_path, Ort::Session& session, Ort::AllocatorWithDefaultOptions& allocator, int img_size) {
    cv::Mat img = cv::imread(image_path);
    if (img.empty()) {
        std::cerr << "❌ 读取图片失败: " << image_path << std::endl;
        return;
    }
    std::vector<float> input_tensor_values = preprocess(img, img_size);

    auto input_name_ptr = session.GetInputNameAllocated(0, allocator);
    auto output_name_ptr = session.GetOutputNameAllocated(0, allocator);
    const char* input_name = input_name_ptr.get();
    const char* output_name = output_name_ptr.get();

    std::vector<int64_t> input_dims = {1, 3, img_size, img_size};
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_values.size(), input_dims.data(), input_dims.size());

    std::vector<const char*> input_names = {input_name};
    std::vector<const char*> output_names = {output_name};

    auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names.data(), &input_tensor, 1, output_names.data(), 1);
    float* output_array = output_tensors[0].GetTensorMutableData<float>();

    // 计算softmax概率
    std::vector<float> probs = softmax(output_array, class_names.size());
    int pred = std::distance(probs.begin(), std::max_element(probs.begin(), probs.end()));
    float confidence = probs[pred];

    // 专业打印
    std::cout << "------------------------------------------------------------\n";
    std::cout << "图片: " << image_path << "\n";
    std::cout << "预测类别: " << class_names[pred]
              << "    置信度: " << std::fixed << std::setprecision(4) << confidence << "\n";
    std::cout << "类别概率分布: ";
    for (size_t i = 0; i < class_names.size(); ++i) {
        std::cout << class_names[i] << ": " << std::fixed << std::setprecision(4) << probs[i] << "  ";
    }
    std::cout << "\n";
}

// 主程序入口，支持单张图片或文件夹批量推理
int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cout << "用法: " << argv[0] << " <onnx模型路径> <图片路径或文件夹> [图片尺寸, 默认224]" << std::endl;
        return 1;
    }
    std::string model_path = argv[1];
    std::string input_path = argv[2];
    int img_size = argc > 3 ? std::stoi(argv[3]) : 224;

    // 初始化 ONNX Runtime 环境与会话
    Ort::Env env(ORT_LOGGING_LEVEL_ERROR, "RepVGG_Inference");
    Ort::SessionOptions session_options;
    Ort::Session session(env, model_path.c_str(), session_options);
    Ort::AllocatorWithDefaultOptions allocator;

    // 判断输入路径类型，支持批量推理
    if (fs::is_directory(input_path)) {
        for (const auto& entry : fs::directory_iterator(input_path)) {
            if (entry.is_regular_file()) {
                infer_image(entry.path().string(), session, allocator, img_size);
            }
        }
    } else if (fs::is_regular_file(input_path)) {
        infer_image(input_path, session, allocator, img_size);
    } else {
        std::cerr << "❌ 输入路径无效: " << input_path << std::endl;
        return 1;
    }
    return 0;
} 